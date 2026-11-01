"""Fine-tuning: AdamW on every weight but the frozen ViTs (and what `--freeze` names), linear warmup
then cosine decay, bf16 autocast on CUDA, gradient clipping, and an EMA of the weights, which is
what validation, snapshots and export use. The normalizer (`norm.*`) is buffers and never moves.

    <run>/args.json                   the command's settings
    <run>/log.jsonl                   a train record every --log-every steps, a val record every
                                      --val-every and at every snapshot
    <run>/last.pt                     trainable weights, EMA, optimizer and step, for --resume
    <run>/ema-<step>.safetensors      the EMA's trainable tensors every --save-every steps, with their
                                      validation loss: what `export` merges into the base. Every one
                                      is kept (about 2 GB each), as candidates to submit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from vector_runtime.net import load

from . import features, model
from .batches import Bank

#: Parameters that are positions or learned tokens: no weight decay, like biases and norms.
NO_DECAY = ("prompt_pos", "obs_pos", "sinks")


class EMA:
    """An exponential moving average of the trainable weights, with the usual warmup of its decay."""

    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in params.items()}

    @torch.no_grad()
    def update(self, params: dict[str, torch.nn.Parameter], step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        names = list(params)
        torch._foreach_lerp_([self.shadow[n] for n in names], [params[n].detach() for n in names], 1 - d)

    @contextmanager
    def applied(self, params: dict[str, torch.nn.Parameter]) -> Iterator[None]:
        """The EMA in the network for the duration, the training weights back after."""
        saved = {n: p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(saved[n])


def trainable(net: torch.nn.Module, freeze: list[str], only: list[str]) -> dict[str, torch.nn.Parameter]:
    """Mark and return what trains: not `vision.*`, nothing under `freeze`, and when `only` is
    given, only what is under it (prefixes of parameter names)."""
    params = {}
    for name, p in net.named_parameters():
        train = not name.startswith("vision.") and not any(name.startswith(f) for f in freeze)
        if only:
            train = train and any(name.startswith(o) for o in only)
        p.requires_grad_(train)
        if train:
            params[name] = p
    if not params:
        raise ValueError("nothing to train: --freeze/--train-only leave no parameter")
    return params


def schedule(warmup: int, total: int, floor: float):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


@torch.no_grad()
def validate(net, sets, autocast, device, seed: int) -> dict[str, float]:
    """Mean loss per task over fixed samples with fixed noise: comparable across steps."""
    was = net.training
    net.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = {}
    for task, batches in sets.items():
        losses = []
        for b in batches:
            with autocast():
                losses.append(model.loss(net, b, generator=g).float())
        out[task] = torch.cat(losses).mean().item()
    net.train(was)
    return out


def _record(log, entry: dict) -> None:
    log.write(json.dumps(entry) + "\n")
    log.flush()


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = args.amp == "bf16" and device.type == "cuda"

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    meta = json.loads((Path(args.cache) / "meta.json").read_text())
    if meta["vision_sha256"] != features.vision_sha256(args.base):
        raise ValueError(
            f"{args.cache} was built from other ViT weights than {args.base}: rebuild the cache"
        )
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1, default=str))

    print(f"loading {args.base}")
    net = load(args.base, device="cpu")
    net.to(device)
    net.vision.to("cpu")  # the cache stands in for it
    net.train()
    params = trainable(net, args.freeze, args.train_only)
    count = sum(p.numel() for p in params.values())
    print(f"training {count:,} parameters in {len(params)} tensors")

    bank = Bank(
        args.cache,
        device,
        tasks=args.tasks,
        held_out=args.held_out,
        task_weights=args.task_weight,
        same_collection=args.same_collection,
    )
    val_sets = bank.fixed("val", args.val_per_task, args.val_batch, seed=args.seed + 1)
    if not val_sets:
        print("no task has two held-out episodes: validation is off")

    decay = [p for n, p in params.items() if p.ndim >= 2 and not n.startswith(NO_DECAY)]
    flat = [p for n, p in params.items() if not (p.ndim >= 2 and not n.startswith(NO_DECAY))]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": flat, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=device.type == "cuda",
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, schedule(args.warmup, args.steps, args.min_lr_ratio)
    )
    ema = EMA(params, args.ema)
    step = 0
    last = run_dir / "last.pt"
    if args.resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(state["params"][n])
        ema.shadow = {n: t.to(device) for n, t in state["ema"].items()}
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        step = int(state["step"])
        print(f"resumed at step {step}")

    log = open(run_dir / "log.jsonl", "a")
    if step == 0 and val_sets:
        base_val = validate(net, val_sets, autocast, device, args.seed)
        _record(log, {"step": 0, "val": base_val, "val_mean": float(np.mean(list(base_val.values()))), "base": True})
        print(f"base val loss {np.mean(list(base_val.values())):.5f}")

    n_tasks = len(bank.tasks)
    task_sum = torch.zeros(n_tasks, device=device)
    task_n = torch.zeros(n_tasks, device=device)
    window_loss, window_steps, skipped = 0.0, 0, 0
    started = time.monotonic()
    while step < args.steps:
        batch = bank.sample(args.batch, "train", rng)
        with autocast():
            per = model.loss(net, batch)
        loss = per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), args.clip)
        if not torch.isfinite(norm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped > 100:
                raise RuntimeError("over 100 steps with non-finite gradients: lower --lr")
            continue
        opt.step()
        sched.step()
        step += 1
        ema.update(params, step)
        task_sum += torch.bincount(batch.task, weights=per.detach(), minlength=n_tasks).float()
        task_n += torch.bincount(batch.task, minlength=n_tasks).float()
        window_loss += loss.item()
        window_steps += 1

        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.monotonic() - started
            per_task = (task_sum / task_n.clamp(min=1)).tolist()
            entry = {
                "step": step,
                "loss": window_loss / window_steps,
                "lr": sched.get_last_lr()[0],
                "grad_norm": float(norm),
                "steps_per_s": window_steps / elapsed,
                "skipped": skipped,
                "tasks": {t: v for t, v, c in zip(bank.tasks, per_task, task_n.tolist(), strict=True) if c},
            }
            _record(log, entry)
            print(
                f"step {step:7d}  loss {entry['loss']:.5f}  lr {entry['lr']:.2e}  "
                f"|g| {entry['grad_norm']:.3f}  {entry['steps_per_s']:.2f} it/s"
            )
            task_sum.zero_()
            task_n.zero_()
            window_loss, window_steps = 0.0, 0
            started = time.monotonic()

        saving = step % args.save_every == 0 or step == args.steps
        val_mean = None
        if val_sets and (saving or step % args.val_every == 0):
            with ema.applied(params):
                val = validate(net, val_sets, autocast, device, args.seed)
            val_mean = float(np.mean(list(val.values())))
            _record(log, {"step": step, "val": val, "val_mean": val_mean, "ema": True})
            print(f"step {step:7d}  val (EMA) {val_mean:.5f}")

        if saving:
            save_snapshot(run_dir, step, ema, args, val_mean)
            tmp = last.with_suffix(".tmp")
            torch.save(
                {
                    "step": step,
                    "params": {n: p.detach().cpu() for n, p in params.items()},
                    "ema": {n: t.cpu() for n, t in ema.shadow.items()},
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state,
                },
                tmp,
            )
            os.replace(tmp, last)
    log.close()
    return run_dir


def save_snapshot(
    run_dir: Path, step: int, ema: EMA, args: argparse.Namespace, val_mean: float | None
) -> Path:
    """The EMA's tensors at `step`, written whole or not at all (a crash leaves only a .tmp)."""
    from safetensors.torch import save_file

    path = run_dir / f"ema-{step:07d}.safetensors"
    tmp = path.with_suffix(".tmp")
    tensors = {n: t.detach().float().cpu().contiguous() for n, t in ema.shadow.items()}
    meta = {"base": str(args.base), "step": str(step)}
    if val_mean is not None:
        meta["val_mean"] = repr(val_mean)
    save_file(tensors, str(tmp), metadata=meta)
    os.replace(tmp, path)
    print(f"saved {path}")
    return path
