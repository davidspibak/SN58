"""End-to-end checks of the pipeline against the validator's own code, on a few episodes. A CPU is
enough. Each check prints PASS, FAIL or SKIP with its numbers.

 1. colours   frames decode to RGB: the mat colour scene_info names is in frame 0
 2. rows      endpose -> network action -> endpose (`demo.action_row`) round-trips
 3. cache     a small feature cache builds (and one fed red/blue swapped, for check 8)
 4. prompt    the sampler's prompt is `demo.prompt`'s: frames kept, proprioception, actions, padding
 5. features  cached features match the live ViT on the same frames
 6. forward   the training condition matches `Net.encode_prompt` + `Net.predict` on live images,
              per sample and in one padded batch
 7. unet      the per-sample-timestep U-Net matches `Unet.forward`
 8. sanity    the base weights' loss is lower on correctly decoded colours than on swapped ones,
              and the normalizer maps the data's actions into [-1, 1]
 9. policy    `VectorPolicy` (what the validator serves) with the base weights: its first actions
              against the recorded ones, and against standing still
10. train     a few steps, a resume, both snapshots kept and listed with their validation loss,
              two exports that pass `vector_runtime.check` with distinct weights, and
              `VectorPolicy` serving one
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from vector_runtime import demo
from vector_runtime.net import (
    ACTION_DIM,
    CAMERAS,
    CHUNK,
    EXECUTED,
    HISTORY,
    HORIZON,
    PROPRIO,
    TOKENS,
    WIDTH,
    load,
)

from . import episodes as E
from . import export, features, model, snapshots, train
from .batches import Bank

#: Mat colours scene_info names, as RGB; colours a red/blue swap leaves alone tell nothing.
COLOURS = {
    "Red": (255, 0, 0), "Blue": (0, 0, 255), "Cyan": (0, 255, 255), "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0), "Purple": (128, 0, 128), "Pink": (255, 192, 203),
}  # fmt: skip


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool | None, str]] = []

    def __call__(self, name: str, ok: bool | None, detail: str) -> None:
        self.rows.append((name, ok, detail))
        mark = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}", flush=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


def _arrays(pose: np.ndarray, frames: dict[str, np.ndarray], row: int | None = None) -> dict:
    """An episode (or one row of it) as the benchmark hands it to a policy."""
    pick = (lambda a: a) if row is None else (lambda a: a[row])
    out = {"endpose": pick(pose)}
    for cam in CAMERAS:
        out["frames_" + E.EPISODE_CAMERAS[cam]] = pick(frames[cam])
    return out


def check_colours(picked: list[E.Episode], report: Report) -> None:
    import h5py

    votes = {"rgb": 0, "swapped": 0}
    for collection in sorted({e.path.parent.parent for e in picked}):
        info_path = collection / "scene_info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        for key, scene in list(info.items())[:40]:
            name = (scene.get("info") or {}).get("{B}")
            path = collection / "data" / f"{key.replace('_', '')}.hdf5"
            if name not in COLOURS or not path.exists():
                continue
            with h5py.File(path, "r") as f:
                rgb = E.decode(f["observation/head_camera/rgb"][0]).astype(np.int16)
            target = np.array(COLOURS[name], np.int16)
            a = int((np.abs(rgb - target).max(-1) < 60).sum())
            b = int((np.abs(rgb[..., ::-1] - target).max(-1) < 60).sum())
            if a > 2 * b + 20:
                votes["rgb"] += 1
            elif b > 2 * a + 20:
                votes["swapped"] += 1
    decided = votes["rgb"] + votes["swapped"]
    ok = None if decided < 5 else votes["rgb"] > 4 * votes["swapped"]
    report("colours", ok, f"mat colour found as named in {votes['rgb']} frames, swapped in {votes['swapped']}")


def check_rows(episode: E.Episode, report: Report) -> None:
    pose, _ = E.read(episode.path, images=False)
    back = np.stack([demo.action_row(a) for a in E.action_rows(pose)])
    pos = max(float(np.abs(back[:, 8 * n : 8 * n + 3] - pose[:, 8 * n : 8 * n + 3]).max()) for n in range(2))
    dots = []
    for n in range(2):
        q1, q2 = back[:, 8 * n + 3 : 8 * n + 7], pose[:, 8 * n + 3 : 8 * n + 7]
        dots.append(np.abs((q1 * q2).sum(-1)) / (np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1)))
    dot = float(np.min(dots))
    grip = max(float(np.abs(back[:, 8 * n + 7] - np.clip(pose[:, 8 * n + 7], 0, 1)).max()) for n in range(2))
    ok = pos < 1e-5 and dot > 0.9999 and grip < 1e-6
    report("rows", ok, f"position {pos:.2e} m, rotation |dot| min {dot:.6f}, gripper {grip:.1e}")


def reference_condition(net, prompt_pose, prompt_frames, pose, frames, t: int, device) -> torch.Tensor:
    """The condition as `VectorPolicy` + `Net` compute it, on live images (no cache)."""
    steps = len(prompt_pose) - 1
    groups = -(-steps // CHUNK)
    keep = np.arange(0, groups * CHUNK, CHUNK)
    pobs = {k: v[keep] for k, v in demo.proprio(prompt_pose[:-1]).items()}
    for cam in CAMERAS:
        pobs[cam] = np.moveaxis(features.frames_224(prompt_frames[cam][keep]), -1, 1).astype(np.float32) / 255.0
    action = E.action_rows(prompt_pose)[1:]
    action = np.concatenate([action, np.zeros((groups * CHUNK - steps, ACTION_DIM), np.float32)])

    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(device)

    mem = net.encode_prompt({k: tensor(v) for k, v in pobs.items()}, tensor(action.reshape(groups, CHUNK, -1)))
    rows = [max(t - 1, 0), t]
    obs = {k: v[rows] for k, v in demo.proprio(pose).items()}
    for cam in CAMERAS:
        obs[cam] = np.moveaxis(features.frames_224(frames[cam][rows]), -1, 1).astype(np.float32) / 255.0
    obs = {k: tensor(v) for k, v in obs.items()}
    obs = {k: net.norm[k](v) if k in PROPRIO else v for k, v in obs.items()}
    now = net.tokens(obs).reshape(1, HISTORY * TOKENS, WIDTH)
    x = now + net.obs_pos
    for layer in net.decoder["layers"]:
        x = layer(x, mem)
    return torch.cat([now, x], dim=1).reshape(1, -1)


def live_features(net, frames, rows, device) -> torch.Tensor:
    out = [
        features.camera_features(net, cam, torch.from_numpy(features.frames_224(frames[cam][rows])).to(device))
        for cam in CAMERAS
    ]
    return torch.stack(out, dim=1)  # (rows, cameras, 768)


def ours(net, batch) -> torch.Tensor:
    mem, pad = model.memory(net, batch.prompt_feats, batch.prompt_prop, batch.prompt_act, batch.prompt_len)
    return model.condition(net, batch.hist_feats, batch.hist_prop, mem, pad)


@torch.no_grad()
def check_model(base: str, cache: Path, cache_sw: Path, device: str, report: Report) -> None:
    net = load(base, device=device)
    quiet = lambda s: None  # noqa: E731
    bank = Bank(cache, device, held_out=0.0, log=quiet)
    bank_sw = Bank(cache_sw, device, held_out=0.0, log=quiet)

    # 4. the prompt, against demo.prompt on the same episode
    task0 = int(bank.task[0])
    same = np.flatnonzero(bank.task == task0)
    e, f = int(same[0]), int(same[1])
    t = int(min(20, bank.length[f] - 2))
    b = bank.batch(np.array([f]), np.array([e]), np.array([t]))
    p_pose, p_frames = E.read(bank.paths[e])
    t_pose, t_frames = E.read(bank.paths[f])
    obs, action = demo.prompt(_arrays(p_pose, p_frames))
    n = int(b.prompt_len[0])
    prop_ref = np.concatenate([obs[k] for k in PROPRIO], axis=-1)
    prop_err = float(np.abs(prop_ref - b.prompt_prop[0, :n].cpu().numpy()).max()) if len(prop_ref) == n else float("inf")
    act_err = float(np.abs(action - b.prompt_act[0, :n].cpu().numpy()).max()) if len(action) == n else float("inf")
    report("prompt", prop_err < 1e-6 and act_err < 1e-6,
           f"{n} prompt frames (demo.prompt: {len(action)}); proprioception {prop_err:.1e}, actions {act_err:.1e}")  # fmt: skip

    # 5. cached features against the live ViT
    keep = np.arange(n) * CHUNK
    live = live_features(net, p_frames, keep, device)
    rel = _rel(b.prompt_feats[0, :n], live)
    report("features", rel < 1e-2, f"cache (float16) vs live ViT: max relative error {rel:.1e}")

    # 6. the condition, per sample on live features, from the cache, and in a padded batch
    ref = reference_condition(net, p_pose, p_frames, t_pose, t_frames, t, device)
    b_live = bank.batch(np.array([f]), np.array([e]), np.array([t]))
    b_live.prompt_feats = live.unsqueeze(0)
    b_live.hist_feats = live_features(net, t_frames, [max(t - 1, 0), t], device).unsqueeze(0)
    exact, cached = _rel(ours(net, b_live), ref), _rel(ours(net, b), ref)
    frames_of = -(-(bank.length - 1) // CHUNK)
    other = np.flatnonzero(frames_of != frames_of[e])
    if len(other):
        e2 = int(other[0])
        f2 = int(np.flatnonzero((bank.task == bank.task[e2]) & (np.arange(len(bank.task)) != e2))[0])
        t2 = int(min(5, bank.length[f2] - 2))
        both = ours(net, bank.batch(np.array([f, f2]), np.array([e, e2]), np.array([t, t2])))
        alone = torch.cat([ours(net, bank.batch(np.array([x]), np.array([y]), np.array([s])))
                           for x, y, s in ((f, e, t), (f2, e2, t2))])  # fmt: skip
        padded = _rel(both, alone)
        pad_note = f"padded batch ({frames_of[e]} and {frames_of[e2]} frames) vs alone {padded:.1e}"
    else:
        padded, pad_note = 0.0, "padded batch: every prompt has the same length (not tested)"
    report("forward", exact < 1e-3 and cached < 2e-2 and padded < 1e-3,
           f"vs Net on live images: {exact:.1e} (live features), {cached:.1e} (cache); {pad_note}")  # fmt: skip

    # 7. the U-Net, one timestep per sample
    cond = torch.cat([ref, ref * 0.5])
    x = torch.randn(2, HORIZON, ACTION_DIM, device=device)
    ts = torch.tensor([3, 40], device=device)
    want = torch.cat([net.unet(x[i : i + 1], ts[i], cond[i : i + 1]) for i in range(2)])
    got = model.unet(net.unet, x, ts, cond)
    rel = _rel(got, want)
    report("unet", rel < 1e-4, f"per-sample timesteps vs Unet.forward: {rel:.1e}")

    # 8. colours through the base weights' own loss, and the normalizer's range on the data
    def mean_loss(b: Bank) -> float:
        rng = np.random.default_rng(0)
        g = torch.Generator(device=device)
        g.manual_seed(0)
        return float(torch.cat([model.loss(net, b.sample(4, "train", rng), g) for _ in range(8)]).mean())

    right, swapped = mean_loss(bank), mean_loss(bank_sw)
    normed = net.norm["action"](bank.act)
    outside = float((normed.abs() > 1).float().mean())
    report("sanity", right < swapped and outside < 0.01,
           f"base loss {right:.4f} on RGB vs {swapped:.4f} on swapped colours; "
           f"{outside:.2%} of normalized actions outside [-1, 1]")  # fmt: skip
    del net, bank, bank_sw


def check_policy(weights: str | Path, pairs: list[tuple[str, str]], device: str, report: Report, name: str) -> None:
    from vector_runtime.policy import VectorPolicy

    policy = VectorPolicy(weights=str(Path(weights).resolve()), device=device)
    errors, still, shapes = [], [], []
    for prompt_path, target_path in pairs:
        p_pose, p_frames = E.read(prompt_path)
        pose, frames = E.read(target_path)
        policy.reset(seed=1)
        policy.set_demonstration(_arrays(p_pose, p_frames), {"action_dims": {"ee": demo.ENDPOSE_DIM}})
        out = policy.act(_arrays(pose, frames, row=0))["action"]
        shapes.append(out.shape)
        k = len(out)
        recorded = pose[1 : 1 + k]
        # the arm that moves most in the episode is the one doing the task
        arm = int(np.argmax([np.ptp(pose[:, 8 * a : 8 * a + 3], axis=0).sum() for a in range(2)]))
        cols = slice(8 * arm, 8 * arm + 3)
        errors.append(float(np.linalg.norm(out[:, cols] - recorded[: len(out), cols], axis=-1).mean()))
        still.append(float(np.linalg.norm(pose[0, cols] - recorded[:, cols], axis=-1).mean()))
    policy.close()
    del policy
    gc.collect()
    expected = EXECUTED - (HISTORY - 1)
    finite = all(np.isfinite(e) for e in errors)
    ok = finite and all(s == (expected, demo.ENDPOSE_DIM) for s in shapes) and np.mean(errors) < np.mean(still)
    report(name, ok,
           f"first {expected} actions of {len(pairs)} episodes: acting arm off the recording by "
           f"{100 * np.mean(errors):.1f} cm on average (standing still: {100 * np.mean(still):.1f} cm)")  # fmt: skip


def held_out_fraction(cache: Path) -> float:
    """A held-out fraction that leaves some task two episodes on each side, so validation runs."""
    meta = json.loads((cache / "meta.json").read_text())
    for fraction in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8):
        for info in meta["tasks"].values():
            held = sum(E.held_out(e["key"], fraction) for e in info["episodes"])
            if held >= 2 and len(info["episodes"]) - held >= 2:
                return fraction
    return 0.0


def check_training(base: str, cache: Path, work: Path, device: str, report: Report) -> Path | None:
    from .cli import build_parser

    run_dir = work / "run"
    shutil.rmtree(run_dir, ignore_errors=True)
    common = ["--base", base, "--cache", str(cache), "--run", str(run_dir), "--device", device,
              "--amp", "none", "--batch", "2", "--train-only", "sinks,obs_pos,decoder.layers.5.",
              "--held-out", str(held_out_fraction(cache)), "--val-every", "3", "--val-per-task", "2",
              "--val-batch", "2", "--save-every", "3", "--log-every", "1", "--warmup", "1"]  # fmt: skip
    train.run(build_parser().parse_args(["train", *common, "--steps", "3"]))
    gc.collect()
    train.run(build_parser().parse_args(["train", *common, "--steps", "5", "--resume"]))
    gc.collect()
    log = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in log if "loss" in r]
    finite = all(np.isfinite(r["loss"]) for r in log if "loss" in r)
    listed = {r["step"]: r["val_mean"] for r in snapshots.find([run_dir])}
    kept = sorted(listed) == [3, 5] and all(v is not None for v in listed.values())
    ok = steps == [1, 2, 3, 4, 5] and finite and kept
    vals = sum(1 for r in log if "val" in r)
    report("train", ok, f"logged steps {steps}, losses finite: {finite}, val records {vals}, resume to "
           f"step 5, snapshots with their val loss: {listed}")  # fmt: skip
    if not ok:
        return None
    exports = work / "exports"
    shutil.rmtree(exports, ignore_errors=True)
    pair = [run_dir / "ema-0000003.safetensors", run_dir / "ema-0000005.safetensors"]
    try:
        summaries = export.run_many(base, pair, exports, device=device)
    except ValueError as e:
        report("export", False, str(e))
        return None
    names = sorted(json.loads((exports / "exports.json").read_text()))
    moved = sorted(g for g, rel in summaries[-1]["drift_rel_l2"].items() if rel)
    ok = names == ["run-0000003", "run-0000005"] and all(
        s["check_ok"] and s["weights_sha256"] != s["base_sha256"] for s in summaries
    )
    report("export", ok, f"{names}, vector_runtime.check ok, distinct sha256s "
           f"{[s['weights_sha256'][:12] for s in summaries]}, changed: {', '.join(moved)}")  # fmt: skip
    return Path(summaries[-1]["weights"]) if ok else None


def run(args: argparse.Namespace) -> int:
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(0)
    report = Report()

    found = E.find(args.data, args.tasks)
    if not found:
        raise SystemExit(f"no episodes under {args.data}")
    collections_of: dict[str, set[str]] = {}
    for e in found:
        collections_of.setdefault(e.task, set()).add(f"{e.root}/{e.collection}")
    tasks = sorted(collections_of, key=lambda t: (-len(collections_of[t]), t))[: args.max_tasks]
    picked, seen = [], {}
    for e in found:
        key = (e.root, e.task, e.collection)
        if e.task in tasks and seen.get(key, 0) < args.per_collection:
            picked.append(e)
            seen[key] = seen.get(key, 0) + 1
    print(f"self-test on {len(picked)} episodes of {', '.join(tasks)} ({device})")

    check_colours(picked, report)
    check_rows(picked[0], report)

    cache = features.build(args.base, picked, work / "cache", device=device, batch=32, workers=0)
    gc.collect()
    cache_sw = features.build(args.base, picked, work / "cache-swapped", device=device, batch=32, workers=0, swap_rb=True)
    gc.collect()
    rows = sum(t["rows"] for t in json.loads((cache / "meta.json").read_text())["tasks"].values())
    report("cache", True, f"{len(picked)} episodes, {rows} rows, 2 caches built")

    check_model(args.base, cache, cache_sw, device, report)
    gc.collect()

    by_task: dict[str, list[E.Episode]] = {}
    for e in picked:
        by_task.setdefault(e.task, []).append(e)
    pairs = [(str(eps[0].path), str(eps[-1].path)) for eps in by_task.values() if len(eps) > 1]
    pairs += [(str(eps[-1].path), str(eps[0].path)) for eps in by_task.values() if len(eps) > 1]
    check_policy(args.base, pairs, device, report, "policy")

    exported = check_training(args.base, cache, work, device, report)
    gc.collect()
    if exported is not None:
        check_policy(exported, pairs[:1], device, report, "export served")

    failed = [name for name, ok, _ in report.rows if ok is False]
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0
