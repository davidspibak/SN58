"""Training samples from a feature cache, held in memory on the training device.

A sample is a target episode of a task, another episode of the same task as its prompt, and a step
`t` of the target. The network sees the target's frames `t-1` and `t` (frame 0 twice at the start,
as `VectorPolicy` pads its history) and is taught the HORIZON actions after `t`: endpose rows `t+1`
on, the last row repeated past the end. The prompt is `demo.prompt` without its JPEG round trip,
which stored frames went through when they were recorded: one frame every CHUNK steps from row 0,
and every action (rows 1 on) CHUNK to a frame, the last group padded with zeros.

Prompts of different lengths share a batch padded to the longest; `prompt_len` says where each one
ends and the training forward masks the rest, so no sample sees padding (inference never pads).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vector_runtime.net import ACTION_DIM, CHUNK, HISTORY, HORIZON, MAX_PROMPT

from . import episodes as E


@dataclass
class Batch:
    hist_feats: torch.Tensor  # (B, HISTORY, cameras, 768) float32
    hist_prop: torch.Tensor  # (B, HISTORY, 20)
    target: torch.Tensor  # (B, HORIZON, ACTION_DIM), raw (normalized in the forward)
    prompt_feats: torch.Tensor  # (B, P, cameras, 768) float32
    prompt_prop: torch.Tensor  # (B, P, 20)
    prompt_act: torch.Tensor  # (B, P, CHUNK, ACTION_DIM), raw, zeros past the demonstration
    prompt_len: torch.Tensor  # (B,) int64: each prompt's frames; the rest of P is padding
    task: torch.Tensor  # (B,) int64 index into `Bank.tasks`

    def __len__(self) -> int:
        return self.target.shape[0]


class Bank:
    """Every cached row of the chosen tasks on `device`: camera features, proprioception and
    action, with the episodes they belong to and the train/val split."""

    def __init__(
        self,
        cache: str | Path,
        device: str | torch.device,
        *,
        tasks: Sequence[str] | None = None,
        held_out: float = 0.05,
        task_weights: Mapping[str, float] | None = None,
        same_collection: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        cache = Path(cache)
        self.meta = json.loads((cache / "meta.json").read_text())
        self.device = torch.device(device)
        self.tasks = sorted(t for t in self.meta["tasks"] if not tasks or t in tasks)
        if not self.tasks:
            raise ValueError(f"{cache} holds none of the tasks {tasks}")
        unknown = set(task_weights or {}) - set(self.tasks)
        if unknown:
            raise ValueError(f"task weights for tasks not in the cache: {sorted(unknown)}")
        self.same_collection = same_collection

        feats, prop, act = [], [], []
        start, length, task, collection, held, paths = [], [], [], [], [], []
        collections: dict[str, int] = {}
        offset = 0
        for ti, name in enumerate(self.tasks):
            info = self.meta["tasks"][name]
            f = np.load(cache / f"{name}.feats.npy", mmap_mode="r")
            pose = np.load(cache / f"{name}.endpose.npy")
            if len(f) != info["rows"] or len(pose) != info["rows"]:
                raise ValueError(f"{cache}: {name}'s arrays do not hold {info['rows']} rows")
            feats.append(torch.from_numpy(np.ascontiguousarray(f)))
            prop.append(torch.from_numpy(E.proprio_rows(pose)))
            act.append(torch.from_numpy(E.action_rows(pose)))
            for e in info["episodes"]:
                if e["length"] < 2:
                    continue
                start.append(offset + e["start"])
                length.append(e["length"])
                task.append(ti)
                collection.append(collections.setdefault(e["collection"], len(collections)))
                held.append(E.held_out(e["key"], held_out))
                paths.append(e["path"])
            offset += info["rows"]
        self.feats = torch.cat(feats).to(self.device)  # float16
        self.prop = torch.cat(prop).to(self.device)
        self.act = torch.cat(act).to(self.device)
        self.start = np.asarray(start, np.int64)
        self.length = np.asarray(length, np.int64)
        self.task = np.asarray(task, np.int64)
        self.collection = np.asarray(collection, np.int64)
        #: Each episode's HDF5 file, for checks against the source.
        self.paths = paths
        held_mask = np.asarray(held, bool)
        #: A demonstration longer than MAX_PROMPT frames is refused at inference; never a prompt.
        self.prompt_ok = -(-(self.length - 1) // CHUNK) <= MAX_PROMPT

        weights = np.array([float((task_weights or {}).get(t, 1.0)) for t in self.tasks])
        self.pools: dict[str, list[np.ndarray]] = {}
        self.weights: dict[str, np.ndarray | None] = {}
        for split, mask in (("train", ~held_mask), ("val", held_mask)):
            pools = [np.flatnonzero(mask & (self.task == ti)) for ti in range(len(self.tasks))]
            usable = np.array([len(p) >= 2 and self.prompt_ok[p].any() for p in pools])
            w = np.where(usable, weights, 0.0)
            self.pools[split] = pools
            self.weights[split] = w / w.sum() if w.sum() > 0 else None
        for ti, name in enumerate(self.tasks):
            log(
                f"  {name:24} train {len(self.pools['train'][ti]):4} episodes, "
                f"val {len(self.pools['val'][ti]):3}, weight {weights[ti]:g}"
            )
        log(f"bank: {len(self.start)} episodes, {offset} rows, {len(collections)} collections")

    def usable(self, split: str) -> list[int]:
        """The tasks `split` can draw samples from (two or more episodes)."""
        w = self.weights[split]
        return [] if w is None else [int(i) for i in np.flatnonzero(w > 0)]

    def _pair(self, split: str, ti: int, rng: np.random.Generator) -> tuple[int, int]:
        pool = self.pools[split][ti]
        target = int(rng.choice(pool))
        candidates = pool[(pool != target) & self.prompt_ok[pool]]
        if self.same_collection:
            same = candidates[self.collection[candidates] == self.collection[target]]
            if len(same):
                candidates = same
        if not len(candidates):
            raise ValueError(f"{self.tasks[ti]} has no {split} episode to prompt with")
        return target, int(rng.choice(candidates))

    def sample(self, n: int, split: str, rng: np.random.Generator, task: int | None = None) -> Batch:
        """`n` samples of `split`, tasks drawn by weight (or all of `task`)."""
        w = self.weights[split]
        if w is None:
            raise ValueError(f"no task has two {split} episodes")
        tasks = np.full(n, task) if task is not None else rng.choice(len(self.tasks), size=n, p=w)
        pairs = [self._pair(split, int(ti), rng) for ti in tasks]
        targets = np.array([p[0] for p in pairs], np.int64)
        prompts = np.array([p[1] for p in pairs], np.int64)
        steps = rng.integers(0, self.length[targets] - 1)
        return self.batch(targets, prompts, steps)

    def fixed(self, split: str, per_task: int, batch: int, seed: int) -> dict[str, list[Batch]]:
        """The same `per_task` samples of every usable task each call: a validation set."""
        out = {}
        for ti in self.usable(split):
            rng = np.random.default_rng([seed, ti])
            sizes = [min(batch, per_task - a) for a in range(0, per_task, batch)]
            out[self.tasks[ti]] = [self.sample(s, split, rng, task=ti) for s in sizes]
        return out

    def batch(self, targets: np.ndarray, prompts: np.ndarray, steps: np.ndarray) -> Batch:
        """The samples (target episode, prompt episode, step), gathered on the device."""
        t0, tn = self.start[targets], self.length[targets]
        hist = t0[:, None] + np.maximum(steps[:, None] + np.arange(1 - HISTORY, 1), 0)
        future = t0[:, None] + np.minimum(steps[:, None] + 1 + np.arange(HORIZON), tn[:, None] - 1)

        p0, demo_steps = self.start[prompts], self.length[prompts] - 1
        frames = -(-demo_steps // CHUNK)
        width = int(frames.max())
        # Frames past a prompt's end point at its last kept row; they are masked.
        keep = p0[:, None] + np.minimum(np.arange(width) * CHUNK, (frames[:, None] - 1) * CHUNK)
        i = np.arange(width * CHUNK)
        act_rows = p0[:, None] + 1 + np.minimum(i, demo_steps[:, None] - 1)
        act_live = i[None, :] < demo_steps[:, None]

        def at(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

        b = len(targets)
        prompt_act = self.act[at(act_rows)] * at(act_live)[..., None].to(self.act.dtype)
        return Batch(
            hist_feats=self.feats[at(hist)].float(),
            hist_prop=self.prop[at(hist)],
            target=self.act[at(future)],
            prompt_feats=self.feats[at(keep)].float(),
            prompt_prop=self.prop[at(keep)],
            prompt_act=prompt_act.reshape(b, width, CHUNK, ACTION_DIM),
            prompt_len=at(frames.astype(np.int64)),
            task=at(self.task[targets]),
        )
