"""Frozen camera features: each frame's CLS token from each camera's ViT, computed once.

`Net` reads a camera only through `vision[cam](pixel_norm(crop(image)))[:, 0]`, so while the ViTs
are frozen those 768 numbers per frame and camera stand in for the image wherever training needs
it, the prompt's frames and the live history alike. Frames go in as inference sends them: decoded
RGB, resized to 224 (`demo.resize`), scaled to [0, 1]. A cache is good only for the ViT weights it
was built with (`vision_sha256`), which fine-tuning leaves alone.

    <cache>/meta.json             base weights, ViT fingerprint, each task's episodes and rows
    <cache>/<task>.feats.npy      (rows, cameras, 768) float16
    <cache>/<task>.endpose.npy    (rows, 16) float64
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vector_runtime import demo
from vector_runtime.net import CAMERAS, WIDTH, load

from . import episodes as E

VERSION = 1


def vision_sha256(weights: str | Path) -> str:
    """A fingerprint of the ViT tensors in a weights file."""
    from safetensors import safe_open

    digest = hashlib.sha256()
    with safe_open(str(weights), framework="np") as f:
        for key in sorted(k for k in f.keys() if k.startswith("vision.")):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(f.get_tensor(key)).tobytes())
    return digest.hexdigest()


@torch.no_grad()
def camera_features(net: Any, cam: str, images: torch.Tensor) -> torch.Tensor:
    """(N, 224, 224, 3) uint8 -> (N, 768): the token `Net.tokens` takes from `cam` per image."""
    x = images.permute(0, 3, 1, 2).float().div(255.0)
    return net.vision[cam](net.pixel_norm(net.crop(x)))[:, 0, :]


def frames_224(frames: np.ndarray, swap_rb: bool = False) -> np.ndarray:
    """(T, H, W, 3) RGB -> (T, 224, 224, 3), resized as inference resizes a live frame."""
    out = np.stack([demo.resize(f) for f in frames])
    return np.ascontiguousarray(out[..., ::-1]) if swap_rb else out


class _Frames(torch.utils.data.Dataset):
    """An episode's endpose and resized frames, read in a loader worker."""

    def __init__(self, eps: Sequence[E.Episode], swap_rb: bool) -> None:
        self.eps = list(eps)
        self.swap_rb = swap_rb

    def __len__(self) -> int:
        return len(self.eps)

    def __getitem__(self, i: int):
        pose, frames = E.read(self.eps[i].path)
        return i, pose, {cam: frames_224(frames[cam], self.swap_rb) for cam in CAMERAS}


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def build(
    base: str | Path,
    eps: Sequence[E.Episode],
    out: str | Path,
    *,
    device: str = "cuda",
    batch: int = 256,
    workers: int = 8,
    amp: bool = False,
    swap_rb: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Cache `eps` under `out`, task by task; a task already cached with the same episodes is
    kept. `amp` runs the ViTs in bf16 (faster, slightly off fp32 inference); `swap_rb` feeds them
    red and blue swapped (for the self-test only)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    fingerprint = vision_sha256(base)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("vision_sha256") != fingerprint or meta.get("swap_rb", False) != swap_rb:
            raise ValueError(f"{out} was built from other ViT weights or frames; use a new directory")
    else:
        meta = {
            "version": VERSION,
            "base": str(Path(base).resolve()),
            "vision_sha256": fingerprint,
            "dtype": "float16",
            "amp": amp,
            "swap_rb": swap_rb,
            "tasks": {},
        }
    by_task: dict[str, list[E.Episode]] = {}
    for e in eps:
        by_task.setdefault(e.task, []).append(e)

    net = load(base, device=device)
    dev = torch.device(device)
    for task, task_eps in sorted(by_task.items()):
        done = meta["tasks"].get(task)
        if done and [e["key"] for e in done["episodes"]] == [e.key for e in task_eps]:
            log(f"{task}: already cached ({done['rows']} rows)")
            continue
        lengths = [E.length(e.path) for e in task_eps]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        rows = int(sum(lengths))
        feats = np.lib.format.open_memmap(
            out / f"{task}.feats.npy", mode="w+", dtype=np.float16, shape=(rows, len(CAMERAS), WIDTH)
        )
        poses = np.zeros((rows, demo.ENDPOSE_DIM), np.float64)
        loader = torch.utils.data.DataLoader(
            _Frames(task_eps, swap_rb),
            batch_size=None,
            num_workers=workers,
            prefetch_factor=2 if workers else None,
        )
        started = time.monotonic()
        for k, (i, pose, frames) in enumerate(loader):
            s, n = int(starts[i]), lengths[i]
            pose = np.asarray(pose)
            if len(pose) != n:
                raise ValueError(f"{task_eps[i].path}: {len(pose)} rows, expected {n}")
            poses[s : s + n] = pose
            for c, cam in enumerate(CAMERAS):
                images = torch.as_tensor(frames[cam])
                for a in range(0, n, batch):
                    chunk = images[a : a + batch].to(dev, non_blocking=True)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                        f = camera_features(net, cam, chunk)
                    feats[s + a : s + a + len(chunk), c] = f.float().cpu().numpy()
            if (k + 1) % 50 == 0 or k + 1 == len(task_eps):
                rate = (k + 1) / (time.monotonic() - started)
                log(f"{task}: {k + 1}/{len(task_eps)} episodes ({rate:.2f}/s)")
        feats.flush()
        del feats
        np.save(out / f"{task}.endpose.npy", poses)
        meta["tasks"][task] = {
            "rows": rows,
            "episodes": [
                {"key": e.key, "collection": e.collection, "path": str(e.path),
                 "start": int(s), "length": int(n)}
                for e, s, n in zip(task_eps, starts, lengths, strict=True)
            ],
        }  # fmt: skip
        _write_json(meta_path, meta)
    return out
