"""RoboTwin HDF5 episodes, read the way the network sees an episode.

An episode is `<root>/<task>/<collection>/data/episode<N>.hdf5`: the layout of the RoboTwin 2.0
dataset archives and of RoboTwin's own collector. Its `endpose` rows are the benchmark's (per arm, a
pose `[x y z qw qx qy qz]` then the gripper) and each camera frame is stored as JPEG bytes.

RoboTwin encoded the frames with OpenCV straight from RGB arrays, so OpenCV's decoder hands RGB
back while any other JPEG decoder (PIL, torchvision) hands back red and blue swapped: the mat
colours `scene_info.json` names only match after `cv2.imdecode`. The benchmark serves the policy
RGB, so frames are decoded with OpenCV and used as they come.

Every conversion of a row is `vector_runtime.demo`'s, the code the validator runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vector_runtime import demo
from vector_runtime.net import CAMERAS, PROPRIO

#: Each of the network's cameras, as an episode names it.
EPISODE_CAMERAS = demo.BENCH_CAMERAS


def _slices() -> dict[str, tuple[int, int]]:
    out, at = {}, 0
    for key, width in PROPRIO.items():
        out[key] = (at, at + width)
        at += width
    return out


#: Where each proprioception key sits in a `proprio_rows` row, in `PROPRIO` (token) order.
PROPRIO_SLICES = _slices()
PROPRIO_DIM = sum(PROPRIO.values())


@dataclass(frozen=True)
class Episode:
    root: str
    task: str
    collection: str
    path: Path

    @property
    def key(self) -> str:
        """Unique across roots; also what decides the held-out split."""
        return f"{self.root}/{self.task}/{self.collection}/{self.path.stem}"


def _number(path: Path) -> int:
    return int(path.stem[len("episode") :])


def find(roots: Iterable[str | Path], tasks: Iterable[str] | None = None) -> list[Episode]:
    """Every episode under `roots`, in root, task, collection and episode-number order."""
    wanted = set(tasks) if tasks else None
    found: list[Episode] = []
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")
        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if wanted is not None and task_dir.name not in wanted:
                continue
            paths = sorted(
                task_dir.glob("*/data/episode*.hdf5"),
                key=lambda p: (p.parent.parent.name, _number(p)),
            )
            found += [Episode(root.name, task_dir.name, p.parent.parent.name, p) for p in paths]
    return found


def endpose(f) -> np.ndarray:
    """An open episode's (T, 16) float64 endpose rows: per arm, the pose then the gripper."""
    parts = []
    for arm in demo.ARMS:
        parts += [f[f"endpose/{arm}_endpose"][()], f[f"endpose/{arm}_gripper"][()][:, None]]
    return np.concatenate(parts, axis=1).astype(np.float64)


def decode(jpeg: bytes) -> np.ndarray:
    """A frame as RoboTwin stored it -> (H, W, 3) uint8 RGB."""
    import cv2

    rgb = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("a stored frame is not a JPEG")
    return rgb


def read(path: str | Path, images: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(endpose, {camera: (T, H, W, 3) uint8 RGB})`, frames at the recorded size; no frames
    without `images`."""
    import h5py

    with h5py.File(path, "r") as f:
        pose = endpose(f)
        frames = {}
        if images:
            for cam in CAMERAS:
                stored = f[f"observation/{EPISODE_CAMERAS[cam]}/rgb"][()]
                frames[cam] = np.stack([decode(b) for b in stored])
                if len(frames[cam]) != len(pose):
                    raise ValueError(f"{path}: {cam} has {len(frames[cam])} frames, {len(pose)} rows")
    return pose, frames


def length(path: str | Path) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        return int(f["endpose/left_endpose"].shape[0])


def proprio_rows(pose: np.ndarray) -> np.ndarray:
    """(T, 16) endpose -> (T, 20) float32 proprioception, keys in `PROPRIO` order."""
    parts = demo.proprio(pose)
    return np.concatenate([parts[key] for key in PROPRIO], axis=-1)


def action_rows(pose: np.ndarray) -> np.ndarray:
    """(T, 16) endpose -> (T, 20) float32 actions, per arm position, rot6d and gripper, as
    `demo.prompt` builds a demonstration's actions."""
    parts = []
    for n in range(len(demo.ARMS)):
        arm = pose[:, 8 * n : 8 * n + 8]
        parts += [arm[:, :3], demo.rot6d(arm[:, 3:7]), arm[:, 7:8]]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def held_out(key: str, fraction: float) -> bool:
    """Whether an episode is in the validation split: a fixed function of its key."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction
