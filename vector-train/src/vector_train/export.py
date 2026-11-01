"""A snapshot -> a submission: the base weights with the snapshot's tensors in place, float32,
checked by `vector_runtime.check` exactly as the validator checks it.

    <out>/submission/model.safetensors   the only file a submission repo may hold
    <out>/local/policy.yaml              serves it in a local duel (`--local REPO=<out>/local`)
    <out>/export.json                    sha256s, the drift from the base per module, the check report

`run_many` exports several snapshots, each to `<root>/<run>-<step>/`, and indexes every export
under `<root>` in `<root>/exports.json`. The validator refuses weights identical to an earlier
commitment's, so no two exports there may share a sha256.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "unet" else parts[0]


def run(base: str | Path, snapshot: str | Path, out: str | Path, device: str = "cuda:0") -> dict:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from vector_runtime.check import check

    base, snapshot, out = Path(base).resolve(), Path(snapshot).resolve(), Path(out).resolve()
    with safe_open(str(base), framework="pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    with safe_open(str(snapshot), framework="pt") as f:
        made = f.metadata() or {}
    update = load_file(str(snapshot))
    unknown = sorted(set(update) - set(state))
    if unknown:
        raise ValueError(f"the snapshot holds tensors the architecture does not: {unknown[:5]}")

    drift: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, tensor in update.items():
        old = state[name]
        if tensor.shape != old.shape:
            raise ValueError(f"{name}: {tuple(tensor.shape)} in the snapshot, {tuple(old.shape)} in the base")
        new = tensor.to(torch.float32).contiguous()
        if not torch.isfinite(new).all():
            raise ValueError(f"{name} holds non-finite values")
        d = drift[group(name)]
        d[0] += float((new.double() - old.double()).pow(2).sum())
        d[1] += float(old.double().pow(2).sum())
        state[name] = new

    weights = out / "submission" / "model.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    for stale in weights.parent.iterdir():
        if stale.name != weights.name:
            raise ValueError(f"{weights.parent} holds {stale.name}: a submission holds only model.safetensors")
    save_file(state, str(weights))
    del state, update

    report = check(weights.parent, compute_sha256=True)
    base_sha = sha256(base)
    summary = {
        "weights": str(weights),
        "weights_sha256": report.weights_sha256,
        "base": str(base),
        "base_sha256": base_sha,
        "snapshot": str(snapshot),
        "step": int(made["step"]) if "step" in made else None,
        "val_mean": float(made["val_mean"]) if "val_mean" in made else None,
        "check_ok": report.ok,
        "check_errors": report.errors,
        "check_warnings": report.warnings,
        "file_bytes": report.file_bytes,
        "drift_rel_l2": {g: (s / r) ** 0.5 if r else None for g, (s, r) in sorted(drift.items())},
    }
    if not report.ok:
        raise ValueError(f"the export fails the validator's check: {report.errors}")
    if report.weights_sha256 == base_sha:
        raise ValueError("the export is byte-identical to the base: it would be refused as a duplicate")

    local = out / "local"
    local.mkdir(exist_ok=True)
    (local / "policy.yaml").write_text(
        "api: 1\n"
        "policy: vector_runtime.policy:VectorPolicy\n"
        f"kwargs: {{weights: {json.dumps(weights.as_posix())}, device: {json.dumps(device)}}}\n"
    )
    (out / "export.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {weights} (sha256 {report.weights_sha256})")
    for g, rel in summary["drift_rel_l2"].items():
        print(f"  drift {g:28} {rel:.5f}")
    return summary


def name(snapshot: str | Path) -> str:
    """`run1/ema-0010000.safetensors` -> `run1-0010000`, the directory its export goes to."""
    snapshot = Path(snapshot).resolve()
    return f"{snapshot.parent.name}-{snapshot.stem.removeprefix('ema-')}"


def run_many(
    base: str | Path, snapshots: list[str | Path], root: str | Path, device: str = "cuda:0"
) -> list[dict]:
    """Each snapshot exported to `<root>/<run>-<step>/`; `<root>/exports.json` updated. Raises when
    two exports under `root` hold identical weights, since only one of them could be submitted."""
    root = Path(root).resolve()
    names = [name(s) for s in snapshots]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f"snapshots that would export to the same directory: {clash}")
    index_path = root / "exports.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index = {n: s for n, s in index.items() if Path(s["weights"]).exists()}

    for n, snapshot in zip(names, snapshots, strict=True):
        print(f"[{n}]")
        index[n] = run(base, snapshot, root / n, device=device)
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))

    by_sha: dict[str, list[str]] = defaultdict(list)
    for n, s in index.items():
        by_sha[s["weights_sha256"]].append(n)
    same = [sorted(ns) for ns in by_sha.values() if len(ns) > 1]
    if same:
        raise ValueError(f"exports with identical weights (only one could be submitted): {same}")

    print(f"\n{len(index)} exports in {root}, all distinct:")
    for n, s in sorted(index.items()):
        val = f"{s['val_mean']:.5f}" if s.get("val_mean") is not None else "-"
        print(f"  {n:24} val {val:>9}  sha256 {s['weights_sha256'][:16]}  {s['weights']}")
    return [index[n] for n in names]
