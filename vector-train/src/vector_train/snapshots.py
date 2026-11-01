"""The snapshots of one or more runs, each with its EMA validation loss against the base's (the
step-0 record): the candidates `export` turns into submissions. Nothing here deletes a snapshot.

A lower validation loss is only a hint: duels decide. Snapshots of one run are close relatives, so
candidates from different runs (seeds, task weights, learning rates) differ more.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def _vals(run: Path) -> tuple[dict | None, dict[int, dict]]:
    """The base's validation record and the EMA's by step (the last wins, as after a resume)."""
    base, by_step = None, {}
    log = run / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            record = json.loads(line)
            if record.get("base"):
                base = record
            elif record.get("ema"):
                by_step[record["step"]] = record
    return base, by_step


def find(runs: Iterable[str | Path]) -> list[dict]:
    """One row per `<run>/ema-*.safetensors`: run, step, path, val_mean, vs_base (relative change of
    the mean), tasks_better ((tasks below the base's loss, tasks compared))."""
    rows = []
    for run in (Path(r).resolve() for r in runs):
        base, by_step = _vals(run)
        for path in sorted(run.glob("ema-*.safetensors")):
            step = int(path.stem.removeprefix("ema-"))
            row = {"run": run.name, "step": step, "path": str(path),
                   "val_mean": None, "vs_base": None, "tasks_better": None}  # fmt: skip
            val = by_step.get(step)
            if val:
                row["val_mean"] = val["val_mean"]
                if base:
                    row["vs_base"] = val["val_mean"] / base["val_mean"] - 1
                    common = [t for t in val["val"] if t in base["val"]]
                    better = sum(val["val"][t] < base["val"][t] for t in common)
                    row["tasks_better"] = (better, len(common))
            rows.append(row)
    return rows


def show(rows: list[dict]) -> None:
    """The rows as a table, lowest validation loss first (unvalidated ones last)."""
    ranked = sorted(rows, key=lambda r: (r["val_mean"] is None, r["val_mean"] or 0.0, r["run"], r["step"]))
    print(f"{'run':16} {'step':>8} {'val loss':>10} {'vs base':>8} {'tasks better':>13}  path")
    for r in ranked:
        val = f"{r['val_mean']:.5f}" if r["val_mean"] is not None else "-"
        vs = f"{r['vs_base']:+.2%}" if r["vs_base"] is not None else "-"
        better = "{}/{}".format(*r["tasks_better"]) if r["tasks_better"] else "-"
        print(f"{r['run']:16} {r['step']:>8} {val:>10} {vs:>8} {better:>13}  {r['path']}")
