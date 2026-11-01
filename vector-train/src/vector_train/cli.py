"""`vector-train`: fine-tune `vector_v1.1` weights.

    vector-train cache     --base KING.safetensors --data ROOT [--data ROOT ...] --out CACHE
    vector-train train     --base KING.safetensors --cache CACHE --run RUN [...]
    vector-train snapshots --run RUN [RUN ...]
    vector-train export    --base KING.safetensors --snapshot RUN/ema-NNNNNNN.safetensors [...] --out DIR
    vector-train selftest  --base KING.safetensors --data ROOT [--data ROOT ...] --work DIR

A data root holds `<task>/<collection>/data/episode*.hdf5`, as the RoboTwin 2.0 archives unpack and
as RoboTwin's collector writes them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _csv(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def _weight(value: str) -> tuple[str, float]:
    task, sep, w = value.partition("=")
    if not sep or not task:
        raise argparse.ArgumentTypeError(f"{value!r} is not TASK=WEIGHT")
    return task, float(w)


def cmd_cache(args: argparse.Namespace) -> int:
    from . import episodes, features

    eps = episodes.find(args.data, args.tasks)
    if args.limit:
        kept, seen = [], {}
        for e in eps:
            key = (e.root, e.task, e.collection)
            if seen.get(key, 0) < args.limit:
                kept.append(e)
                seen[key] = seen.get(key, 0) + 1
        eps = kept
    if not eps:
        print(f"error: no episodes under {args.data}", file=sys.stderr)
        return 2
    print(f"{len(eps)} episodes of {len({e.task for e in eps})} tasks")
    features.build(
        args.base, eps, args.out, device=args.device, batch=args.batch, workers=args.workers, amp=args.amp
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import train

    args.task_weight = dict(args.task_weight or [])
    train.run(args)
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    from . import snapshots

    rows = snapshots.find(args.run)
    if not rows:
        print(f"error: no ema-*.safetensors in {args.run}", file=sys.stderr)
        return 2
    snapshots.show(rows)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from . import export

    export.run_many(args.base, args.snapshot, args.out, device=args.policy_device)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from . import selftest

    return selftest.run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vector-train", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cache", help="compute every episode's frozen camera features once")
    c.add_argument("--base", required=True, help="the weights whose ViTs to use (the king's)")
    c.add_argument("--data", action="append", required=True, help="a data root (repeatable)")
    c.add_argument("--out", required=True, help="the cache directory")
    c.add_argument("--tasks", type=_csv, default=None, help="only these tasks (comma-separated)")
    c.add_argument("--limit", type=int, default=None, help="at most N episodes per task and collection")
    c.add_argument("--device", default="cuda")
    c.add_argument("--batch", type=int, default=256, help="frames per ViT batch")
    c.add_argument("--workers", type=int, default=8, help="processes decoding episodes")
    c.add_argument("--amp", action="store_true", help="bf16 ViTs: faster, slightly off fp32 inference")
    c.set_defaults(func=cmd_cache)

    t = sub.add_parser("train", help="fine-tune on a cache")
    t.add_argument("--base", required=True, help="the weights to start from (the king's)")
    t.add_argument("--cache", required=True)
    t.add_argument("--run", required=True, help="where logs, snapshots and the resume state go")
    t.add_argument("--device", default="cuda")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--min-lr-ratio", type=float, default=0.1, help="the cosine's floor, as a share of --lr")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--weight-decay", type=float, default=1e-6)
    t.add_argument("--clip", type=float, default=1.0, help="gradient norm clip")
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    t.add_argument("--freeze", type=_csv, default=[], help="parameter-name prefixes to keep fixed")
    t.add_argument("--train-only", type=_csv, default=[], help="train only these prefixes")
    t.add_argument("--tasks", type=_csv, default=None)
    t.add_argument("--task-weight", type=_weight, action="append", metavar="TASK=W",
                   help="sample TASK W times as often (default 1 each)")  # fmt: skip
    t.add_argument("--held-out", type=float, default=0.05, help="share of episodes kept for validation")
    t.add_argument("--same-collection", action="store_true",
                   help="prompt each target with an episode of its own collection")  # fmt: skip
    t.add_argument("--val-every", type=int, default=1000)
    t.add_argument("--val-per-task", type=int, default=64)
    t.add_argument("--val-batch", type=int, default=64)
    t.add_argument("--save-every", type=int, default=2000,
                   help="keep an EMA snapshot (about 2 GB), validated, every N steps")  # fmt: skip
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    t.set_defaults(func=cmd_train)

    n = sub.add_parser("snapshots", help="list runs' snapshots, lowest validation loss first")
    n.add_argument("--run", action="extend", nargs="+", required=True, help="run directories")
    n.set_defaults(func=cmd_snapshots)

    e = sub.add_parser("export", help="merge snapshots into the base and check each")
    e.add_argument("--base", required=True)
    e.add_argument("--snapshot", action="extend", nargs="+", required=True,
                   help="<run>/ema-NNNNNNN.safetensors files (several, or repeat the flag)")  # fmt: skip
    e.add_argument("--out", required=True, help="each export goes to <out>/<run>-<step>/")
    e.add_argument("--policy-device", default="cuda:0", help="the device local/policy.yaml asks for")
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("selftest", help="check the pipeline against the validator's code")
    s.add_argument("--base", required=True)
    s.add_argument("--data", action="append", required=True)
    s.add_argument("--work", default="selftest-work")
    s.add_argument("--tasks", type=_csv, default=None)
    s.add_argument("--max-tasks", type=int, default=2)
    s.add_argument("--per-collection", type=int, default=2, help="episodes per task and collection")
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
