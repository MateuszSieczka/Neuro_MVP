"""
_sweep_all.py — Single entry point for all benchmarks.

Replaces _sweep.py (CartPole) and _sweep_mc.py (MountainCar).

Usage
-----
    # Run all registered tasks with default seeds:
    python _sweep_all.py

    # Run a specific task:
    python _sweep_all.py CartPole-v1

    # Run multiple tasks with a custom seed set:
    python _sweep_all.py CartPole-v1 MountainCar-v0 --seeds 1 42 99
"""

from __future__ import annotations

import argparse
import sys

from arena.benchmark import Benchmark
from arena.task_config import REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNN benchmarks. No task-specific logic here — "
                    "all configuration lives in arena/task_config.py."
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        default=list(REGISTRY.keys()),
        metavar="TASK",
        help="Task env_ids to benchmark (default: all registered tasks).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seeds (default: canonical set from Benchmark).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    unknown = [t for t in args.tasks if t not in REGISTRY]
    if unknown:
        print(f"ERROR: Unknown task(s): {unknown}")
        print(f"Registered tasks: {list(REGISTRY.keys())}")
        sys.exit(1)

    results = Benchmark.compare(args.tasks, seeds=args.seeds)

    # Final cross-task summary
    print("\n" + "=" * 62)
    print("  CROSS-TASK SUMMARY")
    print("=" * 62)
    for env_id, result in results.items():
        status = "SOLVED" if result.solve_rate >= 0.5 else "partial"
        print(
            f"  {env_id:<20} | "
            f"score={result.mean_score:8.1f} | "
            f"solved={result.n_solved}/{result.n_seeds} | "
            f"{status}"
        )
    print("=" * 62)


if __name__ == "__main__":
    main()