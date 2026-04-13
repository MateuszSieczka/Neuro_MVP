"""
arena.benchmark — Unified, task-agnostic benchmark runner.

Every task runs through the identical code path:
  GymEnv → Trainer → TrainResult → BenchmarkResult

The Benchmark class never reads the env_id after constructing GymEnv.
It knows nothing about CartPole vs MountainCar semantics — it only
knows "run N episodes, measure mean reward over last K, compare to
threshold".

Usage
-----
    from arena.benchmark import Benchmark
    from arena import task_config

    results = Benchmark.run("CartPole-v1", seeds=[1, 17, 42])
    results = Benchmark.run("MountainCar-v0", seeds=[1, 17, 42])
    Benchmark.print_summary(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from arena.agent_factory import make_agent
from arena.core import Trainer, TrainResult
from arena.gym_env import GymEnv
from arena.task_config import TaskConfig, get as get_task


# =====================================================================
# Benchmark configuration
# =====================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark evaluation parameters (previously hardcoded)."""
    default_seeds: tuple[int, ...] = (1, 17, 42, 99, 145, 256, 500)


_DEFAULT_BENCHMARK_CFG = BenchmarkConfig()


# =====================================================================
# Result containers
# =====================================================================

@dataclass
class SeedResult:
    """Outcome for a single (task, seed) run."""
    seed: int
    final_score: float
    solved: bool
    first_solved_ep: int | None      # None if never solved
    train_result: TrainResult
    elapsed_s: float


@dataclass
class BenchmarkResult:
    """Aggregated outcome across all seeds for one task."""
    task: TaskConfig
    seed_results: list[SeedResult] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [r.final_score for r in self.seed_results]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        return float(np.std(self.scores)) if self.scores else 0.0

    @property
    def n_solved(self) -> int:
        return sum(1 for r in self.seed_results if r.solved)

    @property
    def n_seeds(self) -> int:
        return len(self.seed_results)

    @property
    def solve_rate(self) -> float:
        return self.n_solved / self.n_seeds if self.n_seeds else 0.0

    @property
    def median_first_solved(self) -> float | None:
        """Median episode at which seeds first crossed threshold. None if none solved."""
        solved_eps = [r.first_solved_ep for r in self.seed_results if r.first_solved_ep is not None]
        return float(np.median(solved_eps)) if solved_eps else None


# =====================================================================
# Benchmark runner
# =====================================================================

class Benchmark:
    """
    Runs a reproducible multi-seed benchmark for any registered task.

    The runner is intentionally stateless (all class methods) so it
    cannot accumulate task-specific state between calls.
    """

    @classmethod
    def run(
        cls,
        env_id: str,
        seeds: Sequence[int] | None = None,
        verbose: bool = True,
        benchmark_cfg: BenchmarkConfig | None = None,
    ) -> BenchmarkResult:
        """
        Execute the full benchmark for *env_id* across all seeds.

        Parameters
        ----------
        env_id : str
            Key into the task registry (e.g. 'CartPole-v1').
        seeds : sequence of int, optional
            Random seeds to use.  Defaults to the canonical set.
        verbose : bool
            Print per-seed progress lines.
        benchmark_cfg : BenchmarkConfig, optional
            Benchmark evaluation parameters.
        """
        bcfg = benchmark_cfg or _DEFAULT_BENCHMARK_CFG
        seeds = list(seeds) if seeds is not None else list(bcfg.default_seeds)

        task = get_task(env_id)
        result = BenchmarkResult(task=task)

        if verbose:
            threshold_str = f"{task.solved_threshold:+.0f}"
            print(
                f"\n{'='*62}\n"
                f"  Task   : {task.env_id}\n"
                f"  Budget : {task.n_episodes} episodes × {task.max_steps} steps\n"
                f"  Solved : mean(last {task.eval_window}) > {threshold_str}\n"
                f"  Seeds  : {seeds}\n"
                f"{'='*62}"
            )

        for seed in seeds:
            sr = cls._run_seed(task, seed, verbose=verbose)
            result.seed_results.append(sr)

        if verbose:
            cls.print_summary(result)

        return result

    @classmethod
    def _run_seed(cls, task: TaskConfig, seed: int, verbose: bool) -> SeedResult:
        np.random.seed(seed)

        if task.env_class is not None:
            env = task.env_class()
        else:
            env = GymEnv(
                task.env_id,
                normalize=True,
                fixed_bounds=task.obs_bounds,
                reward_scale=task.reward_scale,
            )
        env.reset(seed=seed)

        # make_agent receives only dimensions + configs — no env_id
        agent = make_agent(task, env)

        t0 = time.perf_counter()
        trainer = Trainer(env, agent)
        train_result = trainer.train(
            n_episodes=task.n_episodes,
            max_steps=task.max_steps,
        )
        elapsed = time.perf_counter() - t0

        env.close()

        # Score is computed the same way for every task
        final_score = train_result.mean_reward(last_n=task.eval_window)
        solved = final_score >= task.solved_threshold

        # First episode that crossed the threshold (rolling check, 1-indexed display)
        first_solved_ep: int | None = None
        window = task.eval_window
        logs = train_result.episode_logs
        for i in range(window, len(logs) + 1):
            window_mean = float(np.mean([l.total_reward for l in logs[i - window:i]]))
            if window_mean >= task.solved_threshold:
                first_solved_ep = i  # episode index where the window first crossed
                break

        if verbose:
            ep_str = str(first_solved_ep) if first_solved_ep is not None else f">{task.n_episodes}"
            status = "✓" if solved else "✗"
            print(
                f"  {status} Seed {seed:3d} | "
                f"score={final_score:8.1f} | "
                f"first solved ep={ep_str:<8} | "
                f"{elapsed:.1f}s"
            )

        return SeedResult(
            seed=seed,
            final_score=final_score,
            solved=solved,
            first_solved_ep=first_solved_ep,
            train_result=train_result,
            elapsed_s=elapsed,
        )

    @staticmethod
    def print_summary(result: BenchmarkResult) -> None:
        """Print a formatted summary table for a completed BenchmarkResult."""
        task = result.task
        med = result.median_first_solved
        med_str = f"{med:.0f}" if med is not None else "never"

        print(
            f"\n{'─'*62}\n"
            f"  SUMMARY  {task.env_id}\n"
            f"  Mean ± Std : {result.mean_score:.1f} ± {result.std_score:.1f}\n"
            f"  Solved     : {result.n_solved}/{result.n_seeds} seeds "
            f"({result.solve_rate:.0%})\n"
            f"  Median first solved ep : {med_str}\n"
            f"  Config     : {task.description}\n"
            f"{'─'*62}\n"
        )

    @classmethod
    def compare(
        cls,
        env_ids: list[str],
        seeds: Sequence[int] | None = None,
    ) -> dict[str, BenchmarkResult]:
        """
        Run benchmarks for multiple tasks and return all results.

        Convenient for a multi-task evaluation in a single script.
        """
        return {eid: cls.run(eid, seeds=seeds) for eid in env_ids}