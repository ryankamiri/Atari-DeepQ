#!/usr/bin/env python3
"""
Batch runner for all offline RL experiments.

Launches multiple offline training runs in parallel using
multiprocessing. Each run is an independent process running
scripts/offline_train.py with a specific config and seed.

Usage:
    # Dry run - see what would be launched without running:
    python scripts/run_offline_experiments.py --dry-run

    # Run everything (default: 3 parallel workers):
    python scripts/run_offline_experiments.py

    # Control parallelism (recommended: number of CPU cores - 1):
    python scripts/run_offline_experiments.py --workers 4

    # Run only specific env/method/split:
    python scripts/run_offline_experiments.py --env pong --split random
    python scripts/run_offline_experiments.py --method cql --split expert_ish

    # Run only best hyperparams (skip sweep variants):
    python scripts/run_offline_experiments.py --best-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Experiment dimensions
# ---------------------------------------------------------------------------
ENVS   = ["pong", "breakout", "boxing"]
SPLITS = ["random", "mixed", "expert_ish"]
SEEDS  = [0, 1, 2]

# Methods with their hyperparam variants
# Format: (method_name, config_suffix)
METHODS = [
    ("bc",          ""),
    ("dqn_offline", ""),
    ("dqn_bc",      "_lam0p1"),
    ("dqn_bc",      "_lam1p0"),
    ("dqn_bc",      "_lam3p0"),
    ("cql",         "_alpha0p5"),
    ("cql",         "_alpha1p0"),
    ("cql",         "_alpha3p0"),
]

# Best hyperparams only (for --best-only flag)
BEST_METHODS = [
    ("bc",          ""),
    ("dqn_offline", ""),
    ("dqn_bc",      "_lam1p0"),
    ("cql",         "_alpha1p0"),
]


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------
def build_jobs(
    envs: list[str],
    splits: list[str],
    seeds: list[int],
    methods: list[tuple[str, str]],
    config_dir: Path,
) -> list[dict]:
    jobs = []
    for env, split, seed, (method, suffix) in product(envs, splits, seeds, methods):
        config_name = f"{env}_{method}_{split}{suffix}.yaml"
        config_path = config_dir / config_name
        if not config_path.exists():
            continue
        jobs.append({
            "config": str(config_path),
            "seed":   seed,
            "label":  f"{env}/{split}/{method}{suffix}/seed{seed}",
        })
    return jobs


# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------
def run_job(job: dict) -> dict:
    config  = job["config"]
    seed    = job["seed"]
    label   = job["label"]
    script  = str(ROOT / "scripts" / "offline_train.py")

    cmd = [sys.executable, script, "--config", config, "--seed", str(seed)]

    start = time.time()
    print(f"[START] {label}")

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    success = result.returncode == 0

    status = "OK" if success else "FAIL"
    print(f"[{status}] {label}  ({elapsed:.0f}s)")

    if not success:
        # Print last 10 lines of stderr to help debug
        lines = result.stderr.strip().splitlines()
        for line in lines[-10:]:
            print(f"  ERR | {line}")

    return {
        "label":   label,
        "config":  config,
        "seed":    seed,
        "success": success,
        "elapsed": elapsed,
        "stdout":  result.stdout[-2000:] if result.stdout else "",
        "stderr":  result.stderr[-2000:] if result.stderr else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch offline RL runner.")
    parser.add_argument("--workers",   type=int, default=3,
                        help="Number of parallel processes (default: 3). "
                             "Recommended: number of CPU cores - 1.")
    parser.add_argument("--config-dir", type=str,
                        default=str(ROOT / "configs" / "offline"))
    parser.add_argument("--env",    type=str, nargs="+",
                        choices=ENVS, default=ENVS)
    parser.add_argument("--split",  type=str, nargs="+",
                        choices=SPLITS, default=SPLITS)
    parser.add_argument("--method", type=str, nargs="+", default=None,
                        help="Filter by method name (bc, dqn_offline, dqn_bc, cql).")
    parser.add_argument("--seeds",  type=int, nargs="+", default=SEEDS)
    parser.add_argument("--best-only", action="store_true",
                        help="Only run best hyperparam variant per method.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print jobs without running them.")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    methods    = BEST_METHODS if args.best_only else METHODS

    # Filter by method name if specified
    if args.method:
        methods = [(m, s) for m, s in methods if m in args.method]

    jobs = build_jobs(
        envs=args.env,
        splits=args.split,
        seeds=args.seeds,
        methods=methods,
        config_dir=config_dir,
    )

    if not jobs:
        print("No matching configs found. Check --env / --split / --method filters.")
        return

    print(f"\nOffline batch runner")
    print(f"  Jobs:    {len(jobs)}")
    print(f"  Workers: {args.workers}")
    print(f"  Envs:    {args.env}")
    print(f"  Splits:  {args.split}")
    print(f"  Seeds:   {args.seeds}")
    print()

    if args.dry_run:
        print("--- DRY RUN (not executing) ---")
        for j in jobs:
            print(f"  config={Path(j['config']).name}  seed={j['seed']}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    # Confirm before launching
    ans = input(f"Launch {len(jobs)} jobs with {args.workers} workers? [y/N] ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
        from tqdm import tqdm

    t0 = time.time()
    results = []
    elapsed_times = []

    with Pool(processes=args.workers) as pool:
        with tqdm(
            total=len(jobs),
            desc="Offline runs",
            unit="run",
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for result in pool.imap_unordered(run_job, jobs):
                results.append(result)
                elapsed_times.append(result["elapsed"])

                # Update progress bar postfix with live stats
                n_done = len(results)
                n_ok   = sum(1 for r in results if r["success"])
                n_fail = n_done - n_ok
                avg    = sum(elapsed_times) / len(elapsed_times)

                # Estimate remaining time accounting for parallelism
                remaining_jobs = len(jobs) - n_done
                eta_sec = (remaining_jobs / args.workers) * avg

                pbar.set_postfix({
                    "ok":    n_ok,
                    "fail":  n_fail,
                    "avg":   f"{avg:.0f}s",
                    "ETA":   f"{eta_sec/60:.1f}m",
                })
                pbar.update(1)

    total = time.time() - t0
    n_ok   = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok

    print(f"\n{'='*50}")
    print(f"Batch complete in {total/60:.1f} min")
    print(f"  Succeeded: {n_ok}/{len(results)}")
    print(f"  Failed:    {n_fail}/{len(results)}")
    print(f"  Avg time per run: {sum(elapsed_times)/len(elapsed_times):.0f}s")

    if n_fail > 0:
        print("\nFailed jobs:")
        for r in results:
            if not r["success"]:
                print(f"  {r['label']}")
                print(f"  config: {r['config']}")


if __name__ == "__main__":
    main()