#!/usr/bin/env python3
"""
Aggregate offline experiment runs and generate paper-ready outputs.

Mirrors the structure of scripts/plot_results.py (online) but for
offline runs stored under artifacts/offline_runs/.

Produces per-environment learning curves (eval mean return vs gradient
steps) and a final summary table (mean +/- std across seeds).

Usage:
    # All envs, all methods, all splits:
    python scripts/plot_offline_results.py

    # Single env:
    python scripts/plot_offline_results.py --env ALE/Pong-ram-v5

    # Single split:
    python scripts/plot_offline_results.py --split random

    # Single method:
    python scripts/plot_offline_results.py --method cql
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Matplotlib backend setup (mirrors plot_results.py)
_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR",
                     Path(tempfile.gettempdir()) / "mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_ENVS     = ("ALE/Pong-ram-v5", "ALE/Breakout-ram-v5", "ALE/Boxing-ram-v5")
EXPECTED_METHODS  = ("bc", "dqn_offline", "dqn_bc", "cql")
EXPECTED_SPLITS   = ("random", "mixed", "expert_ish")

METHOD_LABELS = {
    "bc":          "BC",
    "dqn_offline": "Naive DQN",
    "dqn_bc":      "DQN+BC",
    "cql":         "CQL",
}
SPLIT_LABELS = {
    "random":     "D_random",
    "mixed":      "D_mixed",
    "expert_ish": "D_expert-ish",
}
METHOD_COLORS = {
    "bc":          "#1f77b4",
    "dqn_offline": "#ff7f0e",
    "dqn_bc":      "#2ca02c",
    "cql":         "#d62728",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class OfflineRunRecord:
    run_dir:      Path
    env_id:       str
    method:       str
    split:        str
    seed:         int
    lam:          float | None      # dqn_bc only
    cql_alpha:    float | None      # cql only
    step_to_mean: dict[int, float]
    final_mean:   float


# ---------------------------------------------------------------------------
# Run collection
# ---------------------------------------------------------------------------
def _load_eval_points(eval_dir: Path) -> dict[int, float]:
    step_to_mean: dict[int, float] = {}
    for path in eval_dir.glob("eval_step_*.json"):
        suffix = path.stem.replace("eval_step_", "")
        if not suffix.isdigit():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        mean_return = payload.get("mean_return")
        if isinstance(mean_return, (int, float)):
            step_to_mean[int(suffix)] = float(mean_return)
    return step_to_mean


def _split_from_dataset_path(dataset_path: str) -> str | None:
    """Extract split name from dataset path string."""
    p = Path(dataset_path).stem   # e.g. "random", "mixed", "expert_ish"
    return p if p in EXPECTED_SPLITS else None


def collect_offline_runs(
    runs_root: Path,
    filter_env: str | None = None,
    filter_method: str | None = None,
    filter_split: str | None = None,
) -> list[OfflineRunRecord]:
    records: list[OfflineRunRecord] = []

    for run_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        config_path = run_dir / "config.yaml"
        eval_dir    = run_dir / "eval"
        if not config_path.exists() or not eval_dir.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        env_id  = (config.get("env") or {}).get("id", "")
        offline = config.get("offline") or {}
        method  = offline.get("method", "")
        split   = _split_from_dataset_path(offline.get("dataset_path", ""))
        seed    = int((config.get("experiment") or {}).get("seed", -1))

        if not env_id or not method or not split:
            continue
        if filter_env    and env_id != filter_env:
            continue
        if filter_method and method != filter_method:
            continue
        if filter_split  and split  != filter_split:
            continue

        step_to_mean = _load_eval_points(eval_dir)
        if not step_to_mean:
            continue

        final_step = max(step_to_mean.keys())
        lam        = float(offline["lam"])       if "lam"       in offline else None
        cql_alpha  = float(offline["cql_alpha"]) if "cql_alpha" in offline else None

        records.append(OfflineRunRecord(
            run_dir=run_dir,
            env_id=env_id,
            method=method,
            split=split,
            seed=seed,
            lam=lam,
            cql_alpha=cql_alpha,
            step_to_mean=step_to_mean,
            final_mean=float(step_to_mean[final_step]),
        ))

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env_short(env_id: str) -> str:
    return env_id.split("/")[-1].replace("-ram-v5", "").lower()


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def _group_key(r: OfflineRunRecord) -> tuple:
    """Group runs that share env / method / split / hyperparams."""
    return (r.env_id, r.method, r.split, r.lam, r.cql_alpha)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
def write_learning_curves(
    records: list[OfflineRunRecord],
    output_dir: Path,
    split: str | None = None,
) -> list[Path]:
    if not records:
        return []

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # One figure per (env, split) combination
    from collections import defaultdict
    env_split_groups: dict[tuple[str, str], list[OfflineRunRecord]] = defaultdict(list)
    for r in records:
        env_split_groups[(r.env_id, r.split)].append(r)

    figure_paths: list[Path] = []

    for (env_id, spl), group in sorted(env_split_groups.items()):
        fig, ax = plt.subplots(figsize=(8, 4.8))

        # Sub-group by method (+ hyperparams)
        method_groups: dict[tuple, list[OfflineRunRecord]] = defaultdict(list)
        for r in group:
            method_groups[_group_key(r)].append(r)

        for key, runs in sorted(method_groups.items()):
            method = runs[0].method
            lam    = runs[0].lam
            alpha  = runs[0].cql_alpha

            # Label
            label = METHOD_LABELS.get(method, method)
            if lam is not None:
                label += f" λ={lam}"
            if alpha is not None:
                label += f" α={alpha}"

            # Curve
            common_steps = sorted(
                set.intersection(*(set(r.step_to_mean.keys()) for r in runs))
            )
            if not common_steps:
                continue
            means, stds = [], []
            for step in common_steps:
                vals = [r.step_to_mean[step] for r in runs]
                m, s = _mean_std(vals)
                means.append(m)
                stds.append(s)

            x = np.array(common_steps, dtype=np.int64)
            y = np.array(means, dtype=np.float64)
            s = np.array(stds,  dtype=np.float64)

            color = METHOD_COLORS.get(method, None)
            ax.plot(x, y, label=label, color=color)
            if len(runs) >= 2:
                ax.fill_between(x, y - s, y + s, alpha=0.2, color=color)

        ax.set_title(f"{env_id}  |  {SPLIT_LABELS.get(spl, spl)}")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("eval mean return")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=8)

        fname = f"offline_curve_{_env_short(env_id)}_{spl}.png"
        out   = output_dir / fname
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        figure_paths.append(out)
        print(f"  Saved curve: {out}")

    return figure_paths


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def write_summary_table(
    records: list[OfflineRunRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    from collections import defaultdict
    grouped: dict[tuple, list[OfflineRunRecord]] = defaultdict(list)
    for r in records:
        grouped[_group_key(r)].append(r)

    csv_path = output_dir / "offline_final_eval_table.csv"
    md_path  = output_dir / "offline_final_eval_table.md"

    fieldnames = ["env_id", "split", "method", "lam", "cql_alpha",
                  "n_seeds", "mean_final_eval", "std_final_eval", "mean_pm_std"]

    rows = []
    for env_id in EXPECTED_ENVS:
        for split in EXPECTED_SPLITS:
            for method in EXPECTED_METHODS:
                # Collect all hyperparam variants for this combo
                matching = {k: v for k, v in grouped.items()
                            if k[0] == env_id and k[1] == method and k[2] == split}
                if not matching:
                    rows.append({
                        "env_id": env_id, "split": split, "method": method,
                        "lam": "", "cql_alpha": "",
                        "n_seeds": 0, "mean_final_eval": "",
                        "std_final_eval": "", "mean_pm_std": "NA",
                    })
                    continue
                for key, runs in sorted(matching.items()):
                    finals = [r.final_mean for r in runs]
                    mean, std = _mean_std(finals)
                    rows.append({
                        "env_id":         env_id,
                        "split":          split,
                        "method":         method,
                        "lam":            key[3] if key[3] is not None else "",
                        "cql_alpha":      key[4] if key[4] is not None else "",
                        "n_seeds":        len(finals),
                        "mean_final_eval": round(mean, 4),
                        "std_final_eval":  round(std,  4),
                        "mean_pm_std":     _fmt(mean, std),
                    })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown table
    lines = [
        "| env | split | method | lam | cql_alpha | n_seeds | mean +/- std |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['env_id']} | {row['split']} | {row['method']} "
            f"| {row['lam']} | {row['cql_alpha']} "
            f"| {row['n_seeds']} | {row['mean_pm_std']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"  Saved table (csv): {csv_path}")
    print(f"  Saved table (md):  {md_path}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
def write_summary_json(
    records: list[OfflineRunRecord],
    figure_paths: list[Path],
    csv_path: Path,
    md_path: Path,
    output_dir: Path,
) -> Path:
    payload = {
        "n_runs": len(records),
        "runs_included": [
            {
                "run_dir": str(r.run_dir),
                "env_id":  r.env_id,
                "method":  r.method,
                "split":   r.split,
                "seed":    r.seed,
                "lam":     r.lam,
                "cql_alpha": r.cql_alpha,
                "final_mean_return": r.final_mean,
            }
            for r in records
        ],
        "outputs": {
            "figures":        [str(p) for p in figure_paths],
            "table_csv":      str(csv_path),
            "table_md":       str(md_path),
        },
    }
    summary_path = output_dir / "offline_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved summary:     {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate offline runs and produce figures/tables."
    )
    parser.add_argument("--runs-root",  type=str,
                        default="artifacts/offline_runs")
    parser.add_argument("--output-dir", type=str,
                        default="artifacts/figures/offline")
    parser.add_argument("--env",    type=str, default=None,
                        help="Filter to one env id.")
    parser.add_argument("--method", type=str, default=None,
                        choices=list(EXPECTED_METHODS),
                        help="Filter to one method.")
    parser.add_argument("--split",  type=str, default=None,
                        choices=list(EXPECTED_SPLITS),
                        help="Filter to one dataset split.")
    args = parser.parse_args()

    runs_root  = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning runs in: {runs_root}")
    records = collect_offline_runs(
        runs_root,
        filter_env=args.env,
        filter_method=args.method,
        filter_split=args.split,
    )

    if not records:
        print("No completed offline runs found.")
        return

    print(f"Found {len(records)} run(s). Generating outputs...\n")
    figure_paths      = write_learning_curves(records, output_dir, args.split)
    csv_path, md_path = write_summary_table(records, output_dir)
    write_summary_json(records, figure_paths, csv_path, md_path, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()