#!/usr/bin/env python3
"""
Aggregate full online-matrix runs and generate paper-ready outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", Path(tempfile.gettempdir()) / "xdg-cache"))
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

FULL_STEPS = 2_000_000
EVAL_INTERVAL = 50_000
EVAL_EPISODES = 20
EXPECTED_ENVS = ("ALE/Pong-ram-v5", "ALE/Breakout-ram-v5", "ALE/Boxing-ram-v5")
EXPECTED_VARIANTS = ("DQN", "DDQN+Dueling+PER")


@dataclass
class RunRecord:
    run_dir: Path
    env_id: str
    variant: str
    seed: int
    step_to_mean: dict[int, float]
    final_mean: float


def infer_variant(config: dict) -> str | None:
    alg = config.get("algorithm") or {}
    model = config.get("model") or {}
    replay = config.get("replay") or {}
    is_dqn = (
        not bool(alg.get("double_dqn", False))
        and not bool(model.get("dueling", False))
        and not bool(replay.get("prioritized", False))
    )
    is_main = (
        bool(alg.get("double_dqn", False))
        and bool(model.get("dueling", False))
        and bool(replay.get("prioritized", False))
    )
    if is_dqn:
        return "DQN"
    if is_main:
        return "DDQN+Dueling+PER"
    return None


def _load_eval_points(eval_dir: Path) -> dict[int, float]:
    step_to_mean: dict[int, float] = {}
    for path in eval_dir.glob("eval_step_*.json"):
        suffix = path.stem.replace("eval_step_", "")
        if not suffix.isdigit():
            continue
        step = int(suffix)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        mean_return = payload.get("mean_return")
        if isinstance(mean_return, (int, float)):
            step_to_mean[step] = float(mean_return)
    return step_to_mean


def collect_headline_runs(runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        config_path = run_dir / "config.yaml"
        eval_dir = run_dir / "eval"
        if not config_path.exists() or not eval_dir.exists():
            continue
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        env_id = ((config.get("env") or {}).get("id")) or ""
        variant = infer_variant(config)
        if env_id not in EXPECTED_ENVS or variant is None:
            continue

        eval_cfg = config.get("eval") or {}
        train_cfg = config.get("train") or {}
        if (
            int(train_cfg.get("total_steps", -1)) != FULL_STEPS
            or int(eval_cfg.get("eval_interval", -1)) != EVAL_INTERVAL
            or int(eval_cfg.get("n_episodes", -1)) != EVAL_EPISODES
        ):
            continue

        step_to_mean = _load_eval_points(eval_dir)
        if FULL_STEPS not in step_to_mean:
            continue
        seed = int((config.get("experiment") or {}).get("seed", -1))
        records.append(
            RunRecord(
                run_dir=run_dir,
                env_id=env_id,
                variant=variant,
                seed=seed,
                step_to_mean=step_to_mean,
                final_mean=float(step_to_mean[FULL_STEPS]),
            )
        )
    return records


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), (float(np.std(arr)) if len(arr) > 1 else 0.0)


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def _headline_status(records: list[RunRecord]) -> str:
    if not records:
        return "no_completed_runs"

    grouped: dict[tuple[str, str], set[int]] = {}
    for rec in records:
        grouped.setdefault((rec.env_id, rec.variant), set()).add(rec.seed)

    expected_groups = {(env_id, variant) for env_id in EXPECTED_ENVS for variant in EXPECTED_VARIANTS}
    if not expected_groups.issubset(set(grouped.keys())):
        return "partial"
    if all(len(grouped[(env_id, variant)]) >= 3 for env_id, variant in expected_groups):
        return "complete"
    return "partial"


def write_final_tables(records: list[RunRecord], output_dir: Path) -> tuple[Path, Path]:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.env_id, rec.variant), []).append(rec)

    csv_path = output_dir / "online_final_eval_table.csv"
    md_path = output_dir / "online_final_eval_table.md"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env_id", "variant", "n_seeds", "mean_final_eval", "std_final_eval", "mean_pm_std"],
        )
        writer.writeheader()
        for env_id in EXPECTED_ENVS:
            for variant in EXPECTED_VARIANTS:
                runs = grouped.get((env_id, variant), [])
                finals = [r.final_mean for r in runs]
                if finals:
                    mean, std = _mean_std(finals)
                    mean_pm_std = _fmt_mean_std(mean, std)
                else:
                    mean, std, mean_pm_std = "", "", "NA"
                writer.writerow(
                    {
                        "env_id": env_id,
                        "variant": variant,
                        "n_seeds": len(finals),
                        "mean_final_eval": mean,
                        "std_final_eval": std,
                        "mean_pm_std": mean_pm_std,
                    }
                )

    lines: list[str] = []
    if not records:
        lines.append("No completed full-protocol headline runs were found.")
        lines.append("")
    lines.extend(
        [
            "| env_id | variant | n_seeds | final_eval_mean +/- std |",
            "|---|---:|---:|---:|",
        ]
    )
    for env_id in EXPECTED_ENVS:
        for variant in EXPECTED_VARIANTS:
            runs = grouped.get((env_id, variant), [])
            finals = [r.final_mean for r in runs]
            if finals:
                mean, std = _mean_std(finals)
                text = _fmt_mean_std(mean, std)
            else:
                text = "NA"
            lines.append(f"| {env_id} | {variant} | {len(finals)} | {text} |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def _env_short(env_id: str) -> str:
    return env_id.split("/")[-1].replace("-ram-v5", "").lower()


def write_learning_curves(records: list[RunRecord], output_dir: Path) -> list[Path]:
    if not records:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "plot_results.py needs matplotlib to render figures when completed runs are present. "
            "Install project requirements first."
        ) from exc

    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.env_id, rec.variant), []).append(rec)

    figure_paths: list[Path] = []
    for env_id in EXPECTED_ENVS:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for variant in EXPECTED_VARIANTS:
            runs = grouped.get((env_id, variant), [])
            if not runs:
                continue
            common_steps = sorted(set.intersection(*(set(r.step_to_mean.keys()) for r in runs)))
            if not common_steps:
                continue
            means = []
            stds = []
            for step in common_steps:
                vals = [r.step_to_mean[step] for r in runs]
                m, s = _mean_std(vals)
                means.append(m)
                stds.append(s)
            x = np.array(common_steps, dtype=np.int64)
            y = np.array(means, dtype=np.float64)
            s = np.array(stds, dtype=np.float64)
            ax.plot(x, y, label=variant)
            if len(runs) >= 2:
                ax.fill_between(x, y - s, y + s, alpha=0.2)
        ax.set_title(f"{env_id} online learning curves")
        ax.set_xlabel("global_step")
        ax.set_ylabel("eval_mean_return")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        out = output_dir / f"learning_curve_{_env_short(env_id)}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        figure_paths.append(out)
    return figure_paths


def write_summary(records: list[RunRecord], figure_paths: list[Path], table_csv: Path, table_md: Path, output_dir: Path) -> Path:
    grouped: dict[str, dict[str, list[RunRecord]]] = {}
    for rec in records:
        grouped.setdefault(rec.env_id, {}).setdefault(rec.variant, []).append(rec)
    payload = {
        "status": _headline_status(records),
        "protocol_filter": {
            "total_steps": FULL_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "n_episodes": EVAL_EPISODES,
            "required_final_eval_step": FULL_STEPS,
        },
        "runs_included": [
            {
                "run_dir": str(r.run_dir),
                "env_id": r.env_id,
                "variant": r.variant,
                "seed": r.seed,
                "final_mean_return": r.final_mean,
            }
            for r in records
        ],
        "group_counts": {
            env_id: {variant: len(variant_runs) for variant, variant_runs in by_variant.items()}
            for env_id, by_variant in grouped.items()
        },
        "outputs": {
            "figures": [str(p) for p in figure_paths],
            "final_table_csv": str(table_csv),
            "final_table_md": str(table_md),
        },
    }
    summary_path = output_dir / "online_matrix_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate online matrix runs and produce figures/tables.")
    parser.add_argument("--runs-root", type=str, default="artifacts/runs", help="Root with run directories.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/figures/online",
        help="Directory for figure/table outputs.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_headline_runs(runs_root)
    figure_paths = write_learning_curves(records, output_dir)
    table_csv, table_md = write_final_tables(records, output_dir)
    summary_path = write_summary(records, figure_paths, table_csv, table_md, output_dir)
    status = _headline_status(records)
    if status == "no_completed_runs":
        print(
            "No completed full-protocol headline runs were found. "
            f"Wrote status/table outputs to {output_dir}. Summary: {summary_path}"
        )
        return
    print(
        f"Aggregated {len(records)} runs. Wrote figures/tables to {output_dir}. Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
