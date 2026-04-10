#!/usr/bin/env python3
"""
Combined online vs offline plotting for the paper.

Reads:
  - artifacts/figures/online/online_final_eval_table.csv
  - artifacts/figures/offline/offline_final_eval_table.csv

Produces (under artifacts/figures/combined/):
  1. bar_chart_<env>.png
     Per-env grouped bar chart: online baselines + offline methods
     across dataset qualities. The key "online vs offline gap" figure.

  2. bar_chart_all_envs.png
     Single figure with all three envs as subplots, expert_ish split only.
     Good for the paper's main results section.

  3. dataset_quality_effect.png
     Line plot showing how each offline method improves as dataset
     quality increases (random -> mixed -> expert_ish).
     One subplot per env.

Usage:
    python scripts/plot_combined_results.py
    python scripts/plot_combined_results.py \
        --online-table artifacts/figures/online/online_final_eval_table.csv \
        --offline-table artifacts/figures/offline/offline_final_eval_table.csv
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR",
                     Path(tempfile.gettempdir()) / "mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENVS = [
    "ALE/Pong-ram-v5",
    "ALE/Breakout-ram-v5",
    "ALE/Boxing-ram-v5",
]
ENV_SHORT = {
    "ALE/Pong-ram-v5":    "Pong",
    "ALE/Breakout-ram-v5": "Breakout",
    "ALE/Boxing-ram-v5":   "Boxing",
}
SPLITS = ["random", "mixed", "expert_ish"]
SPLIT_LABELS = {
    "random":     "D_random",
    "mixed":      "D_mixed",
    "expert_ish": "D_expert-ish",
}

# Colors per method — consistent across all figures
COLORS = {
    "DQN":             "#1f77b4",
    "DDQN+Dueling+PER": "#ff7f0e",
    "BC":              "#2ca02c",
    "Naive DQN":       "#d62728",
    "DQN+BC":          "#9467bd",
    "CQL":             "#8c564b",
}

OFFLINE_METHOD_LABELS = {
    "bc":          "BC",
    "dqn_offline": "Naive DQN",
    "dqn_bc":      "DQN+BC",
    "cql":         "CQL",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_online(path: Path) -> dict:
    """
    Returns dict keyed by (env_id, variant) -> (mean, std).
    """
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            env  = row["env_id"]
            var  = row["variant"]
            mean = float(row["mean_final_eval"]) if row["mean_final_eval"] else None
            std  = float(row["std_final_eval"])  if row["std_final_eval"]  else 0.0
            if mean is not None:
                data[(env, var)] = (mean, std)
    return data


def load_offline(path: Path) -> dict:
    """
    Returns dict keyed by (env_id, split, method_label) -> (mean, std).
    Only loads best-hyperparam rows (lam=1.0 for dqn_bc, alpha=1.0 for cql).
    """
    import csv
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            env    = row["env_id"]
            split  = row["split"]
            method = row["method"]
            lam    = row["lam"]
            alpha  = row["cql_alpha"]
            mean   = row["mean_final_eval"]
            std    = row["std_final_eval"]

            # Skip sweep variants — only keep best hyperparams
            if method == "dqn_bc"  and lam   not in ("", "1.0"):
                continue
            if method == "cql"     and alpha not in ("", "1.0"):
                continue
            if not mean:
                continue

            label = OFFLINE_METHOD_LABELS.get(method, method)
            data[(env, split, label)] = (float(mean), float(std))
    return data


# ---------------------------------------------------------------------------
# Figure 1: Per-env grouped bar chart (all splits + online)
# ---------------------------------------------------------------------------
def plot_per_env_bar(online: dict, offline: dict, output_dir: Path) -> list[Path]:
    """
    For each env: grouped bar chart with online baselines and offline
    methods per dataset split.
    """
    paths = []
    offline_methods = ["BC", "Naive DQN", "DQN+BC", "CQL"]

    for env in ENVS:
        fig, ax = plt.subplots(figsize=(12, 5))
        env_s = ENV_SHORT[env]

        # Build groups: one group per split + one group for online
        group_labels = [SPLIT_LABELS[s] for s in SPLITS] + ["Online"]
        n_groups     = len(group_labels)

        # All methods in order
        online_methods = ["DQN", "DDQN+Dueling+PER"]
        all_methods    = offline_methods + online_methods
        n_methods      = len(all_methods)

        bar_w   = 0.12
        offsets = np.linspace(
            -(n_methods - 1) * bar_w / 2,
             (n_methods - 1) * bar_w / 2,
            n_methods
        )
        x = np.arange(n_groups)

        for m_idx, method in enumerate(all_methods):
            means, errs = [], []
            for g_idx, group in enumerate(group_labels):
                if group == "Online":
                    # Online methods only shown in the "Online" group
                    if method in online_methods:
                        key = (env, method)
                        if key in online:
                            means.append(online[key][0])
                            errs.append(online[key][1])
                        else:
                            means.append(0.0)
                            errs.append(0.0)
                    else:
                        means.append(np.nan)
                        errs.append(0.0)
                else:
                    # Offline methods only shown in split groups
                    split = [s for s in SPLITS if SPLIT_LABELS[s] == group][0]
                    if method in offline_methods:
                        key = (env, split, method)
                        if key in offline:
                            means.append(offline[key][0])
                            errs.append(offline[key][1])
                        else:
                            means.append(np.nan)
                            errs.append(0.0)
                    else:
                        means.append(np.nan)
                        errs.append(0.0)

            means = np.array(means, dtype=float)
            errs  = np.array(errs,  dtype=float)
            color = COLORS.get(method, "#888888")
            mask  = ~np.isnan(means)

            ax.bar(
                x[mask] + offsets[m_idx],
                means[mask],
                bar_w,
                yerr=errs[mask],
                label=method,
                color=color,
                capsize=3,
                alpha=0.85,
            )

        # Vertical separator before "Online" group
        ax.axvline(x=n_groups - 1.5, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.6)

        ax.set_title(f"{env_s} — offline vs online final eval return")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_ylabel("mean eval return (± std)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=0.3)

        out = output_dir / f"bar_chart_{env_s.lower()}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")
        paths.append(out)

    return paths


# ---------------------------------------------------------------------------
# Figure 2: All envs — expert_ish only, side by side
# ---------------------------------------------------------------------------
def plot_all_envs_expert(online: dict, offline: dict, output_dir: Path) -> Path:
    """
    Three-panel figure showing expert_ish offline vs online for all envs.
    Good for the paper's main results section.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    offline_methods = ["BC", "Naive DQN", "DQN+BC", "CQL"]
    online_methods  = ["DQN", "DDQN+Dueling+PER"]
    all_methods     = offline_methods + online_methods

    for ax, env in zip(axes, ENVS):
        env_s = ENV_SHORT[env]
        means, errs, labels, colors = [], [], [], []

        for method in offline_methods:
            key = (env, "expert_ish", method)
            if key in offline:
                means.append(offline[key][0])
                errs.append(offline[key][1])
                labels.append(method)
                colors.append(COLORS.get(method, "#888"))

        # Separator
        means.append(np.nan)
        errs.append(0.0)
        labels.append("")
        colors.append("white")

        for method in online_methods:
            key = (env, method)
            if key in online:
                means.append(online[key][0])
                errs.append(online[key][1])
                labels.append(method)
                colors.append(COLORS.get(method, "#888"))

        x    = np.arange(len(means))
        mask = ~np.isnan(np.array(means, dtype=float))

        bars = ax.bar(
            x[mask],
            np.array(means, dtype=float)[mask],
            yerr=np.array(errs, dtype=float)[mask],
            color=[colors[i] for i in range(len(means)) if mask[i]],
            capsize=4,
            alpha=0.85,
        )

        ax.set_title(f"{env_s}")
        ax.set_xticks(x[mask])
        ax.set_xticklabels(
            [labels[i] for i in range(len(labels)) if mask[i]],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylabel("mean eval return (± std)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Dashed line at best online performance
        best_online = max(
            (online.get((env, m), (float("-inf"), 0))[0] for m in online_methods),
            default=None,
        )
        if best_online is not None:
            ax.axhline(best_online, color="black", linestyle="--",
                       linewidth=1.0, alpha=0.5, label=f"best online ({best_online:.1f})")
            ax.legend(fontsize=7)

    fig.suptitle("Expert-ish offline vs online — final eval return", fontsize=12)
    fig.tight_layout()
    out = output_dir / "bar_chart_all_envs_expert_ish.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: Dataset quality effect per method
# ---------------------------------------------------------------------------
def plot_dataset_quality_effect(offline: dict, output_dir: Path) -> Path:
    """
    Line plot: x=dataset quality, y=mean return, one line per method.
    One subplot per env. Shows how each method benefits from better data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    offline_methods = ["BC", "Naive DQN", "DQN+BC", "CQL"]

    for ax, env in zip(axes, ENVS):
        env_s = ENV_SHORT[env]
        x = np.arange(len(SPLITS))

        for method in offline_methods:
            means, errs = [], []
            for split in SPLITS:
                key = (env, split, method)
                if key in offline:
                    means.append(offline[key][0])
                    errs.append(offline[key][1])
                else:
                    means.append(np.nan)
                    errs.append(0.0)

            means = np.array(means, dtype=float)
            errs  = np.array(errs,  dtype=float)
            color = COLORS.get(method, "#888")
            ax.plot(x, means, marker="o", label=method, color=color)
            ax.fill_between(x, means - errs, means + errs,
                            alpha=0.15, color=color)

        ax.set_title(f"{env_s}")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=9)
        ax.set_ylabel("mean eval return (± std)" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Effect of dataset quality on offline method performance", fontsize=12)
    fig.tight_layout()
    out = output_dir / "dataset_quality_effect.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Combined table (online + offline side by side)
# ---------------------------------------------------------------------------
def write_combined_table(online: dict, offline: dict, output_dir: Path) -> tuple[Path, Path]:
    import csv

    offline_methods = ["BC", "Naive DQN", "DQN+BC", "CQL"]
    online_methods  = ["DQN", "DDQN+Dueling+PER"]

    rows = []
    for env in ENVS:
        # Offline rows — one per method per split
        for split in SPLITS:
            for method in offline_methods:
                key = (env, split, method)
                mean, std = offline.get(key, (None, None))
                rows.append({
                    "env_id":    env,
                    "setting":   "offline",
                    "method":    method,
                    "split":     split,
                    "mean_final_eval": round(mean, 4) if mean is not None else "",
                    "std_final_eval":  round(std,  4) if std  is not None else "",
                    "mean_pm_std": _fmt(mean, std) if mean is not None else "NA",
                })
        # Online rows — no split
        for method in online_methods:
            key = (env, method)
            mean, std = online.get(key, (None, None))
            rows.append({
                "env_id":    env,
                "setting":   "online",
                "method":    method,
                "split":     "—",
                "mean_final_eval": round(mean, 4) if mean is not None else "",
                "std_final_eval":  round(std,  4) if std  is not None else "",
                "mean_pm_std": _fmt(mean, std) if mean is not None else "NA",
            })

    # CSV
    csv_path = output_dir / "combined_final_eval_table.csv"
    fieldnames = ["env_id", "setting", "split", "method",
                  "mean_final_eval", "std_final_eval", "mean_pm_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_path = output_dir / "combined_final_eval_table.md"
    lines = [
        "| env | setting | split | method | mean +/- std |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['env_id']} | {row['setting']} | {row['split']} "
            f"| {row['method']} | {row['mean_pm_std']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"  Saved table (csv): {csv_path}")
    print(f"  Saved table (md):  {md_path}")
    return csv_path, md_path


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined online vs offline result plots."
    )
    parser.add_argument("--online-table",  type=str,
                        default="artifacts/figures/online/online_final_eval_table.csv")
    parser.add_argument("--offline-table", type=str,
                        default="artifacts/figures/offline/offline_final_eval_table.csv")
    parser.add_argument("--output-dir",    type=str,
                        default="artifacts/figures/combined")
    args = parser.parse_args()

    online_path  = Path(args.online_table)
    offline_path = Path(args.offline_table)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not online_path.exists():
        raise FileNotFoundError(f"Online table not found: {online_path}")
    if not offline_path.exists():
        raise FileNotFoundError(f"Offline table not found: {offline_path}")

    print("Loading tables...")
    online  = load_online(online_path)
    offline = load_offline(offline_path)
    print(f"  Online entries:  {len(online)}")
    print(f"  Offline entries: {len(offline)}")

    print("\nGenerating figures...")
    plot_per_env_bar(online, offline, output_dir)
    plot_all_envs_expert(online, offline, output_dir)
    plot_dataset_quality_effect(offline, output_dir)

    print("\nGenerating combined table...")
    write_combined_table(online, offline, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()