"""
Generate all figures for prompt compression research.

Figures:
1. Main curve: Accuracy vs compression rate (hero figure)
2. Cliff detection: First derivative of accuracy
3. Strategy comparison: Bar chart at 50% compression
4. Section ablation: Bar chart
5. Variance plot: Std dev across seeds
"""

import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")
AGG_PATH = os.path.join(RESULTS_DIR, "aggregated_results.csv")
ABLATION_PATH = os.path.join(RESULTS_DIR, "ablation_results.csv")
STATS_PATH = os.path.join(RESULTS_DIR, "statistical_tests.json")

STRATEGY_LABELS = {
    "random_dropout": "Random Dropout",
    "stopword_removal": "Stopword Removal",
    "entity_preserving": "Entity-Preserving",
    "llm_guided": "LLM-Guided",
}

STRATEGY_COLORS = {
    "random_dropout": "#e74c3c",
    "stopword_removal": "#3498db",
    "entity_preserving": "#2ecc71",
    "llm_guided": "#9b59b6",
}

TASK_LABELS = {"alc": "Arbitrary Label Classification (ALC)", "slr": "Synthetic Logic Rules (SLR)"}


def load_agg():
    rows = []
    with open(AGG_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["compression_rate"] = float(row["compression_rate"])
            row["mean_accuracy"] = float(row["mean_accuracy"])
            row["std"] = float(row["std"])
            row["ci_lower"] = float(row["ci_lower"])
            row["ci_upper"] = float(row["ci_upper"])
            rows.append(row)
    return rows


def load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return []
    rows = []
    with open(ABLATION_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["compression_rate"] = float(row["compression_rate"])
            row["mean_accuracy"] = float(row["mean_accuracy"])
            row["std"] = float(row["std"])
            rows.append(row)
    return rows


def load_stats():
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Figure 1: Main Accuracy vs Compression Curve
# ──────────────────────────────────────────────

def plot_main_curve(data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for idx, task in enumerate(["alc", "slr"]):
        ax = axes[idx]
        task_data = [r for r in data if r["task"] == task]

        for strategy in STRATEGY_LABELS:
            s_data = [r for r in task_data if r["strategy"] == strategy]
            s_data.sort(key=lambda r: r["compression_rate"])

            if not s_data:
                continue

            rates = [r["compression_rate"] for r in s_data]
            accs = [r["mean_accuracy"] for r in s_data]
            ci_lo = [r["ci_lower"] for r in s_data]
            ci_hi = [r["ci_upper"] for r in s_data]

            color = STRATEGY_COLORS[strategy]
            label = STRATEGY_LABELS[strategy]

            ax.plot(rates, accs, "o-", color=color, label=label, linewidth=2.5, markersize=6)
            ax.fill_between(rates, ci_lo, ci_hi, color=color, alpha=0.12)

        # Random baseline
        baseline = 1/3 if task == "alc" else 0.25
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.6, label=f"Random ({baseline:.0%})")

        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_xlabel("Compression Rate", fontsize=12)
        ax.set_xlim(-0.02, 0.92)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    fig.suptitle("Prompt Compression: Accuracy vs. Compression Rate", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "main_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 2: Cliff Detection (derivative)
# ──────────────────────────────────────────────

def plot_cliff_detection(data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for idx, task in enumerate(["alc", "slr"]):
        ax = axes[idx]
        task_data = [r for r in data if r["task"] == task]

        for strategy in STRATEGY_LABELS:
            s_data = [r for r in task_data if r["strategy"] == strategy]
            s_data.sort(key=lambda r: r["compression_rate"])

            if len(s_data) < 2:
                continue

            rates = [r["compression_rate"] for r in s_data]
            accs = [r["mean_accuracy"] for r in s_data]

            # Compute first derivative (change in accuracy per 10% compression step)
            deriv_rates = [(rates[i] + rates[i+1]) / 2 for i in range(len(rates)-1)]
            deriv_vals = [(accs[i+1] - accs[i]) / (rates[i+1] - rates[i]) if rates[i+1] != rates[i] else 0
                          for i in range(len(rates)-1)]

            color = STRATEGY_COLORS[strategy]
            label = STRATEGY_LABELS[strategy]
            ax.plot(deriv_rates, deriv_vals, "o-", color=color, label=label, linewidth=2, markersize=5)

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3, label="Steep drop threshold")
        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_xlabel("Compression Rate", fontsize=12)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if idx == 0:
            ax.set_ylabel("Accuracy Change Rate (dAcc/dRate)", fontsize=12)
            ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle("Cliff Detection: Where Does Performance Drop Sharpest?", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cliff_detection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 3: Strategy Comparison at 50%
# ──────────────────────────────────────────────

def plot_strategy_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, task in enumerate(["alc", "slr"]):
        ax = axes[idx]
        task_data = [r for r in data if r["task"] == task and abs(r["compression_rate"] - 0.5) < 0.01]

        strategies = []
        accs = []
        stds = []
        colors = []

        for strategy in STRATEGY_LABELS:
            match = [r for r in task_data if r["strategy"] == strategy]
            if match:
                r = match[0]
                strategies.append(STRATEGY_LABELS[strategy])
                accs.append(r["mean_accuracy"])
                stds.append(r["std"])
                colors.append(STRATEGY_COLORS[strategy])

        bars = ax.bar(strategies, accs, yerr=stds, color=colors, capsize=5, edgecolor="white", linewidth=1.5)

        # Add value labels
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")

        # Random baseline
        baseline = 1/3 if task == "alc" else 0.25
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.6)
        ax.text(len(strategies)-0.5, baseline + 0.02, f"Random ({baseline:.0%})",
                fontsize=9, color="gray")

        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Accuracy" if idx == 0 else "")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    fig.suptitle("Strategy Comparison at 50% Compression", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "strategy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 4: Section Ablation
# ──────────────────────────────────────────────

def plot_section_ablation(abl_data, main_data):
    if not abl_data:
        print("No ablation data — skipping ablation plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    section_labels = {"instructions": "Instructions Only", "examples": "Examples Only", "both": "Both Sections"}
    section_colors = {"instructions": "#e67e22", "examples": "#1abc9c", "both": "#e74c3c"}

    for idx, task in enumerate(["alc", "slr"]):
        ax = axes[idx]
        task_abl = [r for r in abl_data if r["task"] == task]

        # Get baseline (0% compression from main experiment)
        baseline_data = [r for r in main_data
                         if r["task"] == task and r["strategy"] == "entity_preserving"
                         and abs(r["compression_rate"]) < 0.01]
        baseline_acc = baseline_data[0]["mean_accuracy"] if baseline_data else 1.0

        names = ["Full Prompt\n(control)"]
        accs_vals = [baseline_acc]
        stds_vals = [0.0]
        bar_colors = ["#95a5a6"]

        for section in ["instructions", "examples", "both"]:
            matches = [r for r in task_abl if r["section"] == section]
            if matches:
                names.append(section_labels[section])
                accs_vals.append(matches[0]["mean_accuracy"])
                stds_vals.append(matches[0]["std"])
                bar_colors.append(section_colors[section])

        bars = ax.bar(names, accs_vals, yerr=stds_vals, color=bar_colors, capsize=5,
                      edgecolor="white", linewidth=1.5)

        for bar, acc in zip(bars, accs_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")

        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Accuracy" if idx == 0 else "")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    fig.suptitle("Section Ablation: Which Part of the Prompt Matters Most? (50% Compression)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "section_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Figure 5: Variance Plot
# ──────────────────────────────────────────────

def plot_variance(data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for idx, task in enumerate(["alc", "slr"]):
        ax = axes[idx]
        task_data = [r for r in data if r["task"] == task]

        for strategy in STRATEGY_LABELS:
            s_data = [r for r in task_data if r["strategy"] == strategy]
            s_data.sort(key=lambda r: r["compression_rate"])

            if not s_data:
                continue

            rates = [r["compression_rate"] for r in s_data]
            stds = [r["std"] for r in s_data]

            color = STRATEGY_COLORS[strategy]
            label = STRATEGY_LABELS[strategy]
            ax.plot(rates, stds, "s-", color=color, label=label, linewidth=2, markersize=6)

        # High-variance threshold
        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="High variance (5%)")

        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_xlabel("Compression Rate", fontsize=12)
        ax.set_xlim(-0.02, 0.92)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if idx == 0:
            ax.set_ylabel("Std Dev Across Seeds", fontsize=12)
            ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.suptitle("Variance by Strategy: Which Compression Methods Are Unstable?",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "variance_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading aggregated results...")
    data = load_agg()
    abl_data = load_ablation()
    stats_data = load_stats()

    print(f"  {len(data)} aggregated conditions")
    print(f"  {len(abl_data)} ablation conditions")

    print("\nGenerating figures...")
    plot_main_curve(data)
    plot_cliff_detection(data)
    plot_strategy_comparison(data)
    plot_section_ablation(abl_data, data)
    plot_variance(data)

    print("\nAll figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
