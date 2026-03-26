"""
Statistical analysis of prompt compression experiment results.

Computes: mean accuracy, std, 95% CIs, pairwise t-tests, cliff detection.
Outputs: aggregated_results.csv and statistical_tests.json
"""

import json
import os
import csv
import numpy as np
from collections import defaultdict
from scipy import stats

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
RAW_PATH = os.path.join(RESULTS_DIR, "raw_results.jsonl")
AGG_PATH = os.path.join(RESULTS_DIR, "aggregated_results.csv")
STATS_PATH = os.path.join(RESULTS_DIR, "statistical_tests.json")
ABLATION_PATH = os.path.join(RESULTS_DIR, "ablation_results.csv")


def load_raw_results():
    results = []
    with open(RAW_PATH) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def aggregate_results(results):
    """Group by (task, strategy, compression_rate) and compute stats across seeds."""

    # Group by condition: {(task, strategy, rate): {seed: [correct bools]}}
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["task"], r["strategy"], r["compression_rate"])
        seed = r.get("seed", "none")
        groups[key][seed].append(r["correct"])

    rows = []
    for (task, strategy, rate), seed_data in sorted(groups.items()):
        # Compute accuracy per seed
        seed_accuracies = []
        total_correct = 0
        total_count = 0
        for seed, corrects in seed_data.items():
            acc = sum(corrects) / len(corrects) if corrects else 0
            seed_accuracies.append(acc)
            total_correct += sum(corrects)
            total_count += len(corrects)

        mean_acc = np.mean(seed_accuracies)
        std_acc = np.std(seed_accuracies, ddof=1) if len(seed_accuracies) > 1 else 0.0
        n_seeds = len(seed_accuracies)

        # 95% CI
        if n_seeds > 1:
            ci = stats.t.interval(0.95, df=n_seeds-1, loc=mean_acc, scale=std_acc/np.sqrt(n_seeds))
            ci_lower, ci_upper = ci
        else:
            # For single-seed (LLM-guided), use binomial CI
            ci = stats.binom.interval(0.95, total_count, mean_acc) if total_count > 0 else (0, 0)
            ci_lower = ci[0] / total_count if total_count > 0 else 0
            ci_upper = ci[1] / total_count if total_count > 0 else 0

        n_examples = total_count
        high_variance = std_acc > 0.05

        rows.append({
            "task": task,
            "strategy": strategy,
            "compression_rate": rate,
            "mean_accuracy": round(mean_acc, 4),
            "std": round(std_acc, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_seeds": n_seeds,
            "n_examples": n_examples,
            "high_variance": high_variance,
        })

    return rows


def aggregate_ablation(results):
    """Aggregate ablation results separately."""
    ablation = [r for r in results if r["strategy"].startswith("ablation_")]
    if not ablation:
        return []

    groups = defaultdict(lambda: defaultdict(list))
    for r in ablation:
        section = r["strategy"].replace("ablation_", "")
        key = (r["task"], section, r["compression_rate"])
        seed = r.get("seed", "none")
        groups[key][seed].append(r["correct"])

    rows = []
    for (task, section, rate), seed_data in sorted(groups.items()):
        seed_accs = [sum(c)/len(c) for c in seed_data.values() if c]
        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else 0.0

        rows.append({
            "task": task,
            "section": section,
            "compression_rate": rate,
            "mean_accuracy": round(mean_acc, 4),
            "std": round(std_acc, 4),
            "n_seeds": len(seed_accs),
        })
    return rows


def pairwise_ttests(results):
    """Run pairwise t-tests between strategies at each compression level."""
    # Main strategies only (not ablation)
    main_strategies = ["random_dropout", "stopword_removal", "entity_preserving", "llm_guided"]

    # Group: {(task, strategy, rate): [per-seed accuracies]}
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["strategy"] in main_strategies:
            key = (r["task"], r["strategy"], r["compression_rate"])
            seed = r.get("seed", "none")
            groups[key][seed].append(r["correct"])

    # Compute per-seed accuracy arrays
    acc_arrays = {}
    for key, seed_data in groups.items():
        accs = [sum(c)/len(c) for c in seed_data.values() if c]
        acc_arrays[key] = accs

    # Pairwise comparisons
    test_results = []
    for task in ["alc", "slr"]:
        for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for i, s1 in enumerate(main_strategies):
                for s2 in main_strategies[i+1:]:
                    a1 = acc_arrays.get((task, s1, rate), [])
                    a2 = acc_arrays.get((task, s2, rate), [])

                    if len(a1) >= 2 and len(a2) >= 2:
                        t_stat, p_val = stats.ttest_ind(a1, a2)
                        significant = p_val < 0.05
                    else:
                        t_stat, p_val, significant = None, None, None

                    test_results.append({
                        "task": task,
                        "compression_rate": rate,
                        "strategy_1": s1,
                        "strategy_2": s2,
                        "mean_1": round(np.mean(a1), 4) if a1 else None,
                        "mean_2": round(np.mean(a2), 4) if a2 else None,
                        "t_statistic": round(t_stat, 4) if t_stat is not None else None,
                        "p_value": round(p_val, 6) if p_val is not None else None,
                        "significant": bool(significant) if significant is not None else None,
                    })

    return test_results


def detect_cliffs(agg_rows):
    """Find cliff point: first rate where accuracy drops >10% from baseline."""
    cliffs = {}
    for task in ["alc", "slr"]:
        for strategy in ["random_dropout", "stopword_removal", "entity_preserving", "llm_guided"]:
            relevant = [r for r in agg_rows
                        if r["task"] == task and r["strategy"] == strategy]
            relevant.sort(key=lambda r: r["compression_rate"])

            if not relevant:
                continue

            baseline = relevant[0]["mean_accuracy"]  # 0% compression
            cliff_rate = None

            for r in relevant[1:]:
                drop = baseline - r["mean_accuracy"]
                if drop > 0.10:
                    cliff_rate = r["compression_rate"]
                    break

            cliffs[f"{task}|{strategy}"] = {
                "task": task,
                "strategy": strategy,
                "baseline_accuracy": baseline,
                "cliff_rate": cliff_rate,
                "cliff_description": f">{10}% drop at {cliff_rate:.0%}" if cliff_rate else "No cliff detected (≤10% drop across all rates)",
            }

    return cliffs


def main():
    print("Loading raw results...")
    results = load_raw_results()
    print(f"  {len(results)} total trial results")

    # Filter main vs ablation
    main_results = [r for r in results if not r["strategy"].startswith("ablation_")]
    ablation_results = [r for r in results if r["strategy"].startswith("ablation_")]

    print(f"  Main: {len(main_results)}, Ablation: {len(ablation_results)}")

    # Aggregate
    print("\nAggregating main results...")
    agg_rows = aggregate_results(main_results)

    # Write CSV
    fieldnames = ["task", "strategy", "compression_rate", "mean_accuracy", "std",
                   "ci_lower", "ci_upper", "n_seeds", "n_examples", "high_variance"]
    with open(AGG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"  Saved {len(agg_rows)} rows to {AGG_PATH}")

    # Ablation
    if ablation_results:
        print("\nAggregating ablation results...")
        abl_rows = aggregate_ablation(ablation_results)
        abl_fields = ["task", "section", "compression_rate", "mean_accuracy", "std", "n_seeds"]
        with open(ABLATION_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=abl_fields)
            writer.writeheader()
            writer.writerows(abl_rows)
        print(f"  Saved {len(abl_rows)} rows to {ABLATION_PATH}")

    # Statistical tests
    print("\nRunning pairwise t-tests...")
    ttests = pairwise_ttests(main_results)
    sig_count = sum(1 for t in ttests if t.get("significant"))
    print(f"  {len(ttests)} comparisons, {sig_count} significant (p<0.05)")

    # Cliff detection
    print("\nDetecting performance cliffs...")
    cliffs = detect_cliffs(agg_rows)
    for key, info in cliffs.items():
        print(f"  {key}: {info['cliff_description']}")

    # Save stats
    stats_output = {
        "pairwise_ttests": ttests,
        "cliff_detection": cliffs,
        "summary": {
            "total_trials": len(results),
            "main_trials": len(main_results),
            "ablation_trials": len(ablation_results),
            "conditions": len(agg_rows),
            "significant_comparisons": sig_count,
        }
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nSaved statistical tests to {STATS_PATH}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Task':<6} {'Strategy':<22} {'Rate':<6} {'Accuracy':<10} {'Std':<8} {'95% CI':<20}")
    print(f"{'='*80}")
    for r in agg_rows:
        ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        flag = " ⚠" if r["high_variance"] else ""
        print(f"{r['task']:<6} {r['strategy']:<22} {r['compression_rate']:<6.0%} "
              f"{r['mean_accuracy']:<10.4f} {r['std']:<8.4f} {ci:<20}{flag}")


if __name__ == "__main__":
    main()
