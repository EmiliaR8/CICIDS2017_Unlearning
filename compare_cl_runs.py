"""
compare_cl_runs.py

Compares multiple naive_cl_pipeline.py / joint_cl_pipeline.py (or any future
pipeline sharing the same JSON schema) result files side by side, one line
per run, on a shared 2x2 grid:

  (a) pooled balanced accuracy per task
  (b) mean-per-task balanced accuracy per task
  (c) red-agent evasion rate per task (held-out test samples; task 0 has no
      red agent and is skipped)
  (d) achieved poison rate per task (n_poisoned / n_malicious_train; task 0
      is always 0 by construction and is skipped for readability)

Each input JSON becomes one color-consistent line across all four panels,
labeled by its filename (no directory, no ".json"). Works across runs with
different task counts or configs -- each line just plots its own file's
task_ids, nothing is assumed to line up across inputs beyond that.

Usage:
    python compare_cl_runs.py naive1.json naive2.json joint1.json \\
        --out comparison1.png [--include_table]

Output: Unlearning_experiments/comparative_plots/<--out> (created if it
doesn't exist yet). With --include_table, also writes a same-named .csv
summary table (final-task accuracy/evasion/poison-rate per run) next to the
PNG, and prints it to stdout.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "comparative_plots"


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    label = os.path.splitext(os.path.basename(path))[0]
    return label, data


def extract_series(results):
    """Pulls the four comparison metrics out of one run's `results` list,
    keyed by task_id, skipping points where the metric doesn't apply
    (task 0 has no red_agent; a task with zero malicious training samples
    has no meaningful poison rate)."""
    pooled_acc, mean_task_acc = [], []
    evasion, poison_rate = [], []

    for r in results:
        tid = r["task_id"]
        pooled_acc.append((tid, r["pooled_eval"]["balanced_accuracy"]))
        mean_task_acc.append((tid, r["mean_per_task_balanced_accuracy"]))

        if r["red_agent"] is not None:
            evasion.append((tid, r["red_agent"]["test_evasion_rate"]))

        n_mal = r.get("n_malicious_train", 0)
        if n_mal > 0 and r.get("n_poisoned") is not None:
            poison_rate.append((tid, r["n_poisoned"] / n_mal))

    return {
        "pooled_acc": pooled_acc,
        "mean_task_acc": mean_task_acc,
        "evasion": evasion,
        "poison_rate": poison_rate,
    }


def plot_series(ax, all_series, key, colors, title, ylabel):
    for label, series in all_series:
        points = series[key]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", color=colors[label], label=label)
    ax.set_title(title)
    ax.set_xlabel("Task id")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)


def build_summary_table(runs):
    rows = []
    for label, data, series in runs:
        strategy = data.get("config", {}).get("strategy", "naive" if "naive" in label else "unknown")
        last_pooled = series["pooled_acc"][-1][1] if series["pooled_acc"] else None
        last_mean_task = series["mean_task_acc"][-1][1] if series["mean_task_acc"] else None
        last_evasion = series["evasion"][-1][1] if series["evasion"] else None
        avg_poison_rate = (
            sum(v for _, v in series["poison_rate"]) / len(series["poison_rate"])
            if series["poison_rate"] else None
        )
        rows.append({
            "run": label,
            "strategy": strategy,
            "final_pooled_balanced_accuracy": last_pooled,
            "final_mean_per_task_balanced_accuracy": last_mean_task,
            "final_test_evasion_rate": last_evasion,
            "avg_poison_rate_achieved": avg_poison_rate,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_files", nargs="+", help="Result JSON files to compare")
    ap.add_argument("--out", type=str, default="comparison.png", help="Output PNG filename")
    ap.add_argument("--include_table", action="store_true",
                     help="Also write/print a final-task summary table (CSV alongside the PNG)")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = []
    for path in args.json_files:
        label, data = load_run(path)
        series = extract_series(data["results"])
        runs.append((label, data, series))

    cmap = matplotlib.colormaps["tab10"]
    colors = {label: cmap(i % 10) for i, (label, _, _) in enumerate(runs)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_series(axes[0, 0], [(l, s) for l, _, s in runs], "pooled_acc", colors,
                "Pooled balanced accuracy per task", "Balanced accuracy")
    plot_series(axes[0, 1], [(l, s) for l, _, s in runs], "mean_task_acc", colors,
                "Mean per-task balanced accuracy per task", "Balanced accuracy")
    plot_series(axes[1, 0], [(l, s) for l, _, s in runs], "evasion", colors,
                "Red agent evasion rate per task (test)", "Evasion rate")
    plot_series(axes[1, 1], [(l, s) for l, _, s in runs], "poison_rate", colors,
                "Achieved poison rate per task", "n_poisoned / n_malicious_train")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(runs), 5), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = os.path.join(OUTPUT_DIR, args.out)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")

    if args.include_table:
        table = build_summary_table(runs)
        print("\n" + table.to_string(index=False))
        csv_path = os.path.join(OUTPUT_DIR, os.path.splitext(args.out)[0] + "_table.csv")
        table.to_csv(csv_path, index=False)
        print(f"\nSaved summary table to {csv_path}")


if __name__ == "__main__":
    main()
