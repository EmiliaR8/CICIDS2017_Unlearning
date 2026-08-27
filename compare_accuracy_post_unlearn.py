"""
compare_accuracy_post_unlearn.py

Compares an arbitrary number of pipeline result JSONs (naive/joint/MADAR/
MADAR+Unlearning -- any file sharing the shared results-list schema) side by
side on two panels:

  (a) mean-per-task balanced accuracy per task
  (b) pooled balanced accuracy per task

The one thing this does differently from compare_cl_runs.py: for any task
where a result entry's "unlearning" dict has a POST-unlearning snapshot
(post_unlearn_mean_per_task_balanced_accuracy / post_unlearn_pooled_eval,
see madar_unlearning_cl_pipeline.py's main()), that POST-unlearning value is
plotted instead of the PRE-unlearning one -- i.e. this shows each run's
ACTUAL, deployed accuracy after its full per-task pipeline (including
unlearning, when the file has it), not the snapshot taken right after CL
training and before the forget step. Tasks with no post-unlearning snapshot
(task 0; a task whose forget set ended up empty; any file that isn't a
MADAR+Unlearning run at all) fall back to the PRE-unlearning value, since
that IS the final state there. Any run that used at least one post-unlearning
value anywhere gets " (post-unlearn)" appended to its legend label, so it's
visually clear at a glance which lines reflect unlearning's own effect.

Each input JSON becomes one color-consistent line across both panels,
labeled by its filename (no directory, no ".json"). Works across runs with
different task counts or configs -- each line just plots its own file's
task_ids, nothing is assumed to line up across inputs beyond that.

Usage:
    python compare_accuracy_post_unlearn.py madar1.json madar_u1.json \\
        --out accuracy_comparison.png

    python compare_accuracy_post_unlearn.py runs/madar/m1/m1.json \\
        runs/madar_unlearning/mu1/mu1.json runs/madar_unlearning/mu2/mu2.json \\
        --out madar_vs_two_unlearning_runs.png

Output: comparative_plots/<--out> (created if it doesn't exist yet).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "comparative_plots"


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    label = os.path.splitext(os.path.basename(path))[0]
    return label, data


def extract_series(results):
    """Pulls mean-per-task and pooled balanced accuracy out of one run's
    `results` list, keyed by task_id -- substituting the POST-unlearning
    snapshot wherever a task has one, falling back to the PRE-unlearning
    value otherwise (see module docstring). Returns the two series plus
    whether any post-unlearning value was actually used, so the caller can
    label the line accordingly."""
    pooled_acc, mean_task_acc = [], []
    used_post_unlearn = False

    for r in results:
        tid = r["task_id"]
        pre_pooled = r["pooled_eval"]["balanced_accuracy"]
        pre_mean_task = r["mean_per_task_balanced_accuracy"]

        u = r.get("unlearning") or {}
        post_pooled_eval = u.get("post_unlearn_pooled_eval")
        post_mean_task = u.get("post_unlearn_mean_per_task_balanced_accuracy")

        pooled = post_pooled_eval["balanced_accuracy"] if post_pooled_eval else pre_pooled
        mean_task = post_mean_task if post_mean_task is not None else pre_mean_task
        if post_pooled_eval is not None or post_mean_task is not None:
            used_post_unlearn = True

        pooled_acc.append((tid, pooled))
        mean_task_acc.append((tid, mean_task))

    return {
        "pooled_acc": pooled_acc,
        "mean_task_acc": mean_task_acc,
        "used_post_unlearn": used_post_unlearn,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_files", nargs="+", help="Result JSON files to compare")
    ap.add_argument("--out", type=str, default="accuracy_comparison_post_unlearn.png",
                     help="Output PNG filename")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = []
    for path in args.json_files:
        label, data = load_run(path)
        series = extract_series(data["results"])
        if series["used_post_unlearn"]:
            label = f"{label} (post-unlearn)"
        runs.append((label, series))

    cmap = matplotlib.colormaps["tab10"]
    colors = {label: cmap(i % 10) for i, (label, _) in enumerate(runs)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_series(ax1, runs, "mean_task_acc", colors,
                "Mean per-task balanced accuracy per task", "Balanced accuracy")
    plot_series(ax2, runs, "pooled_acc", colors,
                "Pooled balanced accuracy per task", "Balanced accuracy")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(runs), 4), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = os.path.join(OUTPUT_DIR, args.out)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
