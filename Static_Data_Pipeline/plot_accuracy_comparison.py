"""
Overlay per-task accuracy curves from any number of experiment result JSONs onto one
color-coded plot. Works with naive/joint/MADAR/unlearning's `{LOG_NAME}_seed{SEED}.json`
("results" key) and mu_inner_loop_cicids.py's reward JSON ("checkpoints" key) without
knowing in advance which produced a given file, how many tasks it ran, or how many files
are being compared -- nothing about experiment names/counts/task counts is hardcoded.

Usage:
    python plot_accuracy_comparison.py file1.json file2.json ... [options]

Examples:
    python plot_accuracy_comparison.py naive_run_seed42.json joint_run_seed42.json \
        madar_run_seed42.json unlearn_run1_seed42.json

    python plot_accuracy_comparison.py joint_experiment_1_seed42.json joint_experiment_2_seed7.json \
        --metric both --labels "joint exp1" "joint exp2" --out joint_compare.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANDIDATE_LIST_KEYS = ["results", "checkpoints", "per_task_checkpoints"]
METRIC_STYLE = {"pooled_accuracy": "-", "mean_per_task_accuracy": "--"}


def find_task_records(payload):
    """Return the list of per-task checkpoint dicts inside a loaded JSON payload,
    regardless of which script produced it."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in CANDIDATE_LIST_KEYS:
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
        # Fallback: first top-level list of dicts, whatever it's called
        for val in payload.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
    return None


def extract_series(records, metric):
    xs, ys = [], []
    for i, rec in enumerate(records):
        if metric not in rec:
            continue
        tid = rec.get("task_id", i)
        xs.append(tid)
        ys.append(rec[metric])
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def label_for(path, override):
    if override is not None:
        return override
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description="Plot per-task accuracy from experiment JSON logs")
    parser.add_argument("files", nargs="+", help="Paths to experiment result JSON files")
    parser.add_argument("--metric", choices=["pooled_accuracy", "mean_per_task_accuracy", "both"],
                        default="pooled_accuracy", help="Which accuracy field to plot (default: pooled_accuracy)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels, one per file in order (default: filename)")
    parser.add_argument("--title", default="Accuracy Comparison")
    parser.add_argument("--out", default="accuracy_comparison.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.files):
        parser.error(f"--labels needs one entry per file ({len(args.files)} files, {len(args.labels)} labels)")

    metrics = ["pooled_accuracy", "mean_per_task_accuracy"] if args.metric == "both" else [args.metric]
    cmap = plt.get_cmap("tab20" if len(args.files) > 10 else "tab10")

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for i, path in enumerate(args.files):
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {path}: {e}")
            continue

        records = find_task_records(payload)
        if not records:
            print(f"[skip] {path}: no per-task results list found")
            continue

        color = cmap(i / max(1, len(args.files) - 1)) if len(args.files) > 1 else cmap(0)
        label = label_for(path, args.labels[i] if args.labels else None)

        for metric in metrics:
            xs, ys = extract_series(records, metric)
            if not xs:
                print(f"[skip] {path}: metric '{metric}' not found in any record")
                continue
            line_label = label if len(metrics) == 1 else f"{label} ({metric})"
            plt.plot(xs, ys, marker="o", linestyle=METRIC_STYLE.get(metric, "-"), color=color, label=line_label)
            plotted_any = True

    if not plotted_any:
        raise SystemExit("Nothing plottable found in any input file.")

    plt.xlabel("Task")
    plt.ylabel("Accuracy (%)")
    plt.title(args.title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
