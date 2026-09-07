"""
plot_pipeline_metrics.py

Parses madar_pocket_pipeline.py's pipeline_log.txt (text log, not the .pt
checkpoint) and plots per-task comparison metrics across all 5 lineages --
clean, poisoned_baseline, dropped_rows, amnesiac, opposite_class -- so you
can see at a glance how the three unlearning variants compare to the no-fix
baseline and the never-poisoned reference over the whole run.

Metrics plotted (one subplot each, one line per lineage, fixed color+marker
per lineage across every subplot):
  - task acc            this task's own clean test accuracy
  - adv acc             this task's own perturbed test accuracy (from the
                         "Adversarial test-set breakdown" section's row for
                         the current task -- the only place clean/
                         poisoned_baseline's adversarial accuracy is logged)
  - pooled acc          accuracy pooled over every clean+adversarial test
                         set seen so far (sample-weighted)
  - mean acc            same set of test-set accuracies, task-weighted
  - malicious recall    from the per-lineage classification report
  - benign recall       from the per-lineage classification report
  - still-evades %      dropped_rows/amnesiac/opposite_class only -- not
                         defined for clean/poisoned_baseline
  - genuine pocket rate task-level (not per-lineage): how much of that
                         task's test set was successfully perturbed

Task 0 has no poisoning yet, so adv acc/recall/still-evades are left blank
(NaN) there and pooled/mean acc are set equal to task acc (only one test
set exists at that point, so pooling is trivial).

Usage:
    python plot_pipeline_metrics.py --log runs/.../logs/pipeline_log.txt
    python plot_pipeline_metrics.py --log pipeline_log.txt --out comparison.png
"""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINEAGE_NAMES = ["clean", "poisoned_baseline", "dropped_rows", "amnesiac", "opposite_class"]
FIX_NAMES = ["dropped_rows", "amnesiac", "opposite_class"]
LINEAGE_ALT = "clean|poisoned_baseline|dropped_rows|amnesiac|opposite_class"

# Fixed color + marker per lineage, assigned once and reused identically
# across every subplot -- color/shape follow the entity, never re-cycled
# depending on which lineages happen to have data for a given metric.
STYLE = {
    "clean":             dict(color="#2a78d6", marker="o"),
    "poisoned_baseline": dict(color="#eb6834", marker="s"),
    "dropped_rows":      dict(color="#1baf7a", marker="^"),
    "amnesiac":          dict(color="#eda100", marker="D"),
    "opposite_class":    dict(color="#e87ba4", marker="v"),
}

# NUM tolerates a literal "nan" -- e.g. still-evades % prints "nan%" when a
# task has zero successful genuine pockets to measure recovery against
# (float("nan") parses "nan" back out fine, so downstream code needn't care).
NUM = r"(?:[\d.]+|nan)"

TASK_SPLIT_RE = re.compile(r"^=+\n=== Task (\d+) ===\n=+\n", re.MULTILINE)
TASK0_LINE_RE = re.compile(rf"^({LINEAGE_ALT}): task test acc = ({NUM})\s*$", re.MULTILINE)
ADAPT_TABLE_RE = re.compile(rf"^(clean|poisoned_baseline)\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)
POST_UNLEARN_RE = re.compile(
    rf"^(dropped_rows|amnesiac|opposite_class)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})%\s*$",
    re.MULTILINE,
)
GENUINE_POCKETS_RE = re.compile(rf"genuine pockets found: \d+/\d+ \(({NUM})%\)")
CLASS_MARKER_RE = re.compile(rf"\[({LINEAGE_ALT})\] classification report")
CLASS_ROW_RE = re.compile(rf"^\s*(Benign|Malicious)\s+{NUM}\s+({NUM})\s+{NUM}\s+\d+\s*$", re.MULTILINE)
BREAKDOWN_ROW_RE = re.compile(
    rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE,
)


def _parse_recall(chunk):
    """Returns {lineage: {"Benign": recall, "Malicious": recall}} for every
    lineage whose classification report appears in this task's chunk."""
    out = {}
    for m in CLASS_MARKER_RE.finditer(chunk):
        name = m.group(1)
        window = chunk[m.end():m.end() + 600]
        rows = {row.group(1): float(row.group(2)) for row in CLASS_ROW_RE.finditer(window)}
        if rows:
            out[name] = rows
    return out


def parse_log(log_text):
    """Returns (tasks: sorted list of task ids, data: {lineage: {metric: {task: value}}},
    genuine_pocket_rate: {task: pct})."""
    data = {name: {m: {} for m in
                   ["task_acc", "adv_acc", "pooled_acc", "mean_acc", "Malicious", "Benign", "still_evades_pct"]}
            for name in LINEAGE_NAMES}
    genuine_pocket_rate = {}

    headers = list(re.finditer(r"^=+\n=== Task (\d+) ===\n=+\n", log_text, re.MULTILINE))
    tasks = []
    for i, h in enumerate(headers):
        t = int(h.group(1))
        tasks.append(t)
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(log_text)
        chunk = log_text[start:end]

        if TASK0_LINE_RE.search(chunk):
            for m in TASK0_LINE_RE.finditer(chunk):
                name, acc = m.group(1), float(m.group(2))
                data[name]["task_acc"][t] = acc
                data[name]["pooled_acc"][t] = acc
                data[name]["mean_acc"][t] = acc
            continue

        for m in ADAPT_TABLE_RE.finditer(chunk):
            name, task_acc, pooled_acc, mean_acc = m.group(1), *map(float, m.groups()[1:])
            data[name]["task_acc"][t] = task_acc
            data[name]["pooled_acc"][t] = pooled_acc
            data[name]["mean_acc"][t] = mean_acc

        for m in POST_UNLEARN_RE.finditer(chunk):
            name = m.group(1)
            task_acc, pooled_acc, mean_acc, adv_acc, still_evades = map(float, m.groups()[1:])
            data[name]["task_acc"][t] = task_acc
            data[name]["pooled_acc"][t] = pooled_acc
            data[name]["mean_acc"][t] = mean_acc
            data[name]["adv_acc"][t] = adv_acc
            data[name]["still_evades_pct"][t] = still_evades

        for m in BREAKDOWN_ROW_RE.finditer(chunk):
            row = m.groups()
            if int(row[0]) != t:
                continue
            for name, acc in zip(LINEAGE_NAMES, map(float, row[1:])):
                data[name]["adv_acc"][t] = acc

        recall = _parse_recall(chunk)
        for name, rows in recall.items():
            if "Malicious" in rows:
                data[name]["Malicious"][t] = rows["Malicious"]
            if "Benign" in rows:
                data[name]["Benign"][t] = rows["Benign"]

        gm = GENUINE_POCKETS_RE.search(chunk)
        if gm:
            genuine_pocket_rate[t] = float(gm.group(1))

    return sorted(tasks), data, genuine_pocket_rate


def _series(d, tasks):
    return np.array([d.get(t, np.nan) for t in tasks], dtype=float)


def plot_metrics(tasks, data, genuine_pocket_rate, out_path):
    panels = [
        ("Task accuracy (this task's clean test)", "task_acc", LINEAGE_NAMES),
        ("Adversarial accuracy (this task's perturbed test)", "adv_acc", LINEAGE_NAMES),
        ("Pooled accuracy (all clean+adv sets so far)", "pooled_acc", LINEAGE_NAMES),
        ("Mean accuracy (task-weighted)", "mean_acc", LINEAGE_NAMES),
        ("Malicious recall (this task's clean test)", "Malicious", LINEAGE_NAMES),
        ("Benign recall (this task's clean test)", "Benign", LINEAGE_NAMES),
        ("Still-evades % (of original genuine pockets)", "still_evades_pct", FIX_NAMES),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    axes = axes.ravel()

    for ax, (title, metric, lineages) in zip(axes, panels):
        for name in lineages:
            y = _series(data[name][metric], tasks)
            style = STYLE[name]
            ax.plot(tasks, y, label=name, color=style["color"], marker=style["marker"],
                    linewidth=2, markersize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("task")
        ax.set_xticks(tasks)
        ax.grid(True, alpha=0.25)

    ax = axes[7]
    gp = _series(genuine_pocket_rate, tasks)
    ax.plot(tasks, gp, color="#52514e", marker="x", linewidth=2, markersize=7)
    ax.set_title("Genuine pocket rate (task-level, not per-lineage)", fontsize=10)
    ax.set_xlabel("task")
    ax.set_xticks(tasks)
    ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0], [0], color=STYLE[n]["color"], marker=STYLE[n]["marker"],
                           linewidth=2, markersize=7, label=n) for n in LINEAGE_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("MADAR pocket-pipeline: lineage comparison across tasks", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(tasks, data, genuine_pocket_rate, csv_path):
    """Long-format table (task, lineage, metric, value) alongside the plot --
    a plain-data view, since 3 of the 5 categorical colors sit below the
    minimum contrast ratio against the chart's light surface."""
    metrics = ["task_acc", "adv_acc", "pooled_acc", "mean_acc", "Malicious", "Benign", "still_evades_pct"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "lineage", "metric", "value"])
        for t in tasks:
            for name in LINEAGE_NAMES:
                for metric in metrics:
                    v = data[name][metric].get(t)
                    if v is not None:
                        w.writerow([t, name, metric, v])
            if t in genuine_pocket_rate:
                w.writerow([t, "(pipeline)", "genuine_pocket_rate", genuine_pocket_rate[t]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="Path to pipeline_log.txt")
    ap.add_argument("--out", default=None, help="Output PNG path (default: alongside --log)")
    args = ap.parse_args()

    with open(args.log) as f:
        log_text = f.read()

    tasks, data, genuine_pocket_rate = parse_log(log_text)
    if not tasks:
        raise ValueError(f"No '=== Task N ===' sections found in {args.log} -- is this a pipeline_log.txt?")

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)), "metrics_comparison.png")
    csv_path = os.path.splitext(out_path)[0] + ".csv"

    plot_metrics(tasks, data, genuine_pocket_rate, out_path)
    write_csv(tasks, data, genuine_pocket_rate, csv_path)
    print(f"Parsed {len(tasks)} tasks from {args.log}")
    print(f"Wrote plot to {out_path}")
    print(f"Wrote table to {csv_path}")


if __name__ == "__main__":
    main()
