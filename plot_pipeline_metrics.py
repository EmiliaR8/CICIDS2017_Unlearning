"""
plot_pipeline_metrics.py

Parses one or more of madar_pocket_pipeline.py's pipeline_log.txt files
(text logs, not the .pt checkpoint) and plots per-task comparison metrics
across all 5 lineages -- clean, poisoned_baseline, dropped_rows, amnesiac,
opposite_class. Pass multiple logs (e.g. the same config run under several
seeds) to get a mean line with a +/-1 std shaded band per lineage instead
of a single run's raw numbers.

Each metric is its own opt-in flag and its own output PNG -- nothing is
plotted unless you ask for it (pass --all for the old "everything" behavior).

Available metrics:
  --task-acc              this task's own clean test accuracy
  --adv-acc               this task's own perturbed test accuracy
  --combined-acc          this task's own COMBINED (clean+adversarial,
                           pooled) accuracy -- needs the "...COMBINED..."
                           breakdown table this pipeline logs; older logs
                           that predate it will show gaps for this metric
  --pooled-acc            accuracy pooled over every clean+adversarial test
                           set seen so far (sample-weighted)
  --mean-acc              same set of test-set accuracies, task-weighted
  --malicious-recall      malicious-class recall, this task's clean test
  --benign-recall         benign-class recall, this task's clean test
  --still-evades          dropped_rows/amnesiac/opposite_class only -- % of
                           the task's original genuine pockets still evaded
  --genuine-pocket-rate   task-level (not per-lineage): how much of that
                           task's test set was successfully perturbed

Usage:
    python plot_pipeline_metrics.py --logs run1/pipeline_log.txt --task-acc --adv-acc
    python plot_pipeline_metrics.py --logs seed1/pipeline_log.txt seed2/pipeline_log.txt seed3/pipeline_log.txt \
        --out-dir plots/ --mean-acc --pooled-acc --combined-acc
    python plot_pipeline_metrics.py --logs *.txt --all --out-dir plots/
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

STYLE = {
    "clean":             dict(color="#2a78d6", marker="o"),
    "poisoned_baseline": dict(color="#eb6834", marker="s"),
    "dropped_rows":      dict(color="#1baf7a", marker="^"),
    "amnesiac":          dict(color="#eda100", marker="D"),
    "opposite_class":    dict(color="#e87ba4", marker="v"),
}

NUM = r"(?:[\d.]+|nan)"  # tolerates the literal "nan" printed for undefined still-evades %

TASK0_LINE_RE = re.compile(rf"^({LINEAGE_ALT}): task test acc = ({NUM})\s*$", re.MULTILINE)
ADAPT_TABLE_RE = re.compile(rf"^(clean|poisoned_baseline)\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)
POST_UNLEARN_RE = re.compile(
    rf"^(dropped_rows|amnesiac|opposite_class)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})%\s*$",
    re.MULTILINE,
)
GENUINE_POCKETS_RE = re.compile(rf"genuine pockets found: \d+/\d+ \(({NUM})%\)")
CLASS_MARKER_RE = re.compile(rf"\[({LINEAGE_ALT})\] classification report")
CLASS_ROW_RE = re.compile(rf"^\s*(Benign|Malicious)\s+{NUM}\s+({NUM})\s+{NUM}\s+\d+\s*$", re.MULTILINE)
ADV_HEADER_RE = re.compile(r"Task \d+'s \(post-unlearning\) classifier accuracy on each source task's adv-test-set:")
COMBINED_HEADER_RE = re.compile(
    r"Task \d+'s \(post-unlearning\) classifier COMBINED \(clean\+adversarial, pooled\) "
    r"accuracy on each source task's test-set:"
)
BREAKDOWN_ROW_RE = re.compile(rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)

METRICS = {
    "task_acc":            ("Task accuracy (this task's clean test)", LINEAGE_NAMES),
    "adv_acc":              ("Adversarial accuracy (this task's perturbed test)", LINEAGE_NAMES),
    "combined_acc":         ("Combined accuracy (this task's clean+adversarial, pooled)", LINEAGE_NAMES),
    "pooled_acc":           ("Pooled accuracy (all clean+adv sets so far)", LINEAGE_NAMES),
    "mean_acc":             ("Mean accuracy (task-weighted)", LINEAGE_NAMES),
    "Malicious":            ("Malicious recall (this task's clean test)", LINEAGE_NAMES),
    "Benign":               ("Benign recall (this task's clean test)", LINEAGE_NAMES),
    "still_evades_pct":     ("Still-evades % (of original genuine pockets)", FIX_NAMES),
}
FLAG_TO_METRIC = {
    "task_acc": "task_acc", "adv_acc": "adv_acc", "combined_acc": "combined_acc",
    "pooled_acc": "pooled_acc", "mean_acc": "mean_acc",
    "malicious_recall": "Malicious", "benign_recall": "Benign",
    "still_evades": "still_evades_pct",
}


def _breakdown_diagonal(chunk, header_re, t):
    """{lineage: acc} for the row where source task == t (this task's own
    row) of one of the per-source-task breakdown tables."""
    header = header_re.search(chunk)
    if not header:
        return {}
    next_boundary = chunk.find("\nTask ", header.end())
    window = chunk[header.end(): next_boundary if next_boundary != -1 else len(chunk)]
    for m in BREAKDOWN_ROW_RE.finditer(window):
        if int(m.group(1)) == t:
            return dict(zip(LINEAGE_NAMES, (float(x) for x in m.groups()[1:])))
    return {}


def _parse_recall(chunk):
    out = {}
    for m in CLASS_MARKER_RE.finditer(chunk):
        name = m.group(1)
        window = chunk[m.end():m.end() + 600]
        rows = {row.group(1): float(row.group(2)) for row in CLASS_ROW_RE.finditer(window)}
        if rows:
            out[name] = rows
    return out


def parse_log(log_text):
    """Returns (tasks, data, genuine_pocket_rate) for ONE log -- same shape
    as before, plus a "combined_acc" metric per lineage."""
    data = {name: {m: {} for m in
                   ["task_acc", "adv_acc", "combined_acc", "pooled_acc", "mean_acc", "Malicious", "Benign",
                    "still_evades_pct"]}
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
                data[name]["combined_acc"][t] = acc
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

        adv_diag = _breakdown_diagonal(chunk, ADV_HEADER_RE, t)
        for name, acc in adv_diag.items():
            data[name]["adv_acc"].setdefault(t, acc)  # fills in clean/poisoned_baseline, which lack a table row

        combined_diag = _breakdown_diagonal(chunk, COMBINED_HEADER_RE, t)
        for name, acc in combined_diag.items():
            data[name]["combined_acc"][t] = acc

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


def aggregate(runs):
    """runs: list of (tasks, data, genuine_pocket_rate) from parse_log(),
    one per input log. Returns (all_tasks, mean_data, std_data, n_data,
    mean_gpr, std_gpr) where mean/std_data[lineage][metric][task] is over
    however many runs actually had a value for that cell (NaNs skipped)."""
    all_tasks = sorted(set(t for tasks, _, _ in runs for t in tasks))
    metrics = list(METRICS.keys())
    mean_data = {n: {m: {} for m in metrics} for n in LINEAGE_NAMES}
    std_data = {n: {m: {} for m in metrics} for n in LINEAGE_NAMES}
    n_data = {n: {m: {} for m in metrics} for n in LINEAGE_NAMES}

    for name in LINEAGE_NAMES:
        for metric in metrics:
            for t in all_tasks:
                vals = [data[name][metric][t] for _, data, _ in runs
                        if t in data[name][metric] and data[name][metric][t] == data[name][metric][t]]
                if vals:
                    mean_data[name][metric][t] = float(np.mean(vals))
                    std_data[name][metric][t] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    n_data[name][metric][t] = len(vals)

    gpr_vals_by_task = {}
    for _, _, gpr in runs:
        for t, v in gpr.items():
            gpr_vals_by_task.setdefault(t, []).append(v)
    mean_gpr = {t: float(np.mean(v)) for t, v in gpr_vals_by_task.items()}
    std_gpr = {t: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for t, v in gpr_vals_by_task.items()}

    return all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr


def _series(d, tasks):
    return np.array([d.get(t, np.nan) for t in tasks], dtype=float)


def plot_one_metric(metric_key, tasks, mean_data, std_data, out_path, n_runs):
    title, lineages = METRICS[metric_key]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in lineages:
        y = _series(mean_data[name][metric_key], tasks)
        e = _series(std_data[name][metric_key], tasks)
        style = STYLE[name]
        ax.plot(tasks, y, label=name, color=style["color"], marker=style["marker"], linewidth=2, markersize=7)
        if n_runs > 1:
            ax.fill_between(tasks, y - e, y + e, color=style["color"], alpha=0.15, linewidth=0)
    ax.set_title(title + (f"  (mean ± std, n={n_runs} runs)" if n_runs > 1 else ""), fontsize=11)
    ax.set_xlabel("task")
    ax.set_xticks(tasks)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_genuine_pocket_rate(tasks, mean_gpr, std_gpr, out_path, n_runs):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = _series(mean_gpr, tasks)
    e = _series(std_gpr, tasks)
    ax.plot(tasks, y, color="#52514e", marker="x", linewidth=2, markersize=7)
    if n_runs > 1:
        ax.fill_between(tasks, y - e, y + e, color="#52514e", alpha=0.15, linewidth=0)
    ax.set_title("Genuine pocket rate (task-level, not per-lineage)"
                 + (f"  (mean ± std, n={n_runs} runs)" if n_runs > 1 else ""), fontsize=11)
    ax.set_xlabel("task")
    ax.set_xticks(tasks)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_csv(tasks, mean_data, std_data, n_data, mean_gpr, std_gpr, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "lineage", "metric", "mean", "std", "n_runs"])
        for t in tasks:
            for name in LINEAGE_NAMES:
                for metric in METRICS:
                    if t in mean_data[name][metric]:
                        w.writerow([t, name, metric, mean_data[name][metric][t],
                                    std_data[name][metric][t], n_data[name][metric][t]])
            if t in mean_gpr:
                w.writerow([t, "(pipeline)", "genuine_pocket_rate", mean_gpr[t], std_gpr[t], ""])
    print(f"Wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="One or more pipeline_log.txt paths")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: alongside the first log)")
    for flag in ["task-acc", "adv-acc", "combined-acc", "pooled-acc", "mean-acc",
                 "malicious-recall", "benign-recall", "still-evades", "genuine-pocket-rate"]:
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--all", action="store_true", help="Plot every metric (each still gets its own file)")
    args = ap.parse_args()

    flag_map = {
        "task_acc": args.task_acc, "adv_acc": args.adv_acc, "combined_acc": args.combined_acc,
        "pooled_acc": args.pooled_acc, "mean_acc": args.mean_acc,
        "malicious_recall": args.malicious_recall, "benign_recall": args.benign_recall,
        "still_evades": args.still_evades,
    }
    selected_metrics = {FLAG_TO_METRIC[k] for k, v in flag_map.items() if v}
    want_gpr = args.genuine_pocket_rate

    if args.all:
        selected_metrics = set(METRICS.keys())
        want_gpr = True

    if not selected_metrics and not want_gpr:
        ap.error(
            "No plot selected. Pass one or more of: --task-acc --adv-acc --combined-acc --pooled-acc "
            "--mean-acc --malicious-recall --benign-recall --still-evades --genuine-pocket-rate, or --all."
        )

    runs = []
    for path in args.logs:
        with open(path) as f:
            log_text = f.read()
        tasks, data, gpr = parse_log(log_text)
        if not tasks:
            raise ValueError(f"No '=== Task N ===' sections found in {path} -- is this a pipeline_log.txt?")
        runs.append((tasks, data, gpr))

    task_counts = {len(tasks) for tasks, _, _ in runs}
    if len(task_counts) > 1:
        print(f"WARNING: input logs have different numbers of tasks ({sorted(task_counts)}) -- "
              f"aggregating per-task over however many logs actually reach that task.\n")

    all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr = aggregate(runs)
    n_runs = len(runs)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.logs[0]))
    os.makedirs(out_dir, exist_ok=True)

    for metric in selected_metrics:
        plot_one_metric(metric, all_tasks, mean_data, std_data,
                         os.path.join(out_dir, f"metrics_{metric}.png"), n_runs)
    if want_gpr:
        plot_genuine_pocket_rate(all_tasks, mean_gpr, std_gpr,
                                  os.path.join(out_dir, "metrics_genuine_pocket_rate.png"), n_runs)

    write_csv(all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr,
              os.path.join(out_dir, "metrics_summary.csv"))
    print(f"\nParsed {n_runs} log(s), {len(all_tasks)} task(s) total.")


if __name__ == "__main__":
    main()
