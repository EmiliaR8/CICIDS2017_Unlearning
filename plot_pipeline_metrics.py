"""
plot_pipeline_metrics.py

Parses one or more pipeline_log.txt files from any of this project's
"pocket" pipelines (madar_pocket_pipeline.py's 5 lineages -- clean,
poisoned_baseline, dropped_rows, amnesiac, opposite_class -- or
madar_pocket_pipeline_si_agem.py's 4 -- clean, poisoned_baseline, si, agem --
or any future pipeline following the same log shape) and plots per-task
comparison metrics across whichever lineages that log actually contains.
Lineage names are auto-detected from each log's own Task 0 section and
per-task tables -- nothing is hardcoded, so old and new logs both work, and a
future pipeline with a different lineage set needs no changes here. Pass
multiple logs (e.g. the same config run under several seeds) to get a mean
line with a +/-1 std shaded band per lineage instead of a single run's raw
numbers. Mixing logs from pipelines with different lineage sets in one
--logs call is allowed -- each lineage is aggregated only over the runs that
actually contain it.

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
  --still-evades          the "under test" lineages only (whichever ones get
                           a 5-column post-fix/post-adaptation table in the
                           log -- dropped_rows/amnesiac/opposite_class in the
                           old pipeline, si/agem in the new one) -- % of the
                           task's original genuine pockets still evaded
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
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUM = r"(?:[\d.]+|nan)"  # tolerates the literal "nan" printed for undefined still-evades %

# Known lineages get a fixed, stable color/marker across runs and pipelines;
# anything else (a future pipeline's new lineage name) gets one assigned
# deterministically from FALLBACK_STYLE the first time it's seen.
KNOWN_STYLE = {
    "clean":             dict(color="#2a78d6", marker="o"),
    "poisoned_baseline": dict(color="#eb6834", marker="s"),
    "dropped_rows":      dict(color="#1baf7a", marker="^"),
    "amnesiac":          dict(color="#eda100", marker="D"),
    "opposite_class":    dict(color="#e87ba4", marker="v"),
    "si":                dict(color="#7a5cdb", marker="P"),
    "agem":              dict(color="#2fb5b0", marker="X"),
}
FALLBACK_STYLE = [
    dict(color="#8c8c8c", marker="*"), dict(color="#c94141", marker="h"),
    dict(color="#4d9e4d", marker="8"), dict(color="#b8860b", marker="p"),
]


def style_for(name, style_cache):
    if name in KNOWN_STYLE:
        return KNOWN_STYLE[name]
    if name not in style_cache:
        style_cache[name] = FALLBACK_STYLE[len(style_cache) % len(FALLBACK_STYLE)]
    return style_cache[name]


# Lineage names are captured generically (a name, not a fixed alternation)
# -- the surrounding literal text (column shape, trailing '%', end-of-line
# anchors) is specific enough to this pipeline family's log format that no
# other line accidentally matches. NAME requires a leading letter/underscore
# (never a bare digit): the per-source-task breakdown tables have rows
# shaped "<source task index> <NUM> <NUM> ...", and a pipeline with exactly
# 3 lineages produces exactly 3 numeric columns there -- indistinguishable
# from an Adaptation-step row's shape if the name could be a plain integer.
# See git history for the previous fixed-alternation version if the
# leading-letter assumption ever breaks (it would need a name that is a bare
# number, which nothing in this project's pipelines uses).
NAME = r"[A-Za-z_]\w*"
TASK0_LINE_RE = re.compile(rf"^({NAME}): task test acc = ({NUM})\s*$", re.MULTILINE)
ADAPT_TABLE_RE = re.compile(rf"^({NAME})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)
POST_FIX_RE = re.compile(
    rf"^({NAME})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})%\s*$",
    re.MULTILINE,
)
GENUINE_POCKETS_RE = re.compile(rf"genuine pockets found: \d+/\d+ \(({NUM})%\)")
CLASS_MARKER_RE = re.compile(rf"\[({NAME})\] classification report")
CLASS_ROW_RE = re.compile(rf"^\s*(Benign|Malicious)\s+{NUM}\s+({NUM})\s+{NUM}\s+\d+\s*$", re.MULTILINE)
# The parenthesized phrase varies by pipeline ("post-unlearning" in the
# detector+fix-variant pipeline, "post-adaptation" in the SI/A-GEM one).
ADV_HEADER_RE = re.compile(r"Task \d+'s \([\w -]+\) classifier accuracy on each source task's adv-test-set:")
COMBINED_HEADER_RE = re.compile(
    r"Task \d+'s \([\w -]+\) classifier COMBINED \(clean\+adversarial, pooled\) "
    r"accuracy on each source task's test-set:"
)

METRIC_KEYS = ["task_acc", "adv_acc", "combined_acc", "pooled_acc", "mean_acc",
               "Malicious", "Benign", "still_evades_pct"]

METRIC_TITLES = {
    "task_acc":         "Task accuracy (this task's clean test)",
    "adv_acc":          "Adversarial accuracy (this task's perturbed test)",
    "combined_acc":     "Combined accuracy (this task's clean+adversarial, pooled)",
    "pooled_acc":       "Pooled accuracy (all clean+adv sets so far)",
    "mean_acc":         "Mean accuracy (task-weighted)",
    "Malicious":        "Malicious recall (this task's clean test)",
    "Benign":           "Benign recall (this task's clean test)",
    "still_evades_pct": "Still-evades % (of original genuine pockets)",
}
FLAG_TO_METRIC = {
    "task_acc": "task_acc", "adv_acc": "adv_acc", "combined_acc": "combined_acc",
    "pooled_acc": "pooled_acc", "mean_acc": "mean_acc",
    "malicious_recall": "Malicious", "benign_recall": "Benign",
    "still_evades": "still_evades_pct",
}


def _new_lineage_data():
    return {m: {} for m in METRIC_KEYS}


def _breakdown_diagonal(chunk, header_re, t):
    """{lineage: acc} for the row where source task == t (this task's own
    row) of one of the per-source-task breakdown tables. Column names come
    from that table's own header line, not a hardcoded list, so tables with
    any number/order of lineages parse correctly."""
    header = header_re.search(chunk)
    if not header:
        return {}
    header_line_start = chunk.find("\n", header.end()) + 1
    if header_line_start == 0:
        return {}
    header_line_end = chunk.find("\n", header_line_start)
    header_line = chunk[header_line_start: header_line_end if header_line_end != -1 else len(chunk)]
    tokens = header_line.split()
    if len(tokens) < 3 or tokens[0] != "source" or tokens[1] != "task":
        return {}
    names = tokens[2:]

    row_re = re.compile(rf"^(\d+)\s+" + r"\s+".join(f"({NUM})" for _ in names) + r"\s*$", re.MULTILINE)
    body_start = header_line_end + 1 if header_line_end != -1 else len(chunk)
    next_boundary = chunk.find("\nTask ", body_start)
    window = chunk[body_start: next_boundary if next_boundary != -1 else len(chunk)]
    for m in row_re.finditer(window):
        if int(m.group(1)) == t:
            return dict(zip(names, (float(x) for x in m.groups()[1:])))
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
    """Returns (tasks, data, genuine_pocket_rate, lineage_order, fix_names)
    for ONE log. lineage_order lists every lineage name this log contains,
    in first-seen order (Task 0's lines, normally). fix_names is the subset
    that gets a 5-column post-fix/post-adaptation table (the ones eligible
    for the still-evades metric)."""
    data = defaultdict(_new_lineage_data)
    genuine_pocket_rate = {}
    lineage_order = []
    fix_names = set()

    def _seen(name):
        if name not in lineage_order:
            lineage_order.append(name)

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
                _seen(name)
                data[name]["task_acc"][t] = acc
                data[name]["pooled_acc"][t] = acc
                data[name]["mean_acc"][t] = acc
                data[name]["combined_acc"][t] = acc
            continue

        for m in ADAPT_TABLE_RE.finditer(chunk):
            name, task_acc, pooled_acc, mean_acc = m.group(1), *map(float, m.groups()[1:])
            _seen(name)
            data[name]["task_acc"][t] = task_acc
            data[name]["pooled_acc"][t] = pooled_acc
            data[name]["mean_acc"][t] = mean_acc

        for m in POST_FIX_RE.finditer(chunk):
            name = m.group(1)
            task_acc, pooled_acc, mean_acc, adv_acc, still_evades = map(float, m.groups()[1:])
            _seen(name)
            fix_names.add(name)
            data[name]["task_acc"][t] = task_acc
            data[name]["pooled_acc"][t] = pooled_acc
            data[name]["mean_acc"][t] = mean_acc
            data[name]["adv_acc"][t] = adv_acc
            data[name]["still_evades_pct"][t] = still_evades

        adv_diag = _breakdown_diagonal(chunk, ADV_HEADER_RE, t)
        for name, acc in adv_diag.items():
            _seen(name)
            data[name]["adv_acc"].setdefault(t, acc)  # fills in the reference lineages, which lack a table row

        combined_diag = _breakdown_diagonal(chunk, COMBINED_HEADER_RE, t)
        for name, acc in combined_diag.items():
            _seen(name)
            data[name]["combined_acc"][t] = acc

        recall = _parse_recall(chunk)
        for name, rows in recall.items():
            _seen(name)
            if "Malicious" in rows:
                data[name]["Malicious"][t] = rows["Malicious"]
            if "Benign" in rows:
                data[name]["Benign"][t] = rows["Benign"]

        gm = GENUINE_POCKETS_RE.search(chunk)
        if gm:
            genuine_pocket_rate[t] = float(gm.group(1))

    return sorted(tasks), dict(data), genuine_pocket_rate, lineage_order, fix_names


def aggregate(runs):
    """runs: list of (tasks, data, genuine_pocket_rate, lineage_order,
    fix_names) from parse_log(), one per input log. Returns (all_tasks,
    mean_data, std_data, n_data, mean_gpr, std_gpr, lineage_order,
    fix_names) where mean/std_data[lineage][metric][task] is over however
    many runs actually had a value for that cell (NaNs skipped), and
    lineage_order/fix_names are the union across all runs, in first-seen
    order."""
    all_tasks = sorted(set(t for tasks, *_ in runs for t in tasks))

    lineage_order = []
    for _, _, _, order, _ in runs:
        for name in order:
            if name not in lineage_order:
                lineage_order.append(name)
    fix_names = set()
    for _, _, _, _, fn in runs:
        fix_names |= fn

    mean_data = {n: {m: {} for m in METRIC_KEYS} for n in lineage_order}
    std_data = {n: {m: {} for m in METRIC_KEYS} for n in lineage_order}
    n_data = {n: {m: {} for m in METRIC_KEYS} for n in lineage_order}

    for name in lineage_order:
        for metric in METRIC_KEYS:
            for t in all_tasks:
                vals = [data[name][metric][t] for _, data, _, _, _ in runs
                        if name in data and t in data[name][metric]
                        and data[name][metric][t] == data[name][metric][t]]
                if vals:
                    mean_data[name][metric][t] = float(np.mean(vals))
                    std_data[name][metric][t] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    n_data[name][metric][t] = len(vals)

    gpr_vals_by_task = {}
    for _, _, gpr, _, _ in runs:
        for t, v in gpr.items():
            gpr_vals_by_task.setdefault(t, []).append(v)
    mean_gpr = {t: float(np.mean(v)) for t, v in gpr_vals_by_task.items()}
    std_gpr = {t: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for t, v in gpr_vals_by_task.items()}

    return all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr, lineage_order, fix_names


def _series(d, tasks):
    return np.array([d.get(t, np.nan) for t in tasks], dtype=float)


def plot_one_metric(metric_key, lineages, tasks, mean_data, std_data, out_path, n_runs, style_cache):
    title = METRIC_TITLES[metric_key]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in lineages:
        y = _series(mean_data[name][metric_key], tasks)
        e = _series(std_data[name][metric_key], tasks)
        style = style_for(name, style_cache)
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


def write_csv(lineages, tasks, mean_data, std_data, n_data, mean_gpr, std_gpr, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "lineage", "metric", "mean", "std", "n_runs"])
        for t in tasks:
            for name in lineages:
                for metric in METRIC_KEYS:
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
        selected_metrics = set(METRIC_KEYS)
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
        tasks, data, gpr, order, fix_names = parse_log(log_text)
        if not tasks:
            raise ValueError(f"No '=== Task N ===' sections found in {path} -- is this a pipeline_log.txt?")
        runs.append((tasks, data, gpr, order, fix_names))
        print(f"{path}: detected lineages {order}")

    task_counts = {len(tasks) for tasks, *_ in runs}
    if len(task_counts) > 1:
        print(f"WARNING: input logs have different numbers of tasks ({sorted(task_counts)}) -- "
              f"aggregating per-task over however many logs actually reach that task.\n")

    all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr, lineage_order, fix_names = aggregate(runs)
    n_runs = len(runs)
    fix_order = [n for n in lineage_order if n in fix_names]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.logs[0]))
    os.makedirs(out_dir, exist_ok=True)
    style_cache = {}

    for metric in selected_metrics:
        lineages = fix_order if metric == "still_evades_pct" else lineage_order
        plot_one_metric(metric, lineages, all_tasks, mean_data, std_data,
                         os.path.join(out_dir, f"metrics_{metric}.png"), n_runs, style_cache)
    if want_gpr:
        plot_genuine_pocket_rate(all_tasks, mean_gpr, std_gpr,
                                  os.path.join(out_dir, "metrics_genuine_pocket_rate.png"), n_runs)

    write_csv(lineage_order, all_tasks, mean_data, std_data, n_data, mean_gpr, std_gpr,
              os.path.join(out_dir, "metrics_summary.csv"))
    print(f"\nParsed {n_runs} log(s), {len(all_tasks)} task(s) total, lineages: {lineage_order}.")


if __name__ == "__main__":
    main()
