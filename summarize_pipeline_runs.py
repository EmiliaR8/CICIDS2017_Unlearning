"""
summarize_pipeline_runs.py

Aggregates N runs of madar_pocket_pipeline.py's pipeline_log.txt (e.g. the
same config run under different seeds) into one mean +/- std table per
lineage, at the FINAL task of each log:

  method | mean acc @ task N | pooled acc @ task N | task R acc @ task N |
  task N acc | precision (macro) | recall (macro) | F1 (macro)

N = each log's own final task; R = --reference-task (default 1).

"task R acc @ task N" and "task N acc" are COMBINED (clean + adversarial)
accuracy for that one task's data specifically -- not the pooled/mean
figures, which mix every task's clean+adversarial sets together. Since a
task's clean and adversarial test sets are always the same size, combining
them is just the plain average of the two accuracies (equal weighting).

"task R acc @ task N" needs the final task's "...CLEAN test-set..." per-
source-task breakdown table, which this pipeline only started logging
alongside the pre-existing "...adv-test-set..." one -- logs from before
that change will show n/a for that column (a warning is printed once,
naming the affected files); every other column works on any log.

Usage:
    python summarize_pipeline_runs.py --logs run1/pipeline_log.txt run2/pipeline_log.txt ...
    python summarize_pipeline_runs.py --logs runs/*/logs/pipeline_log.txt --reference-task 1
"""
import argparse
import re
import statistics

LINEAGE_NAMES = ["clean", "poisoned_baseline", "dropped_rows", "amnesiac", "opposite_class"]
LINEAGE_ALT = "clean|poisoned_baseline|dropped_rows|amnesiac|opposite_class"
NUM = r"(?:[\d.]+|nan)"  # tolerates the literal "nan" the log prints for undefined still-evades %

TASK_HEADER_RE = re.compile(r"^=+\n=== Task (\d+) ===\n=+\n", re.MULTILINE)
ADAPT_TABLE_RE = re.compile(rf"^(clean|poisoned_baseline)\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)
POST_UNLEARN_RE = re.compile(
    rf"^(dropped_rows|amnesiac|opposite_class)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})%\s*$",
    re.MULTILINE,
)
CLASS_MARKER_RE = re.compile(rf"\[({LINEAGE_ALT})\] classification report")
MACRO_ROW_RE = re.compile(rf"^\s*macro avg\s+({NUM})\s+({NUM})\s+({NUM})\s+\d+\s*$", re.MULTILINE)
ADV_BREAKDOWN_HEADER_RE = re.compile(
    r"Task \d+'s \(post-unlearning\) classifier accuracy on each source task's adv-test-set:"
)
CLEAN_BREAKDOWN_HEADER_RE = re.compile(
    r"Task \d+'s \(post-unlearning\) classifier accuracy on each source task's CLEAN test-set:"
)
BREAKDOWN_ROW_RE = re.compile(rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)


def _split_last_task(log_text):
    headers = list(TASK_HEADER_RE.finditer(log_text))
    if not headers:
        raise ValueError("No '=== Task N ===' sections found -- is this a pipeline_log.txt?")
    last = headers[-1]
    return int(last.group(1)), log_text[last.end():]


def _final_task_table_metrics(chunk):
    """{lineage: {task_acc, pooled_acc, mean_acc, adv_acc}} from the final
    task's Adaptation-step (clean/poisoned_baseline, no adv_acc there) and
    Post-unlearning (3 fix variants, has adv_acc) tables."""
    out = {}
    for m in ADAPT_TABLE_RE.finditer(chunk):
        name = m.group(1)
        task_acc, pooled_acc, mean_acc = (float(x) for x in m.groups()[1:])
        out[name] = dict(task_acc=task_acc, pooled_acc=pooled_acc, mean_acc=mean_acc, adv_acc=None)
    for m in POST_UNLEARN_RE.finditer(chunk):
        name = m.group(1)
        task_acc, pooled_acc, mean_acc, adv_acc, _still_evades = (float(x) for x in m.groups()[1:])
        out[name] = dict(task_acc=task_acc, pooled_acc=pooled_acc, mean_acc=mean_acc, adv_acc=adv_acc)
    return out


def _final_task_macro_prf1(chunk):
    """{lineage: (precision, recall, f1)} from the "macro avg" row of each
    lineage's classification report in the final task's chunk."""
    out = {}
    for m in CLASS_MARKER_RE.finditer(chunk):
        name = m.group(1)
        window = chunk[m.end():m.end() + 700]
        row = MACRO_ROW_RE.search(window)
        if row:
            out[name] = tuple(float(x) for x in row.groups())
    return out


def _breakdown_row(chunk, header_re, source_task):
    """{lineage: accuracy} for one source-task row of either breakdown
    table (adv or clean). Window is bounded to end right before the NEXT
    "Task N's..." header (the other breakdown table), since both tables
    share an identical row format and would otherwise bleed into each
    other whenever `source_task` isn't present in the first table."""
    header = header_re.search(chunk)
    if not header:
        return None
    next_boundary = chunk.find("\nTask ", header.end())
    window = chunk[header.end(): next_boundary if next_boundary != -1 else len(chunk)]
    for m in BREAKDOWN_ROW_RE.finditer(window):
        if int(m.group(1)) == source_task:
            return dict(zip(LINEAGE_NAMES, (float(x) for x in m.groups()[1:])))
    return None


def parse_run(log_text, reference_task):
    final_task, chunk = _split_last_task(log_text)
    table_metrics = _final_task_table_metrics(chunk)
    prf1 = _final_task_macro_prf1(chunk)
    adv_ref_row = _breakdown_row(chunk, ADV_BREAKDOWN_HEADER_RE, reference_task)
    clean_ref_row = _breakdown_row(chunk, CLEAN_BREAKDOWN_HEADER_RE, reference_task)
    adv_final_row = _breakdown_row(chunk, ADV_BREAKDOWN_HEADER_RE, final_task)
    missing_clean_breakdown = clean_ref_row is None

    per_lineage = {}
    for name in LINEAGE_NAMES:
        m = table_metrics.get(name, {})
        task_acc = m.get("task_acc")
        adv_acc = m.get("adv_acc")
        if adv_acc is None and adv_final_row is not None:
            adv_acc = adv_final_row.get(name)  # clean/poisoned_baseline: only logged in the breakdown table
        final_combined = (task_acc + adv_acc) / 2 if task_acc is not None and adv_acc is not None else float("nan")

        if clean_ref_row is not None and adv_ref_row is not None:
            ref_combined = (clean_ref_row[name] + adv_ref_row[name]) / 2
        else:
            ref_combined = float("nan")

        p, r, f1 = prf1.get(name, (float("nan"),) * 3)
        per_lineage[name] = dict(
            mean_acc=m.get("mean_acc", float("nan")),
            pooled_acc=m.get("pooled_acc", float("nan")),
            ref_task_acc_combined=ref_combined,
            final_task_acc_combined=final_combined,
            precision=p, recall=r, f1=f1,
        )
    return final_task, per_lineage, missing_clean_breakdown


def _fmt(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="Paths to N pipeline_log.txt files")
    ap.add_argument("--reference-task", type=int, default=1,
                     help="Which early task's data to report the final classifier's combined "
                          "accuracy on (default: task 1)")
    args = ap.parse_args()

    missing_clean_breakdown = []
    per_lineage_runs = {name: {"mean_acc": [], "pooled_acc": [], "ref_task_acc_combined": [],
                                "final_task_acc_combined": [], "precision": [], "recall": [], "f1": []}
                         for name in LINEAGE_NAMES}
    final_tasks = set()

    for path in args.logs:
        with open(path) as f:
            log_text = f.read()
        final_task, per_lineage, missing_clean = parse_run(log_text, args.reference_task)
        final_tasks.add(final_task)
        if missing_clean:
            missing_clean_breakdown.append(path)
        for name in LINEAGE_NAMES:
            for k, v in per_lineage[name].items():
                per_lineage_runs[name][k].append(v)

    if len(final_tasks) > 1:
        print(f"WARNING: logs disagree on final task number: {sorted(final_tasks)} -- "
              f"comparing across runs at DIFFERENT final tasks.\n")
    if missing_clean_breakdown:
        print(f"WARNING: {len(missing_clean_breakdown)} log(s) predate the CLEAN test-set breakdown "
              f"table -- 'task {args.reference_task} acc @ final' is n/a for those runs:")
        for p in missing_clean_breakdown:
            print(f"  {p}")
        print()

    final_task_label = final_tasks.pop() if len(final_tasks) == 1 else "final"
    headers = [
        "method",
        f"mean acc @ t{final_task_label}",
        f"pooled acc @ t{final_task_label}",
        f"task {args.reference_task} acc @ t{final_task_label}",
        f"task {final_task_label} acc",
        "precision (macro)",
        "recall (macro)",
        "F1 (macro)",
    ]
    rows = []
    for name in LINEAGE_NAMES:
        d = per_lineage_runs[name]
        rows.append([
            name,
            _fmt(d["mean_acc"]),
            _fmt(d["pooled_acc"]),
            _fmt(d["ref_task_acc_combined"]),
            _fmt(d["final_task_acc_combined"]),
            _fmt(d["precision"]),
            _fmt(d["recall"]),
            _fmt(d["f1"]),
        ])

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(f"Aggregated over {len(args.logs)} run(s), final task = {final_task_label}, "
          f"reference task = {args.reference_task}\n")
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))


if __name__ == "__main__":
    main()
