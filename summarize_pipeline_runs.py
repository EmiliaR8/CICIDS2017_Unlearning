"""
summarize_pipeline_runs.py

Aggregates N runs of any of this project's "pocket" pipelines' (e.g.
madar_pocket_pipeline.py's 5 lineages, or madar_pocket_pipeline_si_agem.py's
4 -- any pipeline following the same log shape) pipeline_log.txt files (e.g.
the same config run under different seeds) into one mean +/- std table per
lineage, at the FINAL task of each log:

  method | mean acc @ task N | pooled acc @ task N | task R acc @ task N |
  task N acc | precision (macro) | recall (macro) | F1 (macro)

N = each log's own final task; R = --reference-task (default 1).

Lineage names are auto-detected per log from its own tables -- not
hardcoded -- so old and new logs both work, and the table's rows are the
union of every lineage seen across all input logs (a lineage missing from
some logs just gets fewer samples in its mean +/- std, same as any other
missing-data cell).

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

NUM = r"(?:[\d.]+|nan)"  # tolerates the literal "nan" the log prints for undefined still-evades %

# Lineage names are captured generically -- see plot_pipeline_metrics.py for
# the rationale. NAME requires a leading letter/underscore (never a bare
# digit): the per-source-task breakdown tables have rows shaped "<source
# task index> <NUM> <NUM> ...", and a pipeline with exactly 3 lineages
# produces exactly 3 numeric columns there -- indistinguishable from an
# Adaptation-step row's shape if the name could be a plain integer.
NAME = r"[A-Za-z_]\w*"
TASK_HEADER_RE = re.compile(r"^=+\n=== Task (\d+) ===\n=+\n", re.MULTILINE)
ADAPT_TABLE_RE = re.compile(rf"^({NAME})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", re.MULTILINE)
POST_FIX_RE = re.compile(
    rf"^({NAME})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})%\s*$",
    re.MULTILINE,
)
CLASS_MARKER_RE = re.compile(rf"\[({NAME})\] classification report")
MACRO_ROW_RE = re.compile(rf"^\s*macro avg\s+({NUM})\s+({NUM})\s+({NUM})\s+\d+\s*$", re.MULTILINE)
# The parenthesized phrase varies by pipeline ("post-unlearning" vs "post-adaptation").
ADV_BREAKDOWN_HEADER_RE = re.compile(
    r"Task \d+'s \([\w -]+\) classifier accuracy on each source task's adv-test-set:"
)
CLEAN_BREAKDOWN_HEADER_RE = re.compile(
    r"Task \d+'s \([\w -]+\) classifier accuracy on each source task's CLEAN test-set:"
)


def _split_last_task(log_text):
    headers = list(TASK_HEADER_RE.finditer(log_text))
    if not headers:
        raise ValueError("No '=== Task N ===' sections found -- is this a pipeline_log.txt?")
    last = headers[-1]
    return int(last.group(1)), log_text[last.end():]


def _final_task_table_metrics(chunk):
    """{lineage: {task_acc, pooled_acc, mean_acc, adv_acc}} from the final
    task's Adaptation-step (reference lineages, no adv_acc there) and
    post-fix/post-adaptation (under-test lineages, has adv_acc) tables."""
    out = {}
    for m in ADAPT_TABLE_RE.finditer(chunk):
        name = m.group(1)
        task_acc, pooled_acc, mean_acc = (float(x) for x in m.groups()[1:])
        out[name] = dict(task_acc=task_acc, pooled_acc=pooled_acc, mean_acc=mean_acc, adv_acc=None)
    for m in POST_FIX_RE.finditer(chunk):
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
    table (adv or clean). Column names come from that table's own header
    line, not a hardcoded list. Window is bounded to end right before the
    NEXT "Task N's..." header (the other breakdown table), since both
    tables share an identical row format and would otherwise bleed into
    each other whenever `source_task` isn't present in the first table."""
    header = header_re.search(chunk)
    if not header:
        return None
    header_line_start = chunk.find("\n", header.end()) + 1
    if header_line_start == 0:
        return None
    header_line_end = chunk.find("\n", header_line_start)
    header_line = chunk[header_line_start: header_line_end if header_line_end != -1 else len(chunk)]
    tokens = header_line.split()
    if len(tokens) < 3 or tokens[0] != "source" or tokens[1] != "task":
        return None
    names = tokens[2:]

    row_re = re.compile(rf"^(\d+)\s+" + r"\s+".join(f"({NUM})" for _ in names) + r"\s*$", re.MULTILINE)
    body_start = header_line_end + 1 if header_line_end != -1 else len(chunk)
    next_boundary = chunk.find("\nTask ", body_start)
    window = chunk[body_start: next_boundary if next_boundary != -1 else len(chunk)]
    for m in row_re.finditer(window):
        if int(m.group(1)) == source_task:
            return dict(zip(names, (float(x) for x in m.groups()[1:])))
    return None


def parse_run(log_text, reference_task):
    """Returns (final_task, per_lineage, missing_clean_breakdown, lineage_order)
    where lineage_order lists this run's own lineages in first-seen order."""
    final_task, chunk = _split_last_task(log_text)
    table_metrics = _final_task_table_metrics(chunk)
    prf1 = _final_task_macro_prf1(chunk)
    adv_ref_row = _breakdown_row(chunk, ADV_BREAKDOWN_HEADER_RE, reference_task)
    clean_ref_row = _breakdown_row(chunk, CLEAN_BREAKDOWN_HEADER_RE, reference_task)
    adv_final_row = _breakdown_row(chunk, ADV_BREAKDOWN_HEADER_RE, final_task)
    missing_clean_breakdown = clean_ref_row is None

    lineage_order = []
    for names in (table_metrics.keys(), prf1.keys(),
                  (adv_ref_row or {}).keys(), (adv_final_row or {}).keys()):
        for name in names:
            if name not in lineage_order:
                lineage_order.append(name)

    per_lineage = {}
    for name in lineage_order:
        m = table_metrics.get(name, {})
        task_acc = m.get("task_acc")
        adv_acc = m.get("adv_acc")
        if adv_acc is None and adv_final_row is not None:
            adv_acc = adv_final_row.get(name)  # reference lineages: only logged in the breakdown table
        final_combined = (task_acc + adv_acc) / 2 if task_acc is not None and adv_acc is not None else float("nan")

        if clean_ref_row is not None and adv_ref_row is not None and name in clean_ref_row and name in adv_ref_row:
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
    return final_task, per_lineage, missing_clean_breakdown, lineage_order


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
    final_tasks = set()
    lineage_order = []
    per_lineage_runs = {}

    def _ensure(name):
        if name not in per_lineage_runs:
            lineage_order.append(name)
            per_lineage_runs[name] = {"mean_acc": [], "pooled_acc": [], "ref_task_acc_combined": [],
                                       "final_task_acc_combined": [], "precision": [], "recall": [], "f1": []}

    for path in args.logs:
        with open(path) as f:
            log_text = f.read()
        final_task, per_lineage, missing_clean, order = parse_run(log_text, args.reference_task)
        final_tasks.add(final_task)
        if missing_clean:
            missing_clean_breakdown.append(path)
        for name in order:
            _ensure(name)
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
    for name in lineage_order:
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
