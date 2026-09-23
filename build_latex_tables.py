"""
build_latex_tables.py

Fills in ONE method's rows in both of the paper's LaTeX tables (the
per-seed table and the mean+-std comparison table), from N seed runs of the
SAME method/config. Mirrors plot_pipeline_metrics.py / summarize_pipeline_runs.py's
usage: one method per invocation -- run it once per row-group you need,
then paste the printed LaTeX into the document by hand.

Reuses summarize_pipeline_runs.parse_run() directly (same "Task 1"/"Task N"/
mean-acc/precision/recall/f1 numbers that script already computes at the
final task), so a log file that works with summarize_pipeline_runs.py works
here unchanged, and a fix to that parsing logic does not need to be
duplicated in two places.

NOTE ON COLUMNS: the paper table's example has one combined "P/R/F1" column.
This script prints P, R and F1 as THREE SEPARATE columns instead (per your
choice) -- the LaTeX table's column spec needs 2 extra `c`s to match
(e.g. `llccccc` -> `llccccccc` for the per-seed table).

===========================================================================
HOW TO FILL IN THE TWO TABLES, ROW GROUP BY ROW GROUP
===========================================================================
Each row group ("Method" in Table 1, one method row in Table 2) is ONE
invocation of this script, naming --lineage (the column inside the log
files) and --method-name (the label to print). Several methods share the
same log files -- only --lineage changes:

  Joint                        --> madar_pocket_pipeline_naive_joint.py logs, --lineage joint
  Naive                        --> madar_pocket_pipeline_naive_joint.py logs, --lineage naive
  Synaptic Intelligence        --> madar_pocket_pipeline_si_agem.py logs,     --lineage si
  A-GEM                        --> madar_pocket_pipeline_si_agem.py logs,     --lineage agem
  DEDUCE                       --> madar_pocket_pipeline_deduce.py logs,      --lineage deduce
  SSF                          --> madar_pocket_pipeline_ssf.py logs,         --lineage ssf
  MADAR                        --> madar_pocket_pipeline.py logs,             --lineage poisoned_baseline
  MADAR + Unlearning (Dropped Rows)  --> madar_pocket_pipeline.py logs,       --lineage dropped_rows
  MADAR + Unlearning (Amnesiac)      --> madar_pocket_pipeline.py logs,       --lineage amnesiac
  MADAR + Unlearning (Flip Label)    --> madar_pocket_pipeline.py logs,       --lineage opposite_class
  MalCL                         --> not implemented (scoped out -- see madar_pocket_pipeline_deduce.py's
                                     module docstring for why); leave this row blank.
  Meta-Unlearning                --> not run yet; leave this row blank.

"MADAR" is this project's poisoned_baseline lineage (MADAR-style curated
replay buffer, no forget-set-targeted fix applied) -- not a separate file.

Table 2's "Group" column (Baselines / Prior Work / L2U) spans several
method rows under one \\multirow -- this script only ever knows about ONE
method per run, so it prints each method's row body only; add the
\\multirow{N}{*}{Group} wrapper yourself once you have all of a group's rows.

===========================================================================
USAGE
===========================================================================
  python build_latex_tables.py \\
      --logs /mnt/erivas6/runs/madar_pocket_pipeline/0908_piplai_pipeline/logs/pipeline_log.txt \\
             /mnt/erivas6/runs/madar_pocket_pipeline/0908_piplai_pipeline1/logs/pipeline_log.txt \\
             /mnt/erivas6/runs/madar_pocket_pipeline/0908_piplai_pipeline2/logs/pipeline_log.txt \\
      --seeds 130 131 132 \\
      --lineage poisoned_baseline \\
      --method-name "MADAR"

Prints two blocks to stdout: "TABLE 1 rows" (one row per seed, wrapped in
\\multirow{N}{*}{method-name}) and "TABLE 2 row" (one aggregated mean+-std
row). --seeds is optional; if omitted, seeds are labeled by position (1, 2,
3, ...) and a reminder is printed to fill them in by hand.
"""
from __future__ import annotations

import argparse

import summarize_pipeline_runs as summarize

METRICS = ["mean_acc", "final_task_acc_combined", "ref_task_acc_combined", "precision", "recall", "f1"]


def _pct(v):
    return "--" if v != v else f"{v * 100:.2f}\\%"  # v != v is the NaN check


def _pct_pm_std(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return "--"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean * 100:.2f}\\%"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = var ** 0.5
    return f"{mean * 100:.2f}\\% $\\pm$ {std:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="Paths to N seed pipeline_log.txt files")
    ap.add_argument("--seeds", nargs="+", default=None,
                     help="Seed label per log, same order as --logs. Defaults to 1, 2, 3, ...")
    ap.add_argument("--lineage", required=True,
                     help="Which lineage/column inside these logs to extract (e.g. clean, "
                          "poisoned_baseline, si, agem, deduce, naive, joint, dropped_rows, "
                          "amnesiac, opposite_class). See the module docstring for the full mapping.")
    ap.add_argument("--method-name", required=True,
                     help="Label for the LaTeX table's Method column, e.g. 'MADAR + Unlearning (Amnesiac)'")
    ap.add_argument("--reference-task", type=int, default=1,
                     help="Which early task's data becomes the 'Task 1' column (default: task 1, "
                          "matching summarize_pipeline_runs.py's default).")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else [str(i + 1) for i in range(len(args.logs))]
    if len(seeds) != len(args.logs):
        ap.error(f"--seeds has {len(seeds)} entries but --logs has {len(args.logs)}")

    per_seed = []  # one dict per log, from METRICS
    final_tasks = set()
    for path, seed in zip(args.logs, seeds):
        with open(path) as f:
            log_text = f.read()
        final_task, per_lineage, missing_clean, lineage_order = summarize.parse_run(log_text, args.reference_task)
        final_tasks.add(final_task)
        if args.lineage not in per_lineage:
            raise ValueError(
                f"lineage {args.lineage!r} not found in {path} (found: {lineage_order}). "
                f"Wrong --lineage name for this file?"
            )
        row = per_lineage[args.lineage]
        if missing_clean:
            print(f"NOTE: {path} predates the CLEAN test-set breakdown table -- "
                  f"its 'Task {args.reference_task}' column will be '--'.")
        per_seed.append(row)

    if len(final_tasks) > 1:
        print(f"WARNING: input logs disagree on final task number: {sorted(final_tasks)} -- "
              f"the 'Task N' column mixes runs of different length.\n")
    final_task_label = str(next(iter(final_tasks)) + 1) if len(final_tasks) == 1 else "N"

    print(f"{'=' * 70}\nTABLE 1 rows -- {args.method_name} "
          f"(Mean Acc. / Task {final_task_label} / Task {args.reference_task} / P / R / F1)\n{'=' * 70}")
    print(f"\\multirow{{{len(seeds)}}}{{*}}{{{args.method_name}}}")
    for seed, row in zip(seeds, per_seed):
        print(
            f"& {seed} & {_pct(row['mean_acc'])} & {_pct(row['final_task_acc_combined'])} & "
            f"{_pct(row['ref_task_acc_combined'])} & {_pct(row['precision'])} & "
            f"{_pct(row['recall'])} & {_pct(row['f1'])} \\\\"
        )

    print(f"\n{'=' * 70}\nTABLE 2 row -- {args.method_name} "
          f"(aggregated mean $\\pm$ std over {len(seeds)} seed(s))\n{'=' * 70}")
    agg = {m: _pct_pm_std([row[m] for row in per_seed]) for m in METRICS}
    print(f"& {args.method_name}\n& {agg['mean_acc']} & {agg['final_task_acc_combined']} & "
          f"{agg['ref_task_acc_combined']} & {agg['precision']} & {agg['recall']} & {agg['f1']} \\\\")


if __name__ == "__main__":
    main()
