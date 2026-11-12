"""
build_rebuttal_tables.py

Rebuttal table for the informed-DEDUCE comparison: reads the
logs/rebuttal_metrics.json written by

  (c) madar_pocket_pipeline.py        --dump_stream DIR   (L2U fixes + MADAR)
  (a) madar_pocket_pipeline_deduce.py --load_stream DIR   (original DEDUCE)
  (b) madar_pocket_pipeline_deduce.py --load_stream DIR --informed_forget

for each seed, and prints:
  * the same per-seed TABLE 1 rows and mean+-std TABLE 2 row that
    build_latex_tables.py prints (Mean Acc. / Task N / Task 1 / P / R / F1),
    using that script's own number formatting;
  * an extra per-seed table of |U_t| per task and the malicious-side attack
    success rate at the final task;
  * a CSV (one row per method x seed) with all of the above.

Column definitions match summarize_pipeline_runs.parse_run exactly:
  Mean Acc. -- final task's mean over all clean + adversarial test sets seen
  Task N    -- final task's (clean acc + adversarial acc) / 2 on its own test set
  Task 1    -- (clean + adversarial) / 2 on source task --reference-task
               (default 1, i.e. task index 1 -- same convention as the paper tables)
  P/R/F1    -- macro, final task's clean test set

Optional --ssf / --informed_ssf add the SSF rows; --deduce / --informed are
optional too. A clean-only Mean Acc (clean test sets only, no adversarial
sets) is printed and written to the CSV for every row.

Before printing anything it checks, per seed and per task, that the three
runs used the same poisoned rows and the same U_t (sha256 of the index
arrays, recorded independently by each run) and refuses to continue on a
mismatch.

Usage (paths use {seed} as a placeholder):
  python build_rebuttal_tables.py --seeds 130 131 132 \\
      --l2u      RUNS/madar_pocket_pipeline/rebuttal_l2u_s{seed}/logs/rebuttal_metrics.json \\
      --deduce   RUNS/madar_pocket_deduce/rebuttal_deduce_s{seed}/logs/rebuttal_metrics.json \\
      --informed RUNS/madar_pocket_deduce/rebuttal_informed_s{seed}/logs/rebuttal_metrics.json \\
      --out_csv  rebuttal_informed_deduce.csv
"""
from __future__ import annotations

import argparse
import csv
import json

from build_latex_tables import _pct, _pct_pm_std

# (row label, which run, lineage inside that run)
METHODS = [
    ("MADAR", "l2u", "poisoned_baseline"),
    ("Informed MADAR (detector-filtered)", "madar_filtered", "poisoned_baseline_filtered"),
    ("DEDUCE", "deduce", "deduce"),
    ("DEDUCE (informed)", "informed", "deduce"),
    ("L2U (Approximate Unlearning)", "l2u", "amnesiac"),
    ("L2U (Dropped Rows)", "l2u", "dropped_rows"),
    ("L2U (Flip Label)", "l2u", "opposite_class"),
    ("SSF", "ssf", "ssf"),
    ("SSF (informed)", "informed_ssf", "ssf"),
]
# run name -> (must have --informed_forget?)  None = not checked (l2u)
INFORMED_FLAG = {"l2u": None, "deduce": False, "informed": True, "ssf": False, "informed_ssf": True,
                 "madar_filtered": True}
HASH_KEYS = ["sha256_U_t", "sha256_poison_ben", "sha256_poison_mal", "sha256_X_train_poisoned"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _row(run, lineage, reference_task):
    tasks = run["tasks"]
    final = tasks[-1]
    m = final["lineages"][lineage]
    ref = str(reference_task)
    if ref in m["per_source_clean"] and ref in m["per_source_adv"]:
        ref_comb = (m["per_source_clean"][ref] + m["per_source_adv"][ref]) / 2
    else:
        ref_comb = float("nan")
    return {
        "final_task": final["task"],
        "mean_acc": m["mean_acc"], "final_task_acc_combined": m["final_task_acc_combined"],
        "ref_task_acc_combined": ref_comb, "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "asr_malicious": m["asr_malicious"],
        "asr_malicious_pockets": m["asr_malicious_pockets"],
        "U_t_sizes": [e["U_t_size"] for e in tasks],
        # Clean-only Mean Acc: mean over the CLEAN test sets of tasks 0..final
        # only (no adversarial sets) -- unaffected by the attack's surrogate.
        "clean_mean_acc": (sum(m["per_source_clean"].values()) / len(m["per_source_clean"])
                           if m["per_source_clean"] else float("nan")),
    }


def _check_same_stream(seed, runs):
    """Every run must record identical poison / U_t hashes, task by task."""
    ref_name, ref = "l2u", runs["l2u"]
    ref_by_task = {e["task"]: e.get("stream_check") for e in ref["tasks"]}
    for name, run in runs.items():
        if name == ref_name:
            continue
        if run["config"].get("val_fraction", 0):
            raise SystemExit(f"seed {seed}: {name} run used --val_fraction (training rows held out) -- "
                             f"not a final-evaluation run.")
        for e in run["tasks"]:
            a, b = ref_by_task.get(e["task"]), e.get("stream_check")
            if a is None or b is None:
                raise SystemExit(f"seed {seed} task {e['task']}: missing stream_check in {ref_name} or {name}")
            bad = [k for k in HASH_KEYS if a[k] != b[k]]
            if bad:
                raise SystemExit(f"seed {seed} task {e['task']}: {name} vs {ref_name} hash mismatch on {bad}")
        if len(run["tasks"]) != len(ref["tasks"]):
            raise SystemExit(f"seed {seed}: {name} has {len(run['tasks'])} tasks, {ref_name} has {len(ref['tasks'])}")
    n = len(ref["tasks"])
    print(f"seed {seed}: poison + U_t sha256 identical across {sorted(runs)} for all {n} poisoned task(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--l2u", required=True, help="rebuttal_metrics.json path template with {seed}")
    ap.add_argument("--deduce", default=None, help="original DEDUCE run (optional)")
    ap.add_argument("--informed", default=None, help="informed DEDUCE run (optional)")
    ap.add_argument("--ssf", default=None, help="original SSF run (optional)")
    ap.add_argument("--informed_ssf", default=None, help="informed SSF run (optional)")
    ap.add_argument("--madar_filtered", default=None,
                    help="informed MADAR run: madar_pocket_pipeline_baseline_filtered.py --load_stream "
                         "--filter_source detector (optional)")
    ap.add_argument("--no_stream_check", action="store_true",
                     help="For runs that did NOT use a shared stream (each crafted its own poison): skip "
                          "the poison/U_t hash comparison. Rows are then NOT on identical data -- the "
                          "output is labeled accordingly.")
    ap.add_argument("--reference-task", type=int, default=1)
    ap.add_argument("--out_csv", default="rebuttal_informed_deduce.csv")
    args = ap.parse_args()

    paths = {"l2u": args.l2u, "deduce": args.deduce, "informed": args.informed, "ssf": args.ssf,
             "informed_ssf": args.informed_ssf, "madar_filtered": args.madar_filtered}
    paths = {k: v for k, v in paths.items() if v}
    methods = [m for m in METHODS if m[1] in paths]
    if args.no_stream_check:
        print("NOTE: --no_stream_check -- runs are NOT hash-verified to share poison / U_t "
              "(independent-poison comparison).")
    rows = {label: [] for label, _, _ in methods}
    for seed in args.seeds:
        runs = {k: _load(v.format(seed=seed)) for k, v in paths.items()}
        if not args.no_stream_check:
            _check_same_stream(seed, runs)
        for name, run in runs.items():
            want = INFORMED_FLAG[name]
            if want is not None and bool(run["config"].get("informed_forget")) != want:
                raise SystemExit(f"seed {seed}: --{name} run has informed_forget="
                                 f"{run['config'].get('informed_forget')}, expected {want}")
        for label, run_name, lineage in methods:
            rows[label].append((seed, _row(runs[run_name], lineage, args.reference_task)))

    final_tasks = {r["final_task"] for v in rows.values() for _, r in v}
    task_n = str(final_tasks.pop() + 1) if len(final_tasks) == 1 else "N"
    if final_tasks:
        print("WARNING: runs disagree on the final task number")

    print(f"\n{'=' * 70}\nTABLE 1 rows (Mean Acc. / Task {task_n} / Task {args.reference_task} / P / R / F1)\n{'=' * 70}")
    for label, per_seed in rows.items():
        print(f"\\multirow{{{len(per_seed)}}}{{*}}{{{label}}}")
        for seed, r in per_seed:
            print(f"& {seed} & {_pct(r['mean_acc'])} & {_pct(r['final_task_acc_combined'])} & "
                  f"{_pct(r['ref_task_acc_combined'])} & {_pct(r['precision'])} & "
                  f"{_pct(r['recall'])} & {_pct(r['f1'])} \\\\")

    print(f"\n{'=' * 70}\nTABLE 2 rows (mean $\\pm$ std over {len(args.seeds)} seed(s))\n{'=' * 70}")
    for label, per_seed in rows.items():
        agg = {k: _pct_pm_std([r[k] for _, r in per_seed])
               for k in ["mean_acc", "final_task_acc_combined", "ref_task_acc_combined", "precision",
                         "recall", "f1"]}
        print(f"& {label}\n& {agg['mean_acc']} & {agg['final_task_acc_combined']} & "
              f"{agg['ref_task_acc_combined']} & {agg['precision']} & {agg['recall']} & {agg['f1']} \\\\")

    print(f"\n{'=' * 70}\nClean-only Mean Acc (clean test sets of all tasks; no adversarial sets), "
          f"mean $\\pm$ std\n{'=' * 70}")
    for label, per_seed in rows.items():
        print(f"{label:<30} {_pct_pm_std([r['clean_mean_acc'] for _, r in per_seed])}")

    print(f"\n{'=' * 70}\n|U_t| per task and final-task attack success rate (malicious -> benign)\n{'=' * 70}")
    for label, per_seed in rows.items():
        for seed, r in per_seed:
            print(f"{label:<30} seed {seed}: |U_t|={r['U_t_sizes']}  "
                  f"ASR(all mal adv)={_pct(r['asr_malicious'])}  "
                  f"ASR(genuine-pocket mal)={_pct(r['asr_malicious_pockets'])}")

    fields = ["method", "seed", "final_task", "mean_acc", f"task{task_n}_combined",
              f"task{args.reference_task}_combined", "precision", "recall", "f1",
              "asr_malicious", "asr_malicious_pockets", "clean_mean_acc", "U_t_sizes"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for label, per_seed in rows.items():
            for seed, r in per_seed:
                w.writerow([label, seed, r["final_task"], r["mean_acc"], r["final_task_acc_combined"],
                            r["ref_task_acc_combined"], r["precision"], r["recall"], r["f1"],
                            r["asr_malicious"], r["asr_malicious_pockets"], r["clean_mean_acc"],
                            ";".join("" if u is None else str(u) for u in r["U_t_sizes"])])
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
