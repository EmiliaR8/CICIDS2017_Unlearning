"""
madar_pocket_pipeline_baseline_purge.py

Same poisoning, test-time attack, and pocket-targeting criterion as
madar_pocket_pipeline.py -- imported directly from it below, not
reimplemented, so the two files cannot drift apart. Tests ONE change to
poisoned_baseline: its replay buffer is ORACLE-PURGED of perturbed rows after
every fill, with NO refill.

THREE lineages, all starting from the same task-0 model:

  clean                    -- never poisoned; reference. Still required: it
                              crafts every task's poison (craft_task_poison
                              targets ITS decision boundary) and is the
                              "still correct" half of the genuine-pocket
                              attack criterion. Its replay buffer is filled
                              from its own clean data each task (same
                              simplified design as
                              madar_pocket_pipeline_si_agem.py's clean), so no
                              detector is needed anywhere in this file.
  poisoned_baseline        -- IDENTICAL to madar_pocket_pipeline.py's: poisoned
                              every task, never fixed, own UNFILTERED buffer.
  poisoned_baseline_purged -- identical to poisoned_baseline in every respect
                              (same poisoned data, same epochs/optimizer, same
                              IsolationForest buffer selection over its own
                              latent space) EXCEPT that right after each buffer
                              fill, every entry whose ORACLE category is
                              benign_perturbed / malicious_perturbed is
                              removed, and the removed slots are NOT
                              backfilled. The buffer is simply smaller from
                              then on, until ordinary admission next task
                              (IsolationForest over the surviving entries +
                              the new task's rows, re-selected up to budget)
                              grows it back -- after which that task's
                              perturbed rows are purged again. Same "purge,
                              no refill" semantics as madar_pocket_pipeline.py's
                              --joint_buffer_purge_after_fill ablation, but
                              with oracle ground truth instead of the detector,
                              so this is the upper bound for buffer cleaning.

Poison depends only on `clean` (its weights + this task's data), and clean
never touches either baseline's buffer -- so poisoned_baseline and
poisoned_baseline_purged see IDENTICAL poison every task, and any difference
between them is attributable to the purge alone.

No detector and no fix lineages (dropped_rows / amnesiac / opposite_class)
run here. Logging/checkpoints match madar_pocket_pipeline_meta_detect.py:
timestamped timing log (logs/meta_log.txt, also printed), adversarial
classification reports alongside clean ones, a per-task "Pocket recovery
summary (debug)" section (printed + logged, then a breakpoint() unless
--no_breakpoint), and per-task checkpoints (logs/classifier_checkpoint_task<t>.pt)
with buffers stored as ids only. Log table shapes match the other pocket
pipelines, so plot_pipeline_metrics.py / summarize_pipeline_runs.py /
build_latex_tables.py read this file's logs unchanged
(--lineage poisoned_baseline_purged).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import madar_pocket_pipeline as base

PURGED = "poisoned_baseline_purged"
LINEAGE_NAMES = ["clean", "poisoned_baseline", PURGED]
BASELINE_NAMES = [PURGED]  # the one under test; gets the full metrics table


def _buffer_ids_only(label_buffers):
    """Drops each replay-buffer entry's feature row (index 0), keeping only
    (label, category, sample_id) -- used ONLY when writing checkpoints, since
    MEM_SIZE-scale feature rows dominate checkpoint file size. Same helper as
    madar_pocket_pipeline_meta_detect.py's."""
    return {lbl: [tuple(e[1:]) for e in entries] for lbl, entries in label_buffers.items()}


def _replay_ids_only(replay_buffer):
    """Same as _buffer_ids_only, for the flattened list form."""
    return [tuple(e[1:]) for e in replay_buffer]


# ---------------------------------------------------------------------------
# Timestamped step-by-step timing log ("meta_log.txt") -- same mechanism and
# file name as madar_pocket_pipeline_meta_detect.py's.
# ---------------------------------------------------------------------------
_META_LOG_PATH = None
_META_LOG_T0 = None


def _tlog(msg):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    elapsed = time.perf_counter() - _META_LOG_T0 if _META_LOG_T0 is not None else 0.0
    line = f"[{now} | +{elapsed:8.1f}s] {msg}"
    print(line)
    if _META_LOG_PATH is not None:
        with open(_META_LOG_PATH, "a") as f:
            f.write(line + "\n")


def purge_perturbed(label_buffers):
    """Oracle purge: drops every buffer entry whose category (entry[2], set
    from oracle ground truth in main()) is benign_perturbed or
    malicious_perturbed. NO backfill -- removed slots stay empty. Returns
    (flattened replay list, {label: n_removed})."""
    removed = {}
    for lbl, entries in label_buffers.items():
        kept = [e for e in entries if not str(e[2]).endswith("_perturbed")]
        removed[lbl] = len(entries) - len(kept)
        label_buffers[lbl] = kept
    replay = []
    for buf in label_buffers.values():
        replay.extend(buf)
    return replay, removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _META_LOG_PATH, _META_LOG_T0
    start_time = time.perf_counter()
    _META_LOG_T0 = start_time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_baseline_purge_run")
    ap.add_argument("--h5-path", type=str, default=base.H5_DATASET_PATH)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--hidden_sizes", type=str,
                     default=",".join(str(h) for h in base.DEFAULT_HIDDEN_SIZES),
                     help="Comma-separated hidden-layer widths for ClassifierNN (all 3 lineages). "
                          "Same meaning/default as in madar_pocket_pipeline.py.")
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline.py. "
                          "Off by default.")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pauses: after every task's pocket "
                          "recovery summary, and at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}. Needed for a non-interactive/headless run.")
    args = ap.parse_args()
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_baseline_purge", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    meta_log_path = os.path.join(out_dir, "logs", "meta_log.txt")

    def checkpoint_path_for(task_id):
        return os.path.join(out_dir, "logs", f"classifier_checkpoint_task{task_id}.pt")

    _META_LOG_PATH = meta_log_path
    with open(meta_log_path, "w") as f:
        f.write(f"BASELINE-PURGE TIMING LOG -- run started {datetime.datetime.now().isoformat()}\n"
                f"args: {vars(args)}\n\n")
    _tlog("Run starting")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (poisoned_baseline oracle buffer purge)\n"
            "=====================================================================\n"
            "3 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "poisoned_baseline_purged (identical to poisoned_baseline, except its replay\n"
            "buffer is ORACLE-purged of every perturbed row after each fill, with NO\n"
            "refill). Same poisoning/attack/pocket-targeting as madar_pocket_pipeline.py --\n"
            "see that file for those mechanics. Both baselines see identical poison.\n"
            f"Classifier hidden layer sizes: {hidden_sizes}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
        )

    print(f"Loading {args.h5_path} and building {base.NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = base.load_pooled_chronological_tasks(
        args.h5_path, base.TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    scaler = None
    lineages = {}
    clean_label_buffers, clean_replay_buffer = {}, []
    baseline_label_buffers, baseline_replay_buffer = {}, []
    purged_label_buffers, purged_replay_buffer = {}, []
    task_test_splits, task_test_gids = {}, {}
    results = []

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -base.FEATURE_CLIP,
                       base.FEATURE_CLIP).astype(np.float32)

    def buffer_sizes(label_buffers):
        return {lbl: len(entries) for lbl, entries in sorted(label_buffers.items())}

    for t in range(base.NUM_TASKS):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        _tlog(f"=== Task {t}: start ===")
        task = tasks[t]
        X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y_all = task["labels"].astype(np.int64)
        gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

        X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
            X_raw, y_all, gid_all, test_size=base.TASK_TEST_FRAC, random_state=args.seed,
            stratify=y_all,
        )
        _tlog(f"Task {t}: train/test split done (train={len(y_train)}, test={len(y_test)})")

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = to_scaled(X_train_raw)
            X_test_scaled = to_scaled(X_test_raw)

            base_model = base.ClassifierNN(feature_dim, 2, hidden_sizes=hidden_sizes).to(base.DEVICE)
            Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
            yt = torch.tensor(y_train, dtype=torch.long)
            opt0 = torch.optim.Adam(base_model.parameters(), lr=base.TASK0_LR)
            loss_fn0 = nn.CrossEntropyLoss()
            base_model.train()
            n = len(Xt)
            _tlog(f"Task 0: pretraining for {base.TASK0_EPOCHS} epochs on {n} rows")
            for epoch in range(base.TASK0_EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, base.TASK0_BATCH_SIZE):
                    idx = perm[i:i + base.TASK0_BATCH_SIZE]
                    if len(idx) < 2:
                        continue
                    opt0.zero_grad()
                    loss = loss_fn0(base_model(Xt[idx]), yt[idx])
                    loss.backward()
                    opt0.step()
                _tlog(f"  Task 0: pretraining epoch {epoch + 1}/{base.TASK0_EPOCHS} done")
            base_model.eval()
            _tlog("Task 0: pretraining done")

            for name in LINEAGE_NAMES:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            # Task 0 has no poison, so the purge is a no-op here; all three
            # buffers start identical (same weights, same data).
            category = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, gid_train,
                benign_label, mal_label,
            )
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )
            purged_replay_buffer = base.update_shared_buffer(
                lineages[PURGED], purged_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )
            _tlog("Task 0: replay buffers filled")

            train_section = (
                f"malicious: {int((y_train == mal_label).sum())}, benign: {int((y_train == benign_label).sum())}\n"
                f"malicious_perturbed: 0, benign_perturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = (
                f"malicious: {int((y_test == mal_label).sum())}, benign: {int((y_test == benign_label).sum())}\n"
                f"genuine pockets: N/A (task 0 -- no poisoning yet)\n"
            )
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            cl_section = "N/A -- task 0 has no prior model to poison against (nothing to purge)."

            base.write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Continual-learning baselines step", cl_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
                "label_mapping": label_mapping,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                # Feature rows intentionally NOT persisted -- ids/labels/categories only.
                "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
                "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
                "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
                "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
                "purged_label_buffers": _buffer_ids_only(purged_label_buffers),
                "purged_replay_buffer": _replay_ids_only(purged_replay_buffer),
                "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
                "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path_for(t))
            _tlog(f"=== Task {t}: done (log + checkpoint written) ===")
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> adapt -> attack, every task.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        # Each buffer as it stood at the END of the previous task.
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)
        purged_replay_X, purged_replay_y = base.flatten_replay(purged_replay_buffer)
        purged_buffer_size_used = buffer_sizes(purged_label_buffers)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        _tlog(f"Task {t}: step 2 -- adapting clean lineage")
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison, shared by both baselines.
        _tlog(f"Task {t}: step 3 -- crafting poison (poison_fraction={poison_fraction})")
        X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = base.craft_task_poison(
            lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
        )
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])
        category_all = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        category_all[idx_poison_ben] = "benign_perturbed"
        category_all[idx_poison_mal] = "malicious_perturbed"
        _tlog(f"Task {t}: step 3 done ({len(poison_idx)} poisoned rows)")

        # Step 4: both baselines adapt on the SAME poisoned data, each with
        # its OWN buffer (purged's is the only thing that differs).
        _tlog(f"Task {t}: step 4 -- adapting poisoned_baseline")
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                            replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                            epochs=base.ADAPT_EPOCHS)
        _tlog(f"Task {t}: step 4 -- adapting {PURGED} (buffer used: {purged_buffer_size_used})")
        lineages[PURGED].adapt(X_train_poisoned, y_train,
                               replay_X=purged_replay_X, replay_y=purged_replay_y,
                               epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )
        acc_on_forced_labels_purged = (
            lineages[PURGED].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )
        _tlog(f"Task {t}: step 4 done")

        # Step 5: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean) -- same points re-scored under
        # every lineage below.
        _tlog(f"Task {t}: step 5 -- crafting this task's adversarial test attack")
        eps_this_task = base.typical_class_gap(X_test_scaled, y_test, benign_label,
                                               mal_label) * base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, per_feature_epsilon=args.per_feature_epsilon,
        )
        _tlog(f"Task {t}: step 5 done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())})")

        # Step 6: spillover check -- re-attack every PRIOR task's test set.
        _tlog(f"Task {t}: step 6 -- spillover re-attack over {len(task_test_splits)} prior task(s)")
        historical_adv = {}
        for s, (Xs_raw, ys) in task_test_splits.items():
            Xs_scaled = to_scaled(Xs_raw)
            eps_s = base.typical_class_gap(Xs_scaled, ys, benign_label, mal_label)
            eps_s = eps_s * base.ATTACK_EPS_MULTIPLIER if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = base.adversarial_attack_pocket(
                lineages["poisoned_baseline"], lineages["clean"], Xs_scaled, ys, epsilon_max=eps_s,
                per_feature_epsilon=args.per_feature_epsilon,
            )
            historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)
            _tlog(f"  Task {t}: step 6 -- re-attacked source task {s} ({len(ys)} rows)")
        _tlog(f"Task {t}: step 6 done")

        all_clean_sets = {s: (to_scaled(Xs_raw), ys) for s, (Xs_raw, ys) in task_test_splits.items()}
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pocket_info_by_source = {s: (v[2], v[3]) for s, v in historical_adv.items()}
        pocket_info_by_source[t] = (succ_pocket, eps_this_task)

        _tlog(f"Task {t}: computing pooled/mean/per-class accuracy for all lineages")
        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = base.pooled_and_per_task_accuracy(
                lineages[name], all_test_sets_full)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test)
            _tlog(f"  Task {t}: accuracy -- {name} done")

        still_evades = {}
        for name in BASELINE_NAMES:
            pred = lineages[name].predict(X_test_adv)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 7: update all three buffers, only now, after every lineage's
        # adaptation this task is done. Both baselines fill from the SAME
        # unfiltered poisoned batch via IsolationForest over their OWN latent
        # space; the purged one then has every oracle-perturbed entry removed,
        # with no backfill.
        _tlog(f"Task {t}: step 7 -- updating replay buffers")
        clean_category = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, gid_train,
            benign_label, mal_label,
        )
        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers, X_train_poisoned, y_train,
            category_all, gid_train, benign_label, mal_label,
        )
        purged_replay_buffer = base.update_shared_buffer(
            lineages[PURGED], purged_label_buffers, X_train_poisoned, y_train,
            category_all, gid_train, benign_label, mal_label,
        )
        purged_dist_before = base.buffer_distribution(purged_label_buffers)
        purged_replay_buffer, purge_counts = purge_perturbed(purged_label_buffers)
        _tlog(f"Task {t}: step 7 done (purged benign={purge_counts.get(benign_label, 0)}, "
              f"malicious={purge_counts.get(mal_label, 0)}; purged buffer now "
              f"{buffer_sizes(purged_label_buffers)})")

        task_test_splits[t] = (X_test_raw, y_test)
        task_test_gids[t] = gid_test

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        n_ben_train = int((y_train == benign_label).sum())
        n_mal_train = int((y_train == mal_label).sum())
        train_section = (
            f"malicious: {n_mal_train}, benign: {n_ben_train}\n"
            f"malicious_perturbed (oracle): {len(idx_poison_mal)}, "
            f"benign_perturbed (oracle): {len(idx_poison_ben)}\n"
            f"poison_fraction used: {poison_fraction}\n"
            f"poisoned_baseline accuracy on poisoned points' forced labels: {acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
            f"{PURGED} accuracy on poisoned points' forced labels: {acc_on_forced_labels_purged:.3f}\n"
        )

        n_ben_test = int((y_test == benign_label).sum())
        n_mal_test = int((y_test == mal_label).sum())
        succ_ben = int(succ_pocket[y_test == benign_label].sum())
        succ_mal = int(succ_pocket[y_test == mal_label].sum())
        test_section = (
            f"malicious: {n_mal_test}, benign: {n_ben_test}\n"
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})\n"
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        clean_dist = base.buffer_distribution(clean_label_buffers)
        baseline_dist = base.buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"clean's OWN replay buffer distribution (post-update, this task): {clean_dist}")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution (post-update, this task): "
                            f"{baseline_dist}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
            adapt_lines.append(f"[{name}] classification report (this task's adversarial test):")
            adapt_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        adapt_section = "\n".join(adapt_lines)

        cl_lines = [
            f"{PURGED} buffer used for THIS task's adaptation (end of previous task, post-purge): "
            f"{purged_buffer_size_used}",
            f"{PURGED} buffer after IsolationForest fill, BEFORE purge: {purged_dist_before}",
            f"oracle purge removed (NOT backfilled): benign={purge_counts.get(benign_label, 0)}, "
            f"malicious={purge_counts.get(mal_label, 0)}",
            f"{PURGED} buffer AFTER purge (used next task): "
            f"{base.buffer_distribution(purged_label_buffers)}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}",
        ]
        for name in BASELINE_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adv_acc_name = lineages[name].score(X_test_adv, y_test)
            cl_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        cl_lines.append("")
        for name in BASELINE_NAMES:
            cl_lines.append(f"[{name}] classification report (this task's clean test):")
            cl_lines.append(per_class_reports[name])
            cl_lines.append(f"[{name}] classification report (this task's adversarial test):")
            cl_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        cl_section = "\n".join(cl_lines)

        breakdown_lines = [
            "Genuine-pocket rate per source task's adv-test-set, attacked FRESH this task",
            "(against THIS task's poisoned_baseline/clean -- not the rate recorded when",
            "that set was first created at its own task):",
            "",
            f"{'source task':<12} {'n':>8} {'genuine pockets':>18} {'rate':>8} {'eps':>8}",
        ]
        for s in sorted(pocket_info_by_source.keys()):
            succ_s, eps_s = pocket_info_by_source[s]
            n_s = len(succ_s)
            breakdown_lines.append(
                f"{s:<12} {n_s:>8} {int(succ_s.sum()):>10}/{n_s:<7} {base._fmt_pct(succ_s.mean()):>8} {eps_s:>8.4f}"
            )
        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's adv-test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>26}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>26.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>26}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>26.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-adaptation) classifier COMBINED (clean+adversarial, pooled) "
            f"accuracy on each source task's test-set:"
        )
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>26}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            Xs_clean, ys_clean = all_clean_sets[s]
            Xs_adv, ys_adv = all_test_sets_full[f"{s}_adversarial"]
            X_comb = np.vstack([Xs_clean, Xs_adv])
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb):>26.3f}"
            breakdown_lines.append(row)
        breakdown_section = "\n".join(breakdown_lines)

        # ---------------------------------------------------------------
        # Debug: pocket send/recovery summary -- same block as
        # madar_pocket_pipeline_meta_detect.py's, for the purged baseline
        # (poisoned_baseline is left out: pockets are DEFINED as points it
        # gets wrong, so its own recovery is 0% by construction).
        # ---------------------------------------------------------------
        n_pocketed = int(succ_pocket.sum())
        pocket_lines = [
            f"Sent into pockets (genuine pockets found, this task's test set): "
            f"{n_pocketed}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})",
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}",
            "",
            f"Recovered after adaptation (of the {n_pocketed} pocketed points, no longer evading):",
        ]
        lineage_still_evading = {}
        for name in BASELINE_NAMES:
            pred_name = lineages[name].predict(X_test_adv)
            still_evading_mask = (pred_name != y_test) & succ_pocket
            lineage_still_evading[name] = still_evading_mask
            n_recovered = n_pocketed - int(still_evading_mask.sum())
            pct_recovered = base._fmt_pct(n_recovered / n_pocketed) if n_pocketed else "N/A"
            pocket_lines.append(f"  {name:<26}: {n_recovered}/{n_pocketed} recovered ({pct_recovered})")
        pocket_lines.append("")
        pocket_lines.append("How well each lineage under test reads NORMAL (non-attacked) traffic right now:")
        for name in BASELINE_NAMES:
            pred_clean = lineages[name].predict(X_test_scaled)
            catch_ben = (pred_clean[y_test == benign_label] == benign_label).mean() if n_ben_test else float("nan")
            catch_mal = (pred_clean[y_test == mal_label] == mal_label).mean() if n_mal_test else float("nan")
            pocket_lines.append(
                f"  {name:<26}: catches {base._fmt_pct(catch_ben)} of benign, "
                f"{base._fmt_pct(catch_mal)} of malicious"
            )
        pocket_lines.append("")
        pockets_to_check = [
            (s, historical_adv[s]) for s in sorted(historical_adv.keys()) if int(historical_adv[s][2].sum()) > 0
        ]
        if pockets_to_check:
            pocket_lines.append(
                "Checking back on earlier tasks' pockets (did adapting to THIS task "
                "accidentally reopen any of them?):"
            )
            for s, (Xs_adv, ys, succ_s, eps_s) in pockets_to_check:
                n_pocketed_s = int(succ_s.sum())
                pocket_lines.append(f"  Task {s}'s pockets ({n_pocketed_s} total):")
                for name in BASELINE_NAMES:
                    pred_s = lineages[name].predict(Xs_adv)
                    n_still_closed = int((succ_s & (pred_s == ys)).sum())
                    pocket_lines.append(
                        f"    {name:<26}: {n_still_closed}/{n_pocketed_s} still closed "
                        f"({base._fmt_pct(n_still_closed / n_pocketed_s)})"
                    )
        else:
            pocket_lines.append("No earlier tasks with pockets to check yet.")
        pocket_summary = "\n".join(pocket_lines)
        print(f"\n--- Task {t}: pocket recovery summary ---\n{pocket_summary}")
        _tlog(f"Task {t}: pocket recovery summary\n{pocket_summary}")

        _tlog(f"Task {t}: writing pipeline_log.txt")
        base.write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Continual-learning baselines step", cl_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
            ("Pocket recovery summary (debug)", pocket_summary),
        ])

        if not args.no_breakpoint:
            print(f"\n[breakpoint] Task {t}: pocket recovery summary above -- inspect `succ_pocket`, "
                  f"`lineage_still_evading`, `X_test_adv`, `y_test`, `lineages`. Continue with `c`.")
            breakpoint()

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()),
            "purge_removed": {int(k): int(v) for k, v in purge_counts.items()},
            "purged_buffer_size": {int(k): int(v) for k, v in buffer_sizes(purged_label_buffers).items()},
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            # Feature rows intentionally NOT persisted -- ids/labels/categories only.
            "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
            "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
            "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
            "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
            "purged_label_buffers": _buffer_ids_only(purged_label_buffers),
            "purged_replay_buffer": _replay_ids_only(purged_replay_buffer),
            "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
            "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path_for(t))
        _tlog(f"Task {t}: checkpoint saved")

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")
        _tlog(f"=== Task {t}: done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())}) ===")

        if t == base.NUM_TASKS - 1:
            _tlog(f"Task {t}: rendering final PCA correctness-grid plot")
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled)
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels)
            _tlog(f"Task {t}: plot saved")

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`purged_label_buffers`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    _tlog(f"Run done. Total runtime: {time.perf_counter() - start_time:.1f}s")
    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
