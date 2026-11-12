"""
madar_pocket_pipeline_baseline_filtered.py

Same poisoning, test-time attack, and pocket-targeting criterion as
madar_pocket_pipeline.py -- imported directly from it below, not
reimplemented. Runs ONE variant of poisoned_baseline, with the smallest
lineage set possible to keep runtime low:

  clean                      -- never poisoned; reference. Cannot be dropped:
                                it crafts every task's poison
                                (craft_task_poison targets ITS decision
                                boundary) and is the "still correct" half of
                                the genuine-pocket attack criterion. Its
                                replay buffer is filled from its own clean
                                data each task (same simplified design as
                                madar_pocket_pipeline_si_agem.py's clean), so
                                no detector runs anywhere in this file.
  poisoned_baseline_filtered -- poisoned_baseline with an ALWAYS-CLEAN replay
                                buffer. Trains on the full poisoned batch
                                every task exactly like poisoned_baseline
                                (same data, epochs, optimizer), but its
                                buffer is filled ONLY from rows that are
                                clean by ORACLE ground truth: every
                                benign_perturbed / malicious_perturbed row is
                                excluded BEFORE IsolationForest selection, so
                                the buffer still fills to full budget (unlike
                                madar_pocket_pipeline_baseline_purge.py's
                                purged variant, which removes perturbed rows
                                AFTER selection and does not refill). Same
                                idea as the three fix lineages' detector-clean
                                joint buffer in madar_pocket_pipeline.py, but
                                oracle-filtered, so it is truly always clean.

poisoned_baseline_filtered takes poisoned_baseline's ROLE here: it is the
model the genuine-pocket test attack is crafted against (reference = clean),
so its numbers are directly comparable to the poisoned_baseline row in the
other pocket pipelines' runs -- the only difference is the buffer. Because
pockets are defined as points the attacked model gets wrong, there is no
"Pocket recovery summary" section (its own recovery is 0% by construction).

Logging/checkpoints otherwise match madar_pocket_pipeline_meta_detect.py:
timestamped timing log (logs/meta_log.txt, also printed), adversarial
classification reports alongside clean ones, and per-task checkpoints
(logs/classifier_checkpoint_task<t>.pt) with buffers stored as ids only. Log
table shapes match the other pocket pipelines, so plot_pipeline_metrics.py /
summarize_pipeline_runs.py / build_latex_tables.py read this file's logs
unchanged (--lineage poisoned_baseline_filtered).

INFORMED MADAR (ICLR 2027 rebuttal; opt-in, defaults unchanged -- verified
bit-identical weights with the new flags off).
--load_stream DIR uses madar_pocket_pipeline.py --dump_stream's scaler, task-0
model, training rows, poison, U_t and adversarial test sets for --seed (sha256
checked; refuses a stream with fewer tasks than --num_tasks), so this lineage
is trained and scored on exactly the data L2U, DEDUCE and SSF were. Under the
stream the adversarial test sets are the producer's (crafted against the L2U
pipeline's poisoned_baseline), i.e. a transfer attack for this lineage, like
for every other method -- no longer white-box against itself.
--filter_source detector (requires --load_stream) replaces the ORACLE filter
with the shared detector's flagged set U_t (same detector, same 240-row
annotated set as L2U): U_t rows are removed from BOTH the training batch and
the replay-buffer candidate pool. No unlearning step. (--filter_source oracle,
the default, keeps the original behavior: train on the full poisoned batch,
oracle-filter the buffer only.) logs/rebuttal_metrics.json is written for
build_rebuttal_tables.py (--madar_filtered).
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
import rebuttal_stream as rs

FILTERED = "poisoned_baseline_filtered"
LINEAGE_NAMES = ["clean", FILTERED]


def _buffer_ids_only(label_buffers):
    """Drops each replay-buffer entry's feature row (index 0), keeping only
    (label, category, sample_id) -- used ONLY when writing checkpoints. Same
    helper as madar_pocket_pipeline_meta_detect.py's."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _META_LOG_PATH, _META_LOG_T0
    start_time = time.perf_counter()
    _META_LOG_T0 = start_time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_baseline_filtered_run")
    ap.add_argument("--h5-path", type=str, default=base.H5_DATASET_PATH)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--hidden_sizes", type=str,
                     default=",".join(str(h) for h in base.DEFAULT_HIDDEN_SIZES),
                     help="Comma-separated hidden-layer widths for ClassifierNN (both lineages). "
                          "Same meaning/default as in madar_pocket_pipeline.py.")
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline.py. "
                          "Off by default.")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}. Needed for a non-interactive/headless run.")
    # --- Rebuttal (informed-MADAR comparison) options; all default to the
    # original behavior. See rebuttal_stream.py. ---
    ap.add_argument("--load_stream", type=str, default=None,
                     help="Directory written by madar_pocket_pipeline.py --dump_stream: use its scaler, "
                          "task-0 model, training rows, poison, U_t and adversarial test sets for --seed "
                          "(sha256 checked). --h5-path/--poison_fraction/--per_feature_epsilon are then "
                          "ignored. Refuses a stream with fewer tasks than --num_tasks.")
    ap.add_argument("--filter_source", type=str, default="oracle", choices=["oracle", "detector"],
                     help="'oracle' (default, original): train on the full poisoned batch, filter the "
                          "buffer with ground-truth poison labels. 'detector' (requires --load_stream): "
                          "remove the shared detector's flagged rows U_t from BOTH the training batch and "
                          "the buffer candidates (informed MADAR; no unlearning).")
    ap.add_argument("--num_tasks", type=int, default=base.NUM_TASKS,
                     help=f"Run this many tasks (default {base.NUM_TASKS}).")
    ap.add_argument("--runs_dir", type=str, default=base.RUNS_BASE_DIR,
                     help=f"Root output directory (default {base.RUNS_BASE_DIR}).")
    args = ap.parse_args()
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))
    if not 1 <= args.num_tasks <= base.NUM_TASKS:
        ap.error(f"--num_tasks must be in [1, {base.NUM_TASKS}]")
    if args.filter_source == "detector" and not args.load_stream:
        ap.error("--filter_source detector requires --load_stream (U_t comes from the shared stream)")
    reader = None
    if args.load_stream:
        reader = rs.StreamReader(args.load_stream, args.seed)
        print(f"Shared stream {reader.dir}: {reader.num_tasks} task(s) present, {args.num_tasks} requested")
        if reader.num_tasks < args.num_tasks:
            ap.error(f"stream {reader.dir} has only {reader.num_tasks} task(s) but --num_tasks is "
                     f"{args.num_tasks} -- the producer run did not finish. Re-run madar_pocket_pipeline.py "
                     f"--dump_stream for --seed {args.seed}, or pass --num_tasks {reader.num_tasks}.")
        if tuple(reader.meta["hidden_sizes"]) != hidden_sizes:
            ap.error(f"--hidden_sizes {hidden_sizes} != stream's {tuple(reader.meta['hidden_sizes'])}")
        args.poison_fraction = reader.meta["poison_fraction"]
        args.per_feature_epsilon = reader.meta["per_feature_epsilon"]

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(args.runs_dir, "madar_pocket_baseline_filtered", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    meta_log_path = os.path.join(out_dir, "logs", "meta_log.txt")

    def checkpoint_path_for(task_id):
        return os.path.join(out_dir, "logs", f"classifier_checkpoint_task{task_id}.pt")

    _META_LOG_PATH = meta_log_path
    with open(meta_log_path, "w") as f:
        f.write(f"BASELINE-FILTERED TIMING LOG -- run started {datetime.datetime.now().isoformat()}\n"
                f"args: {vars(args)}\n\n")
    _tlog("Run starting")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (poisoned_baseline with oracle-filtered buffer)\n"
            "=====================================================================\n"
            "2 lineages per task: clean (reference), poisoned_baseline_filtered\n"
            "(poisoned_baseline, but its replay buffer is filled ONLY from oracle-clean\n"
            "rows -- perturbed rows excluded before IsolationForest selection, buffer\n"
            "fills to full budget). poisoned_baseline_filtered is the model the pocket\n"
            "attack targets. Same poisoning/attack as madar_pocket_pipeline.py.\n"
            f"Classifier hidden layer sizes: {hidden_sizes}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
            + (f"filter_source={args.filter_source}, load_stream={args.load_stream}, "
               f"num_tasks={args.num_tasks}\n" if (reader is not None or args.num_tasks != base.NUM_TASKS) else "")
        )

    if reader is None:
        print(f"Loading {args.h5_path} and building {base.NUM_TASKS} pooled chronological tasks...")
        tasks, day_mapping, label_mapping = base.load_pooled_chronological_tasks(
            args.h5_path, base.TASK_FRACTIONS)
        benign_label = label_mapping["Benign"]
        mal_label = 1 - benign_label
        feature_dim = tasks[0]["features"].shape[1]
        print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
              f"task sizes={[len(t['labels']) for t in tasks]}")

        task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])
    else:
        benign_label, mal_label = int(reader.meta["benign_label"]), int(reader.meta["mal_label"])
        label_mapping = {"Benign": benign_label}
        feature_dim = int(reader.meta["feature_dim"])

    metrics_log = rs.MetricsLog(os.path.join(out_dir, "logs", "rebuttal_metrics.json"), config={
        "pipeline": "madar_pocket_pipeline_baseline_filtered.py", "seed": args.seed,
        "filter_source": args.filter_source, "informed_forget": args.filter_source == "detector",
        "load_stream": args.load_stream, "num_tasks": args.num_tasks,
        "stream_meta": reader.meta if reader is not None else None,
    })

    scaler = None
    lineages = {}
    clean_label_buffers, clean_replay_buffer = {}, []
    filtered_label_buffers, filtered_replay_buffer = {}, []
    task_test_splits, task_test_gids = {}, {}
    results = []

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -base.FEATURE_CLIP,
                       base.FEATURE_CLIP).astype(np.float32)

    for t in range(args.num_tasks):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        _tlog(f"=== Task {t}: start ===")
        if reader is None:
            task = tasks[t]
            X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
            y_all = task["labels"].astype(np.int64)
            gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

            X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
                X_raw, y_all, gid_all, test_size=base.TASK_TEST_FRAC, random_state=args.seed,
                stratify=y_all,
            )
        else:
            stream_task, _stream_hashes = reader.load_task(t)
            y_train, gid_train = stream_task["y_train"], stream_task["gid_train"]
            X_test_raw, y_test, gid_test = (stream_task["X_test_raw"], stream_task["y_test"],
                                            stream_task["gid_test"])
            print(f"  stream task {t}: {len(y_train)} train rows, {len(y_test)} test rows, "
                  f"sha256 verified for {len(_stream_hashes)} arrays")
        _tlog(f"Task {t}: train/test split done (train={len(y_train)}, test={len(y_test)})")

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            if reader is None:
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
            else:
                # Same scaler and task-0 model as the producer run.
                extras = reader.load_task0_extras()
                scaler = extras["scaler"]
                X_train_scaled = stream_task["X_train_scaled"]
                X_test_scaled = to_scaled(X_test_raw)
                base_model = base.ClassifierNN(feature_dim, 2, hidden_sizes=hidden_sizes).to(base.DEVICE)
                base_model.load_state_dict(extras["model_state_dict"])
                base_model.eval()
                _tlog("Task 0: loaded shared ClassifierNN + scaler from stream")

            for name in LINEAGE_NAMES:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            # Task 0 has no poison, so the filter is a no-op here.
            category = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, gid_train,
                benign_label, mal_label,
            )
            filtered_replay_buffer = base.update_shared_buffer(
                lineages[FILTERED], filtered_label_buffers, X_train_scaled, y_train,
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
            filter_section = "N/A -- task 0 has no prior model to poison against (nothing to filter)."

            base.write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Buffer filtering step", filter_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
                "label_mapping": label_mapping,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                # Feature rows intentionally NOT persisted -- ids/labels/categories only.
                "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
                "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
                "filtered_label_buffers": _buffer_ids_only(filtered_label_buffers),
                "filtered_replay_buffer": _replay_ids_only(filtered_replay_buffer),
                "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
                "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path_for(t))
            _tlog(f"=== Task {t}: done (log + checkpoint written) ===")
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> adapt -> attack, every task.
        # -------------------------------------------------------------
        if reader is None:
            X_train_scaled = to_scaled(X_train_raw)
        else:
            X_train_scaled = stream_task["X_train_scaled"]
        X_test_scaled = to_scaled(X_test_raw)
        # Each buffer as it stood at the END of the previous task.
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        filtered_replay_X, filtered_replay_y = base.flatten_replay(filtered_replay_buffer)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        _tlog(f"Task {t}: step 2 -- adapting clean lineage")
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison.
        _tlog(f"Task {t}: step 3 -- crafting poison (poison_fraction={poison_fraction})")
        if reader is None:
            X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = base.craft_task_poison(
                lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
            )
        else:
            # Shared poison from the producer run (not re-crafted here).
            X_train_poisoned = stream_task["X_train_poisoned"]
            idx_poison_ben = stream_task["idx_poison_ben"]
            idx_poison_mal = stream_task["idx_poison_mal"]
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])
        category_all = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        category_all[idx_poison_ben] = "benign_perturbed"
        category_all[idx_poison_mal] = "malicious_perturbed"
        _tlog(f"Task {t}: step 3 done ({len(poison_idx)} poisoned rows)")

        # Rows kept by the filter: oracle-clean rows (original), or -- with
        # --filter_source detector -- every row NOT in the shared detector's U_t.
        U_t = stream_task["detected_poison_idx"] if reader is not None else None
        if args.filter_source == "detector":
            keep = np.ones(len(y_train), dtype=bool)
            keep[U_t] = False
        else:
            keep = np.ones(len(y_train), dtype=bool)
            keep[poison_idx] = False

        # Step 4: the filtered baseline adapts on the FULL poisoned batch --
        # exactly like poisoned_baseline -- replaying its always-clean buffer.
        # Informed MADAR (--filter_source detector): on the U_t-filtered batch.
        _tlog(f"Task {t}: step 4 -- adapting {FILTERED}")
        if args.filter_source == "detector":
            lineages[FILTERED].adapt(X_train_poisoned[keep], y_train[keep],
                                     replay_X=filtered_replay_X, replay_y=filtered_replay_y,
                                     epochs=base.ADAPT_EPOCHS)
        else:
            lineages[FILTERED].adapt(X_train_poisoned, y_train,
                                     replay_X=filtered_replay_X, replay_y=filtered_replay_y,
                                     epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages[FILTERED].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )
        _tlog(f"Task {t}: step 4 done")

        if reader is None:
            # Step 5: craft this task's genuine-pocket test attack against the
            # filtered baseline (poisoned_baseline's role in the other pipelines),
            # reference = clean.
            _tlog(f"Task {t}: step 5 -- crafting this task's adversarial test attack")
            eps_this_task = base.typical_class_gap(X_test_scaled, y_test, benign_label,
                                                   mal_label) * base.ATTACK_EPS_MULTIPLIER
            X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
                lineages[FILTERED], lineages["clean"], X_test_scaled, y_test,
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
                    lineages[FILTERED], lineages["clean"], Xs_scaled, ys, epsilon_max=eps_s,
                    per_feature_epsilon=args.per_feature_epsilon,
                )
                historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)
                _tlog(f"  Task {t}: step 6 -- re-attacked source task {s} ({len(ys)} rows)")
            _tlog(f"Task {t}: step 6 done")
        else:
            # Steps 5-6 from the stream: the producer's adversarial test sets
            # (crafted against the L2U pipeline's poisoned_baseline).
            historical_adv, X_test_adv, succ_pocket, norms_pocket, eps_this_task = \
                rs.unpack_eval_sets(stream_task)
            _tlog(f"Task {t}: steps 5-6 loaded from stream ({len(historical_adv)} prior adversarial sets)")

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

        # Step 7: update both buffers, only now, after adaptation. The filtered
        # buffer's candidate pool is ONLY this task's oracle-clean rows (the
        # poisoned rows are excluded BEFORE IsolationForest selection), so it
        # still fills to full budget from clean data.
        _tlog(f"Task {t}: step 7 -- updating replay buffers")
        clean_category = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, gid_train,
            benign_label, mal_label,
        )
        filtered_replay_buffer = base.update_shared_buffer(
            lineages[FILTERED], filtered_label_buffers,
            X_train_poisoned[keep], y_train[keep],
            category_all[keep], gid_train[keep],
            benign_label, mal_label,
        )
        n_excluded = {benign_label: int((~keep & (y_train == benign_label)).sum()),
                      mal_label: int((~keep & (y_train == mal_label)).sum())}
        filtered_dist = base.buffer_distribution(filtered_label_buffers)
        n_perturbed_in_buffer = sum(
            1 for entries in filtered_label_buffers.values() for e in entries if str(e[2]).endswith("_perturbed")
        )
        _tlog(f"Task {t}: step 7 done (excluded benign={n_excluded[benign_label]}, "
              f"malicious={n_excluded[mal_label]} before selection; filtered buffer {filtered_dist})")

        task_test_splits[t] = (X_test_raw, y_test)
        task_test_gids[t] = gid_test

        # Rebuttal metrics (read-only).
        metrics_log.add_task({
            "task": t,
            "U_t_size": int(len(U_t)) if U_t is not None else None,
            "oracle_poison_size": int(len(poison_idx)),
            "U_t_true_positives": int(len(np.intersect1d(U_t, poison_idx))) if U_t is not None else None,
            "filter_source": args.filter_source,
            "rows_filtered": int((~keep).sum()),
            "poisoned_rows_trained_on": int(len(poison_idx)) if args.filter_source == "oracle"
                                        else int(keep[poison_idx].sum()),
            "poisoned_rows_in_buffer": int(n_perturbed_in_buffer),
            "stream_check": ({
                "sha256_U_t": rs.hash_array(U_t),
                "sha256_poison_ben": rs.hash_array(idx_poison_ben),
                "sha256_poison_mal": rs.hash_array(idx_poison_mal),
                "sha256_X_train_poisoned": rs.hash_array(X_train_poisoned),
            } if reader is not None else None),
            "lineages": {name: rs.lineage_metrics(
                lineages[name], t, all_clean_sets, all_test_sets_full, X_test_scaled, y_test,
                X_test_adv, succ_pocket, benign_label, mal_label,
                mean_results[name], pooled_results[name]) for name in LINEAGE_NAMES},
        })

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
            f"{FILTERED} accuracy on poisoned points' forced labels: {acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
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
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in LINEAGE_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"clean's OWN replay buffer distribution (post-update, this task): {clean_dist}")
        adapt_lines.append(f"{FILTERED}'s OWN replay buffer distribution (post-update, this task): "
                            f"{filtered_dist}")
        adapt_lines.append("")
        for name in LINEAGE_NAMES:
            adapt_lines.append(f"[{name}] classification report (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
            adapt_lines.append(f"[{name}] classification report (this task's adversarial test):")
            adapt_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        adapt_section = "\n".join(adapt_lines)

        if args.filter_source == "detector":
            filter_section = (
                f"DETECTOR-flagged rows (shared U_t, |U_t|={len(U_t)}) EXCLUDED from {FILTERED}'s TRAINING "
                f"batch AND buffer candidate pool: benign={n_excluded[benign_label]}, "
                f"malicious={n_excluded[mal_label]}; {int(keep[poison_idx].sum())} oracle-poisoned rows "
                f"missed by the detector were trained on\n"
                f"{FILTERED} buffer after fill (used next task): {filtered_dist}\n"
                f"perturbed entries in {FILTERED}'s buffer: {n_perturbed_in_buffer} (detector misses)"
            )
        else:
            filter_section = (
                f"oracle-poisoned rows EXCLUDED from {FILTERED}'s buffer candidate pool (before "
                f"IsolationForest selection): benign={n_excluded[benign_label]}, "
                f"malicious={n_excluded[mal_label]}\n"
                f"{FILTERED} buffer after fill (used next task): {filtered_dist}\n"
                f"perturbed entries in {FILTERED}'s buffer: {n_perturbed_in_buffer} (should always be 0)"
            )

        breakdown_lines = [
            "Genuine-pocket rate per source task's adv-test-set, attacked FRESH this task",
            f"(against THIS task's {FILTERED}/clean -- not the rate recorded when",
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
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>28}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>28.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>28}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>28.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-adaptation) classifier COMBINED (clean+adversarial, pooled) "
            f"accuracy on each source task's test-set:"
        )
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>28}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            Xs_clean, ys_clean = all_clean_sets[s]
            Xs_adv, ys_adv = all_test_sets_full[f"{s}_adversarial"]
            X_comb = np.vstack([Xs_clean, Xs_adv])
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb):>28.3f}"
            breakdown_lines.append(row)
        breakdown_section = "\n".join(breakdown_lines)

        _tlog(f"Task {t}: writing pipeline_log.txt")
        base.write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Buffer filtering step", filter_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
        ])

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()),
            "filter_excluded": {int(k): int(v) for k, v in n_excluded.items()},
            "filtered_buffer_size": {int(k): len(v) for k, v in filtered_label_buffers.items()},
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            # Feature rows intentionally NOT persisted -- ids/labels/categories only.
            "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
            "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
            "filtered_label_buffers": _buffer_ids_only(filtered_label_buffers),
            "filtered_replay_buffer": _replay_ids_only(filtered_replay_buffer),
            "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
            "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path_for(t))
        _tlog(f"Task {t}: checkpoint saved")

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")
        _tlog(f"=== Task {t}: done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())}) ===")

        if t == args.num_tasks - 1:
            _tlog(f"Task {t}: rendering final PCA correctness-grid plot")
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled)
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels)
            _tlog(f"Task {t}: plot saved")

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`filtered_label_buffers`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    _tlog(f"Run done. Total runtime: {time.perf_counter() - start_time:.1f}s")
    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
