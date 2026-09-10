"""
madar_pocket_pipeline_naive_joint.py

Same poisoning, test-time attack, and pocket-targeting criterion as
madar_pocket_pipeline.py -- imported directly from it below, not
reimplemented, so the two files cannot drift apart. Adds the two
continual-learning REFERENCE POINTS every other lineage in this project is
implicitly measured against, ported from EmiliaR8/Meta-Unlearning's
experiments/naive.py and experiments/joint.py:

FOUR lineages, all starting from the same task-0 model (see
madar_pocket_pipeline.py for task 0's plain-pretraining details):

  clean             -- never poisoned; reference. Own bounded, curated
                       replay buffer (same simplified design as
                       madar_pocket_pipeline_si_agem.py's clean).
  poisoned_baseline -- poisoned every task, never fixed. IDENTICAL to
                       madar_pocket_pipeline.py's poisoned_baseline.
  naive             -- THE LOWER BOUND. Poisoned every task (the SAME
                       crafted poison as poisoned_baseline), fine-tuned with
                       PLAIN cross-entropy on that task's data ONLY -- no
                       replay buffer, no regularization, no detection,
                       nothing. Whatever accuracy poisoned_baseline (or any
                       fix/baseline in the other files) retains above this
                       line is what its buffer or mechanism bought; without
                       this row that number was never measured, only
                       assumed. Note this project's other "no buffer"
                       baselines (si, agem, deduce) are NOT naive: each adds
                       its own mechanism (a penalty, gradient projection, or
                       detect/LUM/GUM) even though none uses a replay
                       buffer. naive adds nothing at all.
  joint             -- THE ORACLE CEILING. NEVER poisoned -- this is the
                       reference's own definition, not a "clean lineage
                       done again": at every task boundary the model is
                       DISCARDED and a freshly-initialized network is
                       trained from scratch on ALL clean data seen so far
                       (tasks 0..t, unbounded, no buffer curation), at the
                       same epoch/lr/batch budget as task 0's own plain
                       pretraining. This is NOT the same thing as this
                       project's `clean` lineage, which fine-tunes
                       (continues) the same weights every task and only
                       rehearses from a bounded, curated buffer -- `clean`
                       is what a realistic bounded-memory continual learner
                       achieves with zero poisoning; `joint` is what
                       unbounded memory and no continual-learning
                       constraint at all achieves. joint is still evaluated
                       against every task's genuine-pocket test attack (the
                       same attacks poisoned_baseline/naive face), even
                       though it never saw poisoned data during training --
                       that is the point of tracking it as a ceiling.

Because naive and joint are reference points rather than "fixes" for
poisoning, there is no detector and no pre/post-unlearning split here either
-- both are logged as one post-adaptation snapshot per task, in the SAME
table shape (task/pooled/mean/adv accuracy + still-evades %) every other
"under test" lineage in this project's pipelines uses, so the existing
(already-generalized) analysis scripts need only see the new lineage names.

A note on optimizers: naive trains via the SAME Adam optimizer
(AdaptableClassifier) every lineage in this project uses, not the
reference's SGD, matching madar_pocket_pipeline_si_agem.py's rationale.
joint necessarily does NOT reuse a persistent AdaptableClassifier across
tasks (there is no continuing optimizer state to persist when the weights
themselves are discarded every task) -- each task wraps a freshly-trained
network in a throwaway AdaptableClassifier purely for the shared
scoring/prediction interface the rest of the pipeline expects.

COST WARNING: joint's per-task training set is the FULL cumulative history
(tasks 0..t), retrained from scratch by epochs (not by a fixed iteration
budget), so its cost grows with both task index AND epoch count -- by the
final task it is training on the sum of every task's data seen so far, once
each. This is inherent to what "joint" measures, not a bug; expect this
lineage's step to dominate this file's runtime as NUM_TASKS grows.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import madar_pocket_pipeline as base

LINEAGE_NAMES = ["clean", "poisoned_baseline", "naive", "joint"]
BASELINE_NAMES = ["naive", "joint"]  # the two reference points; get the full metrics table


def joint_retrain(feature_dim, hidden_sizes, X_hist, y_hist, epochs, batch_size, lr):
    """Reference joint.py: discard the weights and retrain a FRESH network
    from scratch on ALL clean data seen so far, at task 0's own
    epoch/lr/batch budget (matching the reference, which also reuses its
    task-0 training routine for every joint task rather than a
    continual-phase iteration budget -- joint is not a continual method)."""
    model = base.ClassifierNN(feature_dim, 2, hidden_sizes=hidden_sizes).to(base.DEVICE)
    Xt = torch.as_tensor(np.asarray(X_hist), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y_hist), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return base.AdaptableClassifier(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_naive_joint_run")
    ap.add_argument("--h5-path", type=str, default=base.H5_DATASET_PATH)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--hidden_sizes", type=str,
                     default=",".join(str(h) for h in base.DEFAULT_HIDDEN_SIZES),
                     help="Comma-separated hidden-layer widths for ClassifierNN. "
                          "Same meaning/default as in madar_pocket_pipeline.py.")
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline.py. "
                          "Off by default.")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}.")
    args = ap.parse_args()
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_naive_joint", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (naive / joint reference points)\n"
            "=====================================================================\n"
            "4 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "naive (lower bound: poisoned, zero buffer, zero mechanism), joint\n"
            "(oracle ceiling: never poisoned, discarded and retrained from scratch\n"
            "on all clean history every task). Same poisoning/attack/pocket-targeting\n"
            "as madar_pocket_pipeline.py -- see that file for those mechanics.\n"
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
    joint_X_history, joint_y_history = [], []
    task_test_splits, task_test_gids = {}, {}
    results = []

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -base.FEATURE_CLIP,
                       base.FEATURE_CLIP).astype(np.float32)

    for t in range(base.NUM_TASKS):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        task = tasks[t]
        X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y_all = task["labels"].astype(np.int64)
        gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

        X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
            X_raw, y_all, gid_all, test_size=base.TASK_TEST_FRAC, random_state=args.seed,
            stratify=y_all,
        )

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # joint's task-0 state IS this shared pretraining -- "discard and
        # retrain from scratch on all history so far" is identical to plain
        # task-0 training when the history is only task 0.
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
            for _ in range(base.TASK0_EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, base.TASK0_BATCH_SIZE):
                    idx = perm[i:i + base.TASK0_BATCH_SIZE]
                    if len(idx) < 2:
                        continue
                    opt0.zero_grad()
                    loss = loss_fn0(base_model(Xt[idx]), yt[idx])
                    loss.backward()
                    opt0.step()
            base_model.eval()

            for name in LINEAGE_NAMES:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            joint_X_history.append(X_train_scaled)
            joint_y_history.append(y_train)

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            category = np.where(y_train == benign_label, "benign", "malicious_clean")
            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, gid_train,
                benign_label, mal_label,
            )
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )

            train_section = (
                f"malicious: {int((y_train == mal_label).sum())}, benign: {int((y_train == benign_label).sum())}\n"
                f"malicious_perturbed: 0, benign_perturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = (
                f"malicious: {int((y_test == mal_label).sum())}, benign: {int((y_test == benign_label).sum())}\n"
                f"genuine pockets: N/A (task 0 -- no poisoning yet)\n"
            )
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            cl_section = "N/A -- task 0 has no prior model to poison against."

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
                "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
                "baseline_label_buffers": baseline_label_buffers,
                "baseline_replay_buffer": baseline_replay_buffer,
                "joint_history_size": sum(len(y) for y in joint_y_history),
                "task_test_splits": task_test_splits, "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
                "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path)
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> attack, every task.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison, shared by poisoned_baseline/naive.
        X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = base.craft_task_poison(
            lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
        )
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix"),
        # identical to madar_pocket_pipeline.py.
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                            replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                            epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )

        # Step 5: naive adapts on the SAME poisoned data, plain CE, no
        # buffer, no replay, no regularization of any kind -- the lower
        # bound. Same epoch/batch budget as every other continual step in
        # this project (poisoned_baseline/si/agem/deduce), so the buffer is
        # the only thing that differs from poisoned_baseline.
        lineages["naive"].adapt(X_train_poisoned, y_train, epochs=base.ADAPT_EPOCHS)

        # Step 6: joint -- discard the weights, retrain a FRESH network from
        # scratch on ALL clean data seen so far (never poisoned).
        joint_X_history.append(X_train_scaled)
        joint_y_history.append(y_train)
        lineages["joint"] = joint_retrain(
            feature_dim, hidden_sizes, np.concatenate(joint_X_history), np.concatenate(joint_y_history),
            epochs=base.TASK0_EPOCHS, batch_size=base.TASK0_BATCH_SIZE, lr=base.TASK0_LR,
        )

        # Step 7: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean) -- same points re-scored under
        # every lineage below.
        eps_this_task = base.typical_class_gap(X_test_scaled, y_test, benign_label,
                                               mal_label) * base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, per_feature_epsilon=args.per_feature_epsilon,
        )

        # Step 8: spillover check -- re-attack every PRIOR task's test set.
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

        all_clean_sets = {s: (to_scaled(Xs_raw), ys) for s, (Xs_raw, ys) in task_test_splits.items()}
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pocket_info_by_source = {s: (v[2], v[3]) for s, v in historical_adv.items()}
        pocket_info_by_source[t] = (succ_pocket, eps_this_task)

        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = base.pooled_and_per_task_accuracy(
                lineages[name], all_test_sets_full)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test)

        still_evades = {}
        for name in BASELINE_NAMES:
            pred = lineages[name].predict(X_test_adv)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 9: update clean's and poisoned_baseline's buffers. naive and
        # joint have no buffer to update.
        category_all = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        category_all[idx_poison_ben] = "benign_perturbed"
        category_all[idx_poison_mal] = "malicious_perturbed"
        clean_category = np.where(y_train == benign_label, "benign", "malicious_clean")
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, gid_train,
            benign_label, mal_label,
        )
        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers, X_train_poisoned, y_train,
            category_all, gid_train, benign_label, mal_label,
        )

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
        adapt_section = "\n".join(adapt_lines)

        cl_lines = [
            f"naive: fine-tuned {base.ADAPT_EPOCHS} epoch(s) on this task's poisoned data only, "
            f"zero buffer, zero mechanism.",
            f"joint: retrained from scratch on {sum(len(y) for y in joint_y_history)} cumulative "
            f"CLEAN samples (tasks 0..{t}), {base.TASK0_EPOCHS} epoch(s).",
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
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-adaptation) classifier COMBINED (clean+adversarial, pooled) "
            f"accuracy on each source task's test-set:"
        )
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            Xs_clean, ys_clean = all_clean_sets[s]
            Xs_adv, ys_adv = all_test_sets_full[f"{s}_adversarial"]
            X_comb = np.vstack([Xs_clean, Xs_adv])
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb):>18.3f}"
            breakdown_lines.append(row)
        breakdown_section = "\n".join(breakdown_lines)

        base.write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Continual-learning baselines step", cl_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
        ])

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()),
            "joint_history_size": sum(len(y) for y in joint_y_history),
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
            "baseline_label_buffers": baseline_label_buffers,
            "baseline_replay_buffer": baseline_replay_buffer,
            "joint_history_size": sum(len(y) for y in joint_y_history),
            "task_test_splits": task_test_splits, "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
            "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path)

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")

        if t == base.NUM_TASKS - 1:
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled)
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels)

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`joint_X_history`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
