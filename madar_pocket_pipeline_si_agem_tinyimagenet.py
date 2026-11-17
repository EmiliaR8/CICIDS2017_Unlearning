"""
madar_pocket_pipeline_si_agem_tinyimagenet.py

Image/class-incremental port of madar_pocket_pipeline_si_agem.py -- same
relationship as madar_pocket_pipeline_tinyimagenet.py is to
madar_pocket_pipeline.py: same poisoning, test-time attack, and
pocket-targeting criterion, imported directly from
madar_pocket_pipeline_tinyimagenet below rather than reimplemented, so the
two files cannot drift apart. Compares two GENERAL continual-learning
baselines that were never designed around a forget set at all -- Synaptic
Intelligence (Zenke et al., 2017) and A-GEM (Chaudhry et al., 2019) -- to see
whether either incidentally provides some robustness to the poisoning attack
on its own, with no targeted correction.

FOUR lineages, all starting from the same task-0 model (see
madar_pocket_pipeline_tinyimagenet.py for task 0's plain-pretraining
details):

  clean             -- never poisoned; reference. Own replay buffer, filled
                       from its own (always-clean) training data each task.
  poisoned_baseline -- poisoned every task, never fixed. IDENTICAL to
                       madar_pocket_pipeline_tinyimagenet.py's
                       poisoned_baseline.
  si                -- poisoned every task; adapts with plain cross-entropy
                       plus a Synaptic Intelligence penalty. NO replay
                       buffer at all (matches the network-flow version's own
                       choice: "the SI-only number is not interpretable
                       without" the si_c=0 matched control -- run both).
  agem              -- poisoned every task; adapts with plain cross-entropy
                       on the current batch only, gradient-projected against
                       a reference gradient from its OWN reservoir-sampled
                       episodic memory whenever the two conflict. Memory is
                       NEVER mixed into the training batch.

SynapticIntelligence/ReservoirBuffer/si_adapt/agem_adapt are ported
VERBATIM from the network-flow file -- both operate purely on
model.named_parameters()/flat gradients, with no assumption about
architecture or input shape, so they work unchanged on ClassifierCNN's
(N, C, H, W) image tensors exactly as they did on flat feature vectors.

Neither SI nor A-GEM does forget-set-targeted correction, so there is no
detector here: si/agem are logged as one post-adaptation snapshot per task,
in the same table shape (task/pooled/mean/adv accuracy + still-evades %) the
image pipeline's fix variants used.

Masked-prefix evaluation (n_active): same convention as
madar_pocket_pipeline_tinyimagenet.py -- threaded through every
.score()/.predict()/_fmt_report call so a not-yet-introduced class's
untrained head weights can never win an argmax.
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

import madar_pocket_pipeline_tinyimagenet as base
from tinyimagenet_data_loader import DEFAULT_TASK_SETUP, build_schedule, load_tinyimagenet_tasks

LINEAGE_NAMES = ["clean", "poisoned_baseline", "si", "agem"]
BASELINE_NAMES = ["si", "agem"]  # the two under test; get the full metrics table


# ---------------------------------------------------------------------------
# Synaptic Intelligence -- ported UNCHANGED from madar_pocket_pipeline_si_agem.py
# (Zenke et al., 2017 path-integral importance). Operates purely on
# model.named_parameters(), so it makes no assumption about architecture and
# works with ClassifierCNN unchanged.
# ---------------------------------------------------------------------------
class SynapticIntelligence:
    """W accumulates -grad * delta over each optimizer step; at a task
    boundary it folds into omega, normalized by the squared distance the
    parameter actually moved. p_old anchors to the start of the current task.
    """

    def __init__(self, model, si_c=1.0, eps=0.1):
        self.si_c = float(si_c)
        self.eps = float(eps)
        self.W = {self._key(n): torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}
        self.omega = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.p_old = {self._key(n): p.detach().clone()
                      for n, p in model.named_parameters() if p.requires_grad}

    @staticmethod
    def _key(name):
        return name.replace(".", "__")

    def penalty(self, model):
        if self.si_c == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        total = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                k = self._key(n)
                total = total + (self.omega[k] * (p - self.p_old[k]) ** 2).sum()
        return total

    def snapshot(self, model):
        return {self._key(n): (p.grad.detach().clone(), p.detach().clone())
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None}

    def accumulate(self, model, snap):
        for n, p in model.named_parameters():
            k = self._key(n)
            if p.requires_grad and k in snap:
                grad, before = snap[k]
                self.W[k].add_(-grad * (p.detach() - before))

    def end_task(self, model, advance_anchor=True):
        with torch.no_grad():
            pre_fold_penalty = float(self.penalty(model).item())
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = self._key(n)
            current = p.detach().clone()
            self.omega[k] += self.W[k] / ((current - self.p_old[k]) ** 2 + self.eps)
            self.W[k].zero_()
            if advance_anchor:
                self.p_old[k] = current
        return pre_fold_penalty

    def advance_anchor(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.p_old[self._key(n)] = p.detach().clone()


def si_adapt(lineage, si, X, y, epochs, batch_size):
    """Plain CE + si.si_c * si.penalty(model), on lineage.model/lineage.opt."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    n = len(Xt)
    lineage.model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            lineage.opt.zero_grad()
            loss = lineage.loss_fn(lineage.model(Xt[idx]), yt[idx]) + si.si_c * si.penalty(lineage.model)
            loss.backward()
            snap = si.snapshot(lineage.model)
            lineage.opt.step()
            si.accumulate(lineage.model, snap)
    lineage.model.eval()


# ---------------------------------------------------------------------------
# A-GEM -- ported UNCHANGED (Chaudhry et al., ICLR 2019). Reservoir memory is
# A-GEM's own published policy, independent of this project's anomaly/inlier
# buffers.
# ---------------------------------------------------------------------------
class ReservoirBuffer:
    def __init__(self, mem_size, seed=0):
        self.mem_size = int(mem_size)
        self.rng = np.random.default_rng(seed)
        self.X = []
        self.y = []
        self.n_seen = 0

    def __len__(self):
        return len(self.X)

    def is_empty(self):
        return not self.X

    def add_stream(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        admitted = replaced = 0
        for i in range(len(y)):
            self.n_seen += 1
            if len(self.X) < self.mem_size:
                self.X.append(X[i].copy())
                self.y.append(int(y[i]))
                admitted += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.mem_size:
                    self.X[j] = X[i].copy()
                    self.y[j] = int(y[i])
                    replaced += 1
        return {"size": len(self.X), "n_seen": self.n_seen,
                "admitted": admitted, "replaced": replaced}

    def sample(self, n):
        if not self.X:
            return None, None
        k = min(int(n), len(self.X))
        idx = self.rng.choice(len(self.X), k, replace=False)
        X = torch.as_tensor(np.stack([self.X[i] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.array([self.y[i] for i in idx]), dtype=torch.long)
        return X, y


def _flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters() if p.requires_grad])


def _write_grad(model, flat):
    i = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def agem_adapt(lineage, memory, X, y, epochs, batch_size):
    """CE on the current batch; project the gradient against a fresh draw
    from `memory` whenever the two conflict. Memory is NEVER concatenated
    into the training batch -- it only ever supplies the reference gradient.
    """
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    n = len(Xt)
    ce = nn.CrossEntropyLoss()
    lineage.model.train()
    n_steps = n_projected = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue

            g_ref = None
            if not memory.is_empty():
                mx, my = memory.sample(batch_size)
                if mx is not None:
                    lineage.opt.zero_grad()
                    ce(lineage.model(mx), my).backward()
                    g_ref = _flat_grad(lineage.model).clone()

            lineage.opt.zero_grad()
            loss = ce(lineage.model(Xt[idx]), yt[idx])
            loss.backward()

            if g_ref is not None:
                g = _flat_grad(lineage.model)
                dot = float(torch.dot(g, g_ref))
                if dot < 0:
                    denom = float(torch.dot(g_ref, g_ref))
                    if denom > 0:
                        _write_grad(lineage.model, g - (dot / denom) * g_ref)
                        n_projected += 1
            lineage.opt.step()
            n_steps += 1
    lineage.model.eval()
    return {"n_steps": n_steps, "n_projected": n_projected,
            "projection_rate": n_projected / max(1, n_steps)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_si_agem_tinyimagenet_run")
    ap.add_argument("--data-root", type=str, default=base.DATA_ROOT)
    ap.add_argument("--cache-root", type=str, default=None)
    ap.add_argument("--task_setup", type=str, default=DEFAULT_TASK_SETUP)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--cnn_channels", type=str, default=",".join(str(c) for c in base.DEFAULT_CNN_CHANNELS),
                     help="Comma-separated conv channel widths. Same meaning/default as in "
                          "madar_pocket_pipeline_tinyimagenet.py.")
    ap.add_argument("--latent_dim", type=int, default=base.DEFAULT_LATENT_DIM)
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline_tinyimagenet.py.")
    ap.add_argument("--si_c", type=float, default=1.0,
                     help="Synaptic Intelligence penalty weight. Pass 0 for the matched "
                          "control (identical harness, penalty removed) -- the si numbers "
                          "are not interpretable without also running that.")
    ap.add_argument("--si_eps", type=float, default=0.1,
                     help="Synaptic Intelligence damping term (denominator floor).")
    ap.add_argument("--agem_mem_size", type=int, default=base.MEM_SIZE,
                     help="A-GEM's own reservoir memory capacity. Defaults to the same "
                          "mem_size as this project's other buffers, so the comparison is "
                          "at equal memory budget (A-GEM's own selection policy, though).")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}.")
    args = ap.parse_args()
    cnn_channels = tuple(int(c) for c in args.cnn_channels.split(","))
    schedule = build_schedule(args.task_setup)

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_si_agem_tinyimagenet", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (TinyImageNet, SI / A-GEM continual-learning baselines)\n"
            "===================================================================================\n"
            "4 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "si (Synaptic Intelligence), agem (A-GEM gradient projection). Same\n"
            "poisoning/attack/pocket-targeting as madar_pocket_pipeline_tinyimagenet.py -- see\n"
            "that file for those mechanics. si/agem are general continual-learning methods,\n"
            "not forget-set-targeted fixes: no detector runs in this pipeline.\n"
            f"Task schedule: {schedule.spec} ({schedule.n_tasks} tasks, {schedule.n_classes} classes)\n"
            f"CNN channels: {cnn_channels}, latent_dim: {args.latent_dim}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
            f"si_c={args.si_c}, si_eps={args.si_eps}, agem_mem_size={args.agem_mem_size}\n"
        )

    print(f"Loading TinyImageNet from {args.data_root} under schedule {schedule.spec}...")
    tasks, wnids = load_tinyimagenet_tasks(args.data_root, schedule, cache_root=args.cache_root)
    print(f"n_tasks={schedule.n_tasks}, n_classes={schedule.n_classes}, "
          f"task sizes(train)={[len(t['y_train']) for t in tasks]}")

    scaler_mean, scaler_std = None, None
    lineages = {}
    clean_label_buffers, clean_replay_buffer = {}, []
    baseline_label_buffers, baseline_replay_buffer = {}, []
    si_obj = None
    agem_memory = None
    task_test_splits = {}
    class_centroids = {}
    results = []

    def to_scaled(X_raw_uint8):
        X = X_raw_uint8.astype(np.float32) / 255.0
        return ((X - scaler_mean[None, :, None, None]) / scaler_std[None, :, None, None]).astype(np.float32)

    for t in range(schedule.n_tasks):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        task = tasks[t]
        new_classes = schedule.classes_for(t)
        n_active = schedule.active_count(t)
        X_train_raw, y_train = task["X_train"], task["y_train"]
        X_test_raw, y_test = task["X_test"], task["y_test"]

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            X_f = X_train_raw.astype(np.float32) / 255.0
            scaler_mean = X_f.mean(axis=(0, 2, 3))
            scaler_std = X_f.std(axis=(0, 2, 3))
            scaler_std[scaler_std <= 0] = 1.0
            X_train_scaled = to_scaled(X_train_raw)
            X_test_scaled = to_scaled(X_test_raw)

            base_model = base.ClassifierCNN(schedule.n_classes, channels=cnn_channels,
                                             latent_dim=args.latent_dim).to(base.DEVICE)
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
            # SI anchors to the END of task 0 (the state every lineage starts
            # continual adaptation from).
            si_obj = SynapticIntelligence(lineages["si"].model, si_c=args.si_c, eps=args.si_eps)
            agem_memory = ReservoirBuffer(mem_size=args.agem_mem_size, seed=args.seed)

            task_acc = {name: lineages[name].score(X_test_scaled, y_test, n_active=n_active)
                        for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_scaled, y_test)
            base.update_class_centroids(class_centroids, X_train_scaled, y_train, new_classes)

            sample_id = np.arange(len(y_train))
            category = np.array(["clean"] * len(y_train), dtype=object)
            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, sample_id,
                new_classes,
            )
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, sample_id, new_classes,
            )
            agem_memory.add_stream(X_train_scaled, y_train)

            train_section = (
                f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
                f"n_train: {len(y_train)}\nperturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = f"n_test: {len(y_test)}\ngenuine pockets: N/A (task 0 -- no poisoning yet)\n"
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
                "task_id": t, "seed": args.seed, "schedule": schedule.as_dict(), "wnids": wnids,
                "scaler_mean": scaler_mean, "scaler_std": scaler_std,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
                "baseline_label_buffers": baseline_label_buffers,
                "baseline_replay_buffer": baseline_replay_buffer,
                "si_state": {"W": si_obj.W, "omega": si_obj.omega, "p_old": si_obj.p_old,
                             "si_c": si_obj.si_c, "si_eps": si_obj.eps},
                "agem_memory": {"X": agem_memory.X, "y": agem_memory.y, "n_seen": agem_memory.n_seen},
                "task_test_splits": task_test_splits, "class_centroids": class_centroids,
                "results": results, "poison_fraction": poison_fraction,
                "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
                "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path)
            continue

        # -------------------------------------------------------------
        # Tasks 1..N-1: poison -> attack, every task's own new classes.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                 replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison, shared by poisoned_baseline/si/agem.
        X_train_poisoned, poison_idx, poison_target = base.craft_task_poison_images(
            lineages["clean"], X_train_scaled, y_train, new_classes, class_centroids,
            poison_fraction, n_active,
        )

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix"),
        # identical to madar_pocket_pipeline_tinyimagenet.py.
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                             replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                             epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(
                X_train_poisoned[poison_idx],
                np.array([poison_target[int(i)] for i in poison_idx]), n_active=n_active,
            ) if len(poison_idx) else float("nan")
        )

        # Step 5: si adapts on poisoned data, CE + SI penalty, no buffer.
        si_adapt(lineages["si"], si_obj, X_train_poisoned, y_train,
                 epochs=base.ADAPT_EPOCHS, batch_size=base.ADAPT_BATCH_SIZE)
        si_penalty = si_obj.end_task(lineages["si"].model)

        # Step 6: agem adapts on poisoned data, CE with gradient projection
        # against its own reservoir memory.
        agem_info = agem_adapt(lineages["agem"], agem_memory, X_train_poisoned, y_train,
                                epochs=base.ADAPT_EPOCHS, batch_size=base.ADAPT_BATCH_SIZE)

        # Step 7: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean) -- same points re-scored under
        # every lineage below.
        eps_this_task = base.typical_class_gap_multiclass(class_centroids, new_classes)
        if eps_this_task is None:
            eps_this_task = 1.0
        eps_this_task *= base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, n_active=n_active, per_feature_epsilon=args.per_feature_epsilon,
        )

        # Step 8: spillover check -- re-attack every PRIOR task's test set.
        historical_adv = {}
        for s, (Xs_scaled, ys) in task_test_splits.items():
            eps_s = base.typical_class_gap_multiclass(class_centroids, schedule.classes_for(s))
            eps_s = (eps_s * base.ATTACK_EPS_MULTIPLIER) if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = base.adversarial_attack_pocket(
                lineages["poisoned_baseline"], lineages["clean"], Xs_scaled, ys,
                epsilon_max=eps_s, n_active=n_active, per_feature_epsilon=args.per_feature_epsilon,
            )
            historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)

        all_clean_sets = dict(task_test_splits)
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
                lineages[name], all_test_sets_full, n_active)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test, n_active)

        still_evades = {}
        for name in BASELINE_NAMES:
            pred = lineages[name].predict(X_test_adv, n_active=n_active)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 9: update clean's and poisoned_baseline's buffers, AFTER every
        # lineage's adaptation this task is fully done. clean's buffer is
        # filled from its OWN clean data; poisoned_baseline's is unfiltered.
        category_all = np.array(["clean"] * len(y_train), dtype=object)
        category_all[poison_idx] = "perturbed"
        clean_category = np.array(["clean"] * len(y_train), dtype=object)
        sample_id = np.arange(len(y_train))
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, sample_id,
            new_classes,
        )
        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers, X_train_poisoned, y_train,
            category_all, sample_id, new_classes,
        )
        agem_mem_update = agem_memory.add_stream(X_train_poisoned, y_train)

        task_test_splits[t] = (X_test_scaled, y_test)

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        train_section = (
            f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
            f"n_train: {len(y_train)}\n"
            f"perturbed (oracle): {len(poison_idx)}\n"
            f"poison_fraction used: {poison_fraction}\n"
            f"poisoned_baseline accuracy on poisoned points' forced (target-class) labels: "
            f"{acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
        )

        test_section = (
            f"n_test: {len(y_test)}\n"
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        clean_dist = base.buffer_distribution(clean_label_buffers)
        baseline_dist = base.buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"clean's OWN replay buffer distribution (post-update, this task, "
                            f"{len(clean_dist)} classes buffered): total={sum(v['total'] for v in clean_dist.values())}")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution (post-update, this task, "
                            f"{len(baseline_dist)} classes buffered): "
                            f"total={sum(v['total'] for v in baseline_dist.values())}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report summary (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
        adapt_section = "\n".join(adapt_lines)

        cl_lines = [
            f"si_penalty (post-task, measures whether SI is doing anything -- "
            f"compare against a si_c=0 run): {si_penalty:.6g}",
            f"agem projections this task: {agem_info['n_projected']}/{agem_info['n_steps']} "
            f"({agem_info['projection_rate'] * 100:.1f}%) -- how often the gradient conflicted with "
            f"memory and was actually projected; near 0% means A-GEM reduced to plain fine-tuning here",
            f"agem memory (post-update, this task): size={agem_mem_update['size']} "
            f"n_seen={agem_mem_update['n_seen']} admitted={agem_mem_update['admitted']} "
            f"replaced={agem_mem_update['replaced']}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}",
        ]
        for name in BASELINE_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adv_acc_name = lineages[name].score(X_test_adv, y_test, n_active=n_active)
            cl_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        cl_lines.append("")
        for name in BASELINE_NAMES:
            cl_lines.append(f"[{name}] classification report summary (this task's clean test):")
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
            X_comb = np.concatenate([Xs_clean, Xs_adv], axis=0)
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb, n_active=n_active):>18.3f}"
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
            "si_penalty": si_penalty, "agem": agem_info,
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "schedule": schedule.as_dict(), "wnids": wnids,
            "scaler_mean": scaler_mean, "scaler_std": scaler_std,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
            "baseline_label_buffers": baseline_label_buffers,
            "baseline_replay_buffer": baseline_replay_buffer,
            "si_state": {"W": si_obj.W, "omega": si_obj.omega, "p_old": si_obj.p_old,
                         "si_c": si_obj.si_c, "si_eps": si_obj.eps},
            "agem_memory": {"X": agem_memory.X, "y": agem_memory.y, "n_seen": agem_memory.n_seen},
            "task_test_splits": task_test_splits, "class_centroids": class_centroids,
            "results": results, "poison_fraction": poison_fraction,
            "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
            "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path)

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")

        if t == schedule.n_tasks - 1:
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled.reshape(len(X_train_scaled), -1))
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels, n_active)

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`si_obj`, `agem_memory`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
