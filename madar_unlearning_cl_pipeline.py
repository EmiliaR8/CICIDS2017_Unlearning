"""
madar_unlearning_cl_pipeline.py

Usage:
    python madar_unlearning_cl_pipeline.py --seed 42 --log_name madar_unlearn_10task_run \\
        --h5-path /mnt/processed_data/subsampled_dataset.h5

MADAR+Unlearning: adds a targeted-forgetting phase after each CL task's regular
MADAR training, per the project's phasing plan (naive -> joint -> MADAR ->
MADAR+Unlearning). Built directly on madar_cl_pipeline.py -- task construction,
red agent, poisoning mechanic, classifier (ClassifierNN/TorchIDSWrapper), MADAR-IF
replay buffer, and the ER+KD+SI training step are copied verbatim, not
reimplemented, so results stay comparable to the naive/joint/MADAR runs already
collected. See madar_cl_pipeline.py's docstring for that shared rationale.

Adapted from two references: clmdu_ember18_mk6B_cd.py (a colleague's EMBER
unlearning script) and Static_Data_Pipeline/clmd_cicids_unlearning_cd.py (an
earlier, not-fully-trusted CICIDS port of it). This file is a fresh port onto
madar_cl_pipeline.py's foundation rather than a fix-up of either -- several
decisions below deliberately diverge from both, spelled out here so they're easy
to correct rather than silently baked in.

Outline this implements (task loop, for each task t with a red agent):
  1. Train on Tt via MADAR's existing ER+KD+SI step (train_cl_er, unchanged).
  2/6. Update replay buffer P -- see "buffer ordering" below for the one change
       from the outline's literal order.
  3. Detect poison in Tt -> Df_t -- see "poison detection" below.
  4. If Df_t is empty, skip unlearning for this task.
  5. Unlearn Df_t (unlearn_teacher_guided) -- SI-protected, KD-anchored on the
     retain set + replay buffer, forget loss pushes Df_t toward uniform.
  7. Verify recovery -- see "verify recovery" below for what's logged.
  8. Move to task t+1.

Four decisions made, not silently assumed (flag anything you want changed):

  - POISON DETECTION (step 3) is being built iteratively, on purpose, rather
    than landing fully-formed. This is ITERATION 1: phi1 only -- an
    IsolationForest decision_function score over the CURRENT model's latent
    embedding of this task's malicious-labeled training samples, forgetting
    the MIDDLE band (DONUT_FORGET_RATIO of the population) rather than the
    extremes. Ported from split_option_b_donut_hole in the EMBER unlearning
    reference (clmdu_ember18_mk6B_cd.py), with its per-malware-family
    joint-fit machinery dropped: that reference fits IF jointly across
    several concurrent "families"; CICIDS poisoning only ever touches the
    malicious-labeled population (benign is never poisoned -- see
    madar_cl_pipeline.py), so there is exactly one population to search and
    no family-level joint fit is meaningful. detect_poison_if_latent() below
    never reads poison_idx or any ground truth -- Df_t is chosen purely from
    the classifier's own latent geometry, unlike the oracle stand-in this
    file used before phi1 was implemented (see git history for that version).
    ITERATION 2 (not yet built) adds phi3 (per-sample CE loss), phi4
    (normalized predictive entropy), and phi5 (top-two probability margin) --
    model-behavior features, from the same paper's feature list, expected to
    matter more than phi1/phi2's raw/latent geometry alone against this
    pipeline's evasion-crafted poisoning specifically (minimal, decision-
    boundary-targeted perturbations are optimized to NOT look geometrically
    anomalous -- see discussion in commit history / conversation). phi6
    (distance to family centroid) and phi7 (log family size) from that same
    paper are EMBER-specific (malware-family concept) and don't transfer to
    CICIDS's flat binary labeling -- not planned for any iteration here.

  - WEIGHT PROTECTION during unlearning uses the existing SI (omega/p_old_task)
    machinery already implemented and validated in madar_cl_pipeline.py, not
    MAS (Memory Aware Synapses) as literally named in the outline. Both
    reference scripts also use SI here, not MAS, despite the outline's
    wording -- MAS is a different importance computation (gradient of squared
    output-norm, not SI's path integral) that neither reference implements and
    that would need to be built and validated from scratch for no
    demonstrated benefit yet.

  - BUFFER ORDERING deviates from the outline's literal steps 2/6 (add Tt to
    the buffer in full, forget-detect, then filter Df_t back out afterward).
    Both reference scripts instead compute the forget/retain split FIRST,
    unlearn, and update the buffer ONCE from the retain set only -- poison
    never enters the buffer even transiently, and it's one buffer touch per
    task instead of two. Adopted as-is here.

  - VERIFY RECOVERY (step 7) was under-logged in both references: they report
    forget/retain-set accuracy before/after (measure_unlearning_efficacy,
    kept as-is below), which is all THIS task's own data -- not "accuracy on
    held-out clean data from prior tasks" as the outline asks for. Added here:
    prior_tasks_recovery in each checkpoint's "unlearning" block snapshots
    every earlier task's held-out test accuracy immediately before and after
    the unlearning phase specifically (not just before/after the whole task,
    which the regular per_task_eval already gives you) -- isolating what
    unlearning itself did to old-task retention from what CL training did.

Everything else (SI bookkeeping order, KD-against-model_pre_unlearn instead of
the CL replay teacher, the alpha=0 control-ablation semantics, forget loss
targeting the full 2-class distribution since CICIDS never has a
prev_active_count:active_count "newly introduced class range" to target
specifically) is carried over from the two references unchanged; see the
docstrings on unlearn_teacher_guided / measure_unlearning_efficacy below for
exactly what and why.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from adversary_env import NetworkAttackEnv, ContrastiveBank
from h5_data_loader import load_pooled_chronological_tasks

# ---------------------------------------------------------------------------
# Top-of-file config. Task/red-agent/MADAR constants kept identical in name/
# value to madar_cl_pipeline.py wherever shared, so runs are directly
# comparable. Only --seed and --log_name are CLI.
# ---------------------------------------------------------------------------
H5_DATASET_PATH = "/mnt/processed_data/subsampled_dataset.h5"

SEED = 42  # overwritten from --seed at the top of main(); module-level so
           # update_buffer_madar's IsolationForest(random_state=SEED) sees it.

NUM_TASKS = 10
# task0 = 30% warm start; tasks1-9 taper 9.18% -> 6.38% of the pooled dataset.
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]
TASK_TEST_FRAC = 0.20

POISON_FRACTION = 0.3
REQUIRE_EVASION_SUCCESS = True  # only successfully-evasive perturbations poison training data

RED_EPSILON = 0.25
RED_MAX_STEPS = 25
RED_TIMESTEPS_PER_TASK = 2500
ALPHA_CONTRAST = 0.5
CONTRASTIVE_EMA = 0.95
CONTRASTIVE_RECENCY_DECAY = 0.5  # weight of task (k-1) vs (k-2) vs ... in the diversity reward

# Caps evaluate_agent_on_batch's per-task episode count for runtime; None = every malicious sample.
MAX_EVAL_SAMPLES_PER_TASK = 500

SAC_POLICY_KWARGS = dict(net_arch=[512, 256, 128], optimizer_kwargs={"weight_decay": 1e-4})
SAC_KWARGS = dict(
    learning_rate=1e-3, buffer_size=50_000, batch_size=128, gamma=0.95,
    ent_coef="auto_0.01", device="auto",
)

# --- MADAR classifier / CL hyperparameters (identical to madar_cl_pipeline.py) ---
TRAIN_DEVICE = "cpu"
DEVICE = torch.device(TRAIN_DEVICE)

FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30
CL_ITERS = 2000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

MEM_SIZE = 4000
MADAR_CONTAMINATION = 0.1

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1
RNT_FLOOR = 0.3  # see madar_cl_pipeline.py's definition of this constant for the
                  # full rationale (madar1.json vs madar2.json validated the fix)

# --- Unlearning-specific hyperparameters (values carried over from both
# references -- generic optimization hyperparameters, not detection-specific,
# so no reason to change them for this draft) ---
UNLEARN_EPOCHS = 3
UNLEARN_LR = 1e-4
UNLEARN_ALPHA = 0.2      # weight of the forget loss; 0.0 = extra-training control
                          # ablation (identical steps/optimizer/data flow, no forget
                          # objective -- isolates "more training" from "actually
                          # targeting forgetting")
POISON_DETECTOR = "if_latent_phi1_donut_hole"  # logged into config/results for
                                                # provenance; see module docstring's
                                                # "poison detection" note
DONUT_FORGET_RATIO = 0.1  # fraction of this task's malicious-labeled samples
                           # forgotten (the middle band of the phi1 IF-score
                           # distribution) -- value carried over unchanged from
                           # split_option_b_donut_hole's forget_ratio default


# ---------------------------------------------------------------------------
# Recency-weighted contrastive diversity (identical to madar_cl_pipeline.py)
# ---------------------------------------------------------------------------
class RecencyWeightedContrastiveBank(ContrastiveBank):
    """
    Same EMA-prototype machinery as adversary_env.ContrastiveBank, but
    contrastive_reward compares against a geometrically recency-weighted
    average of every prior task's prototype instead of a flat mean, so the
    immediately-previous task's strategy is weighted most heavily while
    older tasks still contribute.
    """

    def __init__(self, dim, ema=0.95, recency_decay=0.5):
        super().__init__(dim, ema=ema)
        self.recency_decay = recency_decay
        self.task_order = []  # ids in the order their prototypes were first created

    def update(self, agent_id, emb):
        super().update(agent_id, emb)
        if agent_id not in self.task_order:
            self.task_order.append(agent_id)

    def contrastive_reward(self, agent_id, emb):
        others_oldest_first = [pid for pid in self.task_order if pid != agent_id and pid in self.protos]
        if not others_oldest_first:
            return 0.0

        emb = self._norm(emb)
        others_recent_first = list(reversed(others_oldest_first))
        weights = np.array([self.recency_decay ** i for i in range(len(others_recent_first))])
        weights = weights / weights.sum()
        neg_sims = np.array([float(np.dot(emb, self.protos[pid])) for pid in others_recent_first])
        neg_mean = float(np.dot(weights, neg_sims))

        pos_sim = float(np.dot(emb, self.protos[agent_id])) if agent_id in self.protos else 0.0
        return pos_sim - neg_mean


# ---------------------------------------------------------------------------
# Red agent training / evaluation (identical to madar_cl_pipeline.py -- the red
# agent's retraining cadence and poisoning mechanic don't depend on which blue
# CL strategy or unlearning is in use).
# ---------------------------------------------------------------------------
class EpisodeTimerCallback(BaseCallback):
    """Measures wall-clock duration of each environment episode during model.learn()."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_times = []
        self._start_times = None
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("# episode_index, seconds\n")

    def _on_training_start(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        now = time.perf_counter()
        self._start_times = [now for _ in range(n_envs)]

    def _on_step(self):
        dones = self.locals.get("dones", None)
        if dones is None:
            return True
        dones_list = list(dones) if isinstance(dones, (list, tuple, np.ndarray)) else [bool(dones)]

        now = time.perf_counter()
        if self._start_times is None:
            self._start_times = [now for _ in range(len(dones_list))]

        for env_idx, done in enumerate(dones_list):
            if self._start_times[env_idx] is None:
                self._start_times[env_idx] = now
            if done:
                elapsed = now - self._start_times[env_idx]
                self.episode_times.append(elapsed)
                with open(self.log_path, "a") as f:
                    f.write(f"{len(self.episode_times)},{elapsed:.6f}\n")
                self._start_times[env_idx] = now
        return True

    def _on_training_end(self):
        if not self.episode_times:
            return
        avg = sum(self.episode_times) / len(self.episode_times)
        with open(self.log_path, "a") as f:
            f.write(f"# total_episodes={len(self.episode_times)}, avg_seconds={avg:.6f}\n")


def run_single_agent_attack(env, model, start_index, benign_label, deterministic=True):
    """Runs one episode attacking one sample index. Returns
    (final_obs, total_reward, pred_label, cum_perturbation)."""
    obs, _ = env.reset(options={"index": start_index})
    total_reward = 0.0
    cum_pert = np.zeros(env.action_space.shape[0], dtype=np.float32)

    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        a = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)
        a = np.clip(a, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(a)
        total_reward += float(reward)
        cum_pert += a
        done = bool(terminated) or bool(truncated)

    return obs, total_reward, info["prediction"], cum_pert


def evaluate_agent_on_batch(env, model, X, y, benign_label, only_malicious=True,
                             deterministic=True, max_test=None):
    """
    Runs the agent over (up to max_test of) the malicious samples. Returns
    (X_pert, evasion_rate, rewards, avg_pert_norm, n_attacked, evaded_mask).
    evaded_mask[i] is True only for samples where the attack succeeded -- this is
    the exact set poisoning (and, downstream, the oracle detector) should draw
    from.
    """
    X_pert = X.copy()
    evaded_mask = np.zeros(X.shape[0], dtype=bool)
    rewards, pert_norms = [], []
    evasion = 0
    attacked = 0

    if only_malicious:
        indices = np.where(y != benign_label)[0]
    else:
        indices = np.arange(X.shape[0])
    if max_test is not None:
        indices = indices[:max_test]

    for i in indices:
        final_obs, total_reward, pred_label, cum_pert = run_single_agent_attack(
            env, model, start_index=i, benign_label=benign_label, deterministic=deterministic
        )
        rewards.append(total_reward)
        pert_norms.append(float(np.linalg.norm(cum_pert, ord=2)))
        attacked += 1

        if pred_label == benign_label and y[i] != benign_label:
            evasion += 1
            X_pert[i] = final_obs
            evaded_mask[i] = True

    evasion_rate = evasion / max(attacked, 1)
    avg_pert_norm = float(np.mean(pert_norms)) if pert_norms else 0.0
    return X_pert, evasion_rate, rewards, avg_pert_norm, attacked, evaded_mask


def train_red_agent_for_task(task_id, classifier, X_train, y_train, benign_label, bank, seed, out_dir):
    def _thunk():
        return NetworkAttackEnv(
            classifier, X_train, y_train,
            benign_label=benign_label,
            max_steps=RED_MAX_STEPS,
            epsilon=RED_EPSILON,
            agent_id=task_id,
            contrastive_bank=bank,
            alpha_contrast=ALPHA_CONTRAST,
        )

    vec_env = make_vec_env(_thunk, n_envs=1, seed=seed + task_id)
    agent = SAC(
        "MlpPolicy", vec_env, verbose=0, policy_kwargs=SAC_POLICY_KWARGS,
        seed=seed + task_id, **SAC_KWARGS,
    )
    timer_cb = EpisodeTimerCallback(
        log_path=os.path.join(out_dir, "logs", f"red_episode_times_task{task_id}.txt")
    )
    agent.learn(total_timesteps=RED_TIMESTEPS_PER_TASK, callback=timer_cb)
    return vec_env.envs[0], agent


# ---------------------------------------------------------------------------
# Blue classifier: PyTorch MLP with a latent layer (needed for MADAR-IF buffer
# selection AND for the unlearning phase's latent-space bookkeeping), identical
# to madar_cl_pipeline.py. See that file's docstring for why this replaces
# XGBoostIDSWrapper.
# ---------------------------------------------------------------------------
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(128, num_classes)

    def forward(self, x, return_latent=False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


class TorchIDSWrapper:
    """
    predict()/predict_proba() interface matching XGBoostIDSWrapper's contract, so
    this drops into evaluate_classifier() and adversary_env.NetworkAttackEnv
    unchanged. Takes RAW ([0,1]-clipped, unscaled) features and applies the
    task-0-fit StandardScaler + FEATURE_CLIP internally, mirroring to_tensor() in
    main(). `model` is held by reference and mutated in place by training AND by
    unlearning, so this wrapper's output always reflects the classifier's current
    state without needing to be reconstructed.
    """

    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device

    def _forward_probs(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.clip(self.scaler.transform(X), -FEATURE_CLIP, FEATURE_CLIP)
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model(Xt), dim=1)
        return probs.cpu().numpy()

    def predict_proba(self, X):
        return self._forward_probs(X)

    def predict(self, X):
        return np.argmax(self._forward_probs(X), axis=1)


# ---------------------------------------------------------------------------
# MADAR replay buffer (identical to madar_cl_pipeline.py -- keyed by binary
# label, UNIFORM 50/50 budget). The only behavioral change from that file is at
# the CALL SITE in main(): this file calls update_buffer_madar with the
# post-unlearning RETAIN set only, never the full (possibly poison-containing)
# task batch -- see module docstring's "buffer ordering" note. The function
# itself is unchanged.
# ---------------------------------------------------------------------------
label_buffers = {}   # {0: [(X_scaled, y, category), ...], 1: [...]}
replay_buffer = []   # flattened label_buffers.values()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    return F.kl_div(log_scores_norm, targets_norm, reduction='batchmean') * (T ** 2)


def _embed(model, Xt: torch.Tensor, device):
    loader = data.DataLoader(data.TensorDataset(Xt), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    parts = []
    model.eval()
    with torch.no_grad():
        for (v,) in loader:
            _, latent = model(v.to(device), return_latent=True)
            parts.append(latent.cpu())
    return torch.cat(parts).numpy()


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, benign_label, mal_label,
                         model, device):
    """
    Selects, per label (benign/malicious), an interleaved anomalies+inliers sample
    (by IsolationForest decision_function over the model's LATENT space, budget
    fixed at MEM_SIZE // 2 for both groups regardless of observed class ratio) to
    keep in that label's buffer slot. Re-ranks the UNION of that label's EXISTING
    buffer entries (re-embedded under the CURRENT model, since weights have moved
    since they were last selected) and the newly-passed-in samples for this call,
    so exemplars from earlier tasks can survive across updates. `category` is
    attached to each stored sample purely for the buffer_composition_summary()
    diagnostic -- it plays no role in selection, which sees only latent vectors.
    """
    global label_buffers, replay_buffer
    model.eval()

    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)
    L_np = _embed(model, X, device)
    latent_dim = L_np.shape[1]

    budget_per_label = MEM_SIZE // 2

    for lbl in (benign_label, mal_label):
        old_entries = label_buffers.get(lbl, [])
        if old_entries:
            old_X = np.stack([e[0].numpy() for e in old_entries])
            old_Y = np.array([e[1].item() for e in old_entries], dtype=Y_np.dtype)
            old_cat = np.array([e[2] for e in old_entries], dtype=object)
            old_L = _embed(model, torch.tensor(old_X), device)
        else:
            old_X = np.empty((0, X_np.shape[1]), dtype=X_np.dtype)
            old_Y = np.empty((0,), dtype=Y_np.dtype)
            old_cat = np.empty((0,), dtype=object)
            old_L = np.empty((0, latent_dim), dtype=np.float32)

        mask = (Y_np == lbl)
        pool_X = np.concatenate([old_X, X_np[mask]], axis=0)
        pool_Y = np.concatenate([old_Y, Y_np[mask]], axis=0)
        pool_L = np.concatenate([old_L, L_np[mask]], axis=0)
        pool_cat = np.concatenate([old_cat, cat_np[mask]], axis=0)

        n_select = min(budget_per_label, len(pool_X))
        if n_select == 0:
            continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(pool_L)
        scores = iso.decision_function(pool_L)
        sorted_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_idx = sorted_idx[:half]
        inliers_idx = sorted_idx[-(n_select - half):]
        interleaved_idx = [idx for pair in zip(anomalies_idx, inliers_idx) for idx in pair]
        if n_select % 2 != 0:
            interleaved_idx.append(inliers_idx[-1])

        label_buffers[lbl] = [
            (torch.tensor(pool_X[i]), torch.tensor(pool_Y[i]), pool_cat[i]) for i in interleaved_idx
        ]

    replay_buffer.clear()
    for buf in label_buffers.values():
        replay_buffer.extend(buf)


def buffer_composition_summary():
    """Diagnostic only: what fraction of each label's buffer slot is benign /
    malicious_clean / malicious_perturbed right now. Never used by training or
    selection."""
    summary = {}
    for lbl, entries in label_buffers.items():
        cats = [e[2] for e in entries]
        counts = {}
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
        summary[str(lbl)] = {"total": len(entries), "category_counts": counts}
    return summary


def train_cl_er(model, teacher_model, optimizer, loader, iters, W, omega, p_old_task, tid, si_c, device):
    """
    One task's CL training (identical to madar_cl_pipeline.py's train_cl_er, incl.
    the RNT_FLOOR fix). Each step mixes a batch from the current task's own loader
    with a batch from the (previous-tasks-only) replay buffer. Current-task samples
    get plain cross-entropy; buffer samples get KD against the frozen teacher (the
    model as it stood at the end of the previous task's UNLEARNING phase -- see
    main()) instead of their stored hard labels. `rnt` shifts weight from
    current-task loss toward replay KD as tasks accumulate (1/(tid+1), floored at
    RNT_FLOOR). SI adds a per-parameter quadratic penalty against drifting from
    p_old_task, weighted by omega.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    loader_iter = iter(cycle(loader))

    assert replay_buffer, "train_cl_er requires a non-empty replay buffer"
    b_v = torch.stack([entry[0] for entry in replay_buffer])
    b_l = torch.stack([entry[1] for entry in replay_buffer])
    buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l), batch_size=BATCH_SIZE, shuffle=True)
    buf_iter = iter(cycle(buf_loader))

    for step in range(iters):
        v, l = next(loader_iter)
        v, l = v.to(device), l.to(device)
        optimizer.zero_grad()

        mem_v, mem_l = next(buf_iter)
        mem_v, mem_l = mem_v.to(device), mem_l.to(device)

        combined_v = torch.cat([v, mem_v], dim=0)
        combined_l = torch.cat([l, mem_l], dim=0)
        loss_cur = nn.CrossEntropyLoss()(model(combined_v), combined_l)

        rnt = max(1.0 / (tid + 1), RNT_FLOOR)
        outputs_mem = model(mem_v)
        with torch.no_grad():
            teacher_logits = teacher_model(mem_v)

        loss_replay = loss_fn_kd(outputs_mem, teacher_logits, T=KD_TEMP)
        loss_main = (rnt * loss_cur) + ((1.0 - rnt) * loss_replay)

        si_loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                n_key = n.replace('.', '__')
                si_loss += (omega[n_key] * (p - p_old_task[n_key]) ** 2).sum()

        total_loss = loss_main + (si_c * si_loss)
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at task {tid}, step {step} - training diverged")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # SI path integral: pair each gradient with the parameter step it actually
        # caused. Snapshot params and (clipped) grads, take the optimizer step,
        # then accumulate W += -g_t * (theta_{t+1} - theta_t).
        snap = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                n_key = n.replace('.', '__')
                snap[n_key] = (p.grad.detach().clone(), p.detach().clone())

        optimizer.step()

        for n, p in model.named_parameters():
            if p.requires_grad:
                n_key = n.replace('.', '__')
                if n_key in snap:
                    g, p_before = snap[n_key]
                    W[n_key].add_(-g * (p.detach() - p_before))


# ---------------------------------------------------------------------------
# Blue classifier evaluation (identical schema to naive/joint/MADAR's
# evaluate_classifier -- so compare_cl_runs.py works on this pipeline's JSON
# unmodified).
# ---------------------------------------------------------------------------
def evaluate_classifier(classifier, X, y):
    y_pred = classifier.predict(X)
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }


# ---------------------------------------------------------------------------
# Step 3 -- poison detection, iteration 1 (phi1 only; see module docstring) +
# unlearning phase itself.
# ---------------------------------------------------------------------------
def detect_poison_if_latent(Xtr, ytr, mal_label, model, device, forget_ratio=DONUT_FORGET_RATIO):
    """
    Step 3, iteration 1: phi1 -- IsolationForest decision_function score over
    the CURRENT model's latent embedding of this task's malicious-labeled
    training samples, forgetting the MIDDLE band of that score distribution
    (neither the strongest anomalies nor the strongest inliers). Ported from
    split_option_b_donut_hole in the EMBER unlearning reference
    (clmdu_ember18_mk6B_cd.py); the reference's per-malware-family joint IF
    fit is dropped -- CICIDS poisoning only ever touches the malicious-labeled
    population (benign is never poisoned), so there's exactly one population
    to search, not several families to fit jointly over.

    Reads NO ground truth -- Df_t is chosen purely from the classifier's own
    latent geometry (unlike this file's earlier oracle stand-in, kept in git
    history). Benign samples are never candidates and are always retained.

    Returns (forget_idx, retain_idx) as sorted int64 index arrays into this
    task's own Xtr/ytr -- the contract the rest of the pipeline depends on,
    so iteration 2 (phi3/phi4/phi5 folded in) is a same-shape swap here.
    """
    mal_idx = np.where(ytr.numpy() == mal_label)[0]
    n_samples = len(ytr)
    if len(mal_idx) == 0:
        return np.array([], dtype=np.int64), np.arange(n_samples, dtype=np.int64)

    L_np = _embed(model, Xtr[mal_idx], device)
    iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
    iso.fit(L_np)
    scores = iso.decision_function(L_np)
    sorted_local = np.argsort(scores)

    n_total = len(sorted_local)
    n_forget = int(n_total * forget_ratio)
    if n_forget == 0:
        return np.array([], dtype=np.int64), np.arange(n_samples, dtype=np.int64)

    mid_start = (n_total // 2) - (n_forget // 2)
    mid_end = mid_start + n_forget
    forget_local = sorted_local[mid_start:mid_end]

    forget_idx = np.sort(mal_idx[forget_local])
    retain_idx = np.setdiff1d(np.arange(n_samples, dtype=np.int64), forget_idx)
    return forget_idx, retain_idx


def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, omega, p_old_task,
                           si_c, epochs, lr, alpha, device):
    """
    Pushes the forget set's predictions toward uniform over BOTH classes
    (CICIDS is a fixed 2-class problem -- unlike the EMBER lineage this is
    adapted from, there's no prev_active_count:active_count "newly introduced
    class range" to target specifically, so the full 2-class distribution is
    the target instead), anchored by ground-truth CE + KD on the retain set +
    replay buffer against `teacher_model` (a snapshot taken right after this
    task's own CL training finished -- model_pre_unlearn in main() -- NOT the
    CL phase's replay teacher from the previous task), under an SI penalty
    protecting importance accumulated over all previous tasks' training
    (including this task's own just-finished CL step -- omega was already
    updated for it in main() before this runs, so unlearning is penalized for
    drifting important parameters away from where THIS TASK STARTED, except
    through the forget objective -- intentional, matches both references).

    alpha=0 is the extra-training control ablation: identical steps, optimizer,
    and data flow, with zero forget-loss weight -- isolates "does more training
    alone help" from "does actually targeting the forget set help".
    """
    global replay_buffer
    model.train()
    teacher_model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    retain_iter = iter(cycle(retain_loader))
    if replay_buffer:
        b_v = torch.stack([entry[0] for entry in replay_buffer])
        b_l = torch.stack([entry[1] for entry in replay_buffer])
        buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l), batch_size=BATCH_SIZE, shuffle=True)
        buf_iter = iter(cycle(buf_loader))
    else:
        buf_iter = None

    for epoch in range(epochs):
        for f_v, f_l in forget_loader:
            f_v, f_l = f_v.to(device), f_l.to(device)
            optimizer.zero_grad()

            # --- 1. TARGETED ENTROPY (Forget) -- uniform over both classes ---
            logits = model(f_v)
            log_probs = F.log_softmax(logits, dim=1)
            uniform_probs = torch.full_like(log_probs, 1.0 / logits.shape[1])
            forget_loss = F.kl_div(log_probs, uniform_probs, reduction='batchmean')

            # --- 2. GROUND TRUTH + DISTILLATION (Retain + Replay) ---
            r_v, r_l = next(retain_iter)
            r_v, r_l = r_v.to(device), r_l.to(device)

            if buf_iter:
                mem_v, mem_l = next(buf_iter)
                mem_v, mem_l = mem_v.to(device), mem_l.to(device)
                r_v = torch.cat([r_v, mem_v], dim=0)
                r_l = torch.cat([r_l, mem_l], dim=0)

            with torch.no_grad():
                teacher_logits = teacher_model(r_v)

            student_logits = model(r_v)
            retain_ce = nn.CrossEntropyLoss()(student_logits, r_l)
            retain_kd = loss_fn_kd(student_logits, teacher_logits, T=KD_TEMP)
            retain_loss = (0.5 * retain_ce) + (0.5 * retain_kd)

            # --- 3. SYNAPTIC INTELLIGENCE (Protect Past Tasks) ---
            si_loss = 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    if n_key in p_old_task:
                        si_loss += (omega[n_key] * (p - p_old_task[n_key]) ** 2).sum()

            # --- 4. COMBINED UPDATE ---
            total_loss = (alpha * forget_loss) + ((1.0 - alpha) * retain_loss) + (si_c * si_loss)
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite loss during unlearning - training diverged")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def measure_unlearning_efficacy(model_before, model_after, loader, device):
    """
    Accuracy before/after (fractions, 0-1, matching this project's convention --
    NOT the 0-100 percent scale both references use), prediction entropy (raw
    and normalized by log(2), since CICIDS is always 2-class -- unlike the
    EMBER lineage, no per-task class-count growth to normalize against), and
    latent shift (mean/median/max -- median/max alongside mean since a handful
    of extreme samples can inflate the mean, per mk6B's own observation).
    """
    model_before.eval()
    model_after.eval()

    acc_before = 0.0
    acc_after = 0.0
    entropies = []
    shifts = []
    total_samples = 0

    with torch.no_grad():
        for v, l in loader:
            v, l = v.to(device), l.to(device)

            logits_before, latent_before = model_before(v, return_latent=True)
            logits_after, latent_after = model_after(v, return_latent=True)

            probs = F.softmax(logits_after, dim=1)

            acc_before += (torch.argmax(logits_before, dim=1) == l).sum().item()
            acc_after += (torch.argmax(probs, dim=1) == l).sum().item()

            entropies.append(-(probs * torch.log(probs + 1e-9)).sum(dim=1).cpu())
            shifts.append(torch.norm(latent_after - latent_before, p=2, dim=1).cpu())
            total_samples += l.size(0)

    entropies = torch.cat(entropies)
    shifts = torch.cat(shifts)
    mean_entropy = entropies.mean().item()
    return {
        'acc_before': acc_before / total_samples,
        'acc_after': acc_after / total_samples,
        'entropy': mean_entropy,
        'entropy_norm': mean_entropy / float(np.log(2)),
        'shift_mean': shifts.mean().item(),
        'shift_median': shifts.median().item(),
        'shift_max': shifts.max().item(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_task_metrics(results, out_path):
    """Identical to madar_cl_pipeline.py's plot_task_metrics, title relabeled."""
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc = [r["mean_per_task_balanced_accuracy"] for r in results]
    train_evasion = [r["red_agent"]["train_evasion_rate"] if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"]["test_evasion_rate"] if r["red_agent"] else None for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc, marker="o", label="pooled balanced accuracy")
    ax1.plot(task_ids, mean_per_task_bal_acc, marker="s", label="mean per-task balanced accuracy")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("MADAR+Unlearning: classifier accuracy per task boundary")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xs = [t for t, v in zip(task_ids, train_evasion) if v is not None]
    ys_train = [v for v in train_evasion if v is not None]
    ys_test = [v for v in test_evasion if v is not None]
    ax2.plot(xs, ys_train, marker="o", color="tab:red", label="evasion rate (train samples)")
    ax2.plot(xs, ys_test, marker="s", color="tab:orange", label="evasion rate (held-out test samples)")
    ax2.set_xlabel("Task id")
    ax2.set_ylabel("Evasion rate")
    ax2.set_title("Red agent evasion rate per task boundary (task 0 has no red agent)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_unlearning_metrics(results, out_path):
    """
    NEW (not in either reference): forget/retain-set accuracy before-vs-after
    unlearning, and step 7's prior-task recovery check (mean held-out balanced
    accuracy across every earlier task, before vs. after THIS task's unlearning
    phase specifically -- isolating unlearning's own effect from CL training's).
    Tasks with no unlearning this run (task 0, or an empty oracle Df_t) simply
    don't appear on the x-axis.
    """
    unl_tasks = [r for r in results if r.get("unlearning")]
    if not unl_tasks:
        print("[Plot] No tasks ran unlearning yet -- skipping unlearning_metrics plot.")
        return

    task_ids = [r["task_id"] for r in unl_tasks]
    forget_before = [r["unlearning"]["forget_set"]["acc_before"] for r in unl_tasks]
    forget_after = [r["unlearning"]["forget_set"]["acc_after"] for r in unl_tasks]
    retain_before = [r["unlearning"]["retain_set"]["acc_before"] for r in unl_tasks]
    retain_after = [r["unlearning"]["retain_set"]["acc_after"] for r in unl_tasks]

    def _prior_mean(r, key):
        vals = r["unlearning"]["prior_tasks_recovery"][key]
        if not vals:
            return None
        return float(np.mean([v["balanced_accuracy"] for v in vals.values()]))

    prior_before = [_prior_mean(r, "pre_unlearn") for r in unl_tasks]
    prior_after = [_prior_mean(r, "post_unlearn") for r in unl_tasks]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax = axes[0]
    ax.plot(task_ids, forget_before, marker="o", label="forget set (Df_t), before")
    ax.plot(task_ids, forget_after, marker="s", label="forget set (Df_t), after")
    ax.set_ylabel("Accuracy")
    ax.set_title("Forget set (oracle poison flag): accuracy before vs. after unlearning")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(task_ids, retain_before, marker="o", label="retain set, before")
    ax.plot(task_ids, retain_after, marker="s", label="retain set, after")
    ax.set_ylabel("Accuracy")
    ax.set_title("Retain set: accuracy before vs. after unlearning (should stay high)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    xs = [t for t, v in zip(task_ids, prior_before) if v is not None]
    ys_before = [v for v in prior_before if v is not None]
    ys_after = [v for t, v in zip(task_ids, prior_after) if v is not None]
    ax.plot(xs, ys_before, marker="o", label="prior tasks, before")
    ax.plot(xs, ys_after, marker="s", label="prior tasks, after")
    ax.set_xlabel("Task id")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Prior tasks' held-out accuracy: before vs. after THIS task's unlearning step")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prototype_heatmap(bank, out_path):
    agent_ids, S = bank.cosine_matrix(agent_ids=sorted(bank.protos.keys()))
    if S is None or len(agent_ids) < 2:
        print("[Plot] Not enough task prototypes to plot a heatmap yet.")
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(S, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(agent_ids)), agent_ids)
    plt.yticks(range(len(agent_ids)), agent_ids)
    plt.title("Red agent policy diversity across tasks\n(prototype cosine similarity)")
    plt.xlabel("Task id")
    plt.ylabel("Task id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_episode_clouds(bank, out_path):
    task_ids = sorted(bank.episode_embs.keys())
    X, y = [], []
    for tid in task_ids:
        for e in bank.episode_embs.get(tid, []):
            X.append(e)
            y.append(tid)

    if len(X) < 5:
        print("[Plot] Not enough episode embeddings to plot.")
        return

    X = np.stack(X, axis=0)
    y = np.array(y)

    X0 = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    Z = X0 @ Vt[:2].T

    cmap = matplotlib.colormaps["tab10"]
    plt.figure(figsize=(7, 6))
    for tid in task_ids:
        mask = y == tid
        if mask.sum() == 0:
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.6, color=cmap(tid % 10), label=f"task {tid}")

    plt.title("Episode embedding clouds (PCA-2D of cumulative perturbations)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_unlearn_cl_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    args = ap.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join("runs", "madar_unlearning", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    print(f"Loading {args.h5_path} and building {NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(args.h5_path, TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    bank = RecencyWeightedContrastiveBank(
        dim=feature_dim, ema=CONTRASTIVE_EMA, recency_decay=CONTRASTIVE_RECENCY_DECAY
    )
    bank.init_distance_logger(
        path=os.path.join(out_dir, "logs", "prototype_cosine_over_time.txt"),
        agent_ids=list(range(NUM_TASKS)),
    )

    scaler = None
    model = None
    classifier_wrapper = None
    teacher_model = None
    W = omega = p_old_task = None

    def to_tensor(X_raw):
        X_scaled = np.clip(scaler.transform(X_raw.astype(np.float32)), -FEATURE_CLIP, FEATURE_CLIP)
        return torch.tensor(X_scaled, dtype=torch.float32)

    task_test_splits = {}
    results = []
    warnings_log = []

    for t, task in enumerate(tasks):
        X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)  # data is documented [0,1]-scaled already
        y = task["labels"].astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TASK_TEST_FRAC, random_state=args.seed, stratify=y
        )
        task_test_splits[t] = (X_test, y_test)

        mal_idx_train = np.where(y_train == mal_label)[0]
        red_report = None
        n_poisoned = 0
        poison_idx = np.array([], dtype=int)

        if t == 0:
            print(f"\n===== Task 0 (warm start, no red agent): "
                  f"{len(y_train)} train / {len(y_test)} test =====")
            X_train_for_classifier = X_train

        else:
            print(f"\n===== Task {t}: training fresh red agent against pre-task-{t} classifier "
                  f"({len(y_train)} train / {len(y_test)} test, {len(mal_idx_train)} malicious train) =====")

            if len(mal_idx_train) == 0:
                warnings_log.append(f"Task {t}: no malicious training samples, skipping red agent.")
                X_train_for_classifier = X_train
            else:
                env, agent = train_red_agent_for_task(
                    t, classifier_wrapper, X_train, y_train, benign_label, bank, args.seed, out_dir
                )

                X_train_pert, train_evasion_rate, train_rewards, train_avg_norm, train_attacked, evaded_mask = \
                    evaluate_agent_on_batch(
                        env, agent, X_train, y_train, benign_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                    )
                _, test_evasion_rate, test_rewards, test_avg_norm, test_attacked, _ = evaluate_agent_on_batch(
                    env, agent, X_test, y_test, benign_label,
                    only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                )

                red_report = {
                    "train_evasion_rate": train_evasion_rate, "train_attacked": train_attacked,
                    "train_avg_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
                    "train_avg_pert_l2": train_avg_norm,
                    "test_evasion_rate": test_evasion_rate, "test_attacked": test_attacked,
                    "test_avg_reward": float(np.mean(test_rewards)) if test_rewards else 0.0,
                    "test_avg_pert_l2": test_avg_norm,
                }
                print(f"[Task {t} red agent] train_evasion={train_evasion_rate:.3f} "
                      f"test_evasion={test_evasion_rate:.3f} avg_pert_L2={train_avg_norm:.3f}")

                poison_pool = np.where(evaded_mask)[0] if REQUIRE_EVASION_SUCCESS else mal_idx_train
                n_to_poison = min(int(round(POISON_FRACTION * len(mal_idx_train))), len(poison_pool))
                rng = np.random.RandomState(args.seed + t)
                poison_idx = rng.choice(poison_pool, size=n_to_poison, replace=False) if n_to_poison > 0 else \
                    np.array([], dtype=int)
                n_poisoned = len(poison_idx)

                X_train_for_classifier = X_train.copy()
                X_train_for_classifier[poison_idx] = X_train_pert[poison_idx]
                print(f"[Task {t}] poisoned {n_poisoned}/{len(mal_idx_train)} malicious train samples "
                      f"(poison_fraction={POISON_FRACTION})")

        # Category tracking (identical to madar_cl_pipeline.py) -- feeds the
        # buffer composition diagnostic only in this file; detect_poison_if_latent()
        # below reads no ground truth from this array (see its docstring).
        category = np.full(len(y_train), "malicious_clean", dtype=object)
        category[y_train == benign_label] = "benign"
        if len(poison_idx) > 0:
            category[poison_idx] = "malicious_perturbed"

        if t == 0:
            scaler = StandardScaler()
            scaler.fit(X_train_for_classifier)

            model = ClassifierNN(feature_dim, 2).to(DEVICE)
            classifier_wrapper = TorchIDSWrapper(model, scaler, DEVICE)
            W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

        Xtr = to_tensor(X_train_for_classifier)
        ytr = torch.tensor(y_train, dtype=torch.long)

        unlearning_metrics = None

        if t == 0:
            print(f"  Samples: {len(Xtr)} | pure supervised pretraining, {TASK0_EPOCHS} epochs")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.train()

            grad_steps = 0
            for _ in range(TASK0_EPOCHS):
                for v, l in loader:
                    v, l = v.to(DEVICE), l.to(DEVICE)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(v), l)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at task {t} - training diverged")
                    loss.backward()
                    optimizer.step()
                    grad_steps += 1

            for n, p in model.named_parameters():
                if p.requires_grad:
                    p_old_task[n.replace('.', '__')] = p.detach().clone()
            teacher_model = copy.deepcopy(model); teacher_model.eval()
            update_buffer_madar(Xtr, ytr, category, benign_label, mal_label, model, DEVICE)

        else:
            print(f"  Samples: {len(Xtr)} (+{len(replay_buffer)} replay) | "
                  f"MADAR ER+KD+SI, {CL_ITERS} iters")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = CL_ITERS

            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, W, omega, p_old_task, t, SI_C, DEVICE)

            # SI omega update happens NOW, right after CL training -- W was only
            # accumulated during train_cl_er, so the displacement must be
            # measured at this same point. p_old_task is intentionally NOT
            # updated yet: the unlearning phase's SI penalty below anchors to
            # start-of-task weights, so any drift during unlearning has to be
            # justified by the forget objective specifically. Matches both
            # references' ordering exactly.
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_post_cl = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_post_cl - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()

            # --- Steps 3 + 4: detect poison (phi1, iteration 1), skip if empty ---
            forget_idx, retain_idx = detect_poison_if_latent(Xtr, ytr, mal_label, model, DEVICE)
            print(f"    [Detect] Df_t (phi1 latent-IF donut-hole) = {len(forget_idx)} samples, "
                  f"retain = {len(retain_idx)}")

            if len(forget_idx) == 0:
                print("    [Unlearning] Df_t empty -- skipping unlearning phase for this task.")
                Xtr_for_buffer, ytr_for_buffer, category_for_buffer = Xtr, ytr, category
            else:
                forget_idx_t = torch.tensor(forget_idx, dtype=torch.long)
                retain_idx_t = torch.tensor(retain_idx, dtype=torch.long)
                forget_loader = data.DataLoader(
                    data.TensorDataset(Xtr[forget_idx_t], ytr[forget_idx_t]), batch_size=BATCH_SIZE, shuffle=True
                )
                retain_loader = data.DataLoader(
                    data.TensorDataset(Xtr[retain_idx_t], ytr[retain_idx_t]), batch_size=BATCH_SIZE, shuffle=True
                )

                # --- Step 5 setup: snapshot prior-task held-out accuracy BEFORE
                # unlearning touches the weights (step 7, part 1) ---
                pre_unlearn_prior_eval = {
                    j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t)
                }

                model_pre_unlearn = copy.deepcopy(model)
                print(f"    [Unlearn] task {t} (alpha={UNLEARN_ALPHA}, "
                      f"n_forget={len(forget_idx)}, n_retain={len(retain_idx)})...")
                unlearn_teacher_guided(
                    model=model, teacher_model=model_pre_unlearn,
                    forget_loader=forget_loader, retain_loader=retain_loader,
                    omega=omega, p_old_task=p_old_task, si_c=SI_C,
                    epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA, device=DEVICE,
                )
                grad_steps += UNLEARN_EPOCHS * len(forget_loader)

                f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, DEVICE)
                r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, DEVICE)

                # --- Step 7, part 2: same prior-task snapshot AFTER unlearning ---
                post_unlearn_prior_eval = {
                    j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t)
                }

                for tag, m in (("Forget set", f_m), ("Retain set", r_m)):
                    print(f"    [{tag}] acc {m['acc_before']:.4f} -> {m['acc_after']:.4f}, "
                          f"entropy_norm={m['entropy_norm']:.3f}, latent shift "
                          f"mean/med/max={m['shift_mean']:.3f}/{m['shift_median']:.3f}/{m['shift_max']:.1f}")
                if pre_unlearn_prior_eval:
                    pre_mean = float(np.mean([v["balanced_accuracy"] for v in pre_unlearn_prior_eval.values()]))
                    post_mean = float(np.mean([v["balanced_accuracy"] for v in post_unlearn_prior_eval.values()]))
                    print(f"    [Recovery] prior-task mean balanced_accuracy: {pre_mean:.4f} -> {post_mean:.4f}")

                unlearning_metrics = {
                    "detector": POISON_DETECTOR,
                    "n_forget": int(len(forget_idx)), "n_retain": int(len(retain_idx)),
                    "forget_set": f_m, "retain_set": r_m,
                    "prior_tasks_recovery": {
                        "pre_unlearn": pre_unlearn_prior_eval,
                        "post_unlearn": post_unlearn_prior_eval,
                    },
                }

                # Buffer refreshed from the RETAIN set only (see module
                # docstring's "buffer ordering" note) -- forgotten samples
                # never enter memory, not even transiently.
                Xtr_for_buffer = Xtr[retain_idx_t]
                ytr_for_buffer = ytr[retain_idx_t]
                category_for_buffer = category[retain_idx]

            teacher_model = copy.deepcopy(model); teacher_model.eval()
            update_buffer_madar(Xtr_for_buffer, ytr_for_buffer, category_for_buffer,
                                benign_label, mal_label, model, DEVICE)

        # p_old_task updated to the FINAL (post-unlearning, if it ran this task)
        # weights, for ALL tasks -- ready as next task's start-of-task SI anchor.
        for n, p in model.named_parameters():
            if p.requires_grad:
                p_old_task[n.replace('.', '__')] = p.detach().clone()

        buffer_summary = buffer_composition_summary()
        print(f"    [Buffer] composition: {buffer_summary}")

        per_task_eval = {j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t + 1)}
        pooled_X = np.concatenate([task_test_splits[j][0] for j in range(t + 1)])
        pooled_y = np.concatenate([task_test_splits[j][1] for j in range(t + 1)])
        pooled_eval = evaluate_classifier(classifier_wrapper, pooled_X, pooled_y)
        mean_per_task_bal_acc = float(np.mean([per_task_eval[j]["balanced_accuracy"] for j in range(t + 1)]))

        print(f"[Task {t} classifier] this-task bal_acc={per_task_eval[t]['balanced_accuracy']:.4f} "
              f"pooled bal_acc={pooled_eval['balanced_accuracy']:.4f} "
              f"mean-per-task bal_acc={mean_per_task_bal_acc:.4f}")

        results.append({
            "task_id": t,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "n_malicious_train": int(len(mal_idx_train)),
            "n_poisoned": n_poisoned,
            "red_agent": red_report,
            "per_task_eval": per_task_eval,
            "pooled_eval": pooled_eval,
            "mean_per_task_balanced_accuracy": mean_per_task_bal_acc,
            "grad_steps": grad_steps,
            "replay_buffer_composition": buffer_summary,
            "unlearning": unlearning_metrics,
        })

    config = {
        "strategy": "madar_er_kd_si_plus_unlearning",
        "poison_detector": POISON_DETECTOR,
        "h5_path": args.h5_path, "seed": args.seed, "num_tasks": NUM_TASKS,
        "task_fractions": TASK_FRACTIONS, "task_test_frac": TASK_TEST_FRAC,
        "poison_fraction": POISON_FRACTION, "require_evasion_success": REQUIRE_EVASION_SUCCESS,
        "red_epsilon": RED_EPSILON, "red_max_steps": RED_MAX_STEPS,
        "red_timesteps_per_task": RED_TIMESTEPS_PER_TASK, "alpha_contrast": ALPHA_CONTRAST,
        "contrastive_ema": CONTRASTIVE_EMA, "contrastive_recency_decay": CONTRASTIVE_RECENCY_DECAY,
        "max_eval_samples_per_task": MAX_EVAL_SAMPLES_PER_TASK, "feature_dim": feature_dim,
        "day_mapping": day_mapping,
        "mem_size": MEM_SIZE, "buffer_strategy": "uniform_50_50",
        "madar_contamination": MADAR_CONTAMINATION, "kd_temp": KD_TEMP,
        "si_c": SI_C, "si_eps": SI_EPS, "rnt_floor": RNT_FLOOR, "task0_epochs": TASK0_EPOCHS,
        "cl_iters": CL_ITERS, "batch_size": BATCH_SIZE, "feature_clip": FEATURE_CLIP,
        "unlearn_epochs": UNLEARN_EPOCHS, "unlearn_lr": UNLEARN_LR, "unlearn_alpha": UNLEARN_ALPHA,
    }
    with open(os.path.join(out_dir, f"{args.log_name}.json"), "w") as f:
        json.dump({"config": config, "warnings": warnings_log, "results": results}, f, indent=2)

    plot_task_metrics(results, os.path.join(out_dir, "plots", "task_metrics.png"))
    plot_unlearning_metrics(results, os.path.join(out_dir, "plots", "unlearning_metrics.png"))
    plot_prototype_heatmap(bank, os.path.join(out_dir, "plots", "prototype_heatmap.png"))
    plot_episode_clouds(bank, os.path.join(out_dir, "plots", "episode_clouds.png"))

    print(f"\nDone. Results + plots written to {out_dir}/")
    print("full time elapsed: %.2f seconds" % (time.perf_counter() - start_time))


if __name__ == "__main__":
    main()
