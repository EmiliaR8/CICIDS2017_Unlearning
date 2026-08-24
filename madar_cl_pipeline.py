"""
madar_cl_pipeline.py

Usage:
    python madar_cl_pipeline.py --seed 42 --log_name madar_10task_run \\
        --h5-path /mnt/processed_data/subsampled_dataset.h5

MADAR (ER + Knowledge Distillation + Synaptic Intelligence, buffer selection via
IsolationForest over the classifier's latent space) counterpart to naive_cl_pipeline.py
/ joint_cl_pipeline.py, over the SAME pooled/chronological 10-task split of
subsampled_dataset.h5 / full_dataset.h5, per the project's phasing plan
(naive -> joint -> MADAR -> MADAR+Unlearning).

Task construction, the red agent (SAC against a snapshot of the blue classifier at
the end of the previous task), and the poisoning mechanic are IDENTICAL to
naive_cl_pipeline.py / joint_cl_pipeline.py -- copied verbatim, not reimplemented --
so results are directly comparable. See those files' docstrings for the full
rationale on task windowing and red-agent retraining cadence.

THE THING THAT DIFFERS is the blue classifier and its training strategy:

  - Naive: XGBoostIDSWrapper, sequential fine-tune only (no replay).
  - Joint: XGBoostIDSWrapper, fresh full retrain on the pool every task boundary.
  - MADAR (this file): a from-scratch PyTorch MLP (ClassifierNN), trained with
    Experience Replay + Knowledge Distillation + Synaptic Intelligence against a
    bounded exemplar buffer selected by IsolationForest over the model's latent
    embedding space -- see clmd_cicids_madar_cd.py (Static_Data_Pipeline/) and
    the EMBER reference (clmd_ember18_mk9_cd.py) this is adapted from.

Why a new classifier instead of reusing XGBoostIDSWrapper: MADAR's mechanism is
built entirely around a gradient-trained network with a differentiable latent
space (IsolationForest selects buffer exemplars from that latent space; KD
distills softmax logits; SI regularizes per-parameter drift via gradients). None
of that exists for a tree ensemble. adversary_env.NetworkAttackEnv only calls
classifier.predict_proba(X) (confirmed by inspection), so TorchIDSWrapper below
satisfies that contract without touching adversary_env.py, evaluate_classifier(),
or the red-agent code at all -- only the object passed in as `classifier` changes.

Buffer strategy (confirmed during planning): UNIFORM 50/50 budget split across the
two label groups (benign / malicious), not proportional to observed class ratio.
The measured malicious-train fraction per task in this dataset swings from 1.2%
(task 6) to 96.8% (task 3) -- a ratio-proportional buffer would starve whichever
class is momentarily rare, exactly when a replay buffer is most needed to protect
it from forgetting. MEM_SIZE=4000 (2000/label) keeps the buffer to roughly 2-3% of
the final pooled dataset size (~187k rows by task 9), enough for IsolationForest's
anomaly/inlier selection to matter without approaching "just keep everything".

Since CICIDS is a fixed 2-class problem (benign/malicious present in every task
from task 0 onward -- unlike EMBER's growing malware-family set), the
active_count/prev_active_count output-masking machinery in both reference scripts
is dead code here (num_classes is always 2, so the mask never actually masks
anything) and has been dropped rather than ported.

UPDATE (REWORK): test data IS now poisoned (POISON_TEST_DATA=True), reversing
the "test is never poisoned" convention this file originally shared with
naive/joint. Made specifically so this file's structure matches
madar_unlearning_cl_pipeline.py -- with test poisoning absent here but present
there, per_task_eval/pooled_eval/mean_per_task_balanced_accuracy were being
computed on genuinely different (easier vs. harder) test distributions between
the two pipelines, confounding any comparison between them. Also added to
match that file: global sample_id tracking (gid_train/gid_test,
poisoned_sample_ids/poisoned_test_sample_ids in results, sample_id carried in
the replay buffer) -- see update_buffer_madar's and buffer_composition_summary's
docstrings. The original reasoning below (evasion rate already measures
generalization; poisoning eval labels would conflate "evaded" with "ground
truth malicious") still holds for naive/joint, which this update does not
touch. The one MADAR-specific diagnostic kept from the original design is
the replay buffer's category composition (benign / malicious_clean /
malicious_perturbed per label slot) -- purely observational, computed from
ground truth already on hand, never fed back into selection or training --
added as an extra key on top of naive/joint's existing results schema so
compare_cl_runs.py keeps working unmodified.

Classifier runs on CPU (TRAIN_DEVICE below), matching the caution already on
record in this repo's naive/joint pipelines and the old CICIDS MADAR script:
"auto"/"cuda" has been observed to crash on this environment's torch build.

UPDATE (REWORK, dual red agents): each task now trains TWO red agents
instead of one -- red_train_pert_agent (env built directly from X_train,
only ever perturbs train data) and red_test_pert_agent (env built directly
from X_test, only ever perturbs test data, trained second). This fixes a
latent bug in the old single-agent design: NetworkAttackEnv.reset() always
samples from self.X_data, fixed at construction; the old code built ONE env
from X_train and reused it unmodified for the "test" evaluate_agent_on_batch
call (update_data() existed on the class but was never called), so what was
logged as test_evasion_rate/X_test_pert was actually attacking train rows at
test-derived indices, never real test data. agent_id is now a compound
"{task_id}_{agent_type}" string registered in the SAME contrastive bank, so
each agent's reward is pushed away from every previously-registered agent
(train and test, all earlier tasks) via the existing recency-weighted
mechanism -- train_t vs train_{t-1}, test_t vs test_{t-1}, and (since test
trains second, same task) test_t vs train_t, though that last one is
necessarily one-directional: train_t finishes training before test_t exists,
so train_t's reward never references test_t. Evasion success stays a hard
requirement independent of this diversity pressure: the +10 reward bonus for
successful misclassification dominates the reward magnitude, and
REQUIRE_EVASION_SUCCESS (unchanged) still means only genuinely-evasive
perturbations (evaded_mask=True) ever become poison candidates for either
pool. red_test_pert_agent only trains when POISON_TEST_DATA is on AND this
task actually has malicious test samples, to avoid training a whole second
SAC agent for nothing. Also: POISON_SCHEDULE_ENABLED flipped to False (flat
POISON_FRACTION_FLAT=0.30 for both train and test, every task) -- moving
away from the per-task schedule per explicit request; the schedule code
itself is untouched and can be re-enabled by flipping the toggle back.
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
# Top-of-file config. Task/red-agent constants kept identical in name/value to
# naive_cl_pipeline.py / joint_cl_pipeline.py wherever the two share behavior,
# so runs are directly comparable. Only --seed and --log_name are CLI.
# ---------------------------------------------------------------------------
H5_DATASET_PATH = "/mnt/processed_data/subsampled_dataset.h5"
RUNS_BASE_DIR = "/mnt/erivas6/runs"  # base directory for all run outputs (logs/plots/results json)

SEED = 42  # overwritten from --seed at the top of main(); module-level so
           # update_buffer_madar's IsolationForest(random_state=SEED) sees it.

NUM_TASKS = 10
# task0 = 30% warm start; tasks1-9 taper 9.18% -> 6.38% of the pooled dataset.
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]
TASK_TEST_FRAC = 0.20

REQUIRE_EVASION_SUCCESS = True  # only successfully-evasive perturbations poison training data
POISON_TEST_DATA = True  # REWORK: matches madar_unlearning_cl_pipeline.py's POISON_TEST_DATA
                          # (previously False/absent here -- test was never poisoned). Needed
                          # so per_task_eval/pooled_eval are computed on the SAME kind of data
                          # in both pipelines; see that file's module docstring BRANCH NOTE for
                          # the original rationale (evasion rate alone doesn't check whether
                          # detection/unlearning generalizes to evasive samples never trained
                          # on). Same evaded-mask-restricted substitution as train (schedule
                          # below), applied to each task's test split.

# Per-task poison injection schedule (REWORK, replaces the old flat
# POISON_FRACTION=0.3). Linear interpolation from task 1 to task NUM_TASKS-1
# (task 0 has no red agent, so no poisoning either way). TRAIN and TEST are
# deliberate mirror images of each other -- train front-loads poison (heavy
# early, tapering later), test back-loads it (light early, heavy later).
# REWORK 2: endpoints pushed to a much more drastic 0.99/0.01 split (was
# 0.50/0.10) -- crossing point is now ~0.50 around the midpoint task, no
# longer ~0.30. POISON_FRACTION_FLAT (below) was NOT changed to match --
# it's an independently-set fallback for the schedule-off case, not derived
# from the schedule's endpoints, so it no longer coincides with the
# schedule's midpoint the way it used to. Plain global variables (not
# derived/computed) specifically so they're easy to change later -- see
# poison_fraction_for_task() below.
POISON_SCHEDULE_ENABLED = False  # REWORK 3: off by default now -- moving away from the
                                 # per-task schedule in favor of a flat rate, per explicit
                                 # request, alongside the dual train/test red-agent rework
                                 # below. toggle: True = the linear train/test schedule
                                 # below. False = flat POISON_FRACTION_FLAT for BOTH train
                                 # and test, every task (current setting).
POISON_FRACTION_FLAT = 0.30  # fallback fraction (both train and test) when
                              # POISON_SCHEDULE_ENABLED is False -- i.e. the ACTIVE rate
                              # right now. Originally matched the schedule's own
                              # midpoint/crossing value; no longer does since REWORK 2
                              # moved the schedule's endpoints (see above).
POISON_FRACTION_TRAIN_START = 0.99  # task 1
POISON_FRACTION_TRAIN_END = 0.01    # task NUM_TASKS-1
POISON_FRACTION_TEST_START = 0.01   # task 1
POISON_FRACTION_TEST_END = 0.99     # task NUM_TASKS-1


def poison_fraction_for_task(t, start, end, num_tasks=NUM_TASKS):
    """Linear interpolation from `start` (task 1) to `end` (task num_tasks-1).
    t=0 is never called (task 0 has no red agent / no poisoning). Returns
    POISON_FRACTION_FLAT instead, ignoring start/end, whenever
    POISON_SCHEDULE_ENABLED is False."""
    if not POISON_SCHEDULE_ENABLED:
        return POISON_FRACTION_FLAT
    if num_tasks <= 2:
        return start
    frac = (t - 1) / (num_tasks - 2)
    return start + frac * (end - start)

RED_EPSILON = 0.25
RED_MAX_STEPS = 25
RED_TIMESTEPS_PER_TASK = 2500
ALPHA_CONTRAST = 0.5
CONTRASTIVE_EMA = 0.95
CONTRASTIVE_RECENCY_DECAY = 0.5  # weight of task (k-1) vs (k-2) vs ... in the diversity reward

RED_TARGET_MARGIN_CONFIDENCE = 0.55  # REWORK (margin-minimizing evasion): passed to every
                                      # NetworkAttackEnv (both train_pert_agent and
                                      # test_pert_agent) so their reward peaks at just-barely-
                                      # evasive confidence instead of rewarding maximal
                                      # confidence deep into the target class. See
                                      # NetworkAttackEnv.step()'s confidence_term for the exact
                                      # shape. Rationale: plain MADAR's buffer selection
                                      # (IsolationForest anomaly/inlier over the latent
                                      # embedding) has no notion of margin/decision-boundary
                                      # distance, so margin-hugging poison is a blind spot for
                                      # it specifically, unlike MADAR+Unlearning's
                                      # perturbation_classifier (an explicitly boundary-adjacent
                                      # detector). Set to None to restore the original
                                      # confidence-maximizing reward for both agents.

# Caps evaluate_agent_on_batch's per-task episode count for runtime; None = every malicious sample.
MAX_EVAL_SAMPLES_PER_TASK = 5000

SAC_POLICY_KWARGS = dict(net_arch=[512, 256, 128], optimizer_kwargs={"weight_decay": 1e-4})
SAC_KWARGS = dict(
    learning_rate=1e-3, buffer_size=50_000, batch_size=128, gamma=0.95,
    ent_coef="auto_0.01", device="auto",
)

# --- MADAR classifier / CL hyperparameters ---
TRAIN_DEVICE = "cpu"
DEVICE = torch.device(TRAIN_DEVICE)

FEATURE_CLIP = 10.0     # clip standardized features to [-FEATURE_CLIP, FEATURE_CLIP]
TASK0_EPOCHS = 30       # task 0 is pure supervised (no replay yet) -- epoch-based
CL_ITERS = 2000         # tasks 1+ mix current-task and replay-buffer batches via two
                         # independently-cycling loaders, so a fixed iteration count is
                         # used instead of "epochs over the current task's own loader"
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

MEM_SIZE = 4000          # total replay buffer budget, split UNIFORM 50/50 across
                          # {benign, malicious} regardless of observed class ratio
                          # (see module docstring for why ratio-based was rejected)
MADAR_CONTAMINATION = 0.1  # only shifts IsolationForest's decision threshold; selection
                            # is purely rank-based on decision_function scores, so this
                            # does not affect which samples get chosen (kept for API compat)

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1
# rnt = max(1/(tid+1), RNT_FLOOR): weight of current-task hard-label loss vs.
# replay KD loss (see train_cl_er). Unfloored 1/(tid+1) (borrowed from the
# EMBER class-incremental reference, where "new" each round is a handful of
# brand-new classes on top of a large stable set) decays to ~0.1 by task 9,
# which -- in this fixed 2-class, concept-drift setting where every task can
# be a large, informative new slice of malicious traffic -- suppresses
# learning of new tasks rather than protecting old ones (confirmed via
# madar1.json: task 0's held-out accuracy barely moves, 99.4%->97.8% over 9
# more tasks, while tasks 8/9's OWN fresh-eval accuracy collapses to ~0.52
# despite abundant current-task data, tracking rnt's decay almost exactly).
# Floored at 0.3 so current-task loss never drops below ~3x its unfloored
# tail-end weight, while leaving the early-task decay (task 1: 0.5, task 2:
# 0.33) that's still above the floor untouched.
RNT_FLOOR = 0.3


# ---------------------------------------------------------------------------
# Recency-weighted contrastive diversity (identical to naive_cl_pipeline.py)
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
# Red agent training / evaluation (identical to naive_cl_pipeline.py -- the red
# agent's retraining cadence and poisoning mechanic don't depend on which blue
# CL strategy is in use, per the project's phasing plan).
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
    the exact set poisoning should draw from.
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


def train_red_agent_for_task(task_id, agent_type, classifier, X_data, y_data, benign_label, bank, seed, out_dir,
                              target_margin_confidence=None):
    """
    REWORK: one call per (task, agent_type) pair now, not one call per task.
    agent_type in {"train", "test"} -- the caller passes X_data/y_data as
    whichever split this agent is dedicated to (X_train for the train-side
    perturbation agent, X_test for the test-side one), and this builds a
    NetworkAttackEnv directly over THAT data. Fixes a latent bug: previously
    a single agent's env was built once from X_train, then reused unmodified
    for the "test" evaluate_agent_on_batch call -- NetworkAttackEnv.reset()
    always samples from self.X_data (fixed at construction, update_data()
    existed but was never called), so that second call was actually
    attacking train rows at test-derived indices, not test data at all.

    agent_id is now a compound "{task_id}_{agent_type}" string (was task_id
    alone) so the contrastive bank tracks train/test agents as fully
    distinct entries -- train_t's reward pushes it away from every agent
    registered before it (..., test_{t-1}, train_{t-1}, ...), and test_t's
    reward (trained second, see main()) additionally sees train_t's
    now-registered prototype, giving same-task train/test dissimilarity.
    Distinct seed offset per agent_type so the two agents' SAC init/training
    RNG streams don't collide within the same task.

    target_margin_confidence (REWORK, margin-minimizing evasion): forwarded
    straight to NetworkAttackEnv -- None (default) keeps the original
    confidence-maximizing reward; a value like RED_TARGET_MARGIN_CONFIDENCE
    switches both train and test agents to reward landing just past the
    decision boundary instead of deep past it. See that constant's
    definition and NetworkAttackEnv.step() for the full rationale.
    """
    seed_offset = 0 if agent_type == "train" else 100_000
    agent_id = f"{task_id}_{agent_type}"

    def _thunk():
        return NetworkAttackEnv(
            classifier, X_data, y_data,
            benign_label=benign_label,
            max_steps=RED_MAX_STEPS,
            epsilon=RED_EPSILON,
            agent_id=agent_id,
            contrastive_bank=bank,
            alpha_contrast=ALPHA_CONTRAST,
            target_margin_confidence=target_margin_confidence,
        )

    vec_env = make_vec_env(_thunk, n_envs=1, seed=seed + task_id + seed_offset)
    agent = SAC(
        "MlpPolicy", vec_env, verbose=0, policy_kwargs=SAC_POLICY_KWARGS,
        seed=seed + task_id + seed_offset, **SAC_KWARGS,
    )
    timer_cb = EpisodeTimerCallback(
        log_path=os.path.join(out_dir, "logs", f"red_episode_times_task{task_id}_{agent_type}.txt")
    )
    agent.learn(total_timesteps=RED_TIMESTEPS_PER_TASK, callback=timer_cb)
    return vec_env.envs[0], agent


# ---------------------------------------------------------------------------
# Blue classifier: PyTorch MLP with a latent layer (needed for MADAR-IF buffer
# selection) + KD/SI machinery, replacing XGBoostIDSWrapper for this pipeline
# only. See module docstring for why.
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
    unchanged. Takes RAW ([0,1]-clipped, unscaled) features -- exactly what
    naive/joint pass around and what the red agent's epsilon ball is calibrated
    to -- and applies the task-0-fit StandardScaler + FEATURE_CLIP internally,
    mirroring to_tensor() in main(). `model` is held by reference and mutated in
    place by training, so this wrapper's output always reflects the classifier's
    current state without needing to be reconstructed, exactly like naive/joint
    pass their (also mutable) XGBoostIDSWrapper instance directly.
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
# MADAR replay buffer (keyed by binary label, UNIFORM budget across the two
# groups -- see module docstring for why ratio-based budgeting was rejected).
# ---------------------------------------------------------------------------
label_buffers = {}   # {0: [(X_scaled, y, category, sample_id), ...], 1: [...]}
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


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, sample_id: np.ndarray,
                         benign_label, mal_label, model, device):
    """
    Selects, per label (benign/malicious), an interleaved anomalies+inliers sample
    (by IsolationForest decision_function over the model's LATENT space, budget
    fixed at MEM_SIZE // 2 for both groups regardless of observed class ratio) to
    keep in that label's buffer slot. Re-ranks the UNION of that label's EXISTING
    buffer entries (re-embedded under the CURRENT model, since weights have moved
    since they were last selected) and this task's new same-label samples, so
    exemplars from earlier tasks can survive across updates. `category` is
    attached to each stored sample purely for the buffer_composition_summary()
    diagnostic -- it plays no role in selection, which sees only latent vectors.
    `sample_id` is the GLOBAL pool row index each sample was assigned in main()
    (stable across the whole run, survives train/test splitting and poisoning
    substitution) -- carried alongside category purely for traceability (tSNE
    plots, cross-task comparison of which physical samples persist in the
    buffer), also not used by selection. Structure matches
    madar_unlearning_cl_pipeline.py's update_buffer_madar exactly, so buffer
    contents are directly comparable between the two pipelines.
    """
    global label_buffers, replay_buffer
    model.eval()

    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)
    id_np = np.asarray(sample_id)
    L_np = _embed(model, X, device)
    latent_dim = L_np.shape[1]

    budget_per_label = MEM_SIZE // 2

    for lbl in (benign_label, mal_label):
        old_entries = label_buffers.get(lbl, [])
        if old_entries:
            old_X = np.stack([e[0].numpy() for e in old_entries])
            old_Y = np.array([e[1].item() for e in old_entries], dtype=Y_np.dtype)
            old_cat = np.array([e[2] for e in old_entries], dtype=object)
            old_id = np.array([e[3] for e in old_entries], dtype=id_np.dtype)
            old_L = _embed(model, torch.tensor(old_X), device)
        else:
            old_X = np.empty((0, X_np.shape[1]), dtype=X_np.dtype)
            old_Y = np.empty((0,), dtype=Y_np.dtype)
            old_cat = np.empty((0,), dtype=object)
            old_id = np.empty((0,), dtype=id_np.dtype)
            old_L = np.empty((0, latent_dim), dtype=np.float32)

        mask = (Y_np == lbl)
        pool_X = np.concatenate([old_X, X_np[mask]], axis=0)
        pool_Y = np.concatenate([old_Y, Y_np[mask]], axis=0)
        pool_L = np.concatenate([old_L, L_np[mask]], axis=0)
        pool_cat = np.concatenate([old_cat, cat_np[mask]], axis=0)
        pool_id = np.concatenate([old_id, id_np[mask]], axis=0)

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
            (torch.tensor(pool_X[i]), torch.tensor(pool_Y[i]), pool_cat[i], int(pool_id[i]))
            for i in interleaved_idx
        ]

    replay_buffer.clear()
    for buf in label_buffers.values():
        replay_buffer.extend(buf)


def buffer_composition_summary():
    """Diagnostic only: what fraction of each label's buffer slot is benign /
    malicious_clean / malicious_perturbed right now, plus the GLOBAL sample_id
    of every entry so buffer membership can be cross-referenced against
    poisoned_sample_ids (per-task, in results) or reloaded from the pooled
    dataset for tSNE later -- see update_buffer_madar's docstring for what
    sample_id means. Never used by training or selection."""
    summary = {}
    for lbl, entries in label_buffers.items():
        cats = [e[2] for e in entries]
        counts = {}
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
        summary[str(lbl)] = {
            "total": len(entries),
            "category_counts": counts,
            "sample_ids": [e[3] for e in entries],
        }
    return summary


def train_cl_er(model, teacher_model, optimizer, loader, iters, W, omega, p_old_task, tid, si_c, device):
    """
    One task's CL training: each step mixes a batch from the current task's own
    loader with a batch from the (previous-tasks-only) replay buffer. Current-task
    samples get plain cross-entropy; buffer samples get KD against the frozen
    teacher (the model as it stood at the end of the previous task) instead of
    their stored hard labels, so the model doesn't just re-memorize the buffer's
    literal contents. `rnt` shifts weight from current-task loss toward replay KD
    as tasks accumulate (1/(tid+1), floored at RNT_FLOOR -- see its definition
    above for why the unfloored schedule was suppressing new-task learning).
    SI adds a per-parameter quadratic penalty
    against drifting from p_old_task, weighted by omega (importance accumulated
    over all previous tasks' training).

    Unlike the EMBER lineage this is adapted from, there's no active_count output
    masking: CICIDS is a fixed 2-class problem from task 0 onward, so the mask
    would never mask anything -- dropped rather than carried as dead code.
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
# Blue classifier evaluation (identical schema to naive/joint's evaluate_classifier
# -- balanced_accuracy as a 0-1 fraction, same keys -- so compare_cl_runs.py works
# on this pipeline's JSON unmodified).
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
# Plots (identical to naive_cl_pipeline.py, titles relabeled MADAR)
# ---------------------------------------------------------------------------
def plot_task_metrics(results, out_path):
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc = [r["mean_per_task_balanced_accuracy"] for r in results]
    train_evasion = [r["red_agent"]["train_evasion_rate"] if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"]["test_evasion_rate"] if r["red_agent"] else None for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc, marker="o", label="pooled balanced accuracy")
    ax1.plot(task_ids, mean_per_task_bal_acc, marker="s", label="mean per-task balanced accuracy")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("MADAR CL (ER+KD+SI): classifier accuracy per task boundary")
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


# agent_ids are "{task_id}_{agent_type}" compound strings, agent_type in
# {"train", "test", "train_benign", "test_benign"} -- split on the FIRST
# underscore only (agent_type itself would contain one if benign-side
# agents were ever added here), and sort/mark deterministically rather than
# relying on alphabetical order.
AGENT_TYPE_ORDER = {"train": 0, "test": 1, "train_benign": 2, "test_benign": 3}
AGENT_TYPE_MARKERS = {"train": "o", "test": "^", "train_benign": "s", "test_benign": "D"}


def _parse_agent_id(agent_id):
    task_num, agent_type = agent_id.split("_", 1)
    return int(task_num), agent_type


def _agent_id_sort_key(agent_id):
    task_num, agent_type = _parse_agent_id(agent_id)
    return (task_num, AGENT_TYPE_ORDER.get(agent_type, 99))


def plot_prototype_heatmap(bank, out_path):
    ordered_ids = sorted(bank.protos.keys(), key=_agent_id_sort_key)
    agent_ids, S = bank.cosine_matrix(agent_ids=ordered_ids)
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
    task_ids = sorted(bank.episode_embs.keys(), key=_agent_id_sort_key)
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
        task_num, agent_type = _parse_agent_id(tid)
        marker = AGENT_TYPE_MARKERS.get(agent_type, "x")
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.6, color=cmap(task_num % 10),
                    marker=marker, label=f"task {tid}")

    plt.title("Episode embedding clouds (PCA-2D of cumulative perturbations)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Decision-boundary visualization. Evaluates the CLASSIFIER itself over a
# dense grid -- an actual decision surface, not a data embedding -- on a
# FIXED run-wide 2D PCA plane (pca_mean/pca_components/pca_extent, computed
# once in main() right after the scaler is fit at task 0, over ALL tasks'
# pooled scaled features), reused unchanged for every subsequent
# task/checkpoint so panels are directly comparable -- a fresh per-call PCA
# would make the axes mean something different every panel, defeating the
# point of watching the boundary move. agent_type in {"train", "test",
# "train_benign", "test_benign"} is reused as the color+marker key for
# perturbed samples -- this file only ever populates "train"/"test" (no
# benign-side agents), but the dict covers all four for portability.
# ---------------------------------------------------------------------------
AGENT_PERTURBED_COLORS = {
    "train": "tab:red", "test": "firebrick",
    "train_benign": "tab:purple", "test_benign": "indigo",
}
CATEGORY_BG_COLORS = {"benign": "tab:blue", "malicious_clean": "tab:orange"}


def _pca_project(X_scaled_np, mean, components):
    return (X_scaled_np - mean) @ components.T


def _decision_grid(model, mean, components, extent, device, mal_label, resolution=150):
    """Dense grid over the FIXED 2D PCA plane, inverse-transformed back into
    the model's actual input space, evaluated through `model` for real --
    returns (xx, yy, proba_malicious) for contour plotting."""
    xmin, xmax, ymin, ymax = extent
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_highdim = grid_2d @ components + mean

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(grid_highdim, dtype=torch.float32, device=device))
        proba = F.softmax(logits, dim=1)[:, mal_label].cpu().numpy()
    return xx, yy, proba.reshape(xx.shape)


def _draw_decision_boundary(ax, xx, yy, proba, Z_train, category_train, Z_test, category_test):
    """Draws one panel from PRECOMPUTED grid/scatter data onto `ax`. Shared
    by plot_decision_boundary (computes then draws, saves its own file) and
    plot_decision_boundary_grid (redraws every checkpoint's cached data into
    one summary figure). Background categories (benign/malicious_clean) are
    TRAIN-split only, small and faint; perturbed categories are drawn from
    BOTH train and test splits, colored+marker-coded by which agent
    produced them. Returns per-agent perturbed counts (for the caller's
    legend)."""
    ax.contourf(xx, yy, proba, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=1.2, linestyles="--")

    for cat, color in CATEGORY_BG_COLORS.items():
        mask = category_train == cat
        if mask.sum() > 0:
            ax.scatter(Z_train[mask, 0], Z_train[mask, 1], s=5, alpha=0.2, color=color, marker=".")

    n_by_agent = {}
    for cat, agent_type, Z, cat_arr in [
        ("malicious_perturbed", "train", Z_train, category_train),
        ("malicious_perturbed", "test", Z_test, category_test),
        ("benign_perturbed", "train_benign", Z_train, category_train),
        ("benign_perturbed", "test_benign", Z_test, category_test),
    ]:
        mask = cat_arr == cat
        n_by_agent[agent_type] = int(mask.sum())
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=22, alpha=0.85,
                   color=AGENT_PERTURBED_COLORS[agent_type], marker=AGENT_TYPE_MARKERS[agent_type],
                   edgecolors="black", linewidths=0.3)
    return n_by_agent


def plot_decision_boundary(model, device, pca_mean, pca_components, pca_extent, mal_label,
                            X_train_scaled_np, category_train,
                            X_test_scaled_np, category_test,
                            task_id, out_path, checkpoint_label=""):
    """One task/checkpoint's decision-boundary figure. Returns a dict of the
    computed grid/scatter data so main() can cache it and reuse it in
    plot_decision_boundary_grid's end-of-run summary without recomputing."""
    xx, yy, proba = _decision_grid(model, pca_mean, pca_components, pca_extent, device, mal_label)
    Z_train = _pca_project(X_train_scaled_np, pca_mean, pca_components)
    Z_test = _pca_project(X_test_scaled_np, pca_mean, pca_components)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_by_agent = _draw_decision_boundary(ax, xx, yy, proba, Z_train, category_train, Z_test, category_test)

    handles = [
        plt.Line2D([0], [0], marker=".", color=c, linestyle="", markersize=8,
                   label=f"{cat} (n={int((category_train == cat).sum())})")
        for cat, c in CATEGORY_BG_COLORS.items()
    ] + [
        plt.Line2D([0], [0], marker=AGENT_TYPE_MARKERS[agent], color=AGENT_PERTURBED_COLORS[agent],
                   linestyle="", markersize=8, label=f"perturbed via {agent} (n={n_by_agent.get(agent, 0)})")
        for agent in ("train", "test", "train_benign", "test_benign") if n_by_agent.get(agent, 0) > 0
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")

    title = f"Task {task_id} decision boundary"
    if checkpoint_label:
        title += f" ({checkpoint_label})"
    ax.set_title(title)
    ax.set_xlabel("PC1 (fixed, run-wide)")
    ax.set_ylabel("PC2 (fixed, run-wide)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "xx": xx, "yy": yy, "proba": proba,
        "Z_train": Z_train, "category_train": category_train,
        "Z_test": Z_test, "category_test": category_test,
        "task_id": task_id, "checkpoint_label": checkpoint_label,
    }


def plot_decision_boundary_grid(panels, out_path):
    """panels: list of dicts as returned by plot_decision_boundary, one per
    captured checkpoint across the whole run -- laid out as a grid of
    subplots (same fixed PCA basis, so directly comparable) for an
    at-a-glance view of how the boundary drifts as poisoning accumulates."""
    n = len(panels)
    if n == 0:
        print("[Plot] No decision-boundary panels captured -- skipping summary grid.")
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for i, p in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        _draw_decision_boundary(ax, p["xx"], p["yy"], p["proba"], p["Z_train"], p["category_train"],
                                 p["Z_test"], p["category_test"])
        label = f"Task {p['task_id']}" + (f" ({p['checkpoint_label']})" if p["checkpoint_label"] else "")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle("Decision boundary evolution across the run (fixed PCA basis)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_cl_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    args = ap.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join(RUNS_BASE_DIR, "madar", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    print(f"Loading {args.h5_path} and building {NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(args.h5_path, TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    # GLOBAL sample ids: task t's rows are a contiguous slice of the pooled,
    # timestamp-sorted array load_pooled_chronological_tasks builds internally
    # (see that function -- tasks[t] = pool[starts[t]:starts[t]+len]). task_offsets[t]
    # reconstructs starts[t] purely from each task's row count, so
    # task_offsets[t] + (a row's position within task t's pool, BEFORE the
    # train/test split below) is a stable identifier for that exact physical
    # row -- same value every run given the same h5 file/task_fractions,
    # independent of which task's buffer or forget/retain set it ends up in.
    # Logged per-task (poisoned_sample_ids) and per-buffer-entry
    # (buffer_composition_summary) purely for traceability -- e.g. reloading
    # the pooled dataset by id later for tSNE -- never used by training or
    # selection. Matches madar_unlearning_cl_pipeline.py's tracking exactly.
    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    bank = RecencyWeightedContrastiveBank(
        dim=feature_dim, ema=CONTRASTIVE_EMA, recency_decay=CONTRASTIVE_RECENCY_DECAY
    )
    # REWORK: agent_ids are now "{task_id}_train"/"{task_id}_test" compound
    # strings (one train + one test agent per task), not bare task ids -- see
    # train_red_agent_for_task's docstring. Chronological registration order
    # (train before test, same task) fixed explicitly here rather than left
    # to sorted(), since string-sorting "0_test" before "0_train" would
    # misorder the log/plots relative to when each agent actually trained.
    bank.init_distance_logger(
        path=os.path.join(out_dir, "logs", "prototype_cosine_over_time.txt"),
        agent_ids=[f"{t}_{typ}" for t in range(NUM_TASKS) for typ in ("train", "test")],
    )

    # Classifier, scaler, SI bookkeeping: all lazily created at task 0, once
    # feature_dim/X_train are known. Persistent across every task -- MADAR fine
    # -tunes ONE model forward (like naive), it just protects earlier tasks via
    # the ER/KD/SI machinery instead of never revisiting them.
    scaler = None
    model = None
    classifier_wrapper = None
    teacher_model = None
    W = omega = p_old_task = None

    def to_tensor(X_raw):
        X_scaled = np.clip(scaler.transform(X_raw.astype(np.float32)), -FEATURE_CLIP, FEATURE_CLIP)
        return torch.tensor(X_scaled, dtype=torch.float32)

    task_test_splits = {}
    task_test_gids = {}  # t -> gid_test, so later checkpoints can identify which rows in
                          # task_test_splits[t] are the poisoned ones (see task_poisoned_test_gids)
    task_poisoned_test_gids = {}  # t -> that task's own poisoned_test_sample_ids, recorded once
                                   # per task (immutable afterward) -- feeds perturbed_test_eval
    results = []
    warnings_log = []
    boundary_panels = []  # accumulates plot_decision_boundary's return dict per captured
                           # checkpoint, for the end-of-run summary grid (plot_decision_boundary_grid)
    pca_mean = pca_components = pca_extent = None  # set once at task 0, below

    for t, task in enumerate(tasks):
        X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)  # data is documented [0,1]-scaled already
        y = task["labels"].astype(np.int64)
        gid = task_offsets[t] + np.arange(len(y), dtype=np.int64)  # this task's pool-local id -> global id

        X_train, X_test, y_train, y_test, gid_train, gid_test = train_test_split(
            X, y, gid, test_size=TASK_TEST_FRAC, random_state=args.seed, stratify=y
        )
        task_test_splits[t] = (X_test, y_test)
        task_test_gids[t] = gid_test

        mal_idx_train = np.where(y_train == mal_label)[0]
        red_report = None
        n_poisoned = 0
        poison_idx = np.array([], dtype=int)
        n_poisoned_test = 0
        poison_idx_test = np.array([], dtype=int)  # initialized here (not just inside the nested
                                                     # TEST-SIDE block) so category_test construction
                                                     # below can always safely reference it
        poisoned_test_sample_ids = []

        if t == 0:
            print(f"\n===== Task 0 (warm start, no red agent): "
                  f"{len(y_train)} train / {len(y_test)} test =====")
            X_train_for_classifier = X_train

        else:
            print(f"\n===== Task {t}: training fresh red agent against pre-task-{t} classifier "
                  f"({len(y_train)} train / {len(y_test)} test, {len(mal_idx_train)} malicious train) =====")

            mal_idx_test = np.where(y_test == mal_label)[0]

            if len(mal_idx_train) == 0:
                warnings_log.append(f"Task {t}: no malicious training samples, skipping red agents.")
                X_train_for_classifier = X_train
            else:
                # REWORK: two separate agents per task -- red_train_pert_agent
                # trains directly against X_train and only ever perturbs train
                # data; red_test_pert_agent trains second (so its contrastive
                # reward can reference train's now-registered prototype) and
                # only ever perturbs test data. See train_red_agent_for_task's
                # docstring for why this also fixes a latent env/data mismatch
                # bug in the old single-agent design.
                env_train, agent_train = train_red_agent_for_task(
                    t, "train", classifier_wrapper, X_train, y_train, benign_label, bank, args.seed, out_dir,
                    target_margin_confidence=RED_TARGET_MARGIN_CONFIDENCE,
                )
                X_train_pert, train_evasion_rate, train_rewards, train_avg_norm, train_attacked, evaded_mask = \
                    evaluate_agent_on_batch(
                        env_train, agent_train, X_train, y_train, benign_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                    )

                red_report = {
                    "train_evasion_rate": train_evasion_rate, "train_attacked": train_attacked,
                    "train_avg_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
                    "train_avg_pert_l2": train_avg_norm,
                }
                print(f"[Task {t} red_train_pert_agent] train_evasion={train_evasion_rate:.3f} "
                      f"avg_pert_L2={train_avg_norm:.3f}")

                poison_fraction_train = poison_fraction_for_task(
                    t, POISON_FRACTION_TRAIN_START, POISON_FRACTION_TRAIN_END
                )
                poison_pool = np.where(evaded_mask)[0] if REQUIRE_EVASION_SUCCESS else mal_idx_train
                n_to_poison = min(int(round(poison_fraction_train * len(mal_idx_train))), len(poison_pool))
                rng = np.random.RandomState(args.seed + t)
                poison_idx = rng.choice(poison_pool, size=n_to_poison, replace=False) if n_to_poison > 0 else \
                    np.array([], dtype=int)
                n_poisoned = len(poison_idx)

                X_train_for_classifier = X_train.copy()
                X_train_for_classifier[poison_idx] = X_train_pert[poison_idx]
                print(f"[Task {t}] poisoned {n_poisoned}/{len(mal_idx_train)} malicious train samples "
                      f"(poison_fraction_train={poison_fraction_train:.3f})")

                # TEST-SIDE poisoning (POISON_TEST_DATA -- see module docstring
                # BRANCH NOTE and POISON_TEST_DATA's definition for why).
                # red_test_pert_agent (its own dedicated agent, see above) is
                # only trained when there's actually test-side poisoning to do,
                # to avoid the compute cost otherwise. Overwrites
                # task_test_splits[t] so every downstream eval (per_task_eval,
                # pooled_eval) for this task sees the poisoned test set from
                # here on.
                if POISON_TEST_DATA and len(mal_idx_test) > 0:
                    env_test, agent_test = train_red_agent_for_task(
                        t, "test", classifier_wrapper, X_test, y_test, benign_label, bank, args.seed, out_dir,
                        target_margin_confidence=RED_TARGET_MARGIN_CONFIDENCE,
                    )
                    X_test_pert, test_evasion_rate, test_rewards, test_avg_norm, test_attacked, evaded_mask_test = \
                        evaluate_agent_on_batch(
                            env_test, agent_test, X_test, y_test, benign_label,
                            only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                        )
                    red_report["test_evasion_rate"] = test_evasion_rate
                    red_report["test_attacked"] = test_attacked
                    red_report["test_avg_reward"] = float(np.mean(test_rewards)) if test_rewards else 0.0
                    red_report["test_avg_pert_l2"] = test_avg_norm
                    print(f"[Task {t} red_test_pert_agent] test_evasion={test_evasion_rate:.3f} "
                          f"avg_pert_L2={test_avg_norm:.3f}")

                    poison_fraction_test = poison_fraction_for_task(
                        t, POISON_FRACTION_TEST_START, POISON_FRACTION_TEST_END
                    )
                    poison_pool_test = np.where(evaded_mask_test)[0] if REQUIRE_EVASION_SUCCESS else mal_idx_test
                    n_to_poison_test = min(int(round(poison_fraction_test * len(mal_idx_test))), len(poison_pool_test))
                    rng_test = np.random.RandomState(args.seed + t + 10_000)  # distinct stream from train's rng
                    poison_idx_test = rng_test.choice(poison_pool_test, size=n_to_poison_test, replace=False) \
                        if n_to_poison_test > 0 else np.array([], dtype=int)
                    n_poisoned_test = len(poison_idx_test)

                    X_test = X_test.copy()
                    X_test[poison_idx_test] = X_test_pert[poison_idx_test]
                    task_test_splits[t] = (X_test, y_test)
                    poisoned_test_sample_ids = gid_test[poison_idx_test].tolist() if n_poisoned_test > 0 else []
                    print(f"[Task {t}] poisoned {n_poisoned_test}/{len(mal_idx_test)} malicious TEST samples "
                          f"(poison_fraction_test={poison_fraction_test:.3f}) -- "
                          f"test_evasion_rate was {test_evasion_rate:.3f}")
                else:
                    n_poisoned_test = 0
                    poisoned_test_sample_ids = []

        # Category tracking for the buffer-composition diagnostic ONLY (never fed
        # back into training/selection) -- benign / malicious_clean /
        # malicious_perturbed, built from TRAIN data only; test poisoning above
        # never touches training/selection.
        category = np.full(len(y_train), "malicious_clean", dtype=object)
        category[y_train == benign_label] = "benign"
        if len(poison_idx) > 0:
            category[poison_idx] = "malicious_perturbed"
        poisoned_sample_ids = gid_train[poison_idx].tolist() if len(poison_idx) > 0 else []

        # TEST-side mirror of the category array above -- feeds
        # plot_decision_boundary's test-split scatter (poison_idx_test rows
        # are exactly what red_test_pert_agent produced).
        category_test = np.full(len(y_test), "malicious_clean", dtype=object)
        category_test[y_test == benign_label] = "benign"
        if len(poison_idx_test) > 0:
            category_test[poison_idx_test] = "malicious_perturbed"

        if t == 0:
            # Scaler fit on TASK 0 (pre-poisoning, but task 0 has no red agent
            # anyway) training data only, matching both reference scripts'
            # convention. FEATURE_CLIP guards against later tasks' families/
            # traffic patterns producing extreme z-scores under a task-0-only
            # scaler, which destabilizes Adam without gradient clipping.
            scaler = StandardScaler()
            scaler.fit(X_train_for_classifier)

            model = ClassifierNN(feature_dim, 2).to(DEVICE)
            classifier_wrapper = TorchIDSWrapper(model, scaler, DEVICE)
            W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

            # GLOBAL (run-wide) PCA basis for plot_decision_boundary -- fit
            # ONCE, here, on ALL tasks' scaled features (already fully
            # loaded in `tasks`), reused unchanged for every subsequent
            # task/checkpoint's boundary plot for the rest of the run.
            all_X_scaled = to_tensor(
                np.concatenate([np.clip(tk["features"].astype(np.float32), 0.0, 1.0) for tk in tasks], axis=0)
            ).numpy()
            pca_mean = all_X_scaled.mean(axis=0)
            _, _, pca_Vt = np.linalg.svd(all_X_scaled - pca_mean, full_matrices=False)
            pca_components = pca_Vt[:2]
            Z_all = (all_X_scaled - pca_mean) @ pca_components.T
            pca_pad = 0.5
            pca_extent = (
                float(Z_all[:, 0].min() - pca_pad), float(Z_all[:, 0].max() + pca_pad),
                float(Z_all[:, 1].min() - pca_pad), float(Z_all[:, 1].max() + pca_pad),
            )
            del all_X_scaled, Z_all

        Xtr = to_tensor(X_train_for_classifier)
        ytr = torch.tensor(y_train, dtype=torch.long)

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
            update_buffer_madar(Xtr, ytr, category, gid_train, benign_label, mal_label, model, DEVICE)

        else:
            print(f"  Samples: {len(Xtr)} (+{len(replay_buffer)} replay) | "
                  f"MADAR ER+KD+SI, {CL_ITERS} iters")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = CL_ITERS

            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, W, omega, p_old_task, t, SI_C, DEVICE)

            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_current = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_current - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()
                    p_old_task[n_key] = p_current

            teacher_model = copy.deepcopy(model); teacher_model.eval()
            update_buffer_madar(Xtr, ytr, category, gid_train, benign_label, mal_label, model, DEVICE)

        buffer_summary = buffer_composition_summary()
        #print(f"    [Buffer] composition: {buffer_summary}")

        # Decision-boundary snapshot for THIS task, post-training (i.e. the
        # same model state per_task_eval below scores).
        X_test_scaled_np = to_tensor(task_test_splits[t][0]).numpy()
        boundary_panels.append(plot_decision_boundary(
            model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
            Xtr.numpy(), category, X_test_scaled_np, category_test,
            t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}.png"),
        ))

        per_task_eval = {j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t + 1)}
        pooled_X = np.concatenate([task_test_splits[j][0] for j in range(t + 1)])
        pooled_y = np.concatenate([task_test_splits[j][1] for j in range(t + 1)])
        pooled_eval = evaluate_classifier(classifier_wrapper, pooled_X, pooled_y)
        mean_per_task_bal_acc = float(np.mean([per_task_eval[j]["balanced_accuracy"] for j in range(t + 1)]))

        # perturbed_test_eval: accuracy computed ONLY on the poisoned rows of
        # each earlier task's test split (cross-referenced by global sample_id
        # against that task's own poisoned_test_sample_ids), tracked at EVERY
        # checkpoint the same way per_task_eval is -- unlike per_task_eval,
        # which mixes clean+poisoned test rows together. Omits task j entirely
        # if it has no poisoned test rows (task 0, or POISON_TEST_DATA off).
        task_poisoned_test_gids[t] = poisoned_test_sample_ids
        perturbed_test_eval = {}
        for j in range(t + 1):
            poisoned_gids_j = set(task_poisoned_test_gids.get(j, []))
            if not poisoned_gids_j:
                continue
            mask = np.isin(task_test_gids[j], list(poisoned_gids_j))
            if mask.sum() == 0:
                continue
            X_test_j, y_test_j = task_test_splits[j]
            perturbed_test_eval[j] = evaluate_classifier(classifier_wrapper, X_test_j[mask], y_test_j[mask])

        print(f"[Task {t} classifier] this-task bal_acc={per_task_eval[t]['balanced_accuracy']:.4f} "
              f"pooled bal_acc={pooled_eval['balanced_accuracy']:.4f} "
              f"mean-per-task bal_acc={mean_per_task_bal_acc:.4f}")

        results.append({
            "task_id": t,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "n_malicious_train": int(len(mal_idx_train)),
            "n_poisoned": n_poisoned,
            "poisoned_sample_ids": poisoned_sample_ids,
            "n_poisoned_test": n_poisoned_test,
            "poisoned_test_sample_ids": poisoned_test_sample_ids,
            "red_agent": red_report,
            "per_task_eval": per_task_eval,
            "perturbed_test_eval": perturbed_test_eval,
            "pooled_eval": pooled_eval,
            "mean_per_task_balanced_accuracy": mean_per_task_bal_acc,
            "grad_steps": grad_steps,
            "replay_buffer_composition": buffer_summary,
        })

    config = {
        "strategy": "madar_er_kd_si",
        "h5_path": args.h5_path, "seed": args.seed, "num_tasks": NUM_TASKS,
        "task_fractions": TASK_FRACTIONS, "task_test_frac": TASK_TEST_FRAC,
        "poison_schedule_enabled": POISON_SCHEDULE_ENABLED, "poison_fraction_flat": POISON_FRACTION_FLAT,
        "poison_fraction_train_start": POISON_FRACTION_TRAIN_START,
        "poison_fraction_train_end": POISON_FRACTION_TRAIN_END,
        "poison_fraction_test_start": POISON_FRACTION_TEST_START,
        "poison_fraction_test_end": POISON_FRACTION_TEST_END,
        "poison_test_data": POISON_TEST_DATA,
        "require_evasion_success": REQUIRE_EVASION_SUCCESS,
        "red_epsilon": RED_EPSILON, "red_max_steps": RED_MAX_STEPS,
        "red_timesteps_per_task": RED_TIMESTEPS_PER_TASK, "alpha_contrast": ALPHA_CONTRAST,
        "contrastive_ema": CONTRASTIVE_EMA, "contrastive_recency_decay": CONTRASTIVE_RECENCY_DECAY,
        "max_eval_samples_per_task": MAX_EVAL_SAMPLES_PER_TASK, "feature_dim": feature_dim,
        "day_mapping": day_mapping,
        "mem_size": MEM_SIZE, "buffer_strategy": "uniform_50_50",
        "madar_contamination": MADAR_CONTAMINATION, "kd_temp": KD_TEMP,
        "si_c": SI_C, "si_eps": SI_EPS, "rnt_floor": RNT_FLOOR, "task0_epochs": TASK0_EPOCHS,
        "cl_iters": CL_ITERS, "batch_size": BATCH_SIZE, "feature_clip": FEATURE_CLIP,
    }
    with open(os.path.join(out_dir, f"{args.log_name}.json"), "w") as f:
        json.dump({"config": config, "warnings": warnings_log, "results": results}, f, indent=2)

    plot_task_metrics(results, os.path.join(out_dir, "plots", "task_metrics.png"))
    plot_prototype_heatmap(bank, os.path.join(out_dir, "plots", "prototype_heatmap.png"))
    plot_episode_clouds(bank, os.path.join(out_dir, "plots", "episode_clouds.png"))
    plot_decision_boundary_grid(boundary_panels, os.path.join(out_dir, "plots", "decision_boundary_evolution.png"))

    print(f"\nDone. Results + plots written to {out_dir}/")
    print("full time elapsed: %.2f seconds" % (time.perf_counter() - start_time))


if __name__ == "__main__":
    main()
