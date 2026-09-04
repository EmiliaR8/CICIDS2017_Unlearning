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
so train_t's reward never references test_t. Evasion success stays a hard,
unconditional requirement independent of this diversity pressure: the +10
reward bonus for successful misclassification dominates the reward
magnitude, and only genuinely-evasive perturbations (evaded_mask=True) ever
become poison candidates -- there is no toggle to relax this. red_test_pert_
agent only trains when POISON_TEST_DATA is on AND this task actually has
malicious test samples, to avoid training a whole second SAC agent for
nothing. REWORK (uncertainty-margin pipeline): the old per-task poison
schedule (ramping fractions across tasks) is gone -- both train and test now
use fixed, always-active per-class fractions (TRAIN_UNCERTAIN_FRACTION=0.30,
TEST_AGGREGATE_FRACTION=0.40), and WHICH samples get attacked is chosen by
uncertainty (proximity to C1's decision boundary) rather than randomly.

UPDATE (REWORK, quad red agents): two MORE agents added per task --
red_train_benign_pert_agent and red_test_benign_pert_agent -- mirroring the
malicious pair but attacking BENIGN rows toward mal_label instead of
malicious rows toward benign_label. Reuses NetworkAttackEnv/
train_red_agent_for_task/evaluate_agent_on_batch entirely unmodified: the
"benign_label" argument threaded through that whole call chain is really
just "the target class this agent optimizes toward", so passing mal_label
in for the two new agents is sufficient. All four agents (train, test,
train_benign, test_benign) register in the SAME contrastive bank under
"{task_id}_{agent_type}", so each is pushed away from every agent
registered before it (this task and all earlier ones) via the existing
recency-weighted mechanism -- no new diversity logic needed. Poisoning of
benign data mirrors the malicious mechanic exactly (same TRAIN_UNCERTAIN_
FRACTION quota and uncertainty-ranked selection, same unconditional success
requirement) and is applied on TOP of the existing malicious substitution in
X_train_for_classifier/X_test (disjoint index sets, so no collision).
Ground truth y is left unchanged (still benign) on substitution, matching
how malicious_perturbed keeps its true malicious label -- only X changes.
New category value "benign_perturbed" added alongside the existing
benign/malicious_clean/malicious_perturbed three, purely for the same
buffer-composition/plotting diagnostics (never fed back into selection or
training). No RED_TARGET_MARGIN_CONFIDENCE bias on the two new agents --
see train_red_agent_for_task's docstring.

UPDATE (REWORK, pocket-targeted test poisoning): red_test_pert_agent/
red_test_benign_pert_agent now train AFTER this task's CL training finishes,
not before it (train-side agents/poisoning are unaffected -- still attack
the pre-task-t classifier, since that's what CAUSES the boundary to move). A
frozen pre-training classifier snapshot (pre_train_classifier_wrapper,
captured right before train_cl_er mutates `model`) is passed alongside the
now-POST-training live classifier, and NetworkAttackEnv's new pocket_term
rewards perturbations landing where the two disagree in the evasive
direction -- i.e. a point THIS TASK's own poisoned CL training just flipped
from correctly- to incorrectly-classified, a "pocket" attributable to this
task's training specifically, layered on top of (not replacing)
RED_TARGET_MARGIN_CONFIDENCE's existing margin-minimizing bias. See
POCKET_SHIFT_WEIGHT's definition and NetworkAttackEnv.step()'s pocket_term
for the full mechanics. Identical mechanism/rationale to
madar_unlearning_cl_pipeline.py's UPDATE 7, so both pipelines face the same
attack.
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

POISON_TEST_DATA = True  # REWORK: matches madar_unlearning_cl_pipeline.py's POISON_TEST_DATA
                          # (previously False/absent here -- test was never poisoned). Needed
                          # so per_task_eval/pooled_eval are computed on the SAME kind of data
                          # in both pipelines; see that file's module docstring BRANCH NOTE for
                          # the original rationale (evasion rate alone doesn't check whether
                          # detection/unlearning generalizes to evasive samples never trained
                          # on). Same evaded-mask-restricted substitution as train (schedule
                          # below), applied to each task's test split.

# REWORK (uncertainty-margin pipeline): replaces the old per-task ramp
# schedule (POISON_SCHEDULE_ENABLED/POISON_FRACTION_TRAIN/TEST_START/END) and
# its flat fallback (POISON_FRACTION_FLAT) with two fixed, always-active
# fractions -- no schedule, no toggle. Both are PER-CLASS quotas (malicious
# and benign each get their own independent budget of this size), not a
# combined/shared budget. Identical values to madar_unlearning_cl_pipeline.py's.
TRAIN_UNCERTAIN_FRACTION = 0.30  # per class: fraction of this task's own row count for
                                  # that class selected (most-uncertain-under-C1 first) as
                                  # the attack pool for that class's train-side agent. Every
                                  # episode that successfully flips (see NetworkAttackEnv's
                                  # bare class-flip termination) replaces its clean row --
                                  # so the ACTUAL poisoned count can land under 30% if not
                                  # every attempt succeeds within RED_MAX_STEPS. There is no
                                  # REQUIRE_EVASION_SUCCESS toggle anymore -- success is
                                  # unconditionally required to poison, always.
TEST_AGGREGATE_FRACTION = 0.40  # per class: fraction of this task's own row count for that
                                 # class selected, tiered (C1-correct-by-uncertainty first,
                                 # then C1-incorrect padding), from the aggregated (this
                                 # task's own test split + task (t-1)'s own test split) pool.
                                 # Same "actual count can land under quota" caveat as train
                                 # above -- see the TEST-SIDE blocks for the tiering logic.

# NEW (variant-augmentation branch): see madar_unlearning_cl_pipeline.py's
# identical constants for the full rationale -- mirrored unchanged.
VARIANT_STEPS_PER_LEVEL = 3
MAX_VARIANTS_PER_SAMPLE = 5
VARIANT_GID_OFFSET = 10**9

RECYCLE_TEST_POCKETS = True  # toggle: True (current/default behavior) unions this task's own
                              # test split with task (t-1)'s own test split before selection,
                              # and persists any recycled successes as new rows into this
                              # task's task_test_splits[t] (see the TEST-SIDE blocks below).
                              # False disables recycling entirely -- the perturbable test pool
                              # is just this task's own test rows, selection/tiering/the joint
                              # C1-correct-and-C2-wrong success condition are otherwise
                              # unchanged. Implemented as a single gate on X_prev_test/
                              # y_prev_test/gid_prev_test (forced to empty arrays when off) --
                              # every downstream recycled_*/n_poisoned_test_recycled_prev*
                              # computation naturally reduces to "nothing recycled" without
                              # its own separate branch.

RED_EPSILON = 0.25
RED_MAX_STEPS = 25
RED_C1_MISMATCH_PATIENCE = 3  # FINETUNE (test-side patience investigation): see
                               # madar_unlearning_cl_pipeline.py's identical constant for the
                               # full rationale (a diagnostic run showed 99.5% of malicious
                               # test attacks hard-aborting on a C1 disagreement within ~2
                               # steps, using only ~8-12% of the epsilon budget). Tolerates up
                               # to this many CONSECUTIVE C1-mismatch steps (streak resets the
                               # moment C1 agrees again) before pgd_boundary_search_batch gives
                               # up on a sample.
RED_TIMESTEPS_PER_TASK = 5000  # REWORK (uncertainty-margin pipeline): doubled from 2500 --
                                # POTENTIALLY REVISIT, tune from real run data. See
                                # madar_unlearning_cl_pipeline.py's identical constant for the
                                # full rationale (test-side joint success condition was a
                                # sparse target under the old budget). Identical value here so
                                # both pipelines' red agents get the same training budget.
ALPHA_CONTRAST = 0.5
CONTRASTIVE_EMA = 0.95
CONTRASTIVE_RECENCY_DECAY = 0.5  # weight of task (k-1) vs (k-2) vs ... in the diversity reward

RED_TARGET_MARGIN_CONFIDENCE = 0.55  # REWORK (margin-minimizing evasion): passed to the
                                      # test_pert_agent/test_benign_pert_agent NetworkAttackEnv
                                      # as a DENSE shaping term only -- test-side success/
                                      # termination is fully governed by the joint C1-correct/
                                      # C2-wrong condition (see NetworkAttackEnv's
                                      # shift_reference_classifier), independent of this value.
                                      # Reward peaks at just-barely-evasive confidence instead of
                                      # rewarding maximal confidence deep into the target class.
                                      # See NetworkAttackEnv.step()'s confidence_term for the
                                      # exact shape. Rationale: plain MADAR's buffer selection
                                      # (IsolationForest anomaly/inlier over the latent
                                      # embedding) has no notion of margin/decision-boundary
                                      # distance, so margin-hugging poison is a blind spot for
                                      # it specifically, unlike MADAR+Unlearning's
                                      # perturbation_classifier (an explicitly boundary-adjacent
                                      # detector). Set to None to restore the original
                                      # confidence-maximizing reward for both agents.
RED_TARGET_MARGIN_HIGH = 0.65  # See madar_unlearning_cl_pipeline.py's identical constant --
                                # upper end of the test-side target confidence band
                                # [RED_TARGET_MARGIN_CONFIDENCE, RED_TARGET_MARGIN_HIGH]
                                # pgd_boundary_search_batch keeps stepping toward past the
                                # first boundary crossing.

RED_TRAIN_TARGET_MARGIN_CONFIDENCE = 0.65  # REWORK (uncertainty-margin pipeline): passed to
                                      # BOTH train_pert_agent and train_benign_pert_agent
                                      # (train-side only -- test-side keeps using
                                      # RED_TARGET_MARGIN_CONFIDENCE=0.55 above, unchanged).
                                      # confidence_term (NetworkAttackEnv.step()) peaks at this
                                      # value: 0.5 + 0.15, i.e. 0.15 past uncertain toward the
                                      # target class -- slightly further than the OLD 0.55
                                      # target, favoring confidence closer to 0.65 over landing
                                      # right at 0.55, while still penalizing overshoot deep past
                                      # 0.65 (same symmetric peaked shape as
                                      # RED_TARGET_MARGIN_CONFIDENCE, just re-centered). Episode
                                      # TERMINATION for train-side agents is unaffected by this
                                      # value -- it still fires on the bare class flip (or
                                      # max_steps), same as always; only the reward that shapes
                                      # WHERE within the evasive region the agent prefers to land
                                      # changes. Identical constant/value to
                                      # madar_unlearning_cl_pipeline.py's.

# REWORK (gradient-attack branch): POCKET_SHIFT_WEIGHT/PROXIMITY_SHAPING_ENABLED/
# PROXIMITY_WEIGHT/PROXIMITY_LENGTH_SCALE/PROXIMITY_ANCHOR_MAX removed -- see
# madar_unlearning_cl_pipeline.py's identical removal note for the full
# rationale (no reward being optimized in a direct gradient search, so these
# RL-reward-shaping terms had no analog to carry forward).

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
SI_OMEGA_DECAY = 0.9  # REWORK (SI capacity investigation): see
                       # madar_unlearning_cl_pipeline.py's definition for the
                       # full rationale -- applied identically here so plain
                       # MADAR and MADAR+Unlearning stay directly comparable.
                       # Set to 1.0 to restore the old unbounded-accumulation
                       # behavior.
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
    (final_obs, total_reward, pred_label, cum_perturbation, success).

    REWORK (joint pocket objective / "Option B"): `success` is
    info["success"] from the env's LAST step -- True only when the episode
    ended via `terminated` (the env's own joint C1-still-correct AND
    C2-now-target condition when a reference classifier is configured, or
    plain C2 evasion otherwise), never when it merely ran out of steps
    (`truncated`). Callers should use this instead of re-deriving success
    from pred_label alone, which would silently drop the C1 requirement.
    Kept in step with madar_unlearning_cl_pipeline.py's identical copy.
    """
    obs, _ = env.reset(options={"index": start_index})
    total_reward = 0.0
    cum_pert = np.zeros(env.action_space.shape[0], dtype=np.float32)

    done = False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        a = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)
        a = np.clip(a, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(a)
        total_reward += float(reward)
        cum_pert += a
        done = bool(terminated) or bool(truncated)

    return obs, total_reward, info["prediction"], cum_pert, bool(info.get("success", False))


def evaluate_agent_on_batch(env, model, X, y, benign_label, only_malicious=True,
                             deterministic=True, max_test=None, allowed_start_indices=None):
    """
    Runs the agent over (up to max_test of) the malicious samples. Returns
    (X_pert, evasion_rate, rewards, avg_pert_norm, n_attacked, evaded_mask).
    evaded_mask[i] is True only for samples where the attack succeeded -- this is
    the exact set poisoning should draw from.

    BUGFIX (C1-correct-only sampling prototype): allowed_start_indices, when
    given, restricts `indices` to that set -- MUST be the same array passed as
    `env`'s allowed_start_indices (or a subset of it). Without this, `indices`
    here defaults to the FULL only_malicious pool, independent of whatever
    `env.non_benign_indices` was restricted to at construction. run_single_agent_attack
    calls env.reset(options={"index": i}) for each i; NetworkAttackEnv.reset()
    silently substitutes a RANDOM row from env.non_benign_indices whenever the
    requested i isn't in it -- so for any i outside the allowed set, this loop
    was attacking some other, unrelated row and recording the result under i's
    slot in X_pert/evaded_mask anyway. That silent substitution path was dead
    code before allowed_start_indices existed (indices and non_benign_indices
    were always the same set), which is why it went unnoticed until enabled.
    Kept in step with madar_unlearning_cl_pipeline.py's identical fix.

    REWORK (joint pocket objective / "Option B"): evasion/evaded_mask is now
    keyed to run_single_agent_attack's `success` flag (the env's own
    terminated condition), not a re-derived pred_label==benign_label check --
    for envs with a shift_reference_classifier (test/test_benign), that means
    a sample only counts as evaded when the reference classifier (C1) STILL
    calls the final perturbed point the sample's true label AND the live
    classifier (C2) now calls it benign_label. Train-side envs (no reference
    classifier) are unaffected -- success there is still plain C2 evasion.
    Kept in step with madar_unlearning_cl_pipeline.py's identical copy.
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
    if allowed_start_indices is not None:
        indices = np.intersect1d(indices, np.asarray(allowed_start_indices))
    if max_test is not None:
        indices = indices[:max_test]

    for i in indices:
        final_obs, total_reward, pred_label, cum_pert, success = run_single_agent_attack(
            env, model, start_index=i, benign_label=benign_label, deterministic=deterministic
        )
        rewards.append(total_reward)
        pert_norms.append(float(np.linalg.norm(cum_pert, ord=2)))
        attacked += 1

        if success:
            evasion += 1
            X_pert[i] = final_obs
            evaded_mask[i] = True

    evasion_rate = evasion / max(attacked, 1)
    avg_pert_norm = float(np.mean(pert_norms)) if pert_norms else 0.0
    return X_pert, evasion_rate, rewards, avg_pert_norm, attacked, evaded_mask


def _variant_gid_display(internal_gid):
    """See madar_unlearning_cl_pipeline.py's identical function for the full
    docstring -- ported unchanged."""
    internal_gid = int(internal_gid)
    if internal_gid < VARIANT_GID_OFFSET:
        return internal_gid
    remainder = internal_gid - VARIANT_GID_OFFSET
    original_gid, level = divmod(remainder, 10)
    return f"{original_gid}.{level}"


def generate_train_variants(env, red_agent, X_pert, evaded_mask, gid_array, true_label,
                             steps_per_level=VARIANT_STEPS_PER_LEVEL, max_variants=MAX_VARIANTS_PER_SAMPLE,
                             deterministic=True):
    """See madar_unlearning_cl_pipeline.py's identical function for the full
    docstring -- ported unchanged."""
    # BUGFIX: see madar_unlearning_cl_pipeline.py's identical fix for the full
    # rationale -- env is Monitor-wrapped; unwrap so env.step() below bypasses
    # Monitor's needs_reset bookkeeping entirely.
    env = getattr(env, "unwrapped", env)

    feat_dim = X_pert.shape[1]
    variant_rows, internal_gids, display_gids = [], [], []
    n_variants_by_original = {}

    for i in np.where(evaded_mask)[0]:
        current_state = X_pert[i].copy()
        original_gid = int(gid_array[i])
        n_this = 0
        for level in range(1, max_variants + 1):
            env.state = current_state.copy()
            env.true_label = true_label
            env.cum_action = np.zeros_like(current_state, dtype=np.float32)
            env.steps = 0

            still_evading = False
            for _ in range(steps_per_level):
                a, _ = red_agent.predict(env.state, deterministic=deterministic)
                a = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)
                a = np.clip(a, env.action_space.low, env.action_space.high)
                _, _, _, _, info = env.step(a)
                still_evading = bool(info["success"])

            if not still_evading:
                break

            n_this += 1
            current_state = env.state.copy()
            variant_rows.append(current_state)
            internal_gids.append(VARIANT_GID_OFFSET + original_gid * 10 + level)
            display_gids.append(f"{original_gid}.{level}")

        if n_this > 0:
            n_variants_by_original[original_gid] = n_this

    X = np.stack(variant_rows).astype(np.float32) if variant_rows else np.empty((0, feat_dim), dtype=np.float32)
    return {
        "X": X,
        "internal_gids": np.asarray(internal_gids, dtype=np.int64),
        "display_gids": display_gids,
        "n_variants_by_original": n_variants_by_original,
    }


# NEW (gradient-attack branch): PGD nearest-boundary search, replacing the
# SAC/NetworkAttackEnv rollout for TEST-SIDE agents only. See
# madar_unlearning_cl_pipeline.py's identical function for the full
# docstring/rationale (including the FINETUNE c1_patience addition below)
# -- ported unchanged.
def _log_pgd_break_diagnostics(diag_log_path, diag_label, break_reasons, steps_survived,
                                c1_mismatch_steps, initial_confidences, pert_norms,
                                epsilon, max_steps, attacked, c1_patience=None,
                                recovered_after_mismatch=None):
    """
    DIAGNOSTIC (why-so-few-test-pockets investigation): summarizes, for one
    pgd_boundary_search_batch() call, how many attacked samples ended up
    evaded vs. broke for each reason (C1 disagreeing, the gradient
    vanishing, or simply exhausting max_steps still C1-correct but never
    reaching benign_label), how many steps the non-evaded ones survived
    before that, and how much of the epsilon L2 budget they actually used.
    This is purely observational -- it never affects success/evaded_mask or
    the return contract -- meant to tell apart "ran out of epsilon budget"
    from "C1 disagreed almost immediately" as the dominant failure mode.
    Prints unconditionally; also appended to diag_log_path (the
    pocket_targeting_diagnostic log) when one is given, matching
    c1_correct_pool's dual console+log convention.

    c1_patience/recovered_after_mismatch (FINETUNE, optional): when given,
    also reports how many CONSECUTIVE C1-mismatch steps were tolerated
    before a hard abort, and how many evaded samples had at least one
    tolerated mismatch along the way -- i.e. how much the patience knob
    itself is actually buying.
    """
    if attacked == 0:
        return
    label = diag_label or "pgd_boundary_search_batch"
    counts = {"evaded": 0, "c1_mismatch": 0, "vanishing_grad": 0, "max_steps_exhausted": 0}
    for r in break_reasons:
        counts[r] += 1

    lines = [f"    [PGD break diagnostic] {label}: {attacked} attacked -- "
             f"evaded={counts['evaded']}, c1_mismatch={counts['c1_mismatch']}, "
             f"vanishing_grad={counts['vanishing_grad']}, "
             f"max_steps_exhausted={counts['max_steps_exhausted']}"]

    if c1_patience is not None:
        lines.append(f"      c1_patience={c1_patience} consecutive steps tolerated, "
                     f"recovered_after_mismatch={recovered_after_mismatch or 0}/{counts['evaded']} "
                     f"of the evaded samples had a tolerated mismatch along the way")

    if c1_mismatch_steps:
        frac_step1 = sum(1 for s in c1_mismatch_steps if s == 1) / len(c1_mismatch_steps)
        lines.append(f"      c1_mismatch (hard abort) at step (median)="
                     f"{float(np.median(c1_mismatch_steps)):.1f}/{max_steps}, "
                     f"{frac_step1:.0%} of those broke on step 1")

    if steps_survived:
        lines.append(f"      avg steps survived (non-evaded only)="
                     f"{float(np.mean(steps_survived)):.1f}/{max_steps}")

    if initial_confidences:
        lines.append(f"      initial confidence toward target label: "
                     f"mean={float(np.mean(initial_confidences)):.3f}, "
                     f"median={float(np.median(initial_confidences)):.3f}")

    if pert_norms:
        util = np.asarray(pert_norms) / epsilon
        lines.append(f"      epsilon-budget utilization (final L2 / epsilon={epsilon}): "
                     f"mean={util.mean():.2f}, median={float(np.median(util)):.2f}, "
                     f"max={util.max():.2f}")

    msg = "\n".join(lines)
    print(msg)
    if diag_log_path is not None:
        with open(diag_log_path, "a") as f:
            f.write(msg + "\n")


def pgd_boundary_search_batch(classifier_wrapper, X, y, benign_label, allowed_start_indices,
                               shift_reference_classifier=None, epsilon=None, max_steps=None,
                               margin_low=0.55, margin_high=0.65, device=None, max_test=None,
                               diag_log_path=None, diag_label=None, c1_patience=None):
    epsilon = RED_EPSILON if epsilon is None else epsilon
    max_steps = RED_MAX_STEPS if max_steps is None else max_steps
    c1_patience = RED_C1_MISMATCH_PATIENCE if c1_patience is None else c1_patience
    device = DEVICE if device is None else device
    margin_mid = (margin_low + margin_high) / 2.0

    model, scaler = classifier_wrapper.model, classifier_wrapper.scaler
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    target_t = torch.tensor([benign_label], dtype=torch.long, device=device)

    def _scaled(x_t):
        return torch.clamp((x_t - mean_t) / scale_t, -FEATURE_CLIP, FEATURE_CLIP)

    def _probs(x_np):
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32, device=device)
            return F.softmax(model(_scaled(xt).unsqueeze(0)), dim=1)[0].cpu().numpy()

    indices = np.where(y != benign_label)[0]
    if allowed_start_indices is not None:
        indices = np.intersect1d(indices, np.asarray(allowed_start_indices))
    if max_test is not None:
        indices = indices[:max_test]

    X_pert = X.copy()
    evaded_mask = np.zeros(X.shape[0], dtype=bool)
    rewards, pert_norms = [], []
    evasion = 0
    attacked = 0

    # DIAGNOSTIC (why-so-few-test-pockets investigation) -- see
    # _log_pgd_break_diagnostics's docstring. Tracked per attacked sample,
    # aggregated/printed once after the loop below.
    break_reasons = []
    steps_survived = []
    c1_mismatch_steps = []
    initial_confidences = []
    recovered_after_mismatch = 0  # FINETUNE: evaded samples that had at least
                                   # one tolerated C1 mismatch along the way --
                                   # the direct payoff metric for c1_patience.

    model.eval()
    for i in indices:
        attacked += 1
        x0 = X[i].astype(np.float32)
        true_label = int(y[i])
        x_cur = x0.copy()
        success = False
        final_confidence = 0.0
        initial_confidences.append(float(_probs(x0)[benign_label]))

        steps_taken = 0
        c1_mismatch_at = None
        c1_mismatch_streak = 0
        c1_ever_mismatched = False
        vanishing_grad_hit = False

        for step_num in range(1, max_steps + 1):
            x_t = torch.tensor(x_cur, dtype=torch.float32, device=device, requires_grad=True)
            logits = model(_scaled(x_t).unsqueeze(0))
            loss = F.cross_entropy(logits, target_t)
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad = x_t.grad.detach().cpu().numpy()

            grad_norm = np.linalg.norm(grad, ord=2)
            steps_taken = step_num
            if grad_norm < 1e-12:
                vanishing_grad_hit = True
                break
            step = -grad / grad_norm

            delta = (x_cur + (epsilon / max_steps) * step) - x0
            delta_norm = np.linalg.norm(delta, ord=2)
            if delta_norm > epsilon:
                delta = delta * (epsilon / delta_norm)
            x_cur = np.clip(x0 + delta, 0.0, 1.0).astype(np.float32)

            probs = _probs(x_cur)
            predicted_label = int(np.argmax(probs))
            confidence = float(probs[benign_label])
            final_confidence = confidence

            c1_correct = True
            if shift_reference_classifier is not None:
                ref_pred = shift_reference_classifier.predict(x_cur.reshape(1, -1))[0]
                c1_correct = bool(ref_pred == true_label)
                # FINETUNE (test-side patience investigation): tolerate up to
                # c1_patience CONSECUTIVE mismatched steps (streak resets the
                # moment C1 agrees again) instead of hard-aborting on the
                # first one. A step where C1 disagrees still can't count as
                # success below (c1_correct is False this iteration either
                # way), it just no longer necessarily ends the episode.
                if not c1_correct:
                    c1_ever_mismatched = True
                    c1_mismatch_streak += 1
                    if c1_mismatch_streak > c1_patience:
                        c1_mismatch_at = step_num
                        break
                else:
                    c1_mismatch_streak = 0

            c2_evaded = bool(predicted_label == benign_label)
            if c2_evaded and c1_correct:
                success = True
                if confidence >= margin_low:
                    break

        rewards.append(5 * (0.5 - abs(final_confidence - margin_mid)))
        pert_norms.append(float(np.linalg.norm(x_cur - x0, ord=2)))

        if success:
            evasion += 1
            X_pert[i] = x_cur
            evaded_mask[i] = True
            break_reasons.append("evaded")
            if c1_ever_mismatched:
                recovered_after_mismatch += 1
        else:
            steps_survived.append(steps_taken)
            if c1_mismatch_at is not None:
                break_reasons.append("c1_mismatch")
                c1_mismatch_steps.append(c1_mismatch_at)
            elif vanishing_grad_hit:
                break_reasons.append("vanishing_grad")
            else:
                break_reasons.append("max_steps_exhausted")

    evasion_rate = evasion / max(attacked, 1)
    avg_pert_norm = float(np.mean(pert_norms)) if pert_norms else 0.0

    _log_pgd_break_diagnostics(diag_log_path, diag_label, break_reasons, steps_survived,
                                c1_mismatch_steps, initial_confidences, pert_norms,
                                epsilon, max_steps, attacked, c1_patience, recovered_after_mismatch)

    return X_pert, evasion_rate, rewards, avg_pert_norm, attacked, evaded_mask


RED_AGENT_SEED_OFFSETS = {"train": 0, "test": 100_000, "train_benign": 200_000, "test_benign": 300_000}


def c1_correct_pool(classifier_c1, X_full, idx_pool, target_label, task_id, agent_label, pocket_diag_log_path=None):
    """
    PROTOTYPE (C1-correct-only sampling): narrows idx_pool (indices into
    X_full, e.g. mal_idx_train/mal_idx_test/benign_idx_train/benign_idx_test)
    down to the subset that classifier_c1 -- the classifier state a red
    agent's episodes are meant to be judged against as "before" -- already
    classifies as target_label. Passed to train_red_agent_for_task as
    allowed_start_indices, so every episode starts from a genuinely "was
    correct under C1" row instead of a uniformly-random one.

    Kept in step with madar_unlearning_cl_pipeline.py's identical copy of
    this function -- see the module docstring's note on why poisoning must
    stay identical between the two pipelines. Logs the raw pool size to the
    console (and, now that this pipeline has its own pocket_targeting_
    diagnostic.txt, that log too when a path is given) either way, purely
    as data for recalibrating the minimum-pool fallback threshold later.
    """
    if len(idx_pool) == 0:
        return idx_pool
    preds = classifier_c1.predict(X_full[idx_pool])
    correct_pool = idx_pool[preds == target_label]
    msg = (f"[C1-pool] Task {task_id} {agent_label}: {len(correct_pool)}/{len(idx_pool)} "
           f"candidates are C1-correct.")
    print(f"    {msg}")
    if pocket_diag_log_path is not None:
        with open(pocket_diag_log_path, "a") as f:
            f.write(msg + "\n")
    return correct_pool


def _uncertainty_rank(classifier, X_full, idx_pool):
    """REWORK (uncertainty-margin pipeline): reorders idx_pool (indices into
    X_full) by ASCENDING confidence under `classifier` -- i.e. most uncertain
    (predict_proba's max-class-probability closest to 0.5, so closest to the
    decision boundary) first. Kept in step with
    madar_unlearning_cl_pipeline.py's identical copy of this function.
    Returns the SAME set of indices, just reordered -- does not filter
    anything out."""
    if len(idx_pool) == 0:
        return idx_pool
    proba = classifier.predict_proba(X_full[idx_pool])
    confidence = proba.max(axis=1)
    order = np.argsort(confidence)  # ascending: most uncertain (lowest max-proba) first
    return idx_pool[order]


def train_red_agent_for_task(task_id, agent_type, classifier, X_data, y_data, benign_label, bank, seed, out_dir,
                              target_margin_confidence=None, shift_reference_classifier=None,
                              pocket_shift_weight=None, proximity_anchor_X=None, proximity_weight=None,
                              proximity_length_scale=5.0, allowed_start_indices=None):
    """
    REWORK: one call per (task, agent_type) pair now, not one call per task.
    agent_type in {"train", "test", "train_benign", "test_benign"} -- the
    caller passes X_data/y_data as whichever split this agent is dedicated to
    (X_train for the train-side perturbation agent, X_test for the test-side
    one), and this builds a NetworkAttackEnv directly over THAT data. Fixes a
    latent bug: previously a single agent's env was built once from X_train,
    then reused unmodified for the "test" evaluate_agent_on_batch call --
    NetworkAttackEnv.reset() always samples from self.X_data (fixed at
    construction, update_data() existed but was never called), so that
    second call was actually attacking train rows at test-derived indices,
    not test data at all.

    "train"/"test" attack malicious rows toward benign_label (the original
    mechanism); "train_benign"/"test_benign" (REWORK, quad red agents) attack
    BENIGN rows toward whatever label the CALLER passes in as `benign_label`
    (in practice mal_label) -- this argument is really just "the target class
    this agent optimizes toward", and NetworkAttackEnv's non_benign_indices =
    where(y_data != that target) already generalizes to "the pool being
    attacked" for any target class, so no changes to NetworkAttackEnv/
    evaluate_agent_on_batch were needed to add the benign-side agents.

    agent_id is now a compound "{task_id}_{agent_type}" string (was task_id
    alone) so the contrastive bank tracks all four agent types as fully
    distinct entries -- each agent's reward pushes it away from every agent
    registered before it, this task and all earlier tasks, via the existing
    recency-weighted mechanism. Distinct seed offset per agent_type (see
    RED_AGENT_SEED_OFFSETS) so the four agents' SAC init/training RNG
    streams don't collide within the same task.

    target_margin_confidence (REWORK, margin-minimizing evasion): forwarded
    straight to NetworkAttackEnv -- None (default) keeps the original
    confidence-maximizing reward; a value like RED_TARGET_MARGIN_CONFIDENCE
    switches an agent to reward landing just past the decision boundary
    instead of deep past it. Only wired up to "train"/"test" (the malicious
    evasion agents, see RED_TARGET_MARGIN_CONFIDENCE's call sites in main())
    -- the new "train_benign"/"test_benign" agents are left at the default
    (None, ordinary confidence-maximizing reward) since the margin-minimizing
    tactic was motivated by and validated against malicious-side evasion
    specifically; nothing stops a future run from passing it to the benign
    agents too. See that constant's definition and NetworkAttackEnv.step()
    for the full rationale.

    shift_reference_classifier/pocket_shift_weight (REWORK, pocket-targeted
    test poisoning): forwarded straight to NetworkAttackEnv -- see its
    docstring. Only meaningful for "test"/"test_benign", passed a FROZEN
    pre-CL-training snapshot by main() (train-side agents have no "before"
    state to compare against, since they're what shifts the boundary).

    proximity_anchor_X/proximity_weight/proximity_length_scale (REWORK,
    proximity-anchored pocket targeting): forwarded straight to
    NetworkAttackEnv -- see its docstring. Only meaningful for "test"/
    "test_benign", passed this task's own TRAIN-side poisoned exemplars by
    main().
    """
    seed_offset = RED_AGENT_SEED_OFFSETS.get(agent_type, 0)
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
            shift_reference_classifier=shift_reference_classifier,
            pocket_shift_weight=pocket_shift_weight,
            proximity_anchor_X=proximity_anchor_X,
            proximity_weight=proximity_weight,
            proximity_length_scale=proximity_length_scale,
            allowed_start_indices=allowed_start_indices,
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
# TEMPORARY DIAGNOSTIC (pocket-targeting investigation), test-accuracy
# breakdown -- MADAR version, ported from madar_unlearning_cl_pipeline.py.
# Answers, per task, the 3 questions that apply without an unlearning step:
# (1) how many rows are in this task's test split (own reserved + recycled
# successes appended in), and how many of those are actually-perturbed rows
# counted in the reported accuracy; (2) of the perturbed rows, how many the
# classifier STILL gets right after adaptation, per class; (3) of the clean
# rows, how many it gets right after adaptation. Appended into the SAME
# pocket_targeting_diagnostic.txt log, right after the existing
# correctness-pattern section for that task.
# ---------------------------------------------------------------------------
def _test_accuracy_snapshot(classifier_wrapper, X_test, y_test, poison_idx_test, poison_idx_test_benign,
                             mal_label, benign_label):
    """See madar_unlearning_cl_pipeline.py's identical function for the full
    docstring -- ported unchanged (MADAR just never calls it a second time
    for a "post" snapshot, since there's no unlearning step)."""
    n_test_total = len(y_test)
    poison_idx_test = np.asarray(poison_idx_test, dtype=np.int64)
    poison_idx_test_benign = np.asarray(poison_idx_test_benign, dtype=np.int64)
    all_poison_idx = np.concatenate([poison_idx_test, poison_idx_test_benign])
    clean_idx = np.setdiff1d(np.arange(n_test_total, dtype=np.int64), all_poison_idx)

    mal_pred = classifier_wrapper.predict(X_test[poison_idx_test]) if len(poison_idx_test) > 0 else np.array([])
    ben_pred = classifier_wrapper.predict(X_test[poison_idx_test_benign]) if len(poison_idx_test_benign) > 0 else np.array([])
    clean_pred = classifier_wrapper.predict(X_test[clean_idx]) if len(clean_idx) > 0 else np.array([])
    clean_true = y_test[clean_idx]

    return {
        "n_test_total": int(n_test_total),
        "n_perturbed_mal": int(len(poison_idx_test)), "n_perturbed_ben": int(len(poison_idx_test_benign)),
        "mal_correct": int(np.sum(mal_pred == mal_label)),
        "ben_correct": int(np.sum(ben_pred == benign_label)),
        "clean_idx": clean_idx, "clean_true": clean_true, "clean_pred": clean_pred,
        "clean_total": int(len(clean_idx)),
        "clean_correct": int(np.sum(clean_pred == clean_true)) if len(clean_idx) > 0 else 0,
    }


def _ratio_str(numer, denom):
    return f"{numer}/{denom} ({numer / denom:.3f})" if denom > 0 else f"{numer}/0 (n/a)"


def _confusion_matrix_4x2(snapshot, mal_label, benign_label):
    """See madar_unlearning_cl_pipeline.py's identical function for the full
    docstring -- ported unchanged."""
    clean_true, clean_pred = snapshot["clean_true"], snapshot["clean_pred"]
    mal_mask = clean_true == mal_label
    ben_mask = clean_true == benign_label

    def _counts(pred_subset):
        n_benign = int(np.sum(pred_subset == benign_label)) if len(pred_subset) else 0
        n_mal = int(np.sum(pred_subset == mal_label)) if len(pred_subset) else 0
        return n_benign, n_mal

    clean_mal_pred_benign, clean_mal_pred_mal = _counts(clean_pred[mal_mask])
    clean_ben_pred_benign, clean_ben_pred_mal = _counts(clean_pred[ben_mask])
    pert_mal_pred_mal = snapshot["mal_correct"]
    pert_mal_pred_benign = snapshot["n_perturbed_mal"] - pert_mal_pred_mal
    pert_ben_pred_benign = snapshot["ben_correct"]
    pert_ben_pred_mal = snapshot["n_perturbed_ben"] - pert_ben_pred_benign

    columns = ["malicious", "benign", "perturbed_malicious", "perturbed_benign"]
    row_benign = [clean_mal_pred_benign, clean_ben_pred_benign, pert_mal_pred_benign, pert_ben_pred_benign]
    row_malicious = [clean_mal_pred_mal, clean_ben_pred_mal, pert_mal_pred_mal, pert_ben_pred_mal]
    return columns, row_benign, row_malicious


def _format_confusion_matrix(columns, row_benign, row_malicious):
    col_w = max(max(len(c) for c in columns), 9) + 2
    header = " " * 14 + "".join(f"{c:>{col_w}}" for c in columns)
    line_benign = f"  {'benign':<12}" + "".join(f"{v:>{col_w}}" for v in row_benign)
    line_malicious = f"  {'malicious':<12}" + "".join(f"{v:>{col_w}}" for v in row_malicious)
    return "\n".join([header, line_benign, line_malicious])


def _format_metrics_line(metrics):
    return (f"  balanced_accuracy (this task) = {metrics['balanced_accuracy']:.4f}, "
            f"pooled_accuracy = {metrics['pooled_accuracy']:.4f}, "
            f"mean_accuracy (mean per-task balanced accuracy) = {metrics['mean_accuracy']:.4f}")


def write_test_accuracy_breakdown(log_path, task_id, pre_stats, pre_metrics, mal_label, benign_label):
    """MADAR version -- no post-unlearning phase (no unlearning step)."""
    n_pert = pre_stats["n_perturbed_mal"] + pre_stats["n_perturbed_ben"]
    with open(log_path, "a") as f:
        f.write(f"[Test accuracy breakdown] Task {task_id}\n")
        f.write(f"  1. Reserved for testing (incl. recycled successes persisted this task): "
                f"{pre_stats['n_test_total']} total\n")
        f.write(f"     Perturbed & actually used (counted in the reported accuracy above): "
                f"{n_pert} ({pre_stats['n_perturbed_mal']} malicious + {pre_stats['n_perturbed_ben']} benign)\n")
        f.write(f"  2. Perturbed samples still classified CORRECTLY after adaptation: "
                f"malicious {_ratio_str(pre_stats['mal_correct'], pre_stats['n_perturbed_mal'])}, "
                f"benign {_ratio_str(pre_stats['ben_correct'], pre_stats['n_perturbed_ben'])}\n")
        f.write(f"  3. Clean samples correctly classified after adaptation: "
                f"{_ratio_str(pre_stats['clean_correct'], pre_stats['clean_total'])}\n")
        f.write("  [Confusion matrix] (rows = predicted, columns = true category)\n")
        f.write(_format_confusion_matrix(*_confusion_matrix_4x2(pre_stats, mal_label, benign_label)) + "\n")
        f.write(_format_metrics_line(pre_metrics) + "\n\n")


# ---------------------------------------------------------------------------
# TEMPORARY DIAGNOSTIC (pocket-targeting investigation) -- MADAR version.
# Ported from madar_unlearning_cl_pipeline.py so the two pipelines' pocket-
# targeting behavior can be compared directly (same run, same seed, same
# poisoning -- see the module docstring's note on why the two stay in sync).
# Only TWO checkpoints here, not three: MADAR has no unlearning step, so
# C2 (after this task's CL training) is also this task's FINAL state -- there
# is no C3 to compare against. Not part of the pipeline's normal metrics.
# ---------------------------------------------------------------------------
def write_pocket_targeting_diagnostic(log_path, task_id, entries, benign_label, mal_label):
    """
    Appends one human-readable section to `log_path` for this task's
    test-side perturbed samples (both malicious_perturbed and
    benign_perturbed). `entries` is a list of dicts, each:
        {"gid": int, "group": "malicious_perturbed"|"benign_perturbed",
         "true_label": int, "c1_pred": int, "c2_pred": int,
         "source": str, optional}
    `source`, when present (e.g. "recycled_from_task_3"), marks a row drawn
    from an EARLIER task's own test split (recycled into this task's
    perturbable pool -- see the TEST-SIDE blocks in main()) rather than this
    task's own test rows; absent/None means "this task's own test row". Its
    `gid` is still the row's ORIGINAL global id from whichever task it
    actually came from -- never reassigned.
    C1 = classifier at the end of the previous task (what the red test
    agents actually perturbed against). C2 = after this task's CL training
    -- MADAR's FINAL state for this task, since there's no unlearning step
    to follow it with (contrast madar_unlearning_cl_pipeline.py's version,
    which also tracks C3).

    Since success (evading C2) is unconditionally required to be logged as
    poisoned, "C1 correct -> C2 WRONG" is the only pattern
    that means anything ("a genuine pocket -- this task's training flipped a
    previously-correct sample"); "C1 WRONG -> C2 WRONG" just means the
    sample was already a pre-existing error, unrelated to this task.
    """
    if not entries:
        with open(log_path, "a") as f:
            f.write(f"--- Task {task_id} " + "-" * 50 + "\n"
                    "No test-side perturbed samples this task (no red agents ran, "
                    "POISON_TEST_DATA off, or nothing evaded) -- nothing to check.\n\n")
        return

    def _fmt(pred, true_label):
        correct = (pred == true_label)
        tag = "correct" if correct else "WRONG  "
        return tag, correct

    pattern_counts = {}
    lines_by_group = {"malicious_perturbed": [], "benign_perturbed": []}
    for e in entries:
        true_label = e["true_label"]
        true_name = "malicious" if true_label == mal_label else "benign"
        c1_tag, c1_ok = _fmt(e["c1_pred"], true_label)
        c2_tag, c2_ok = _fmt(e["c2_pred"], true_label)
        pattern = f"{'correct' if c1_ok else 'WRONG'} -> {'correct' if c2_ok else 'WRONG'}"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        pred_name = lambda p: "malicious" if p == mal_label else "benign"
        source = e.get("source")
        source_tag = f"  [{source}]" if source else ""
        lines_by_group[e["group"]].append(
            f"    gid={e['gid']:<8} true={true_name:<9} "
            f"C1={c1_tag}(pred={pred_name(e['c1_pred'])})  "
            f"C2={c2_tag}(pred={pred_name(e['c2_pred'])})"
            f"{source_tag}"
        )

    n_mal = len(lines_by_group["malicious_perturbed"])
    n_ben = len(lines_by_group["benign_perturbed"])
    n_recycled = sum(1 for e in entries if e.get("source"))

    with open(log_path, "a") as f:
        f.write(f"--- Task {task_id} " + "-" * 20 + "\n")
        f.write(f"{len(entries)} total perturbed test samples "
                f"({n_mal} malicious_perturbed, {n_ben} benign_perturbed"
                f"{f', {n_recycled} recycled from an earlier task' if n_recycled else ''})\n\n")
        f.write("Summary by C1 -> C2 correctness pattern:\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda kv: -kv[1]):
            note = ""
            if pattern == "correct -> WRONG":
                note = "  <- genuine pocket -- this task's training flipped a previously-correct sample"
            elif pattern.startswith("WRONG"):
                note = "  <- C1 was already wrong here (pre-existing error, not this task's doing)"
            f.write(f"  {pattern:<20}: {count:>4}{note}\n")
        f.write("\n")

        if lines_by_group["malicious_perturbed"]:
            f.write(f"  Malicious-perturbed samples ({n_mal}):\n")
            f.write("\n".join(lines_by_group["malicious_perturbed"]) + "\n\n")
        if lines_by_group["benign_perturbed"]:
            f.write(f"  Benign-perturbed samples ({n_ben}):\n")
            f.write("\n".join(lines_by_group["benign_perturbed"]) + "\n\n")


# ---------------------------------------------------------------------------
# Plots (identical to naive_cl_pipeline.py, titles relabeled MADAR)
# ---------------------------------------------------------------------------
def plot_task_metrics(results, out_path):
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc = [r["mean_per_task_balanced_accuracy"] for r in results]
    # BUGFIX (C1-correct-only sampling prototype): red_agent can be a
    # non-empty dict missing train_evasion_rate/test_evasion_rate
    # specifically -- e.g. the malicious train-side agent was skipped (empty
    # C1-correct pool) but the benign train-side agent ran and populated its
    # own keys into the SAME dict, or vice versa. .get() instead of direct
    # indexing so a partially-populated red_agent doesn't crash the plot.
    # Kept in step with madar_unlearning_cl_pipeline.py's identical fix.
    train_evasion = [r["red_agent"].get("train_evasion_rate") if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"].get("test_evasion_rate") if r["red_agent"] else None for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc, marker="o", label="pooled balanced accuracy")
    ax1.plot(task_ids, mean_per_task_bal_acc, marker="s", label="mean per-task balanced accuracy")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("MADAR CL (ER+KD+SI): classifier accuracy per task boundary")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # BUGFIX: train_evasion/test_evasion can now be None at DIFFERENT task
    # indices from each other -- a single xs shared between both series
    # would misalign them, or crash outright if the two None-patterns
    # differ in count. Filter each series against its own None pattern.
    xs_train = [t for t, v in zip(task_ids, train_evasion) if v is not None]
    ys_train = [v for v in train_evasion if v is not None]
    xs_test = [t for t, v in zip(task_ids, test_evasion) if v is not None]
    ys_test = [v for v in test_evasion if v is not None]
    ax2.plot(xs_train, ys_train, marker="o", color="tab:red", label="evasion rate (train samples)")
    ax2.plot(xs_test, ys_test, marker="s", color="tab:orange", label="evasion rate (held-out test samples)")
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


def compute_pocket_mask(proba_before, proba_after, threshold=0.5):
    """
    PROTOTYPE (adaptation-split pocket highlighting): boolean grid-cell mask
    over the SAME fixed-PCA grid shape as _decision_grid's output, True
    wherever the predicted class (thresholded at `threshold`, matching the
    plot's own 0.5 contour line) differs between two checkpoints' probability
    grids -- i.e. the region the decision surface moved through between them.
    Kept in step with madar_unlearning_cl_pipeline.py's identical copy.
    """
    return (proba_before >= threshold) != (proba_after >= threshold)


def _draw_decision_boundary(ax, xx, yy, proba, Z_train, category_train, Z_test, category_test,
                             highlight_mask=None):
    """Draws one panel from PRECOMPUTED grid/scatter data onto `ax`. Shared
    by plot_decision_boundary (computes then draws, saves its own file) and
    plot_decision_boundary_grid (redraws every checkpoint's cached data into
    one summary figure). Background categories (benign/malicious_clean) are
    TRAIN-split only, small and faint; perturbed categories are drawn from
    BOTH train and test splits, colored+marker-coded by which agent
    produced them. Returns per-agent perturbed counts (for the caller's
    legend).

    highlight_mask (PROTOTYPE, adaptation-split pocket highlighting): optional
    boolean grid (same shape as `proba`, see compute_pocket_mask) drawn as a
    translucent green wash + outline UNDER the scatter points, so a "pocket"
    -- a grid cell where the predicted class has flipped at ANY checkpoint so
    far this task -- stays visibly marked even on later checkpoints where the
    background probability itself has changed again.
    """
    ax.contourf(xx, yy, proba, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=1.2, linestyles="--")

    if highlight_mask is not None and highlight_mask.any():
        ax.contourf(xx, yy, highlight_mask.astype(float), levels=[0.5, 1.5], colors=["#39FF14"], alpha=0.35)
        ax.contour(xx, yy, highlight_mask.astype(float), levels=[0.5], colors="#0a7d0a", linewidths=1.0)

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
                            task_id, out_path, checkpoint_label="",
                            highlight_mask=None, precomputed_grid=None):
    """One task/checkpoint's decision-boundary figure. Returns a dict of the
    computed grid/scatter data so main() can cache it and reuse it in
    plot_decision_boundary_grid's end-of-run summary without recomputing.

    precomputed_grid (PROTOTYPE, adaptation-split pocket highlighting):
    optional (xx, yy, proba) triple -- when the caller already computed this
    checkpoint's grid via _decision_grid to diff it against the previous
    checkpoint (compute_pocket_mask), pass it here so this function doesn't
    redundantly recompute the same forward pass. highlight_mask is forwarded
    straight to _draw_decision_boundary -- see its docstring.
    """
    if precomputed_grid is not None:
        xx, yy, proba = precomputed_grid
    else:
        xx, yy, proba = _decision_grid(model, pca_mean, pca_components, pca_extent, device, mal_label)
    Z_train = _pca_project(X_train_scaled_np, pca_mean, pca_components)
    Z_test = _pca_project(X_test_scaled_np, pca_mean, pca_components)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_by_agent = _draw_decision_boundary(ax, xx, yy, proba, Z_train, category_train, Z_test, category_test,
                                          highlight_mask=highlight_mask)

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
        "highlight_mask": highlight_mask,
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
                                 p["Z_test"], p["category_test"], highlight_mask=p.get("highlight_mask"))
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

    # TEMPORARY DIAGNOSTIC (pocket-targeting investigation) -- ported from
    # madar_unlearning_cl_pipeline.py so the two pipelines' pocket-targeting
    # behavior can be compared directly. See write_pocket_targeting_
    # diagnostic()'s docstring for the field definitions -- only C1/C2 here
    # (no C3/unlearning), unlike that file's version.
    pocket_diag_log_path = os.path.join(out_dir, "logs", "pocket_targeting_diagnostic.txt")
    with open(pocket_diag_log_path, "w") as f:
        f.write(
            "POCKET-TARGETING DIAGNOSTIC LOG (MADAR)\n"
            "========================================\n"
            "Checks whether red_test_pert_agent/red_test_benign_pert_agent are landing\n"
            "where the decision boundary actually moves. One section per task (tasks > 0\n"
            "only -- task 0 has no red agents/perturbed test samples).\n\n"
            "Two classifier snapshots, both scored on the SAME perturbed test sample:\n"
            "  C1 = classifier at the END of the PREVIOUS task -- this is the exact\n"
            "       classifier state red_test_pert_agent/red_test_benign_pert_agent\n"
            "       perturbed this sample against.\n"
            "  C2 = classifier AFTER this task's CL training (ER+KD+SI). Plain MADAR has\n"
            "       no unlearning step, so C2 is also this task's FINAL state -- there is\n"
            "       no C3 here, unlike madar_unlearning_cl_pipeline.py's version of this\n"
            "       log.\n"
            "'correct'/'WRONG' always means: does the prediction match the sample's TRUE\n"
            "(pre-perturbation) label -- malicious for malicious_perturbed rows, benign for\n"
            "benign_perturbed rows.\n\n"
            "Each task's perturbable test pool is the UNION of that task's own C1-correct\n"
            "test rows AND task (t-1)'s own C1-correct test rows, recycled in (train-side\n"
            "pools are NOT C1-filtered -- they batch straight from the task's own training\n"
            "data). A recycled row's perturbed value, if successful, is now APPENDED as a\n"
            "new row into the recycling task's OWN test split (task_test_splits[t]) -- it\n"
            "never overwrites task (t-1)'s own copy of that row, so the same physical\n"
            "sample can appear twice across the run's cumulative pooled_eval (once as\n"
            "task (t-1)'s original clean row, once here). Entries below tagged\n"
            "'[recycled_from_task_N]' are these; entries without that tag are this task's\n"
            "own test rows.\n\n"
        )

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
        agent_ids=[f"{t}_{typ}" for t in range(NUM_TASKS)
                   for typ in ("train", "test", "train_benign", "test_benign")],
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
        benign_idx_train = np.where(y_train == benign_label)[0]
        red_report = None
        n_poisoned = 0
        poison_idx = np.array([], dtype=int)
        n_poisoned_test = 0
        poison_idx_test = np.array([], dtype=int)  # initialized here (not just inside the nested
                                                     # TEST-SIDE block) so category_test construction
                                                     # below can always safely reference it
        poisoned_test_sample_ids = []
        n_poisoned_benign = 0
        poison_idx_benign = np.array([], dtype=int)
        # NEW (variant-augmentation branch): safe empty defaults -- see
        # madar_unlearning_cl_pipeline.py's identical addition at the same
        # point for the full rationale.
        _empty_variants = {"X": np.empty((0, X.shape[1]), dtype=np.float32),
                            "internal_gids": np.array([], dtype=np.int64),
                            "display_gids": [], "n_variants_by_original": {}}
        mal_variants = dict(_empty_variants)
        ben_variants = dict(_empty_variants)
        n_poisoned_test_benign = 0
        poison_idx_test_benign = np.array([], dtype=int)
        poisoned_test_sample_ids_benign = []
        # REWORK (recycled-pocket test pool): mirror of
        # madar_unlearning_cl_pipeline.py's identical addition -- see its
        # docstring at the same point for the full rationale.
        recycled_gids_mal = np.array([], dtype=np.int64)
        recycled_Xpert_mal = np.empty((0, X.shape[1]), dtype=np.float32)
        recycled_gids_ben = np.array([], dtype=np.int64)
        recycled_Xpert_ben = np.empty((0, X.shape[1]), dtype=np.float32)

        if t == 0:
            print(f"\n===== Task 0 (warm start, no red agent): "
                  f"{len(y_train)} train / {len(y_test)} test =====")
            X_train_for_classifier = X_train

        else:
            print(f"\n===== Task {t}: training fresh red agent against pre-task-{t} classifier "
                  f"({len(y_train)} train / {len(y_test)} test, {len(mal_idx_train)} malicious train) =====")

            mal_idx_test = np.where(y_test == mal_label)[0]
            benign_idx_test = np.where(y_test == benign_label)[0]

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
                # REWORK (uncertainty-margin pipeline): the attack pool is no
                # longer "everything" or "C1-correct only" -- it's the top
                # TRAIN_UNCERTAIN_FRACTION (30%) of this class's own rows,
                # ranked by ascending confidence under C1 (most uncertain --
                # closest to C1's decision boundary -- first). c1_correct_pool()
                # is still called and logged purely as an FYI diagnostic,
                # separate from this selection. Kept in step with
                # madar_unlearning_cl_pipeline.py's identical change.
                _ = c1_correct_pool(
                    classifier_wrapper, X_train, mal_idx_train, mal_label,
                    t, "train-side malicious (FYI only, not filtered)", pocket_diag_log_path,
                )
                mal_train_ranked = _uncertainty_rank(classifier_wrapper, X_train, mal_idx_train)
                n_attack_mal_train = max(1, round(TRAIN_UNCERTAIN_FRACTION * len(mal_idx_train)))
                attack_pool_mal_train = mal_train_ranked[:n_attack_mal_train]
                print(f"    [Uncertainty selection] Task {t} train-side malicious: attacking "
                      f"{len(attack_pool_mal_train)}/{len(mal_idx_train)} "
                      f"({TRAIN_UNCERTAIN_FRACTION:.0%}, most-uncertain-under-C1 first).")

                env_train, agent_train = train_red_agent_for_task(
                    t, "train", classifier_wrapper, X_train, y_train, benign_label, bank, args.seed, out_dir,
                    target_margin_confidence=RED_TRAIN_TARGET_MARGIN_CONFIDENCE,
                    allowed_start_indices=attack_pool_mal_train,
                )
                X_train_pert, train_evasion_rate, train_rewards, train_avg_norm, train_attacked, evaded_mask = \
                    evaluate_agent_on_batch(
                        env_train, agent_train, X_train, y_train, benign_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                        allowed_start_indices=attack_pool_mal_train,
                    )

                red_report = {
                    "train_evasion_rate": train_evasion_rate, "train_attacked": train_attacked,
                    "train_avg_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
                    "train_avg_pert_l2": train_avg_norm,
                }
                print(f"[Task {t} red_train_pert_agent] train_evasion={train_evasion_rate:.3f} "
                      f"avg_pert_L2={train_avg_norm:.3f}")

                # REWORK: no separate quota/subsample step anymore -- the attack
                # pool above IS the quota (already capped at 30%), and success is
                # unconditionally required (no REQUIRE_EVASION_SUCCESS toggle), so
                # every episode that successfully flips (evaded_mask=True) simply
                # becomes the poisoned set. Actual n_poisoned can land under 30%
                # of len(mal_idx_train) if not every attempt succeeds.
                poison_idx = np.where(evaded_mask)[0]
                n_poisoned = len(poison_idx)

                print(f"[Task {t}] poisoned {n_poisoned}/{len(attack_pool_mal_train)} attacked "
                      f"({n_poisoned}/{len(mal_idx_train)} of all malicious train samples) "
                      f"malicious train samples (attack quota was {TRAIN_UNCERTAIN_FRACTION:.0%})")

                # NEW (variant-augmentation branch): see generate_train_variants's
                # docstring (madar_unlearning_cl_pipeline.py) for the full
                # rationale -- mirrored unchanged.
                mal_variants = generate_train_variants(
                    env_train, agent_train, X_train_pert, evaded_mask, gid_train, true_label=mal_label,
                )
                n_mal_variants = len(mal_variants["internal_gids"])
                print(f"    [Train variants] Task {t} malicious: {len(mal_variants['n_variants_by_original'])}/"
                      f"{n_poisoned} evaded originals produced {n_mal_variants} variant(s) "
                      f"(perturbed so far: {n_poisoned + n_mal_variants}, clean target: "
                      f"{len(mal_idx_train) - n_poisoned})")

                X_train_for_classifier = X_train.copy()
                X_train_for_classifier[poison_idx] = X_train_pert[poison_idx]

                # TEST-SIDE poisoning (POISON_TEST_DATA -- see module docstring
                # BRANCH NOTE and POISON_TEST_DATA's definition for why).
                # REWORK (pocket-targeted test poisoning): red_test_pert_agent now
                # trains AFTER this task's CL training finishes, not here -- see
                # the TEST-SIDE block right after train_cl_er, below, and
                # pre_train_classifier_wrapper's capture just before it.

                # BENIGN-SIDE perturbation agents (REWORK, quad red agents):
                # mirror-image objective of the malicious pair above -- attack
                # BENIGN rows, reward/terminate on being misclassified as
                # mal_label instead of benign_label. Reuses NetworkAttackEnv/
                # train_red_agent_for_task/evaluate_agent_on_batch completely
                # unmodified -- passing mal_label in as the "benign_label"
                # argument is sufficient (see train_red_agent_for_task's
                # docstring). Registered in the SAME contrastive bank as the
                # malicious agents, so all four of this task's agents push
                # each other apart via the existing recency-weighted
                # mechanism -- no new diversity logic needed. Ground truth
                # label is left as benign_label on substitution, matching how
                # malicious_perturbed keeps its true malicious label -- only
                # X changes, not y. Uses the SAME RED_TRAIN_TARGET_MARGIN_
                # CONFIDENCE margin bias as the malicious side (see
                # train_red_agent_for_task's docstring).
                if len(benign_idx_train) == 0:
                    warnings_log.append(f"Task {t}: no benign training samples, skipping benign red agents.")
                    X_train_pert_benign = X_train.copy()  # placeholder, mirrors the malicious side
                else:
                    # REWORK (uncertainty-margin pipeline): mirror of the
                    # malicious side above -- top TRAIN_UNCERTAIN_FRACTION (30%)
                    # of benign_idx_train, most-uncertain-under-C1 first.
                    # c1_correct_pool() stays FYI-only.
                    _ = c1_correct_pool(
                        classifier_wrapper, X_train, benign_idx_train, benign_label,
                        t, "train-side benign (FYI only, not filtered)", pocket_diag_log_path,
                    )
                    benign_train_ranked = _uncertainty_rank(classifier_wrapper, X_train, benign_idx_train)
                    n_attack_benign_train = max(1, round(TRAIN_UNCERTAIN_FRACTION * len(benign_idx_train)))
                    attack_pool_benign_train = benign_train_ranked[:n_attack_benign_train]
                    print(f"    [Uncertainty selection] Task {t} train-side benign: attacking "
                          f"{len(attack_pool_benign_train)}/{len(benign_idx_train)} "
                          f"({TRAIN_UNCERTAIN_FRACTION:.0%}, most-uncertain-under-C1 first).")

                    env_train_benign, agent_train_benign = train_red_agent_for_task(
                        t, "train_benign", classifier_wrapper, X_train, y_train, mal_label, bank, args.seed, out_dir,
                        target_margin_confidence=RED_TRAIN_TARGET_MARGIN_CONFIDENCE,
                        allowed_start_indices=attack_pool_benign_train,
                    )
                    (X_train_pert_benign, train_evasion_rate_benign, train_rewards_benign, train_avg_norm_benign,
                     train_attacked_benign, evaded_mask_benign) = evaluate_agent_on_batch(
                        env_train_benign, agent_train_benign, X_train, y_train, mal_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                        allowed_start_indices=attack_pool_benign_train,
                    )

                    red_report["train_benign_evasion_rate"] = train_evasion_rate_benign
                    red_report["train_benign_attacked"] = train_attacked_benign
                    red_report["train_benign_avg_reward"] = float(np.mean(train_rewards_benign)) if train_rewards_benign else 0.0
                    red_report["train_benign_avg_pert_l2"] = train_avg_norm_benign
                    print(f"[Task {t} red_train_benign_pert_agent] train_evasion={train_evasion_rate_benign:.3f} "
                          f"avg_pert_L2={train_avg_norm_benign:.3f}")

                    # REWORK: no separate quota/subsample step -- see the malicious
                    # side's identical note above.
                    poison_idx_benign = np.where(evaded_mask_benign)[0]
                    n_poisoned_benign = len(poison_idx_benign)

                    print(f"[Task {t}] poisoned {n_poisoned_benign}/{len(attack_pool_benign_train)} attacked "
                          f"({n_poisoned_benign}/{len(benign_idx_train)} of all benign train samples) "
                          f"benign train samples (attack quota was {TRAIN_UNCERTAIN_FRACTION:.0%})")

                    # NEW (variant-augmentation branch): mirror of the malicious
                    # side's identical addition above.
                    ben_variants = generate_train_variants(
                        env_train_benign, agent_train_benign, X_train_pert_benign, evaded_mask_benign, gid_train,
                        true_label=benign_label,
                    )
                    n_ben_variants = len(ben_variants["internal_gids"])
                    print(f"    [Train variants] Task {t} benign: {len(ben_variants['n_variants_by_original'])}/"
                          f"{n_poisoned_benign} evaded originals produced {n_ben_variants} variant(s) "
                          f"(perturbed so far: {n_poisoned_benign + n_ben_variants}, clean target: "
                          f"{len(benign_idx_train) - n_poisoned_benign})")

                    X_train_for_classifier[poison_idx_benign] = X_train_pert_benign[poison_idx_benign]

                    # TEST-SIDE (REWORK, pocket-targeted test poisoning): moved to
                    # after CL training, mirroring the malicious side -- see the
                    # TEST-SIDE block right after train_cl_er, below.

        # NEW (variant-augmentation branch): append this task's malicious +
        # benign variant rows -- see madar_unlearning_cl_pipeline.py's
        # identical addition at the same point for the full rationale.
        _variant_blocks = []
        if len(mal_variants["internal_gids"]) > 0:
            _variant_blocks.append(("mal", mal_variants))
        if len(ben_variants["internal_gids"]) > 0:
            _variant_blocks.append(("ben", ben_variants))
        for _tag, _v in _variant_blocks:
            _new_idx = np.arange(len(X_train_for_classifier), len(X_train_for_classifier) + len(_v["X"]))
            if _tag == "mal":
                poison_idx = np.concatenate([poison_idx, _new_idx])
                _v_label = mal_label
            else:
                poison_idx_benign = np.concatenate([poison_idx_benign, _new_idx])
                _v_label = benign_label
            X_train_for_classifier = np.concatenate([X_train_for_classifier, _v["X"]], axis=0)
            y_train = np.concatenate([y_train, np.full(len(_v["X"]), _v_label, dtype=y_train.dtype)], axis=0)
            gid_train = np.concatenate([gid_train, _v["internal_gids"]], axis=0)
        if _variant_blocks:
            n_poisoned = len(poison_idx)
            n_poisoned_benign = len(poison_idx_benign)
            print(f"    [Train variants] Task {t} total: {sum(len(v['X']) for _, v in _variant_blocks)} variant "
                  f"row(s) appended -- train set now {len(y_train)} rows "
                  f"({n_poisoned} malicious_perturbed + {n_poisoned_benign} benign_perturbed).")

        # Category tracking for the buffer-composition diagnostic ONLY (never fed
        # back into training/selection) -- benign / malicious_clean /
        # malicious_perturbed / benign_perturbed, built from TRAIN data only;
        # test poisoning above never touches training/selection.
        category = np.full(len(y_train), "malicious_clean", dtype=object)
        category[y_train == benign_label] = "benign"
        if len(poison_idx) > 0:
            category[poison_idx] = "malicious_perturbed"
        if len(poison_idx_benign) > 0:
            category[poison_idx_benign] = "benign_perturbed"
        # NEW (variant-augmentation branch): _variant_gid_display translates a
        # variant's internal gid to "<original>.<level>"; an ordinary gid
        # passes through unchanged as a plain int.
        poisoned_sample_ids = [_variant_gid_display(g) for g in gid_train[poison_idx]] if len(poison_idx) > 0 else []
        poisoned_sample_ids_benign = [_variant_gid_display(g) for g in gid_train[poison_idx_benign]] \
            if len(poison_idx_benign) > 0 else []

        # TEST-side category array is built further below, AFTER the TEST-SIDE
        # agent block that now runs post-CL-training (REWORK, pocket-targeted
        # test poisoning) -- poison_idx_test/poison_idx_test_benign aren't known
        # yet at this point in the loop.

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
                  f"MADAR ER+KD+SI, {CL_ITERS} iters (single pass)")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = 0

            # REWORK (pocket-targeted test poisoning): frozen snapshot of the
            # classifier as it stood BEFORE this task's CL training -- captured
            # here, right before train_cl_er mutates `model` in place. See the
            # TEST-SIDE block right below for how it's used. Also this task's
            # C1 for the pocket-highlight diffing below.
            pre_train_classifier_wrapper = TorchIDSWrapper(copy.deepcopy(model), scaler, DEVICE)

            # PROTOTYPE (adaptation-split pocket visualization): running
            # per-task highlight state -- see madar_unlearning_cl_pipeline.py's
            # identical block for the full rationale.
            xx0, yy0, proba_c1 = _decision_grid(
                pre_train_classifier_wrapper.model, pca_mean, pca_components, pca_extent, DEVICE, mal_label
            )
            task_pocket_mask = np.zeros_like(proba_c1, dtype=bool)
            prev_proba = proba_c1
            X_test_current_scaled_np = to_tensor(task_test_splits[t][0]).numpy()
            category_test_clean_so_far = np.full(len(y_test), "malicious_clean", dtype=object)
            category_test_clean_so_far[y_test == benign_label] = "benign"

            # REWORK (uncertainty-margin pipeline): reverted to a SINGLE
            # training pass over the full (partially poisoned) train set --
            # the two-phase perturbed/clean split is gone. Perturbed rows are
            # already swapped in place in Xtr/ytr by this point (both classes).
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(
                data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last,
            )
            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS,
                        W, omega, p_old_task, t, SI_C, DEVICE)
            grad_steps += CL_ITERS

            proba_after_adapt = _decision_grid(model, pca_mean, pca_components, pca_extent, DEVICE, mal_label)[2]
            task_pocket_mask |= compute_pocket_mask(prev_proba, proba_after_adapt)
            prev_proba = proba_after_adapt
            boundary_panels.append(plot_decision_boundary(
                model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                Xtr.numpy(), category, X_test_current_scaled_np, category_test_clean_so_far,
                t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}_adapt.png"),
                checkpoint_label="adapt", highlight_mask=task_pocket_mask,
                precomputed_grid=(xx0, yy0, proba_after_adapt),
            ))

            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_current = p.detach().clone()
                    # SI_OMEGA_DECAY (REWORK, SI capacity investigation): see
                    # its definition for the full rationale.
                    omega[n_key] = SI_OMEGA_DECAY * omega[n_key] + \
                        W[n_key] / ((p_current - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()
                    p_old_task[n_key] = p_current

            # PROTOTYPE (adaptation-split pocket visualization): MADAR has no
            # per-task diagnostic log file (that's unlearning-pipeline-only),
            # so this is console-only -- kept in step with
            # madar_unlearning_cl_pipeline.py's identical line.
            omega_l2_after_adapt = float(torch.sqrt(sum((v ** 2).sum() for v in omega.values())).item())
            print(f"    [SI omega] after adaptation: L2 norm = {omega_l2_after_adapt:.4f}")

            # TEST-SIDE poisoning (REWORK, pocket-targeted test poisoning): runs
            # HERE, after this task's CL training has finished, so
            # red_test_pert_agent/red_test_benign_pert_agent can be scored
            # against the POST-training classifier while comparing to
            # pre_train_classifier_wrapper (captured above) -- now
            # pgd_boundary_search_batch's shift_reference_classifier, re-checked
            # every PGD step (REWORK, gradient-attack branch: the dense
            # POCKET_SHIFT_WEIGHT reward bonus this used to also drive is gone,
            # only the hard joint success condition remains -- see that
            # function's definition). Only runs when this task actually had
            # train-side agents/poisoning above (mirrors the nesting this code
            # used to live inside, before the reorder).
            if len(mal_idx_train) > 0:
                # REWORK (recycled-pocket test pool): fetched once here (not
                # inside the malicious/benign sub-blocks) so both classes see it
                # regardless of which sub-block runs. Deliberately re-derived
                # from task (t-1) ALONE, fresh every task. POTENTIALLY REVISIT:
                # a longer-range pool was considered and explicitly deferred,
                # not ruled out. Mirror of madar_unlearning_cl_pipeline.py's
                # identical addition -- see its docstring at the same point
                # for the full rationale.
                if RECYCLE_TEST_POCKETS:
                    X_prev_test, y_prev_test = task_test_splits[t - 1]
                    gid_prev_test = task_test_gids[t - 1]
                else:
                    # RECYCLE_TEST_POCKETS off: force the "recycled" half of every
                    # union below to be empty -- see its definition for the full
                    # rationale.
                    X_prev_test = np.empty((0, X.shape[1]), dtype=X.dtype)
                    y_prev_test = np.empty((0,), dtype=y.dtype)
                    gid_prev_test = np.empty((0,), dtype=np.int64)

                if POISON_TEST_DATA and len(mal_idx_test) > 0:
                    mal_idx_prev_test = np.where(y_prev_test == mal_label)[0]

                    n_own_test = len(X_test)
                    X_test_ext = np.concatenate([X_test, X_prev_test], axis=0)
                    y_test_ext = np.concatenate([y_test, y_prev_test], axis=0)
                    mal_idx_test_ext = np.concatenate([mal_idx_test, mal_idx_prev_test + n_own_test])

                    mal_idx_test_c1 = c1_correct_pool(
                        pre_train_classifier_wrapper, X_test_ext, mal_idx_test_ext, mal_label,
                        t, f"test-side malicious (this task + recycled task {t - 1})", pocket_diag_log_path,
                    )
                    # REWORK (uncertainty-margin pipeline): tiered selection --
                    # tier 1 = C1-correct candidates, tier 2 = C1-incorrect
                    # candidates padded in once tier 1 runs out, up to
                    # TEST_AGGREGATE_FRACTION (40%) of THIS TASK's own malicious
                    # test row count. Tier-2 episodes can essentially never
                    # satisfy the joint C1-correct/C2-wrong success condition
                    # below, so padding trades a lower per-attempt success rate
                    # for more total attempts -- an accepted, deliberate tradeoff.
                    #
                    # FINETUNE (test-side selection/patience investigation):
                    # both tiers now ranked by most-uncertain-under-C2
                    # (classifier_wrapper) instead of under-C1 -- see
                    # madar_unlearning_cl_pipeline.py's identical block for the
                    # full rationale (ranking by C1-uncertainty was picking
                    # points that self-sabotaged against the C1-correctness
                    # requirement). C1-correctness is still required to ENTER
                    # tier 1 (c1_correct_pool above, unchanged).
                    n_target_mal_test = max(1, round(TEST_AGGREGATE_FRACTION * len(mal_idx_test)))
                    mal_tier1 = _uncertainty_rank(classifier_wrapper, X_test_ext, mal_idx_test_c1)
                    mal_tier2_pool = np.setdiff1d(mal_idx_test_ext, mal_idx_test_c1)
                    mal_tier2 = _uncertainty_rank(classifier_wrapper, X_test_ext, mal_tier2_pool)
                    n_tier1_used_mal = min(len(mal_tier1), n_target_mal_test)
                    attack_pool_mal_test = mal_tier1[:n_target_mal_test]
                    if len(attack_pool_mal_test) < n_target_mal_test:
                        attack_pool_mal_test = np.concatenate([
                            attack_pool_mal_test, mal_tier2[:n_target_mal_test - len(attack_pool_mal_test)]
                        ])
                    n_tier2_used_mal = len(attack_pool_mal_test) - n_tier1_used_mal
                    print(f"    [Uncertainty selection] Task {t} test-side malicious: attacking "
                          f"{len(attack_pool_mal_test)}/{n_target_mal_test} target "
                          f"({n_tier1_used_mal} C1-correct, {n_tier2_used_mal} padded C1-incorrect).")

                    if len(attack_pool_mal_test) == 0:
                        warnings_log.append(
                            f"Task {t}: no C1-correct malicious test candidates (own or recycled), "
                            f"skipping red_test_pert_agent."
                        )
                    else:
                        # NEW (gradient-attack branch): test-side malicious agent is
                        # now pgd_boundary_search_batch -- see its definition
                        # (madar_unlearning_cl_pipeline.py has the full rationale) for
                        # why the RL rollout/reward-shaping machinery above is gone.
                        X_test_ext_pert, test_evasion_rate, test_rewards, test_avg_norm, test_attacked, evaded_mask_test = \
                            pgd_boundary_search_batch(
                                classifier_wrapper, X_test_ext, y_test_ext, benign_label,
                                allowed_start_indices=attack_pool_mal_test,
                                shift_reference_classifier=pre_train_classifier_wrapper,
                                epsilon=RED_EPSILON, max_steps=RED_MAX_STEPS,
                                margin_low=RED_TARGET_MARGIN_CONFIDENCE, margin_high=RED_TARGET_MARGIN_HIGH,
                                max_test=MAX_EVAL_SAMPLES_PER_TASK, c1_patience=RED_C1_MISMATCH_PATIENCE,
                                diag_log_path=pocket_diag_log_path, diag_label=f"Task {t} test-side malicious",
                            )
                        red_report["test_evasion_rate"] = test_evasion_rate
                        red_report["test_attacked"] = test_attacked
                        red_report["test_avg_reward"] = float(np.mean(test_rewards)) if test_rewards else 0.0
                        red_report["test_avg_pert_l2"] = test_avg_norm
                        red_report["test_n_tier1_c1_correct"] = int(n_tier1_used_mal)
                        red_report["test_n_tier2_padded"] = int(n_tier2_used_mal)
                        print(f"[Task {t} red_test_pert_agent] test_evasion={test_evasion_rate:.3f} "
                              f"avg_pert_L2={test_avg_norm:.3f}")

                        # REWORK: no separate quota/subsample step anymore -- the
                        # attack pool above IS the quota (already capped at 40% of
                        # this task's own malicious test count), and evaded_mask_
                        # test's "success" already IS the joint C1-correct/C2-wrong
                        # condition, so every episode that satisfies it becomes the
                        # poisoned set directly.
                        poison_idx_test_ext = np.where(evaded_mask_test)[0]

                        own_mask_test = poison_idx_test_ext < n_own_test
                        poison_idx_test_own = poison_idx_test_ext[own_mask_test]
                        recycled_ext_idx_mal = poison_idx_test_ext[~own_mask_test]
                        recycled_local_idx_mal = recycled_ext_idx_mal - n_own_test
                        n_poisoned_test_own = len(poison_idx_test_own)

                        X_test = X_test.copy()
                        X_test[poison_idx_test_own] = X_test_ext_pert[poison_idx_test_own]

                        recycled_gids_mal = gid_prev_test[recycled_local_idx_mal]
                        recycled_Xpert_mal = X_test_ext_pert[recycled_ext_idx_mal]
                        n_recycled_mal = len(recycled_gids_mal)

                        # REWORK (recycled-pocket test pool, now persisted): recycled
                        # successes are appended as NEW rows into task_test_splits[t]/
                        # task_test_gids[t] (they have no "slot" to overwrite in this
                        # task's own split) so they now count toward this task's own
                        # per_task_eval/pooled_eval/perturbed_test_eval, same as an
                        # own-task success. Confirmed tradeoff: the same physical
                        # sample can now appear TWICE across pooled_eval's cumulative
                        # test corpus this run -- once as task (t-1)'s original clean
                        # row (still sitting unmodified in task_test_splits[t-1]),
                        # once here as task t's perturbed row -- not deduplicated.
                        if n_recycled_mal > 0:
                            recycled_new_idx_mal = np.arange(len(X_test), len(X_test) + n_recycled_mal)
                            X_test = np.concatenate([X_test, recycled_Xpert_mal], axis=0)
                            y_test = np.concatenate(
                                [y_test, np.full(n_recycled_mal, mal_label, dtype=y_test.dtype)], axis=0
                            )
                            gid_test = np.concatenate([gid_test, recycled_gids_mal], axis=0)
                        else:
                            recycled_new_idx_mal = np.array([], dtype=np.int64)

                        poison_idx_test = np.concatenate([poison_idx_test_own, recycled_new_idx_mal]).astype(np.int64)
                        n_poisoned_test = len(poison_idx_test)
                        task_test_splits[t] = (X_test, y_test)
                        task_test_gids[t] = gid_test
                        poisoned_test_sample_ids = gid_test[poison_idx_test].tolist() if n_poisoned_test > 0 else []

                        print(f"[Task {t}] poisoned {n_poisoned_test}/{len(attack_pool_mal_test)} attacked "
                              f"({n_tier1_used_mal} C1-correct, {n_tier2_used_mal} padded) malicious TEST samples "
                              f"({n_poisoned_test_own} own + {n_recycled_mal} recycled from task {t - 1}'s test "
                              f"split, both now persisted into task_test_splits[{t}]) -- "
                              f"success rate on C1-correct starts was "
                              f"{(n_poisoned_test / n_tier1_used_mal) if n_tier1_used_mal > 0 else float('nan'):.3f}")

                if len(benign_idx_train) > 0 and POISON_TEST_DATA and len(benign_idx_test) > 0:
                    # REWORK (recycled-pocket test pool): exact mirror of the
                    # malicious block above -- X_prev_test/y_prev_test/
                    # gid_prev_test are already fetched above.
                    benign_idx_prev_test = np.where(y_prev_test == benign_label)[0]

                    n_own_test_benign = len(X_test)
                    X_test_ext_benign = np.concatenate([X_test, X_prev_test], axis=0)
                    y_test_ext_benign = np.concatenate([y_test, y_prev_test], axis=0)
                    benign_idx_test_ext = np.concatenate([benign_idx_test, benign_idx_prev_test + n_own_test_benign])

                    benign_idx_test_c1 = c1_correct_pool(
                        pre_train_classifier_wrapper, X_test_ext_benign, benign_idx_test_ext, benign_label,
                        t, f"test-side benign (this task + recycled task {t - 1})", pocket_diag_log_path,
                    )
                    # REWORK (uncertainty-margin pipeline): mirror of the
                    # malicious tiered selection above -- ranked by C2
                    # (classifier_wrapper), not C1, per the same FINETUNE
                    # rationale noted there.
                    n_target_ben_test = max(1, round(TEST_AGGREGATE_FRACTION * len(benign_idx_test)))
                    ben_tier1 = _uncertainty_rank(classifier_wrapper, X_test_ext_benign, benign_idx_test_c1)
                    ben_tier2_pool = np.setdiff1d(benign_idx_test_ext, benign_idx_test_c1)
                    ben_tier2 = _uncertainty_rank(classifier_wrapper, X_test_ext_benign, ben_tier2_pool)
                    n_tier1_used_ben = min(len(ben_tier1), n_target_ben_test)
                    attack_pool_ben_test = ben_tier1[:n_target_ben_test]
                    if len(attack_pool_ben_test) < n_target_ben_test:
                        attack_pool_ben_test = np.concatenate([
                            attack_pool_ben_test, ben_tier2[:n_target_ben_test - len(attack_pool_ben_test)]
                        ])
                    n_tier2_used_ben = len(attack_pool_ben_test) - n_tier1_used_ben
                    print(f"    [Uncertainty selection] Task {t} test-side benign: attacking "
                          f"{len(attack_pool_ben_test)}/{n_target_ben_test} target "
                          f"({n_tier1_used_ben} C1-correct, {n_tier2_used_ben} padded C1-incorrect).")

                    if len(attack_pool_ben_test) == 0:
                        warnings_log.append(
                            f"Task {t}: no C1-correct benign test candidates (own or recycled), "
                            f"skipping red_test_benign_pert_agent."
                        )
                    else:
                        # NEW (gradient-attack branch): mirror of the malicious
                        # test-side block above.
                        (X_test_ext_pert_benign, test_evasion_rate_benign, test_rewards_benign, test_avg_norm_benign,
                         test_attacked_benign, evaded_mask_test_benign) = pgd_boundary_search_batch(
                            classifier_wrapper, X_test_ext_benign, y_test_ext_benign, mal_label,
                            allowed_start_indices=attack_pool_ben_test,
                            shift_reference_classifier=pre_train_classifier_wrapper,
                            epsilon=RED_EPSILON, max_steps=RED_MAX_STEPS,
                            margin_low=RED_TARGET_MARGIN_CONFIDENCE, margin_high=RED_TARGET_MARGIN_HIGH,
                            max_test=MAX_EVAL_SAMPLES_PER_TASK, c1_patience=RED_C1_MISMATCH_PATIENCE,
                            diag_log_path=pocket_diag_log_path, diag_label=f"Task {t} test-side benign",
                        )
                        red_report["test_benign_evasion_rate"] = test_evasion_rate_benign
                        red_report["test_benign_attacked"] = test_attacked_benign
                        red_report["test_benign_avg_reward"] = float(np.mean(test_rewards_benign)) if test_rewards_benign else 0.0
                        red_report["test_benign_avg_pert_l2"] = test_avg_norm_benign
                        red_report["test_benign_n_tier1_c1_correct"] = int(n_tier1_used_ben)
                        red_report["test_benign_n_tier2_padded"] = int(n_tier2_used_ben)
                        print(f"[Task {t} red_test_benign_pert_agent] test_evasion={test_evasion_rate_benign:.3f} "
                              f"avg_pert_L2={test_avg_norm_benign:.3f}")

                        # REWORK: no separate quota/subsample step -- see the
                        # malicious side's identical note above.
                        poison_idx_test_ext_benign = np.where(evaded_mask_test_benign)[0]

                        own_mask_test_benign = poison_idx_test_ext_benign < n_own_test_benign
                        poison_idx_test_benign_own = poison_idx_test_ext_benign[own_mask_test_benign]
                        recycled_ext_idx_ben = poison_idx_test_ext_benign[~own_mask_test_benign]
                        recycled_local_idx_ben = recycled_ext_idx_ben - n_own_test_benign
                        n_poisoned_test_benign_own = len(poison_idx_test_benign_own)

                        X_test = X_test.copy()
                        X_test[poison_idx_test_benign_own] = X_test_ext_pert_benign[poison_idx_test_benign_own]

                        recycled_gids_ben = gid_prev_test[recycled_local_idx_ben]
                        recycled_Xpert_ben = X_test_ext_pert_benign[recycled_ext_idx_ben]
                        n_recycled_ben = len(recycled_gids_ben)

                        # REWORK (recycled-pocket test pool, now persisted): mirror of
                        # the malicious side's identical append above.
                        if n_recycled_ben > 0:
                            recycled_new_idx_ben = np.arange(len(X_test), len(X_test) + n_recycled_ben)
                            X_test = np.concatenate([X_test, recycled_Xpert_ben], axis=0)
                            y_test = np.concatenate(
                                [y_test, np.full(n_recycled_ben, benign_label, dtype=y_test.dtype)], axis=0
                            )
                            gid_test = np.concatenate([gid_test, recycled_gids_ben], axis=0)
                        else:
                            recycled_new_idx_ben = np.array([], dtype=np.int64)

                        poison_idx_test_benign = np.concatenate(
                            [poison_idx_test_benign_own, recycled_new_idx_ben]
                        ).astype(np.int64)
                        n_poisoned_test_benign = len(poison_idx_test_benign)
                        task_test_splits[t] = (X_test, y_test)
                        task_test_gids[t] = gid_test
                        poisoned_test_sample_ids_benign = gid_test[poison_idx_test_benign].tolist() \
                            if n_poisoned_test_benign > 0 else []

                        print(f"[Task {t}] poisoned {n_poisoned_test_benign}/{len(attack_pool_ben_test)} attacked "
                              f"({n_tier1_used_ben} C1-correct, {n_tier2_used_ben} padded) benign TEST samples "
                              f"({n_poisoned_test_benign_own} own + {n_recycled_ben} recycled from task {t - 1}'s "
                              f"test split, both now persisted into task_test_splits[{t}]) -- "
                              f"success rate on C1-correct starts was "
                              f"{(n_poisoned_test_benign / n_tier1_used_ben) if n_tier1_used_ben > 0 else float('nan'):.3f}")

            teacher_model = copy.deepcopy(model); teacher_model.eval()
            update_buffer_madar(Xtr, ytr, category, gid_train, benign_label, mal_label, model, DEVICE)

        # TEST-side category array (mirrors the train-side one, above) -- feeds
        # plot_decision_boundary's test-split scatter (poison_idx_test/
        # poison_idx_test_benign rows are exactly what red_test_pert_agent/
        # red_test_benign_pert_agent just produced above, if this was a t>0
        # task with agents at all).
        category_test = np.full(len(y_test), "malicious_clean", dtype=object)
        category_test[y_test == benign_label] = "benign"
        if len(poison_idx_test) > 0:
            category_test[poison_idx_test] = "malicious_perturbed"
        if len(poison_idx_test_benign) > 0:
            category_test[poison_idx_test_benign] = "benign_perturbed"

        buffer_summary = buffer_composition_summary()
        #print(f"    [Buffer] composition: {buffer_summary}")

        # Decision-boundary snapshot for THIS task, post-training (i.e. the
        # same model state per_task_eval below scores). This is the "testing"
        # checkpoint -- test-side poisoning above already ran, so
        # category_test now has real perturbed markers, unlike the two
        # adaptation-phase plots.
        X_test_scaled_np = to_tensor(task_test_splits[t][0]).numpy()
        if t > 0:
            # PROTOTYPE (adaptation-split pocket visualization): continue the
            # SAME running task_pocket_mask/prev_proba from the adaptation
            # plots above -- this is the last checkpoint of the task for
            # MADAR (no unlearning phase), so this final union is what
            # carries into the montage/summary plot.
            proba_test = _decision_grid(model, pca_mean, pca_components, pca_extent, DEVICE, mal_label)[2]
            task_pocket_mask |= compute_pocket_mask(prev_proba, proba_test)
            boundary_panels.append(plot_decision_boundary(
                model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                Xtr.numpy(), category, X_test_scaled_np, category_test,
                t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}.png"),
                checkpoint_label="test", highlight_mask=task_pocket_mask,
                precomputed_grid=(xx0, yy0, proba_test),
            ))
        else:
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

        # TEMPORARY DIAGNOSTIC (test-accuracy breakdown, Q1-3): snapshot of
        # THIS task's own (recycled-extended) test split against the current
        # classifier -- see _test_accuracy_snapshot's docstring. Uses
        # task_test_splits[t] directly rather than the loop-local X_test/
        # y_test to guarantee it matches exactly what per_task_eval[t] above
        # was computed against. Computed for every task (including t==0,
        # where it's trivially all-clean) so write_test_accuracy_breakdown
        # below always has something to report.
        X_test_t, y_test_t = task_test_splits[t]
        pre_test_acc_snapshot = _test_accuracy_snapshot(
            classifier_wrapper, X_test_t, y_test_t, poison_idx_test, poison_idx_test_benign, mal_label, benign_label
        )
        pre_test_acc_metrics = {
            "balanced_accuracy": per_task_eval[t]["balanced_accuracy"],
            "pooled_accuracy": pooled_eval["accuracy"],
            "mean_accuracy": mean_per_task_bal_acc,
        }

        # TEMPORARY DIAGNOSTIC (pocket-targeting investigation), ported from
        # madar_unlearning_cl_pipeline.py: `classifier_wrapper` right now IS
        # C2 (this task's FINAL state -- MADAR has no unlearning step to run
        # after this). `pre_train_classifier_wrapper` (captured before this
        # task's CL training, in the t>0 branch of step 1) is C1. Both
        # checkpoints exist in one shot here, unlike the unlearning
        # pipeline's version which has to come back for C3 later.
        if t > 0:
            pocket_diag_entries = []
            X_test_diag, _ = task_test_splits[t]
            # REWORK (recycled-pocket test pool, now persisted): poison_idx_test/
            # poison_idx_test_benign are each [own successes..., recycled
            # successes...] in that order (see the selection blocks above), and
            # task_test_splits[t] now actually holds the recycled rows too (as
            # trailing appended rows) -- so a single predict() call over each
            # already covers both own and recycled entries; local_i past the
            # "own" count identifies a recycled entry for the source tag.
            if len(poison_idx_test) > 0:
                X_mal_pert_diag = X_test_diag[poison_idx_test]
                c1_mal = pre_train_classifier_wrapper.predict(X_mal_pert_diag)
                c2_mal = classifier_wrapper.predict(X_mal_pert_diag)
                for local_i, idx in enumerate(poison_idx_test):
                    entry = {
                        "gid": int(gid_test[idx]), "group": "malicious_perturbed",
                        "true_label": int(mal_label),
                        "c1_pred": int(c1_mal[local_i]), "c2_pred": int(c2_mal[local_i]),
                    }
                    if local_i >= n_poisoned_test_own:
                        entry["source"] = f"recycled_from_task_{t - 1}"
                    pocket_diag_entries.append(entry)
            if len(poison_idx_test_benign) > 0:
                X_ben_pert_diag = X_test_diag[poison_idx_test_benign]
                c1_ben = pre_train_classifier_wrapper.predict(X_ben_pert_diag)
                c2_ben = classifier_wrapper.predict(X_ben_pert_diag)
                for local_i, idx in enumerate(poison_idx_test_benign):
                    entry = {
                        "gid": int(gid_test[idx]), "group": "benign_perturbed",
                        "true_label": int(benign_label),
                        "c1_pred": int(c1_ben[local_i]), "c2_pred": int(c2_ben[local_i]),
                    }
                    if local_i >= n_poisoned_test_benign_own:
                        entry["source"] = f"recycled_from_task_{t - 1}"
                    pocket_diag_entries.append(entry)

            write_pocket_targeting_diagnostic(
                pocket_diag_log_path, t, pocket_diag_entries, benign_label, mal_label,
            )

        write_test_accuracy_breakdown(pocket_diag_log_path, t, pre_test_acc_snapshot, pre_test_acc_metrics,
                                       mal_label, benign_label)

        results.append({
            "task_id": t,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "n_malicious_train": int(len(mal_idx_train)),
            "n_benign_train": int(len(benign_idx_train)),
            "n_poisoned": n_poisoned,
            "poisoned_sample_ids": poisoned_sample_ids,
            "n_poisoned_test": n_poisoned_test,
            "poisoned_test_sample_ids": poisoned_test_sample_ids,
            "n_poisoned_benign": n_poisoned_benign,
            "poisoned_sample_ids_benign": poisoned_sample_ids_benign,
            "n_poisoned_test_benign": n_poisoned_test_benign,
            "poisoned_test_sample_ids_benign": poisoned_test_sample_ids_benign,
            # REWORK (recycled-pocket test pool): mirror of
            # madar_unlearning_cl_pipeline.py's identical fields -- see that
            # file's results dict for the full rationale.
            "n_poisoned_test_recycled_prev": int(len(recycled_gids_mal)),
            "poisoned_test_sample_ids_recycled_prev": recycled_gids_mal.tolist(),
            "n_poisoned_test_recycled_prev_benign": int(len(recycled_gids_ben)),
            "poisoned_test_sample_ids_recycled_prev_benign": recycled_gids_ben.tolist(),
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
        "train_uncertain_fraction": TRAIN_UNCERTAIN_FRACTION,
        "test_aggregate_fraction": TEST_AGGREGATE_FRACTION,
        "recycle_test_pockets": RECYCLE_TEST_POCKETS,
        "variant_steps_per_level": VARIANT_STEPS_PER_LEVEL,
        "max_variants_per_sample": MAX_VARIANTS_PER_SAMPLE,
        "red_train_target_margin_confidence": RED_TRAIN_TARGET_MARGIN_CONFIDENCE,
        "red_target_margin_confidence": RED_TARGET_MARGIN_CONFIDENCE,
        "red_target_margin_high": RED_TARGET_MARGIN_HIGH,
        "test_side_attack_method": "pgd_boundary_search",
        "poison_test_data": POISON_TEST_DATA,
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
