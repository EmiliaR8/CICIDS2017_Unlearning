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

BRANCH NOTE (claude/oracle-unlearn-sample-tracking): main's version of this
file uses detect_poison_if_latent (phi1, iteration 1). This branch swaps
detection back to detect_poison_oracle (Df_t = poison_idx exactly) to
isolate a still-open question -- does the unlearning MECHANISM (steps
5/7: unlearn_teacher_guided, SI, retain+KD anchoring) behave sensibly at
all -- from detector quality, before sinking more effort into phi1/phi3-5.
If oracle-driven unlearning is well-behaved, the swings seen with phi1
(madar_u1.json/madar_u2.json: forget-set accuracy sometimes frozen,
sometimes moving the wrong direction, task 7's -0.17 collateral damage on
task 0) are downstream of what phi1 selects, not the mechanism itself --
narrowing where to look next. If oracle-driven unlearning is ALSO
unstable, the mechanism itself needs fixing before any detector iteration
matters. This branch also adds global sample-id tracking (see
update_buffer_madar/buffer_composition_summary and `gid_train` in main())
so buffer contents and poisoned samples can be traced across tasks for
later tSNE-style analysis -- independent of the oracle-vs-phi1 question,
worth porting back to main either way. UPDATE: this branch also flips
POISON_TEST_DATA on -- naive/joint/madar_cl_pipeline.py's shared
convention (and this file's own original design) was to NEVER poison test
data, since evasion rate already measures test-time generalization and
poisoning eval labels was judged likely to conflate "evaded" with "ground
truth malicious." That reasoning holds for the OTHER three pipelines
(no unlearning to test), but for THIS question it left a gap: with test
always clean, there was no way to check whether detection/unlearning
generalizes to evasive samples the classifier never trained on, only
whether it handles memorized training-time poison. POISON_TEST_DATA
applies the same evaded-mask-restricted POISON_FRACTION substitution to
each task's test split too (see its definition below for exactly what).
Unlearning stays scoped to training data only -- Df_t is never drawn from
test, since there's nothing to "forget" from data never trained on; this
only changes what per_task_eval/pooled_eval/prior_tasks_recovery measure.

UPDATE 2: per-task component order changed. Previously: train -> detect ->
unlearn -> IsolationForest/buffer -> test (test was LAST, so per_task_eval/
pooled_eval always reflected the POST-unlearning model). Now: train -> TEST
-> detect -> unlearn -> IsolationForest/buffer, so per_task_eval/pooled_eval
are the model's PRE-unlearning snapshot for task t, and that buffer refresh
is what task t+1's training reads next iteration (unchanged -- buffer was
always "this task's refresh feeds next task's training", moving test earlier
doesn't touch that). detect_poison_oracle was already maximally targeted at
this task's own perturbation samples (Df_t = poison_idx, drawn only from
Xtr/ytr, never broader) before this change and still is -- nothing to adjust
there. To avoid losing the post-unlearning view of THIS task's own numbers
(previously the headline; useful for exactly the kind of before/after
analysis done on madar_u1-u4), unlearning_metrics now separately logs
post_unlearn_this_task_eval alongside the existing prior_tasks_recovery
bracket (which still covers OLDER tasks and is untouched by this reorder).

UPDATE 3: detect_poison_oracle can now forget a partial random subsample of
poison_idx instead of always all of it -- ORACLE_FORGET_FRACTION (default
1.0 = unchanged prior behavior). Still zero false positives (Df_t is always
a subset of true poison; precision stays 1.0), but recall drops to
forget_fraction, so this simulates "the unlearner only gets told about SOME
of the poison" without touching detection logic or inventing a fake
detector. Whatever poison isn't selected stays in the retain set --
unforgotten poison is buffer-eligible exactly like phi1's false negatives
were on main. detector_eval (precision/recall) is no longer trivially
1.0/1.0 below forget_fraction=1.0 -- it now measures the subsample rate
directly, useful for sweeping "how much of the poison does the unlearner
actually need to see to keep collateral damage low."

UPDATE 5 (REWORK -- SUPERSEDES detect_poison_oracle/ORACLE_FORGET_FRACTION
and the "BUFFER ORDERING" decision below entirely; both are kept in this
docstring for history, not because they're still accurate). Forget-set
construction no longer draws directly from oracle ground truth. Instead:

  1. Oracle ground truth (poison_idx) is used ONLY to seed training labels
     for a small per-task 3-class RandomForestClassifier,
     "perturbation_classifier" (benign / malicious_clean / malicious_perturbed),
     trained on PERTURBATION_CLASSIFIER_N (default 50, shrinking uniformly
     across all three classes if any pool is short this task) samples per
     class, on raw/scaled Xtr features. See
     build_perturbation_classifier_forget_set()'s docstring for the full
     mechanics, including why raw features over the blue model's latent
     embedding (decouples this classifier from the blue model's
     still-evolving, task-to-task-shifting latent space).
  2. It then predicts on every OTHER Xtr row this task (never its own
     150-ish training rows, to avoid trivially-correct leakage in the
     logged eval metrics). potentially_perturbed_pool = predicted
     malicious_perturbed AND true (binary) label malicious -- a benign
     sample mispredicted "perturbed" is excluded, since it isn't malicious
     at all. forget_idx = potentially_perturbed_pool exactly; this is what
     unlearn_teacher_guided forgets (same mechanism as before, just a
     different-provenance forget set).
  3. BUFFER POPULATION IS NO LONGER RETAIN-ONLY. update_buffer_madar now
     always fills from the FULL task batch, unfiltered by forget/retain --
     identical call pattern to task 0 and to plain MADAR. The forget/retain
     split above is used ONLY to build unlearning's retain_loader (the
     CE+KD anchor signal), not to gate buffer eligibility anymore. This is
     a deliberate reversal of the old "poison never enters the buffer even
     transiently" design (see BUFFER ORDERING below, now stale) -- IsolationForest
     may legitimately select a potentially_perturbed_pool member into the
     buffer.
  4. CLEAN_REPLAY_BUFFER_OF_PERTURBED (toggle, default True): immediately
     after buffer fill, scan the just-refreshed buffer for entries whose
     sample_id matches potentially_perturbed_pool's global ids and remove
     them, THEN refill the malicious slot back up to budget with this task's
     own malicious samples that are NOT in potentially_perturbed_pool (the
     classifier's judgment, never oracle ground truth), ranked via the same
     IsolationForest anomaly+inlier selection as everywhere else
     (_anomaly_inlier_interleave, shared with update_buffer_madar). REWORK:
     originally no backfill (removed slots stayed empty until the next
     task's refresh happened to top them up); added because leaving the
     slot chronically under-filled confounds "effect of forgetting perturbed
     samples" with "effect of a smaller buffer" -- plain MADAR's buffer
     never shrinks this way, so the size gap alone could bias any comparison
     between the two pipelines. If fewer clean-malicious candidates exist
     than the deficit, the slot stays under-filled by whatever's left over.
     When the toggle is off, the buffer is left exactly as IsolationForest
     selected it (may contain potentially_perturbed_pool members).
  5. Both the 3-class confusion matrix (against full oracle ground truth on
     the eval rows) and the 2-class collapsed accuracy/balanced_accuracy
     are logged per task under unlearning.perturbation_classifier, so
     precision/recall of the classifier itself -- e.g. how many true
     malicious_perturbed rows it actually caught, how many
     malicious_clean rows it mislabeled as perturbed -- is directly visible
     without cross-referencing anything else.

UPDATE 6 (REWORK, quad red agents / benign perturbation classifier):
perturbation_classifier is now 4-class (benign / malicious_clean /
malicious_perturbed / benign_perturbed), mirroring points 1-5 above exactly
on the benign side -- poison_idx_benign (from the new
red_train_benign_pert_agent/red_test_benign_pert_agent, see
train_red_agent_for_task's docstring and main()) seeds benign_perturbed's
training labels the same way poison_idx seeds malicious_perturbed's;
potentially_perturbed_benign_pool = predicted benign_perturbed AND true
(binary) label benign, same construction as potentially_perturbed_pool.
forget_idx fed to unlearn_teacher_guided is now the UNION of both pools
(one unlearning pass covers both populations -- unlearn_teacher_guided
doesn't care why a sample is in the forget set). CLEAN_REPLAY_BUFFER_OF_
PERTURBED's clean+refill (point 4 above) is likewise mirrored per label
slot: the malicious slot is cleaned/refilled using only forget_idx_malicious
as before (still via uncertainty sampling + historical_clean_mal_pool, see
that pool's definition and HISTORICAL_POOL_MAX_SIZE), and the benign slot is
now ALSO cleaned of forget_idx_benign matches and refilled from a mirrored
cross-task pool (historical_clean_benign_pool), via the same
_clean_and_refill helper in main(). See
build_perturbation_classifier_forget_set()'s docstring for the full
mechanics.

UPDATE 7 (REWORK, pocket-targeted test poisoning): red_test_pert_agent/
red_test_benign_pert_agent now train AFTER this task's CL training finishes,
not before it (train-side agents/poisoning are unaffected -- still attack the
pre-task-t classifier, since that's what CAUSES the boundary to move). A
frozen pre-training classifier snapshot (pre_train_classifier_wrapper,
captured right before train_cl_er mutates `model`) is passed alongside the
now-POST-training live classifier, and NetworkAttackEnv's new pocket_term
rewards perturbations landing where the two disagree in the evasive
direction -- i.e. a point THIS TASK's own poisoned CL training just flipped
from correctly- to incorrectly-classified, a "pocket" attributable to this
task's training specifically, layered on top of (not replacing)
RED_TARGET_MARGIN_CONFIDENCE's existing margin-minimizing bias. See
POCKET_SHIFT_WEIGHT's definition and NetworkAttackEnv.step()'s pocket_term
for the full mechanics. Motivation: make plain MADAR's accuracy drop
specifically where MADAR+Unlearning's detection/forgetting has real leverage
to recover it (an artifact of this task's own poisoned training), rather
than at boundary weaknesses that predate this task and neither pipeline is
positioned to fix.

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

  - POISON DETECTION (step 3): on main, being built iteratively (iteration 1
    shipped phi1 -- an IsolationForest decision_function score over the
    latent embedding of this task's malicious-labeled samples, forgetting the
    MIDDLE band; ported from split_option_b_donut_hole in the EMBER unlearning
    reference, clmdu_ember18_mk6B_cd.py, with its per-malware-family joint-fit
    machinery dropped since CICIDS poisoning only ever touches the
    malicious-labeled population). Planned iteration 2 there: phi3 (per-sample
    CE loss), phi4 (normalized entropy), phi5 (top-two margin) -- model-
    behavior features expected to matter more than phi1/phi2's raw geometry
    against evasion-crafted poisoning specifically, computed against the
    PRE-task-t teacher_model snapshot rather than the post-CL-training model
    (by the time detection runs post-training, the model has already fit Tt's
    labels, poison included, so post-training confidence features are
    measuring "did training just memorize this," not "does this look
    evasive"). phi6/phi7 (malware-family based) don't transfer to CICIDS.

    THIS BRANCH originally overrode all of that with detect_poison_oracle()
    (Df_t = poison_idx directly, no detection logic at all -- see the BRANCH
    NOTE above for why). STALE as of UPDATE 5 above: forget-set construction
    is now build_perturbation_classifier_forget_set(), which uses poison_idx
    only to seed a small classifier's training labels, not to build Df_t
    directly.

  - WEIGHT PROTECTION during unlearning uses the existing SI (omega/p_old_task)
    machinery already implemented and validated in madar_cl_pipeline.py, not
    MAS (Memory Aware Synapses) as literally named in the outline. Both
    reference scripts also use SI here, not MAS, despite the outline's
    wording -- MAS is a different importance computation (gradient of squared
    output-norm, not SI's path integral) that neither reference implements and
    that would need to be built and validated from scratch for no
    demonstrated benefit yet. The first smoke test (madar_u1.json) showed
    forget-set accuracy completely unmoved on several tasks (2/3/4: exactly
    1.000 -> 1.000 despite real gradient steps) alongside catastrophic
    prior-task collateral damage on another (task 7: -0.17 balanced accuracy
    on task 0, isolated to the unlearning phase specifically by
    prior_tasks_recovery) -- inconsistent, task-dependent symptoms consistent
    with SI's diagonal per-parameter penalty sometimes freezing the forget
    objective and sometimes failing to protect against it, rather than a
    single miscalibration to fix with one number. Rather than swap frameworks
    on a hypothesis, unlearn_teacher_guided now returns raw (pre-weighting)
    loss-component magnitudes and omega's L2 norm every call (see its
    docstring) so this can be checked directly, and UNLEARN_SI_C decouples
    unlearning's SI weight from CL training's SI_C (still 1.0, unchanged) --
    set to 0.1 as a first-pass guess, meant to be tuned against the next
    smoke test's diagnostics rather than treated as settled.

  - BUFFER ORDERING deviates from the outline's literal steps 2/6 (add Tt to
    the buffer in full, forget-detect, then filter Df_t back out afterward).
    Both reference scripts instead compute the forget/retain split FIRST,
    unlearn, and update the buffer ONCE from the retain set only -- poison
    never enters the buffer even transiently, and it's one buffer touch per
    task instead of two. Adopted as-is here. STALE as of UPDATE 5 above: the
    buffer now fills from the FULL task batch (unfiltered), closer to the
    outline's literal steps 2/6 after all, with CLEAN_REPLAY_BUFFER_OF_PERTURBED
    doing the "filter back out" part post-hoc instead of pre-filtering.

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

from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
RUNS_BASE_DIR = "/mnt/erivas6/runs"  # base directory for all run outputs (logs/plots/results json)

SEED = 42  # overwritten from --seed at the top of main(); module-level so
           # update_buffer_madar's IsolationForest(random_state=SEED) sees it.

NUM_TASKS = 10
# task0 = 30% warm start; tasks1-9 taper 9.18% -> 6.38% of the pooled dataset.
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]
TASK_TEST_FRAC = 0.20

REQUIRE_EVASION_SUCCESS = True  # only successfully-evasive perturbations poison training data
POISON_TEST_DATA = True  # BRANCH-SPECIFIC divergence from naive/joint/madar_cl_pipeline.py's
                          # shared convention (test is normally NEVER poisoned -- see this
                          # file's module docstring, BRANCH NOTE). When True, the SAME
                          # evaded-mask-restricted substitution (schedule below) applied to
                          # train is also applied to each task's test split, so per_task_eval
                          # /pooled_eval/prior_tasks_recovery measure accuracy on data that
                          # includes evasive perturbations the classifier never trained on --
                          # i.e. does detection/unlearning generalize to unseen poison, not just
                          # memorized training-time poison. Unlearning itself stays scoped to
                          # TRAINING data only (Df_t is never drawn from test) -- there is
                          # nothing to "forget" from data that was never trained on.

# Per-task poison injection schedule (REWORK, replaces the old flat
# POISON_FRACTION=0.3). Linear interpolation from task 1 to task NUM_TASKS-1
# (task 0 has no red agent, so no poisoning either way). TRAIN and TEST are
# deliberate mirror images of each other -- train front-loads poison (heavy
# early, tapering later), test back-loads it (light early, heavy later) --
# crossing at ~0.30 around the midpoint task, matching the flat fraction
# this replaces. Plain global variables (not derived/computed) specifically
# so they're easy to change later -- see poison_fraction_for_task() below.
# Identical to madar_cl_pipeline.py's, so both pipelines see the SAME
# poisoned data each task -- any difference in outcomes is attributable to
# the STRATEGY, not to one pipeline facing easier/harder data.
# REWORK 2: endpoints pushed to a much more drastic 0.99/0.01 split (was
# 0.50/0.10) -- crossing point is now ~0.50 around the midpoint task, no
# longer ~0.30. POISON_FRACTION_FLAT (below) was NOT changed to match --
# it's an independently-set fallback for the schedule-off case, not derived
# from the schedule's endpoints, so it no longer coincides with the
# schedule's midpoint the way it used to.
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
                                      # it specifically, unlike THIS file's
                                      # perturbation_classifier (an explicitly boundary-adjacent
                                      # detector). Set to None to restore the original
                                      # confidence-maximizing reward for both agents. Identical
                                      # constant/value to madar_cl_pipeline.py's, so both
                                      # pipelines face the same attack.

POCKET_SHIFT_WEIGHT = 5.0  # REWORK (pocket-targeted test poisoning): weight on
                            # NetworkAttackEnv's pocket_term (see its docstring) for
                            # red_test_pert_agent/red_test_benign_pert_agent ONLY --
                            # train-side agents are unchanged, since they're what
                            # SHIFTS the boundary in the first place (there's no
                            # "before" snapshot to compare against for them; using
                            # one would be circular). Same 5x scale as
                            # confidence_term so neither term structurally dominates
                            # by construction. Rationale: rather than a generic
                            # evasion direction, this specifically rewards landing in
                            # a PROBED "pocket" -- a point THIS TASK's own poisoned
                            # CL training just pushed from correctly-classified to
                            # misclassified -- which should make plain MADAR's
                            # accuracy drop specifically where MADAR+Unlearning's
                            # detection/forgetting has the most leverage to recover
                            # it (an actual perturbed-training-data artifact, not a
                            # boundary weakness that predates this task). See
                            # main()'s test-side agent construction for how the
                            # frozen pre-CL-training snapshot is captured. Set to
                            # None/0 to disable (falls back to whatever
                            # target_margin_confidence alone produces).

PROXIMITY_WEIGHT = 3.0  # REWORK (proximity-anchored pocket targeting): weight on
                         # NetworkAttackEnv's proximity_term (see its docstring) for
                         # red_test_pert_agent/red_test_benign_pert_agent ONLY.
                         # Slightly lighter than POCKET_SHIFT_WEIGHT/confidence_term's
                         # 5x -- a bias pulling the test agent toward the train-side
                         # forget-set's neighborhood, not a hard requirement that
                         # overrides evasion/pocket-shift entirely. Motivation: the
                         # pocket-targeted test poisoning experiment found unlearning
                         # recovers accuracy on the test batch in only ~2/9 tasks
                         # (madar_u_pocket_run_0826_630.json), with NO correlation to
                         # how much got forgotten -- e.g. tasks 3/9 forgot thousands of
                         # malicious rows yet showed ZERO test-batch recovery. Read:
                         # the test agent (trained independently of the train agent)
                         # was landing in a DIFFERENT pocket than whatever unlearning
                         # actually corrects. This anchors it to the SAME neighborhood
                         # by construction, while still letting it search
                         # independently (still tests real generalization, not just
                         # memorization of the train agent's exact perturbation). Set
                         # to None/0 to disable.
PROXIMITY_LENGTH_SCALE = 5.0  # First-pass decay length scale for proximity_term's
                               # exp(-nearest_dist / scale) -- raw feature space here
                               # is [0,1]-per-dimension, unnormalized, so this needs
                               # tuning against actually observed anchor distances for
                               # this dataset once a run's logs are available.
PROXIMITY_ANCHOR_MAX = 200  # Caps how many of this task's TRAIN-side poisoned
                             # exemplars get passed as proximity_term's anchor set --
                             # NetworkAttackEnv.step() computes a nearest-neighbor
                             # distance against ALL of them every single RL step, so
                             # this bounds that cost. Random subsample when the actual
                             # poisoned-row count exceeds it.

# Caps evaluate_agent_on_batch's per-task episode count for runtime; None = every malicious sample.
MAX_EVAL_SAMPLES_PER_TASK = 5000

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

HISTORICAL_POOL_MAX_SIZE = MEM_SIZE * 2  # REWORK (task 8/9 collapse investigation):
                                          # historical_clean_mal_pool (see its definition
                                          # below) used to grow unbounded over the whole run.
                                          # Uncertainty-sampling refill always pulls the LEAST
                                          # confident candidates from it first, every task --
                                          # so an ever-larger pool means an ever-larger set of
                                          # ambiguous, boundary-hugging exemplars competing to
                                          # get selected, which can compound over tasks even
                                          # though the pool isn't dominated by literally STALE
                                          # (old-task) samples (checked directly against
                                          # madar_u_0824.json's replay_buffer_composition --
                                          # task 9's buffer was 79% from tasks 8/9 themselves).
                                          # Capped here via FIFO eviction (oldest-appended
                                          # samples dropped first) right after each task's
                                          # growth step. 2x MEM_SIZE is a first-pass guess --
                                          # big enough to keep real cross-task diversity, small
                                          # enough to stop the pool from just accumulating every
                                          # ambiguous sample from every task forever. Also used
                                          # (REWORK, quad red agents) to cap
                                          # historical_clean_benign_pool, the exact mirror of
                                          # historical_clean_mal_pool on the benign side.

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
UNLEARN_SI_C = 0.1  # SI penalty weight during UNLEARNING ONLY -- deliberately
                     # decoupled from SI_C (still 1.0 for ordinary CL training,
                     # unchanged). First smoke test (madar_u1.json) showed forget-set
                     # accuracy completely unmoved on several tasks despite real
                     # gradient steps, consistent with SI's accumulated-omega penalty
                     # dominating the forget objective's gradient at si_c=1.0 -- this
                     # is a first-pass guess at a softer value, NOT yet tuned; see the
                     # loss-component diagnostics logged in unlearn_teacher_guided's
                     # return value for the data to tune it against.
POISON_DETECTOR = "perturbation_classifier"  # logged into config/results for provenance --
                                              # REWORK, see module docstring: replaces the old
                                              # oracle-fraction forget set entirely.

# --- Perturbation-classifier forget-set construction (REWORK, replaces
# detect_poison_oracle/ORACLE_FORGET_FRACTION entirely -- see module
# docstring). Oracle ground truth is now used only to SEED a small 3-class
# classifier's training labels, not to build the forget set directly. ---
PERTURBATION_CLASSIFIER_N = 50  # target sample count PER CLASS (benign / clean_malicious /
                                 # perturbed) used to train perturbation_classifier each task.
                                 # Shrinks uniformly across all three classes (same N for all)
                                 # if any one pool has fewer than this many available this task
                                 # -- see build_perturbation_classifier_forget_set().
CLEAN_REPLAY_BUFFER_OF_PERTURBED = True  # toggle: after IsolationForest/buffer fill (which now
                                          # draws from the FULL task batch, unfiltered -- see
                                          # module docstring), remove any buffer entries that
                                          # match potentially_perturbed_pool's global sample ids,
                                          # then refill the malicious slot back up to budget with
                                          # this task's own clean-malicious (not in
                                          # potentially_perturbed_pool) samples, ranked via the
                                          # same IsolationForest anomaly+inlier selection as
                                          # everywhere else. Stays under-filled only if too few
                                          # clean-malicious candidates exist this task to cover
                                          # the deficit.


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


RED_AGENT_SEED_OFFSETS = {"train": 0, "test": 100_000, "train_benign": 200_000, "test_benign": 300_000}


def train_red_agent_for_task(task_id, agent_type, classifier, X_data, y_data, benign_label, bank, seed, out_dir,
                              target_margin_confidence=None, shift_reference_classifier=None,
                              pocket_shift_weight=None, proximity_anchor_X=None, proximity_weight=None,
                              proximity_length_scale=5.0):
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
    instead of deep past it. Only wired up to "train"/"test" (see
    RED_TARGET_MARGIN_CONFIDENCE's call sites in main()) -- "train_benign"/
    "test_benign" are left at the default (ordinary confidence-maximizing
    reward), since the margin-minimizing tactic was motivated by and
    validated against malicious-side evasion specifically.

    shift_reference_classifier/pocket_shift_weight (REWORK, pocket-targeted
    test poisoning): forwarded straight to NetworkAttackEnv -- see its
    docstring. Only meaningful for "test"/"test_benign", passed a FROZEN
    pre-CL-training snapshot by main() (train-side agents have no "before"
    state to compare against, since they're what shifts the boundary).

    proximity_anchor_X/proximity_weight/proximity_length_scale (REWORK,
    proximity-anchored pocket targeting): forwarded straight to
    NetworkAttackEnv -- see its docstring. Only meaningful for "test"/
    "test_benign", passed this task's own TRAIN-side poisoned exemplars by
    main() -- the exact rows about to become the forget set.
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


def _predict_confidence(model, Xt: torch.Tensor, device):
    """Per-sample confidence = max class probability under the CURRENT model
    (softmax over its raw logits; Xt is already-scaled, same convention as
    _embed). For this 2-class problem, ranking by this ascending is
    equivalent to ranking by |P(malicious) - 0.5| ascending -- i.e. "closest
    to the decision boundary first". Used by the buffer refill step's
    uncertainty-sampling selection (see main(), REWORK) in place of
    IsolationForest anomaly/inlier ranking."""
    loader = data.DataLoader(data.TensorDataset(Xt), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    parts = []
    model.eval()
    with torch.no_grad():
        for (v,) in loader:
            probs = F.softmax(model(v.to(device)), dim=1)
            parts.append(probs.max(dim=1).values.cpu())
    return torch.cat(parts).numpy()


def _anomaly_inlier_interleave(scores, n_select):
    """Given IsolationForest decision_function scores (lower = more anomalous)
    over a candidate pool, returns n_select LOCAL indices into that pool:
    half most-anomalous-first, half most-inlier-first, interleaved. Factored
    out of update_buffer_madar so the buffer-cleaning refill step in main()
    (REWORK) can draw new entries using identical selection logic instead of
    a separately-invented method."""
    sorted_idx = np.argsort(scores)
    n_total = len(sorted_idx)
    half = n_select // 2
    n_inlier = n_select - half
    anomalies_idx = sorted_idx[:half]
    inliers_idx = sorted_idx[n_total - n_inlier:] if n_inlier > 0 else np.array([], dtype=int)
    interleaved = [idx for pair in zip(anomalies_idx, inliers_idx) for idx in pair]
    if n_select % 2 != 0 and len(inliers_idx) > 0:
        interleaved.append(inliers_idx[-1])
    return interleaved


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, sample_id: np.ndarray,
                         benign_label, mal_label, model, device):
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
    `sample_id` is the GLOBAL pool row index each sample was assigned in main()
    (stable across the whole run, survives train/test splitting and poisoning
    substitution) -- carried alongside category purely for traceability (tSNE
    plots, cross-task comparison of which physical samples persist in the
    buffer), also not used by selection.
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
        interleaved_idx = _anomaly_inlier_interleave(scores, n_select)

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
# Step 3 -- forget-set construction, PERTURBATION-CLASSIFIER REWORK (see
# module docstring). Replaces the old detect_poison_oracle/ORACLE_FORGET_FRACTION
# entirely -- oracle ground truth (poison_idx) is now used only to SEED a
# small 3-class classifier's training labels, never to build the forget set
# directly. main branch keeps phi1 (detect_poison_if_latent); untouched, this
# is a separate branch, not a replacement for it.
# ---------------------------------------------------------------------------
LABEL_BENIGN = "benign"
LABEL_CLEAN_MALICIOUS = "malicious_clean"
LABEL_PERTURBED = "malicious_perturbed"
LABEL_BENIGN_PERTURBED = "benign_perturbed"
PERTURBATION_CLASSIFIER_CLASSES = [LABEL_BENIGN, LABEL_CLEAN_MALICIOUS, LABEL_PERTURBED, LABEL_BENIGN_PERTURBED]


def build_perturbation_classifier_forget_set(Xtr, ytr, poison_idx, poison_idx_benign, benign_label, mal_label,
                                              rng, n_target=PERTURBATION_CLASSIFIER_N):
    """
    Trains a small per-task RandomForestClassifier ("perturbation_classifier")
    to distinguish benign / malicious_clean / malicious_perturbed /
    benign_perturbed (REWORK, quad red agents) on RAW (already-scaled) Xtr
    features, then uses its predictions -- not oracle ground truth -- to
    build the forget set. benign_perturbed mirrors malicious_perturbed
    exactly (same construction, same eval, same downstream forget-pool
    logic), just on the benign side: poison_idx_benign is to benign_idx what
    poison_idx is to mal_idx.

    1. Oracle draws N per class (benign_clean = benign minus
       poison_idx_benign, malicious_clean = malicious minus poison_idx,
       malicious_perturbed = poison_idx, benign_perturbed =
       poison_idx_benign), N shrinking uniformly across all FOUR classes if
       any pool has fewer than n_target available this task, so the
       training set stays class-balanced. This is the ONLY place
       poison_idx/poison_idx_benign's ground truth "perturbed" label is
       ever handed to a model as a training target -- nothing downstream of
       this classifier sees it.
    2. Classifier trains on those 4*N samples only, then predicts on every
       OTHER Xtr sample this task (never its own training rows -- avoids
       leaking their trivially-known labels into the eval metrics below).
    3. potentially_perturbed_pool = predicted malicious_perturbed AND whose
       actual ytr label is mal_label (the ordinary binary label the rest of
       the pipeline sees -- a benign sample the classifier mislabels
       "perturbed" is excluded, since it isn't malicious at all).
       potentially_perturbed_benign_pool = the mirror: predicted
       benign_perturbed AND whose actual ytr label is benign_label.

    forget_idx = potentially_perturbed_pool UNION
    potentially_perturbed_benign_pool -- fed to a single
    unlearn_teacher_guided call (that function doesn't care WHY something's
    in the forget set, just pushes it toward uniform). forget_idx_malicious/
    forget_idx_benign are also returned individually so the buffer-clean/
    refill step (main()) can clean each label's buffer slot using only the
    matches relevant to that slot. retain_idx = everything else -- used
    ONLY to build unlearning's retain_loader (the CE+KD anchor signal), NOT
    for buffer eligibility (see module docstring/UPDATE 5 -- buffer now
    fills from the full task batch, unfiltered, and gets cleaned of
    forget-pool matches afterward instead).

    Returns None if n_target shrinks to 0 for any class this task (no
    poison on either side, or one of the four pools is empty) -- caller
    should skip the classifier/forget-set step entirely for this task, same
    as the old empty-Df_t path. Otherwise returns a dict: forget_idx,
    forget_idx_malicious, forget_idx_benign, retain_idx (sorted int64
    arrays into Xtr), n_used (the actual per-class N after shrinking),
    confusion_matrix (4x4, row/col order == PERTURBATION_CLASSIFIER_CLASSES,
    computed against oracle ground truth on the held-out eval rows only),
    two_class_metric (predicted benign OR benign_perturbed -> compare to
    true benign; predicted malicious_clean OR malicious_perturbed ->
    compare to true malicious; accuracy + balanced_accuracy over the SAME
    held-out eval rows), and n_eval (how many rows the classifier was
    scored against).
    """
    poison_idx = np.asarray(poison_idx, dtype=np.int64)
    poison_idx_benign = np.asarray(poison_idx_benign, dtype=np.int64)
    y_np = ytr.numpy()
    n_samples = len(y_np)

    mal_idx = np.where(y_np == mal_label)[0]
    benign_idx = np.where(y_np == benign_label)[0]
    clean_mal_idx = np.setdiff1d(mal_idx, poison_idx)
    clean_benign_idx = np.setdiff1d(benign_idx, poison_idx_benign)

    n_used = min(n_target, len(poison_idx), len(clean_mal_idx), len(clean_benign_idx), len(poison_idx_benign))
    if n_used == 0:
        return None

    train_perturbed = rng.choice(poison_idx, size=n_used, replace=False)
    train_clean_mal = rng.choice(clean_mal_idx, size=n_used, replace=False)
    train_benign = rng.choice(clean_benign_idx, size=n_used, replace=False)
    train_benign_perturbed = rng.choice(poison_idx_benign, size=n_used, replace=False)
    train_idx = np.concatenate([train_benign, train_clean_mal, train_perturbed, train_benign_perturbed])
    train_labels = np.array(
        [LABEL_BENIGN] * n_used + [LABEL_CLEAN_MALICIOUS] * n_used
        + [LABEL_PERTURBED] * n_used + [LABEL_BENIGN_PERTURBED] * n_used
    )

    X_np = Xtr.numpy()
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
    clf.fit(X_np[train_idx], train_labels)

    eval_idx = np.setdiff1d(np.arange(n_samples, dtype=np.int64), train_idx)
    eval_pred = clf.predict(X_np[eval_idx])

    # Oracle 4-way ground truth for the eval rows, for scoring only -- never
    # fed to the classifier itself.
    eval_true_4way = np.full(len(eval_idx), LABEL_CLEAN_MALICIOUS, dtype=object)
    eval_true_4way[y_np[eval_idx] == benign_label] = LABEL_BENIGN
    poison_set = set(int(i) for i in poison_idx)
    poison_set_benign = set(int(i) for i in poison_idx_benign)
    eval_true_4way[np.array([i in poison_set for i in eval_idx])] = LABEL_PERTURBED
    eval_true_4way[np.array([i in poison_set_benign for i in eval_idx])] = LABEL_BENIGN_PERTURBED

    cm = confusion_matrix(eval_true_4way, eval_pred, labels=PERTURBATION_CLASSIFIER_CLASSES)

    pred_binary = np.where(
        np.isin(eval_pred, [LABEL_BENIGN, LABEL_BENIGN_PERTURBED]), benign_label, mal_label
    )
    true_binary = y_np[eval_idx]
    two_class_metric = {
        "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "balanced_accuracy": float(balanced_accuracy_score(true_binary, pred_binary)),
    }

    potentially_perturbed_pool = eval_idx[(eval_pred == LABEL_PERTURBED) & (true_binary == mal_label)]
    potentially_perturbed_benign_pool = eval_idx[(eval_pred == LABEL_BENIGN_PERTURBED) & (true_binary == benign_label)]

    forget_idx_malicious = np.asarray(sorted(int(i) for i in potentially_perturbed_pool), dtype=np.int64)
    forget_idx_benign = np.asarray(sorted(int(i) for i in potentially_perturbed_benign_pool), dtype=np.int64)
    forget_idx = np.union1d(forget_idx_malicious, forget_idx_benign)
    retain_idx = np.setdiff1d(np.arange(n_samples, dtype=np.int64), forget_idx)

    return {
        "forget_idx": forget_idx,
        "forget_idx_malicious": forget_idx_malicious,
        "forget_idx_benign": forget_idx_benign,
        "retain_idx": retain_idx,
        "n_used": int(n_used),
        "confusion_matrix": {
            "classes": PERTURBATION_CLASSIFIER_CLASSES,
            "matrix": cm.tolist(),
        },
        "two_class_metric": two_class_metric,
        "n_eval": int(len(eval_idx)),
    }


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

    Returns a diagnostics dict (raw, PRE-weighting loss component magnitudes,
    averaged across all steps, plus omega's L2 norm at call time) -- added after
    the first smoke test showed forget-set accuracy completely unmoved on
    several tasks despite real gradient steps, to see directly whether the SI
    term is dominating the forget objective rather than just hypothesizing it.
    """
    global replay_buffer
    model.train()
    teacher_model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    omega_l2_norm = float(torch.sqrt(sum((v ** 2).sum() for v in omega.values())).item())

    retain_iter = iter(cycle(retain_loader))
    if replay_buffer:
        b_v = torch.stack([entry[0] for entry in replay_buffer])
        b_l = torch.stack([entry[1] for entry in replay_buffer])
        buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l), batch_size=BATCH_SIZE, shuffle=True)
        buf_iter = iter(cycle(buf_loader))
    else:
        buf_iter = None

    raw_forget_losses, raw_retain_losses, raw_si_losses = [], [], []

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

            raw_forget_losses.append(forget_loss.item())
            raw_retain_losses.append(retain_loss.item())
            raw_si_losses.append(float(si_loss.item()) if torch.is_tensor(si_loss) else float(si_loss))

    return {
        "omega_l2_norm": omega_l2_norm,
        "n_steps": len(raw_forget_losses),
        "raw_forget_loss_mean": float(np.mean(raw_forget_losses)) if raw_forget_losses else None,
        "raw_retain_loss_mean": float(np.mean(raw_retain_losses)) if raw_retain_losses else None,
        "raw_si_loss_mean": float(np.mean(raw_si_losses)) if raw_si_losses else None,
    }


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
    """Identical to madar_cl_pipeline.py's plot_task_metrics, title relabeled, plus
    a third panel (REWORK, pocket-targeted test poisoning follow-up) showing THIS
    task's own pocket-targeted test batch accuracy at decision boundary N
    (pre-unlearning, perturbed_test_eval) vs. decision boundary J (post-unlearning,
    post_unlearn_perturbed_test_eval) -- the direct before/after comparison on the
    identical rows, rather than inferring recovery from the whole-test-set
    post_unlearn_this_task_eval number."""
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc = [r["mean_per_task_balanced_accuracy"] for r in results]
    train_evasion = [r["red_agent"]["train_evasion_rate"] if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"]["test_evasion_rate"] if r["red_agent"] else None for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc, marker="o", label="pooled balanced accuracy")
    ax1.plot(task_ids, mean_per_task_bal_acc, marker="s", label="mean per-task balanced accuracy")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("MADAR+Unlearning: classifier accuracy per task boundary (PRE-unlearning snapshot)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xs = [t for t, v in zip(task_ids, train_evasion) if v is not None]
    ys_train = [v for v in train_evasion if v is not None]
    ys_test = [v for v in test_evasion if v is not None]
    ax2.plot(xs, ys_train, marker="o", color="tab:red", label="evasion rate (train samples)")
    ax2.plot(xs, ys_test, marker="s", color="tab:orange", label="evasion rate (held-out test samples)")
    ax2.set_ylabel("Evasion rate")
    ax2.set_title("Red agent evasion rate per task boundary (task 0 has no red agent)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    pocket_task_ids, pocket_pre, pocket_post = [], [], []
    for r in results:
        t = r["task_id"]
        pre = r.get("perturbed_test_eval", {}).get(t)
        u = r.get("unlearning") or {}
        post = u.get("post_unlearn_perturbed_test_eval", {}).get(t)
        if pre is not None and post is not None:
            pocket_task_ids.append(t)
            pocket_pre.append(pre["balanced_accuracy"])
            pocket_post.append(post["balanced_accuracy"])
    if pocket_task_ids:
        ax3.plot(pocket_task_ids, pocket_pre, marker="o", color="tab:red",
                  label="pocket-targeted test batch, boundary N (pre-unlearn)")
        ax3.plot(pocket_task_ids, pocket_post, marker="s", color="tab:green",
                  label="pocket-targeted test batch, boundary J (post-unlearn)")
        ax3.set_ylim(-0.05, 1.05)
    else:
        ax3.text(0.5, 0.5, "No task ran unlearning with a pocket-targeted test batch yet",
                  ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Task id")
    ax3.set_ylabel("Balanced accuracy")
    ax3.set_title("This task's own pocket-targeted test batch: before vs. after its unlearning step")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_unlearning_metrics(results, out_path):
    """
    NEW (not in either reference): forget/retain-set accuracy before-vs-after
    unlearning, and step 7's prior-task recovery check (mean held-out balanced
    accuracy across every earlier task, before vs. after THIS task's unlearning
    phase specifically -- isolating unlearning's own effect from CL training's).
    Tasks with no unlearning this run (task 0, too few samples for the
    perturbation classifier, or an empty potentially_perturbed_pool -- these
    still log a "perturbation_classifier"-only entry, so filter on
    "forget_set" specifically, not just truthiness) simply don't appear on
    the x-axis.
    """
    unl_tasks = [r for r in results if r.get("unlearning") and "forget_set" in r["unlearning"]]
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


# agent_ids are "{task_id}_{agent_type}" compound strings, agent_type in
# {"train", "test", "train_benign", "test_benign"} -- split on the FIRST
# underscore only, and sort/mark deterministically rather than relying on
# alphabetical order.
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
# dense grid on a FIXED run-wide 2D PCA plane (pca_mean/pca_components/
# pca_extent, computed once in main() right after the scaler is fit at task
# 0), reused unchanged for every task/checkpoint so panels are directly
# comparable. TWO checkpoints get captured per task where unlearning
# actually ran: "pre_unlearn" (right after CL training) and "post_unlearn"
# (right after unlearn_teacher_guided) -- isolating what unlearning itself
# does to the boundary. Tasks where unlearning didn't run get just the one
# "pre_unlearn"-labeled snapshot.
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
    one summary figure). Returns per-agent perturbed counts (for the
    caller's legend)."""
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
    at-a-glance view of how the boundary drifts as poisoning accumulates,
    and (unique to this file) how unlearning shifts it back within a task."""
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
    ap.add_argument("--log_name", type=str, default="madar_unlearn_cl_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    args = ap.parse_args()

    global SEED
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join(RUNS_BASE_DIR, "madar_unlearning", args.log_name)
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
    # the pooled dataset by id later for tSNE -- never used by training,
    # detection, or selection.
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

    historical_clean_mal_pool = []  # REWORK (uncertainty-based cross-task refill): list of
                                     # (X, y, category, gid) 4-tuples -- every PAST task's
                                     # clean-malicious samples (malicious AND NOT in that
                                     # task's own potentially_perturbed_pool, the classifier's
                                     # own judgment, never oracle ground truth), accumulated
                                     # across the whole run. Grown AFTER each task's own
                                     # refill step (see below) so a task's own samples are
                                     # never candidates for its own refill -- only genuinely
                                     # PAST tasks' samples are. Capped to HISTORICAL_POOL_MAX_SIZE
                                     # via FIFO eviction after each task's growth step (REWORK,
                                     # task 8/9 collapse investigation -- see that constant's
                                     # definition for rationale).
    historical_clean_benign_pool = []  # REWORK (quad red agents): exact mirror of
                                        # historical_clean_mal_pool above, on the benign side --
                                        # every PAST task's clean-benign samples (benign AND NOT
                                        # in that task's own potentially_perturbed_benign_pool),
                                        # used to refill the benign buffer slot the same way.

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
        n_poisoned_test_benign = 0
        poison_idx_test_benign = np.array([], dtype=int)
        poisoned_test_sample_ids_benign = []

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

                # TEST-SIDE poisoning (POISON_TEST_DATA, this branch only -- see module
                # docstring BRANCH NOTE and POISON_TEST_DATA's definition for why).
                # REWORK (pocket-targeted test poisoning): red_test_pert_agent now
                # trains AFTER this task's CL training finishes, not here -- it needs
                # the POST-training classifier plus a frozen pre-training snapshot to
                # target "pockets" the boundary just shifted through (see
                # pre_train_classifier_wrapper's capture, right before train_cl_er,
                # and the TEST-SIDE block right after it, both below). Train-side
                # poisoning (above) is unaffected -- it still attacks the pre-task-t
                # classifier, since it's what CAUSES the boundary shift being probed.

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
                # X changes, not y. This new "benign_perturbed" population
                # feeds the reconfigured (4-class) perturbation_classifier
                # below exactly like malicious_perturbed does. No
                # target_margin_confidence bias here (see
                # train_red_agent_for_task's docstring).
                if len(benign_idx_train) == 0:
                    warnings_log.append(f"Task {t}: no benign training samples, skipping benign red agents.")
                else:
                    env_train_benign, agent_train_benign = train_red_agent_for_task(
                        t, "train_benign", classifier_wrapper, X_train, y_train, mal_label, bank, args.seed, out_dir,
                    )
                    (X_train_pert_benign, train_evasion_rate_benign, train_rewards_benign, train_avg_norm_benign,
                     train_attacked_benign, evaded_mask_benign) = evaluate_agent_on_batch(
                        env_train_benign, agent_train_benign, X_train, y_train, mal_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                    )

                    red_report["train_benign_evasion_rate"] = train_evasion_rate_benign
                    red_report["train_benign_attacked"] = train_attacked_benign
                    red_report["train_benign_avg_reward"] = float(np.mean(train_rewards_benign)) if train_rewards_benign else 0.0
                    red_report["train_benign_avg_pert_l2"] = train_avg_norm_benign
                    print(f"[Task {t} red_train_benign_pert_agent] train_evasion={train_evasion_rate_benign:.3f} "
                          f"avg_pert_L2={train_avg_norm_benign:.3f}")

                    poison_fraction_train_benign = poison_fraction_for_task(
                        t, POISON_FRACTION_TRAIN_START, POISON_FRACTION_TRAIN_END
                    )
                    poison_pool_benign = np.where(evaded_mask_benign)[0] if REQUIRE_EVASION_SUCCESS else benign_idx_train
                    n_to_poison_benign = min(
                        int(round(poison_fraction_train_benign * len(benign_idx_train))), len(poison_pool_benign)
                    )
                    rng_benign = np.random.RandomState(args.seed + t + 20_000)  # distinct stream from mal train/test rngs
                    poison_idx_benign = rng_benign.choice(poison_pool_benign, size=n_to_poison_benign, replace=False) \
                        if n_to_poison_benign > 0 else np.array([], dtype=int)
                    n_poisoned_benign = len(poison_idx_benign)

                    X_train_for_classifier[poison_idx_benign] = X_train_pert_benign[poison_idx_benign]
                    print(f"[Task {t}] poisoned {n_poisoned_benign}/{len(benign_idx_train)} benign train samples "
                          f"(poison_fraction_train_benign={poison_fraction_train_benign:.3f})")

                    # TEST-SIDE (REWORK, pocket-targeted test poisoning): moved to
                    # after CL training, mirroring the malicious side -- see the
                    # TEST-SIDE block right after train_cl_er, below.

        # Category tracking (identical to madar_cl_pipeline.py) -- feeds the
        # buffer composition diagnostic; build_perturbation_classifier_forget_set()
        # below reads poison_idx/poison_idx_benign directly (see its docstring),
        # not this array.
        category = np.full(len(y_train), "malicious_clean", dtype=object)
        category[y_train == benign_label] = "benign"
        if len(poison_idx) > 0:
            category[poison_idx] = "malicious_perturbed"
        if len(poison_idx_benign) > 0:
            category[poison_idx_benign] = "benign_perturbed"
        poisoned_sample_ids = gid_train[poison_idx].tolist() if len(poison_idx) > 0 else []
        poisoned_sample_ids_benign = gid_train[poison_idx_benign].tolist() if len(poison_idx_benign) > 0 else []

        # TEST-side category array (mirrors the one above) is built further
        # below, AFTER the TEST-SIDE agent block that now runs post-CL-
        # training (REWORK, pocket-targeted test poisoning) -- poison_idx_test/
        # poison_idx_test_benign aren't known yet at this point in the loop.

        if t == 0:
            scaler = StandardScaler()
            scaler.fit(X_train_for_classifier)

            model = ClassifierNN(feature_dim, 2).to(DEVICE)
            classifier_wrapper = TorchIDSWrapper(model, scaler, DEVICE)
            W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
            omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

            # GLOBAL (run-wide) PCA basis for plot_decision_boundary -- fit
            # ONCE, here, reused unchanged for every task/checkpoint's
            # boundary plot for the rest of the run.
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

        unlearning_metrics = None

        # ============================================================
        # 1. TRAIN on this task's train split.
        # ============================================================
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

        else:
            print(f"  Samples: {len(Xtr)} (+{len(replay_buffer)} replay) | "
                  f"MADAR ER+KD+SI, {CL_ITERS} iters")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = CL_ITERS

            # REWORK (pocket-targeted test poisoning): frozen snapshot of the
            # classifier as it stood BEFORE this task's CL training -- captured
            # here, right before train_cl_er mutates `model` in place, so the
            # TEST-SIDE agent block below (which runs AFTER CL training) can
            # compare "did this exact point's predicted class change" between
            # the two. Independent copy of `model` (not just the wrapper), since
            # TorchIDSWrapper holds its model by reference and `model` itself
            # keeps training.
            pre_train_classifier_wrapper = TorchIDSWrapper(copy.deepcopy(model), scaler, DEVICE)

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

        # TEST-SIDE poisoning (REWORK, pocket-targeted test poisoning): runs HERE,
        # after this task's CL training has finished, so red_test_pert_agent/
        # red_test_benign_pert_agent can be scored against the POST-training
        # classifier (classifier_wrapper/model, now mutated by train_cl_er) while
        # comparing to pre_train_classifier_wrapper (frozen snapshot captured
        # right before train_cl_er, above). NetworkAttackEnv's pocket_term
        # rewards landing where the two disagree in the evasive direction -- a
        # point the CURRENT model calls more benign(-ish) than the PRE-training
        # one did, i.e. a "pocket" THIS TASK's own poisoned training just opened
        # up, not a generic weakness that predates it. See POCKET_SHIFT_WEIGHT's
        # definition for the full rationale. Only runs when this task actually
        # had train-side agents/poisoning above (mirrors the nesting this code
        # used to live inside, before the reorder).
        if t > 0 and len(mal_idx_train) > 0:
            if POISON_TEST_DATA and len(mal_idx_test) > 0:
                # REWORK (proximity-anchored pocket targeting): this task's own
                # TRAIN-side poisoned malicious exemplars -- the exact rows about
                # to become the forget set -- subsampled to PROXIMITY_ANCHOR_MAX
                # for per-step nearest-neighbor cost. See PROXIMITY_WEIGHT's
                # definition for why this exists.
                mal_anchor_X = X_train_pert[poison_idx]
                if len(mal_anchor_X) > PROXIMITY_ANCHOR_MAX:
                    anchor_rng = np.random.RandomState(args.seed + t + 50_000)
                    anchor_sel = anchor_rng.choice(len(mal_anchor_X), size=PROXIMITY_ANCHOR_MAX, replace=False)
                    mal_anchor_X = mal_anchor_X[anchor_sel]

                env_test, agent_test = train_red_agent_for_task(
                    t, "test", classifier_wrapper, X_test, y_test, benign_label, bank, args.seed, out_dir,
                    target_margin_confidence=RED_TARGET_MARGIN_CONFIDENCE,
                    shift_reference_classifier=pre_train_classifier_wrapper,
                    pocket_shift_weight=POCKET_SHIFT_WEIGHT,
                    proximity_anchor_X=mal_anchor_X,
                    proximity_weight=PROXIMITY_WEIGHT,
                    proximity_length_scale=PROXIMITY_LENGTH_SCALE,
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

            if len(benign_idx_train) > 0 and POISON_TEST_DATA and len(benign_idx_test) > 0:
                # REWORK (proximity-anchored pocket targeting): mirror of
                # mal_anchor_X above, on the benign side.
                benign_anchor_X = X_train_pert_benign[poison_idx_benign]
                if len(benign_anchor_X) > PROXIMITY_ANCHOR_MAX:
                    anchor_rng_benign = np.random.RandomState(args.seed + t + 60_000)
                    anchor_sel_benign = anchor_rng_benign.choice(
                        len(benign_anchor_X), size=PROXIMITY_ANCHOR_MAX, replace=False
                    )
                    benign_anchor_X = benign_anchor_X[anchor_sel_benign]

                env_test_benign, agent_test_benign = train_red_agent_for_task(
                    t, "test_benign", classifier_wrapper, X_test, y_test, mal_label, bank, args.seed, out_dir,
                    shift_reference_classifier=pre_train_classifier_wrapper,
                    pocket_shift_weight=POCKET_SHIFT_WEIGHT,
                    proximity_anchor_X=benign_anchor_X,
                    proximity_weight=PROXIMITY_WEIGHT,
                    proximity_length_scale=PROXIMITY_LENGTH_SCALE,
                )
                (X_test_pert_benign, test_evasion_rate_benign, test_rewards_benign, test_avg_norm_benign,
                 test_attacked_benign, evaded_mask_test_benign) = evaluate_agent_on_batch(
                    env_test_benign, agent_test_benign, X_test, y_test, mal_label,
                    only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                )
                red_report["test_benign_evasion_rate"] = test_evasion_rate_benign
                red_report["test_benign_attacked"] = test_attacked_benign
                red_report["test_benign_avg_reward"] = float(np.mean(test_rewards_benign)) if test_rewards_benign else 0.0
                red_report["test_benign_avg_pert_l2"] = test_avg_norm_benign
                print(f"[Task {t} red_test_benign_pert_agent] test_evasion={test_evasion_rate_benign:.3f} "
                      f"avg_pert_L2={test_avg_norm_benign:.3f}")

                poison_fraction_test_benign = poison_fraction_for_task(
                    t, POISON_FRACTION_TEST_START, POISON_FRACTION_TEST_END
                )
                poison_pool_test_benign = np.where(evaded_mask_test_benign)[0] if REQUIRE_EVASION_SUCCESS \
                    else benign_idx_test
                n_to_poison_test_benign = min(
                    int(round(poison_fraction_test_benign * len(benign_idx_test))), len(poison_pool_test_benign)
                )
                rng_test_benign = np.random.RandomState(args.seed + t + 40_000)  # distinct stream, avoids
                                                                                  # colliding with rng_test
                                                                                  # (+10_000), rng_benign
                                                                                  # (+20_000), and pc_rng
                                                                                  # (+30_000, below)
                poison_idx_test_benign = rng_test_benign.choice(
                    poison_pool_test_benign, size=n_to_poison_test_benign, replace=False
                ) if n_to_poison_test_benign > 0 else np.array([], dtype=int)
                n_poisoned_test_benign = len(poison_idx_test_benign)

                X_test = X_test.copy()
                X_test[poison_idx_test_benign] = X_test_pert_benign[poison_idx_test_benign]
                task_test_splits[t] = (X_test, y_test)
                poisoned_test_sample_ids_benign = gid_test[poison_idx_test_benign].tolist() \
                    if n_poisoned_test_benign > 0 else []
                print(f"[Task {t}] poisoned {n_poisoned_test_benign}/{len(benign_idx_test)} benign TEST samples "
                      f"(poison_fraction_test_benign={poison_fraction_test_benign:.3f}) -- "
                      f"test_evasion_rate was {test_evasion_rate_benign:.3f}")

        # TEST-side category array (mirrors the train-side one, above) -- feeds
        # plot_decision_boundary's test-split scatter (poison_idx_test/
        # poison_idx_test_benign rows are exactly what red_test_pert_agent/
        # red_test_benign_pert_agent just produced above).
        category_test = np.full(len(y_test), "malicious_clean", dtype=object)
        category_test[y_test == benign_label] = "benign"
        if len(poison_idx_test) > 0:
            category_test[poison_idx_test] = "malicious_perturbed"
        if len(poison_idx_test_benign) > 0:
            category_test[poison_idx_test_benign] = "benign_perturbed"

        # Decision-boundary snapshot -- model state right after CL training,
        # BEFORE any unlearning attempt for this task (task 0 has no
        # unlearning phase at all, so this is its only snapshot). Whether
        # unlearning actually runs for t>0 is decided further below, so this
        # is captured unconditionally here and labeled "pre_unlearn" for
        # every t>0 regardless of outcome.
        X_test_scaled_np = to_tensor(task_test_splits[t][0]).numpy()
        boundary_panels.append(plot_decision_boundary(
            model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
            Xtr.numpy(), category, X_test_scaled_np, category_test,
            t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}.png"),
            checkpoint_label="pre_unlearn" if t > 0 else "",
        ))

        # ============================================================
        # 2. TEST on the (poisoned) test split -- REORDERED here, right after
        # training and BEFORE this task's forget-set population/unlearning,
        # per explicit request. These per_task_eval/pooled_eval numbers are
        # now the OFFICIAL logged per-task numbers, reflecting the model
        # PRE-unlearning. Nothing is lost by moving them earlier: unlearning's
        # effect on THIS task's own numbers is still captured separately below
        # (unlearning_metrics.post_unlearn_this_task_eval), and its effect on
        # every OLDER task is still captured by prior_tasks_recovery, unchanged.
        # ============================================================
        per_task_eval = {j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t + 1)}
        pooled_X = np.concatenate([task_test_splits[j][0] for j in range(t + 1)])
        pooled_y = np.concatenate([task_test_splits[j][1] for j in range(t + 1)])
        pooled_eval = evaluate_classifier(classifier_wrapper, pooled_X, pooled_y)
        mean_per_task_bal_acc = float(np.mean([per_task_eval[j]["balanced_accuracy"] for j in range(t + 1)]))

        # perturbed_test_eval: accuracy computed ONLY on the poisoned rows of
        # each earlier task's test split (cross-referenced by global sample_id
        # against that task's own poisoned_test_sample_ids), tracked at EVERY
        # checkpoint the same way per_task_eval is -- unlike per_task_eval,
        # which mixes clean+poisoned test rows together. Computed here (step
        # 2, PRE-unlearning) to match per_task_eval's own timing. REWORK
        # (pocket-targeted test poisoning follow-up): a POST-unlearning
        # version of this exact computation (post_unlearn_perturbed_test_eval)
        # is now ALSO logged, further below where post_unlearn_this_task_eval
        # is -- so both decision-boundary-N (this dict) and decision-boundary-J
        # (the post-unlearn one) accuracy on the identical pocket-targeted
        # batch are directly comparable, not just inferred from the
        # whole-test-set post_unlearn_this_task_eval number. Omits task j
        # entirely if it has no poisoned test rows (task 0, or POISON_TEST_DATA
        # off).
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

        print(f"[Task {t} classifier, PRE-unlearning] this-task bal_acc={per_task_eval[t]['balanced_accuracy']:.4f} "
              f"pooled bal_acc={pooled_eval['balanced_accuracy']:.4f} "
              f"mean-per-task bal_acc={mean_per_task_bal_acc:.4f}")

        # ============================================================
        # 3. FORGET SET POPULATION -- build_perturbation_classifier_forget_set
        # trains a small 3-class RandomForest per task on PERTURBATION_
        # CLASSIFIER_N oracle-seeded samples per class, then uses ITS
        # predictions (not oracle ground truth) to build the forget set --
        # see that function's docstring and the module docstring's REWORK
        # note for the full design.
        # 4. ISOLATION FOREST / REPLAY BUFFER POPULATION -- update_buffer_madar
        # now always fills from the FULL task batch, unfiltered (REWORK --
        # forget/retain above shapes ONLY unlearning's retain_loader anchor
        # now, not buffer eligibility). Whatever this leaves in replay_buffer
        # feeds task t+1's training (step 1, next iteration) via train_cl_er.
        # 5. REPLAY BUFFER CLEANING (CLEAN_REPLAY_BUFFER_OF_PERTURBED toggle)
        # -- after buffer fill, remove any entries matching
        # potentially_perturbed_pool's global sample ids, then refill with
        # this task's own clean-malicious samples up to budget.
        # ============================================================
        if t == 0:
            teacher_model = copy.deepcopy(model); teacher_model.eval()
            update_buffer_madar(Xtr, ytr, category, gid_train, benign_label, mal_label, model, DEVICE)

        else:
            pc_rng = np.random.RandomState(args.seed + t + 30_000)  # distinct stream from train/test poison rngs
            pc_result = build_perturbation_classifier_forget_set(
                Xtr, ytr, poison_idx, poison_idx_benign, benign_label, mal_label, rng=pc_rng
            )

            if pc_result is None:
                print(f"    [Perturbation classifier] fewer than {PERTURBATION_CLASSIFIER_N} samples available "
                      f"in one of benign/malicious_clean/malicious_perturbed/benign_perturbed this task -- "
                      f"skipping classifier/unlearning phase for this task.")
                unlearning_metrics = {"detector": POISON_DETECTOR, "n_forget": 0}
            else:
                forget_idx, retain_idx = pc_result["forget_idx"], pc_result["retain_idx"]
                cm, tcm = pc_result["confusion_matrix"], pc_result["two_class_metric"]
                print(f"    [Perturbation classifier] n_used={pc_result['n_used']}/class, "
                      f"n_eval={pc_result['n_eval']}, 2-class acc={tcm['accuracy']:.3f} "
                      f"bal_acc={tcm['balanced_accuracy']:.3f}")
                print(f"    [Perturbation classifier confusion] classes={cm['classes']}, matrix={cm['matrix']}")
                print(f"    [Forget set] potentially_perturbed_pool (malicious) = "
                      f"{len(pc_result['forget_idx_malicious'])}, potentially_perturbed_benign_pool = "
                      f"{len(pc_result['forget_idx_benign'])}, combined forget_idx = {len(forget_idx)} samples "
                      f"(retain = {len(retain_idx)})")

                if len(forget_idx) == 0:
                    print("    [Unlearning] forget set empty -- "
                          "skipping unlearning phase for this task.")
                    unlearning_metrics = {
                        "detector": POISON_DETECTOR, "n_forget": 0,
                        "n_forget_malicious": int(len(pc_result["forget_idx_malicious"])),
                        "n_forget_benign": int(len(pc_result["forget_idx_benign"])),
                        "perturbation_classifier": {
                            "n_used": pc_result["n_used"], "n_eval": pc_result["n_eval"],
                            "confusion_matrix": cm, "two_class_metric": tcm,
                        },
                    }
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
                    print(f"    [Unlearn] task {t} (alpha={UNLEARN_ALPHA}, si_c={UNLEARN_SI_C}, "
                          f"n_forget={len(forget_idx)}, n_retain={len(retain_idx)})...")
                    unlearn_diag = unlearn_teacher_guided(
                        model=model, teacher_model=model_pre_unlearn,
                        forget_loader=forget_loader, retain_loader=retain_loader,
                        omega=omega, p_old_task=p_old_task, si_c=UNLEARN_SI_C,
                        epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA, device=DEVICE,
                    )
                    grad_steps += UNLEARN_EPOCHS * len(forget_loader)
                    print(f"    [Unlearn diag] omega_l2={unlearn_diag['omega_l2_norm']:.3f}, raw losses "
                          f"forget={unlearn_diag['raw_forget_loss_mean']:.4f} "
                          f"retain={unlearn_diag['raw_retain_loss_mean']:.4f} "
                          f"si={unlearn_diag['raw_si_loss_mean']:.4f} (unweighted, before alpha/si_c)")

                    # Decision-boundary snapshot -- model state right after
                    # unlearning, paired with the "pre_unlearn" snapshot taken
                    # above (same task, same fixed PCA basis, same category
                    # arrays) so the two are directly comparable.
                    boundary_panels.append(plot_decision_boundary(
                        model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                        Xtr.numpy(), category, X_test_scaled_np, category_test,
                        t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}_post_unlearn.png"),
                        checkpoint_label="post_unlearn",
                    ))

                    f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, DEVICE)
                    r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, DEVICE)

                    # --- Step 7, part 2: same prior-task snapshot AFTER unlearning ---
                    post_unlearn_prior_eval = {
                        j: evaluate_classifier(classifier_wrapper, *task_test_splits[j]) for j in range(t)
                    }

                    # per_task_eval[t]/pooled_eval above are the PRE-unlearning
                    # snapshot, so this task's own post-unlearning number needs
                    # its own explicit checkpoint -- otherwise it would only
                    # ever be visible once task t+1 evaluates it as a "prior task".
                    post_unlearn_this_task_eval = evaluate_classifier(classifier_wrapper, *task_test_splits[t])
                    print(f"    [This task] bal_acc after its own unlearning: "
                          f"{per_task_eval[t]['balanced_accuracy']:.4f} -> "
                          f"{post_unlearn_this_task_eval['balanced_accuracy']:.4f}")

                    # POST-unlearn perturbed_test_eval (REWORK, pocket-targeted test
                    # poisoning follow-up): same rows/mask logic as perturbed_test_eval
                    # above (isolated to just the poisoned test rows, per earlier task
                    # j), re-scored against THIS classifier state -- i.e. decision
                    # boundary J from the pocket-targeting algorithm: A (pre-training)
                    # -> N (post-CL-training, what perturbed_test_eval above already
                    # measures) -> J (post-unlearning, measured here). Lets a future
                    # run check directly whether unlearning recovers accuracy on the
                    # exact pocket-targeted batch, instead of only inferring it from
                    # post_unlearn_this_task_eval's whole-test-set number (which mixes
                    # in clean rows too).
                    post_unlearn_perturbed_test_eval = {}
                    for j in range(t + 1):
                        poisoned_gids_j = set(task_poisoned_test_gids.get(j, []))
                        if not poisoned_gids_j:
                            continue
                        mask = np.isin(task_test_gids[j], list(poisoned_gids_j))
                        if mask.sum() == 0:
                            continue
                        X_test_j, y_test_j = task_test_splits[j]
                        post_unlearn_perturbed_test_eval[j] = evaluate_classifier(
                            classifier_wrapper, X_test_j[mask], y_test_j[mask]
                        )
                    if t in perturbed_test_eval and t in post_unlearn_perturbed_test_eval:
                        print(f"    [Pocket-targeted test batch] bal_acc after this task's unlearning: "
                              f"{perturbed_test_eval[t]['balanced_accuracy']:.4f} -> "
                              f"{post_unlearn_perturbed_test_eval[t]['balanced_accuracy']:.4f}")

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
                        "n_forget_malicious": int(len(pc_result["forget_idx_malicious"])),
                        "n_forget_benign": int(len(pc_result["forget_idx_benign"])),
                        "perturbation_classifier": {
                            "n_used": pc_result["n_used"], "n_eval": pc_result["n_eval"],
                            "confusion_matrix": cm, "two_class_metric": tcm,
                        },
                        "unlearn_diagnostics": unlearn_diag,
                        "forget_set": f_m, "retain_set": r_m,
                        "post_unlearn_this_task_eval": post_unlearn_this_task_eval,
                        "post_unlearn_perturbed_test_eval": post_unlearn_perturbed_test_eval,
                        "prior_tasks_recovery": {
                            "pre_unlearn": pre_unlearn_prior_eval,
                            "post_unlearn": post_unlearn_prior_eval,
                        },
                    }

            teacher_model = copy.deepcopy(model); teacher_model.eval()

            # Buffer fills from the FULL task batch now, unfiltered (REWORK,
            # see module docstring) -- matches plain MADAR's/task-0's call
            # pattern exactly.
            update_buffer_madar(Xtr, ytr, category, gid_train, benign_label, mal_label, model, DEVICE)

            # Step 5: post-hoc buffer cleaning (toggleable). Only meaningful
            # when the classifier actually ran and found something. Mirrored
            # per label (REWORK, quad red agents): malicious slot cleaned of
            # forget_idx_malicious matches / refilled from
            # historical_clean_mal_pool candidates (unchanged from before
            # benign agents existed); benign slot cleaned of forget_idx_benign
            # matches / refilled from historical_clean_benign_pool candidates
            # (new, exact mirror) -- same uncertainty-sampling + cross-task
            # pool + FIFO-cap mechanics either way.
            buffer_refill_diag = {}
            if CLEAN_REPLAY_BUFFER_OF_PERTURBED and pc_result is not None:
                budget_per_label = MEM_SIZE // 2

                def _clean_and_refill(label, forget_idx_for_label, historical_pool, category_tag):
                    """REFILL (REWORK, uncertainty sampling): top `label`'s buffer
                    slot back up to budget with clean samples of that label -- i.e.
                    that label AND NOT in this task's forget pool for it (the
                    classifier's judgment, not oracle ground truth -- staying
                    consistent with the rest of this pipeline never peeking at
                    poison_idx/poison_idx_benign beyond seeding the classifier's
                    training labels). Candidates come from THIS task's own
                    leftovers AND every past task's clean samples of that label
                    (historical_pool), combined into one pool. Selection is
                    uncertainty sampling: the CURRENT model (post this task's CL
                    training + unlearning) scores every candidate's confidence
                    (_predict_confidence, max class probability), and refill takes
                    the LEAST confident first. Still only fills the deficit left
                    by cleaning -- not a full replace of whatever survived
                    cleaning."""
                    forget_gids = set(int(g) for g in gid_train[forget_idx_for_label])
                    before = len(label_buffers.get(label, []))
                    label_buffers[label] = [
                        e for e in label_buffers.get(label, []) if e[3] not in forget_gids
                    ]
                    n_removed = before - len(label_buffers[label])

                    deficit = budget_per_label - len(label_buffers[label])
                    n_refilled = 0
                    refill_confidence_diag = None
                    if deficit > 0:
                        already_buffered = set(int(e[3]) for e in label_buffers[label])
                        ytr_np = ytr.numpy()
                        forgotten_set = set(int(i) for i in forget_idx_for_label)
                        this_task_entries = [
                            (Xtr[i].clone(), ytr[i].clone(), category[i], int(gid_train[i]))
                            for i in range(len(Xtr))
                            if ytr_np[i] == label and i not in forgotten_set
                            and int(gid_train[i]) not in already_buffered
                        ]
                        historical_entries = [
                            e for e in historical_pool if e[3] not in already_buffered
                        ]
                        combined_entries = this_task_entries + historical_entries

                        if combined_entries:
                            n_refill = min(deficit, len(combined_entries))
                            cand_X = torch.stack([e[0] for e in combined_entries])
                            confidence = _predict_confidence(model, cand_X, DEVICE)
                            ranked_idx = np.argsort(confidence)  # ascending: least confident first
                            local_idx = ranked_idx[:n_refill]

                            for li in local_idx:
                                label_buffers[label].append(combined_entries[li])
                            n_refilled = len(local_idx)

                            # Diagnostic (task 8/9 collapse investigation): confidence
                            # distribution of the SELECTED refill samples vs. the full
                            # candidate pool, so a future run can directly check whether
                            # refill is pulling from a narrow band of near-boundary
                            # samples and whether that band shifts/widens across tasks.
                            selected_conf = confidence[local_idx]
                            refill_confidence_diag = {
                                "n_candidates": int(len(combined_entries)),
                                "n_this_task_candidates": int(len(this_task_entries)),
                                "n_historical_candidates": int(len(historical_entries)),
                                "pool_confidence_mean": float(np.mean(confidence)),
                                "pool_confidence_min": float(np.min(confidence)),
                                "pool_confidence_max": float(np.max(confidence)),
                                "selected_confidence_mean": float(np.mean(selected_conf)),
                                "selected_confidence_min": float(np.min(selected_conf)),
                                "selected_confidence_max": float(np.max(selected_conf)),
                            }

                    print(f"    [Buffer clean] label={category_tag}: removed {n_removed} forget-pool "
                          f"matches, refilled {n_refilled}/{deficit} with least-confident clean "
                          f"{category_tag} samples (this task + {len(historical_pool)} historical "
                          f"candidates).")
                    if refill_confidence_diag is not None:
                        print(f"    [Buffer refill confidence] label={category_tag} selected "
                              f"mean/min/max={refill_confidence_diag['selected_confidence_mean']:.3f}/"
                              f"{refill_confidence_diag['selected_confidence_min']:.3f}/"
                              f"{refill_confidence_diag['selected_confidence_max']:.3f} vs. pool "
                              f"mean={refill_confidence_diag['pool_confidence_mean']:.3f}")

                    return {
                        "n_removed": int(n_removed),
                        "deficit": int(deficit),
                        "n_refilled": int(n_refilled),
                        "historical_pool_size_before_refill": int(len(historical_pool)),
                        "refill_confidence": refill_confidence_diag,
                    }

                if len(pc_result["forget_idx_malicious"]) > 0:
                    buffer_refill_diag["malicious"] = _clean_and_refill(
                        mal_label, pc_result["forget_idx_malicious"], historical_clean_mal_pool, "malicious"
                    )
                if len(pc_result["forget_idx_benign"]) > 0:
                    buffer_refill_diag["benign"] = _clean_and_refill(
                        benign_label, pc_result["forget_idx_benign"], historical_clean_benign_pool, "benign"
                    )

                replay_buffer.clear()
                for buf in label_buffers.values():
                    replay_buffer.extend(buf)

                if buffer_refill_diag:
                    unlearning_metrics["buffer_refill"] = buffer_refill_diag

            # Grow the cross-task historical pools with THIS task's own clean
            # samples, AFTER this task's own refill above (so a task's own
            # samples are never candidates for its own refill -- only
            # genuinely PAST tasks' samples are). Independent of the
            # CLEAN_REPLAY_BUFFER_OF_PERTURBED toggle -- both pools are always
            # maintained when a classifier judgment exists, regardless of
            # whether cleaning/refill happened to run this task. Mirrored per
            # label (REWORK, quad red agents): historical_clean_mal_pool grows
            # from malicious/forget_idx_malicious as before,
            # historical_clean_benign_pool grows from benign/forget_idx_benign
            # (new, exact mirror).
            if pc_result is not None:
                ytr_np = ytr.numpy()
                forgotten_set_malicious = set(int(i) for i in pc_result["forget_idx_malicious"])
                forgotten_set_benign = set(int(i) for i in pc_result["forget_idx_benign"])
                for i in range(len(Xtr)):
                    if ytr_np[i] == mal_label and i not in forgotten_set_malicious:
                        historical_clean_mal_pool.append(
                            (Xtr[i].clone(), ytr[i].clone(), category[i], int(gid_train[i]))
                        )
                    elif ytr_np[i] == benign_label and i not in forgotten_set_benign:
                        historical_clean_benign_pool.append(
                            (Xtr[i].clone(), ytr[i].clone(), category[i], int(gid_train[i]))
                        )
                # FIFO cap (see HISTORICAL_POOL_MAX_SIZE definition): drop the
                # oldest-appended entries first once either pool exceeds budget.
                if len(historical_clean_mal_pool) > HISTORICAL_POOL_MAX_SIZE:
                    historical_clean_mal_pool[:] = historical_clean_mal_pool[-HISTORICAL_POOL_MAX_SIZE:]
                if len(historical_clean_benign_pool) > HISTORICAL_POOL_MAX_SIZE:
                    historical_clean_benign_pool[:] = historical_clean_benign_pool[-HISTORICAL_POOL_MAX_SIZE:]
                # Always logged (regardless of whether cleaning/refill ran this
                # task) so pool-size-over-time is visible in the results JSON
                # even on tasks where the classifier found nothing to forget.
                unlearning_metrics["historical_pool_size"] = {
                    "malicious": int(len(historical_clean_mal_pool)),
                    "benign": int(len(historical_clean_benign_pool)),
                }

        # p_old_task updated to the FINAL (post-unlearning, if it ran this task)
        # weights, for ALL tasks -- ready as next task's start-of-task SI anchor.
        for n, p in model.named_parameters():
            if p.requires_grad:
                p_old_task[n.replace('.', '__')] = p.detach().clone()

        buffer_summary = buffer_composition_summary()
        #print(f"    [Buffer] composition: {buffer_summary}")

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
            "red_agent": red_report,
            "per_task_eval": per_task_eval,
            "perturbed_test_eval": perturbed_test_eval,
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
        "poison_schedule_enabled": POISON_SCHEDULE_ENABLED, "poison_fraction_flat": POISON_FRACTION_FLAT,
        "poison_fraction_train_start": POISON_FRACTION_TRAIN_START,
        "poison_fraction_train_end": POISON_FRACTION_TRAIN_END,
        "poison_fraction_test_start": POISON_FRACTION_TEST_START,
        "poison_fraction_test_end": POISON_FRACTION_TEST_END,
        "require_evasion_success": REQUIRE_EVASION_SUCCESS,
        "poison_test_data": POISON_TEST_DATA,
        "perturbation_classifier_n": PERTURBATION_CLASSIFIER_N,
        "clean_replay_buffer_of_perturbed": CLEAN_REPLAY_BUFFER_OF_PERTURBED,
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
        "unlearn_si_c": UNLEARN_SI_C,
    }
    with open(os.path.join(out_dir, f"{args.log_name}.json"), "w") as f:
        json.dump({"config": config, "warnings": warnings_log, "results": results}, f, indent=2)

    plot_task_metrics(results, os.path.join(out_dir, "plots", "task_metrics.png"))
    plot_unlearning_metrics(results, os.path.join(out_dir, "plots", "unlearning_metrics.png"))
    plot_prototype_heatmap(bank, os.path.join(out_dir, "plots", "prototype_heatmap.png"))
    plot_episode_clouds(bank, os.path.join(out_dir, "plots", "episode_clouds.png"))
    plot_decision_boundary_grid(boundary_panels, os.path.join(out_dir, "plots", "decision_boundary_evolution.png"))

    print(f"\nDone. Results + plots written to {out_dir}/")
    print("full time elapsed: %.2f seconds" % (time.perf_counter() - start_time))


if __name__ == "__main__":
    main()
