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

# REWORK (uncertainty-margin pipeline): replaces the old per-task ramp
# schedule (POISON_SCHEDULE_ENABLED/POISON_FRACTION_TRAIN/TEST_START/END) and
# its flat fallback (POISON_FRACTION_FLAT) with two fixed, always-active
# fractions -- no schedule, no toggle. Both are PER-CLASS quotas (malicious
# and benign each get their own independent budget of this size), not a
# combined/shared budget.
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

# NEW (variant-augmentation branch): train-side variant generation. For every
# row that successfully evades C2 (poison_idx/poison_idx_benign), the SAME
# trained agent keeps perturbing that exact point for up to
# MAX_VARIANTS_PER_SAMPLE more "levels," VARIANT_STEPS_PER_LEVEL actions each,
# CHAINED (level 2 continues from level 1's landing point, not from the
# original evasion point). A level only becomes a real variant if the point is
# STILL evading C2 after its steps; the first level that fails to still evade
# stops generation for that sample (see generate_train_variants). Goal (not
# enforced as a hard constraint -- see that function's docstring): bring the
# perturbed row count up to roughly match the (1 - TRAIN_UNCERTAIN_FRACTION)
# clean row count for that class; if MAX_VARIANTS_PER_SAMPLE isn't enough to
# close that gap, the shortfall is left as-is.
VARIANT_STEPS_PER_LEVEL = 3
MAX_VARIANTS_PER_SAMPLE = 5
# Variant rows get a real, unique gid stored in gid_train (so every existing
# int(gid_train[...]) cast -- buffer sample_id, forget-set gid matching --
# keeps working on them unchanged): VARIANT_GID_OFFSET + original_gid*10 +
# level. Translated back to the requested "<original>.<level>" display string
# only where a human actually reads it (poisoned_sample_ids, logs) via
# _variant_gid_display -- see its docstring. 10**9 is far beyond any real gid
# in this dataset (pooled row counts are in the tens of thousands), so this
# can never collide with a genuine sample's gid.
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
RED_C1_MISMATCH_PATIENCE = 3  # FINETUNE (test-side patience investigation): pgd_boundary_
                               # search_batch's shift_reference_classifier (C1) check used to
                               # hard-abort the walk on the FIRST step where C1 disagreed with
                               # the true label -- a diagnostic run showed this killed 99.5% of
                               # malicious test attacks within ~2 steps, using only ~8-12% of
                               # the epsilon budget, i.e. the walk almost never got a real
                               # chance. This lets it tolerate up to this many CONSECUTIVE
                               # C1-mismatch steps (resets to 0 the moment C1 agrees again)
                               # before giving up, so a transient wobble across C1's own
                               # boundary doesn't necessarily end the episode. Success still
                               # requires C1-correct AND C2-evaded on the SAME step regardless
                               # (the pocket definition itself is unchanged) -- this only
                               # widens how much rope the search gets to find that step.
RED_TIMESTEPS_PER_TASK = 5000  # REWORK (uncertainty-margin pipeline): doubled from 2500 --
                                # POTENTIALLY REVISIT, tune from real run data. Proposed
                                # because the test-side joint C1-correct/C2-wrong success
                                # condition (kept from the previous branch) was already a
                                # sparse target under 2500 steps -- real runs showed test
                                # evasion/success rates collapsing toward 0 once that
                                # condition became mandatory, which reads as SAC not getting
                                # enough training signal to reliably find the narrow
                                # "still-C1-correct" window rather than the objective being
                                # unreachable. Doubling is a first, conservative step (not
                                # tuned against this specific pipeline's real numbers yet) --
                                # revisit upward or downward once a run's actual success
                                # rates are in.
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
                                      # it specifically, unlike THIS file's
                                      # perturbation_classifier (an explicitly boundary-adjacent
                                      # detector). Set to None to restore the original
                                      # confidence-maximizing reward for both agents. Identical
                                      # constant/value to madar_cl_pipeline.py's, so both
                                      # pipelines face the same attack.
RED_TARGET_MARGIN_HIGH = 0.65  # NEW (gradient-attack branch): upper end of the test-side
                                # target confidence band [RED_TARGET_MARGIN_CONFIDENCE,
                                # RED_TARGET_MARGIN_HIGH] pgd_boundary_search_batch keeps
                                # stepping toward past the first boundary crossing. Mirrors
                                # what NetworkAttackEnv's peaked confidence_term reward used
                                # to encourage (without ever enforcing it as a hard bound) --
                                # here it's the PGD loop's actual stopping target.

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
                                      # madar_cl_pipeline.py's.

# REWORK (gradient-attack branch): POCKET_SHIFT_WEIGHT/PROXIMITY_SHAPING_ENABLED/
# PROXIMITY_WEIGHT/PROXIMITY_LENGTH_SCALE/PROXIMITY_ANCHOR_MAX (dense RL reward-
# shaping terms for red_test_pert_agent/red_test_benign_pert_agent, both now
# pgd_boundary_search_batch calls -- see its module-level comment) removed
# entirely: there is no reward being optimized in a direct gradient search,
# just a loss gradient, so these had no analog to carry forward. Test-side
# success is still governed by the same joint C1-correct/C2-wrong condition
# (now pgd_boundary_search_batch's shift_reference_classifier check); only the
# dense shaping bonuses that used to bias RL exploration toward it are gone.

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
SI_OMEGA_DECAY = 0.9  # REWORK (SI capacity investigation): applied to `omega`
                       # (SI's accumulated per-parameter importance) each task,
                       # BEFORE that task's new contribution is added --
                       # omega[k] = SI_OMEGA_DECAY * omega[k] + this_task's_delta
                       # -- same "online EWC" forgetting-factor idea (Schwarz et
                       # al. 2018) used to stop Fisher/importance accumulation
                       # from growing without bound over many tasks. Previously
                       # omega had NO cap or decay at all, unlike every other
                       # accumulating state in this pipeline (replay buffer,
                       # historical_clean_mal_pool/historical_clean_benign_pool
                       # -- both since capped). Motivated by a recurring pattern
                       # across three separate experiments (margin-evasion-only,
                       # quad-agent, and pocket-anchored runs): a SPECIFIC subset
                       # of early tasks (task 1 in one run; tasks 0/1/5 in
                       # another) collapses hard (-17 to -19pp) at the FINAL
                       # checkpoint only, while omega_l2_norm climbs
                       # monotonically the whole run with no anomalous jump at
                       # that point -- consistent with SI's "protected" subspace
                       # quietly shrinking every task until there's no longer
                       # enough free capacity left, at which point fitting new
                       # data forces a large, uncontrolled reallocation.
                       # 0.9 is a first-pass guess (not yet tuned) -- at 0.9,
                       # omega converges toward a geometric-series steady state
                       # instead of growing linearly forever; lower values decay
                       # faster (less long-run protection, more plasticity),
                       # 1.0 exactly reproduces the old unbounded-growth
                       # behavior. Applied identically in madar_cl_pipeline.py
                       # (plain MADAR also uses SI, though it has no unlearning
                       # step to interact with) so results stay comparable.
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
                          # targeting forgetting"). REWORK (uncertainty-margin
                          # pipeline): left UNCHANGED even though the forget loss
                          # itself changed (KL-to-uniform -> corrective cross-entropy,
                          # see unlearn_teacher_guided) -- POTENTIALLY REVISIT. The new
                          # loss is a much more directly targeted signal than the old
                          # entropy-maximizing one, so this alpha/UNLEARN_SI_C balance
                          # (tuned against the OLD loss) may need retuning once real
                          # runs show how it behaves against the new one.
UNLEARN_SI_C = 0.1  # SI penalty weight during UNLEARNING ONLY -- deliberately
                     # decoupled from SI_C (still 1.0 for ordinary CL training,
                     # unchanged). First smoke test (madar_u1.json) showed forget-set
                     # accuracy completely unmoved on several tasks despite real
                     # gradient steps, consistent with SI's accumulated-omega penalty
                     # dominating the forget objective's gradient at si_c=1.0 -- this
                     # is a first-pass guess at a softer value, NOT yet tuned; see the
                     # loss-component diagnostics logged in unlearn_teacher_guided's
                     # return value for the data to tune it against.

FULL_RETRAIN_DIAGNOSTIC_TASKS = {8, 9}  # DIAGNOSTIC (retrain-vs-unlearn comparison): at
                                          # exactly these task ids, skip unlearn_teacher_
                                          # guided entirely and instead fully retrain a
                                          # fresh classifier on ONLY the retain set (see
                                          # retrain_classifier_from_scratch) -- the "exact
                                          # unlearning" gold standard, for comparison
                                          # against the approximate SI/KD-anchored method
                                          # used every other task.
RETRAIN_FULL_EPOCHS = TASK0_EPOCHS  # same budget as the very first from-scratch training
                                     # at task 0, since this is the same kind of
                                     # from-scratch supervised fit, just on the retain
                                     # set instead of task 0's full training split.
RETRAIN_FULL_LR = 1e-3  # matches task 0's pretraining optimizer's lr.
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
# REWORK (revisited Q11 answer): forget-set selection is back to hard
# classification -- every held-out eval row (see build_perturbation_classifier_
# forget_set's eval_idx) whose predicted label is LABEL_PERTURBED (malicious
# side) or LABEL_BENIGN_PERTURBED (benign side) goes into the forget set, 100%
# of them, no probability ranking and no FORGET_SET_TOP_FRACTION quota. This
# replaces the earlier top-50%-by-probability-per-class rule (which itself had
# replaced an identical hard-classify rule) -- forget-set size now tracks
# perturbation_classifier's actual positive rate on each class directly,
# whatever that happens to be a given task, rather than being pinned to a
# fixed fraction of that class's eval-row count.
CLEAN_REPLAY_BUFFER_OF_PERTURBED = True  # toggle: after IsolationForest/buffer fill (which
                                          # draws from the FULL task batch, unfiltered -- see
                                          # module docstring), remove any buffer entries that
                                          # match this task's forget-pool gids. REWORK
                                          # (uncertainty-margin pipeline): no refill anymore --
                                          # POTENTIALLY REVISIT -- the buffer is left under
                                          # budget until a later task's own fill tops it back up.


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
    (final_obs, total_reward, pred_label, cum_perturbation, success).

    REWORK (joint pocket objective / "Option B"): `success` is
    info["success"] from the env's LAST step -- True only when the episode
    ended via `terminated` (the env's own joint C1-still-correct AND
    C2-now-target condition when a reference classifier is configured, or
    plain C2 evasion otherwise), never when it merely ran out of steps
    (`truncated`). Callers should use this instead of re-deriving success
    from pred_label alone, which would silently drop the C1 requirement.
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
    the exact set poisoning (and, downstream, the oracle detector) should draw
    from.

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

    REWORK (joint pocket objective / "Option B"): evasion/evaded_mask is now
    keyed to run_single_agent_attack's `success` flag (the env's own
    terminated condition), not a re-derived pred_label==benign_label check --
    for envs with a shift_reference_classifier (test/test_benign), that means
    a sample only counts as evaded when the reference classifier (C1) STILL
    calls the final perturbed point the sample's true label AND the live
    classifier (C2) now calls it benign_label. Train-side envs (no reference
    classifier) are unaffected -- success there is still plain C2 evasion.
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
    """Translates an internal variant gid (VARIANT_GID_OFFSET +
    original_gid*10 + level) back to the requested "<original>.<level>"
    string; an ordinary (non-variant) gid is returned unchanged as a plain
    int. Used only at output points a human reads (poisoned_sample_ids,
    logs) -- gid_train itself always stores the raw internal integer, so
    every existing int(gid_train[...]) cast elsewhere is unaffected."""
    internal_gid = int(internal_gid)
    if internal_gid < VARIANT_GID_OFFSET:
        return internal_gid
    remainder = internal_gid - VARIANT_GID_OFFSET
    original_gid, level = divmod(remainder, 10)
    return f"{original_gid}.{level}"


def generate_train_variants(env, red_agent, X_pert, evaded_mask, gid_array, true_label,
                             steps_per_level=VARIANT_STEPS_PER_LEVEL, max_variants=MAX_VARIANTS_PER_SAMPLE,
                             deterministic=True):
    """
    NEW (variant-augmentation branch): for every row where evaded_mask[i] is
    True (already successfully evaded C2 -- see evaluate_agent_on_batch),
    keeps `red_agent` perturbing that EXACT landing point for up to
    max_variants more "levels," steps_per_level raw env.step() actions each,
    CHAINED -- level 2 continues from level 1's landing point, not from the
    original evasion point, so each level drifts further. A level is kept
    as a genuine variant only if the point is STILL classified as this
    agent's target class (env.step()'s info["success"], the same joint
    condition run_single_agent_attack/evaluate_agent_on_batch already use --
    for train-side envs this is unconditionally just C2 evasion, no
    shift_reference_classifier) after its steps_per_level actions; the
    FIRST level that fails to still evade stops generation for that sample
    entirely (no further levels attempted, and the failed level itself is
    discarded, per explicit confirmation).

    Bypasses env.reset() -- resuming a specific mid-episode state isn't
    something NetworkAttackEnv supports natively, so this sets env.state/
    env.true_label/env.cum_action/env.steps directly before each level's
    steps, then calls env.step() exactly steps_per_level times. `env` and
    `red_agent` should be the SAME (env, agent) pair evaluate_agent_on_batch
    was just run with, so the agent's learned policy (not a fresh/reset one)
    is what continues the perturbation.

    Known side effect: if env has a contrastive_bank, its per-episode
    update/log_episode/log_cosine_snapshot calls (normally fired once at
    natural episode termination) will fire again on every one of these
    extra steps too, since success re-evaluates True on each still-evading
    step -- harmless (it's still logging this agent's real behavior), just
    more frequent than a normal episode boundary during variant generation.

    Returns a dict: X (stacked variant feature rows, shape (n_variants,
    feat_dim), empty (0, feat_dim) array if none), internal_gids (int array,
    same order, see VARIANT_GID_OFFSET), display_gids (list of "N.L"
    strings, same order), n_variants_by_original (dict original_gid -> how
    many variants it produced, for logging/parity-tracking).
    """
    # BUGFIX: `env` as passed in from main() is train_red_agent_for_task's
    # make_vec_env(...).envs[0] -- a Monitor-wrapped NetworkAttackEnv, not the
    # raw env. Monitor tracks its own "needs_reset" flag and raises
    # RuntimeError on any step() call after an episode ended without an
    # intervening reset() -- exactly what evaluate_agent_on_batch's last
    # episode already triggered before this function ever runs. Unwrap once
    # here so every env.state=.../env.step() call below operates on the raw
    # NetworkAttackEnv directly, bypassing Monitor's bookkeeping entirely
    # (NetworkAttackEnv.step() itself has no such restriction).
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


# ---------------------------------------------------------------------------
# NEW (gradient-attack branch): PGD (Projected Gradient Descent) nearest-
# boundary search, replacing the SAC/NetworkAttackEnv rollout for TEST-SIDE
# agents only (test/test_benign) -- train-side agents (train/train_benign,
# feeding generate_train_variants above) stay RL/SAC, unchanged.
#
# No policy is trained. For each sample, independently: take a small step in
# the direction that most decreases cross-entropy(model(x), target_label)
# w.r.t. x (the normalized input gradient), clipped to stay within an L2
# `epsilon` ball of the ORIGINAL point and within [0,1] feature bounds --
# same perturbation budget/valid-range semantics NetworkAttackEnv's action
# space and self.state clipping already used. Stop as soon as the point
# crosses the boundary (predicted_label == target_label) AND, if a reference
# classifier is given, that reference classifier STILL calls this point the
# sample's true label -- otherwise the joint condition is broken and the
# episode fails right there (this is the SAME check adversary_env.py's
# NetworkAttackEnv.step() runs every step; a gradient step only ever moves
# toward the LIVE classifier's boundary, so once broken it has no mechanism
# to wander back, unlike a stochastic RL policy might within its remaining
# steps). Once evading, a few more of the same small steps continue past the
# first flip specifically to land the confidence toward target_label inside
# [margin_low, margin_high] (mirrors the old RED_TARGET_MARGIN_CONFIDENCE
# reward's peak-at-0.65 band) rather than stopping at the bare first crossing
# -- but a sample that starts evading and never reaches margin_low before
# max_steps runs out (or overshoots past margin_high in a single step) still
# counts as a success once evaded, matching the old RL definition ("any
# sample flipped to the target class is successful").
#
# Two RL-specific reward-shaping terms have no analog here and are dropped
# structurally, not by a tunable toggle: POCKET_SHIFT_WEIGHT/proximity-
# anchoring (both were dense REWARD bonuses shaping an RL policy's
# exploration -- there's no reward being optimized in a direct gradient
# search, just a loss gradient) and the contrastive "dissimilarity from
# previous tasks' agents" bank (per explicit confirmation, dropped).
#
# Same return contract as evaluate_agent_on_batch -- (X_pert, evasion_rate,
# rewards, avg_pert_norm, attacked, evaded_mask) -- so every existing call
# site downstream (poison_idx_test_ext construction, red_report fields,
# print statements) works completely unchanged; only the two TEST-SIDE call
# sites that used to call train_red_agent_for_task + evaluate_agent_on_batch
# now call this instead.
# ---------------------------------------------------------------------------
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

    # Same selection/truncation semantics as evaluate_agent_on_batch, so
    # swapping the two in a caller changes nothing about WHICH rows get
    # attacked, only HOW.
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
            step = -grad / grad_norm  # descend the loss -> toward benign_label

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
                # first one -- see RED_C1_MISMATCH_PATIENCE's comment for why.
                # A step where C1 disagrees still can't count as success below
                # (c1_correct is False this iteration either way), it just no
                # longer necessarily ends the episode.
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
    correct under C1" row instead of a uniformly-random one (see
    NetworkAttackEnv's allowed_start_indices docstring for why that matters).

    Logs the raw pool size (how many candidates were C1-correct out of the
    full label pool) either way -- to the console, and to the pocket-
    targeting diagnostic log when a path is given -- purely as data, so the
    minimum-pool fallback threshold (currently: skip only when this returns
    empty) can be recalibrated from real numbers later instead of guessed.
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
    decision boundary) first. Same underlying notion of "confidence" as
    _predict_confidence (used by buffer refill), but works directly off a
    classifier wrapper's predict_proba on RAW [0,1] features instead of a
    raw model + already-scaled tensor, since callers here (train/test-side
    selection) always have a classifier wrapper and raw X on hand already.
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

    allowed_start_indices (PROTOTYPE, C1-correct-only sampling): forwarded
    straight to NetworkAttackEnv -- see its docstring. Caller passes the
    subset of this agent's label-appropriate index array (mal_idx_train/
    mal_idx_test/benign_idx_train/benign_idx_test) that a reference
    classifier already classifies correctly, so every episode this agent
    runs starts from a genuinely "was correct" row instead of a uniformly-
    random one -- see main()'s call sites for which reference classifier
    each agent_type uses (C1 for all four; train-side agents' "current"
    classifier already IS C1 at their call site, so no separate snapshot is
    needed there).
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
# TEMPORARY DIAGNOSTIC (pocket-targeting investigation), test-accuracy
# breakdown -- answers, per task, the 6 questions: (1) how many rows are in
# this task's test split (own reserved + recycled successes appended in),
# and how many of those are actually-perturbed rows counted in the reported
# accuracy; (2) of the perturbed rows, how many the classifier STILL gets
# right after adaptation, per class; (3) of the clean rows, how many it gets
# right after adaptation; (4) how many perturbed rows unlearning recovers,
# per class; (5) how many clean rows are right after unlearning; (6) how
# many clean rows unlearning flipped right->wrong or wrong->right (not just
# the net delta). Appended into the SAME pocket_targeting_diagnostic.txt
# log, right after the existing correctness-pattern section for that task.
# ---------------------------------------------------------------------------
def _test_accuracy_snapshot(classifier_wrapper, X_test, y_test, poison_idx_test, poison_idx_test_benign,
                             mal_label, benign_label):
    """
    One-shot accuracy snapshot of a task's (possibly recycled-extended) test
    split against whatever classifier state `classifier_wrapper` currently
    holds -- called once right after PRE-unlearning eval (classifier == C2)
    and again right after POST-unlearning eval (classifier == C3), on the
    SAME X_test/y_test/poison_idx_test/poison_idx_test_benign (unlearning
    changes only the classifier's weights, never the test data or which
    rows are flagged perturbed), so the two snapshots' clean_pred arrays are
    directly comparable row-for-row -- see write_test_accuracy_breakdown's
    Q6 computation.

    Returns a dict: n_test_total, n_perturbed_mal/n_perturbed_ben (== the
    count of malicious/benign rows counted in the reported accuracy, own +
    persisted recycled), mal_correct/ben_correct (of those, how many the
    classifier still gets right), clean_total/clean_correct, and
    clean_idx/clean_true/clean_pred (raw arrays, kept for the row-level
    diff against a later snapshot).
    """
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
    """
    2 (predicted: benign, malicious) x 4 (category: malicious, benign,
    perturbed_malicious, perturbed_benign) breakdown, built entirely from a
    _test_accuracy_snapshot dict -- unlike evaluate_classifier's plain 2x2
    confusion matrix, this keeps perturbed rows in their own columns
    instead of folding them into the ordinary malicious/benign columns, so
    a reader can see at a glance where errors concentrate (clean vs
    perturbed). "malicious"/"benign" columns are this task's CLEAN rows
    only (the perturbed ones are split out into their own columns).
    Returns (columns, row_benign, row_malicious).
    """
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


def write_test_accuracy_breakdown(log_path, task_id, pre_stats, pre_metrics, mal_label, benign_label,
                                   post_stats=None, post_metrics=None):
    """
    Appends a human-readable Q1-6 test-accuracy breakdown for this task to
    `log_path` (the same pocket_targeting_diagnostic.txt file), right after
    the existing per-sample correctness-pattern section for this task, plus
    (right after each phase's items) a 2x4 confusion matrix and the
    balanced/pooled/mean accuracy numbers for that phase.
    pre_stats/post_stats come from _test_accuracy_snapshot, captured right
    after this task's PRE-unlearning and POST-unlearning (if it ran)
    evaluations respectively; pre_metrics/post_metrics are dicts with
    balanced_accuracy/pooled_accuracy/mean_accuracy for that same
    checkpoint. post_stats/post_metrics are None when unlearning didn't run
    this task (task 0, or an empty forget set).
    """
    n_pert = pre_stats["n_perturbed_mal"] + pre_stats["n_perturbed_ben"]
    with open(log_path, "a") as f:
        f.write(f"[Test accuracy breakdown] Task {task_id}\n")
        f.write(f"  1. Reserved for testing (incl. recycled successes persisted this task): "
                f"{pre_stats['n_test_total']} total\n")
        f.write(f"     Perturbed & actually used (counted in the reported accuracy above): "
                f"{n_pert} ({pre_stats['n_perturbed_mal']} malicious + {pre_stats['n_perturbed_ben']} benign)\n")
        f.write(f"  2. Perturbed samples still classified CORRECTLY after adaptation (pre-unlearning): "
                f"malicious {_ratio_str(pre_stats['mal_correct'], pre_stats['n_perturbed_mal'])}, "
                f"benign {_ratio_str(pre_stats['ben_correct'], pre_stats['n_perturbed_ben'])}\n")
        f.write(f"  3. Clean samples correctly classified after adaptation (pre-unlearning): "
                f"{_ratio_str(pre_stats['clean_correct'], pre_stats['clean_total'])}\n")
        f.write("  [Confusion matrix -- pre-unlearning] (rows = predicted, columns = true category)\n")
        f.write(_format_confusion_matrix(*_confusion_matrix_4x2(pre_stats, mal_label, benign_label)) + "\n")
        f.write(_format_metrics_line(pre_metrics) + "\n")
        if post_stats is None:
            f.write("  4-6. Unlearning did not run this task -- no post-unlearning comparison.\n\n")
            return
        f.write(f"  4. Perturbed samples RECOVERED by unlearning (now correct, were wrong pre-unlearning): "
                f"malicious {_ratio_str(post_stats['mal_correct'], pre_stats['n_perturbed_mal'])}, "
                f"benign {_ratio_str(post_stats['ben_correct'], pre_stats['n_perturbed_ben'])}\n")
        f.write(f"  5. Clean samples correctly classified after unlearning: "
                f"{_ratio_str(post_stats['clean_correct'], post_stats['clean_total'])}\n")
        gained = int(np.sum((pre_stats["clean_pred"] != pre_stats["clean_true"]) &
                             (post_stats["clean_pred"] == post_stats["clean_true"])))
        lost = int(np.sum((pre_stats["clean_pred"] == pre_stats["clean_true"]) &
                           (post_stats["clean_pred"] != post_stats["clean_true"])))
        net = post_stats["clean_correct"] - pre_stats["clean_correct"]
        f.write(f"  6. Clean samples changed by unlearning: {gained} recovered (wrong->correct), "
                f"{lost} lost (correct->wrong), net {net:+d}\n")
        f.write("  [Confusion matrix -- post-unlearning] (rows = predicted, columns = true category)\n")
        f.write(_format_confusion_matrix(*_confusion_matrix_4x2(post_stats, mal_label, benign_label)) + "\n")
        f.write(_format_metrics_line(post_metrics) + "\n\n")


def write_pocket_targeting_diagnostic(log_path, task_id, entries, unlearning_ran,
                                       benign_label, mal_label):
    """
    Appends one human-readable section to `log_path` for this task's
    test-side perturbed samples (both malicious_perturbed and
    benign_perturbed). `entries` is a list of dicts, each:
        {"gid": int, "group": "malicious_perturbed"|"benign_perturbed",
         "true_label": int, "c1_pred": int, "c2_pred": int, "c3_pred": int,
         "source": str, optional}
    `source`, when present (e.g. "recycled_from_task_3"), marks a row drawn
    from an EARLIER task's own test split (recycled into this task's
    perturbable pool -- see the TEST-SIDE blocks in main()) rather than this
    task's own test rows; absent/None means "this task's own test row". Its
    `gid` is still the row's ORIGINAL global id from whichever task it
    actually came from -- never reassigned.
    C1/C2/C3 are as defined in the log's header (see main()): C1 = classifier
    at the end of the previous task, C2 = after this task's CL training
    (before unlearning), C3 = after this task's unlearning step (or the same
    model as C2, unchanged, if unlearning did not run this task).

    This function only reads/formats -- it never decides what "success"
    means, since that's the judgment call this diagnostic exists to inform.
    It does group entries by (C1,C2,C3) correctness pattern in the summary,
    since eyeballing that distribution directly answers:
      - "correct -> WRONG -> correct" count: pocket-targeting AND unlearning
        both worked as intended on these samples.
      - "correct -> WRONG -> WRONG" count: pocket-targeting found a real
        pocket, but unlearning failed to recover it.
      - "correct -> correct -> *" count: this task's poisoned training didn't
        actually move the boundary through this exact point -- pocket-
        targeting missed (unlearning has nothing to recover here regardless).
      - any pattern starting "WRONG -> ...": C1 itself already misclassified
        this sample -- a pre-existing error unrelated to this task's
        poisoning, not informative for this diagnosis.
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
        c3_tag, c3_ok = _fmt(e["c3_pred"], true_label)
        pattern = f"{'correct' if c1_ok else 'WRONG'} -> {'correct' if c2_ok else 'WRONG'} -> {'correct' if c3_ok else 'WRONG'}"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        pred_name = lambda p: "malicious" if p == mal_label else "benign"
        source = e.get("source")
        source_tag = f"  [{source}]" if source else ""
        lines_by_group[e["group"]].append(
            f"    gid={e['gid']:<8} true={true_name:<9} "
            f"C1={c1_tag}(pred={pred_name(e['c1_pred'])})  "
            f"C2={c2_tag}(pred={pred_name(e['c2_pred'])})  "
            f"C3={c3_tag}(pred={pred_name(e['c3_pred'])})"
            f"{source_tag}"
        )

    n_mal = len(lines_by_group["malicious_perturbed"])
    n_ben = len(lines_by_group["benign_perturbed"])
    n_recycled = sum(1 for e in entries if e.get("source"))

    with open(log_path, "a") as f:
        f.write(f"--- Task {task_id} (unlearning ran this task: {unlearning_ran}) " + "-" * 20 + "\n")
        f.write(f"{len(entries)} total perturbed test samples "
                f"({n_mal} malicious_perturbed, {n_ben} benign_perturbed"
                f"{f', {n_recycled} recycled from an earlier task' if n_recycled else ''})\n\n")
        f.write("Summary by C1 -> C2 -> C3 correctness pattern:\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda kv: -kv[1]):
            note = ""
            if pattern == "correct -> WRONG -> correct":
                note = "  <- pocket-targeting AND unlearning working as intended"
            elif pattern == "correct -> WRONG -> WRONG":
                note = "  <- pocket found, but unlearning failed to recover it"
            elif pattern.startswith("correct -> correct"):
                note = "  <- boundary didn't move here -- pocket-targeting missed"
            elif pattern.startswith("WRONG"):
                note = "  <- C1 was already wrong here (pre-existing error, not this task's doing)"
            f.write(f"  {pattern:<30}: {count:>4}{note}\n")
        f.write("\n")

        if lines_by_group["malicious_perturbed"]:
            f.write(f"  Malicious-perturbed samples ({n_mal}):\n")
            f.write("\n".join(lines_by_group["malicious_perturbed"]) + "\n\n")
        if lines_by_group["benign_perturbed"]:
            f.write(f"  Benign-perturbed samples ({n_ben}):\n")
            f.write("\n".join(lines_by_group["benign_perturbed"]) + "\n\n")


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
    3. REWORK (revisited Q11 answer): forget-set selection is hard-classified
       again, at 100% -- every eval row (see eval_idx below) whose HARD
       .predict() output is LABEL_PERTURBED becomes forget_idx_malicious,
       and every eval row predicted LABEL_BENIGN_PERTURBED becomes
       forget_idx_benign. No probability ranking, no per-class quota --
       forget-set size is whatever the classifier's raw positive rate on
       each class happens to be this task, which can be smaller OR larger
       than the old fixed-50%-per-class rule depending on how confidently
       (and how often) it fires. perturbed_class_metrics (see return value)
       reports precision/recall for LABEL_PERTURBED/LABEL_BENIGN_PERTURBED
       against oracle ground truth on these same eval rows, so a task where
       the forget set stays empty or misses the true pockets shows up
       directly as low recall there, rather than being invisible.

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
    held-out eval rows), n_eval (how many rows the classifier was scored
    against), and perturbed_class_metrics (precision/recall/tp/fp/fn for
    LABEL_PERTURBED and LABEL_BENIGN_PERTURBED specifically, derived from
    confusion_matrix -- logging aid for seeing whether a task's forget set
    is small because there was nothing to catch or because the classifier
    missed it).
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

    # REWORK (revisited Q11 answer): hard-classification selection, 100% --
    # see this function's docstring step 3. A malicious eval row enters the
    # forget set iff the classifier's own .predict() called it
    # LABEL_PERTURBED (no ranking, no quota); mirror for LABEL_BENIGN_PERTURBED
    # on the benign side.
    mal_eval_local = np.where(true_binary == mal_label)[0]
    ben_eval_local = np.where(true_binary == benign_label)[0]

    forget_local_mal = mal_eval_local[eval_pred[mal_eval_local] == LABEL_PERTURBED]
    forget_local_ben = ben_eval_local[eval_pred[ben_eval_local] == LABEL_BENIGN_PERTURBED]

    forget_idx_malicious = np.asarray(sorted(int(eval_idx[i]) for i in forget_local_mal), dtype=np.int64)
    forget_idx_benign = np.asarray(sorted(int(eval_idx[i]) for i in forget_local_ben), dtype=np.int64)
    forget_idx = np.union1d(forget_idx_malicious, forget_idx_benign)
    retain_idx = np.setdiff1d(np.arange(n_samples, dtype=np.int64), forget_idx)

    # LOGGING (added to diagnose the forget set's actual behavior against
    # ground truth, now that its size is no longer pinned to a fixed
    # fraction): precision/recall for the two "perturbed" classes specifically,
    # read off cm -- a task where recall is low means the true pockets are
    # being missed (never entering the forget set at all, regardless of this
    # selection rule), not just that the forget set looks small.
    def _class_precision_recall(class_label):
        ci = PERTURBATION_CLASSIFIER_CLASSES.index(class_label)
        tp = int(cm[ci, ci])
        fn = int(cm[ci, :].sum()) - tp
        fp = int(cm[:, ci].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

    perturbed_class_metrics = {
        "malicious": _class_precision_recall(LABEL_PERTURBED),
        "benign": _class_precision_recall(LABEL_BENIGN_PERTURBED),
    }

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
        "perturbed_class_metrics": perturbed_class_metrics,
        "n_eval": int(len(eval_idx)),
    }


def retrain_classifier_from_scratch(feature_dim, retain_idx, forget_idx, Xtr, ytr, omega, device,
                                     replay_buffer=None, epochs=None, lr=None, batch_size=None):
    """
    DIAGNOSTIC (retrain-vs-unlearn comparison, FULL_RETRAIN_DIAGNOSTIC_TASKS
    only): the "exact unlearning" gold standard -- a freshly-initialized
    classifier (fresh weights, fresh optimizer) trained on the retain set
    PLUS the current replay buffer, so it has structurally never seen the
    forget set at all, unlike unlearn_teacher_guided's approximate SI/KD-
    anchored corrective approach. The replay buffer is included (not just
    this task's retain set) so the comparison is apples-to-apples:
    unlearn_teacher_guided also trains against retain + replay (+ KD) --
    omitting replay here would just catastrophically forget every prior
    task and confound "does full retrain forget the poisoned data better"
    with "this model saw none of the other 7-8 tasks." The buffer is
    already scrubbed of forget-pool matches by _clean_buffer, so including
    it can't leak anything this diagnostic is meant to forget. Swapped in
    for specific tasks only; CL training (and the SI omega/p_old_task
    trackers) for every other task is unaffected -- this function doesn't
    touch omega/p_old_task itself, it only produces a new model's weights
    for the caller to load into the existing model in place.

    Returns (new_model, diag) where diag deliberately mirrors unlearn_
    teacher_guided's return keys (omega_l2_norm/n_steps/raw_forget_loss_mean/
    raw_retain_loss_mean/raw_si_loss_mean) so the existing "[Unlearn diag]"
    print line and unlearning_metrics plumbing need no special-casing:
    omega_l2_norm reports the CURRENT omega (unchanged by this path -- CL
    continues normally next task), raw_si_loss_mean is always 0.0 (no SI
    term is used here), and raw_forget_loss_mean is the NEW model's
    resulting forget-set CE loss (informative even though never optimized
    against, unlike the other two which are true training-loss averages).
    n_replay (extra key) records how many buffer rows were folded in.
    """
    epochs = RETRAIN_FULL_EPOCHS if epochs is None else epochs
    lr = RETRAIN_FULL_LR if lr is None else lr
    batch_size = BATCH_SIZE if batch_size is None else batch_size

    new_model = ClassifierNN(feature_dim, 2).to(device)
    optimizer = optim.Adam(new_model.parameters(), lr=lr)
    retain_idx_t = torch.tensor(retain_idx, dtype=torch.long)

    train_X = [Xtr[retain_idx_t]]
    train_y = [ytr[retain_idx_t]]
    n_replay = 0
    if replay_buffer:
        train_X.append(torch.stack([entry[0] for entry in replay_buffer]))
        train_y.append(torch.stack([entry[1] for entry in replay_buffer]))
        n_replay = len(replay_buffer)
    train_X = torch.cat(train_X, dim=0)
    train_y = torch.cat(train_y, dim=0)

    drop_last = (len(train_X) % batch_size == 1)
    loader = data.DataLoader(
        data.TensorDataset(train_X, train_y), batch_size=batch_size,
        shuffle=True, drop_last=drop_last,
    )

    new_model.train()
    raw_retain_losses = []
    for _ in range(epochs):
        for v, l in loader:
            v, l = v.to(device), l.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(new_model(v), l)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during diagnostic full retrain - training diverged")
            loss.backward()
            optimizer.step()
            raw_retain_losses.append(loss.item())

    new_model.eval()
    raw_forget_loss_mean = None
    if len(forget_idx) > 0:
        forget_idx_t = torch.tensor(forget_idx, dtype=torch.long)
        with torch.no_grad():
            f_v, f_l = Xtr[forget_idx_t].to(device), ytr[forget_idx_t].to(device)
            raw_forget_loss_mean = float(F.cross_entropy(new_model(f_v), f_l).item())

    omega_l2_norm = float(torch.sqrt(sum((v ** 2).sum() for v in omega.values())).item())

    diag = {
        "method": "full_retrain_on_retain_set_plus_replay",
        "omega_l2_norm": omega_l2_norm,
        "n_steps": len(raw_retain_losses),
        "n_replay": n_replay,
        "raw_forget_loss_mean": raw_forget_loss_mean,
        "raw_retain_loss_mean": float(np.mean(raw_retain_losses)) if raw_retain_losses else None,
        "raw_si_loss_mean": 0.0,
    }
    return new_model, diag


def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, omega, p_old_task,
                           si_c, epochs, lr, alpha, device):
    """
    REWORK (uncertainty-margin pipeline): this becomes a corrective/targeted
    cross-entropy toward ground truth rather than an entropy-maximizing
    forget loss -- forget-set samples are pushed to be CONFIDENTLY
    classified as their TRUE (pre-perturbation) label, i.e. the class
    OPPOSITE whatever the poisoned model currently predicts for them,
    instead of toward uniform/maximum uncertainty. ytr (and so f_l in the
    training loop below) is always the true label even for perturbed rows --
    only X changes on substitution, never y -- so this needs no new label
    source, just a different loss against the SAME f_l the old KL-to-uniform
    version received but never used. Anchored by ground-truth CE + KD on the
    retain set +
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

            # --- 1. CORRECTIVE CROSS-ENTROPY (Forget) -- this becomes a
            # corrective/targeted cross-entropy toward ground truth rather
            # than an entropy-maximizing forget loss. f_l is the sample's
            # TRUE (pre-perturbation) label -- pushing logits toward it
            # directly targets confident correction, not uncertainty. ---
            logits = model(f_v)
            forget_loss = F.cross_entropy(logits, f_l)

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
    """Identical to madar_cl_pipeline.py's plot_task_metrics, title relabeled,
    plus (REWORK, task_metrics.png follow-up): the main panel now plots BOTH
    the PRE-unlearning snapshot (pooled_eval/mean_per_task_balanced_accuracy,
    computed at step 2, before this task's forget step) and the POST-unlearning
    one (post_unlearn_pooled_eval/post_unlearn_mean_per_task_balanced_accuracy,
    computed right after unlearning finishes) -- previously only the PRE
    numbers were plotted here (despite the title's own disclaimer), so the
    plot never actually showed the classifier's real, deployed post-unlearning
    performance. Tasks where unlearning didn't run this checkpoint (task 0, or
    any task whose forget set ended up empty) fall back to the PRE value for
    the POST line, since PRE *is* the final state there -- there's no separate
    post-unlearning snapshot to show. Also adds a third panel (REWORK,
    pocket-targeted test poisoning follow-up) showing THIS task's own
    pocket-targeted test batch accuracy at decision boundary N (pre-unlearning,
    perturbed_test_eval) vs. decision boundary J (post-unlearning,
    post_unlearn_perturbed_test_eval) -- the direct before/after comparison on
    the identical rows, rather than inferring recovery from the whole-test-set
    post_unlearn_this_task_eval number."""
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc_pre = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc_pre = [r["mean_per_task_balanced_accuracy"] for r in results]

    def _post_or_fallback(r, key, fallback):
        u = r.get("unlearning") or {}
        v = u.get(key)
        if isinstance(v, dict):
            v = v.get("balanced_accuracy")
        return fallback if v is None else v

    pooled_bal_acc_post = [
        _post_or_fallback(r, "post_unlearn_pooled_eval", pre)
        for r, pre in zip(results, pooled_bal_acc_pre)
    ]
    mean_per_task_bal_acc_post = [
        _post_or_fallback(r, "post_unlearn_mean_per_task_balanced_accuracy", pre)
        for r, pre in zip(results, mean_per_task_bal_acc_pre)
    ]
    # BUGFIX (C1-correct-only sampling prototype): red_agent can be a non-empty
    # dict missing train_evasion_rate/test_evasion_rate specifically -- e.g.
    # the malicious train-side agent was skipped (empty C1-correct pool) but
    # the benign train-side agent ran and populated its own keys into the
    # SAME dict, or vice versa. .get() instead of direct indexing so a
    # partially-populated red_agent doesn't crash the plot.
    train_evasion = [r["red_agent"].get("train_evasion_rate") if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"].get("test_evasion_rate") if r["red_agent"] else None for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc_pre, marker="o", linestyle="--", color="tab:blue", alpha=0.5,
              label="pooled balanced accuracy (pre-unlearn)")
    ax1.plot(task_ids, pooled_bal_acc_post, marker="o", color="tab:blue",
              label="pooled balanced accuracy (post-unlearn)")
    ax1.plot(task_ids, mean_per_task_bal_acc_pre, marker="s", linestyle="--", color="tab:orange", alpha=0.5,
              label="mean per-task balanced accuracy (pre-unlearn)")
    ax1.plot(task_ids, mean_per_task_bal_acc_post, marker="s", color="tab:orange",
              label="mean per-task balanced accuracy (post-unlearn)")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("MADAR+Unlearning: classifier accuracy per task boundary (PRE- vs. POST-unlearning)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # BUGFIX (C1-correct-only sampling prototype): train_evasion/test_evasion
    # can now be None at DIFFERENT task indices from each other (malicious
    # and benign/train and test agents can each be independently skipped) --
    # a single xs shared between both series would misalign them, or crash
    # outright if the two None-patterns differ in count. Filter each series
    # against its own None pattern.
    xs_train = [t for t, v in zip(task_ids, train_evasion) if v is not None]
    ys_train = [v for v in train_evasion if v is not None]
    xs_test = [t for t, v in zip(task_ids, test_evasion) if v is not None]
    ys_test = [v for v in test_evasion if v is not None]
    ax2.plot(xs_train, ys_train, marker="o", color="tab:red", label="evasion rate (train samples)")
    ax2.plot(xs_test, ys_test, marker="s", color="tab:orange", label="evasion rate (held-out test samples)")
    ax2.set_ylabel("Evasion rate")
    ax2.set_title("Red agent evasion rate per task boundary (task 0 has no red agent)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # BUGFIX: perturbed_test_eval/post_unlearn_perturbed_test_eval are built
    # with int task-id keys during a live run, but a dict loaded back from
    # JSON (e.g. via regenerate_task_plots.py) always has STRING keys --
    # dict.get(1) never matches key "1". Try both so this panel works
    # whether `results` came straight from main()'s in-memory list or from
    # a reloaded results JSON.
    pocket_task_ids, pocket_pre, pocket_post = [], [], []
    for r in results:
        t = r["task_id"]
        pte = r.get("perturbed_test_eval", {})
        pre = pte.get(t, pte.get(str(t)))
        u = r.get("unlearning") or {}
        post_pte = u.get("post_unlearn_perturbed_test_eval", {})
        post = post_pte.get(t, post_pte.get(str(t)))
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


def compute_pocket_mask(proba_before, proba_after, threshold=0.5):
    """
    PROTOTYPE (adaptation-split pocket highlighting): boolean grid-cell mask
    over the SAME fixed-PCA grid shape as _decision_grid's output, True
    wherever the predicted class (thresholded at `threshold`, matching the
    plot's own 0.5 contour line) differs between two checkpoints' probability
    grids -- i.e. the region the decision surface moved through between them.
    Both grids must come from the same pca_mean/pca_components/pca_extent/
    resolution (always true here -- both are always the run-wide fixed basis).
    """
    return (proba_before >= threshold) != (proba_after >= threshold)


def _draw_decision_boundary(ax, xx, yy, proba, Z_train, category_train, Z_test, category_test,
                             highlight_mask=None):
    """Draws one panel from PRECOMPUTED grid/scatter data onto `ax`. Shared
    by plot_decision_boundary (computes then draws, saves its own file) and
    plot_decision_boundary_grid (redraws every checkpoint's cached data into
    one summary figure). Returns per-agent perturbed counts (for the
    caller's legend).

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


def _compute_global_pca_basis(tasks, to_tensor, pad=0.5):
    """
    GLOBAL (run-wide) PCA basis for plot_decision_boundary -- a 2D
    projection fit once on ALL tasks' pooled features (scaled via the
    caller's `to_tensor`, which closes over whatever `scaler` is current at
    call time) and reused unchanged for every task/checkpoint's boundary
    plot for the rest of the run. Pure function of `tasks` + the scaler
    `to_tensor` uses -- no dependency on training state -- so a resumed run
    (see main()'s --resume_from) can call this again with its restored
    scaler and get back the IDENTICAL basis the original run would have
    used at this point, rather than needing to persist it in the checkpoint.
    """
    all_X_scaled = to_tensor(
        np.concatenate([np.clip(tk["features"].astype(np.float32), 0.0, 1.0) for tk in tasks], axis=0)
    ).numpy()
    pca_mean = all_X_scaled.mean(axis=0)
    _, _, pca_Vt = np.linalg.svd(all_X_scaled - pca_mean, full_matrices=False)
    pca_components = pca_Vt[:2]
    Z_all = (all_X_scaled - pca_mean) @ pca_components.T
    pca_extent = (
        float(Z_all[:, 0].min() - pad), float(Z_all[:, 0].max() + pad),
        float(Z_all[:, 1].min() - pad), float(Z_all[:, 1].max() + pad),
    )
    return pca_mean, pca_components, pca_extent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log_name", type=str, default="madar_unlearn_cl_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    # NEW (full-resume support): path to a classifier_checkpoint.pt saved by
    # a PREVIOUS run of this same script. When given, the loop skips
    # straight to the task AFTER that checkpoint's task_id, having restored
    # everything the skipped tasks would otherwise have produced -- model/
    # scaler, SI omega/p_old_task, the replay buffer, the contrastive bank's
    # prototypes, the cross-task historical clean-sample pools, every prior
    # task's test split/results entry -- so the resumed tasks run exactly as
    # if the earlier tasks had just finished in THIS process. Primarily for
    # the FULL_RETRAIN_DIAGNOSTIC_TASKS comparison (tasks 8/9): re-run just
    # those two tasks from a task-7 checkpoint instead of the whole run.
    ap.add_argument("--resume_from", type=str, default=None,
                     help="Path to a classifier_checkpoint.pt from a prior run; resumes "
                          "right after that checkpoint's task_id.")
    args = ap.parse_args()

    resume_checkpoint = None
    resume_task_id = None
    if args.resume_from:
        print(f"Loading resume checkpoint from {args.resume_from} ...")
        # weights_only=False: this checkpoint carries plain Python objects
        # (StandardScaler, the contrastive bank's numpy state, task_test_
        # splits, etc.), not just tensors -- torch's newer weights_only=True
        # default would refuse to unpickle those. Trusted input: this is a
        # local research checkpoint written by this same script, never
        # user-supplied/untrusted data.
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        resume_task_id = resume_checkpoint["task_id"]
        if resume_task_id >= NUM_TASKS - 1:
            raise ValueError(
                f"Checkpoint is already at task {resume_task_id}, the last task "
                f"(NUM_TASKS={NUM_TASKS}) -- there is nothing left to resume."
            )
        print(f"Resuming after task {resume_task_id} -- next task to run is {resume_task_id + 1}.")

    # Seed must match the ORIGINAL run's seed for a resumed run's own
    # train_test_split calls (random_state=args.seed) to land on the same
    # rows the original run would have produced for the not-yet-processed
    # tasks -- default to the checkpoint's seed when resuming and none was
    # given explicitly, otherwise fall back to this script's historical
    # default (42).
    if args.seed is None:
        args.seed = resume_checkpoint["seed"] if resume_checkpoint is not None else 42
    elif resume_checkpoint is not None and args.seed != resume_checkpoint["seed"]:
        print(f"WARNING: --seed {args.seed} differs from the checkpoint's original seed "
              f"{resume_checkpoint['seed']} -- task splits from here on will NOT match "
              f"what the original run would have produced.")

    global SEED
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join(RUNS_BASE_DIR, "madar_unlearning", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # NEW: single classifier checkpoint, stored alongside the logs and
    # OVERWRITTEN at the end of every task -- always holds only the most
    # recently completed task's classifier, not one file per task.
    classifier_checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    # TEMPORARY DIAGNOSTIC (pocket-targeting investigation): separate,
    # human-readable log checking whether red_test_pert_agent/
    # red_test_benign_pert_agent are actually landing where the boundary
    # moves, and whether unlearning recovers accuracy there. See
    # write_pocket_targeting_diagnostic()'s docstring for the full field
    # definitions. Not part of the pipeline's normal metrics -- delete this
    # file/call sites once the investigation concludes.
    pocket_diag_log_path = os.path.join(out_dir, "logs", "pocket_targeting_diagnostic.txt")
    with open(pocket_diag_log_path, "w") as f:
        f.write(
            "POCKET-TARGETING DIAGNOSTIC LOG\n"
            "================================\n"
            "Checks whether red_test_pert_agent/red_test_benign_pert_agent are landing\n"
            "where the decision boundary actually moves, and whether unlearning recovers\n"
            "accuracy on those exact points. One section per task (tasks > 0 only -- task 0\n"
            "has no red agents/perturbed test samples).\n\n"
            "Three classifier snapshots, all scored on the SAME perturbed test sample:\n"
            "  C1 = classifier at the END of the PREVIOUS task -- this is the exact\n"
            "       classifier state red_test_pert_agent/red_test_benign_pert_agent\n"
            "       perturbed this sample against.\n"
            "  C2 = classifier AFTER this task's CL training (ER+KD+SI), BEFORE this\n"
            "       task's unlearning step. If pocket-targeting worked, this is where a\n"
            "       sample that C1 got right should flip to wrong -- i.e. this task's own\n"
            "       poisoned training moved the boundary through this exact point.\n"
            "  C3 = classifier AFTER this task's unlearning step. If unlearning worked,\n"
            "       a sample that flipped wrong at C2 should flip back to correct here.\n"
            "       If unlearning did NOT run this task (no forget set found), C3 is the\n"
            "       same model as C2 -- logged as 'unlearning ran: False' in the section\n"
            "       header so a C2==C3 result isn't mistaken for a failed recovery.\n"
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

    def to_tensor(X_raw):
        X_scaled = np.clip(scaler.transform(X_raw.astype(np.float32)), -FEATURE_CLIP, FEATURE_CLIP)
        return torch.tensor(X_scaled, dtype=torch.float32)

    if resume_checkpoint is not None:
        # NEW (full-resume support): restore every carried-across-tasks piece
        # of state a normal run would have accumulated through the end of
        # task resume_task_id, so task resume_task_id+1 onward runs exactly
        # as if this process had been here all along. classifier_wrapper
        # wraps `model` BY REFERENCE (same pattern the t==0 branch below
        # uses) so every later in-place update (CL training, unlearning,
        # this diagnostic's load_state_dict) is automatically visible
        # through it without reconstruction.
        scaler = resume_checkpoint["scaler"]
        model = ClassifierNN(feature_dim, 2).to(DEVICE)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        classifier_wrapper = TorchIDSWrapper(model, scaler, DEVICE)
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        omega = resume_checkpoint["omega"]
        p_old_task = resume_checkpoint["p_old_task"]
        W = {k: torch.zeros_like(v) for k, v in omega.items()}  # always reset per-task anyway

        replay_buffer.clear()
        replay_buffer.extend(resume_checkpoint["replay_buffer"])
        bank.protos = resume_checkpoint["bank_protos"]
        bank.counts = resume_checkpoint["bank_counts"]
        bank.episode_embs = resume_checkpoint["bank_episode_embs"]
        bank.task_order = resume_checkpoint["bank_task_order"]

        task_test_splits = resume_checkpoint["task_test_splits"]
        task_test_gids = resume_checkpoint["task_test_gids"]
        task_poisoned_test_gids = resume_checkpoint["task_poisoned_test_gids"]
        results = resume_checkpoint["results"]
        warnings_log = resume_checkpoint["warnings_log"]
        boundary_panels = resume_checkpoint["boundary_panels"]
        historical_clean_mal_pool = resume_checkpoint["historical_clean_mal_pool"]
        historical_clean_benign_pool = resume_checkpoint["historical_clean_benign_pool"]

        # Global PCA basis (see the t==0 branch below for why this depends
        # on `scaler`) -- recomputed identically here since it's a pure
        # function of tasks + the (now-restored) scaler, not of training
        # state, so there's nothing to gain from trying to persist it.
        pca_mean, pca_components, pca_extent = _compute_global_pca_basis(tasks, to_tensor)
    else:
        scaler = None
        model = None
        classifier_wrapper = None
        teacher_model = None
        W = omega = p_old_task = None

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
        # NEW (full-resume support): everything through resume_task_id was
        # already processed by the run that produced resume_checkpoint --
        # its effects are already restored above, so just skip straight to
        # the next task.
        if resume_checkpoint is not None and t <= resume_task_id:
            continue
        # REWORK (uncertainty-margin pipeline): buffer composition logged at
        # every stage of this task's processing -- "start_of_task" here
        # captures whatever the PREVIOUS task's cleaning step left behind
        # (empty for task 0), i.e. exactly the replay signal this task's own
        # train_cl_er call is about to use. See the other two checkpoints
        # ("after_fill", "after_cleaning") further down, and where this dict
        # gets attached to the task's results entry.
        buffer_snapshots = {"start_of_task": buffer_composition_summary()}

        X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)  # data is documented [0,1]-scaled already
        y = task["labels"].astype(np.int64)
        gid = task_offsets[t] + np.arange(len(y), dtype=np.int64)  # this task's pool-local id -> global id

        X_train, X_test, y_train, y_test, gid_train, gid_test = train_test_split(
            X, y, gid, test_size=TASK_TEST_FRAC, random_state=args.seed, stratify=y
        )
        task_test_splits[t] = (X_test, y_test)
        task_test_gids[t] = gid_test
        n_test_reserved = len(X_test)  # this task's own row count, BEFORE any
                                        # recycled-success rows get appended below --
                                        # used by the test-accuracy-breakdown logging

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
        # NEW (variant-augmentation branch): safe empty defaults so the
        # append step near `category = np.full(...)` below can always
        # reference these, even for t==0 or a task with no benign_idx_train.
        _empty_variants = {"X": np.empty((0, X.shape[1]), dtype=np.float32),
                            "internal_gids": np.array([], dtype=np.int64),
                            "display_gids": [], "n_variants_by_original": {}}
        mal_variants = dict(_empty_variants)
        ben_variants = dict(_empty_variants)
        n_poisoned_test_benign = 0
        poison_idx_test_benign = np.array([], dtype=int)
        poisoned_test_sample_ids_benign = []
        # REWORK (recycled-pocket test pool, now persisted): rows drawn from
        # task (t-1)'s OWN test split that got poisoned this task -- see the
        # TEST-SIDE blocks below for how these are selected. Still NEVER
        # written into task_test_splits[t-1] itself (that split stays exactly
        # as task (t-1) left it, so every later task's re-evaluation of task
        # (t-1) is unaffected) -- but their perturbed feature rows/gids ARE now
        # appended as new rows into THIS task's own task_test_splits[t]/
        # task_test_gids[t] (see the TEST-SIDE blocks below), so they count
        # toward this task's own n_test/per_task_eval/pooled_eval/
        # perturbed_test_eval, same as an own-task success. These task-local
        # copies are kept around only for the pocket-diagnostic capture below
        # (which needs the recycled count to tag entries, even though the
        # actual C1/C2/C3 predictions are now read off task_test_splits[t]
        # directly). Initialized here so every downstream reference is safe
        # even when nothing gets recycled this task.
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
                # is still called and logged purely as an FYI diagnostic (so a
                # collapse like task 6's 0/162 stays visible), separate from
                # this selection.
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
                # docstring. env_train/agent_train are the SAME pair evaluate_agent_
                # on_batch just ran -- variants continue perturbing from each
                # already-evaded landing point using the identical trained policy.
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
                # below exactly like malicious_perturbed does. Uses the SAME
                # RED_TRAIN_TARGET_MARGIN_CONFIDENCE margin bias as the
                # malicious side (see train_red_agent_for_task's docstring).
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
        # benign variant rows as NEW trailing rows into X_train_for_classifier/
        # y_train/gid_train (they have no "slot" to overwrite -- unlike an
        # original evasion, they're not replacing any existing row), and
        # extend poison_idx/poison_idx_benign to include their new positions,
        # so category/build_perturbation_classifier_forget_set/buffer fill/
        # poisoned_sample_ids below all pick them up automatically -- same
        # append pattern used for recycled test-pocket successes.
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
        # NEW (variant-augmentation branch): _variant_gid_display translates a
        # variant's internal gid to "<original>.<level>"; an ordinary gid
        # passes through unchanged as a plain int.
        poisoned_sample_ids = [_variant_gid_display(g) for g in gid_train[poison_idx]] if len(poison_idx) > 0 else []
        poisoned_sample_ids_benign = [_variant_gid_display(g) for g in gid_train[poison_idx_benign]] \
            if len(poison_idx_benign) > 0 else []

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
            # boundary plot for the rest of the run. Factored into
            # _compute_global_pca_basis so a resumed run (see
            # --resume_from above) can recompute the identical basis from
            # its restored scaler without duplicating this code.
            pca_mean, pca_components, pca_extent = _compute_global_pca_basis(tasks, to_tensor)

        Xtr = to_tensor(X_train_for_classifier)
        ytr = torch.tensor(y_train, dtype=torch.long)

        unlearning_metrics = None

        # TEMPORARY DIAGNOSTIC (pocket-targeting investigation): populated
        # below (step 2 area) with C1/C2 predictions, then C3 filled in near
        # the end of the loop body. Declared here so it's always defined,
        # even for t==0 (no red agents -> stays empty -> logged as "nothing
        # to check" by write_pocket_targeting_diagnostic).
        pocket_diag_entries = []
        unlearning_ran_this_task = False
        post_test_acc_snapshot = None  # set below only if unlearning actually runs this task
        post_test_acc_metrics = None

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
            # PROTOTYPE (adaptation-split pocket visualization): this task's
            # single combined train_cl_er call is now TWO sequential calls on
            # the same model/optimizer/W/omega/p_old_task -- a "perturbed"
            # pass on just this task's poisoned rows, then a "clean" pass on
            # everything else. Isolates how much EACH data type shifts the
            # boundary. See module-level notes on the design decisions:
            #   - iteration split is PROPORTIONAL to row count (not flat
            #     50/50 or full-budget-each), so a small perturbed pool
            #     doesn't get force-fed thousands of repeats and overfit.
            #   - buffer replay + KD run in BOTH passes (each is a complete
            #     standalone round, same mechanism as today's single pass).
            #   - teacher_model is the SAME pre-task-t snapshot for both
            #     passes, so KD comparisons are apples-to-apples between them.
            #   - SI omega updates ONCE, after BOTH passes (W accumulates
            #     across both calls uninitialized in between; p_old_task
            #     stays anchored at task start throughout) -- identical
            #     semantics to today's single-pass omega update, just now
            #     covering the combined drift of both passes.
            print(f"  Samples: {len(Xtr)} (+{len(replay_buffer)} replay) | "
                  f"MADAR ER+KD+SI, {CL_ITERS} iters (single pass)")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = 0

            # REWORK (pocket-targeted test poisoning): frozen snapshot of the
            # classifier as it stood BEFORE this task's CL training -- captured
            # here, right before train_cl_er mutates `model` in place, so the
            # TEST-SIDE agent block below (which runs AFTER CL training) can
            # compare "did this exact point's predicted class change" between
            # the two. Independent copy of `model` (not just the wrapper), since
            # TorchIDSWrapper holds its model by reference and `model` itself
            # keeps training. Also this task's C1 for the pocket-highlight
            # diffing below.
            pre_train_classifier_wrapper = TorchIDSWrapper(copy.deepcopy(model), scaler, DEVICE)

            # PROTOTYPE (adaptation-split pocket visualization): running
            # per-task highlight state. task_pocket_mask accumulates (unions)
            # every checkpoint's grid-cell flips from here through this task's
            # test/post-unlearn plots -- reset fresh at the start of every
            # task. prev_proba is C1's grid, this task's first "before".
            # X_test_current_scaled_np/category_test_clean_so_far are for the
            # TWO adaptation-phase plots specifically: test-side poisoning
            # hasn't run yet at this point in the loop, so task_test_splits[t]
            # is still this task's unpoisoned test split -- shown as-is
            # (no perturbed markers exist yet, correctly, at this checkpoint).
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
            # already swapped in place in Xtr/ytr by this point (both classes),
            # same as a plain MADAR train_cl_er call always worked before the
            # two-phase split existed.
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

            # SI omega update happens now, after the single training pass --
            # p_old_task is still exactly what it was at task start.
            # p_old_task is intentionally NOT updated yet: the unlearning
            # phase's SI penalty below anchors to start-of-task weights, so
            # any drift during unlearning has to be justified by the forget
            # objective specifically. Matches both references' ordering
            # exactly.
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_post_cl = p.detach().clone()
                    # SI_OMEGA_DECAY (REWORK, SI capacity investigation): decay
                    # applied to the EXISTING accumulated importance before
                    # adding this task's contribution, so omega converges
                    # instead of growing unboundedly over the whole run -- see
                    # that constant's definition for the full rationale.
                    omega[n_key] = SI_OMEGA_DECAY * omega[n_key] + \
                        W[n_key] / ((p_post_cl - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()

            # PROTOTYPE (adaptation-split pocket visualization): logging SI
            # omega's L2 norm right after this task's adaptation, alongside
            # the C1-pool/pocket data already in this diagnostic log.
            omega_l2_after_adapt = float(torch.sqrt(sum((v ** 2).sum() for v in omega.values())).item())
            print(f"    [SI omega] after adaptation: L2 norm = {omega_l2_after_adapt:.4f}")
            with open(pocket_diag_log_path, "a") as f:
                f.write(f"[SI omega] Task {t}: L2 norm after adaptation = {omega_l2_after_adapt:.4f}\n")

        # TEST-SIDE poisoning (REWORK, pocket-targeted test poisoning): runs HERE,
        # after this task's CL training has finished, so red_test_pert_agent/
        # red_test_benign_pert_agent can be scored against the POST-training
        # classifier (classifier_wrapper/model, now mutated by train_cl_er) while
        # comparing to pre_train_classifier_wrapper (frozen snapshot captured
        # right before train_cl_er, above) -- pgd_boundary_search_batch's
        # shift_reference_classifier is exactly this snapshot, re-checked every
        # PGD step: a point the CURRENT model calls the target class but the
        # PRE-training one still calls correctly is the "pocket" THIS TASK's own
        # poisoned training just opened up, not a generic weakness that predates
        # it. (REWORK, gradient-attack branch: this used to ALSO drive a dense
        # RL reward bonus, POCKET_SHIFT_WEIGHT -- removed, since there's no
        # reward to shape in a direct gradient search; only the hard joint
        # success condition remains.) Only runs when this task actually had
        # train-side agents/poisoning above (mirrors the nesting this code used
        # to live inside, before the reorder).
        if t > 0 and len(mal_idx_train) > 0:
            # REWORK (recycled-pocket test pool): the perturbable test pool for
            # BOTH classes below is no longer just this task's own C1-correct test
            # rows -- it's the UNION of those with task (t-1)'s own test-split rows
            # that C1 (the classifier at the end of task (t-1), i.e.
            # pre_train_classifier_wrapper -- exactly what these episodes perturb
            # against either way) also classifies correctly. Fetched once here
            # (not inside the malicious/benign sub-blocks) so both classes see it
            # regardless of which sub-block runs. Deliberately re-derived from
            # task (t-1) ALONE, fresh every task -- POTENTIALLY REVISIT: a
            # longer-range pool that carries forward unused C1-correct samples
            # across more than one task back was considered and explicitly
            # deferred, not ruled out.
            if RECYCLE_TEST_POCKETS:
                X_prev_test, y_prev_test = task_test_splits[t - 1]
                gid_prev_test = task_test_gids[t - 1]
            else:
                # RECYCLE_TEST_POCKETS off: force the "recycled" half of every
                # union below to be empty, so mal_idx_test_ext/benign_idx_test_ext
                # reduce to exactly this task's own candidates, and every
                # recycled_*/n_poisoned_test_recycled_prev* value downstream
                # comes out as 0/empty with no separate code path needed.
                X_prev_test = np.empty((0, X.shape[1]), dtype=X.dtype)
                y_prev_test = np.empty((0,), dtype=y.dtype)
                gid_prev_test = np.empty((0,), dtype=np.int64)

            if POISON_TEST_DATA and len(mal_idx_test) > 0:
                # X_test_ext/y_test_ext concatenate [this task's X_test, task
                # (t-1)'s X_test]; indices >= n_own_test refer to the RECYCLED
                # half. A single c1_correct_pool call over the combined array
                # naturally produces the union filtered by C1, since C1 is the
                # same classifier instance either way.
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
                # TEST_AGGREGATE_FRACTION (40%) of THIS TASK's own malicious test
                # row count (len(mal_idx_test) -- not the combined pool size).
                # Tier-2 episodes can essentially never satisfy the joint C1-
                # correct/C2-wrong success condition below (they didn't start
                # C1-correct), so padding trades a lower per-attempt success rate
                # for more total attempts -- an accepted, deliberate tradeoff.
                #
                # FINETUNE (test-side selection/patience investigation): both
                # tiers are now ranked by most-uncertain-under-C2 (classifier_
                # wrapper, the actual attack target pgd_boundary_search_batch
                # takes gradients against) instead of under-C1. Picking
                # candidates near C1's OWN boundary (the old scheme) was
                # actively self-sabotaging: a diagnostic run showed 99.5% of
                # malicious test attacks were killed by a C1 disagreement
                # within ~2 steps, long before the walk could make any
                # progress toward C2's boundary. Ranking by C2-uncertainty
                # instead targets points that are actually easy to move
                # across C2's decision surface, which is what the attack is
                # trying to do; C1-correctness is still required to ENTER
                # tier 1 (c1_correct_pool above, unchanged -- that's the
                # pocket definition, not the ranking).
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
                    # NEW (gradient-attack branch): test-side malicious agent is now
                    # pgd_boundary_search_batch (see its definition for the full
                    # rationale) instead of an RL rollout -- no env/agent training,
                    # no contrastive bank, no proximity/pocket-shift reward shaping
                    # (those were RL-reward-specific, dropped structurally). Same
                    # return contract as evaluate_agent_on_batch, so everything
                    # below (poison_idx_test_ext construction, red_report fields,
                    # prints) is unchanged.
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

                    # REWORK: no separate quota/subsample step anymore -- the attack
                    # pool above IS the quota (already capped at 40% of this task's
                    # own malicious test count), and evaded_mask_test's "success"
                    # already IS the joint C1-correct/C2-wrong condition (see
                    # NetworkAttackEnv's shift_reference_classifier), so every
                    # episode that satisfies it becomes the poisoned set directly.
                    poison_idx_test_ext = np.where(evaded_mask_test)[0]

                    # Split the ext-array draw back into "this task's own test rows"
                    # (positions < n_own_test) vs. "recycled from task (t-1)"
                    # (positions >= n_own_test).
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
                    # task's own split -- they belong to task (t-1)'s original test
                    # rows) so they now count toward this task's own per_task_eval/
                    # pooled_eval/perturbed_test_eval, same as an own-task success.
                    # Confirmed tradeoff: the same physical sample can now appear
                    # TWICE across pooled_eval's cumulative test corpus this run --
                    # once as task (t-1)'s original clean row (still sitting
                    # unmodified in task_test_splits[t-1]), once here as task t's
                    # perturbed row -- not deduplicated.
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
                # REWORK (recycled-pocket test pool): exact mirror of the malicious
                # block above, on the benign side. X_prev_test/y_prev_test/
                # gid_prev_test are already computed above (same task (t-1) split
                # for both classes).
                benign_idx_prev_test = np.where(y_prev_test == benign_label)[0]

                n_own_test_benign = len(X_test)
                X_test_ext_benign = np.concatenate([X_test, X_prev_test], axis=0)
                y_test_ext_benign = np.concatenate([y_test, y_prev_test], axis=0)
                benign_idx_test_ext = np.concatenate([benign_idx_test, benign_idx_prev_test + n_own_test_benign])

                benign_idx_test_c1 = c1_correct_pool(
                    pre_train_classifier_wrapper, X_test_ext_benign, benign_idx_test_ext, benign_label,
                    t, f"test-side benign (this task + recycled task {t - 1})", pocket_diag_log_path,
                )
                # REWORK (uncertainty-margin pipeline): mirror of the malicious
                # tiered selection above -- ranked by C2 (classifier_wrapper),
                # not C1, per the same FINETUNE rationale noted there.
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
                    # NEW (gradient-attack branch): mirror of the malicious test-side
                    # block above -- see pgd_boundary_search_batch's definition for
                    # the full rationale. mal_label passed as the "benign_label"
                    # (target class) argument, same convention train_red_agent_for_task
                    # used for this agent type.
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

                    # REWORK: no separate quota/subsample step -- see the malicious
                    # side's identical note above.
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
        # every t>0 regardless of outcome. This is the "testing" checkpoint
        # (test-side poisoning above already ran, so category_test now has
        # real perturbed markers, unlike the two adaptation-phase plots).
        X_test_scaled_np = to_tensor(task_test_splits[t][0]).numpy()
        if t > 0:
            # PROTOTYPE (adaptation-split pocket visualization): continue the
            # SAME running task_pocket_mask/prev_proba from the adaptation
            # plots above -- diff this checkpoint against the last one
            # (end of the clean-adaptation pass) before drawing, so any NEW
            # flips (e.g. from the test-side red agents' own poisoning, which
            # doesn't touch `model` -- but test-side poisoning is applied to
            # X_test, not the classifier, so no NEW flips are expected here;
            # this mainly re-confirms the mask carried over correctly) get
            # unioned in too.
            proba_test = _decision_grid(model, pca_mean, pca_components, pca_extent, DEVICE, mal_label)[2]
            task_pocket_mask |= compute_pocket_mask(prev_proba, proba_test)
            prev_proba = proba_test
            boundary_panels.append(plot_decision_boundary(
                model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                Xtr.numpy(), category, X_test_scaled_np, category_test,
                t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}.png"),
                checkpoint_label="pre_unlearn", highlight_mask=task_pocket_mask,
                precomputed_grid=(xx0, yy0, proba_test),
            ))
        else:
            boundary_panels.append(plot_decision_boundary(
                model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                Xtr.numpy(), category, X_test_scaled_np, category_test,
                t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}.png"),
                checkpoint_label="",
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

        # TEMPORARY DIAGNOSTIC (test-accuracy breakdown, Q1-3): snapshot of
        # THIS task's own (recycled-extended) test split against C2 (pre-
        # unlearning) -- see _test_accuracy_snapshot's docstring. Uses
        # task_test_splits[t] directly rather than the loop-local X_test/
        # y_test to guarantee it matches exactly what per_task_eval[t] above
        # was computed against.
        X_test_t, y_test_t = task_test_splits[t]
        pre_test_acc_snapshot = _test_accuracy_snapshot(
            classifier_wrapper, X_test_t, y_test_t, poison_idx_test, poison_idx_test_benign, mal_label, benign_label
        )
        pre_test_acc_metrics = {
            "balanced_accuracy": per_task_eval[t]["balanced_accuracy"],
            "pooled_accuracy": pooled_eval["accuracy"],
            "mean_accuracy": mean_per_task_bal_acc,
        }

        # TEMPORARY DIAGNOSTIC (pocket-targeting investigation), C1/C2 capture:
        # `classifier_wrapper` right now IS C2 (post-CL-training this task,
        # pre-unlearning -- nothing has touched `model` since train_cl_er/the
        # omega update above). `pre_train_classifier_wrapper` (captured before
        # train_cl_er, in the t>0 branch of step 1 above) is C1. Both are
        # frozen/live at exactly the point pocket-targeting is meant to check:
        # did this task's own poisoned training flip a sample C1 got right?
        # task_test_splits[t] already reflects this task's test-side poisoning
        # (X_test was overwritten in place at the poison indices, same array
        # positions as poison_idx_test/poison_idx_test_benign below).
        if t > 0:
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
                        "c3_pred": None,
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
                        "c3_pred": None,
                    }
                    if local_i >= n_poisoned_test_benign_own:
                        entry["source"] = f"recycled_from_task_{t - 1}"
                    pocket_diag_entries.append(entry)

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
            buffer_snapshots["after_fill"] = buffer_composition_summary()

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
                pcm = pc_result["perturbed_class_metrics"]
                print(f"    [Perturbation classifier] n_used={pc_result['n_used']}/class, "
                      f"n_eval={pc_result['n_eval']}, 2-class acc={tcm['accuracy']:.3f} "
                      f"bal_acc={tcm['balanced_accuracy']:.3f}")
                print(f"    [Perturbation classifier confusion] classes={cm['classes']}, matrix={cm['matrix']}")

                def _fmt_pr(m):
                    p = f"{m['precision']:.3f}" if m['precision'] is not None else "n/a"
                    r = f"{m['recall']:.3f}" if m['recall'] is not None else "n/a"
                    return f"precision={p} recall={r} (tp={m['tp']} fp={m['fp']} fn={m['fn']})"

                print(f"    [Perturbation classifier perturbed-class metrics] "
                      f"malicious_perturbed: {_fmt_pr(pcm['malicious'])} | "
                      f"benign_perturbed: {_fmt_pr(pcm['benign'])}")
                print(f"    [Forget set] potentially_perturbed_pool (malicious) = "
                      f"{len(pc_result['forget_idx_malicious'])}, potentially_perturbed_benign_pool = "
                      f"{len(pc_result['forget_idx_benign'])}, combined forget_idx = {len(forget_idx)} samples "
                      f"(retain = {len(retain_idx)}) -- 100% hard-classified selection")

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
                            "perturbed_class_metrics": pc_result["perturbed_class_metrics"],
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

                    # DIAGNOSTIC (retrain-vs-unlearn comparison): at exactly
                    # FULL_RETRAIN_DIAGNOSTIC_TASKS, skip unlearn_teacher_
                    # guided entirely and fully retrain a fresh classifier on
                    # ONLY the retain set instead -- see
                    # retrain_classifier_from_scratch's docstring. Every
                    # other task's unlearning is completely unaffected.
                    # model_pre_unlearn above was already snapshotted before
                    # either branch touches `model`, so measure_unlearning_
                    # efficacy/downstream "before/after" comparisons below
                    # work identically regardless of which branch ran.
                    if t in FULL_RETRAIN_DIAGNOSTIC_TASKS:
                        print(f"    [Unlearn] task {t}: DIAGNOSTIC full retrain on retain set + "
                              f"replay buffer (n_retain={len(retain_idx)}, n_replay={len(replay_buffer)}, "
                              f"forget set of {len(forget_idx)} samples discarded -- NOT trained on) "
                              f"instead of unlearn_teacher_guided...")
                        retrained_model, unlearn_diag = retrain_classifier_from_scratch(
                            feature_dim, retain_idx, forget_idx, Xtr, ytr, omega, DEVICE,
                            replay_buffer=replay_buffer,
                            epochs=RETRAIN_FULL_EPOCHS, lr=RETRAIN_FULL_LR, batch_size=BATCH_SIZE,
                        )
                        model.load_state_dict(retrained_model.state_dict())
                        grad_steps += unlearn_diag["n_steps"]
                    else:
                        print(f"    [Unlearn] task {t} (alpha={UNLEARN_ALPHA}, si_c={UNLEARN_SI_C}, "
                              f"n_forget={len(forget_idx)}, n_retain={len(retain_idx)})...")
                        unlearn_diag = unlearn_teacher_guided(
                            model=model, teacher_model=model_pre_unlearn,
                            forget_loader=forget_loader, retain_loader=retain_loader,
                            omega=omega, p_old_task=p_old_task, si_c=UNLEARN_SI_C,
                            epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA, device=DEVICE,
                        )
                        grad_steps += UNLEARN_EPOCHS * len(forget_loader)
                    unlearn_diag.setdefault("method", "unlearn_teacher_guided")
                    unlearning_ran_this_task = True  # TEMPORARY DIAGNOSTIC (pocket-targeting)
                    print(f"    [Unlearn diag] omega_l2={unlearn_diag['omega_l2_norm']:.3f}, raw losses "
                          f"forget={unlearn_diag['raw_forget_loss_mean']:.4f} "
                          f"retain={unlearn_diag['raw_retain_loss_mean']:.4f} "
                          f"si={unlearn_diag['raw_si_loss_mean']:.4f} (unweighted, before alpha/si_c)")

                    # Decision-boundary snapshot -- model state right after
                    # unlearning, paired with the "pre_unlearn" snapshot taken
                    # above (same task, same fixed PCA basis, same category
                    # arrays) so the two are directly comparable. PROTOTYPE
                    # (adaptation-split pocket visualization): continues the
                    # same running task_pocket_mask/prev_proba -- this is the
                    # 4th and last checkpoint of the task, so this final union
                    # is what carries forward into the montage/summary plot.
                    proba_post_unlearn = _decision_grid(
                        model, pca_mean, pca_components, pca_extent, DEVICE, mal_label
                    )[2]
                    task_pocket_mask |= compute_pocket_mask(prev_proba, proba_post_unlearn)
                    prev_proba = proba_post_unlearn
                    boundary_panels.append(plot_decision_boundary(
                        model, DEVICE, pca_mean, pca_components, pca_extent, mal_label,
                        Xtr.numpy(), category, X_test_scaled_np, category_test,
                        t, os.path.join(out_dir, "plots", f"decision_boundary_task{t}_post_unlearn.png"),
                        checkpoint_label="post_unlearn", highlight_mask=task_pocket_mask,
                        precomputed_grid=(xx0, yy0, proba_post_unlearn),
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

                    # TEMPORARY DIAGNOSTIC (test-accuracy breakdown, Q4-6): same
                    # snapshot as pre_test_acc_snapshot above, now against C3
                    # (post-unlearning) -- same X_test_t/y_test_t/poison_idx_test/
                    # poison_idx_test_benign (unlearning changes only the
                    # classifier's weights, never the test data), so the two
                    # snapshots' clean_pred arrays are directly comparable
                    # row-for-row in write_test_accuracy_breakdown's Q6.
                    post_test_acc_snapshot = _test_accuracy_snapshot(
                        classifier_wrapper, X_test_t, y_test_t, poison_idx_test, poison_idx_test_benign,
                        mal_label, benign_label
                    )

                    # POST-unlearn per_task_eval/pooled_eval/mean_per_task_balanced_
                    # accuracy (REWORK, task_metrics.png follow-up): the pre-unlearn
                    # versions of these (per_task_eval/pooled_eval/mean_per_task_
                    # bal_acc, computed at step 2 above) are what task_metrics.png's
                    # main panel plots today, labeled "PRE-unlearning snapshot" --
                    # this is the complete POST-unlearning equivalent, at this same
                    # checkpoint, so the plot can show the classifier's ACTUAL
                    # deployed performance instead. post_unlearn_prior_eval already
                    # covers tasks 0..t-1; post_unlearn_this_task_eval covers task t;
                    # combined they span the same range per_task_eval does, with no
                    # extra evaluate_classifier calls beyond pooled_eval's own.
                    post_unlearn_per_task_eval = dict(post_unlearn_prior_eval)
                    post_unlearn_per_task_eval[t] = post_unlearn_this_task_eval
                    post_unlearn_mean_per_task_bal_acc = float(np.mean(
                        [v["balanced_accuracy"] for v in post_unlearn_per_task_eval.values()]
                    ))
                    post_unlearn_pooled_eval = evaluate_classifier(classifier_wrapper, pooled_X, pooled_y)
                    post_test_acc_metrics = {
                        "balanced_accuracy": post_unlearn_this_task_eval["balanced_accuracy"],
                        "pooled_accuracy": post_unlearn_pooled_eval["accuracy"],
                        "mean_accuracy": post_unlearn_mean_per_task_bal_acc,
                    }
                    # NEW: POST-unlearning counterpart of the "[Task {t} classifier,
                    # PRE-unlearning]" line above -- same three numbers
                    # (this-task/pooled/mean-per-task balanced accuracy), now off
                    # post_unlearn_this_task_eval/post_unlearn_pooled_eval/
                    # post_unlearn_mean_per_task_bal_acc instead, so both are
                    # directly diffable line-for-line in the console log without
                    # needing to go dig them out of results.json.
                    print(f"[Task {t} classifier, POST-unlearning] "
                          f"this-task bal_acc={post_unlearn_this_task_eval['balanced_accuracy']:.4f} "
                          f"pooled bal_acc={post_unlearn_pooled_eval['balanced_accuracy']:.4f} "
                          f"mean-per-task bal_acc={post_unlearn_mean_per_task_bal_acc:.4f}")

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
                            "perturbed_class_metrics": pc_result["perturbed_class_metrics"],
                        },
                        "unlearn_diagnostics": unlearn_diag,
                        "forget_set": f_m, "retain_set": r_m,
                        "post_unlearn_this_task_eval": post_unlearn_this_task_eval,
                        "post_unlearn_perturbed_test_eval": post_unlearn_perturbed_test_eval,
                        "post_unlearn_per_task_eval": post_unlearn_per_task_eval,
                        "post_unlearn_pooled_eval": post_unlearn_pooled_eval,
                        "post_unlearn_mean_per_task_balanced_accuracy": post_unlearn_mean_per_task_bal_acc,
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
            buffer_snapshots["after_fill"] = buffer_composition_summary()

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

                def _clean_buffer(label, forget_idx_for_label, category_tag):
                    """REWORK (uncertainty-margin pipeline): removes buffer entries
                    matching this task's forget-pool gids for `label`. NO refill --
                    POTENTIALLY REVISIT: refill (uncertainty-sampled from this
                    task's own leftovers + the historical_clean_*_pool cross-task
                    pools) was removed pending a possible future reimplementation.
                    The buffer is intentionally left under budget for the rest of
                    this run until a LATER task's own update_buffer_madar() call
                    naturally tops it back up from that task's own full batch."""
                    forget_gids = set(int(g) for g in gid_train[forget_idx_for_label])
                    before = len(label_buffers.get(label, []))
                    label_buffers[label] = [
                        e for e in label_buffers.get(label, []) if e[3] not in forget_gids
                    ]
                    n_removed = before - len(label_buffers[label])
                    print(f"    [Buffer clean] label={category_tag}: removed {n_removed} forget-pool "
                          f"matches (no refill -- POTENTIALLY REVISIT). Buffer size now "
                          f"{len(label_buffers[label])}/{MEM_SIZE // 2}.")
                    return {"n_removed": int(n_removed), "buffer_size_after": int(len(label_buffers[label]))}

                if len(pc_result["forget_idx_malicious"]) > 0:
                    buffer_refill_diag["malicious"] = _clean_buffer(
                        mal_label, pc_result["forget_idx_malicious"], "malicious"
                    )
                if len(pc_result["forget_idx_benign"]) > 0:
                    buffer_refill_diag["benign"] = _clean_buffer(
                        benign_label, pc_result["forget_idx_benign"], "benign"
                    )

                replay_buffer.clear()
                for buf in label_buffers.values():
                    replay_buffer.extend(buf)

                if buffer_refill_diag:
                    unlearning_metrics["buffer_refill"] = buffer_refill_diag

            # REWORK (uncertainty-margin pipeline): buffer composition logged
            # here too -- AFTER this task's cleaning step -- so buffer_snapshots
            # (see its other two checkpoints: start-of-task and after-fill)
            # captures the buffer's FINAL state for this task, i.e. exactly what
            # task (t+1)'s own train_cl_er call will inherit as its replay signal.
            buffer_snapshots["after_cleaning"] = buffer_composition_summary()

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

        # TEMPORARY DIAGNOSTIC (pocket-targeting investigation), C3 fill-in:
        # `classifier_wrapper` now reflects the FINAL model state for this
        # task -- post-unlearning if it ran (unlearning_ran_this_task=True),
        # otherwise identical to C2 (nothing touched `model` since the C1/C2
        # capture above). Reuses the same poison_idx_test/poison_idx_test_benign
        # positional indices as the C1/C2 capture (task_test_splits[t]'s row
        # order is unchanged since then -- only values were overwritten, never
        # reordered), so no gid re-lookup is needed.
        if pocket_diag_entries:
            X_test_diag, _ = task_test_splits[t]
            # REWORK (recycled-pocket test pool, now persisted): poison_idx_test/
            # poison_idx_test_benign already include the recycled successes (see
            # the C1/C2 capture block above), and task_test_splits[t] now holds
            # their perturbed values too -- one predict() call per group covers
            # both own and recycled entries, in the same order they were
            # appended to pocket_diag_entries.
            c3_mal = classifier_wrapper.predict(X_test_diag[poison_idx_test]) if len(poison_idx_test) > 0 else None
            c3_ben = classifier_wrapper.predict(X_test_diag[poison_idx_test_benign]) \
                if len(poison_idx_test_benign) > 0 else None
            mal_i = ben_i = 0
            for e in pocket_diag_entries:
                if e["group"] == "malicious_perturbed":
                    e["c3_pred"] = int(c3_mal[mal_i]); mal_i += 1
                else:
                    e["c3_pred"] = int(c3_ben[ben_i]); ben_i += 1

        write_pocket_targeting_diagnostic(
            pocket_diag_log_path, t, pocket_diag_entries, unlearning_ran_this_task,
            benign_label, mal_label,
        )
        write_test_accuracy_breakdown(
            pocket_diag_log_path, t, pre_test_acc_snapshot, pre_test_acc_metrics, mal_label, benign_label,
            post_test_acc_snapshot, post_test_acc_metrics,
        )

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
            # REWORK (recycled-pocket test pool, now persisted): breakdown of
            # n_poisoned_test/n_poisoned_test_benign above -- how many of those
            # successes originated as candidates recycled from task (t-1)'s own
            # test split, rather than this task's own. gids are the ORIGINAL
            # task-(t-1) global ids; their poisoned values are now appended as
            # new rows into task_test_splits[t]/task_test_gids[t] (never written
            # back into task_test_splits[t-1] itself -- see the recycled_*
            # variables' docstring at the top of the loop), so they DO count
            # toward this task's own n_test/per_task_eval/pooled_eval/
            # perturbed_test_eval now, same as an own-task success. The same
            # physical sample can therefore appear twice across pooled_eval's
            # cumulative test corpus this run (once as task (t-1)'s original
            # clean row, once here) -- a confirmed, accepted tradeoff.
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
            "unlearning": unlearning_metrics,
            "buffer_snapshots": buffer_snapshots,
        })

        # NEW: end-of-task checkpoint -- OVERWRITES the previous task's file
        # each time, so classifier_checkpoint_path always holds only the
        # just-finished task's state (this task's C2/C3, i.e. post-
        # unlearning if it ran this task, matching `model`'s current weights
        # at this point in the loop). Carries everything main()'s
        # --resume_from path restores (see that block's comment for why each
        # field is needed) -- not just the classifier -- so a later run can
        # pick up at task t+1 as if this process had kept going.
        torch.save({
            "task_id": t,
            "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "feature_dim": feature_dim,
            "omega": omega,
            "p_old_task": p_old_task,
            "replay_buffer": list(replay_buffer),
            "bank_protos": bank.protos,
            "bank_counts": bank.counts,
            "bank_episode_embs": bank.episode_embs,
            "bank_task_order": bank.task_order,
            "historical_clean_mal_pool": historical_clean_mal_pool,
            "historical_clean_benign_pool": historical_clean_benign_pool,
            "task_test_splits": task_test_splits,
            "task_test_gids": task_test_gids,
            "task_poisoned_test_gids": task_poisoned_test_gids,
            "results": results,
            "warnings_log": warnings_log,
            "boundary_panels": boundary_panels,
        }, classifier_checkpoint_path)

        # TEMPORARY DIAGNOSTIC (pocket-targeting investigation): pause at the
        # end of every task after task 2, so pocket_diag_entries / the
        # written log section / any live variable above can be inspected
        # interactively before the run continues into the next task. Remove
        # once the investigation concludes.
        if t > 2:
            print(f"[Diagnostic] Pausing at breakpoint() after task {t} -- "
                  f"see {pocket_diag_log_path} for this task's pocket-targeting section.")
            breakpoint()

    config = {
        "strategy": "madar_er_kd_si_plus_unlearning",
        "poison_detector": POISON_DETECTOR,
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
        "perturbation_classifier_n": PERTURBATION_CLASSIFIER_N,
        "clean_replay_buffer_of_perturbed": CLEAN_REPLAY_BUFFER_OF_PERTURBED,
        "red_epsilon": RED_EPSILON, "red_max_steps": RED_MAX_STEPS,
        "red_c1_mismatch_patience": RED_C1_MISMATCH_PATIENCE,
        "full_retrain_diagnostic_tasks": sorted(FULL_RETRAIN_DIAGNOSTIC_TASKS),
        "retrain_full_epochs": RETRAIN_FULL_EPOCHS, "retrain_full_lr": RETRAIN_FULL_LR,
        "classifier_checkpoint_path": classifier_checkpoint_path,
        "resumed_from": args.resume_from, "resumed_from_task_id": resume_task_id,
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
