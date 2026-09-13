"""
madar_pocket_pipeline_tinyimagenet.py

Image port of madar_pocket_pipeline.py: SAME attack strategy (closed-form
centroid-shift "pocket creation" poisoning on train, dual-gradient "genuine
pocket" targeting on test -- see adversarial_attack_pocket, ported UNCHANGED),
SAME 5-lineage / shared-detector / shared-buffer design, SAME pipeline_log.txt
text-log format and the same BREAKPOINT_FROM_TASK interactive-inspection
mechanic. What actually changes:

  1. DATA: TinyImageNet-200 images (tinyimagenet_data_loader.py) instead of
     CICIDS network-flow feature vectors. Class-incremental task schedule
     ("<task0>+<step>x<n_increments>", ported from the sister project
     EmiliaR8/Meta-Unlearning's core/tasks.py -- see that repo's
     core/data/{base,tinyimagenet}.py for the non-poisoned original this
     pipeline's data loading is adapted from) REPLACES the chronological
     timestamp-shard split. Default "20+20x9": 10 tasks, 20 NEW classes
     introduced every task (task 0 included), all 200 classes used.
  2. MODEL: a small CNN (ClassifierCNN) instead of the flat-vector MLP --
     images are 3x64x64, not a feature vector, and a conv trunk is what makes
     TinyImageNet classification work at all. Keeps the same
     forward(x, return_latent=False) contract and `fc_last` attribute name
     the shared buffer's embed_latent() and the log's parameter-count
     reporting depend on.
  3. BINARY -> N-CLASS. Every piece of the original that hard-coded
     "benign_label/mal_label" as the only two possible values is generalized:

     - POISONING (craft_task_poison_images): the original's single
       benign<->malicious centroid-shift pair is generalized to NEAREST-CLASS
       PAIRING. A running per-class centroid dict is grown by ONE entry per
       class, computed once from that class's own data the task it is
       introduced and never recomputed (mirrors the original always measuring
       "separating_direction" fresh from that task's own current data). Each
       of THIS task's newly-introduced classes is paired with its nearest
       OTHER already-known class by centroid distance (raw pixel space, same
       space the original used) and a poison_fraction of its
       decision-boundary-closest samples are shifted toward that neighbor's
       centroid -- the direct N-class generalization of "shift benign toward
       malicious's centroid, and malicious toward benign's." Every poisoned
       row's TARGET class is recorded (idx: target_class), something the
       binary version never needed to track since the target was always
       "the other label" -- amnesiac/opposite_class below consume it.
     - "CLOSEST TO THE BOUNDARY" (confidence_margin): the original's
       |P(class1)-0.5| margin generalizes to the TOP1-TOP2 softmax confidence
       gap -- smallest gap = most boundary-adjacent, identical intent for any
       class count.
     - THE ATTACK ITSELF (adversarial_attack_pocket) NEEDS NO LOGIC CHANGE --
       success = poisoned model now wrong (argmax != true) AND clean model
       still right (argmax == true) is already class-count-agnostic. The only
       edits are shape-related: norms that assumed a flat (N, D) array now
       reduce over every non-batch axis (axis=tuple(range(1, arr.ndim))) so
       they work on (N, C, H, W) image tensors instead of flat vectors.
       eps_this_task's source, typical_class_gap, generalizes to the mean
       nearest-neighbor centroid distance among this task's ACTIVE classes
       (was: the one benign-malicious centroid distance).
     - DETECTOR stays conceptually binary (IS this row poisoned, yes/no) --
       that was never actually a 2-CLASS-of-the-underlying-problem fact, it is
       a fact about each row independent of how many classes exist. What
       generalizes is the training-set construction: one (clean, poisoned)
       group pair per active new class this task, not a fixed 4-group
       benign/malicious layout.
     - BUFFER budget: mem_size split evenly across every class seen so far
       (mem_size // n_active_classes) instead of a 50/50 two-way split --
       same anomaly+inlier IsolationForest interleave per class.
     - AMNESIAC: originally duplicated flagged rows under BOTH the only two
       possible labels. Generalizes to duplicating each row under its TRUE
       label and its RECORDED poison target label specifically (which, in
       the binary case, were the only two labels there were -- so this is a
       strict generalization, not a behavior change at n_classes=2).
     - OPPOSITE_CLASS: originally relabeled flagged rows to "the other
       binary label" (benign+mal-true, i.e. the only other option).
       Generalizes to relabeling each row to its RECORDED poison target class
       -- again identical at n_classes=2, since the target always WAS the
       other of the two labels there.
     - MASKED-PREFIX EVALUATION (new, not in the original -- needed only
       because class-incremental has a growing label space the binary
       version never did): logits for classes not yet introduced are masked
       to -inf before softmax/argmax, same convention
       Meta-Unlearning's evaluation protocol uses, so an untrained future
       class's still-random head weights can never win an argmax by chance.

  4. LOGGING: pipeline_log.txt's === Task N === sectioned structure and every
     table (train/test info, adaptation step, unlearning step, adversarial
     breakdown) are kept AS-IS in spirit. The one necessary change is
     _fmt_report: printing a 200-row classification_report every task is
     impractical, so it reports macro/weighted precision/recall/F1 (still
     from classification_report's own output_dict, so the numbers are
     identical to what the full table would show) instead of a full per-class
     table.
  5. BREAKPOINT_FROM_TASK=3 / --no_breakpoint: ported unchanged, at the
     user's request -- pause after every task from task 3 onward (i.e. every
     task after task 2) to inspect pipeline_log.txt and adjust before
     continuing.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # noqa: F401  (kept for parity; TinyImageNet ships its own val split, so no manual split is needed per-task)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    _XGBOOST_AVAILABLE = False

from tinyimagenet_data_loader import (
    DEFAULT_TASK_SETUP, TaskSchedule, build_schedule, load_tinyimagenet_tasks,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/mnt/erivas6/processed"
RUNS_BASE_DIR = "/mnt/erivas6/runs"

IMAGE_SHAPE = (3, 64, 64)

TASK0_EPOCHS = 30
TASK0_BATCH_SIZE = 128
TASK0_LR = 1e-3

ADAPT_EPOCHS = 15
CLEAN_ADAPT_EPOCHS = 5
ADAPT_LR = 1e-4
ADAPT_WEIGHT_DECAY = 1e-5
ADAPT_BATCH_SIZE = 64

POISON_FRACTION = 0.3

DETECTOR_N_PER_GROUP = 60
DETECTOR_UNCERTAIN_FRACTION = 0.5
LOGISTIC_PARAMS = dict(C=0.1, class_weight="balanced", max_iter=2000)
XGBOOST_PARAMS = dict(n_estimators=20, max_depth=2, learning_rate=0.1, reg_lambda=10.0,
                       subsample=0.7, colsample_bytree=0.3, min_child_weight=5,
                       eval_metric="logloss", device="cpu", tree_method="exact")

AMNESIAC_ROUNDS = 15

ATTACK_STEP = 0.04
ATTACK_MAX_STEPS = 100
ATTACK_CLEAN_WEIGHT = 1.0
ATTACK_EPS_MULTIPLIER = 0.15

MEM_SIZE = 4000
BUFFER_CONTAMINATION = 0.1
EMBED_BATCH_SIZE = 256

BREAKPOINT_FROM_TASK = 3

DEVICE = torch.device("cpu")
SEED = 42  # overwritten from --seed in main()

DEFAULT_CNN_CHANNELS = (32, 64, 128, 256)
DEFAULT_LATENT_DIM = 128


# ---------------------------------------------------------------------------
# Model + continual-adaptation wrapper
# ---------------------------------------------------------------------------
class ClassifierCNN(nn.Module):
    """4 conv+BN+ReLU+maxpool blocks (64x64 -> 4x4) -> latent FC -> fc_last.
    Same forward(x, return_latent=False) contract and `fc_last` attribute
    name as the flat-vector ClassifierNN this replaces, so embed_latent()
    (shared with the buffer) needs no changes."""

    def __init__(self, num_classes, in_channels=3, channels=DEFAULT_CNN_CHANNELS,
                 latent_dim=DEFAULT_LATENT_DIM):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        spatial = 64 // (2 ** len(channels))
        if spatial < 1:
            raise ValueError(f"too many conv blocks ({len(channels)}) for a 64x64 input")
        self.flatten_dim = c_in * spatial * spatial
        self.latent_fc = nn.Linear(self.flatten_dim, latent_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc_last = nn.Linear(latent_dim, num_classes)

    def forward(self, x, return_latent=False):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1)
        latent = self.relu(self.latent_fc(x))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


class AdaptableClassifier:
    """Continues training the EXISTING weights (never a fresh model). One
    instance per lineage, created once and reused for the whole run.

    n_active masking (new vs. the binary pipeline): predict_proba/predict/
    score all accept an optional n_active -- when given, logits for classes
    >= n_active are set to -inf before softmax, so a not-yet-introduced
    class's still-untrained head weights can never win an argmax. The binary
    pipeline never needed this since both classes existed from task 0."""

    def __init__(self, torch_model, lr=ADAPT_LR, weight_decay=ADAPT_WEIGHT_DECAY):
        self.model = torch_model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, X, y, replay_X=None, replay_y=None, epochs=5, batch_size=ADAPT_BATCH_SIZE):
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        if replay_X is not None and len(replay_X):
            Xt = torch.cat([Xt, torch.as_tensor(np.asarray(replay_X), dtype=torch.float32)], dim=0)
            yt = torch.cat([yt, torch.as_tensor(np.asarray(replay_y), dtype=torch.long)], dim=0)
        n = len(Xt)
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                self.opt.zero_grad()
                loss = self.loss_fn(self.model(Xt[idx]), yt[idx])
                loss.backward()
                self.opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X, n_active=None):
        self.model.eval()
        logits = self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        if n_active is not None and n_active < logits.shape[1]:
            logits = logits.clone()
            logits[:, n_active:] = float("-inf")
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X, n_active=None):
        return self.predict_proba(X, n_active=n_active).argmax(axis=1)

    def score(self, X, y, n_active=None):
        return (self.predict(X, n_active=n_active) == np.asarray(y)).mean()


@torch.no_grad()
def embed_latent(model_wrapper, X, batch_size=EMBED_BATCH_SIZE):
    model_wrapper.model.eval()
    parts = []
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        _, latent = model_wrapper.model(Xt[i:i + batch_size], return_latent=True)
        parts.append(latent.numpy())
    if parts:
        return np.concatenate(parts, axis=0)
    return np.empty((0, model_wrapper.model.fc_last.in_features), dtype=np.float32)


def confidence_margin(model_wrapper, X, n_active=None):
    """Top1-top2 softmax confidence gap -- generalizes the binary
    |P(class1)-0.5| margin used everywhere "closest to the decision
    boundary" mattered (poisoning target selection, detector training-set
    construction). Smaller gap = more boundary-adjacent, for any class
    count."""
    proba = model_wrapper.predict_proba(X, n_active=n_active)
    top2 = np.sort(proba, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


# ---------------------------------------------------------------------------
# Poisoning (generalizes craft_task_poison/craft_boundary_pocket_poison from
# the binary pipeline -- nearest-CLASS pairing instead of the one fixed
# benign<->malicious pair). See module docstring point 3.
# ---------------------------------------------------------------------------
def update_class_centroids(class_centroids, X_train_scaled, y_train, classes):
    """Grows class_centroids by ONE entry per class in `classes`, computed
    from X_train_scaled/y_train (this task's own data) and never recomputed
    later -- mirrors the original always measuring its centroid fresh from
    that task's own current data, just scoped to the task a class is
    introduced rather than to every task."""
    for c in classes:
        mask = y_train == c
        if mask.any():
            class_centroids[c] = X_train_scaled[mask].mean(axis=0)


def nearest_class(class_centroids, c):
    """Nearest OTHER known class to `c` by centroid L2 distance (flattened).
    Returns None if `c` has no known peers yet (task-0-only edge case)."""
    others = [k for k in class_centroids if k != c]
    if not others:
        return None
    cc = class_centroids[c].ravel()
    dists = [np.linalg.norm(cc - class_centroids[k].ravel()) for k in others]
    return others[int(np.argmin(dists))]


def craft_boundary_pocket_poison_images(model_wrapper, X, y, class_id, target_class_id,
                                          direction, n_points, n_active):
    """Poison the n_points samples of class_id closest to the decision
    boundary (smallest top1-top2 confidence gap), shifting each by
    `direction` -- a measured centroid-to-centroid vector. Labels unchanged
    (clean-label poisoning). Direct N-class generalization of
    craft_boundary_pocket_poison: `direction`/`target_class_id` are now
    PER-CLASS-PAIR instead of the one fixed benign<->malicious pair."""
    candidates = np.where(y == class_id)[0]
    if n_points <= 0 or len(candidates) == 0:
        return X.copy(), np.array([], dtype=np.int64)
    margin = confidence_margin(model_wrapper, X[candidates], n_active=n_active)
    closest = candidates[np.argsort(margin)[:n_points]]
    X_pois = X.copy()
    X_pois[closest] = X[closest] + direction
    return X_pois, closest


def craft_task_poison_images(clean_lineage, X_train_scaled, y_train, new_classes,
                              class_centroids, poison_fraction, n_active):
    """For each of THIS task's newly-introduced classes, pair it with its
    nearest OTHER already-known class (by centroid distance) and poison a
    poison_fraction of its boundary-closest samples toward that neighbor's
    centroid. Returns (X_train_poisoned, poison_idx: 1D array of every
    poisoned row this task, poison_target: dict idx->target_class_id, so
    amnesiac/opposite_class know exactly which class each row was shifted
    toward)."""
    # New classes' own centroids must exist before pairing so a class
    # introduced this task can be paired against ANOTHER class introduced
    # this same task, not just older ones.
    update_class_centroids(class_centroids, X_train_scaled, y_train, new_classes)

    X_train_poisoned = X_train_scaled.copy()
    poison_idx_parts = []
    poison_target = {}
    for c in new_classes:
        target = nearest_class(class_centroids, c)
        if target is None:
            continue  # only possible if new_classes has a single class and it's task 0's only class
        direction = class_centroids[target] - class_centroids[c]
        n_c = int((y_train == c).sum())
        n_poison_c = int(poison_fraction * n_c)
        X_train_poisoned, idx_c = craft_boundary_pocket_poison_images(
            clean_lineage, X_train_poisoned, y_train, class_id=c, target_class_id=target,
            direction=direction, n_points=n_poison_c, n_active=n_active,
        )
        poison_idx_parts.append(idx_c)
        for i in idx_c:
            poison_target[int(i)] = int(target)

    poison_idx = np.concatenate(poison_idx_parts) if poison_idx_parts else np.array([], dtype=np.int64)
    return X_train_poisoned, poison_idx, poison_target


def typical_class_gap_multiclass(class_centroids, active_classes):
    """Mean nearest-neighbor centroid distance among `active_classes` --
    generalizes typical_class_gap's one benign-malicious centroid distance
    to however many classes are active, feeding the same role (scaling the
    attack's epsilon_max budget)."""
    dists = []
    for c in active_classes:
        if c not in class_centroids:
            continue
        nn_c = nearest_class({k: v for k, v in class_centroids.items() if k in active_classes}, c)
        if nn_c is not None:
            dists.append(float(np.linalg.norm(class_centroids[c].ravel() - class_centroids[nn_c].ravel())))
    return float(np.mean(dists)) if dists else None


# ---------------------------------------------------------------------------
# Detector -- stays binary (poisoned vs not); generalizes only the
# training-set construction from a fixed 4-group layout to one (clean,
# poisoned) group pair per active new class this task.
# ---------------------------------------------------------------------------
def build_detector_training_set_mixed(ref_model, X_pois, y, poison_idx, new_classes,
                                       n_per_group, uncertain_fraction, seed, n_active):
    rng = np.random.default_rng(seed)
    poison_idx = np.asarray(poison_idx, dtype=np.int64)
    poison_set = set(poison_idx.tolist())

    def select_group(pool_idx, n):
        pool_idx = np.asarray(pool_idx)
        n = min(n, len(pool_idx))
        n_uncertain = int(round(uncertain_fraction * n))
        n_random = n - n_uncertain
        margin = confidence_margin(ref_model, X_pois[pool_idx], n_active=n_active)
        order = np.argsort(margin)
        uncertain_pick = pool_idx[order[:n_uncertain]]
        remaining = np.setdiff1d(pool_idx, uncertain_pick)
        random_pick = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
        return np.concatenate([uncertain_pick, random_pick])

    clean_parts, pois_parts, composition = [], [], {}
    for c in new_classes:
        class_idx = np.where(y == c)[0]
        clean_c = np.asarray([i for i in class_idx if i not in poison_set])
        pois_c = np.asarray([i for i in class_idx if i in poison_set])
        if len(clean_c):
            clean_parts.append(select_group(clean_c, n_per_group))
        if len(pois_c):
            pois_parts.append(select_group(pois_c, n_per_group))
        composition[int(c)] = {"clean": int(len(clean_c)), "poisoned": int(len(pois_c))}

    clean_all = np.concatenate(clean_parts) if clean_parts else np.array([], dtype=np.int64)
    pois_all = np.concatenate(pois_parts) if pois_parts else np.array([], dtype=np.int64)
    idx_all = np.concatenate([clean_all, pois_all])
    is_poisoned = np.concatenate([np.zeros(len(clean_all)), np.ones(len(pois_all))]).astype(int)
    return X_pois[idx_all], is_poisoned, composition


def train_detector(detector_type, X_det, y_det, seed):
    if detector_type == "xgboost" and not _XGBOOST_AVAILABLE:
        print("  [detector] xgboost not installed -- falling back to logistic regression.")
        detector_type = "logistic"
    X_det_flat = X_det.reshape(len(X_det), -1)
    if detector_type == "logistic":
        detector = LogisticRegression(random_state=seed, **LOGISTIC_PARAMS).fit(X_det_flat, y_det)
    elif detector_type == "xgboost":
        detector = XGBClassifier(random_state=seed, **XGBOOST_PARAMS).fit(X_det_flat, y_det)
    else:
        raise ValueError(f"Unknown detector_type: {detector_type!r}")
    return detector, detector_type


def run_detector(poisoned_baseline, X_train_poisoned, y_train, poison_idx, new_classes,
                  detector_type, seed, n_active):
    X_det, y_det, composition = build_detector_training_set_mixed(
        poisoned_baseline, X_train_poisoned, y_train, poison_idx, new_classes,
        n_per_group=DETECTOR_N_PER_GROUP, uncertain_fraction=DETECTOR_UNCERTAIN_FRACTION,
        seed=seed, n_active=n_active,
    )
    detector, detector_type_used = train_detector(detector_type, X_det, y_det, seed)
    X_det_flat = X_det.reshape(len(X_det), -1)
    train_acc = (detector.predict(X_det_flat) == y_det).mean()

    poison_set = set(int(i) for i in poison_idx)
    flagged_all = []
    for c in new_classes:
        class_indices = np.where(y_train == c)[0]
        flat = X_train_poisoned[class_indices].reshape(len(class_indices), -1)
        preds = detector.predict(flat)
        flagged_all.append(class_indices[preds == 1])
    detected_poison_idx = np.concatenate(flagged_all) if flagged_all else np.array([], dtype=np.int64)

    tp = len(poison_set.intersection(int(i) for i in detected_poison_idx))
    precision = tp / max(len(detected_poison_idx), 1)
    recall = tp / max(len(poison_idx), 1)
    metrics = {
        "detector_type": detector_type_used, "train_accuracy": float(train_acc),
        "composition": composition, "n_detected": int(len(detected_poison_idx)),
        "n_oracle": int(len(poison_idx)), "precision": precision, "recall": recall,
    }
    return detected_poison_idx, metrics, detector


# ---------------------------------------------------------------------------
# Unlearning variants -- dropped_rows is unchanged (index-count-agnostic);
# amnesiac/opposite_class now consume the recorded per-row poison TARGET
# class instead of a hardcoded "the other binary label" (identical behavior
# at n_classes=2, since the target always WAS the other of the two labels).
# ---------------------------------------------------------------------------
def apply_dropped_rows(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y):
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_poison_idx] = False
    lineage.adapt(X_train_poisoned[mask_keep], y_train[mask_keep],
                  replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)


def apply_amnesiac(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y,
                    poison_target):
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_poison_idx] = False
    X_clean_only, y_clean_only = X_train_poisoned[mask_keep], y_train[mask_keep]

    X_flagged = X_train_poisoned[detected_poison_idx]
    if len(X_flagged) == 0:
        lineage.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)
        return
    true_labels = y_train[detected_poison_idx]
    # Detector-flagged rows may include false positives the oracle poison_target
    # dict has no entry for; fall back to the row's own true label (a no-op
    # duplicate) rather than dropping it from the amnesiac batch entirely.
    target_labels = np.array([poison_target.get(int(i), true_labels[j])
                               for j, i in enumerate(detected_poison_idx)])
    X_flagged_dup = np.vstack([X_flagged, X_flagged])
    y_flagged_dup = np.concatenate([true_labels, target_labels])
    for _ in range(AMNESIAC_ROUNDS):
        lineage.adapt(X_flagged_dup, y_flagged_dup, epochs=1)
        lineage.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=1)


def apply_opposite_class(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y,
                          poison_target):
    y_relabel = y_train.copy()
    for i in detected_poison_idx:
        y_relabel[i] = poison_target.get(int(i), y_train[i])  # unknown target (false positive): no-op relabel
    lineage.adapt(X_train_poisoned, y_relabel, replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)


# ---------------------------------------------------------------------------
# Test-time "genuine pocket" attack -- UNCHANGED logic from the binary
# pipeline (success = poisoned model now wrong AND clean model still right
# is already argmax-generic for any class count). Only the norm reductions
# are generalized from a flat (N, D) axis=1 assumption to "every non-batch
# axis," so this works on (N, C, H, W) image tensors.
# ---------------------------------------------------------------------------
def adversarial_attack_pocket(poisoned_model, clean_model, X, y, epsilon_max, n_active,
                               step=ATTACK_STEP, max_steps=ATTACK_MAX_STEPS,
                               clean_weight=ATTACK_CLEAN_WEIGHT, per_feature_epsilon=None):
    X_adv = X.copy()
    x0 = X.copy()
    success = np.zeros(len(X), dtype=bool)
    active = np.ones(len(X), dtype=bool)
    reduce_axes = tuple(range(1, X.ndim))  # all but the batch axis -- was hardcoded axis=1 for flat vectors

    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.where(active)[0]
        Xa = X_adv[idx]
        y_active_np = y[idx]
        y_active = torch.tensor(y_active_np, dtype=torch.long)

        poisoned_model.model.eval()
        xl_p = torch.tensor(Xa, dtype=torch.float32, requires_grad=True)
        logits_p = poisoned_model.model(xl_p)
        if n_active is not None and n_active < logits_p.shape[1]:
            logits_p = logits_p.clone()
            logits_p[:, n_active:] = float("-inf")
        proba_p = torch.softmax(logits_p, dim=1)
        proba_p.gather(1, y_active.unsqueeze(1)).squeeze(1).sum().backward()
        grad_p = xl_p.grad.numpy()
        pred_p = proba_p.detach().numpy().argmax(axis=1)

        clean_model.model.eval()
        xl_c = torch.tensor(Xa, dtype=torch.float32, requires_grad=True)
        logits_c = clean_model.model(xl_c)
        if n_active is not None and n_active < logits_c.shape[1]:
            logits_c = logits_c.clone()
            logits_c[:, n_active:] = float("-inf")
        proba_c = torch.softmax(logits_c, dim=1)
        proba_c.gather(1, y_active.unsqueeze(1)).squeeze(1).sum().backward()
        grad_c = xl_c.grad.numpy()
        pred_c = proba_c.detach().numpy().argmax(axis=1)

        newly = (pred_p != y_active_np) & (pred_c == y_active_np)
        success[idx[newly]] = True
        active[idx[newly]] = False

        still = ~newly
        idx_s = idx[still]
        if len(idx_s):
            d = -grad_p[still] + clean_weight * grad_c[still]
            # np.linalg.norm's tuple-axis form only supports 2 axes (matrix-norm
            # semantics) -- (N, C, H, W) needs 3 reduced (C, H, W), so the L2
            # norm is computed directly via sqrt(sum(x**2)) instead.
            nrm = np.sqrt(np.sum(d ** 2, axis=reduce_axes, keepdims=True))
            nrm[nrm < 1e-8] = 1e-8
            d = d / nrm
            X_adv[idx_s] = X_adv[idx_s] + step * d
            delta = X_adv[idx_s] - x0[idx_s]
            if per_feature_epsilon is not None:
                delta = np.clip(delta, -per_feature_epsilon, per_feature_epsilon)
                X_adv[idx_s] = x0[idx_s] + delta
            dn = np.sqrt(np.sum(delta ** 2, axis=reduce_axes, keepdims=True))
            too_far = dn.reshape(-1) > epsilon_max
            if too_far.any():
                X_adv[idx_s[too_far]] = x0[idx_s[too_far]] + delta[too_far] / dn[too_far] * epsilon_max

    return X_adv, success, np.linalg.norm((X_adv - x0).reshape(len(X), -1), axis=1)


# ---------------------------------------------------------------------------
# Pooled / mean / per-task accuracy
# ---------------------------------------------------------------------------
def pooled_and_per_task_accuracy(model, all_test_sets, n_active):
    per_task = {}
    X_all, y_all = [], []
    for tid, (Xh, yh) in all_test_sets.items():
        acc = model.score(Xh, yh, n_active=n_active)
        per_task[tid] = (acc, len(yh))
        X_all.append(Xh)
        y_all.append(yh)
    X_pooled = np.vstack(X_all)
    y_pooled = np.concatenate(y_all)
    pooled_acc = model.score(X_pooled, y_pooled, n_active=n_active)
    mean_acc = float(np.mean([acc for acc, n in per_task.values()]))
    return pooled_acc, mean_acc, per_task


# ---------------------------------------------------------------------------
# Shared replay buffer -- same anomaly+inlier IsolationForest interleave as
# the binary pipeline; budget generalizes from a 50/50 two-way split to an
# even split across every class seen so far.
# ---------------------------------------------------------------------------
def anomaly_inlier_interleave(scores, n_select):
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


def update_shared_buffer(embed_model, label_buffers, X_clean, y_clean, category_clean, sample_id_clean,
                          active_classes, mem_size=MEM_SIZE, contamination=BUFFER_CONTAMINATION):
    X_np = np.asarray(X_clean, dtype=np.float32)
    Y_np = np.asarray(y_clean)
    cat_np = np.asarray(category_clean, dtype=object)
    id_np = np.asarray(sample_id_clean)
    L_np = embed_latent_np(embed_model, X_np)
    latent_dim = L_np.shape[1]
    budget_per_class = max(mem_size // max(len(active_classes), 1), 2)

    for lbl in active_classes:
        old_entries = label_buffers.get(lbl, [])
        if old_entries:
            old_X = np.stack([e[0] for e in old_entries]).astype(np.float32)
            old_Y = np.array([e[1] for e in old_entries], dtype=Y_np.dtype)
            old_cat = np.array([e[2] for e in old_entries], dtype=object)
            old_id = np.array([e[3] for e in old_entries])
            old_L = embed_latent_np(embed_model, old_X)
        else:
            old_X = np.empty((0,) + X_np.shape[1:], dtype=np.float32)
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

        n_select = min(budget_per_class, len(pool_X))
        if n_select == 0:
            label_buffers[lbl] = []
            continue

        iso = IsolationForest(contamination=contamination, n_jobs=-1, random_state=SEED)
        iso.fit(pool_L)
        scores = iso.decision_function(pool_L)
        interleaved_idx = anomaly_inlier_interleave(scores, n_select)

        label_buffers[lbl] = [
            (pool_X[i], pool_Y[i], pool_cat[i], int(pool_id[i])) for i in interleaved_idx
        ]

    replay = []
    for buf in label_buffers.values():
        replay.extend(buf)
    return replay


def embed_latent_np(model_wrapper, X):
    return embed_latent(model_wrapper, X)


def flatten_replay(replay_buffer):
    if not replay_buffer:
        return None, None
    replay_X = np.stack([e[0] for e in replay_buffer]).astype(np.float32)
    replay_y = np.array([e[1] for e in replay_buffer])
    return replay_X, replay_y


def buffer_distribution(label_buffers):
    dist = {}
    for lbl, entries in label_buffers.items():
        cats = {}
        for e in entries:
            cats[e[2]] = cats.get(e[2], 0) + 1
        dist[lbl] = {"total": len(entries), "by_category": cats}
    return dist


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def _fmt_report(model, X, y, n_active):
    """Macro/weighted precision-recall-F1 summary instead of the binary
    pipeline's full per-class classification_report table -- with up to 200
    active classes a full table is impractical, and these two rows are
    computed from the SAME classification_report call (output_dict=True), so
    the numbers are identical to what the full table would show."""
    preds = model.predict(X, n_active=n_active)
    report = classification_report(y, preds, zero_division=0, output_dict=True)
    lines = [f"n_classes_in_report={len([k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')])}, "
             f"accuracy={report.get('accuracy', float('nan')):.3f}"]
    for key in ("macro avg", "weighted avg"):
        r = report.get(key, {})
        lines.append(f"  {key}: precision={r.get('precision', float('nan')):.3f} "
                     f"recall={r.get('recall', float('nan')):.3f} f1={r.get('f1-score', float('nan')):.3f}")
    return "\n".join(lines)


def write_task_log(log_path, t, sections):
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 78}\n=== Task {t} ===\n{'=' * 78}\n")
        for title, body in sections:
            f.write(f"\n-- {title} --\n{body}\n")


# ---------------------------------------------------------------------------
# PCA correctness plots -- final task only. PCA on raw pixels (flattened)
# rather than latents, same as the binary pipeline used raw scaled features.
# ---------------------------------------------------------------------------
def plot_correctness_grid(out_path, pca_fit, panels, n_active):
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, model, X, y) in zip(axes, panels):
        proj = pca_fit.transform(X.reshape(len(X), -1))
        correct = model.predict(X, n_active=n_active) == y
        ax.scatter(proj[correct, 0], proj[correct, 1], s=8, c="#1f3d7a", label="correct", alpha=0.5)
        ax.scatter(proj[~correct, 0], proj[~correct, 1], s=30, facecolors="none", edgecolors="#a34a12",
                   linewidths=1.2, label="WRONG")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global SEED
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_pipeline_tinyimagenet_run")
    ap.add_argument("--data-root", type=str, default=DATA_ROOT,
                     help="Directory CONTAINING tiny-imagenet-200/ (after unzip).")
    ap.add_argument("--cache-root", type=str, default=None,
                     help="Where to cache decoded uint8 .npy arrays (default: <data-root>/_cache).")
    ap.add_argument("--task_setup", type=str, default=DEFAULT_TASK_SETUP,
                     help="'<task0>+<step>x<n_increments>' class-incremental schedule, e.g. '20+20x9' "
                          "(10 tasks, 20 new classes/task, all 200 TinyImageNet classes).")
    ap.add_argument("--detector_type", type=str, default="xgboost", choices=["xgboost", "logistic"])
    ap.add_argument("--poison_fraction", type=float, default=POISON_FRACTION)
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks >= "
                          f"{BREAKPOINT_FROM_TASK} (needed for a non-interactive/headless run).")
    ap.add_argument("--joint_buffer_allow_perturbed", action="store_true")
    ap.add_argument("--joint_buffer_purge_after_fill", action="store_true")
    ap.add_argument("--cnn_channels", type=str, default=",".join(str(c) for c in DEFAULT_CNN_CHANNELS),
                     help="Comma-separated conv channel widths, e.g. '32,64,128,256' (default). "
                          "Each entry halves the spatial size (64x64 input), so at most 6 entries.")
    ap.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    ap.add_argument("--per_feature_epsilon", type=float, default=None)
    args = ap.parse_args()
    cnn_channels = tuple(int(c) for c in args.cnn_channels.split(","))
    schedule = build_schedule(args.task_setup)

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(RUNS_BASE_DIR, "madar_pocket_pipeline_tinyimagenet", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (TinyImageNet, class-incremental)\n"
            "============================================================\n"
            "5 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "dropped_rows, amnesiac, opposite_class. One shared detector + one shared\n"
            "replay buffer per task. Task 0 is plain initial training only -- no\n"
            "poisoning/detection/unlearning yet.\n"
            f"Task schedule: {schedule.spec} ({schedule.n_tasks} tasks, {schedule.n_classes} classes)\n"
            f"CNN channels: {cnn_channels}, latent_dim: {args.latent_dim}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
        )

    print(f"Loading TinyImageNet from {args.data_root} under schedule {schedule.spec}...")
    tasks, wnids = load_tinyimagenet_tasks(args.data_root, schedule, cache_root=args.cache_root)
    print(f"n_tasks={schedule.n_tasks}, n_classes={schedule.n_classes}, "
          f"task sizes(train)={[len(t['y_train']) for t in tasks]}")

    scaler_mean, scaler_std = None, None  # per-channel, fit on task 0 only
    lineages = {}
    baseline_label_buffers = {}
    baseline_replay_buffer = []
    joint_label_buffers = {}
    joint_replay_buffer = []
    task_test_splits = {}
    class_centroids = {}
    results = []

    def to_scaled(X_raw_uint8):
        X = X_raw_uint8.astype(np.float32) / 255.0
        return ((X - scaler_mean[None, :, None, None]) / scaler_std[None, :, None, None]).astype(np.float32)

    LINEAGE_NAMES = ["clean", "poisoned_baseline", "dropped_rows", "amnesiac", "opposite_class"]
    FIX_NAMES = ["dropped_rows", "amnesiac", "opposite_class"]

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

            base_model = ClassifierCNN(schedule.n_classes, channels=cnn_channels,
                                        latent_dim=args.latent_dim).to(DEVICE)
            Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
            yt = torch.tensor(y_train, dtype=torch.long)
            opt0 = torch.optim.Adam(base_model.parameters(), lr=TASK0_LR)
            loss_fn0 = nn.CrossEntropyLoss()
            base_model.train()
            n = len(Xt)
            for _ in range(TASK0_EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, TASK0_BATCH_SIZE):
                    idx = perm[i:i + TASK0_BATCH_SIZE]
                    if len(idx) < 2:
                        continue
                    opt0.zero_grad()
                    loss = loss_fn0(base_model(Xt[idx]), yt[idx])
                    loss.backward()
                    opt0.step()
            base_model.eval()

            for name in LINEAGE_NAMES:
                lineages[name] = AdaptableClassifier(copy.deepcopy(base_model))

            task_acc = {name: lineages[name].score(X_test_scaled, y_test, n_active=n_active) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_scaled, y_test)
            update_class_centroids(class_centroids, X_train_scaled, y_train, new_classes)

            sample_id = np.arange(len(y_train))
            category = np.array(["clean"] * len(y_train), dtype=object)
            baseline_replay_buffer = update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train, category, sample_id,
                new_classes,
            )
            joint_replay_buffer = update_shared_buffer(
                lineages["poisoned_baseline"], joint_label_buffers, X_train_scaled, y_train, category, sample_id,
                new_classes,
            )

            train_section = (
                f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
                f"n_train: {len(y_train)}\nperturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = (
                f"n_test: {len(y_test)}\ngenuine pockets: N/A (task 0 -- no poisoning yet)\n"
            )
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            unlearn_section = "N/A -- task 0 has no prior model to poison against."

            write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Unlearning step", unlearn_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": SEED, "schedule": schedule.as_dict(), "wnids": wnids,
                "scaler_mean": scaler_mean, "scaler_std": scaler_std,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                "baseline_label_buffers": baseline_label_buffers, "baseline_replay_buffer": baseline_replay_buffer,
                "joint_label_buffers": joint_label_buffers, "joint_replay_buffer": joint_replay_buffer,
                "task_test_splits": task_test_splits, "class_centroids": class_centroids,
                "results": results, "poison_fraction": poison_fraction,
                "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
                "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path)
            continue

        # -------------------------------------------------------------
        # Tasks 1..N-1: poison -> detect -> unlearn, every task's OWN new
        # classes (matches the binary pipeline's "poison every task" --
        # here that means every task's newly-INTRODUCED classes, since
        # older classes' raw rows are never retrained on directly).
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        replay_X, replay_y = flatten_replay(joint_replay_buffer)
        baseline_replay_X, baseline_replay_y = flatten_replay(baseline_replay_buffer)

        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=replay_X, replay_y=replay_y,
                                 epochs=CLEAN_ADAPT_EPOCHS)

        X_train_poisoned, poison_idx, poison_target = craft_task_poison_images(
            lineages["clean"], X_train_scaled, y_train, new_classes, class_centroids,
            poison_fraction, n_active,
        )

        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                             replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                             epochs=ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(
                X_train_poisoned[poison_idx],
                np.array([poison_target[int(i)] for i in poison_idx]), n_active=n_active,
            ) if len(poison_idx) else float("nan")
        )

        eps_this_task = typical_class_gap_multiclass(class_centroids, new_classes)
        if eps_this_task is None:
            eps_this_task = 1.0
        eps_this_task *= ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, n_active=n_active, per_feature_epsilon=args.per_feature_epsilon,
        )

        # Spillover check -- re-attack every PRIOR task's test set, every
        # task, same poisoned_baseline/clean reference pair.
        historical_adv = {}
        for s, (Xs_scaled, ys) in task_test_splits.items():
            eps_s = typical_class_gap_multiclass(class_centroids, schedule.classes_for(s))
            eps_s = (eps_s * ATTACK_EPS_MULTIPLIER) if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = adversarial_attack_pocket(
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

        detected_poison_idx, det_metrics, detector = run_detector(
            lineages["poisoned_baseline"], X_train_poisoned, y_train, poison_idx, new_classes,
            args.detector_type, SEED, n_active,
        )

        pre_unlearn_metrics = {}
        for name in FIX_NAMES:
            lineages[name].adapt(X_train_poisoned, y_train, replay_X=replay_X, replay_y=replay_y,
                                  epochs=ADAPT_EPOCHS)
            pre_task_acc = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            pre_pooled_acc, pre_mean_acc, _ = pooled_and_per_task_accuracy(lineages[name], all_test_sets_full, n_active)
            pre_unlearn_metrics[name] = {
                "task_acc": pre_task_acc, "pooled_acc": pre_pooled_acc, "mean_acc": pre_mean_acc,
            }
        apply_dropped_rows(lineages["dropped_rows"], X_train_poisoned, y_train, detected_poison_idx,
                            replay_X, replay_y)
        apply_amnesiac(lineages["amnesiac"], X_train_poisoned, y_train, detected_poison_idx,
                        replay_X, replay_y, poison_target)
        apply_opposite_class(lineages["opposite_class"], X_train_poisoned, y_train, detected_poison_idx,
                              replay_X, replay_y, poison_target)

        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = pooled_and_per_task_accuracy(lineages[name], all_test_sets_full, n_active)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = _fmt_report(lineages[name], X_test_scaled, y_test, n_active)

        still_evades = {}
        for name in FIX_NAMES:
            pred = lineages[name].predict(X_test_adv, n_active=n_active)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Update BOTH replay buffers -- only now, after every lineage's
        # adaptation/unlearning for this task is fully done.
        category_all = np.array(["clean"] * len(y_train), dtype=object)
        category_all[poison_idx] = "perturbed"
        sample_id = np.arange(len(y_train))  # per-task-local ids; unique within this task's own batch

        baseline_replay_buffer = update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers,
            X_train_poisoned, y_train, category_all, sample_id, new_classes,
        )

        joint_unfiltered = args.joint_buffer_allow_perturbed or args.joint_buffer_purge_after_fill
        if joint_unfiltered:
            joint_X, joint_y, joint_cat, joint_id = X_train_poisoned, y_train, category_all, sample_id
        else:
            clean_mask = np.ones(len(X_train_poisoned), dtype=bool)
            clean_mask[detected_poison_idx] = False
            joint_X, joint_y = X_train_poisoned[clean_mask], y_train[clean_mask]
            joint_cat, joint_id = category_all[clean_mask], sample_id[clean_mask]
        joint_replay_buffer = update_shared_buffer(
            lineages["poisoned_baseline"], joint_label_buffers,
            joint_X, joint_y, joint_cat, joint_id, new_classes,
        )

        joint_purge_counts = {c: 0 for c in new_classes}
        if args.joint_buffer_purge_after_fill:
            for lbl in new_classes:
                entries = joint_label_buffers.get(lbl, [])
                if not entries:
                    continue
                X_buf = np.stack([e[0] for e in entries]).astype(np.float32)
                preds = detector.predict(X_buf.reshape(len(X_buf), -1))
                kept = [e for e, p in zip(entries, preds) if p == 0]
                joint_purge_counts[lbl] = len(entries) - len(kept)
                joint_label_buffers[lbl] = kept
            joint_replay_buffer = []
            for buf in joint_label_buffers.values():
                joint_replay_buffer.extend(buf)

        task_test_splits[t] = (X_test_scaled, y_test)

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        train_section = (
            f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
            f"n_train: {len(y_train)}\n"
            f"perturbed (oracle): {len(poison_idx)}, detector-flagged: {len(detected_poison_idx)}\n"
            f"poison_fraction used: {poison_fraction}\n"
            f"poisoned_baseline accuracy on poisoned points' forced (target-class) labels: "
            f"{acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
        )

        test_section = (
            f"n_test: {len(y_test)}\n"
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({_fmt_pct(succ_pocket.mean())})\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        baseline_dist = buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution "
                            f"(post-update, this task, {len(baseline_dist)} classes buffered): "
                            f"total={sum(v['total'] for v in baseline_dist.values())}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report summary (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
        adapt_section = "\n".join(adapt_lines)

        joint_dist = buffer_distribution(joint_label_buffers)
        unlearn_lines = [
            f"detector type: {det_metrics['detector_type']}, train accuracy: {det_metrics['train_accuracy']:.3f}",
            f"detector precision/recall vs oracle (pooled over {len(new_classes)} new classes): "
            f"P={det_metrics['precision']:.2f} R={det_metrics['recall']:.2f}",
            f"n_detected={det_metrics['n_detected']}, n_oracle={det_metrics['n_oracle']}",
        ]
        if args.joint_buffer_purge_after_fill:
            joint_mode_desc = "UNFILTERED admission + detector purge after fill, no refill"
        elif args.joint_buffer_allow_perturbed:
            joint_mode_desc = "UNFILTERED -- perturbed rows allowed"
        else:
            joint_mode_desc = "detector-clean only"
        unlearn_lines.append(
            f"JOINT replay buffer ({joint_mode_desc}; post-update, this task, "
            f"{len(joint_dist)} classes buffered): total={sum(v['total'] for v in joint_dist.values())}"
        )
        if args.joint_buffer_purge_after_fill:
            unlearn_lines.append(f"JOINT buffer purge this task (removed, NOT backfilled): {joint_purge_counts}")
        unlearn_lines += [
            "",
            "Pre-unlearning (poison-adapted, before any fix) accuracy:",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}",
        ]
        for name in FIX_NAMES:
            pre = pre_unlearn_metrics[name]
            unlearn_lines.append(
                f"{name:<18} {pre['task_acc']:>10.3f} {pre['pooled_acc']:>12.3f} {pre['mean_acc']:>10.3f}"
            )
        unlearn_lines.append("")
        unlearn_lines.append("Post-unlearning accuracy:")
        unlearn_lines.append(
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}"
        )
        for name in FIX_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adv_acc_name = lineages[name].score(X_test_adv, y_test, n_active=n_active)
            unlearn_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        unlearn_lines.append("")
        for name in FIX_NAMES:
            unlearn_lines.append(f"[{name}] classification report summary (this task's clean test):")
            unlearn_lines.append(per_class_reports[name])
        unlearn_section = "\n".join(unlearn_lines)

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
                f"{s:<12} {n_s:>8} {int(succ_s.sum()):>10}/{n_s:<7} {_fmt_pct(succ_s.mean()):>8} {eps_s:>8.4f}"
            )
        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-unlearning) classifier accuracy on each source task's adv-test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-unlearning) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-unlearning) classifier COMBINED (clean+adversarial, pooled) "
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

        write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Unlearning step", unlearn_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
        ])

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()), "detector_metrics": det_metrics,
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": SEED, "schedule": schedule.as_dict(), "wnids": wnids,
            "scaler_mean": scaler_mean, "scaler_std": scaler_std,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            "baseline_label_buffers": baseline_label_buffers, "baseline_replay_buffer": baseline_replay_buffer,
            "joint_label_buffers": joint_label_buffers, "joint_replay_buffer": joint_replay_buffer,
            "task_test_splits": task_test_splits, "class_centroids": class_centroids,
            "results": results, "poison_fraction": poison_fraction,
            "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
            "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path)

        print(f"Task {t} done. Genuine pocket rate: {_fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")

        if t == schedule.n_tasks - 1:
            pca_fit = PCA(n_components=2, random_state=SEED).fit(X_train_scaled.reshape(len(X_train_scaled), -1))
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            plot_correctness_grid(os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels, n_active)

        # Interactive inspection breakpoint -- ported unchanged from the
        # binary pipeline, at the user's request: pause after every task
        # from task 3 onward (i.e. every task after task 2) to look at
        # pipeline_log.txt and adjust before continuing.
        if not args.no_breakpoint and t >= BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`baseline_label_buffers`, `joint_label_buffers`, `class_centroids`, or the log "
                  f"at {log_path}. Continue with `c`.")
            breakpoint()

    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
