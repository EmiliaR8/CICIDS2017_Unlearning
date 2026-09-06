"""
madar_pocket_pipeline.py

10-task extension of the notebook-based poisoning -> detection -> unlearning
pipeline (task8_pipelineCopy1.ipynb), run across every chronological task
instead of a single task-7-checkpoint snapshot. Replaces
madar_unlearning_cl_pipeline.py's RL red-agent training, SI regularization,
KD distillation, RandomForest 4-class forget-set classifier, and PGD/C1-gated
test attack with the notebook's mechanics: closed-form centroid-shift
poisoning, a binary detector, a dual-gradient "genuine pocket" attack, and
three unlearning variants (dropped-rows / amnesiac / opposite-class).

FIVE parallel model lineages, all starting from the same task-0 model:
  clean             -- never poisoned; reference model only (supplies the
                        "still correct" half of the genuine-pocket criterion)
  poisoned_baseline -- poisoned every task, never fixed ("no fix" reference,
                        also the model the shared per-task detector trains on)
  dropped_rows / amnesiac / opposite_class
                    -- poisoned every task (from their OWN prior-task
                       weights), then fixed using the SAME shared per-task
                       detector's flagged rows -- they disagree only on the
                       fix mechanic, never on what counts as poisoned.

Replay buffer is SHARED across all 5 lineages: since there is one detector
per task (trained from poisoned_baseline), "clean vs. poisoned" is a single
task-level fact, so one buffer curated from that fact is consistent for
everyone. The buffer is filled/re-selected ONLY AFTER every lineage's
adaptation + unlearning step has finished for the task -- a lineage's own
.adapt() call this task always reads the buffer as it stood at the END of
the PREVIOUS task, so this task's rows are never trained on twice (once
directly, once via a buffer that already contains them).

No SI, no KD, no RL red agent -- every adaptation step is plain
CrossEntropyLoss on train + replay (AdaptableClassifier, same as the
notebook). Data loading/task splitting (h5_data_loader, NUM_TASKS=10,
TASK_FRACTIONS, TASK_TEST_FRAC) is unchanged from the rest of this repo.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    _XGBOOST_AVAILABLE = False

from h5_data_loader import load_pooled_chronological_tasks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H5_DATASET_PATH = "/mnt/processed_data/subsampled_dataset.h5"
RUNS_BASE_DIR = "/mnt/erivas6/runs"

NUM_TASKS = 10
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]
TASK_TEST_FRAC = 0.20
FEATURE_CLIP = 10.0

TASK0_EPOCHS = 30
TASK0_BATCH_SIZE = 256
TASK0_LR = 1e-3

ADAPT_EPOCHS = 15
CLEAN_ADAPT_EPOCHS = 5
ADAPT_LR = 1e-4
ADAPT_WEIGHT_DECAY = 1e-5
ADAPT_BATCH_SIZE = 128

POISON_FRACTION = 0.3

DETECTOR_N_PER_GROUP = 60
DETECTOR_UNCERTAIN_FRACTION = 0.5
LOGISTIC_PARAMS = dict(C=0.1, class_weight="balanced", max_iter=2000)
XGBOOST_PARAMS = dict(n_estimators=20, max_depth=2, learning_rate=0.1, reg_lambda=10.0,
                       subsample=0.7, colsample_bytree=0.3, min_child_weight=5,
                       eval_metric="logloss")

AMNESIAC_ROUNDS = 15

ATTACK_STEP = 0.04
ATTACK_MAX_STEPS = 100
ATTACK_CLEAN_WEIGHT = 1.0
ATTACK_EPS_MULTIPLIER = 0.15

MEM_SIZE = 4000
BUFFER_CONTAMINATION = 0.1
EMBED_BATCH_SIZE = 512

BREAKPOINT_FROM_TASK = 3

DEVICE = torch.device("cpu")
SEED = 42  # overwritten from --seed in main()


# ---------------------------------------------------------------------------
# Model + continual-adaptation wrapper (same architecture/semantics as the
# notebook; return_latent kept so the shared replay buffer's IsolationForest
# selection has an embedding space to work in, matching this repo's existing
# buffer mechanism).
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


class AdaptableClassifier:
    """Continues training the EXISTING weights (never a fresh model). One
    instance per lineage, created once and reused for the whole run so its
    Adam optimizer state persists naturally across tasks."""

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
    def predict_proba(self, X):
        self.model.eval()
        logits = self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == np.asarray(y)).mean()


@torch.no_grad()
def embed_latent(model_wrapper, X, batch_size=EMBED_BATCH_SIZE):
    model_wrapper.model.eval()
    parts = []
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        _, latent = model_wrapper.model(Xt[i:i + batch_size], return_latent=True)
        parts.append(latent.numpy())
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 128), dtype=np.float32)


# ---------------------------------------------------------------------------
# Poisoning (notebook: craft_boundary_pocket_poison)
# ---------------------------------------------------------------------------
def craft_boundary_pocket_poison(model, X, y, class_id, direction, n_points):
    """Poison the n_points samples of class_id closest to the decision
    boundary (lowest |P(class 1) - 0.5|), shifting each by `direction` -- a
    measured centroid-to-centroid vector, not a hand-picked constant. Labels
    unchanged (clean-label poisoning)."""
    candidates = np.where(y == class_id)[0]
    if n_points <= 0 or len(candidates) == 0:
        return X.copy(), np.array([], dtype=np.int64)
    proba = model.predict_proba(X[candidates])[:, 1]
    margin = np.abs(proba - 0.5)
    closest = candidates[np.argsort(margin)[:n_points]]
    X_pois = X.copy()
    X_pois[closest] = X[closest] + direction
    return X_pois, closest


def craft_task_poison(clean_lineage, X_train_scaled, y_train, benign_label, mal_label, poison_fraction):
    pocket_mal = X_train_scaled[y_train == mal_label].mean(axis=0)
    pocket_ben = X_train_scaled[y_train == benign_label].mean(axis=0)
    separating_direction = pocket_mal - pocket_ben

    n_ben = int((y_train == benign_label).sum())
    n_mal = int((y_train == mal_label).sum())
    n_poison_ben = int(poison_fraction * n_ben)
    n_poison_mal = int(poison_fraction * n_mal)

    X_p1, idx_poison_ben = craft_boundary_pocket_poison(
        clean_lineage, X_train_scaled, y_train, class_id=benign_label,
        direction=separating_direction, n_points=n_poison_ben,
    )
    X_train_poisoned, idx_poison_mal = craft_boundary_pocket_poison(
        clean_lineage, X_p1, y_train, class_id=mal_label,
        direction=-separating_direction, n_points=n_poison_mal,
    )
    return X_train_poisoned, idx_poison_ben, idx_poison_mal, separating_direction


# ---------------------------------------------------------------------------
# Detector (notebook: build_detector_training_set_mixed + DETECTOR_TYPE)
# ---------------------------------------------------------------------------
def build_detector_training_set_mixed(ref_model, X_pois, y, poison_idx_ben, poison_idx_mal,
                                       benign_label, mal_label, n_per_group,
                                       uncertain_fraction=0.5, seed=0):
    rng = np.random.default_rng(seed)
    clean_idx = np.setdiff1d(np.arange(len(y)), np.concatenate([poison_idx_ben, poison_idx_mal]))

    def select_group(pool_idx, n):
        pool_idx = np.asarray(pool_idx)
        n = min(n, len(pool_idx))
        n_uncertain = int(round(uncertain_fraction * n))
        n_random = n - n_uncertain
        proba = ref_model.predict_proba(X_pois[pool_idx])[:, 1]
        margin = np.abs(proba - 0.5)
        order = np.argsort(margin)
        uncertain_pick = pool_idx[order[:n_uncertain]]
        remaining = np.setdiff1d(pool_idx, uncertain_pick)
        random_pick = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
        return np.concatenate([uncertain_pick, random_pick])

    clean_ben = select_group(clean_idx[y[clean_idx] == benign_label], n_per_group)
    clean_mal = select_group(clean_idx[y[clean_idx] == mal_label], n_per_group)
    pois_ben = select_group(poison_idx_ben, n_per_group)
    pois_mal = select_group(poison_idx_mal, n_per_group)

    idx_all = np.concatenate([clean_ben, clean_mal, pois_ben, pois_mal])
    is_poisoned = np.concatenate([
        np.zeros(len(clean_ben)), np.zeros(len(clean_mal)),
        np.ones(len(pois_ben)), np.ones(len(pois_mal)),
    ]).astype(int)
    composition = {
        "clean_benign": len(clean_ben), "clean_malicious": len(clean_mal),
        "poisoned_benign": len(pois_ben), "poisoned_malicious": len(pois_mal),
    }
    return X_pois[idx_all], is_poisoned, composition


def train_detector(detector_type, X_det, y_det, seed):
    if detector_type == "xgboost" and not _XGBOOST_AVAILABLE:
        print("  [detector] xgboost not installed -- falling back to logistic regression.")
        detector_type = "logistic"
    if detector_type == "logistic":
        detector = LogisticRegression(random_state=seed, **LOGISTIC_PARAMS).fit(X_det, y_det)
    elif detector_type == "xgboost":
        detector = XGBClassifier(random_state=seed, **XGBOOST_PARAMS).fit(X_det, y_det)
    else:
        raise ValueError(f"Unknown detector_type: {detector_type!r}")
    return detector, detector_type


def run_detector(poisoned_baseline, X_train_poisoned, y_train, idx_poison_ben, idx_poison_mal,
                  benign_label, mal_label, detector_type, seed):
    X_det, y_det, composition = build_detector_training_set_mixed(
        poisoned_baseline, X_train_poisoned, y_train, idx_poison_ben, idx_poison_mal,
        benign_label, mal_label, n_per_group=DETECTOR_N_PER_GROUP,
        uncertain_fraction=DETECTOR_UNCERTAIN_FRACTION, seed=seed,
    )
    detector, detector_type_used = train_detector(detector_type, X_det, y_det, seed)
    train_acc = (detector.predict(X_det) == y_det).mean()

    detected_by_class = {}
    class_metrics = {}
    for cls, true_poison_this_class in [(benign_label, idx_poison_ben), (mal_label, idx_poison_mal)]:
        class_indices = np.where(y_train == cls)[0]
        preds = detector.predict(X_train_poisoned[class_indices])
        flagged = class_indices[preds == 1]
        detected_by_class[cls] = flagged
        tp = len(np.intersect1d(flagged, true_poison_this_class))
        precision = tp / max(len(flagged), 1)
        recall = tp / max(len(true_poison_this_class), 1)
        class_metrics[cls] = {"n_flagged": len(flagged), "tp": tp, "precision": precision, "recall": recall}

    detected_poison_idx = np.concatenate([detected_by_class[benign_label], detected_by_class[mal_label]])
    metrics = {
        "detector_type": detector_type_used,
        "train_accuracy": float(train_acc),
        "composition": composition,
        "class_metrics": class_metrics,
        "n_detected": len(detected_poison_idx),
        "n_oracle": len(idx_poison_ben) + len(idx_poison_mal),
    }
    return detected_poison_idx, detected_by_class, metrics


# ---------------------------------------------------------------------------
# Unlearning variants (notebook cell 18)
# ---------------------------------------------------------------------------
def apply_dropped_rows(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y):
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_poison_idx] = False
    lineage.adapt(X_train_poisoned[mask_keep], y_train[mask_keep],
                  replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)


def apply_amnesiac(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y,
                    benign_label, mal_label):
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_poison_idx] = False
    X_clean_only, y_clean_only = X_train_poisoned[mask_keep], y_train[mask_keep]

    X_flagged = X_train_poisoned[detected_poison_idx]
    if len(X_flagged) == 0:
        lineage.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)
        return
    X_flagged_dup = np.vstack([X_flagged, X_flagged])
    y_flagged_dup = np.concatenate([
        np.full(len(X_flagged), benign_label), np.full(len(X_flagged), mal_label),
    ])
    for _ in range(AMNESIAC_ROUNDS):
        lineage.adapt(X_flagged_dup, y_flagged_dup, epochs=1)
        lineage.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=1)


def apply_opposite_class(lineage, X_train_poisoned, y_train, detected_poison_idx, replay_X, replay_y,
                          benign_label, mal_label):
    y_relabel = y_train.copy()
    y_relabel[detected_poison_idx] = benign_label + mal_label - y_train[detected_poison_idx]
    lineage.adapt(X_train_poisoned, y_relabel, replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)


# ---------------------------------------------------------------------------
# Test-time "genuine pocket" attack (notebook cell 16, batched dual-gradient)
# ---------------------------------------------------------------------------
def adversarial_attack_pocket(poisoned_model, clean_model, X, y, epsilon_max,
                               step=ATTACK_STEP, max_steps=ATTACK_MAX_STEPS,
                               clean_weight=ATTACK_CLEAN_WEIGHT):
    X_adv = X.copy()
    x0 = X.copy()
    success = np.zeros(len(X), dtype=bool)
    active = np.ones(len(X), dtype=bool)

    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.where(active)[0]
        Xa = X_adv[idx]
        y_active_np = y[idx]
        y_active = torch.tensor(y_active_np, dtype=torch.long)

        poisoned_model.model.eval()
        xl_p = torch.tensor(Xa, dtype=torch.float32, requires_grad=True)
        proba_p = torch.softmax(poisoned_model.model(xl_p), dim=1)
        proba_p.gather(1, y_active.unsqueeze(1)).squeeze(1).sum().backward()
        grad_p = xl_p.grad.numpy()
        pred_p = proba_p.detach().numpy().argmax(axis=1)

        clean_model.model.eval()
        xl_c = torch.tensor(Xa, dtype=torch.float32, requires_grad=True)
        proba_c = torch.softmax(clean_model.model(xl_c), dim=1)
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
            nrm = np.linalg.norm(d, axis=1, keepdims=True)
            nrm[nrm < 1e-8] = 1e-8
            d = d / nrm
            X_adv[idx_s] = X_adv[idx_s] + step * d
            delta = X_adv[idx_s] - x0[idx_s]
            dn = np.linalg.norm(delta, axis=1, keepdims=True)
            too_far = dn.ravel() > epsilon_max
            if too_far.any():
                X_adv[idx_s[too_far]] = x0[idx_s[too_far]] + delta[too_far] / dn[too_far] * epsilon_max

    return X_adv, success, np.linalg.norm(X_adv - x0, axis=1)


def typical_class_gap(X, y, benign_label, mal_label):
    if len(np.unique(y)) < 2:
        return None
    pocket_mal = X[y == mal_label].mean(axis=0)
    return float(np.linalg.norm(X[y == benign_label] - pocket_mal, axis=1).mean())


# ---------------------------------------------------------------------------
# Pooled / mean / per-task accuracy (notebook cell 23)
# ---------------------------------------------------------------------------
def pooled_and_per_task_accuracy(model, all_test_sets):
    per_task = {}
    X_all, y_all = [], []
    for tid, (Xh, yh) in all_test_sets.items():
        acc = model.score(Xh, yh)
        per_task[tid] = (acc, len(yh))
        X_all.append(Xh)
        y_all.append(yh)
    X_pooled = np.vstack(X_all)
    y_pooled = np.concatenate(y_all)
    pooled_acc = model.score(X_pooled, y_pooled)
    mean_acc = float(np.mean([acc for acc, n in per_task.values()]))
    return pooled_acc, mean_acc, per_task


# ---------------------------------------------------------------------------
# Shared replay buffer (fill/select only -- called ONCE per task, AFTER all
# adaptation + unlearning is finished; reuses this repo's existing anomaly+
# inlier IsolationForest selection so exemplars from earlier tasks can
# survive across updates).
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
                          benign_label, mal_label, mem_size=MEM_SIZE, contamination=BUFFER_CONTAMINATION):
    X_np = np.asarray(X_clean, dtype=np.float32)
    Y_np = np.asarray(y_clean)
    cat_np = np.asarray(category_clean, dtype=object)
    id_np = np.asarray(sample_id_clean)
    L_np = embed_latent_np(embed_model, X_np)
    latent_dim = L_np.shape[1]
    budget_per_label = mem_size // 2

    for lbl in (benign_label, mal_label):
        old_entries = label_buffers.get(lbl, [])
        if old_entries:
            old_X = np.stack([e[0] for e in old_entries]).astype(np.float32)
            old_Y = np.array([e[1] for e in old_entries], dtype=Y_np.dtype)
            old_cat = np.array([e[2] for e in old_entries], dtype=object)
            old_id = np.array([e[3] for e in old_entries])
            old_L = embed_latent_np(embed_model, old_X)
        else:
            old_X = np.empty((0, X_np.shape[1]), dtype=np.float32)
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


def _fmt_report(model, X, y):
    return classification_report(y, model.predict(X), target_names=["Benign", "Malicious"],
                                  zero_division=0, digits=3)


def write_task_log(log_path, t, sections):
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 78}\n=== Task {t} ===\n{'=' * 78}\n")
        for title, body in sections:
            f.write(f"\n-- {title} --\n{body}\n")


# ---------------------------------------------------------------------------
# PCA correctness plots (notebook cell 20) -- final task only
# ---------------------------------------------------------------------------
def plot_correctness_grid(out_path, pca_fit, panels):
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, model, X, y) in zip(axes, panels):
        proj = pca_fit.transform(X)
        correct = model.predict(X) == y
        for cls, color, name in [(0, "#1f3d7a", "class0"), (1, "#a34a12", "class1")]:
            m_c = (y == cls) & correct
            m_w = (y == cls) & ~correct
            ax.scatter(proj[m_c, 0], proj[m_c, 1], s=10, c=color, label=f"{name} correct")
            ax.scatter(proj[m_w, 0], proj[m_w, 1], s=40, facecolors="none", edgecolors=color,
                       linewidths=1.4, label=f"{name} WRONG")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="best")
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
    ap.add_argument("--log_name", type=str, default="madar_pocket_pipeline_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    ap.add_argument("--detector_type", type=str, default="xgboost", choices=["xgboost", "logistic"])
    ap.add_argument("--poison_fraction", type=float, default=POISON_FRACTION)
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks >= "
                          f"{BREAKPOINT_FROM_TASK} (needed for a non-interactive/headless run).")
    args = ap.parse_args()

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(RUNS_BASE_DIR, "madar_pocket_pipeline", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG\n"
            "=========================\n"
            "5 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "dropped_rows, amnesiac, opposite_class. One shared detector + one shared\n"
            "replay buffer per task (see module docstring). Task 0 is plain initial\n"
            "training only -- no poisoning/detection/unlearning yet.\n"
        )

    print(f"Loading {args.h5_path} and building {NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(args.h5_path, TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    scaler = None
    lineages = {}
    label_buffers = {}
    replay_buffer = []
    task_test_splits = {}
    task_test_gids = {}
    results = []

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)

    LINEAGE_NAMES = ["clean", "poisoned_baseline", "dropped_rows", "amnesiac", "opposite_class"]
    FIX_NAMES = ["dropped_rows", "amnesiac", "opposite_class"]

    for t in range(NUM_TASKS):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        task = tasks[t]
        X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y_all = task["labels"].astype(np.int64)
        gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

        X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
            X_raw, y_all, gid_all, test_size=TASK_TEST_FRAC, random_state=SEED, stratify=y_all,
        )

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = to_scaled(X_train_raw)
            X_test_scaled = to_scaled(X_test_raw)

            base_model = ClassifierNN(feature_dim, 2).to(DEVICE)
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

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            sample_id = gid_train
            category = np.where(y_train == benign_label, "benign", "malicious_clean")
            replay_buffer = update_shared_buffer(
                lineages["poisoned_baseline"], label_buffers, X_train_scaled, y_train, category, sample_id,
                benign_label, mal_label,
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
            unlearn_section = "N/A -- task 0 has no prior model to poison against."

            write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Unlearning step", unlearn_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": SEED, "feature_dim": feature_dim, "scaler": scaler,
                "label_mapping": label_mapping,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                "label_buffers": label_buffers, "replay_buffer": replay_buffer,
                "task_test_splits": task_test_splits, "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
            }, checkpoint_path)
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> detect -> unlearn, every task.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        replay_X, replay_y = flatten_replay(replay_buffer)  # buffer as it stood at end of PREVIOUS task

        # Step 2: clean lineage adapts on clean data only.
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=replay_X, replay_y=replay_y,
                                 epochs=CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison (shared across all poisoned lineages).
        X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = craft_task_poison(
            lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
        )
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix").
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train, replay_X=replay_X, replay_y=replay_y,
                                             epochs=ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )

        # Step 5: ONE shared detector per task, trained from poisoned_baseline.
        detected_poison_idx, detected_by_class, det_metrics = run_detector(
            lineages["poisoned_baseline"], X_train_poisoned, y_train, idx_poison_ben, idx_poison_mal,
            benign_label, mal_label, args.detector_type, SEED,
        )

        # Step 6: each fix lineage adapts on poisoned data from ITS OWN prior
        # weights first (this lineage's own "just got poisoned" state), then
        # applies its own fix to the SAME shared detected_poison_idx.
        for name in FIX_NAMES:
            lineages[name].adapt(X_train_poisoned, y_train, replay_X=replay_X, replay_y=replay_y,
                                  epochs=ADAPT_EPOCHS)
        apply_dropped_rows(lineages["dropped_rows"], X_train_poisoned, y_train, detected_poison_idx,
                            replay_X, replay_y)
        apply_amnesiac(lineages["amnesiac"], X_train_poisoned, y_train, detected_poison_idx,
                        replay_X, replay_y, benign_label, mal_label)
        apply_opposite_class(lineages["opposite_class"], X_train_poisoned, y_train, detected_poison_idx,
                              replay_X, replay_y, benign_label, mal_label)

        # Step 7: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean) -- matches the notebook: the
        # same crafted points are then just re-scored under every lineage.
        eps_this_task = typical_class_gap(X_test_scaled, y_test, benign_label, mal_label) * ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test, epsilon_max=eps_this_task,
        )

        # Step 8: spillover check -- re-attack every PRIOR task's test set,
        # every task, same poisoned_baseline/clean reference pair.
        historical_adv = {}
        for s, (Xs_raw, ys) in task_test_splits.items():
            Xs_scaled = to_scaled(Xs_raw)
            eps_s = typical_class_gap(Xs_scaled, ys, benign_label, mal_label)
            eps_s = eps_s * ATTACK_EPS_MULTIPLIER if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = adversarial_attack_pocket(
                lineages["poisoned_baseline"], lineages["clean"], Xs_scaled, ys, epsilon_max=eps_s,
            )
            historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)

        # Step 9: pooled/mean/per-class accuracy across all clean + adversarial
        # test sets seen so far, for every lineage (clean included).
        all_clean_sets = {s: (to_scaled(Xs_raw), ys) for s, (Xs_raw, ys) in task_test_splits.items()}
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pooled_results, mean_results, per_class_reports = {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, _ = pooled_and_per_task_accuracy(lineages[name], all_test_sets_full)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_class_reports[name] = _fmt_report(lineages[name], X_test_scaled, y_test)

        still_evades = {}
        for name in FIX_NAMES:
            pred = lineages[name].predict(X_test_adv)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 10: update the SHARED replay buffer -- only now, after every
        # lineage's adaptation/unlearning for this task is fully done.
        clean_mask = np.ones(len(X_train_poisoned), dtype=bool)
        clean_mask[detected_poison_idx] = False
        category = np.where(y_train == benign_label, "benign", "malicious_clean")
        replay_buffer = update_shared_buffer(
            lineages["poisoned_baseline"], label_buffers,
            X_train_poisoned[clean_mask], y_train[clean_mask], category[clean_mask], gid_train[clean_mask],
            benign_label, mal_label,
        )

        task_test_splits[t] = (X_test_raw, y_test)
        task_test_gids[t] = gid_test

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        n_ben_train = int((y_train == benign_label).sum())
        n_mal_train = int((y_train == mal_label).sum())
        n_flagged_ben = det_metrics["class_metrics"][benign_label]["n_flagged"]
        n_flagged_mal = det_metrics["class_metrics"][mal_label]["n_flagged"]
        train_section = (
            f"malicious: {n_mal_train}, benign: {n_ben_train}\n"
            f"malicious_perturbed (oracle): {len(idx_poison_mal)}, "
            f"benign_perturbed (oracle): {len(idx_poison_ben)}\n"
            f"detector-flagged malicious: {n_flagged_mal}, detector-flagged benign: {n_flagged_ben}\n"
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
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({_fmt_pct(succ_pocket.mean())})\n"
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
        adapt_section = "\n".join(adapt_lines)

        dist = buffer_distribution(label_buffers)
        unlearn_lines = [
            f"detector type: {det_metrics['detector_type']}, train accuracy: {det_metrics['train_accuracy']:.3f}",
            f"detector precision/recall vs oracle -- benign: "
            f"P={det_metrics['class_metrics'][benign_label]['precision']:.2f} "
            f"R={det_metrics['class_metrics'][benign_label]['recall']:.2f}, malicious: "
            f"P={det_metrics['class_metrics'][mal_label]['precision']:.2f} "
            f"R={det_metrics['class_metrics'][mal_label]['recall']:.2f}",
            f"detector training composition: {det_metrics['composition']}",
            f"replay buffer distribution (post-update, this task): {dist}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}",
        ]
        for name in FIX_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adv_acc_name = lineages[name].score(X_test_adv, y_test)
            unlearn_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        unlearn_lines.append("")
        for name in FIX_NAMES:
            unlearn_lines.append(f"[{name}] classification report (this task's clean test):")
            unlearn_lines.append(per_class_reports[name])
        unlearn_section = "\n".join(unlearn_lines)

        write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Unlearning step", unlearn_section),
        ])

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()), "detector_metrics": det_metrics,
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": SEED, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            "label_buffers": label_buffers, "replay_buffer": replay_buffer,
            "task_test_splits": task_test_splits, "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
        }, checkpoint_path)

        print(f"Task {t} done. Genuine pocket rate: {_fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")

        # Final task: PCA correctness plots.
        if t == NUM_TASKS - 1:
            pca_fit = PCA(n_components=2, random_state=SEED).fit(X_train_scaled)
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            plot_correctness_grid(os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels)

        if not args.no_breakpoint and t >= BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, `label_buffers`, "
                  f"or the log at {log_path}. Continue with `c`.")
            breakpoint()

    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
