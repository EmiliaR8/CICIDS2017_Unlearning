"""
task0_9_pipeline_4arch.py

Script port of notebooks/task0-9_pipeline_4arch-updatedmeta.ipynb ("Tasks 0-9: poison / adapt /
pocket-attack / oracle vs. meta_al detection / unlearn+TTA -- full pipeline, 4 architectures").

Every notebook code cell is copied VERBATIM below (marked "notebook cell <id>"); nothing in the
pipeline logic is changed. The only additions are:
  * a CLI (paths, SEED, which architecture, how many tasks) applied right after the config cell,
    replacing the notebook's hard-coded H5_PATH / LOGS_DIR / SEED / output dirs;
  * a timing harness (bottom of file) that runs Task 0 .. Task N-1 for ONE architecture, records
    wall-clock per task and per phase (attacks, detectors, unlearning, TTA), writes
    <results_dir>/timing_seed<SEED>_depth<D>_tasks<N>.json, and extrapolates the full
    10-task x 4-architecture run.

To time "one task": --num_tasks 2 runs Task 0 (from-scratch training, required as the starting point)
and Task 1 (the first full poison/attack/detect/unlearn/TTA task).

NOTE ON GPU: the notebook defines DEVICE = cuda but never moves any model or tensor to it, and
XGBoost runs on CPU by default -- the pipeline runs entirely on CPU. The harness reports this.
Requires checkpoint files classifier_checkpoint_task{0..N-1}.pt (per-task, ids-only replay buffers,
as written by madar_pocket_pipeline_meta_detect.py / madar_pocket_pipeline_ssf.py) in --logs_dir.
"""
import argparse as _argparse
import time as _time

_ap = _argparse.ArgumentParser(description="Timing run of the Task 0-9 / 4-architecture notebook pipeline")
_ap.add_argument("--h5", required=True, help="path to subsampled_dataset.h5 (notebook's H5_PATH)")
_ap.add_argument("--logs_dir", required=True,
                 help="dir with classifier_checkpoint_task{t}.pt files (notebook's LOGS_DIR)")
_ap.add_argument("--seed", type=int, default=100, help="notebook's SEED (default 100, as in the notebook)")
_ap.add_argument("--depth", type=int, default=4, choices=[1, 2, 3, 4],
                 help="architecture to run: HIDDEN_SIZES_BY_DEPTH[depth] (default 4 = largest)")
_ap.add_argument("--num_tasks", type=int, default=2,
                 help="run Task 0..num_tasks-1 (default 2 = Task 0 + Task 1); 10 = the full run")
_ap.add_argument("--results_dir", default="./results_datasensitivity")
_ap.add_argument("--model_dir", default="./saved_models_datasensitivity")
_args = _ap.parse_args()
if not 1 <= _args.num_tasks <= 10:
    _ap.error("--num_tasks must be in [1, 10]")
_T_START = _time.perf_counter()

# ===================== notebook cell 3b5888ed (config) =====================
import numpy as np
import torch
import torch.nn as nn
import copy
import json
import h5py
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd

# ---- EDIT THESE PATHS ----
H5_PATH = "subsampled_dataset.h5"
LOGS_DIR = Path("/home/aritran/ICLR/logs")
# ---------------------------

TASK_FRACTIONS = [
    0.3000, 0.0918, 0.0883, 0.0848, 0.0813,
    0.0778, 0.0743, 0.0708, 0.0673, 0.0638,
]
N_TASKS = len(TASK_FRACTIONS)

# ---- CHANGE THIS TO RUN THE WHOLE PIPELINE AGAIN WITH DIFFERENT RANDOMNESS ----
# task_seed = SEED -- the SAME seed is used for every task and every architecture (no per-task or
# per-step derivation). torch's global RNG is seeded once per task with task_seed and then left to
# evolve naturally through every step that follows (adaptation, unlearning, TTA -- no reseeding in
# between), and one shared np.random.Generator(task_seed) is passed into the detectors for their numpy
# randomness (including drawing the XGBoost random_state via rng.integers(...)). Bump SEED and rerun
# for a fully independent realization of the whole run.
SEED = 100
# ---------------------------------------------------------------------------------

POISON_FRACTION = 0.3
ADAPT_EPOCHS = 15
CLEAN_ADAPT_EPOCHS = 5
TASK0_EPOCHS = 30   # Task 0 trains from scratch (no prior model to continue from), so it gets more
                    # epochs than the incremental CLEAN_ADAPT_EPOCHS/ADAPT_EPOCHS used from Task 1 on

N_OUTER_EPISODES = 8
N_INNER_STEPS = 3
SAMPLES_PER_INNER_STEP = 4

N_PER_GROUP = 12
UNCERTAIN_FRACTION = 0.3

# NOTE: no random_state baked in here -- every XGBClassifier(...) call below passes its own seed
# drawn from the task's shared rng, instead of every task silently sharing a fixed random_state.
XGBOOST_PARAMS_BASE = dict(n_estimators=20, max_depth=2, learning_rate=0.1, reg_lambda=10.0,
                            subsample=0.7, colsample_bytree=0.3, min_child_weight=5,
                            eval_metric="logloss")

def xgb_params(seed):
    return dict(**XGBOOST_PARAMS_BASE, random_state=seed)

CALIB_FRAC = 0.02   # fraction of each task's genuine pockets held out for the calibration/TTA probe

# ---- architecture variants: 1/2/3/4 hidden layers, each a full independent Task 0-9 run ----
HIDDEN_SIZES_BY_DEPTH = {
    1: (512,),
    2: (512, 256),
    3: (512, 256, 128),
    4: (1024, 512, 256, 128),   # matches the real checkpoints' architecture
}

MODEL_SAVE_DIR_BASE = Path("./saved_models_datasensitivity")     # models saved under MODEL_SAVE_DIR_BASE/seed{SEED}_depth{depth}/
RESULTS_DIR = Path("./results_datasensitivity")                  # <-- EDIT to your actual target folder
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", DEVICE)


# ===================== CLI overrides (replace the notebook's hard-coded values) =====================
H5_PATH = _args.h5
LOGS_DIR = Path(_args.logs_dir)
SEED = _args.seed
MODEL_SAVE_DIR_BASE = Path(_args.model_dir)
RESULTS_DIR = Path(_args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================== notebook cell 3dcc8b2c =====================
with h5py.File(H5_PATH, "r") as f:
    features = f["features"][:]
    label = f["label"][:]

N = features.shape[0]
print("features:", features.shape, "label:", label.shape)


# ===================== notebook cell ff2838a9 =====================
def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def get_ckpts_through(task_id, ckpts_all):
    """Loads (and caches) whatever checkpoints 0..task_id are needed. `ckpts_all` is a dict you
    keep around across task cells so earlier checkpoints aren't re-read from disk each time."""
    for t in range(task_id + 1):
        if t not in ckpts_all:
            p = LOGS_DIR / f"classifier_checkpoint_task{t}.pt"
            if not p.exists():
                raise FileNotFoundError(f"need {p} to reconstruct through task {task_id}")
            ckpts_all[t] = load_ckpt(p)
    return ckpts_all


def reconstruct_task_gids(max_task_id, ckpts):
    """Generalized version of the single-task boundary-correction logic (validated earlier on
    Task 1 alone): returns (task_train_gids, task_test_gids_final, INDEX_OFFSET) with keys 0..max_task_id.
    Only needs classifier_checkpoint_task{0..max_task_id}.pt, exactly like the original."""
    test_gids = {}
    for t, ck in ckpts.items():
        for task_id, gids in ck["task_test_gids"].items():
            gids = np.asarray(gids)
            if task_id not in test_gids or len(gids) > len(test_gids[task_id]):
                test_gids[task_id] = gids

    cum = np.cumsum(TASK_FRACTIONS)
    frac_upper = np.round(cum * N).astype(int)
    frac_upper[-1] = N
    frac_lower = np.concatenate([[0], frac_upper[:-1]])

    def out_of_bounds_count(offset):
        total = 0
        for g in test_gids.values():
            gi = g - offset
            total += ((gi < 0) | (gi >= N)).sum()
        return total

    def seam_mismatch_count(offset):
        total = 0
        for t, g in test_gids.items():
            gi = g - offset
            bad = (gi < frac_lower[t]) | (gi >= frac_upper[t])
            total += bad.sum()
        return total

    candidates = [0, 1]
    oob = {o: out_of_bounds_count(o) for o in candidates}
    seam = {o: seam_mismatch_count(o) for o in candidates}
    feasible = [o for o in candidates if oob[o] == 0]
    INDEX_OFFSET = min(feasible, key=lambda o: seam[o]) if feasible else min(candidates, key=lambda o: oob[o])
    print(f"  INDEX_OFFSET = {INDEX_OFFSET}  (out-of-bounds counts: {oob})")

    test_idx = {t: np.asarray(g) - INDEX_OFFSET for t, g in test_gids.items()}

    lower = frac_lower.copy()
    upper = frac_upper.copy()
    for t in range(1, N_TASKS):
        lo_side, hi_side = t - 1, t
        if lo_side not in test_idx or hi_side not in test_idx:
            continue
        constraint_lo = int(test_idx[lo_side].max()) + 1
        constraint_hi = int(test_idx[hi_side].min())
        if constraint_lo > constraint_hi:
            print(f"  !! seam {lo_side}/{hi_side}: irreconcilable -- investigate")
            continue
        boundary = min(max(frac_upper[lo_side], constraint_lo), constraint_hi)
        upper[lo_side] = boundary
        lower[hi_side] = boundary
    upper[-1] = N
    lower[0] = 0

    task_pool_idx = {t: set(range(int(lower[t]), int(upper[t]))) for t in range(N_TASKS)}

    task_train_gids, task_test_gids_final = {}, {}
    for t in range(max_task_id + 1):
        test_set = set(test_idx[t].tolist())
        missing = test_set - task_pool_idx[t]
        if missing:
            print(f"  task {t}: dropping {len(missing)} test indices outside computed pool")
            test_set -= missing
        train_set = task_pool_idx[t] - test_set
        task_train_gids[t] = np.array(sorted(train_set))
        task_test_gids_final[t] = np.array(sorted(test_set))
        print(f"  task {t}: train={len(train_set)} test={len(test_set)}")

    return task_train_gids, task_test_gids_final, INDEX_OFFSET


def rows_for(gids):
    idx = np.asarray(sorted(gids))
    return features[idx], label[idx], idx


def reconstruct_replay_buffer(buffer_list, offset):
    labels, categories, gids = zip(*buffer_list)
    gids = np.array(gids)
    idx = gids - offset
    X = features[idx].astype("float32")
    y = np.array(labels)
    categories = np.array(categories)
    n_poison = sum("perturbed" in c for c in categories)
    if n_poison:
        print(f"  note: {n_poison}/{len(buffer_list)} replay-buffer entries are *_perturbed -- "
              f"their X is the PRE-poison original feature vector, not the true perturbed one")
    return X, y, categories, gids


# ===================== notebook cell 6c92dd24 =====================
class ClassifierNN(nn.Module):
    """Matches the REAL checkpoints' key naming (layers.0-3 / bns.0-3 / fc_last) for the 4-layer
    architecture; for the 1/2/3-layer variants the same naming scheme just has fewer entries."""
    def __init__(self, input_dim, num_classes, hidden_sizes=(1024, 512, 256, 128)):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes)
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_sizes))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(h) for h in hidden_sizes])
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        for layer, bn in zip(self.layers, self.bns):
            x = self.relu(bn(layer(x)))
        return self.fc_last(x)


class AdaptableClassifier:
    def __init__(self, torch_model, lr=1e-4, weight_decay=1e-5):
        self.model = torch_model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, X, y, replay_X=None, replay_y=None, epochs=5, batch_size=128):
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.long)
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
        logits = self.model(torch.as_tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ===================== notebook cell 52e3ba60 =====================
def craft_boundary_pocket_poison(model, X, y, class_id, direction, n_points):
    candidates = np.where(y == class_id)[0]
    proba = model.predict_proba(X[candidates])[:, 1]
    margin = np.abs(proba - 0.5)
    closest = candidates[np.argsort(margin)[:n_points]]
    X_pois = X.copy()
    X_pois[closest] = X[closest] + direction
    return X_pois, closest


def adversarial_attack_pocket(poisoned_model, clean_model, X, y, epsilon_max,
                               step=0.04, max_steps=50, clean_weight=1.0, verbose_every=1000):
    X_adv = X.copy()
    x0 = X.copy()
    success = np.zeros(len(X), dtype=bool)
    active = np.ones(len(X), dtype=bool)

    for s in range(max_steps):
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

        if (s + 1) % verbose_every == 0 or s == max_steps - 1:
            print(f"    step {s+1}/{max_steps}: {success.sum()} succeeded so far, {active.sum()} still active")

    return X_adv, success, np.linalg.norm(X_adv - x0, axis=1)


def honest_adv(X_scaled, X_adv, succ):
    """Perturbed features only where the attack genuinely succeeded; clean everywhere else."""
    out = X_scaled.copy()
    out[succ] = X_adv[succ]
    return out


# ===================== notebook cell 04e391ed =====================
def build_detector_training_set_mixed(ref_model, X_pois, y, poison_idx_A, poison_idx_B,
                                        n_per_group, uncertain_fraction, rng):
    """`rng` is the ONE shared np.random.Generator for the whole task -- pass the same object in
    everywhere rather than re-deriving a fresh generator from a seed, so randomness continues to
    evolve across every randomized step in the task instead of each step restarting from a fixed
    state."""
    clean_idx = np.setdiff1d(np.arange(len(y)), np.concatenate([poison_idx_A, poison_idx_B]))

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

    clean_benign = select_group(clean_idx[y[clean_idx] == 0], n_per_group)
    clean_malicious = select_group(clean_idx[y[clean_idx] == 1], n_per_group)
    pois_benign = select_group(poison_idx_A, n_per_group)
    pois_malicious = select_group(poison_idx_B, n_per_group)

    idx_all = np.concatenate([clean_benign, clean_malicious, pois_benign, pois_malicious])
    is_poisoned = np.concatenate([
        np.zeros(len(clean_benign)), np.zeros(len(clean_malicious)),
        np.ones(len(pois_benign)), np.ones(len(pois_malicious)),
    ]).astype(int)
    return X_pois[idx_all], is_poisoned, idx_all


def run_oracle_detector(poisoned_model, X_train_poisoned, y_train, idx_pocket_A, idx_pocket_B, rng):
    """`rng` is the task's one shared Generator -- used both for sample-selection (via
    build_detector_training_set_mixed) and to draw the XGBoost model's random_state (XGBoost needs a
    concrete int, so we pull one from the shared stream with rng.integers(...) rather than resetting
    anything to a fixed seed)."""
    X_det, y_det, _ = build_detector_training_set_mixed(
        poisoned_model, X_train_poisoned, y_train, idx_pocket_A, idx_pocket_B,
        n_per_group=N_PER_GROUP, uncertain_fraction=UNCERTAIN_FRACTION, rng=rng,
    )
    detector = XGBClassifier(**xgb_params(int(rng.integers(0, 2**31 - 1)))).fit(X_det, y_det)

    detected = []
    for class_id, true_poison_this_class in [(0, idx_pocket_A), (1, idx_pocket_B)]:
        class_indices = np.where(y_train == class_id)[0]
        preds = detector.predict(X_train_poisoned[class_indices])
        flagged = class_indices[preds == 1]
        tp = len(np.intersect1d(flagged, true_poison_this_class))
        precision = tp / max(len(flagged), 1)
        recall = tp / max(len(true_poison_this_class), 1)
        print(f"    [oracle class={class_id}] flagged {len(flagged)}; precision={precision:.2f} recall={recall:.2f}")
        detected.append(flagged)
    detected_poison_idx = np.concatenate(detected)
    print(f"    oracle total detected: {len(detected_poison_idx)}")
    return detected_poison_idx


def get_latent_features(model, X, batch_size=512):
    model.eval()
    captured = {}
    def _hook(module, inp, out):
        captured["z"] = out.detach()
    handle = model.bns[-1].register_forward_hook(_hook)
    latents = []
    with torch.no_grad():
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        for i in range(0, len(Xt), batch_size):
            _ = model(Xt[i:i + batch_size])
            latents.append(torch.relu(captured["z"]).numpy())
    handle.remove()
    return np.concatenate(latents, axis=0)


def replay_knn_distance(X, y, replay_X_np, replay_y_np, k=5):
    dist = np.full(len(X), np.nan)
    for c in np.unique(y):
        mask = y == c
        replay_mask = replay_y_np == c
        if replay_mask.sum() == 0:
            continue
        kk = min(k, replay_mask.sum())
        nn = NearestNeighbors(n_neighbors=kk).fit(replay_X_np[replay_mask])
        d, _ = nn.kneighbors(X[mask])
        dist[mask] = d.mean(axis=1)
    return dist


def fit_density_ratio_model(X_current, replay_X_np):
    Xd = np.vstack([replay_X_np, X_current])
    yd = np.concatenate([np.zeros(len(replay_X_np)), np.ones(len(X_current))])
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0).fit(Xd, yd)


def density_ratio_score(clf, X):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return p / (1 - p)


def extract_meta_features(adaptable_model, X, y, replay_X_np, replay_y_np, k=5):
    net = adaptable_model.model
    net.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    latent = get_latent_features(net, X)

    iso_raw = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
    iso_latent = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(latent)
    density_clf = fit_density_ratio_model(X, replay_X_np)

    iso_raw_score = -iso_raw.score_samples(X)
    iso_latent_score = -iso_latent.score_samples(latent)
    density_ratio = density_ratio_score(density_clf, X)

    with torch.no_grad():
        logits = net(Xt)
        proba = torch.softmax(logits, dim=1)
        per_sample_loss = nn.functional.cross_entropy(logits, yt, reduction="none").numpy()
        p = proba.numpy()
        eps_ = 1e-12
        entropy = -(p * np.log(p + eps_)).sum(axis=1)
        sorted_p = np.sort(p, axis=1)
        top2_margin = sorted_p[:, -1] - sorted_p[:, -2]

    dist_to_replay = replay_knn_distance(X, y, replay_X_np, replay_y_np, k=k)

    feats = np.column_stack([iso_raw_score, iso_latent_score, per_sample_loss, entropy, top2_margin,
                              density_ratio, dist_to_replay])
    return feats


def run_meta_al_detector(poisoned_model, X_train_poisoned, y_train, poison_idx, replay_X, replay_y, rng):
    """Active-learning episodic sampling: same ~250-label budget as the original random-sampling
    meta detector, but each inner step ALWAYS queries SAMPLES_PER_INNER_STEP // 2 points from the
    poison pool and the rest from the clean pool -- never letting the split drift -- picking within
    each pool the CURRENT detector's least-confident untouched points (once a detector is trainable).
    No test-time access, no borrowed oracle points. `rng` is the task's one shared Generator, same as
    run_oracle_detector."""
    replay_y_np = np.asarray(replay_y)
    feats_meta = extract_meta_features(poisoned_model, X_train_poisoned, y_train, replay_X, replay_y_np)
    is_poisoned_true = np.zeros(len(y_train), dtype=int)
    is_poisoned_true[poison_idx] = 1
    feats_meta_std = StandardScaler().fit_transform(feats_meta)

    poison_pool = poison_idx
    clean_pool = np.setdiff1d(np.arange(len(y_train)), poison_idx)

    n_p = SAMPLES_PER_INNER_STEP // 2
    n_c = SAMPLES_PER_INNER_STEP - n_p

    touched_al, labels_al = set(), {}

    seed_p = rng.choice(poison_pool, n_p, replace=False)
    seed_c = rng.choice(clean_pool, n_c, replace=False)
    for i in np.concatenate([seed_p, seed_c]):
        touched_al.add(int(i)); labels_al[int(i)] = int(is_poisoned_true[i])

    for ep in range(N_OUTER_EPISODES):
        for step in range(N_INNER_STEPS):
            idx_touched = np.array(sorted(touched_al))
            y_touched = np.array([labels_al[i] for i in idx_touched])

            untouched_poison = np.setdiff1d(poison_pool, idx_touched)
            untouched_clean = np.setdiff1d(clean_pool, idx_touched)

            if len(np.unique(y_touched)) < 2:
                # no AL detector trainable yet -- random draw within each pool, still split n_p/n_c
                idx_p = rng.choice(untouched_poison, min(n_p, len(untouched_poison)), replace=False)
                idx_c = rng.choice(untouched_clean, min(n_c, len(untouched_clean)), replace=False)
            else:
                al_detector = XGBClassifier(**xgb_params(int(rng.integers(0, 2**31 - 1)))).fit(
                    feats_meta_std[idx_touched], y_touched)
                # rank uncertainty SEPARATELY within each pool so the split stays fixed regardless of
                # which class the detector currently finds more ambiguous overall
                margin_p = np.abs(al_detector.predict_proba(feats_meta_std[untouched_poison])[:, 1] - 0.5)
                margin_c = np.abs(al_detector.predict_proba(feats_meta_std[untouched_clean])[:, 1] - 0.5)
                idx_p = untouched_poison[np.argsort(margin_p)[:min(n_p, len(untouched_poison))]]
                idx_c = untouched_clean[np.argsort(margin_c)[:min(n_c, len(untouched_clean))]]

            new_idx = np.concatenate([idx_p, idx_c])
            for i in new_idx:
                touched_al.add(int(i)); labels_al[int(i)] = int(is_poisoned_true[i])

    idx_touched_al = np.array(sorted(touched_al))
    y_touched_al = np.array([labels_al[i] for i in idx_touched_al])
    print(f"    active-learning budget used: {len(idx_touched_al)} labels "
          f"(poisoned: {y_touched_al.sum()}, clean: {(y_touched_al==0).sum()})")

    xgb_meta_al = XGBClassifier(**xgb_params(int(rng.integers(0, 2**31 - 1)))).fit(
        feats_meta_std[idx_touched_al], y_touched_al)

    query_mask_al = np.ones(len(is_poisoned_true), dtype=bool)
    query_mask_al[idx_touched_al] = False
    Xq_al, yq_al = feats_meta_std[query_mask_al], is_poisoned_true[query_mask_al]
    preds_al = xgb_meta_al.predict(Xq_al)
    scores_al = xgb_meta_al.predict_proba(Xq_al)[:, 1]
    prec_al, rec_al, f1_al, _ = precision_recall_fscore_support(yq_al, preds_al, average="binary", zero_division=0)
    auc_al = roc_auc_score(yq_al, scores_al) if len(np.unique(yq_al)) > 1 else float("nan")
    print(f"    [meta_al] held-out ({query_mask_al.sum()} pts): precision={prec_al:.3f} recall={rec_al:.3f} "
          f"f1={f1_al:.3f} auc={auc_al:.3f}")

    meta_al_detected = np.where(query_mask_al)[0][preds_al == 1]
    return meta_al_detected

def unlearn_dropped_rows(start_model, detected_idx, X_train_poisoned, y_train, replay_X, replay_y,
                          epochs=20):
    """Continues from `start_model` (the SAME lineage's own state from the previous task) -- trains
    on this task's own data with the flagged points dropped. No manual_seed call here -- torch's global
    RNG is seeded ONCE per task (in run_task_pipeline) and left to evolve naturally through every step
    that follows, rather than being reset before each one."""
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_idx] = False
    X_clean_only, y_clean_only = X_train_poisoned[mask_keep], y_train[mask_keep]
    model = AdaptableClassifier(copy.deepcopy(start_model.model))
    model.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=epochs)
    return model


def unlearn_amnesiac(start_model, detected_idx, X_train_poisoned, y_train, replay_X, replay_y,
                      epochs=20):
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[detected_idx] = False
    X_clean_only, y_clean_only = X_train_poisoned[mask_keep], y_train[mask_keep]
    X_flagged = X_train_poisoned[detected_idx]
    y_flagged = y_train[detected_idx]
    X_flagged_dup = np.vstack([X_flagged, X_flagged])
    y_flagged_dup = np.concatenate([np.zeros(len(X_flagged)), np.ones(len(X_flagged))])
    model = AdaptableClassifier(copy.deepcopy(start_model.model))
    for _ in range(epochs):
        model.adapt(X_flagged_dup, y_flagged_dup, epochs=1)
        model.adapt(X_clean_only, y_clean_only, replay_X=replay_X, replay_y=replay_y, epochs=1)
    return model


def unlearn_opposite_class(start_model, detected_idx, X_train_poisoned, y_train, replay_X, replay_y,
                            epochs=20):
    y_relabel = y_train.copy()
    y_relabel[detected_idx] = 1 - y_train[detected_idx]
    model = AdaptableClassifier(copy.deepcopy(start_model.model))
    model.adapt(X_train_poisoned, y_relabel, replay_X=replay_X, replay_y=replay_y, epochs=epochs)
    return model


def test_time_adapt(start_model, flagged_idx, X_probe, y_probe, X_train_poisoned, y_train,
                     replay_X, replay_y, epochs=3, lr=1e-4):
    """Fixed version: retrains on the FULL kept-training set (not just probe+replay), so the
    model keeps its anchor to what it already learned instead of catastrophically collapsing
    clean accuracy."""
    mask_keep = np.ones(len(X_train_poisoned), dtype=bool)
    mask_keep[flagged_idx] = False
    tta_model = AdaptableClassifier(copy.deepcopy(start_model.model), lr=lr)
    X_combined = np.vstack([X_train_poisoned[mask_keep], X_probe])
    y_combined = np.concatenate([y_train[mask_keep], y_probe])
    tta_model.adapt(X_combined, y_combined, replay_X=replay_X, replay_y=replay_y, epochs=epochs)
    return tta_model


# ===================== notebook cell 10b8c0f6 =====================
def save_task_models(task_id, models_dict, seeds_dict, extra_meta=None):
    task_dir = MODEL_SAVE_DIR / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models_dict.items():
        fname = (name.replace(" ", "_").replace("(", "").replace(")", "")
                     .replace("/", "-").replace("+", "plus"))
        torch.save(model.model.state_dict(), task_dir / f"{fname}.pt")
    meta = {"seeds": seeds_dict}
    if extra_meta:
        meta.update(extra_meta)
    with open(task_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  saved {len(models_dict)} models + run_meta.json to {task_dir}")


# ===================== notebook cell 7bb221ce =====================
def run_task0_pipeline(state, hidden_sizes, epochs=TASK0_EPOCHS):
    """Task 0: trained FROM SCRATCH on task 0's own data -- no replay buffer (no prior task to replay
    from) and no poisoning (the real pipeline never poisoned task 0 either -- task0.pt's replay buffer
    has 0% *_perturbed entries). hidden_sizes fixes the architecture for this whole run: every later
    task just deep-copies this model's structure forward, so it's never passed again after this."""
    print(f"\n{'='*100}\nTASK 0  (hidden_sizes={hidden_sizes})\n{'='*100}")
    state.setdefault("ckpts_all", {})
    state.setdefault("results", {})
    state["hidden_sizes"] = hidden_sizes

    task_seed = SEED
    torch.manual_seed(task_seed)
    rng = np.random.default_rng(task_seed)

    ckpts = get_ckpts_through(0, state["ckpts_all"])
    task_train_gids, task_test_gids_final, INDEX_OFFSET = reconstruct_task_gids(0, ckpts)

    X_train_raw, y_train, gid_train = rows_for(task_train_gids[0])
    print(f"Task 0: X_train={X_train_raw.shape}")

    ckpt0 = ckpts[0]
    scaler = ckpt0["scaler"]
    X_train_scaled = scaler.transform(X_train_raw.astype("float32"))

    X_test_raw, y_test, _ = rows_for(task_test_gids_final[0])
    X_test_scaled = scaler.transform(X_test_raw.astype("float32"))
    test_data = {0: (X_test_scaled, y_test)}

    feature_dim = ckpt0["feature_dim"]
    net = ClassifierNN(feature_dim, num_classes=2, hidden_sizes=hidden_sizes)
    model0 = AdaptableClassifier(copy.deepcopy(net))
    model0.adapt(X_train_scaled, y_train, epochs=epochs)   # no replay_X/replay_y passed at all
    t0_acc = model0.score(*test_data[0])
    print(f"[task0-trained, {len(hidden_sizes)}-hidden-layer] task 0 test acc: {t0_acc:.3f}")

    y_pred0 = model0.predict(X_test_scaled)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred0, average="binary", zero_division=0)

    final_df = pd.DataFrame([{
        "model": "task0-trained (clean, no poison)",
        "t0_clean": t0_acc, "t0_adv": float("nan"), "t0_combined": t0_acc,
        "t0_precision": prec, "t0_recall": rec, "t0_f1": f1,
        "mean_combined": t0_acc,
    }])

    # this single model seeds ALL 8 lineages entering Task 1 -- task 0 has no poison yet, so there's
    # no distinction between clean/poisoned/fixed lineages until Task 1's own poisoning happens
    models_to_save = {"clean_model": model0, "poisoned_model": model0,
                       **{k: model0 for k in ["dropped_oracle", "amnesiac_oracle", "opposite_oracle",
                                               "dropped_al_tta", "amnesiac_al_tta", "opposite_al_tta"]}}
    save_task_models(0, {"clean_model": model0}, {"task_seed": task_seed})

    result = dict(final_df=final_df, test_data=test_data, models=models_to_save)
    state["results"][0] = result
    return result


# ===================== notebook cell e9969b53 =====================
def run_task_pipeline(task_id, state):
    """state: dict persisted across task cells.
        ckpts_all         -- {task_id: raw checkpoint dict}, cached across calls
        results[t]        -- everything returned from task t, including the live model objects that
                              the NEXT task's cell continues from (results[0] comes from
                              run_task0_pipeline, everything else from this function)
    """
    print(f"\n{'='*100}\nTASK {task_id}\n{'='*100}")
    state.setdefault("ckpts_all", {})
    state.setdefault("results", {})

    base_task_id = task_id - 1
    task_seed = SEED
    torch.manual_seed(task_seed)
    rng = np.random.default_rng(task_seed)

    ckpts = get_ckpts_through(task_id, state["ckpts_all"])
    task_train_gids, task_test_gids_final, INDEX_OFFSET = reconstruct_task_gids(task_id, ckpts)

    X_train_raw, y_train, gid_train = rows_for(task_train_gids[task_id])
    print(f"Task {task_id}: X_train={X_train_raw.shape}")

    base_ckpt = ckpts[base_task_id]
    scaler = base_ckpt["scaler"]
    X_train_scaled = scaler.transform(X_train_raw.astype("float32"))

    replay_X_raw, replay_y, replay_cat, replay_gid = reconstruct_replay_buffer(
        base_ckpt["baseline_replay_buffer"], offset=INDEX_OFFSET)
    replay_X = scaler.transform(replay_X_raw)
    print(f"replay buffer (end of task {base_task_id}, from checkpoint_task{base_task_id}.pt): {replay_X.shape}")

    test_data = {}
    for t in range(task_id + 1):
        X_t_raw, y_t, _ = rows_for(task_test_gids_final[t])
        test_data[t] = (scaler.transform(X_t_raw.astype("float32")), y_t)
        print(f"  test set task {t}: {test_data[t][0].shape}")

    # every task from 1 onward (including task 1) chains from the PREVIOUS task's own live models --
    # task 1 reads state["results"][0], which run_task0_pipeline populated
    prev_models = state["results"][task_id - 1]["models"]
    prev_clean = prev_models["clean_model"]
    prev_poisoned = prev_models["poisoned_model"]
    prev_fixed = {k: prev_models[k] for k in
                  ["dropped_oracle", "amnesiac_oracle", "opposite_oracle",
                   "dropped_al_tta", "amnesiac_al_tta", "opposite_al_tta"]}

    clean_model = AdaptableClassifier(copy.deepcopy(prev_clean.model))
    clean_model.adapt(X_train_scaled, y_train, replay_X=replay_X, replay_y=replay_y,
                       epochs=CLEAN_ADAPT_EPOCHS)
    print(f"[clean-adapted] task {task_id} test acc: {clean_model.score(*test_data[task_id]):.3f}")

    pocket_toward_1 = X_train_scaled[y_train == 1].mean(axis=0)
    pocket_toward_0 = X_train_scaled[y_train == 0].mean(axis=0)
    separating_direction = pocket_toward_1 - pocket_toward_0

    n_class0 = (y_train == 0).sum()
    n_class1 = (y_train == 1).sum()
    n_poison_0 = int(POISON_FRACTION * n_class0)
    n_poison_1 = int(POISON_FRACTION * n_class1)

    X_p1, idx_pocket_A = craft_boundary_pocket_poison(
        clean_model, X_train_scaled, y_train, class_id=0, direction=separating_direction, n_points=n_poison_0)
    X_train_poisoned, idx_pocket_B = craft_boundary_pocket_poison(
        clean_model, X_p1, y_train, class_id=1, direction=-separating_direction, n_points=n_poison_1)
    poison_idx = np.concatenate([idx_pocket_A, idx_pocket_B])
    print(f"poisoned {len(poison_idx)}/{len(X_train_scaled)} = {len(poison_idx)/len(X_train_scaled):.1%} of task {task_id}'s train set")

    poisoned_model = AdaptableClassifier(copy.deepcopy(prev_poisoned.model))
    poisoned_model.adapt(X_train_poisoned, y_train, replay_X=replay_X, replay_y=replay_y, epochs=ADAPT_EPOCHS)
    print(f"[poisoned-adapted] clean task {task_id} test acc: {poisoned_model.score(*test_data[task_id]):.3f}")

    typical_gap = np.linalg.norm(test_data[task_id][0][test_data[task_id][1] == 0] - pocket_toward_1, axis=1).mean()
    EPS_TEST = typical_gap * 0.15

    adv_honest, succ_by_task = {}, {}
    for t in range(task_id + 1):
        Xt, yt = test_data[t]
        print(f"  attacking task {t}'s test set ({len(yt)} pts)...")
        X_adv_t, succ_t, _ = adversarial_attack_pocket(poisoned_model, clean_model, Xt, yt, epsilon_max=EPS_TEST)
        adv_honest[t] = honest_adv(Xt, X_adv_t, succ_t)
        succ_by_task[t] = succ_t
        print(f"    task {t} genuine pockets: {succ_t.sum()}/{len(yt)} ({succ_t.mean():.1%})")

    print("running oracle detector...")
    detected_poison_idx = run_oracle_detector(poisoned_model, X_train_poisoned, y_train,
                                               idx_pocket_A, idx_pocket_B, rng=rng)
    print("running meta_al detector...")
    meta_al_detected = run_meta_al_detector(poisoned_model, X_train_poisoned, y_train, poison_idx,
                                             replay_X, replay_y, rng=rng)

    dropped_oracle = unlearn_dropped_rows(prev_fixed["dropped_oracle"], detected_poison_idx,
                                           X_train_poisoned, y_train, replay_X, replay_y)
    amnesiac_oracle = unlearn_amnesiac(prev_fixed["amnesiac_oracle"], detected_poison_idx,
                                        X_train_poisoned, y_train, replay_X, replay_y)
    opposite_oracle = unlearn_opposite_class(prev_fixed["opposite_oracle"], detected_poison_idx,
                                              X_train_poisoned, y_train, replay_X, replay_y)
    print(f"  [oracle] dropped/amnesiac/opposite each continued from their own prior state, "
          f"fixed on {len(detected_poison_idx)} newly-flagged points")

    dropped_al = unlearn_dropped_rows(prev_fixed["dropped_al_tta"], meta_al_detected,
                                       X_train_poisoned, y_train, replay_X, replay_y)
    amnesiac_al = unlearn_amnesiac(prev_fixed["amnesiac_al_tta"], meta_al_detected,
                                    X_train_poisoned, y_train, replay_X, replay_y)
    opposite_al = unlearn_opposite_class(prev_fixed["opposite_al_tta"], meta_al_detected,
                                          X_train_poisoned, y_train, replay_X, replay_y)
    print(f"  [meta_al] dropped/amnesiac/opposite each continued from their own prior state, "
          f"fixed on {len(meta_al_detected)} newly-flagged points")

    calib_X_list, calib_y_list, eval_pockets = [], [], {}
    for t in range(task_id + 1):
        Xt, yt = test_data[t]
        succ_t = succ_by_task[t]
        pocket_indices = np.where(succ_t)[0]
        n_calib = int(CALIB_FRAC * len(pocket_indices))
        calib_idx_t = rng.choice(pocket_indices, n_calib, replace=False) if n_calib > 0 else np.array([], dtype=int)
        eval_idx_t = np.setdiff1d(pocket_indices, calib_idx_t)
        calib_X_list.append(adv_honest[t][calib_idx_t])
        calib_y_list.append(yt[calib_idx_t])
        eval_pockets[t] = (adv_honest[t][eval_idx_t], yt[eval_idx_t])
        print(f"  task {t}: calib pockets={len(calib_idx_t)}  held-out eval pockets={len(eval_idx_t)}")

    X_calib_combined = np.vstack(calib_X_list)
    y_calib_combined = np.concatenate(calib_y_list)
    print(f"combined calibration probe across tasks 0..{task_id}: {len(y_calib_combined)} points")

    dropped_al_tta = test_time_adapt(dropped_al, meta_al_detected, X_calib_combined, y_calib_combined,
                                      X_train_poisoned, y_train, replay_X, replay_y)
    amnesiac_al_tta = test_time_adapt(amnesiac_al, meta_al_detected, X_calib_combined, y_calib_combined,
                                       X_train_poisoned, y_train, replay_X, replay_y)
    opposite_al_tta = test_time_adapt(opposite_al, meta_al_detected, X_calib_combined, y_calib_combined,
                                       X_train_poisoned, y_train, replay_X, replay_y)

    final_models = [
        ("poisoned (baseline / MADAR)", poisoned_model),
        ("dropped-rows (oracle)", dropped_oracle),
        ("amnesiac (oracle)", amnesiac_oracle),
        ("opposite-class (oracle)", opposite_oracle),
        ("dropped-rows (meta_al+TTA)", dropped_al_tta),
        ("amnesiac (meta_al+TTA)", amnesiac_al_tta),
        ("opposite-class (meta_al+TTA)", opposite_al_tta),
    ]

    header = f"{'model':<32}"
    for t in range(task_id + 1):
        header += f" {'T'+str(t)+' clean':>9} {'T'+str(t)+' adv':>8} {'T'+str(t)+' comb':>9} {'T'+str(t)+' f1':>7}"
    header += f" {'MEAN COMB':>10}"
    print("\n" + header)
    print("=" * len(header))

    final_rows = []
    for name, model in final_models:
        row = {"model": name}
        combined_accs = []
        line = f"{name:<32}"
        for t in range(task_id + 1):
            Xt, yt = test_data[t]
            Xp, yp = eval_pockets[t]
            clean_acc = model.score(Xt, yt)
            adv_acc = model.score(Xp, yp) if len(yp) else float("nan")
            if len(yp):
                X_eval, y_eval = np.vstack([Xt, Xp]), np.concatenate([yt, yp])
            else:
                X_eval, y_eval = Xt, yt
            comb_acc = model.score(X_eval, y_eval)
            y_pred_eval = model.predict(X_eval)
            prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred_eval, average="binary", zero_division=0)
            combined_accs.append(comb_acc)
            row[f"t{t}_clean"] = clean_acc
            row[f"t{t}_adv"] = adv_acc
            row[f"t{t}_combined"] = comb_acc
            row[f"t{t}_precision"] = prec
            row[f"t{t}_recall"] = rec
            row[f"t{t}_f1"] = f1
            line += f" {clean_acc:>9.3f} {adv_acc:>8.3f} {comb_acc:>9.3f} {f1:>7.3f}"
        mean_combined = float(np.mean(combined_accs))
        row["mean_combined"] = mean_combined
        line += f" {mean_combined:>10.3f}"
        print(line)
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    models_to_save = {
        "clean_model": clean_model,
        "poisoned_model": poisoned_model,
        "dropped_oracle": dropped_oracle, "amnesiac_oracle": amnesiac_oracle, "opposite_oracle": opposite_oracle,
        "dropped_al_tta": dropped_al_tta, "amnesiac_al_tta": amnesiac_al_tta, "opposite_al_tta": opposite_al_tta,
    }
    seeds_used = {"task_seed": task_seed}
    save_task_models(task_id, models_to_save, seeds_used,
                      extra_meta={"n_oracle_detected": int(len(detected_poison_idx)),
                                  "n_meta_al_detected": int(len(meta_al_detected)),
                                  "n_calib_combined": int(len(y_calib_combined))})

    result = dict(final_df=final_df, test_data=test_data, adv_honest=adv_honest, succ_by_task=succ_by_task,
                  eval_pockets=eval_pockets, X_calib_combined=X_calib_combined, y_calib_combined=y_calib_combined,
                  detected_poison_idx=detected_poison_idx, meta_al_detected=meta_al_detected,
                  models=models_to_save)
    state["results"][task_id] = result
    return result


# ===================== notebook cell 3d75e2b3 =====================
all_architecture_results = {}


def run_one_architecture(depth):
    """Runs Task 0-9 for HIDDEN_SIZES_BY_DEPTH[depth] with a fresh state, saves models under their own
    MODEL_SAVE_DIR, writes this architecture's p/r/f1 table to RESULTS_DIR, and returns that table."""
    hidden_sizes = HIDDEN_SIZES_BY_DEPTH[depth]
    print(f"\n{'#'*100}\n RUNNING {depth}-HIDDEN-LAYER ARCHITECTURE {hidden_sizes}\n{'#'*100}")
    global MODEL_SAVE_DIR
    MODEL_SAVE_DIR = MODEL_SAVE_DIR_BASE / f"seed{SEED}_depth{depth}"
    state = {}
    run_task0_pipeline(state, hidden_sizes)
    for t in range(1, 10):
        run_task_pipeline(t, state)

    # ---- stack every task's final_df (with p/r/f1) into one table for this architecture ----
    rows = []
    for t in range(0, 10):
        df = state["results"][t]["final_df"]
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["task"] = t
            row_dict["depth"] = depth
            rows.append(row_dict)
    depth_df = pd.DataFrame(rows)
    cols = ["depth", "task", "model"] + [c for c in depth_df.columns if c not in ("depth", "task", "model")]
    depth_df = depth_df[cols]

    out_path = RESULTS_DIR / f"prf_report_seed{SEED}_depth{depth}.csv"
    depth_df.to_csv(out_path, index=False)
    print(f"saved {len(depth_df)} rows to {out_path}")
    return depth_df


# ===================== timing harness (not part of the notebook) =====================
import os as _os
import platform as _platform
import resource as _resource

_T_DATA_LOADED = _time.perf_counter()   # module code above (incl. reading the h5) has run by now
_PHASE_SECONDS = {}                     # (task_id, phase) -> seconds
_ATTACK_ROWS = {}                       # task_id -> test rows attacked
_CURRENT_TASK = [None]


def _timed(phase, fn):
    """Wraps a notebook function so its wall-clock time is charged to (current task, phase).
    The notebook functions call each other through module globals, so rebinding the global name
    is enough -- the wrapped function's behavior is unchanged."""
    def wrapper(*a, **k):
        t0 = _time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            key = (_CURRENT_TASK[0], phase)
            _PHASE_SECONDS[key] = _PHASE_SECONDS.get(key, 0.0) + _time.perf_counter() - t0
            if phase == "test-time pocket attacks":
                X = a[2] if len(a) > 2 else k.get("X")
                _ATTACK_ROWS[_CURRENT_TASK[0]] = _ATTACK_ROWS.get(_CURRENT_TASK[0], 0) + len(X)
    return wrapper


for _name, _phase in [
    ("adversarial_attack_pocket", "test-time pocket attacks"),
    ("run_oracle_detector", "oracle detector"),
    ("run_meta_al_detector", "meta_al detector"),
    ("unlearn_dropped_rows", "unlearning (6 lineages)"),
    ("unlearn_amnesiac", "unlearning (6 lineages)"),
    ("unlearn_opposite_class", "unlearning (6 lineages)"),
    ("test_time_adapt", "test-time adaptation (3 models)"),
    ("save_task_models", "saving models"),
]:
    globals()[_name] = _timed(_phase, globals()[_name])


def _fmt(sec):
    return f"{sec:8.1f}s ({sec / 3600:.3f} h)"


def _extrapolate(task_totals, depth):
    """Rough full-run estimate from the measured Task 1: attack time scales with the number of
    test rows attacked (Task t attacks the test sets of tasks 0..t); everything else scales with
    Task t's training rows + the 4000-row replay buffer."""
    if 1 not in task_totals:
        return None
    attack1 = _PHASE_SECONDS.get((1, "test-time pocket attacks"), 0.0)
    rows1 = _ATTACK_ROWS.get(1, 0)
    other1 = task_totals[1] - attack1
    n_rows = N
    sizes = [int(round(f * n_rows)) for f in TASK_FRACTIONS]
    train = [s * 0.8 for s in sizes]
    test = [s * 0.2 for s in sizes]
    replay = 4000
    per_task = {0: task_totals[0]}
    for t in range(1, N_TASKS):
        rows_t = sum(test[:t + 1])
        attack_t = attack1 * rows_t / rows1 if rows1 else 0.0
        other_t = other1 * (train[t] + replay) / (train[1] + replay)
        per_task[t] = attack_t + other_t
    one_arch = sum(per_task.values())
    return per_task, one_arch


def _main():
    depth, num_tasks = _args.depth, _args.num_tasks
    hidden_sizes = HIDDEN_SIZES_BY_DEPTH[depth]
    globals()["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR_BASE / f"seed{SEED}_depth{depth}"
    print(f"\n[timing] depth={depth} hidden_sizes={hidden_sizes} tasks=0..{num_tasks - 1} seed={SEED}")
    print(f"[timing] host={_platform.node()} cpus={_os.cpu_count()} torch_threads={torch.get_num_threads()} "
          f"cuda_available={torch.cuda.is_available()}")
    state = {}
    task_totals = {}
    for t in range(num_tasks):
        _CURRENT_TASK[0] = t
        t0 = _time.perf_counter()
        if t == 0:
            run_task0_pipeline(state, hidden_sizes)
        else:
            run_task_pipeline(t, state)
        task_totals[t] = _time.perf_counter() - t0
        print(f"[timing] Task {t} wall-clock: {_fmt(task_totals[t])}")

    model_device = str(next(state["results"][num_tasks - 1]["models"]["clean_model"].model.parameters()).device)
    total = _time.perf_counter() - _T_START
    load = _T_DATA_LOADED - _T_START
    peak_rss_gb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

    lines = ["", "=" * 78, f"TIMING SUMMARY  (depth {depth} {hidden_sizes}, seed {SEED}, tasks 0..{num_tasks - 1})",
             "=" * 78,
             f"startup + loading the h5:        {_fmt(load)}"]
    for t, sec in task_totals.items():
        lines.append(f"Task {t} total:                    {_fmt(sec)}")
        phases = sorted({p for (tt, p) in _PHASE_SECONDS if tt == t})
        accounted = 0.0
        for p in phases:
            s = _PHASE_SECONDS[(t, p)]
            accounted += s
            extra = f"  [{_ATTACK_ROWS.get(t, 0)} test rows]" if p == "test-time pocket attacks" else ""
            lines.append(f"    {p:<34}{_fmt(s)}{extra}")
        if t > 0:
            lines.append(f"    {'clean/poisoned adaptation + rest':<34}{_fmt(sec - accounted)}")
    lines += [f"TOTAL wall-clock (this run):      {_fmt(total)}",
              f"peak memory (RSS):                {peak_rss_gb:.1f} GB",
              f"models ran on device:             {model_device}  "
              f"(cuda available: {torch.cuda.is_available()})"]
    if model_device == "cpu":
        lines.append("  -> GPU-hours actually used: 0 (code is CPU-only as written). If a GPU is reserved for the "
                     "job anyway, allocated GPU-hours = wall-clock hours.")

    est = _extrapolate(task_totals, depth)
    if est is not None:
        per_task, one_arch = est
        lines += ["", "ROUGH EXTRAPOLATION from the measured Task 1 (attacks scale with test rows attacked,",
                  "everything else with training rows) -- treat as an order-of-magnitude estimate:"]
        for t in range(N_TASKS):
            tag = "measured" if t in task_totals else "estimated"
            lines.append(f"    Task {t}: {_fmt(per_task[t] if t not in task_totals else task_totals[t])}  {tag}")
        full_one = load + sum(task_totals.get(t, per_task[t]) for t in range(N_TASKS))
        lines += [f"Full Task 0-9, depth {depth} only:   {_fmt(full_one)}",
                  f"Full Task 0-9, all 4 depths:       <= {_fmt(4 * full_one)}  "
                  f"(upper bound: depths 1-3 are smaller networks; time --depth 1 too for a tighter range)"]
    text = "\n".join(lines)
    print(text)

    out = RESULTS_DIR / f"timing_seed{SEED}_depth{depth}_tasks{num_tasks}.json"
    with open(out, "w") as f:
        json.dump({
            "seed": SEED, "depth": depth, "hidden_sizes": list(hidden_sizes), "num_tasks": num_tasks,
            "host": _platform.node(), "cpu_count": _os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "model_device": model_device,
            "load_seconds": load, "task_seconds": task_totals, "total_seconds": total,
            "phase_seconds": {f"task{t}:{p}": s for (t, p), s in _PHASE_SECONDS.items()},
            "attack_rows": _ATTACK_ROWS, "peak_rss_gb": peak_rss_gb,
            "estimate_full_one_depth_seconds": (load + sum(task_totals.get(t, est[0][t]) for t in range(N_TASKS)))
            if est else None,
        }, f, indent=2)
    print(f"\n[timing] wrote {out}")


if __name__ == "__main__":
    _main()
