"""
CICIDS inner loop for meta-learning forget-set selection.

This is clmd_cicids_unlearning_cd.py's data construction, MADAR replay, SI/KD, and
targeted-unlearning machinery, refactored to be called many times cheaply by an outer
optimizer (es_meta_train_cicids.py) or standalone for baselines -- same shape as the
EMBER version of this file (mu_inner_loop.py), with two CICIDS-specific changes:

  * Outer-loop windowing: --window_start/--window_rows slice a contiguous, time-ordered
    chunk out of the full dataset BEFORE the usual TASK0_ROW_FRACTION task split runs on
    it. The outer driver slides this window forward (with overlap) across generations;
    this file just executes whichever window it's told to use.
  * Forget-set selection is budget-based (MEM_SIZE per label), not ratio-based like
    EMBER's --forget_ratio. --selector nn1 computes the same budget (so donut/random/nn1
    all forget the same COUNT per label, for a fair comparison) and asks nn1_scorer to
    rank samples within that budget.

  * --checkpoint: task-0 model + scaler + replay buffer, keyed to (window_start,
    window_rows, seed) -- cheap to skip when many ES candidates share one generation's
    window/seed (they all bottleneck on the same task-0 training otherwise).
  * --selector: donut (hand-crafted IsolationForest baseline) | random | nn1 (learned).
  * --scorer_weights: flat parameter vector for nn1_scorer (npz, key 'params').
  * --out: reward JSON ({mean_acc, final_acc, per_task, ...}) for the driver, plus full
    per-task diagnostics for later inspection.

Examples:
  # build a window's task-0 checkpoint once, reused by every candidate that generation
  python mu_inner_loop_cicids.py --window_start 0 --window_rows 20000 \
      --checkpoint ckpt_win0.pt --make_checkpoint
  # baseline evaluation under that window
  python mu_inner_loop_cicids.py --window_start 0 --window_rows 20000 \
      --checkpoint ckpt_win0.pt --selector donut --out reward_donut.json
  # one ES candidate
  python mu_inner_loop_cicids.py --window_start 0 --window_rows 20000 \
      --checkpoint ckpt_win0.pt --selector nn1 --scorer_weights cand_003.npz --out r3.json
"""

import os
import json
import copy
import random
import pickle
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import nn1_scorer

# --- CLI ---
parser = argparse.ArgumentParser(description="CICIDS MU inner loop for meta-learned forget-set selection")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset_path', type=str, default="./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset2.pkl")
parser.add_argument('--window_start', type=int, default=0,
                    help='Row offset into the time-sorted ORIGINAL rows where this window begins')
parser.add_argument('--window_rows', type=int, default=None,
                    help='Rows in the window, starting at --window_start. Default: everything '
                         'to the end of the dataset (i.e. behave like the un-windowed script).')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='npz/pt path caching task-0 (model + buffer), keyed to window+seed. '
                         'Omit to always train task 0 fresh.')
parser.add_argument('--make_checkpoint', action='store_true',
                    help='Train/load task 0, save the checkpoint, then exit')
parser.add_argument('--selector', type=str, default='donut', choices=['donut', 'random', 'nn1'])
parser.add_argument('--scorer_weights', type=str, default=None, help='npz with key "params" (selector=nn1)')
parser.add_argument('--n_tasks', type=int, default=6, help='Total tasks including task 0 (matches NUM_TASKS)')
parser.add_argument('--cl_iters', type=int, default=2000)
parser.add_argument('--unlearn_epochs', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.2, help='Weight of the forget loss during unlearning')
parser.add_argument('--out', type=str, default=None, help='Write reward JSON here')
parser.add_argument('--quiet', action='store_true')
args = parser.parse_args()

assert args.n_tasks >= 2, "--n_tasks must be at least 2 (task 0 plus one CL task)"

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TRAIN_DEVICE = "cpu"           # see clmd_cicids_naive_cd.py -- "auto"/"cuda" can crash on
                                # newer GPUs than the installed torch build supports
DEVICE = torch.device(TRAIN_DEVICE)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION
#  (kept as constants, matching the naive/joint/MADAR/unlearning sibling files --
#   only what the ES driver actually needs to vary is exposed as a CLI flag above)
# ==========================================
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30
BATCH_SIZE = 256

MEM_SIZE = 5000
MADAR_CONTAMINATION = 0.1

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1

UNLEARN_LR = 1e-4

TASK0_ROW_FRACTION = 0.35
TASK_TEST_FRAC = 0.20

TASK0_CLEAN_ONLY = True
AGENT_MODE = "mixed"
FIXED_AGENT_ID = 0
REQUIRE_EVASION_SUCCESS = True
POISON_FRACTION = 0.8

ENSURE_CATEGORY_DIVERSITY = True
MIN_CATEGORY_COUNT = 5

EVAL_BATCH_SIZE = 512
# ==========================================


def log(*a):
    if not args.quiet:
        print(*a, flush=True)


# --- 1. Data Loading ---

def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Point --dataset_path at the red_agent_perturbed_dataset.pkl "
            f"produced by generate_red_agent_dataset.py."
        )
    with open(path, "rb") as f:
        blob = pickle.load(f)
    df = blob["data_original_scale"].copy()
    feature_cols = blob["feature_columns"]
    return df, feature_cols, blob.get("metadata", {})


def encode_labels(df: pd.DataFrame, label_column: str):
    le = LabelEncoder()
    y_all = le.fit_transform(df[label_column].values)
    return y_all, len(le.classes_), le


# --- 2. Task windowing: outer-loop slice first, then the usual TASK0_ROW_FRACTION split ---

def assign_task_windows(df: pd.DataFrame, num_tasks: int, task0_fraction: float,
                        window_start: int = 0, window_rows: int = None) -> pd.DataFrame:
    originals = df[df["row_type"] == "original"].copy()
    originals = originals.sort_values("source_row_id").reset_index(drop=True)
    n_total = len(originals)

    if window_rows is None:
        window_rows = n_total - window_start
    assert 0 <= window_start < n_total, f"window_start {window_start} out of range [0, {n_total})"
    assert window_start + window_rows <= n_total, (
        f"window [{window_start}, {window_start + window_rows}) runs past the dataset ({n_total} rows); "
        f"the outer loop is responsible for wrapping window_start back to 0 before calling this script"
    )
    originals = originals.iloc[window_start:window_start + window_rows].reset_index(drop=True)
    n = len(originals)

    n_task0 = int(round(n * task0_fraction))
    n_task0 = max(1, min(n_task0, n - (num_tasks - 1)))
    remaining = n - n_task0
    per_task = remaining // (num_tasks - 1) if num_tasks > 1 else 0

    task_ids = np.empty(n, dtype=int)
    task_ids[:n_task0] = 0
    idx = n_task0
    for t in range(1, num_tasks):
        end = idx + per_task if t < num_tasks - 1 else n
        task_ids[idx:end] = t
        idx = end

    originals["task_id"] = task_ids
    return originals


def assign_perturbed_task_ids(df: pd.DataFrame, originals_with_task: pd.DataFrame) -> pd.DataFrame:
    task_lookup = originals_with_task.set_index("orig_row_id")["task_id"]
    perturbed = df[df["row_type"] == "perturbed"].copy()
    perturbed["task_id"] = perturbed["source_row_id"].map(task_lookup)
    perturbed = perturbed.dropna(subset=["task_id"]).copy()
    perturbed["task_id"] = perturbed["task_id"].astype(int)
    return perturbed


# --- 3. Poisoning substitution + category diversity floor (unchanged from clmd_cicids_unlearning_cd.py) ---

def poison_and_diversify(malicious_orig_subset: pd.DataFrame, perturbed_pool: pd.DataFrame,
                          agent_for_task: int, poison_fraction: float, require_evasion_success: bool,
                          ensure_diversity: bool, min_category_count: int, rng: random.Random):
    cand = perturbed_pool[perturbed_pool["agent_id"] == agent_for_task]
    if require_evasion_success:
        cand = cand[cand["evasion_success"] == True]  # noqa: E712
    eligible_ids = set(cand["source_row_id"].tolist())

    all_ids = malicious_orig_subset["orig_row_id"].tolist()
    eligible = [i for i in all_ids if i in eligible_ids]
    n_elig = len(eligible)

    n_poison = int(round(poison_fraction * n_elig))
    if poison_fraction <= 0.0:
        n_poison = 0
    elif ensure_diversity and n_elig > 0:
        if n_elig >= 2 * min_category_count:
            n_poison = max(min_category_count, min(n_poison, n_elig - min_category_count))
        else:
            n_poison = n_elig // 2
    n_poison = max(0, min(n_poison, n_elig))

    rng.shuffle(eligible)
    poison_ids = set(eligible[:n_poison])
    clean_ids = set(all_ids) - poison_ids

    clean_rows = malicious_orig_subset[malicious_orig_subset["orig_row_id"].isin(clean_ids)].copy()
    clean_rows["category"] = "malicious_clean"
    clean_rows["poison_agent_id"] = -1

    poison_rows = cand[cand["source_row_id"].isin(poison_ids)].copy()
    poison_rows["category"] = "malicious_perturbed"
    poison_rows["poison_agent_id"] = poison_rows["agent_id"]

    combined = pd.concat([clean_rows, poison_rows], ignore_index=True)
    return combined, len(poison_ids), len(clean_ids)


# --- 4. Model ---

class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024); self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512);       self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256);        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128);        self.fc4_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(); self.fc_last = nn.Linear(128, num_classes)

    def forward(self, x, return_latent=False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    return F.kl_div(log_scores_norm, targets_norm, reduction='batchmean') * (T ** 2)


# --- 5. MADAR replay buffer (identical to clmd_cicids_unlearning_cd.py) ---

label_buffers = {}
replay_buffer = []


def current_budget_per_label(current_labels):
    all_group_keys = set(label_buffers.keys()) | set(current_labels)
    return MEM_SIZE // max(len(all_group_keys), 1)


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, model, device):
    global label_buffers, replay_buffer
    model.eval()

    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    all_latents = []
    with torch.no_grad():
        for v, l in loader:
            v = v.to(device)
            _, latent = model(v, return_latent=True)
            all_latents.append(latent.cpu())
    L_np = torch.cat(all_latents).numpy()
    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)

    current_labels = np.unique(Y_np).tolist()
    budget_per_label = current_budget_per_label(current_labels)

    for key in list(label_buffers.keys()):
        if len(label_buffers[key]) > budget_per_label:
            current_buffer = label_buffers[key]
            half_budget = budget_per_label // 2
            anomalies = current_buffer[::2]
            inliers = current_buffer[1::2]
            new_buffer = [val for pair in zip(anomalies[:half_budget], inliers[:half_budget]) for val in pair]
            if budget_per_label % 2 != 0 and len(anomalies) > half_budget:
                new_buffer.append(anomalies[half_budget])
            label_buffers[key] = new_buffer

    for lbl in current_labels:
        mask = (Y_np == lbl)
        X_lbl, Y_lbl, L_lbl, cat_lbl = X_np[mask], Y_np[mask], L_np[mask], cat_np[mask]
        n_select = min(budget_per_label, len(X_lbl))
        if n_select == 0:
            continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(L_lbl)
        scores = iso.decision_function(L_lbl)
        sorted_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_idx = sorted_idx[:half]
        inliers_idx = sorted_idx[-(n_select - half):]
        interleaved_idx = [idx for pair in zip(anomalies_idx, inliers_idx) for idx in pair]
        if n_select % 2 != 0:
            interleaved_idx.append(inliers_idx[-1])

        label_buffers[lbl] = [
            (torch.tensor(X_lbl[i]), torch.tensor(Y_lbl[i]), cat_lbl[i]) for i in interleaved_idx
        ]

    replay_buffer.clear()
    for buf in label_buffers.values():
        replay_buffer.extend(buf)


def buffer_composition_summary():
    summary = {}
    for lbl, entries in label_buffers.items():
        cats = [e[2] for e in entries]
        summary[str(lbl)] = {"total": len(entries), "category_counts": dict(Counter(cats))}
    return summary


def train_cl_er(model, teacher_model, optimizer, loader, iters, active_count, prev_active_count,
                W, omega, p_old_task, tid, si_c, device):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    mask = torch.full((model.fc_last.out_features,), -1e9).to(device)
    mask[:active_count] = 0.0

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
        loss_cur = nn.CrossEntropyLoss()(model(combined_v) + mask, combined_l)

        rnt = 1.0 / (tid + 1)
        outputs_mem = model(mem_v)
        with torch.no_grad():
            teacher_logits = teacher_model(mem_v)[:, :prev_active_count]

        loss_replay = loss_fn_kd(outputs_mem[:, :prev_active_count], teacher_logits, T=KD_TEMP)
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


# --- 6. Forget/retain selectors (donut/random unchanged; nn1 new) ---

def compute_nn1_features(X_lbl: torch.Tensor, y_lbl: torch.Tensor, model, active_count, device):
    """Per-sample features for NN-1, within one label group (see nn1_scorer.py docstring
    for the 7-feature layout). z-scored within the group so scales are comparable across
    labels/tasks/windows."""
    model.eval()
    loader = data.DataLoader(data.TensorDataset(X_lbl, y_lbl), batch_size=EVAL_BATCH_SIZE)
    lats, logits_all, losses = [], [], []
    with torch.no_grad():
        for v, l in loader:
            v, l = v.to(device), l.to(device)
            logits, latent = model(v, return_latent=True)
            logits = logits[:, :active_count]
            lats.append(latent.cpu()); logits_all.append(logits.cpu())
            losses.append(F.cross_entropy(logits, l, reduction='none').cpu())
    L = torch.cat(lats).numpy()
    logits = torch.cat(logits_all)
    loss = torch.cat(losses).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    ent = -(probs * np.log(probs + 1e-9)).sum(1) / np.log(active_count)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    iso_l = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(L)
    f_iso_lat = iso_l.decision_function(L)
    Xn = X_lbl.numpy()
    iso_r = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(Xn)
    f_iso_raw = iso_r.decision_function(Xn)

    centroid = L.mean(axis=0)
    cent_dist = np.linalg.norm(L - centroid, axis=1)
    group_size = np.full(len(y_lbl), np.log10(len(y_lbl)))

    feats = np.stack([f_iso_lat, f_iso_raw, loss, ent, margin, cent_dist, group_size], axis=1)
    return nn1_scorer.zscore(feats)


def split_forget_retain(X: torch.Tensor, y: torch.Tensor, budget_per_label: int, model, active_count,
                        device, selector: str, scorer_weights: str, tid: int):
    """Per-LABEL split: keep up to budget_per_label as retain, forget the rest. donut/random
    match clmd_cicids_unlearning_cd.py's Option A exactly; nn1 forgets the SAME count per
    label (so all three selectors are directly comparable under one budget), just choosing
    WHICH samples using the learned scorer instead of a fixed IsolationForest/random rule."""
    X_np = X.numpy()
    y_np = y.numpy()

    retain_idx_all = []
    forget_idx_all = []

    for lbl in np.unique(y_np):
        lbl_indices = np.where(y_np == lbl)[0]
        n_lbl = len(lbl_indices)
        n_select = min(budget_per_label, n_lbl)

        if n_select == n_lbl:
            retain_idx_all.extend(lbl_indices.tolist())
            continue

        if selector == 'donut':
            X_lbl = X_np[lbl_indices]
            iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
            iso.fit(X_lbl)
            scores = iso.decision_function(X_lbl)
            sorted_local = np.argsort(scores)
            half = n_select // 2
            anomalies_local = sorted_local[:half]
            inliers_local = sorted_local[-(n_select - half):]
            selected_local = np.concatenate((anomalies_local, inliers_local))
            unselected_local = np.setdiff1d(np.arange(n_lbl), selected_local)
        elif selector == 'random':
            rng_lbl = np.random.default_rng(SEED * 1000 + tid * 7 + int(lbl))
            perm = rng_lbl.permutation(n_lbl)
            selected_local = perm[:n_select]
            unselected_local = perm[n_select:]
        else:  # nn1
            assert scorer_weights, "--scorer_weights required for --selector nn1"
            params, hidden, nf = nn1_scorer.load(scorer_weights)
            assert nf == nn1_scorer.N_FEATURES, (
                f"scorer expects {nf} features, this pipeline computes {nn1_scorer.N_FEATURES}"
            )
            X_lbl_t, y_lbl_t = X[lbl_indices], y[lbl_indices]
            feats = compute_nn1_features(X_lbl_t, y_lbl_t, model, active_count, device)
            forget_ratio_lbl = (n_lbl - n_select) / n_lbl
            forget_local, retain_local = nn1_scorer.select_forget(params, feats, forget_ratio_lbl, hidden=hidden)
            selected_local, unselected_local = retain_local, forget_local

        retain_idx_all.extend(lbl_indices[selected_local].tolist())
        forget_idx_all.extend(lbl_indices[unselected_local].tolist())

    return retain_idx_all, forget_idx_all


def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, active_count,
                           omega, p_old_task, si_c, epochs, lr, alpha, device):
    global replay_buffer
    model.train()
    teacher_model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mask = torch.full((model.fc_last.out_features,), -1e9).to(device)
    mask[:active_count] = 0.0

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

            logits = model(f_v)[:, :active_count]
            log_probs = F.log_softmax(logits, dim=1)
            uniform_probs = torch.ones_like(log_probs) / active_count
            forget_loss = F.kl_div(log_probs, uniform_probs, reduction='batchmean')

            r_v, r_l = next(retain_iter)
            r_v, r_l = r_v.to(device), r_l.to(device)

            if buf_iter:
                mem_v, mem_l = next(buf_iter)
                mem_v, mem_l = mem_v.to(device), mem_l.to(device)
                r_v = torch.cat([r_v, mem_v], dim=0)
                r_l = torch.cat([r_l, mem_l], dim=0)

            with torch.no_grad():
                teacher_logits = teacher_model(r_v)[:, :active_count]

            student_logits_raw = model(r_v)
            student_logits_masked = student_logits_raw + mask
            student_logits_active = student_logits_raw[:, :active_count]

            retain_ce = nn.CrossEntropyLoss()(student_logits_masked, r_l)
            retain_kd = loss_fn_kd(student_logits_active, teacher_logits, T=KD_TEMP)
            retain_loss = (0.5 * retain_ce) + (0.5 * retain_kd)

            si_loss = 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    if n_key in p_old_task:
                        si_loss += (omega[n_key] * (p - p_old_task[n_key]) ** 2).sum()

            total_loss = (alpha * forget_loss) + ((1.0 - alpha) * retain_loss) + (si_c * si_loss)
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite loss during unlearning - training diverged")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def measure_unlearning_efficacy(model_before, model_after, loader, active_count, device):
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
            logits_before = logits_before[:, :active_count]
            logits_after, latent_after = model_after(v, return_latent=True)
            logits_after = logits_after[:, :active_count]

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
        'acc_before': (acc_before / total_samples) * 100,
        'acc_after': (acc_after / total_samples) * 100,
        'entropy': mean_entropy,
        'entropy_norm': mean_entropy / float(np.log(active_count)),
        'shift_mean': shifts.mean().item(),
        'shift_median': shifts.median().item(),
        'shift_max': shifts.max().item(),
    }


# --- 7. Eval ---

def compute_metrics(y_true, y_pred, class_labels):
    acc = float(accuracy_score(y_true, y_pred)) * 100
    cm = confusion_matrix(y_true, y_pred, labels=class_labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "n_samples": int(len(y_true)),
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(class_labels),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
    }


def evaluate_slice(model, X, y, categories, active_count, device, class_labels):
    model.eval()
    preds = []
    with torch.no_grad():
        loader = data.DataLoader(data.TensorDataset(X), batch_size=EVAL_BATCH_SIZE)
        for (v,) in loader:
            logits = model(v.to(device))[:, :active_count]
            preds.append(torch.argmax(logits, dim=1).cpu())
    preds = torch.cat(preds).numpy()
    y_np = y.numpy()

    overall = compute_metrics(y_np, preds, class_labels)
    per_category = {}
    for cat in sorted(set(categories.tolist())):
        mask = categories == cat
        if mask.sum() == 0:
            continue
        per_category[cat] = compute_metrics(y_np[mask], preds[mask], class_labels)
    return overall, per_category


# --- 8. Main ---

df, feature_cols, ds_metadata = load_dataset(args.dataset_path)
y_all, num_classes, label_encoder = encode_labels(df, LABEL_COLUMN)
df = df.copy()
df["_y_encoded"] = y_all
class_labels = list(range(num_classes))
INPUT_DIM = len(feature_cols)

n_agents = sorted(
    df.loc[df["row_type"] == "perturbed", "agent_id"].dropna().astype(int).unique().tolist()
)
log(f"INPUT_DIM={INPUT_DIM}  num_classes={num_classes}  classes={list(label_encoder.classes_)}")
log(f"window_start={args.window_start}  window_rows={args.window_rows}")

originals_with_task = assign_task_windows(df, args.n_tasks, TASK0_ROW_FRACTION,
                                           args.window_start, args.window_rows)
window_rows = len(originals_with_task)          # resolved value, for the checkpoint key
perturbed_with_task = assign_perturbed_task_ids(df, originals_with_task)

rng = random.Random(SEED)
warnings_log = []
tasks = {}

for tid in range(args.n_tasks):
    task_orig = originals_with_task[originals_with_task["task_id"] == tid]
    benign_orig = task_orig[task_orig["label_binary"] == 0]
    malicious_orig = task_orig[task_orig["label_binary"] == 1]

    if len(benign_orig) == 0 or len(malicious_orig) == 0:
        msg = (f"[WARNING] Task {tid}: window has benign={len(benign_orig)}, "
               f"malicious={len(malicious_orig)} ORIGINAL rows.")
        log(msg); warnings_log.append(msg)

    if len(malicious_orig) >= 2:
        mal_train_orig, mal_test_orig = train_test_split(
            malicious_orig, test_size=TASK_TEST_FRAC, random_state=SEED
        )
    else:
        mal_train_orig, mal_test_orig = malicious_orig, malicious_orig.iloc[0:0]

    if len(benign_orig) >= 2:
        ben_train, ben_test = train_test_split(
            benign_orig, test_size=TASK_TEST_FRAC, random_state=SEED
        )
    else:
        ben_train, ben_test = benign_orig, benign_orig.iloc[0:0]

    ben_train = ben_train.copy(); ben_train["category"] = "benign"; ben_train["poison_agent_id"] = -1
    ben_test = ben_test.copy(); ben_test["category"] = "benign"; ben_test["poison_agent_id"] = -1

    clean_only_this_task = (tid == 0 and TASK0_CLEAN_ONLY)

    if clean_only_this_task:
        mal_train_final = mal_train_orig.copy()
        mal_train_final["category"] = "malicious_clean"; mal_train_final["poison_agent_id"] = -1
        mal_test_final = mal_test_orig.copy()
        mal_test_final["category"] = "malicious_clean"; mal_test_final["poison_agent_id"] = -1
        agent_for_task = None
    else:
        if AGENT_MODE == "uniform":
            agent_for_task = FIXED_AGENT_ID
        else:
            poison_task_index = tid - (1 if TASK0_CLEAN_ONLY else 0)
            agent_for_task = n_agents[poison_task_index % len(n_agents)]

        pert_pool = perturbed_with_task[perturbed_with_task["task_id"] == tid]
        train_pert_pool = pert_pool[pert_pool["source_row_id"].isin(mal_train_orig["orig_row_id"])]
        test_pert_pool = pert_pool[pert_pool["source_row_id"].isin(mal_test_orig["orig_row_id"])]

        mal_train_final, n_p, n_c = poison_and_diversify(
            mal_train_orig, train_pert_pool, agent_for_task, POISON_FRACTION,
            REQUIRE_EVASION_SUCCESS, ENSURE_CATEGORY_DIVERSITY, MIN_CATEGORY_COUNT, rng
        )
        mal_test_final, n_p_t, n_c_t = poison_and_diversify(
            mal_test_orig, test_pert_pool, agent_for_task, POISON_FRACTION,
            REQUIRE_EVASION_SUCCESS, ENSURE_CATEGORY_DIVERSITY, MIN_CATEGORY_COUNT, rng
        )
        log(f"Task {tid}: agent={agent_for_task} | "
            f"train poisoned={n_p} clean={n_c} | test poisoned={n_p_t} clean={n_c_t}")

    train_df = pd.concat([ben_train, mal_train_final], ignore_index=True)
    test_df = pd.concat([ben_test, mal_test_final], ignore_index=True)

    for split_name, split_df in (("train", train_df), ("test", test_df)):
        counts = split_df["category"].value_counts().to_dict()
        for cat in ["benign", "malicious_clean", "malicious_perturbed"]:
            if clean_only_this_task and cat == "malicious_perturbed":
                continue
            c = counts.get(cat, 0)
            if c < MIN_CATEGORY_COUNT:
                msg = (f"[WARNING] Task {tid} {split_name}: category '{cat}' has only "
                       f"{c} samples (< MIN_CATEGORY_COUNT={MIN_CATEGORY_COUNT}).")
                log(msg); warnings_log.append(msg)

    tasks[tid] = {"train": train_df, "test": test_df, "agent_for_task": agent_for_task}

scaler = StandardScaler()
scaler.fit(tasks[0]["train"][feature_cols].to_numpy(dtype=np.float32))


def to_tensors(split_df):
    X = np.clip(scaler.transform(split_df[feature_cols].to_numpy(dtype=np.float32)),
                -FEATURE_CLIP, FEATURE_CLIP)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(split_df["_y_encoded"].to_numpy(), dtype=torch.long)
    categories = split_df["category"].to_numpy()
    return X, y, categories


for tid in tasks:
    tasks[tid]["train_tensors"] = to_tensors(tasks[tid]["train"])
    tasks[tid]["test_tensors"] = to_tensors(tasks[tid]["test"])

model = ClassifierNN(INPUT_DIM, num_classes).to(DEVICE)
W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
per_task_checkpoints = []

# --- Task 0: train-or-load checkpoint (shared across every candidate in a generation) ---
Xtr0, ytr0, catr0 = tasks[0]["train_tensors"]
seen_labels = set(np.unique(ytr0.numpy()).tolist())
active_count0 = len(seen_labels)

if args.checkpoint and os.path.exists(args.checkpoint):
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert (ck['window_start'] == args.window_start and ck['window_rows'] == window_rows
            and ck['seed'] == SEED and ck['n_tasks'] == args.n_tasks), (
        "checkpoint was built for a different window/seed/n_tasks than requested; the outer "
        "loop must build a fresh checkpoint whenever the window, seed, or n_tasks changes"
    )
    model.load_state_dict(ck['model'])
    label_buffers.update(ck['label_buffers'])
    for buf in label_buffers.values():
        replay_buffer.extend(buf)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p_old_task[n.replace('.', '__')] = p.detach().clone()
    seen_labels = set(ck['seen_labels'])
    grad_steps0 = ck['grad_steps']
    log(f"Loaded task-0 checkpoint {args.checkpoint}")
else:
    log(f"Training task 0 (no checkpoint found) | train_samples={len(Xtr0)} | active_count={active_count0} ...")
    drop_last = (len(Xtr0) % BATCH_SIZE == 1)
    loader0 = data.DataLoader(data.TensorDataset(Xtr0, ytr0), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=drop_last)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    mask0 = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask0[:active_count0] = 0.0

    grad_steps0 = 0
    for _ in range(TASK0_EPOCHS):
        for v, l in loader0:
            v, l = v.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(v) + mask0, l)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in task 0 - training diverged")
            loss.backward()
            optimizer.step()
            grad_steps0 += 1

    for n, p in model.named_parameters():
        if p.requires_grad:
            p_old_task[n.replace('.', '__')] = p.detach().clone()
    update_buffer_madar(Xtr0, ytr0, catr0, model, DEVICE)

    if args.checkpoint:
        torch.save({
            'model': model.state_dict(), 'label_buffers': label_buffers,
            'window_start': args.window_start, 'window_rows': window_rows, 'seed': SEED,
            'n_tasks': args.n_tasks, 'seen_labels': list(seen_labels), 'grad_steps': grad_steps0,
        }, args.checkpoint)
        log(f"Saved task-0 checkpoint to {args.checkpoint}")

if args.make_checkpoint:
    log("Checkpoint ready; exiting (--make_checkpoint).")
    raise SystemExit(0)

teacher_model = copy.deepcopy(model); teacher_model.eval()
prev_active_count = active_count0

overall0, per_cat0 = evaluate_slice(model, *tasks[0]["test_tensors"][:2], tasks[0]["test_tensors"][2],
                                    active_count0, DEVICE, class_labels)
per_task_checkpoints.append({
    "task_id": 0, "grad_steps": grad_steps0, "train_samples": int(len(Xtr0)),
    "replay_buffer_composition": buffer_composition_summary(), "unlearning": None,
    "mean_per_task_accuracy": overall0["accuracy"], "pooled_accuracy": overall0["accuracy"],
})
log(f"  -> Task 0 acc: {overall0['accuracy']:.2f}%")

# --- CL + unlearning over tasks 1..n_tasks-1 ---
per_task_accs = []

for tid in range(1, args.n_tasks):
    Xtr, ytr, catr = tasks[tid]["train_tensors"]
    seen_labels.update(np.unique(ytr.numpy()).tolist())
    active_count = len(seen_labels)

    log(f"\n=== Training Task {tid} (MADAR: ER+KD+SI, {args.cl_iters} iters) | "
        f"train_samples={len(Xtr)} (+{len(replay_buffer)} replay) | active_count={active_count} ===")
    drop_last = (len(Xtr) % BATCH_SIZE == 1)
    loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                             shuffle=True, drop_last=drop_last)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)

    train_cl_er(model, teacher_model, optimizer, loader, args.cl_iters, active_count,
                prev_active_count, W, omega, p_old_task, tid, SI_C, DEVICE)

    for n, p in model.named_parameters():
        if p.requires_grad:
            n_key = n.replace('.', '__')
            p_current = p.detach().clone()
            omega[n_key] += W[n_key] / ((p_current - p_old_task[n_key]) ** 2 + SI_EPS)
            W[n_key].zero_()
            p_old_task[n_key] = p_current

    log(f" -> Initiating Unlearning Phase for Task {tid} (selector={args.selector}, alpha={args.alpha})...")
    budget_per_label = current_budget_per_label(np.unique(ytr.numpy()).tolist())
    retain_idx, forget_idx = split_forget_retain(
        Xtr, ytr, budget_per_label, model, active_count, DEVICE, args.selector, args.scorer_weights, tid
    )

    if forget_idx and retain_idx:
        retain_idx_t = torch.tensor(retain_idx, dtype=torch.long)
        forget_idx_t = torch.tensor(forget_idx, dtype=torch.long)
        retain_loader = data.DataLoader(
            data.TensorDataset(Xtr[retain_idx_t], ytr[retain_idx_t]), batch_size=BATCH_SIZE, shuffle=True
        )
        forget_loader = data.DataLoader(
            data.TensorDataset(Xtr[forget_idx_t], ytr[forget_idx_t]), batch_size=BATCH_SIZE, shuffle=True
        )

        model_pre_unlearn = copy.deepcopy(model)
        unlearn_teacher_guided(
            model=model, teacher_model=model_pre_unlearn,
            forget_loader=forget_loader, retain_loader=retain_loader,
            active_count=active_count, omega=omega, p_old_task=p_old_task,
            si_c=SI_C, epochs=args.unlearn_epochs, lr=UNLEARN_LR, alpha=args.alpha, device=DEVICE
        )

        f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, active_count, DEVICE)
        r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, active_count, DEVICE)
        unlearning_metrics = {"forget_set": f_m, "retain_set": r_m,
                              "n_forget": len(forget_idx), "n_retain": len(retain_idx)}
        log(f"    [Forget Set] Acc: {f_m['acc_before']:.2f}% -> {f_m['acc_after']:.2f}%")
        log(f"    [Retain Set] Acc: {r_m['acc_before']:.2f}% -> {r_m['acc_after']:.2f}%")

        Xtr_for_buffer = Xtr[retain_idx_t]
        ytr_for_buffer = ytr[retain_idx_t]
        catr_for_buffer = catr[retain_idx_t.numpy()]
    else:
        log("    [Unlearning] No forget set produced this task -- skipping unlearning phase.")
        unlearning_metrics = None
        Xtr_for_buffer, ytr_for_buffer, catr_for_buffer = Xtr, ytr, catr

    teacher_model = copy.deepcopy(model); teacher_model.eval()
    prev_active_count = active_count
    update_buffer_madar(Xtr_for_buffer, ytr_for_buffer, catr_for_buffer, model, DEVICE)

    checkpoint = {
        "task_id": tid, "train_samples": int(len(Xtr)),
        "replay_buffer_composition": buffer_composition_summary(), "unlearning": unlearning_metrics,
        "per_task_eval": {},
    }

    pooled_X, pooled_y, pooled_cat = [], [], []
    per_task_j_accs = []
    for j in range(tid + 1):
        Xte, yte, catje = tasks[j]["test_tensors"]
        overall, per_cat = evaluate_slice(model, Xte, yte, catje, active_count, DEVICE, class_labels)
        checkpoint["per_task_eval"][j] = {"overall": overall, "per_category": per_cat}
        per_task_j_accs.append(overall["accuracy"])
        pooled_X.append(Xte); pooled_y.append(yte); pooled_cat.append(catje)
        log(f"    [Eval Task {j}] acc={overall['accuracy']:.2f}%  n={overall['n_samples']}")

    pooled_X_cat = torch.cat(pooled_X)
    pooled_y_cat = torch.cat(pooled_y)
    pooled_cat_cat = np.concatenate(pooled_cat)
    pooled_overall, pooled_per_cat = evaluate_slice(
        model, pooled_X_cat, pooled_y_cat, pooled_cat_cat, active_count, DEVICE, class_labels
    )

    checkpoint["mean_per_task_accuracy"] = float(np.mean(per_task_j_accs))
    checkpoint["pooled_accuracy"] = pooled_overall["accuracy"]
    checkpoint["pooled_overall"] = pooled_overall
    checkpoint["pooled_per_category"] = pooled_per_cat
    per_task_accs.append(checkpoint["mean_per_task_accuracy"])

    log(f"  -> Mean-per-task acc: {checkpoint['mean_per_task_accuracy']:.2f}%  "
        f"| Pooled acc: {checkpoint['pooled_accuracy']:.2f}%")

    per_task_checkpoints.append(checkpoint)

reward = {
    'mean_acc': float(np.mean(per_task_accs)) if per_task_accs else 0.0,
    'final_acc': float(per_task_accs[-1]) if per_task_accs else 0.0,
    'per_task': per_task_accs,
    'selector': args.selector, 'seed': SEED,
    'window_start': args.window_start, 'window_rows': window_rows,
    'n_tasks': args.n_tasks, 'cl_iters': args.cl_iters,
    'unlearn_epochs': args.unlearn_epochs, 'alpha': args.alpha,
    'warnings': warnings_log,
    'checkpoints': per_task_checkpoints,
}
log(f"\nMean acc over tasks 1..{args.n_tasks - 1}: {reward['mean_acc']:.2f}% (final {reward['final_acc']:.2f}%)")
if args.out:
    with open(args.out, 'w') as f:
        json.dump(reward, f, indent=2, default=str)
