"""
Run with python clmd_cicids_unlearning_cd.py --seed 42 --log_name <>
Continual Learning Experiment (CICIDS MADAR + Targeted Unlearning)

Builds directly on clmd_cicids_madar_cd.py -- identical data construction, evaluation
methodology, and MADAR training machinery (label-keyed replay buffer, KD-teacher, Synaptic
Intelligence). See that file's docstring and clmd_cicids_naive_cd.py's for the full shared
rationale. This file adds a forget/retain split plus a targeted unlearning phase after each
CL task's regular training, adapted from clmdu_ember18_mk6B_cd.py with two changes forced by
the EMBER-to-CICIDS mapping (confirmed during planning):

  1. ONLY EMBER's "Option A" (split_option_a_scrub_leftovers) is implemented. EMBER's Option B
     was joint IsolationForest selection across the MULTIPLE NEW FAMILIES a class-incremental
     task introduces simultaneously -- a concept with no home here, since our tasks never
     introduce a new class at all (label stays {benign, malicious} throughout; only the time
     slice of data changes). Option A operates within an already-existing group (family in
     EMBER, label_binary here, matching the MADAR buffer's own keying) and has a clean,
     direct mapping. So: per-label (not per-category, no oracle knowledge of poisoning) raw-
     feature IsolationForest selects a retain set (anomalies + inliers), everything else in
     that label group becomes the forget set.

  2. The forget loss is redesigned. EMBER pushes predictions toward uniform specifically over
     `logits[:, prev_active_count:active_count]` -- the newly-introduced class range for the
     current task. Since active_count is flat at 2 from task 0 onward here, that range is
     always empty (`logits[:, 2:2]`) and the mechanism has no target. Redirected at the FULL
     active_count-class prediction instead: forget-set samples get pushed toward uniform
     across benign/malicious entirely, which is the more relevant question for a poisoning
     study anyway (can the model be made to stop being confidently correct about specific
     flagged samples, full stop).

Everything else carries over from mk6B: SI protection during unlearning, KD against a
model_pre_unlearn snapshot (not the CL replay teacher) for the retain loss, the alpha=0
control ablation (identical steps/optimizer/data flow, zero forget pressure -- isolates
"more training" from "actually targeting forgetting"), measure_unlearning_efficacy
(before/after accuracy, entropy raw + normalized, latent shift mean/median/max), and
refreshing the replay buffer from the RETAIN set only after unlearning (forgotten samples
don't re-enter memory). Dropped clmdu_ember18_mk6B_cd.py's plot_global_misclassifications_scatter
(family-task-boundary-specific, no clean analog) -- the per-task confusion matrices already
produced by every run cover the same diagnostic need.

Note: UNLEARN_ALPHA is a CONFIG global here, not a CLI flag like mk6B's --alpha -- kept
consistent with every other hyperparameter in this file and its siblings (naive/joint/MADAR),
all of which are top-of-file globals rather than CLI flags, per the project's established
convention. Only --seed and --log_name are CLI.

Run from the repo root (needs red_agent_perturbed_dataset.pkl under DATASET_PATH):
    python clmd_cicids_unlearning_cd.py --seed 42 --log_name unlearn_run1
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
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- CLI ---
parser = argparse.ArgumentParser(description="Continual Learning Experiment (CICIDS MADAR + Unlearning)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--log_name', type=str, default='cicids_unlearning_run',
                    help='Base name for the JSON results log and PNG plot written by this run')
args = parser.parse_args()

SEED = args.seed
LOG_NAME = args.log_name

# --- Enforce Determinism ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TRAIN_DEVICE = "cpu"           # see clmd_cicids_naive_cd.py -- "auto"/"cuda" can crash on
                                # newer GPUs than the installed torch build supports
DEVICE = torch.device(TRAIN_DEVICE)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
DATASET_PATH = "./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset2.pkl"
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30
CL_ITERS = 2000
BATCH_SIZE = 256

MEM_SIZE = 5000
MADAR_CONTAMINATION = 0.1      # rank-based selection; this only shifts IsolationForest's
                                # internal threshold, not which samples get chosen

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1

UNLEARN_EPOCHS = 3
UNLEARN_LR = 1e-4
UNLEARN_ALPHA = 0.2            # weight of the forget loss; 0.0 = extra-training control
                                # ablation (same steps/optimizer, no forget objective)

# --- Task construction (time-based, not class-based) ---
NUM_TASKS = 6
TASK0_ROW_FRACTION = 0.35
TASK_TEST_FRAC = 0.20

# --- Poisoning scenario ---
TASK0_CLEAN_ONLY = True     # FIXME 
AGENT_MODE = "mixed"      # FIXME 
FIXED_AGENT_ID = 0
REQUIRE_EVASION_SUCCESS = True
POISON_FRACTION = 0.8          # 0.0 always means "never poison", regardless of diversity flag

# --- Category diversity floor ---
ENSURE_CATEGORY_DIVERSITY = True
MIN_CATEGORY_COUNT = 5

HIDDEN_DIMS = [1024, 512, 256, 128]
EVAL_BATCH_SIZE = 512
# ==========================================


# --- 1. Data Loading ---

def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Point DATASET_PATH at the red_agent_perturbed_dataset.pkl "
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


# --- 2. Task Windowing (time-based) ---

def assign_task_windows(df: pd.DataFrame, num_tasks: int, task0_fraction: float) -> pd.DataFrame:
    originals = df[df["row_type"] == "original"].copy()
    originals = originals.sort_values("source_row_id").reset_index(drop=True)
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


# --- 3. Poisoning substitution + category diversity floor ---

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


# --- 4. Model Definition ---

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


# --- 5. MADAR replay buffer (keyed by label_binary, NOT category -- see clmd_cicids_madar_cd.py) ---

label_buffers = {}
replay_buffer = []


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    return F.kl_div(log_scores_norm, targets_norm, reduction='batchmean') * (T ** 2)


def current_budget_per_label(current_labels):
    """UNION, not sum: our label groups (0/1) are the same every task after task 0, unlike
    EMBER's ever-growing non-overlapping family set -- see clmd_cicids_madar_cd.py."""
    all_group_keys = set(label_buffers.keys()) | set(current_labels)
    return MEM_SIZE // max(len(all_group_keys), 1)


def _embed(model, Xt: torch.Tensor, device):
    loader = data.DataLoader(data.TensorDataset(Xt), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    parts = []
    with torch.no_grad():
        for (v,) in loader:
            _, latent = model(v.to(device), return_latent=True)
            parts.append(latent.cpu())
    return torch.cat(parts).numpy()


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, model, device):
    """Re-ranks the UNION of each label's EXISTING buffer entries (re-embedded under the
    CURRENT model, since weights have moved since they were last selected) and this task's
    new same-label samples, so exemplars from earlier tasks can survive across updates
    instead of being discarded every call (as a prior version of this function did -- see
    git history). See clmd_cicids_madar_cd.py for the full selection rationale."""
    global label_buffers, replay_buffer
    model.eval()

    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)
    L_np = _embed(model, X, device)
    latent_dim = L_np.shape[1]

    current_labels = np.unique(Y_np).tolist()
    budget_per_label = current_budget_per_label(current_labels)
    all_group_keys = set(label_buffers.keys()) | set(current_labels)

    for lbl in all_group_keys:
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


# --- 6. Forget/retain split (Option A only -- see module docstring) + unlearning phase ---

def split_option_a_scrub_leftovers(X: torch.Tensor, y: torch.Tensor, budget_per_label: int):
    """Per-LABEL (benign/malicious) split in RAW FEATURE space: keep the MADAR-style
    selection (anomalies + inliers, via IsolationForest) as retain, forget everything else.
    Returns (retain_idx, forget_idx) as plain index lists so the caller can also slice
    category/other metadata using the same split. No family or category oracle is used --
    the split sees only raw features and the binary label."""
    X_np = X.numpy()
    y_np = y.numpy()

    retain_idx_all = []
    forget_idx_all = []

    for lbl in np.unique(y_np):
        lbl_indices = np.where(y_np == lbl)[0]
        X_lbl = X_np[lbl_indices]
        n_select = min(budget_per_label, len(X_lbl))

        if n_select == len(X_lbl):
            retain_idx_all.extend(lbl_indices.tolist())
            continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(X_lbl)
        scores = iso.decision_function(X_lbl)
        sorted_local_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_local = sorted_local_idx[:half]
        inliers_local = sorted_local_idx[-(n_select - half):]

        selected_local = np.concatenate((anomalies_local, inliers_local))
        unselected_local = np.setdiff1d(np.arange(len(X_lbl)), selected_local)

        retain_idx_all.extend(lbl_indices[selected_local].tolist())
        forget_idx_all.extend(lbl_indices[unselected_local].tolist())

    return retain_idx_all, forget_idx_all


def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, active_count,
                           omega, p_old_task, si_c, epochs, lr, alpha, device):
    """Pushes the forget set's FULL active_count-class prediction toward uniform (see module
    docstring for why this differs from mk6B's prev_active_count:active_count sub-range),
    anchored by ground-truth CE + KD on retain/replay data, under SI protection. alpha=0 is
    the extra-training control: identical steps/optimizer/data flow, no forget objective."""
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

            # --- 1. TARGETED ENTROPY (Forget) -- full active_count-class distribution ---
            logits = model(f_v)[:, :active_count]
            log_probs = F.log_softmax(logits, dim=1)
            uniform_probs = torch.ones_like(log_probs) / active_count
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
                teacher_logits = teacher_model(r_v)[:, :active_count]

            student_logits_raw = model(r_v)
            student_logits_masked = student_logits_raw + mask
            student_logits_active = student_logits_raw[:, :active_count]

            retain_ce = nn.CrossEntropyLoss()(student_logits_masked, r_l)
            retain_kd = loss_fn_kd(student_logits_active, teacher_logits, T=KD_TEMP)
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


# --- 7. Eval (identical methodology to naive/joint/MADAR) ---

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


# --- 8. Main Execution ---

if __name__ == "__main__":
    df, feature_cols, ds_metadata = load_dataset(DATASET_PATH)
    y_all, num_classes, label_encoder = encode_labels(df, LABEL_COLUMN)
    df = df.copy()
    df["_y_encoded"] = y_all
    class_labels = list(range(num_classes))
    INPUT_DIM = len(feature_cols)

    n_agents = sorted(
        df.loc[df["row_type"] == "perturbed", "agent_id"].dropna().astype(int).unique().tolist()
    )
    print(f"INPUT_DIM={INPUT_DIM}  num_classes={num_classes}  classes={list(label_encoder.classes_)}")
    print(f"Detected {len(n_agents)} red agents in dataset: {n_agents}")

    originals_with_task = assign_task_windows(df, NUM_TASKS, TASK0_ROW_FRACTION)
    perturbed_with_task = assign_perturbed_task_ids(df, originals_with_task)

    rng = random.Random(SEED)
    warnings_log = []
    tasks = {}

    for tid in range(NUM_TASKS):
        task_orig = originals_with_task[originals_with_task["task_id"] == tid]
        benign_orig = task_orig[task_orig["label_binary"] == 0]
        malicious_orig = task_orig[task_orig["label_binary"] == 1]

        if len(benign_orig) == 0 or len(malicious_orig) == 0:
            msg = (f"[WARNING] Task {tid}: window has benign={len(benign_orig)}, "
                   f"malicious={len(malicious_orig)} ORIGINAL rows.")
            print(msg); warnings_log.append(msg)

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
            print(f"Task {tid}: agent={agent_for_task} | "
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
                    print(msg); warnings_log.append(msg)

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

    # --- MADAR + Unlearning training loop ---
    model = ClassifierNN(INPUT_DIM, num_classes).to(DEVICE)
    W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    teacher_model = None
    prev_active_count = 0
    seen_labels = set()

    run_config = {
        "DATASET_PATH": DATASET_PATH, "LABEL_COLUMN": LABEL_COLUMN, "FEATURE_CLIP": FEATURE_CLIP,
        "TASK0_EPOCHS": TASK0_EPOCHS, "CL_ITERS": CL_ITERS, "BATCH_SIZE": BATCH_SIZE,
        "MEM_SIZE": MEM_SIZE, "MADAR_CONTAMINATION": MADAR_CONTAMINATION,
        "KD_TEMP": KD_TEMP, "SI_C": SI_C, "SI_EPS": SI_EPS,
        "UNLEARN_EPOCHS": UNLEARN_EPOCHS, "UNLEARN_LR": UNLEARN_LR, "UNLEARN_ALPHA": UNLEARN_ALPHA,
        "NUM_TASKS": NUM_TASKS, "TASK0_ROW_FRACTION": TASK0_ROW_FRACTION, "TASK_TEST_FRAC": TASK_TEST_FRAC,
        "TASK0_CLEAN_ONLY": TASK0_CLEAN_ONLY, "AGENT_MODE": AGENT_MODE, "FIXED_AGENT_ID": FIXED_AGENT_ID,
        "REQUIRE_EVASION_SUCCESS": REQUIRE_EVASION_SUCCESS, "POISON_FRACTION": POISON_FRACTION,
        "ENSURE_CATEGORY_DIVERSITY": ENSURE_CATEGORY_DIVERSITY, "MIN_CATEGORY_COUNT": MIN_CATEGORY_COUNT,
        "SEED": SEED, "N_AGENTS_DETECTED": n_agents, "NUM_CLASSES": num_classes,
        "CLASS_LABELS_ORIGINAL": [str(c) for c in label_encoder.classes_], "INPUT_DIM": INPUT_DIM,
        "STRATEGY": "madar_er_kd_si_plus_unlearning_option_a",
    }

    per_task_checkpoints = []

    for tid in range(NUM_TASKS):
        Xtr, ytr, catr = tasks[tid]["train_tensors"]
        seen_labels.update(np.unique(ytr.numpy()).tolist())
        active_count = len(seen_labels)

        if tid == 0:
            print(f"\n=== Training Task {tid} (pure supervised, no replay/unlearning yet) | "
                  f"train_samples={len(Xtr)} | active_count={active_count} ===")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
            mask[:active_count] = 0.0

            grad_steps = 0
            for _ in range(TASK0_EPOCHS):
                for v, l in loader:
                    v, l = v.to(DEVICE), l.to(DEVICE)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(v) + mask, l)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at task {tid} - training diverged")
                    loss.backward()
                    optimizer.step()
                    grad_steps += 1

            for n, p in model.named_parameters():
                if p.requires_grad:
                    p_old_task[n.replace('.', '__')] = p.detach().clone()
            teacher_model = copy.deepcopy(model); teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(Xtr, ytr, catr, model, DEVICE)
            unlearning_metrics = None

        else:
            print(f"\n=== Training Task {tid} (MADAR: ER+KD+SI, {CL_ITERS} iters) | "
                  f"train_samples={len(Xtr)} (+{len(replay_buffer)} replay) | active_count={active_count} ===")
            drop_last = (len(Xtr) % BATCH_SIZE == 1)
            loader = data.DataLoader(data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=drop_last)
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            grad_steps = CL_ITERS

            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, active_count,
                        prev_active_count, W, omega, p_old_task, tid, SI_C, DEVICE)

            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_current = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_current - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()
                    p_old_task[n_key] = p_current

            print(f" -> Initiating Unlearning Phase for Task {tid} (alpha={UNLEARN_ALPHA})...")
            budget_per_label = current_budget_per_label(np.unique(ytr.numpy()).tolist())
            retain_idx, forget_idx = split_option_a_scrub_leftovers(Xtr, ytr, budget_per_label)

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
                    si_c=SI_C, epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA, device=DEVICE
                )
                grad_steps += UNLEARN_EPOCHS * len(forget_loader)

                f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, active_count, DEVICE)
                r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, active_count, DEVICE)
                unlearning_metrics = {"forget_set": f_m, "retain_set": r_m,
                                      "n_forget": len(forget_idx), "n_retain": len(retain_idx)}

                for tag, m in (("Forget Set", f_m), ("Retain Set", r_m)):
                    print(f"    [{tag}] Acc: {m['acc_before']:.2f}% -> {m['acc_after']:.2f}%, "
                          f"Entropy: {m['entropy']:.3f} (norm: {m['entropy_norm']:.3f}), "
                          f"Shift mean/med/max: {m['shift_mean']:.3f}/{m['shift_median']:.3f}/{m['shift_max']:.1f}")

                # Buffer refreshed from the RETAIN set only -- forgotten samples don't re-enter memory
                Xtr_for_buffer = Xtr[retain_idx_t]
                ytr_for_buffer = ytr[retain_idx_t]
                catr_for_buffer = catr[retain_idx_t.numpy()]
            else:
                print("    [Unlearning] No forget set produced this task -- skipping unlearning phase.")
                unlearning_metrics = None
                Xtr_for_buffer, ytr_for_buffer, catr_for_buffer = Xtr, ytr, catr

            teacher_model = copy.deepcopy(model); teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(Xtr_for_buffer, ytr_for_buffer, catr_for_buffer, model, DEVICE)

        buffer_summary = buffer_composition_summary()
        print(f"    [Buffer] composition: {buffer_summary}")

        checkpoint = {
            "task_id": tid, "grad_steps": grad_steps, "train_samples": int(len(Xtr)),
            "replay_buffer_composition": buffer_summary, "unlearning": unlearning_metrics,
            "per_task_eval": {},
        }

        pooled_X, pooled_y, pooled_cat = [], [], []
        per_task_accs = []
        for j in range(tid + 1):
            Xte, yte, catje = tasks[j]["test_tensors"]
            overall, per_cat = evaluate_slice(model, Xte, yte, catje, active_count, DEVICE, class_labels)
            checkpoint["per_task_eval"][j] = {"overall": overall, "per_category": per_cat}
            per_task_accs.append(overall["accuracy"])
            pooled_X.append(Xte); pooled_y.append(yte); pooled_cat.append(catje)
            print(f"    [Eval Task {j}] acc={overall['accuracy']:.2f}%  n={overall['n_samples']}")

        pooled_X_cat = torch.cat(pooled_X)
        pooled_y_cat = torch.cat(pooled_y)
        pooled_cat_cat = np.concatenate(pooled_cat)
        pooled_overall, pooled_per_cat = evaluate_slice(
            model, pooled_X_cat, pooled_y_cat, pooled_cat_cat, active_count, DEVICE, class_labels
        )

        checkpoint["mean_per_task_accuracy"] = float(np.mean(per_task_accs))
        checkpoint["pooled_accuracy"] = pooled_overall["accuracy"]
        checkpoint["pooled_overall"] = pooled_overall
        checkpoint["pooled_per_category"] = pooled_per_cat

        print(f"  -> Mean-per-task acc: {checkpoint['mean_per_task_accuracy']:.2f}%  "
              f"| Pooled acc: {checkpoint['pooled_accuracy']:.2f}%")

        per_task_checkpoints.append(checkpoint)

    log_path = f"{LOG_NAME}_MadarUNLEARN_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}_seed{SEED}.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": run_config,
            "warnings": warnings_log,
            "results": per_task_checkpoints,
        }, f, indent=2, default=str)
    print(f"\nSaved log to {log_path}")

    plt.figure(figsize=(10, 6))
    plt.plot([c["pooled_accuracy"] for c in per_task_checkpoints], marker='o', label='Pooled Acc (all seen tasks)')
    plt.plot([c["mean_per_task_accuracy"] for c in per_task_checkpoints], marker='x', label='Mean Per-Task Acc')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'CICIDS MADAR + Unlearning ({LOG_NAME}, seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{LOG_NAME}_MadarUNLEARN_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}.png")
    print(f"Saved plot to {LOG_NAME}_seed{SEED}.png")
