"""
Run with python clmd_cicids_madar_cd.py --seed 42 --log_name 0725_0.6_perturbed
Continual Learning Experiment (CICIDS MADAR: Experience Replay + Knowledge Distillation +
Synaptic Intelligence)

Same CICIDS adaptation as clmd_cicids_naive_cd.py / clmd_cicids_joint_cd.py for data
construction and evaluation (time-based tasks, the poisoning mechanic, category diversity
floor, per-task test subbatches, per-task accuracy matrix, category-sliced confusion
matrices) -- see clmd_cicids_naive_cd.py's docstring for the full rationale. This file only
replaces the TRAINING strategy, adapting clmd_ember18_mk9_cd.py's MADAR machinery.

Key mapping decision (confirmed during planning): EMBER's replay buffer is keyed by malware
FAMILY, which is simultaneously the training label, the buffer partition key, AND the
diversity-selection group -- one overloaded axis. We deliberately do NOT replace family with
"category" (benign/malicious_clean/malicious_perturbed): the buffer key is just
LABEL_BINARY (benign vs malicious), matching EMBER's collapsed label=key design. This means
the IsolationForest selection within the "malicious" group runs over a MIX of clean and
perturbed samples with zero oracle knowledge of which is which -- whatever the buffer ends
up protecting or discarding from that mix is a genuine emergent property of the anomaly
detection, not something fed in. This mirrors mk6B's unlearning forget/retain split, which
also uses IsolationForest on raw/latent features with no family or category oracle -- so the
eventual MADAR+unlearning experiment tests whether the combined mechanism can discover which
samples matter using only anomaly signal, exactly as intended.

`category` is still tracked, but ONLY for two purposes, both entirely downstream of training:
  1. The same per-category confusion-matrix evaluation slicing naive/joint already do.
  2. A new diagnostic: after each buffer update, log what fraction of each label's buffer
     slot is benign / malicious_clean / malicious_perturbed -- purely observational, computed
     from ground truth we already have, never fed back into selection or training.

Run from the repo root (needs red_agent_perturbed_dataset.pkl under DATASET_PATH):
    python clmd_cicids_madar_cd.py --seed 42 --log_name madar_run1
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

# --- CLI ---
parser = argparse.ArgumentParser(description="Continual Learning Experiment (CICIDS MADAR: ER+KD+SI)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--log_name', type=str, default='cicids_madar_run',
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
DATASET_PATH = "./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset_4500.pkl"
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30              # task 0 is pure supervised (no replay yet) -- epoch-based
CL_ITERS = 2000                # tasks 1+ mix current-task and replay-buffer batches via two
                                # independently-cycling loaders, so a fixed iteration count is
                                # used instead of "epochs over the current task's own loader"
BATCH_SIZE = 256

MEM_SIZE = 800                # total replay buffer budget, split across label groups
MADAR_CONTAMINATION = 0.1      # only shifts IsolationForest's decision threshold; selection
                                # is purely rank-based on decision_function scores, so this
                                # does not affect which samples get chosen (kept for API compat)

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1

# --- Task construction (time-based, not class-based) ---
NUM_TASKS = 6
TASK0_ROW_FRACTION = 0.35
TASK_TEST_FRAC = 0.20

# --- Poisoning scenario ---
TASK0_CLEAN_ONLY = True         # FIXME 
AGENT_MODE = "mixed"          # FIXME 
FIXED_AGENT_ID = 0
REQUIRE_EVASION_SUCCESS = True
POISON_FRACTION = 0.6          # 0.0 always means "never poison", regardless of diversity flag

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
    """Sorted by TIMESTAMP (true chronological order) -- source_row_id/orig_row_id are
    per-day counters that reset to 0 for each source_range, so sorting by them directly
    would interleave rows from different days by coincidence of row number instead of time."""
    originals = df[df["row_type"] == "original"].copy()
    originals = originals.sort_values("timestamp").reset_index(drop=True)
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
    """orig_row_id/source_row_id are only unique WITHIN one day's source file (they reset to
    0 per source_range), so a multi-day dataset needs (source_range, row_id) as the join
    key -- row_id alone collides across days."""
    task_lookup = originals_with_task.set_index(["source_range", "orig_row_id"])["task_id"]
    perturbed = df[df["row_type"] == "perturbed"].copy()
    keys = pd.MultiIndex.from_arrays([perturbed["source_range"], perturbed["source_row_id"]])
    perturbed["task_id"] = task_lookup.reindex(keys).to_numpy()
    perturbed = perturbed.dropna(subset=["task_id"]).copy()
    perturbed["task_id"] = perturbed["task_id"].astype(int)
    return perturbed


# --- 3. Poisoning substitution + category diversity floor ---

def poison_and_diversify(malicious_orig_subset: pd.DataFrame, perturbed_pool: pd.DataFrame,
                          agent_for_task: int, poison_fraction: float, require_evasion_success: bool,
                          ensure_diversity: bool, min_category_count: int, rng: random.Random):
    """Identity here keys on (source_range, row_id), not the bare row id: orig_row_id/
    source_row_id reset per source_range (day), and a single task CAN span more than one
    day's rows, so a bare-id match risks pairing a Tuesday original with a Wednesday
    perturbation that happens to share the same in-day row number."""
    cand = perturbed_pool[perturbed_pool["agent_id"] == agent_for_task]
    if require_evasion_success:
        cand = cand[cand["evasion_success"] == True]  # noqa: E712
    eligible_ids = set(zip(cand["source_range"], cand["source_row_id"]))

    all_ids = list(zip(malicious_orig_subset["source_range"], malicious_orig_subset["orig_row_id"]))
    eligible = [k for k in all_ids if k in eligible_ids]
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

    mal_keys = pd.Series(all_ids, index=malicious_orig_subset.index)
    clean_rows = malicious_orig_subset[mal_keys.isin(clean_ids)].copy()
    clean_rows["category"] = "malicious_clean"
    clean_rows["poison_agent_id"] = -1

    cand_keys = pd.Series(list(zip(cand["source_range"], cand["source_row_id"])), index=cand.index)
    poison_rows = cand[cand_keys.isin(poison_ids)].copy()
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


# --- 5. MADAR replay buffer (keyed by label_binary, NOT category -- see module docstring) ---

label_buffers = {}     # {0: [(X, y, category), ...], 1: [...]}
replay_buffer = []      # flattened label_buffers.values()


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
    with torch.no_grad():
        for (v,) in loader:
            _, latent = model(v.to(device), return_latent=True)
            parts.append(latent.cpu())
    return torch.cat(parts).numpy()


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, model, device):
    """Selects, per label (0/1), an interleaved anomalies+inliers sample (by IsolationForest
    decision_function over the model's LATENT space) to keep in that label's buffer slot.
    Re-ranks the UNION of that label's EXISTING buffer entries (re-embedded under the CURRENT
    model, since weights have moved since they were last selected) and this task's new
    same-label samples, so exemplars from earlier tasks can survive across updates instead of
    being discarded every call (as a prior version of this function did -- see git history).
    category is attached to each stored sample purely for the post-hoc composition diagnostic
    -- it plays no role in selection, which sees only the latent vectors."""
    global label_buffers, replay_buffer
    model.eval()

    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)
    L_np = _embed(model, X, device)
    latent_dim = L_np.shape[1]

    current_labels = np.unique(Y_np).tolist()
    # UNION, not sum: unlike EMBER's ever-growing, non-overlapping family set, our label
    # groups (0/1) are the SAME every task after task 0 -- summing len(label_buffers) +
    # len(current_labels) would double-count and halve the intended budget.
    all_group_keys = set(label_buffers.keys()) | set(current_labels)
    budget_per_label = MEM_SIZE // max(len(all_group_keys), 1)

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
    """Diagnostic only: what fraction of each label's buffer slot is benign / malicious_clean
    / malicious_perturbed right now. Never used by training or selection."""
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


# --- 6. Eval (identical methodology to naive/joint) ---

def compute_metrics(y_true, y_pred, class_labels):
    acc = float(accuracy_score(y_true, y_pred)) * 100
    bala_acc = float(balanced_accuracy_score(y_true, y_pred)) * 100
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
        "balanced_accuracy": bala_acc
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


# --- 7. Main Execution ---

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

            # source_row_id resets per source_range (day) -- must key on (source_range, row_id),
            # not the bare id, or a perturbed row from one day can land in the wrong split
            # because another day's train/test malicious sample shares its in-day row number.
            pert_pool = perturbed_with_task[perturbed_with_task["task_id"] == tid]
            pert_pool_keys = pd.Series(list(zip(pert_pool["source_range"], pert_pool["source_row_id"])),
                                        index=pert_pool.index)
            train_keys = set(zip(mal_train_orig["source_range"], mal_train_orig["orig_row_id"]))
            test_keys = set(zip(mal_test_orig["source_range"], mal_test_orig["orig_row_id"]))
            train_pert_pool = pert_pool[pert_pool_keys.isin(train_keys)]
            test_pert_pool = pert_pool[pert_pool_keys.isin(test_keys)]

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

    # --- MADAR training loop ---
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
        "NUM_TASKS": NUM_TASKS, "TASK0_ROW_FRACTION": TASK0_ROW_FRACTION, "TASK_TEST_FRAC": TASK_TEST_FRAC,
        "TASK0_CLEAN_ONLY": TASK0_CLEAN_ONLY, "AGENT_MODE": AGENT_MODE, "FIXED_AGENT_ID": FIXED_AGENT_ID,
        "REQUIRE_EVASION_SUCCESS": REQUIRE_EVASION_SUCCESS, "POISON_FRACTION": POISON_FRACTION,
        "ENSURE_CATEGORY_DIVERSITY": ENSURE_CATEGORY_DIVERSITY, "MIN_CATEGORY_COUNT": MIN_CATEGORY_COUNT,
        "SEED": SEED, "N_AGENTS_DETECTED": n_agents, "NUM_CLASSES": num_classes,
        "CLASS_LABELS_ORIGINAL": [str(c) for c in label_encoder.classes_], "INPUT_DIM": INPUT_DIM,
        "STRATEGY": "madar_er_kd_si",
    }

    per_task_checkpoints = []

    for tid in range(NUM_TASKS):
        Xtr, ytr, catr = tasks[tid]["train_tensors"]
        seen_labels.update(np.unique(ytr.numpy()).tolist())
        active_count = len(seen_labels)

        if tid == 0:
            print(f"\n=== Training Task {tid} (pure supervised, no replay yet) | "
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

            teacher_model = copy.deepcopy(model); teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(Xtr, ytr, catr, model, DEVICE)

        buffer_summary = buffer_composition_summary()
        print(f"    [Buffer] composition: {buffer_summary}")

        checkpoint = {
            "task_id": tid, "grad_steps": grad_steps, "train_samples": int(len(Xtr)),
            "replay_buffer_composition": buffer_summary, "per_task_eval": {},
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
        print(f"  -> Balanced accuracy: {checkpoint['pooled_overall']['balanced_accuracy']:.2f}%")

        per_task_checkpoints.append(checkpoint)

    log_path = f"{LOG_NAME}_MADAR_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}.json"
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
    plt.title(f'CICIDS MADAR ER+KD+SI ({LOG_NAME}, seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{LOG_NAME}_MADAR_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}.png")
    print(f"Saved plot to {LOG_NAME}_seed{SEED}.png")
