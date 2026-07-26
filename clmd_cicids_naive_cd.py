"""
Run with: python clmd_cicids_naive_cd.py --seed 42 --log_name 0725_pert2
Continual Learning Experiment (CICIDS Naive Baseline)

Adapts the EMBER naive baseline (clmd_ember18_naive_cd.py) to the perturbed CICIDS
dataset produced by generate_red_agent_dataset.py. Key differences from the EMBER
version, per the planning discussion:

  - Data source: red_agent_perturbed_dataset.pkl instead of raw EMBER + embersim.
  - Tasks are TIME-based windows (sorted by source_row_id), not class-based. Task 0
    gets a larger share of rows than the rest (TASK0_ROW_FRACTION).
  - Classification target is read from LABEL_COLUMN and label-encoded generically, so
    swapping in a future family-labeled dataset requires no code changes -- num_classes
    and the active-class mask are derived from the data, not hardcoded.
  - Poisoning: within each task (except an optional clean-only task 0), a POISON_FRACTION
    share of malicious originals are replaced by one red agent's perturbed version of that
    same sample -- never both. AGENT_MODE picks whether that agent is fixed ("uniform") or
    cycles per task ("mixed"). A perturbed row always lands in the same task as its source
    sample (natural timestamp placement).
  - ENSURE_CATEGORY_DIVERSITY guarantees benign / malicious_clean / malicious_perturbed are
    all present (at least MIN_CATEGORY_COUNT each) in every task's train AND test split,
    except malicious_perturbed in a TASK0_CLEAN_ONLY task 0, where it's intentionally absent.
  - Testing: each task carves its own local test subbatch (TASK_TEST_FRAC) before training.
    After training task N, the model is evaluated separately against tasks 0..N's own test
    subbatches (a full per-task accuracy row, not one blended number), plus a pooled-union
    accuracy over all of them and the mean of the per-task accuracies.
  - Every checkpoint's confusion matrix and precision/recall/F1 are computed both overall
    and sliced by category (benign / malicious_clean / malicious_perturbed), so poisoning's
    effect on detection is visible directly rather than hidden inside one aggregate number.
  - Every run writes one JSON log (name set via --log_name) containing the full config
    snapshot and results, so results are self-documenting without re-parsing stdout.

Run from the repo root (needs red_agent_perturbed_dataset.pkl under DATASET_PATH):
    python clmd_cicids_naive_cd.py --seed 42 --log_name naive_run1
"""

import os
import json
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- CLI ---
parser = argparse.ArgumentParser(description="Continual Learning Experiment (CICIDS Naive Baseline)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--log_name', type=str, default='cicids_naive_run',
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

TRAIN_DEVICE = "cpu"           # "auto"/"cuda" would pick CUDA whenever torch.cuda.is_available()
                                # is True, but that only checks a GPU is present -- not that this
                                # torch build has kernels for its compute capability. On a newer
                                # GPU than the installed torch supports, that crashes instead of
                                # falling back to CPU. This workload is small enough that CPU is fine.
DEVICE = torch.device(TRAIN_DEVICE)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
DATASET_PATH = "./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset2.pkl"

# Which column the model is trained/evaluated against. Binary today; point this at
# e.g. "attack_family" once that data exists -- num_classes and the active-class mask
# are derived from whatever column this names, no other code changes needed.
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
EPOCHS_PER_TASK = 30
BATCH_SIZE = 256

# --- Task construction (time-based, not class-based) ---
NUM_TASKS = 6 
TASK0_ROW_FRACTION = 0.35      # share of ORIGINAL (benign+clean-malicious) rows, sorted by
                                # source_row_id, assigned to task 0; remaining rows split
                                # evenly across the other NUM_TASKS - 1 tasks
TASK_TEST_FRAC = 0.20          # each task's own local held-out test subbatch

# --- Poisoning scenario ---
TASK0_CLEAN_ONLY = False  # FIXME      # task 0 = benign + clean malicious only, zero perturbed rows
AGENT_MODE = "mixed"         # "uniform" -> always FIXED_AGENT_ID; "mixed" -> cycles agents per task
FIXED_AGENT_ID = 0
REQUIRE_EVASION_SUCCESS = True # only substitute a perturbed row if it actually evaded the
                                # red agent's target classifier (evasion_success == True)
POISON_FRACTION = 0.8 # FIXME 0.8 for full run          # share of a poisoned task's malicious originals replaced by
                                # their assigned agent's perturbed version; the rest stay
                                # malicious_clean (never both for the same sample)

# --- Category diversity floor ---
ENSURE_CATEGORY_DIVERSITY = True   # guarantee benign / malicious_clean / malicious_perturbed
                                    # are all present per task's train AND test split (except
                                    # malicious_perturbed in a TASK0_CLEAN_ONLY task 0, which
                                    # is intentionally absent there)
MIN_CATEGORY_COUNT = 5 #FIXME increase later 5             # floor per category, per split

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
    """Assigns a task_id to every ORIGINAL row (benign + clean malicious), sorted by
    source_row_id. Task 0 gets task0_fraction of rows; the rest split evenly."""
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
    """A perturbed row inherits its source malicious sample's task_id (natural timestamp
    placement -- no separate poisoning schedule)."""
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
    """
    For one task's malicious ORIGINAL rows (already split into this task's train or test
    partition), decides -- per sample -- clean vs. replaced-by-agent_for_task's perturbation.
    Never both for the same sample. Returns the combined DataFrame plus poison/clean counts.
    """
    cand = perturbed_pool[perturbed_pool["agent_id"] == agent_for_task]
    if require_evasion_success:
        cand = cand[cand["evasion_success"] == True]  # noqa: E712
    eligible_ids = set(cand["source_row_id"].tolist())

    all_ids = malicious_orig_subset["orig_row_id"].tolist()
    eligible = [i for i in all_ids if i in eligible_ids]
    n_elig = len(eligible)

    n_poison = int(round(poison_fraction * n_elig))
    if poison_fraction <= 0.0:
        # Explicit "never poison" override: an all-clean control run must stay all-clean
        # even with ENSURE_CATEGORY_DIVERSITY on, which otherwise exists specifically to
        # force some minimum perturbed presence.
        n_poison = 0
    elif ensure_diversity and n_elig > 0:
        if n_elig >= 2 * min_category_count:
            n_poison = max(min_category_count, min(n_poison, n_elig - min_category_count))
        else:
            # Not enough eligible perturbed candidates to guarantee both floors -- best
            # effort even split; the caller's post-hoc check will warn if still short.
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


# --- 5. Train / Eval ---

def train_task_naive(model, X, y, epochs, batch_size, active_count, device):
    drop_last = (len(X) % batch_size == 1)  # avoid a final batch of size 1 crashing BatchNorm
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size,
                             shuffle=True, drop_last=drop_last)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    mask = torch.full((model.fc_last.out_features,), -1e9).to(device)
    mask[:active_count] = 0.0

    grad_steps = 0
    for _ in range(epochs):
        for v, l in loader:
            v, l = v.to(device), l.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(v) + mask, l)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss - training diverged")
            loss.backward()
            optimizer.step()
            grad_steps += 1
    return grad_steps


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


# --- 6. Main Execution ---

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
        # NOTE: label_binary (not LABEL_COLUMN) drives the benign/malicious split used for
        # poisoning eligibility -- poisoning is a malicious-traffic concept regardless of
        # which column the classifier is trained against (binary today, family later).
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
                    continue  # intentionally absent
                c = counts.get(cat, 0)
                if c < MIN_CATEGORY_COUNT:
                    msg = (f"[WARNING] Task {tid} {split_name}: category '{cat}' has only "
                           f"{c} samples (< MIN_CATEGORY_COUNT={MIN_CATEGORY_COUNT}).")
                    print(msg); warnings_log.append(msg)

        tasks[tid] = {"train": train_df, "test": test_df, "agent_for_task": agent_for_task}

    # --- Scaler: fit on task 0's TRAIN split only, applied as-is to every other task/split ---
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

    # --- Training + per-task-matrix evaluation ---
    model = ClassifierNN(INPUT_DIM, num_classes).to(DEVICE)
    seen_labels = set()

    run_config = {
        "DATASET_PATH": DATASET_PATH, "LABEL_COLUMN": LABEL_COLUMN, "FEATURE_CLIP": FEATURE_CLIP,
        "EPOCHS_PER_TASK": EPOCHS_PER_TASK, "BATCH_SIZE": BATCH_SIZE, "NUM_TASKS": NUM_TASKS,
        "TASK0_ROW_FRACTION": TASK0_ROW_FRACTION, "TASK_TEST_FRAC": TASK_TEST_FRAC,
        "TASK0_CLEAN_ONLY": TASK0_CLEAN_ONLY, "AGENT_MODE": AGENT_MODE, "FIXED_AGENT_ID": FIXED_AGENT_ID,
        "REQUIRE_EVASION_SUCCESS": REQUIRE_EVASION_SUCCESS, "POISON_FRACTION": POISON_FRACTION,
        "ENSURE_CATEGORY_DIVERSITY": ENSURE_CATEGORY_DIVERSITY, "MIN_CATEGORY_COUNT": MIN_CATEGORY_COUNT,
        "SEED": SEED, "N_AGENTS_DETECTED": n_agents, "NUM_CLASSES": num_classes,
        "CLASS_LABELS_ORIGINAL": [str(c) for c in label_encoder.classes_], "INPUT_DIM": INPUT_DIM,
    }

    per_task_checkpoints = []

    for tid in range(NUM_TASKS):
        Xtr, ytr, catr = tasks[tid]["train_tensors"]
        seen_labels.update(np.unique(ytr.numpy()).tolist())
        active_count = len(seen_labels)

        print(f"\n=== Training Task {tid} | train_samples={len(Xtr)} | active_count={active_count} ===")
        grad_steps = train_task_naive(model, Xtr, ytr, EPOCHS_PER_TASK, BATCH_SIZE, active_count, DEVICE)

        checkpoint = {
            "task_id": tid, "grad_steps": grad_steps, "train_samples": int(len(Xtr)),
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

    # --- Save log + plot ---
    log_path = f"{LOG_NAME}_naive_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}.json"
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
    plt.title(f'CICIDS Naive Baseline ({LOG_NAME}, seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{LOG_NAME}_naive_Task0Clean_{TASK0_CLEAN_ONLY}_{AGENT_MODE}.png")
    print(f"Saved plot to {LOG_NAME}_seed{SEED}.png")
