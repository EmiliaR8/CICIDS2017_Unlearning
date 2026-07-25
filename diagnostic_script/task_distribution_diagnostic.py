"""
Task distribution diagnostic: how different is task 1's data from every other task's data?

Three independent metric families, each computed comparing task 1 (task_id=0) against every
other task (task_id=1..NUM_TASKS-1):

  A. Gradient similarity, NO reference-point fix (as originally proposed): train a classifier
     on task 1's data, then -- without any further training -- compute the gradient of each
     other task's loss at those SAME (task-1-trained) weights, and compare it via cosine
     similarity to task 1's own gradient at that same point. Included as requested, with the
     caveat noted during planning: task 1's own gradient here will be close to zero since
     training just converged on it, so this method's task-1-reference signal is weak by
     construction -- kept for comparison against method B below.

  B. Gradient similarity, WITH reference-point fix: both task 1's and the other task's
     gradients are computed at a shared, untrained reference point (the model at random
     initialization, before any training at all). Avoids method A's near-zero-gradient issue.

  C. Direct distributional distance: per-feature KL divergence and Wasserstein distance
     between task 1's and the other task's feature distributions, no model involved at all.
     Adapted directly from adaptor_env.py's DriftAdaptationEnv.compute_kl_divergence /
     compute_wasserstein_distance (same per-feature-histogram methodology).

Run twice per invocation: once using only benign + clean-malicious samples ("clean" mode),
once with malicious samples replaced by one agent's perturbed version ("perturbed" mode,
DIAG_AGENT_ID) -- so the same task boundaries are compared with and without poisoning.

Run from the repo root (needs red_agent_perturbed_dataset.pkl under DATASET_PATH):
    python task_distribution_diagnostic.py --seed 42 --log_name task_diag
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
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CLI ---
parser = argparse.ArgumentParser(description="Task-to-task data distribution diagnostic")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--log_name', type=str, default='task_diag',
                    help='Base name for the JSON results log and PNG plots written by this run')
args = parser.parse_args()

SEED = args.seed
LOG_NAME = args.log_name

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_DEVICE = "cpu"   # see clmd_cicids_naive_cd.py -- "auto"/"cuda" can crash on newer GPUs
                        # than the installed torch build supports; this workload is tiny anyway
DEVICE = torch.device(TRAIN_DEVICE)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
# DATASET_PATH = "./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset2.pkl"
DATASET_PATH = "/home/erivas6/Desktop/Unlearning_experiments/Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset2.pkl"
LABEL_COLUMN = "label_binary"

NUM_TASKS = 10                  # "10 batches" -- task 1 = task_id 0, compared against task_ids 1..9
TASK0_ROW_FRACTION = 0.35       # same windowing convention as clmd_cicids_naive_cd.py

DIAG_AGENT_ID = 0               # which agent's perturbation represents "perturbed" mode
REQUIRE_EVASION_SUCCESS = True  # only substitute a perturbed row if it actually evaded detection;
                                 # falls back to the clean row for a sample with no valid substitute

REFERENCE_TRAIN_EPOCHS = 5      # epochs used to train the Method-A reference model on task 1
BATCH_SIZE = 256
FEATURE_CLIP = 10.0
KL_HIST_BINS = 50

HIDDEN_DIMS = [1024, 512, 256, 128]   # matches clmd_cicids_naive_cd.py's ClassifierNN
# ==========================================


# --- 1. Data loading + task windowing (shared logic with clmd_cicids_naive_cd.py) ---

def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Point DATASET_PATH at your generated pickle.")
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["data_original_scale"].copy(), blob["feature_columns"], blob.get("metadata", {})


def encode_labels(df: pd.DataFrame, label_column: str):
    le = LabelEncoder()
    y_all = le.fit_transform(df[label_column].values)
    return y_all, len(le.classes_), le


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


def build_task_subset(originals_with_task, perturbed_with_task, task_id, mode, agent_id, require_evasion_success):
    """mode='clean' -> benign + malicious_clean only. mode='perturbed' -> benign + malicious
    replaced by agent_id's perturbed version (falls back to clean if no valid substitute)."""
    subset = originals_with_task[originals_with_task["task_id"] == task_id]
    if mode == "clean":
        return subset

    benign = subset[subset["label_binary"] == 0]
    malicious = subset[subset["label_binary"] == 1]

    pert_pool = perturbed_with_task[
        (perturbed_with_task["task_id"] == task_id) & (perturbed_with_task["agent_id"] == agent_id)
    ]
    if require_evasion_success:
        pert_pool = pert_pool[pert_pool["evasion_success"] == True]  # noqa: E712

    eligible_ids = set(pert_pool["source_row_id"].tolist())
    eligible_mal = malicious[malicious["orig_row_id"].isin(eligible_ids)]
    fallback_mal = malicious[~malicious["orig_row_id"].isin(eligible_ids)]
    substituted = pert_pool[pert_pool["source_row_id"].isin(eligible_mal["orig_row_id"])]

    return pd.concat([benign, fallback_mal, substituted], ignore_index=True)


def to_feature_matrix(subset_df, feature_cols, scaler, clip):
    X = np.clip(scaler.transform(subset_df[feature_cols].to_numpy(dtype=np.float32)), -clip, clip)
    y = subset_df["_y_encoded"].to_numpy()
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), X


# --- 2. Model (same architecture as clmd_cicids_naive_cd.py's ClassifierNN) ---

class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024); self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512); self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256); self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128); self.fc4_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4_bn(self.fc4(x)))
        return self.fc_last(x)


def train_reference_model(model, X, y, epochs, batch_size, device):
    drop_last = (len(X) % batch_size == 1)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size, shuffle=True, drop_last=drop_last)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for v, l in loader:
            v, l = v.to(device), l.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(v), l)
            loss.backward()
            optimizer.step()


def compute_avg_gradient(model, X, y, device):
    """Single full-batch forward+backward pass -- the gradient of this task's loss at the
    model's CURRENT weights, without taking an optimizer step."""
    model.train()
    model.zero_grad()
    v, l = X.to(device), y.to(device)
    loss = nn.CrossEntropyLoss()(model(v), l)
    loss.backward()
    grads = [p.grad.detach().flatten().clone() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads).numpy()


def cosine_similarity(g1, g2, eps=1e-12):
    return float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + eps))


# --- 3. Distributional distance (adapted from adaptor_env.py's DriftAdaptationEnv) ---

def compute_kl_divergence(X_old, X_new, bins=50):
    """Per-feature KL divergence with SHARED bin edges (spanning both datasets' combined
    range). adaptor_env.py's original version bins X_old and X_new independently over their
    own min/max, which silently compares distribution *shapes* rather than actual value
    differences and badly underestimates divergence between distributions on very different
    scales -- verified this concretely before shipping (two disjoint-range Gaussians scored
    KL~0.3 with independent binning, spuriously low)."""
    kl_divergences = []
    for i in range(X_old.shape[1]):
        combined_min = min(X_old[:, i].min(), X_new[:, i].min())
        combined_max = max(X_old[:, i].max(), X_new[:, i].max())
        if combined_min == combined_max:
            kl_divergences.append(0.0)
            continue
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        p = np.histogram(X_old[:, i], bins=bin_edges, density=True)[0] + 1e-10
        q = np.histogram(X_new[:, i], bins=bin_edges, density=True)[0] + 1e-10
        kl_divergences.append(stats.entropy(p, q))
    return float(np.mean(kl_divergences))


def compute_wasserstein_distance(X_old, X_new):
    distances = [wasserstein_distance(X_old[:, i], X_new[:, i]) for i in range(X_old.shape[1])]
    return float(np.mean(distances))


# --- 4. Main ---

def run_mode(mode, originals_with_task, perturbed_with_task, feature_cols, num_classes, input_dim):
    print(f"\n========== MODE: {mode} ==========")

    task0_subset = build_task_subset(originals_with_task, perturbed_with_task, 0, mode,
                                     DIAG_AGENT_ID, REQUIRE_EVASION_SUCCESS)
    scaler = StandardScaler()
    scaler.fit(task0_subset[feature_cols].to_numpy(dtype=np.float32))

    task_tensors = {}
    for tid in range(NUM_TASKS):
        subset = task0_subset if tid == 0 else build_task_subset(
            originals_with_task, perturbed_with_task, tid, mode, DIAG_AGENT_ID, REQUIRE_EVASION_SUCCESS
        )
        Xt, yt, Xnp = to_feature_matrix(subset, feature_cols, scaler, FEATURE_CLIP)
        task_tensors[tid] = (Xt, yt, Xnp)
        print(f"  task {tid}: n={len(Xt)}")

    # ---- Method B: shared UNTRAINED reference point ----
    torch.manual_seed(SEED)
    model_ref = ClassifierNN(input_dim, num_classes).to(DEVICE)
    X0, y0, _ = task_tensors[0]
    g0_ref = compute_avg_gradient(model_ref, X0, y0, DEVICE)

    # ---- Method A: reference point = weights AFTER training on task 0 ----
    torch.manual_seed(SEED)   # identical starting point to model_ref, so A and B are paired
    model_trained = ClassifierNN(input_dim, num_classes).to(DEVICE)
    train_reference_model(model_trained, X0, y0, REFERENCE_TRAIN_EPOCHS, BATCH_SIZE, DEVICE)
    g0_trained = compute_avg_gradient(model_trained, X0, y0, DEVICE)

    results = {}
    for tid in range(1, NUM_TASKS):
        Xi, yi, Xi_np = task_tensors[tid]

        gi_ref = compute_avg_gradient(model_ref, Xi, yi, DEVICE)
        grad_cos_no_fix = cosine_similarity(g0_trained, compute_avg_gradient(model_trained, Xi, yi, DEVICE))
        grad_cos_ref_point = cosine_similarity(g0_ref, gi_ref)

        kl = compute_kl_divergence(task_tensors[0][2], Xi_np, bins=KL_HIST_BINS)
        wass = compute_wasserstein_distance(task_tensors[0][2], Xi_np)

        results[tid] = {
            "n_samples": int(len(Xi)),
            "grad_cosine_no_reference_fix": grad_cos_no_fix,
            "grad_cosine_reference_point": grad_cos_ref_point,
            "kl_divergence": kl,
            "wasserstein_distance": wass,
        }
        print(f"  task 1 vs task {tid+1}: grad_cos(no-fix)={grad_cos_no_fix:+.4f}  "
              f"grad_cos(ref-point)={grad_cos_ref_point:+.4f}  KL={kl:.4f}  Wasserstein={wass:.4f}")

    return results


if __name__ == "__main__":
    df, feature_cols, ds_metadata = load_dataset(DATASET_PATH)
    y_all, num_classes, label_encoder = encode_labels(df, LABEL_COLUMN)
    df = df.copy()
    df["_y_encoded"] = y_all
    input_dim = len(feature_cols)
    print(f"INPUT_DIM={input_dim}  num_classes={num_classes}  classes={list(label_encoder.classes_)}")

    originals_with_task = assign_task_windows(df, NUM_TASKS, TASK0_ROW_FRACTION)
    perturbed_with_task = assign_perturbed_task_ids(df, originals_with_task)

    all_results = {}
    for mode in ["clean", "perturbed"]:
        all_results[mode] = run_mode(mode, originals_with_task, perturbed_with_task,
                                     feature_cols, num_classes, input_dim)

    run_config = {
        "DATASET_PATH": DATASET_PATH, "LABEL_COLUMN": LABEL_COLUMN, "NUM_TASKS": NUM_TASKS,
        "TASK0_ROW_FRACTION": TASK0_ROW_FRACTION, "DIAG_AGENT_ID": DIAG_AGENT_ID,
        "REQUIRE_EVASION_SUCCESS": REQUIRE_EVASION_SUCCESS,
        "REFERENCE_TRAIN_EPOCHS": REFERENCE_TRAIN_EPOCHS, "BATCH_SIZE": BATCH_SIZE,
        "FEATURE_CLIP": FEATURE_CLIP, "KL_HIST_BINS": KL_HIST_BINS, "SEED": SEED,
        "INPUT_DIM": input_dim, "NUM_CLASSES": num_classes,
    }

    log_path = f"{LOG_NAME}_seed{SEED}.json"
    with open(log_path, "w") as f:
        json.dump({"config": run_config, "results": all_results}, f, indent=2, default=str)
    print(f"\nSaved log to {log_path}")

    for mode, results in all_results.items():
        task_ids = sorted(results.keys())
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(task_ids, [results[t]["grad_cosine_no_reference_fix"] for t in task_ids],
                    marker='o', label='Grad cosine (no reference fix)')
        axes[0].plot(task_ids, [results[t]["grad_cosine_reference_point"] for t in task_ids],
                    marker='x', label='Grad cosine (reference point)')
        axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
        axes[0].set_ylabel('Cosine similarity to task 1')
        axes[0].set_title(f'Task 1 vs. other tasks -- {mode} mode')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        ax2 = axes[1]
        ax2.plot(task_ids, [results[t]["kl_divergence"] for t in task_ids],
                marker='o', color='tab:red', label='KL divergence')
        ax2.set_ylabel('KL divergence', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax3 = ax2.twinx()
        ax3.plot(task_ids, [results[t]["wasserstein_distance"] for t in task_ids],
                marker='x', color='tab:blue', label='Wasserstein distance')
        ax3.set_ylabel('Wasserstein distance', color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_xlabel('Task index (0 = task 1, the reference)')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_png = f"{LOG_NAME}_{mode}_seed{SEED}.png"
        plt.savefig(out_png)
        plt.close()
        print(f"Saved plot to {out_png}")
