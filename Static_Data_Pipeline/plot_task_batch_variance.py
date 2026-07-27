"""
Run with: python plot_task_batch_variance.py --seed 42 --out_dir task_variance_out

Reproduces the exact train_i/test_i batches that clmd_cicids_unlearning_cd.py /
clmd_cicids_madar_cd.py build for each CL task (same time-based windowing,
poisoning substitution, and category-diversity logic -- copied verbatim from
the fixed clmd_cicids_unlearning_cd.py, including the multi-day row-id fix),
then visualizes how different those batches are from each other.

Meant to check whether later tasks -- and specifically what the replay
buffer/unlearner sees -- carry enough real variance to support learning a
meaningful representation, rather than being near-duplicates of earlier tasks
or collapsed onto a narrow slice of feature space.

Outputs (under --out_dir):
  - variance_report.csv: per task/split/category sample count, total variance
    (trace of covariance in the FULL scaled feature space, not the lossy 2D
    PCA projection below), and mean distance to the group's own centroid.
  - tasks_train_by_category.png / tasks_test_by_category.png: one subplot per
    task, points colored by category (benign / malicious_clean /
    malicious_perturbed), all sharing one PCA basis fit on the pooled training
    data so tasks are visually comparable.
  - tasks_overlay_by_task_id.png: every task's train points overlaid in one
    plot, colored by task, to see cluster separation/overlap directly.
"""

import os
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# --- CLI ---
parser = argparse.ArgumentParser(description="Visualize train/test batch variance across CL tasks")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='task_variance_out')
parser.add_argument('--dataset_path', type=str,
                    default="./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset_4500.pkl")
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION  (mirrors clmd_cicids_unlearning_cd.py)
# ==========================================
DATASET_PATH = args.dataset_path
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0

# --- Task construction (time-based, not class-based) ---
NUM_TASKS = 5
TASK0_ROW_FRACTION = 0.35
TASK_TEST_FRAC = 0.20

# --- Poisoning scenario ---
TASK0_CLEAN_ONLY = True
AGENT_MODE = "mixed"
FIXED_AGENT_ID = 0
REQUIRE_EVASION_SUCCESS = True
POISON_FRACTION = 0.8

# --- Category diversity floor ---
ENSURE_CATEGORY_DIVERSITY = True
MIN_CATEGORY_COUNT = 5
# ==========================================


# --- 1. Data loading (identical to clmd_cicids_unlearning_cd.py) ---

def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    with open(path, "rb") as f:
        blob = pickle.load(f)
    df = blob["data_original_scale"].copy()
    feature_cols = blob["feature_columns"]
    return df, feature_cols, blob.get("metadata", {})


def encode_labels(df, label_column):
    le = LabelEncoder()
    y_all = le.fit_transform(df[label_column].values)
    return y_all, len(le.classes_), le


# --- 2. Task windowing (identical to the fixed clmd_cicids_unlearning_cd.py) ---

def assign_task_windows(df, num_tasks, task0_fraction):
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


def assign_perturbed_task_ids(df, originals_with_task):
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


# --- 3. Poisoning substitution + category diversity floor (identical, fixed) ---

def poison_and_diversify(malicious_orig_subset, perturbed_pool, agent_for_task, poison_fraction,
                          require_evasion_success, ensure_diversity, min_category_count, rng):
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


# --- 4. Build tasks (identical control flow to clmd_cicids_unlearning_cd.py's __main__) ---

def build_tasks(df):
    n_agents = sorted(
        df.loc[df["row_type"] == "perturbed", "agent_id"].dropna().astype(int).unique().tolist()
    )
    originals_with_task = assign_task_windows(df, NUM_TASKS, TASK0_ROW_FRACTION)
    perturbed_with_task = assign_perturbed_task_ids(df, originals_with_task)

    rng = random.Random(SEED)
    tasks = {}

    for tid in range(NUM_TASKS):
        task_orig = originals_with_task[originals_with_task["task_id"] == tid]
        benign_orig = task_orig[task_orig["label_binary"] == 0]
        malicious_orig = task_orig[task_orig["label_binary"] == 1]

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
        else:
            if AGENT_MODE == "uniform":
                agent_for_task = FIXED_AGENT_ID
            else:
                poison_task_index = tid - (1 if TASK0_CLEAN_ONLY else 0)
                agent_for_task = n_agents[poison_task_index % len(n_agents)]

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
            print(f"Task {tid}: agent={agent_for_task} | train poisoned={n_p} clean={n_c} "
                  f"| test poisoned={n_p_t} clean={n_c_t}")

        train_df = pd.concat([ben_train, mal_train_final], ignore_index=True)
        test_df = pd.concat([ben_test, mal_test_final], ignore_index=True)
        tasks[tid] = {"train": train_df, "test": test_df}
        print(f"Task {tid}: train={len(train_df)} test={len(test_df)} "
              f"(train categories: {train_df['category'].value_counts().to_dict()})")

    return tasks


# --- 5. Scale + clip (identical to training: fit on task 0 train only) ---

def scale_tasks(tasks, feature_cols):
    scaler = StandardScaler()
    scaler.fit(tasks[0]["train"][feature_cols].to_numpy(dtype=np.float32))

    def to_matrix(split_df):
        return np.clip(scaler.transform(split_df[feature_cols].to_numpy(dtype=np.float32)),
                        -FEATURE_CLIP, FEATURE_CLIP)

    for tid in tasks:
        tasks[tid]["train_X"] = to_matrix(tasks[tid]["train"])
        tasks[tid]["test_X"] = to_matrix(tasks[tid]["test"])
    return scaler


# --- 6. Variance report (full-dimensional, not just the 2D projection) ---

def variance_report(tasks):
    rows = []
    for tid in tasks:
        for split_name in ("train", "test"):
            split_df = tasks[tid][split_name]
            X = tasks[tid][f"{split_name}_X"]
            for cat in ["benign", "malicious_clean", "malicious_perturbed"]:
                mask = (split_df["category"] == cat).to_numpy()
                n = int(mask.sum())
                if n == 0:
                    continue
                Xc = X[mask]
                total_var = float(Xc.var(axis=0).sum())  # trace of covariance
                mean_dist = None
                if n >= 2:
                    centroid = Xc.mean(axis=0)
                    mean_dist = float(np.linalg.norm(Xc - centroid, axis=1).mean())
                rows.append({
                    "task_id": tid, "split": split_name, "category": cat, "n_samples": n,
                    "total_variance": total_var, "mean_dist_to_centroid": mean_dist,
                })
    return pd.DataFrame(rows)


# --- 7. PCA visualization ---

def plot_tasks(tasks, out_dir):
    all_train_X = np.concatenate([tasks[tid]["train_X"] for tid in tasks], axis=0)
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(all_train_X)
    print(f"PCA on pooled train data: explained variance ratio (PC1, PC2) = "
          f"{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    cat_colors = {"benign": "tab:blue", "malicious_clean": "tab:orange", "malicious_perturbed": "tab:red"}

    for split_name in ("train", "test"):
        fig, axes = plt.subplots(1, NUM_TASKS, figsize=(4.5 * NUM_TASKS, 4.5), sharex=True, sharey=True)
        if NUM_TASKS == 1:
            axes = [axes]
        for tid in range(NUM_TASKS):
            ax = axes[tid]
            split_df = tasks[tid][split_name]
            emb = pca.transform(tasks[tid][f"{split_name}_X"])
            for cat, color in cat_colors.items():
                mask = (split_df["category"] == cat).to_numpy()
                if mask.sum() == 0:
                    continue
                ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.5, color=color, label=cat)
            ax.set_title(f"Task {tid} ({split_name}, n={len(split_df)})")
            ax.set_xlabel("PC1")
            if tid == 0:
                ax.set_ylabel("PC2")
        axes[-1].legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"tasks_{split_name}_by_category.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    cmap = plt.get_cmap("viridis", NUM_TASKS)
    plt.figure(figsize=(8, 7))
    for tid in range(NUM_TASKS):
        emb = pca.transform(tasks[tid]["train_X"])
        plt.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.4, color=cmap(tid), label=f"Task {tid}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("All tasks overlaid (train), colored by task")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "tasks_overlay_by_task_id.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# --- 8. Main ---

if __name__ == "__main__":
    df, feature_cols, ds_metadata = load_dataset(DATASET_PATH)
    y_all, num_classes, label_encoder = encode_labels(df, LABEL_COLUMN)
    df = df.copy()
    df["_y_encoded"] = y_all

    n_agents = sorted(
        df.loc[df["row_type"] == "perturbed", "agent_id"].dropna().astype(int).unique().tolist()
    )
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, agents detected: {n_agents}")

    tasks = build_tasks(df)
    scale_tasks(tasks, feature_cols)

    report = variance_report(tasks)
    report_path = os.path.join(args.out_dir, "variance_report.csv")
    report.to_csv(report_path, index=False)
    print(f"\nSaved variance report to {report_path}")
    print(report.to_string(index=False))

    plot_tasks(tasks, args.out_dir)
    print(f"\nDone. Everything written to {args.out_dir}/")
