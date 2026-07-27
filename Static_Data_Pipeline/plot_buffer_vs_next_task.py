"""
Run with: python plot_buffer_vs_next_task.py --seed 42 --out_dir buffer_vs_next_task_out

Runs the ACTUAL plain-MADAR training pipeline (task-0 supervised warmup, then
per-task replay+KD+SI via train_cl_er, then update_buffer_madar) -- model
architecture, buffer-selection logic (IsolationForest over LATENT embeddings,
union-with-existing-buffer merge), and data construction copied verbatim from
clmd_cicids_madar_cd.py (including the multi-day row-id fix) so the buffer
this script inspects is exactly what a real MADAR run would produce.

After each task's update_buffer_madar() call, before training moves on to the
next task, it snapshots two groups in that SAME model checkpoint's 128-dim
latent space:
  - "buffer": every malicious (label=1) sample currently held in the replay
    buffer (this task's MADAR selection).
  - "next_poison": the malicious_perturbed TRAIN rows belonging to the very
    next task -- the poisoned samples the model is about to be fed.

It then runs t-SNE jointly over both groups and plots them, so you can see
directly whether the buffer's notion of "malicious" is anywhere near the
distribution of poison that's about to arrive, or whether they occupy
disjoint regions of latent space (which would explain why replay/KD --
anchored on the buffer -- doesn't generalize to the new poison).

Outputs (under --out_dir), one set per task boundary tid -> tid+1:
  - boundary_{tid}_to_{tid+1}_tsne.png
  - boundary_{tid}_to_{tid+1}_latents.npz (raw latent vectors + group/category
    labels, for re-plotting or further quantitative analysis)
  - separability_summary.csv: per boundary, group sizes plus a simple
    centroid-distance / intra-group-spread ratio computed in the FULL
    128-dim latent space (t-SNE is for visualization only -- don't trust
    distances read off the 2D plot quantitatively).
"""

import os
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# --- CLI ---
parser = argparse.ArgumentParser(description="Compare MADAR buffer contents vs. next task's poisoned samples")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='buffer_vs_next_task_out')
parser.add_argument('--dataset_path', type=str,
                    default="./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset_4500.pkl")
parser.add_argument('--tsne_perplexity', type=float, default=30.0)
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(args.out_dir, exist_ok=True)

TRAIN_DEVICE = "cpu"           # see clmd_cicids_naive_cd.py -- "auto"/"cuda" can crash on
                                # newer GPUs than the installed torch build supports
DEVICE = torch.device(TRAIN_DEVICE)

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION  (mirrors clmd_cicids_madar_cd.py exactly)
# ==========================================
DATASET_PATH = args.dataset_path
LABEL_COLUMN = "label_binary"

FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30
CL_ITERS = 2000
BATCH_SIZE = 256

MEM_SIZE = 800
MADAR_CONTAMINATION = 0.1

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1

NUM_TASKS = 5
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


# --- 1. Data loading (identical to clmd_cicids_madar_cd.py) ---

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


# --- 2. Task windowing (identical to the fixed clmd_cicids_madar_cd.py) ---

def assign_task_windows(df, num_tasks, task0_fraction):
    """Sorted by TIMESTAMP -- source_row_id/orig_row_id are per-day counters that reset to 0
    for each source_range, so sorting by them directly would interleave rows from different
    days by coincidence of row number instead of time."""
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
    """orig_row_id/source_row_id reset per source_range (day), so the join key must be
    (source_range, row_id), not the bare row id."""
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
    """Identity keys on (source_range, row_id): a single task can span more than one day's
    rows, so a bare-id match risks pairing a Tuesday original with a Wednesday perturbation
    that happens to share the same in-day row number."""
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


# --- 4. Build tasks (identical control flow to clmd_cicids_madar_cd.py's __main__) ---

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

    return tasks


def scale_tasks(tasks, feature_cols):
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
    return scaler


# --- 5. Model + MADAR machinery (copied verbatim from clmd_cicids_madar_cd.py) ---

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


def _embed(model, Xt: torch.Tensor, device):
    loader = data.DataLoader(data.TensorDataset(Xt), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    parts = []
    with torch.no_grad():
        for (v,) in loader:
            _, latent = model(v.to(device), return_latent=True)
            parts.append(latent.cpu())
    return torch.cat(parts).numpy()


def update_buffer_madar(X: torch.Tensor, y: torch.Tensor, category: np.ndarray, model, device):
    global label_buffers, replay_buffer
    model.eval()

    X_np = X.numpy()
    Y_np = y.numpy()
    cat_np = np.asarray(category)
    L_np = _embed(model, X, device)
    latent_dim = L_np.shape[1]

    current_labels = np.unique(Y_np).tolist()
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


# --- 6. Buffer-vs-next-task-poison snapshot + t-SNE ---

def snapshot_and_plot(tid, model, tasks, out_dir, perplexity):
    """Called right after update_buffer_madar() for task `tid`, before training moves on to
    task tid+1. Compares the malicious buffer contents (this task's MADAR selection) against
    task tid+1's poisoned (malicious_perturbed) TRAIN samples, both embedded through this
    SAME model checkpoint."""
    if tid + 1 >= NUM_TASKS:
        return None

    model.eval()  # _embed() relies on eval-mode BatchNorm; don't depend on caller ordering
    buf_entries = label_buffers.get(1, [])
    if not buf_entries:
        print(f"  [skip] boundary {tid}->{tid+1}: buffer has no malicious entries")
        return None

    buf_X = torch.stack([e[0] for e in buf_entries])
    buf_cat = np.array([e[2] for e in buf_entries])
    buf_latent = _embed(model, buf_X, DEVICE)

    next_df = tasks[tid + 1]["train"]
    next_mask = (next_df["category"] == "malicious_perturbed").to_numpy()
    n_next_poison = int(next_mask.sum())
    if n_next_poison == 0:
        print(f"  [skip] boundary {tid}->{tid+1}: task {tid+1} has no malicious_perturbed train rows")
        return None

    next_X = tasks[tid + 1]["train_tensors"][0][next_mask]
    next_latent = _embed(model, next_X, DEVICE)

    combined = np.concatenate([buf_latent, next_latent], axis=0)
    group = np.array(["buffer"] * len(buf_latent) + ["next_task_poison"] * len(next_latent))
    cat = np.concatenate([buf_cat, np.array(["malicious_perturbed"] * len(next_latent))])

    eff_perplexity = max(5.0, min(perplexity, (len(combined) - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=SEED, init="pca")
    emb2d = tsne.fit_transform(combined)

    plt.figure(figsize=(7, 6))
    marker_by_group = {"buffer": "o", "next_task_poison": "^"}
    color_by_cat = {"malicious_clean": "tab:orange", "malicious_perturbed": "tab:red"}
    for g in ("buffer", "next_task_poison"):
        for c, color in color_by_cat.items():
            m = (group == g) & (cat == c)
            if m.sum() == 0:
                continue
            label = f"{g} / {c}"
            plt.scatter(emb2d[m, 0], emb2d[m, 1], s=14, alpha=0.6, marker=marker_by_group[g],
                        color=color, label=label)
    plt.title(f"Buffer after task {tid} vs. task {tid+1}'s poisoned samples\n(t-SNE, latent space)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"boundary_{tid}_to_{tid+1}_tsne.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved {png_path}  (buffer n={len(buf_latent)}, next_poison n={len(next_latent)})")

    npz_path = os.path.join(out_dir, f"boundary_{tid}_to_{tid+1}_latents.npz")
    np.savez(npz_path, latent=combined, group=group, category=cat)

    # Quantitative separability in the FULL latent space (t-SNE is for visualization only)
    buf_centroid = buf_latent.mean(axis=0)
    next_centroid = next_latent.mean(axis=0)
    centroid_dist = float(np.linalg.norm(buf_centroid - next_centroid))
    buf_spread = float(np.linalg.norm(buf_latent - buf_centroid, axis=1).mean())
    next_spread = float(np.linalg.norm(next_latent - next_centroid, axis=1).mean())
    avg_spread = (buf_spread + next_spread) / 2.0
    separation_ratio = centroid_dist / avg_spread if avg_spread > 0 else float("nan")
    print(f"  centroid_dist={centroid_dist:.3f}  avg_intra_group_spread={avg_spread:.3f}  "
          f"separation_ratio={separation_ratio:.3f}  (>~1 means the two groups are farther "
          f"apart than they are internally spread out)")

    return {
        "boundary": f"{tid}->{tid+1}", "buffer_n": len(buf_latent), "next_poison_n": len(next_latent),
        "centroid_dist": centroid_dist, "buffer_spread": buf_spread, "next_poison_spread": next_spread,
        "separation_ratio": separation_ratio,
    }


# --- 7. Main ---

if __name__ == "__main__":
    df, feature_cols, ds_metadata = load_dataset(DATASET_PATH)
    y_all, num_classes, label_encoder = encode_labels(df, LABEL_COLUMN)
    df = df.copy()
    df["_y_encoded"] = y_all
    class_labels = list(range(num_classes))
    INPUT_DIM = len(feature_cols)

    print(f"INPUT_DIM={INPUT_DIM}  num_classes={num_classes}  classes={list(label_encoder.classes_)}")

    tasks = build_tasks(df)
    scale_tasks(tasks, feature_cols)

    model = ClassifierNN(INPUT_DIM, num_classes).to(DEVICE)
    W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    teacher_model = None
    prev_active_count = 0
    seen_labels = set()

    summary_rows = []

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

            for _ in range(TASK0_EPOCHS):
                for v, l in loader:
                    v, l = v.to(DEVICE), l.to(DEVICE)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(v) + mask, l)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at task {tid} - training diverged")
                    loss.backward()
                    optimizer.step()

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

        print(f"    Snapshotting buffer vs. task {tid+1}'s poison...")
        row = snapshot_and_plot(tid, model, tasks, args.out_dir, args.tsne_perplexity)
        if row:
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.out_dir, "separability_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved separability summary to {summary_path}")
        print(summary_df.to_string(index=False))

    print(f"\nDone. Everything written to {args.out_dir}/")
