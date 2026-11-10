"""
rebuttal_stream.py

Shared helpers for the ICLR 2027 rebuttal "informed DEDUCE" comparison.
Nothing here changes any pipeline's default behavior -- every entry point is
only reached when one of the new opt-in flags is passed:

  madar_pocket_pipeline.py         --data_fraction / --dump_stream / --num_tasks
  madar_pocket_pipeline_deduce.py  --load_stream / --informed_forget / --val_fraction

1. TRAIN-SPLIT SUBSAMPLING (`subsample_train_split`). Keeps `fraction` of a
   task's TRAINING split, stratified by label, drawn with a FIXED subset seed
   (independent of --seed; the producer passes [data_subset_seed, task] so
   each task gets its own fixed stream), and re-sorted by global sample id -- ids are
   assigned in timestamp order (h5_data_loader.load_pooled_chronological_tasks
   sorts the pool by timestamp), so the kept rows are in chronological order.
   The train/test split itself still depends on --seed (unchanged), so
   different seeds see different rows; within one seed every config sees the
   same rows because the DEDUCE file reads them from the dumped stream.

2. SHARED POISON + DETECTOR STREAM (`StreamWriter` / `StreamReader`).
   madar_pocket_pipeline.py dumps, per seed and per task, everything a second
   pipeline needs to train and be SCORED on identical data:
     task 0 : scaler, task-0 pretrained model weights, train/test arrays
     task t : clean + poisoned training arrays, labels, ground-truth poison
              indices, the detector's flagged set U_t (detected_poison_idx),
              the raw clean test split, and every adversarial test set built
              this task (this task's own + the re-attacked historical ones)
   Layout: <dir>/seed<seed>/task<t>.npz  (+ task0_extras.pt, manifest.json).
   manifest.json stores a sha256 per array; StreamReader re-hashes every array
   on load and raises on any mismatch, and returns the hashes of the key
   arrays (poison indices, U_t) so the loading pipeline can log them.

3. PER-TASK METRICS (`lineage_metrics`, `MetricsLog`). A machine-readable
   JSON (logs/rebuttal_metrics.json) with exactly the quantities
   build_latex_tables.py reports, computed the same way as
   summarize_pipeline_runs.parse_run (see `lineage_metrics`), plus |U_t| and a
   malicious-side attack success rate. build_rebuttal_tables.py turns these
   into the CSV + per-seed LaTeX rows.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


# ---------------------------------------------------------------------------
# 1. Train-split subsampling
# ---------------------------------------------------------------------------
def subsample_train_split(X, y, gid, fraction, subset_seed):
    """Stratified-by-label subsample of ONE task's training split.

    Per label, keeps round(fraction * n_label) rows (at least 1 if the label is
    present), chosen with np.random.default_rng(subset_seed) -- independent of
    the run's --seed (subset_seed may be an int or a seed sequence such as
    [data_subset_seed, task_id]). Returned rows are sorted by `gid` (= timestamp order).
    fraction >= 1.0 returns the inputs unchanged."""
    if fraction >= 1.0:
        return X, y, gid
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"--data_fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(subset_seed)
    keep = []
    for lbl in np.unique(y):
        idx = np.where(y == lbl)[0]
        n_keep = max(1, int(round(fraction * len(idx))))
        keep.append(rng.choice(idx, size=n_keep, replace=False))
    keep = np.concatenate(keep)
    keep = keep[np.argsort(gid[keep], kind="stable")]
    return X[keep], y[keep], gid[keep]


def stratified_holdout(y, fraction, seed):
    """(train_idx, val_idx) positions into y: a stratified `fraction` held out
    per label with a fixed seed, each side sorted (keeps chronological order
    when y is already in gid order). fraction <= 0 -> (all, empty)."""
    n = len(y)
    if fraction <= 0:
        return np.arange(n), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    val = []
    for lbl in np.unique(y):
        idx = np.where(y == lbl)[0]
        n_val = int(round(fraction * len(idx)))
        if n_val:
            val.append(rng.choice(idx, size=n_val, replace=False))
    val = np.sort(np.concatenate(val)) if val else np.array([], dtype=np.int64)
    train = np.setdiff1d(np.arange(n), val)
    return train, val


# ---------------------------------------------------------------------------
# 2. Shared stream
# ---------------------------------------------------------------------------
def hash_array(a):
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _seed_dir(root, seed):
    return os.path.join(root, f"seed{seed}")


class StreamWriter:
    def __init__(self, root, seed, meta):
        self.dir = _seed_dir(root, seed)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.manifest = {"meta": dict(meta, seed=seed), "tasks": {}}
        self._flush()

    def _flush(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=1, sort_keys=True)

    def save_task(self, t, arrays):
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        np.savez(os.path.join(self.dir, f"task{t}.npz"), **arrays)
        self.manifest["tasks"][str(t)] = {k: hash_array(v) for k, v in arrays.items()}
        self._flush()

    def save_task0_extras(self, scaler, model_state_dict):
        torch.save({"scaler": scaler, "model_state_dict": model_state_dict},
                   os.path.join(self.dir, "task0_extras.pt"))

    def set_meta(self, **kw):
        self.manifest["meta"].update(kw)
        self._flush()


class StreamReader:
    def __init__(self, root, seed):
        self.dir = _seed_dir(root, seed)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"No stream manifest at {self.manifest_path} -- run madar_pocket_pipeline.py "
                f"--dump_stream {root} --seed {seed} first.")
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        self.meta = self.manifest["meta"]
        if int(self.meta["seed"]) != int(seed):
            raise ValueError(f"stream seed {self.meta['seed']} != requested seed {seed}")

    @property
    def num_tasks(self):
        return len(self.manifest["tasks"])

    def load_task(self, t):
        """Returns (arrays, hashes). Re-hashes EVERY array and raises on any
        mismatch with manifest.json."""
        path = os.path.join(self.dir, f"task{t}.npz")
        expected = self.manifest["tasks"].get(str(t))
        if expected is None:
            raise KeyError(f"task {t} not in stream {self.dir} (has {sorted(self.manifest['tasks'])})")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if set(arrays) != set(expected):
            raise ValueError(f"task {t}: array set {sorted(arrays)} != manifest {sorted(expected)}")
        got = {k: hash_array(v) for k, v in arrays.items()}
        bad = [k for k in arrays if got[k] != expected[k]]
        if bad:
            raise ValueError(f"task {t}: sha256 mismatch vs manifest for {bad}")
        return arrays, got

    def train_sizes(self):
        """Training-row count of every task in the stream, in task order.
        Reads only y_train from each npz (no full load / hash check)."""
        sizes = []
        for t in range(self.num_tasks):
            with np.load(os.path.join(self.dir, f"task{t}.npz"), allow_pickle=False) as z:
                sizes.append(int(z["y_train"].shape[0]))
        return sizes

    def load_task0_extras(self):
        return torch.load(os.path.join(self.dir, "task0_extras.pt"), weights_only=False)


def pack_eval_sets(historical_adv, X_test_adv, succ_pocket, norms_pocket, eps_this_task):
    """Flattens this task's adversarial test sets into npz-able arrays.
    historical_adv: {s: (Xs_adv, ys, succ_s, eps_s)} for prior tasks s."""
    out = {
        "X_test_adv": X_test_adv, "succ_pocket": succ_pocket, "norms_pocket": norms_pocket,
        "eps_this_task": np.float64(eps_this_task),
        "hist_sources": np.array(sorted(historical_adv), dtype=np.int64),
    }
    for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
        out[f"hist{s}_X"] = Xs_adv
        out[f"hist{s}_y"] = ys
        out[f"hist{s}_succ"] = succ_s
        out[f"hist{s}_eps"] = np.float64(eps_s)
    return out


def unpack_eval_sets(arrays):
    historical_adv = {}
    for s in arrays["hist_sources"].tolist():
        historical_adv[int(s)] = (arrays[f"hist{s}_X"], arrays[f"hist{s}_y"],
                                  arrays[f"hist{s}_succ"], float(arrays[f"hist{s}_eps"]))
    return (historical_adv, arrays["X_test_adv"], arrays["succ_pocket"], arrays["norms_pocket"],
            float(arrays["eps_this_task"]))


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def lineage_metrics(lineage, t, all_clean_sets, all_test_sets_full, X_test_scaled, y_test,
                    X_test_adv, succ_pocket, benign_label, mal_label, mean_acc, pooled_acc):
    """Same quantities summarize_pipeline_runs.parse_run extracts from a log,
    computed directly:
      mean_acc / pooled_acc     -- pooled_and_per_task_accuracy over ALL clean +
                                   adversarial test sets of tasks 0..t (passed in)
      task_acc / adv_acc        -- this task's clean / adversarial test set
      final_task_acc_combined   -- (task_acc + adv_acc) / 2
      per_source_clean/_adv     -- per source task s (clean, adversarial), used
                                   for the 'Task 1' column: (clean[1]+adv[1])/2
      precision/recall/f1       -- macro, this task's CLEAN test set
    plus the rebuttal-only attack success rates:
      asr_malicious             -- fraction of this task's adversarial test rows
                                   whose TRUE label is malicious that are
                                   predicted benign
      asr_malicious_pockets     -- same, restricted to rows where the attack
                                   found a genuine pocket (succ_pocket)
    """
    task_acc = float(lineage.score(X_test_scaled, y_test))
    adv_acc = float(lineage.score(X_test_adv, y_test))
    pred_clean = lineage.predict(X_test_scaled)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred_clean, average="macro", zero_division=0)
    pred_adv = lineage.predict(X_test_adv)
    mal = y_test == mal_label
    asr = float((pred_adv[mal] == benign_label).mean()) if mal.any() else float("nan")
    mal_p = mal & np.asarray(succ_pocket, dtype=bool)
    asr_p = float((pred_adv[mal_p] == benign_label).mean()) if mal_p.any() else float("nan")
    per_clean = {int(s): float(lineage.score(*all_clean_sets[s])) for s in all_clean_sets}
    per_adv = {}
    for k, (Xs, ys) in all_test_sets_full.items():
        if isinstance(k, str) and k.endswith("_adversarial"):
            per_adv[int(k.split("_")[0])] = float(lineage.score(Xs, ys))
    return {
        "task": int(t), "mean_acc": float(mean_acc), "pooled_acc": float(pooled_acc),
        "task_acc": task_acc, "adv_acc": adv_acc,
        "final_task_acc_combined": (task_acc + adv_acc) / 2,
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "asr_malicious": asr, "asr_malicious_pockets": asr_p,
        "per_source_clean": per_clean, "per_source_adv": per_adv,
    }


class MetricsLog:
    """logs/rebuttal_metrics.json: {"config": {...}, "tasks": [ {task, U_t_size,
    U_t_by_class, oracle_poison_size, lineages: {name: lineage_metrics}} ]}.
    Rewritten after every task so a crashed run still leaves usable rows."""

    def __init__(self, path, config):
        self.path = path
        self.data = {"config": config, "tasks": []}
        self._flush()

    def add_task(self, entry):
        self.data["tasks"].append(entry)
        self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
