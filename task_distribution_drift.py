"""
Measures how much the CLEAN (unperturbed, pre-poisoning) feature distribution
differs from one task's batch to another, using the exact same task split as
load_pooled_chronological_tasks() in h5_data_loader.py -- the pooled,
timestamp-sorted, TASK_FRACTIONS-weighted split that both madar_cl_pipeline.py
and madar_unlearning_cl_pipeline.py build their tasks from. This script does
NOT run any red agent / poisoning -- it only loads and analyzes the raw
per-task feature batches those pipelines start from, before anything is
perturbed.

Two per-feature distributional-difference metrics are computed for every pair
of tasks (all C(10,2)=45 pairs, plus the diagonal-adjacent "consecutive task"
view that matters most for continual learning):
  - Jensen-Shannon divergence (base 2, bounded [0, 1]): a genuine shape-of-
    distribution metric, symmetric, well-defined even when two tasks' value
    ranges don't overlap.
  - Wasserstein (earth-mover) distance: how much "mass" has to move and how
    far, in the feature's own [0, 1] units -- more sensitive to *how far*
    values shifted, not just whether the shape changed.
Both are computed from fixed 25-bin histograms over [0, 1] per feature (the
pipeline's own raw features are already clipped to [0, 1] before use -- see
main()'s `np.clip(task["features"].astype(np.float32), 0.0, 1.0)` -- so a
shared fixed binning is valid across every task/feature without re-fitting
per pair), then averaged across all features into one scalar per task pair.
Reported for three scopes: all rows, benign-only, malicious-only.

Usage:
  python task_distribution_drift.py --h5-path /mnt/processed_data/subsampled_dataset.h5 --out-dir drift_out
"""

import os
import argparse

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from h5_data_loader import load_pooled_chronological_tasks

# Must match TASK_FRACTIONS in madar_cl_pipeline.py / madar_unlearning_cl_pipeline.py --
# load_pooled_chronological_tasks() renormalizes to sum to 1.0 regardless, but the
# per-task SHARE of the pooled data (a big warm-start task 0, tapering after) only
# matches the real pipeline run if this list matches theirs.
NUM_TASKS = 10
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]

N_BINS = 25
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0


def per_feature_histograms(X):
    """X: (n_samples, n_features), already clipped to [0, 1]. Returns
    (n_features, N_BINS) probability histograms, one row per feature."""
    n_features = X.shape[1]
    hists = np.empty((n_features, N_BINS), dtype=np.float64)
    for f in range(n_features):
        counts, _ = np.histogram(X[:, f], bins=BIN_EDGES)
        total = counts.sum()
        hists[f] = counts / total if total > 0 else np.full(N_BINS, 1.0 / N_BINS)
    return hists


def mean_js_and_wasserstein(hist_a, hist_b):
    """hist_a/hist_b: (n_features, N_BINS) probability histograms (same features,
    same bins). Returns (mean_js_divergence, mean_wasserstein) averaged over
    features. JS uses scipy's jensenshannon (a DISTANCE = sqrt(divergence));
    squared here to report the divergence itself, bounded [0, 1] at base=2."""
    n_features = hist_a.shape[0]
    js_vals = np.empty(n_features)
    wd_vals = np.empty(n_features)
    for f in range(n_features):
        js_dist = jensenshannon(hist_a[f], hist_b[f], base=2)
        js_vals[f] = 0.0 if np.isnan(js_dist) else js_dist ** 2
        wd_vals[f] = wasserstein_distance(BIN_CENTERS, BIN_CENTERS, u_weights=hist_a[f], v_weights=hist_b[f])
    return float(js_vals.mean()), float(wd_vals.mean())


def build_scope_histograms(tasks, benign_label):
    """Returns {scope: {task_id: (n_features, N_BINS) histograms}} for
    scope in ("all", "benign", "malicious")."""
    scopes = {"all": {}, "benign": {}, "malicious": {}}
    for t, task in enumerate(tasks):
        X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y = task["labels"].astype(np.int64)
        scopes["all"][t] = per_feature_histograms(X)
        scopes["benign"][t] = per_feature_histograms(X[y == benign_label]) if (y == benign_label).any() else None
        mal_mask = y != benign_label
        scopes["malicious"][t] = per_feature_histograms(X[mal_mask]) if mal_mask.any() else None
    return scopes


def compute_pairwise_report(scopes):
    """Returns a list of dict rows: task_a, task_b, scope, mean_js_divergence,
    mean_wasserstein -- one row per (unordered task pair, scope)."""
    rows = []
    for scope_name, per_task_hists in scopes.items():
        task_ids = sorted(per_task_hists.keys())
        for i, ta in enumerate(task_ids):
            hist_a = per_task_hists[ta]
            if hist_a is None:
                continue
            for tb in task_ids[i + 1:]:
                hist_b = per_task_hists[tb]
                if hist_b is None:
                    continue
                mean_js, mean_wd = mean_js_and_wasserstein(hist_a, hist_b)
                rows.append({
                    "task_a": ta, "task_b": tb, "scope": scope_name,
                    "mean_js_divergence": mean_js, "mean_wasserstein": mean_wd,
                })
    return rows


def print_consecutive_summary(rows):
    print("\nConsecutive-task drift (task t vs task t+1), scope=all:")
    print(f"{'task pair':>10} {'mean JS divergence':>20} {'mean Wasserstein':>18}")
    by_pair = {(r["task_a"], r["task_b"]): r for r in rows if r["scope"] == "all"}
    for t in range(NUM_TASKS - 1):
        r = by_pair.get((t, t + 1))
        if r is None:
            continue
        print(f"{t:>4}->{t+1:<4} {r['mean_js_divergence']:>20.4f} {r['mean_wasserstein']:>18.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="task_distribution_drift_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.h5_path} and building {NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(args.h5_path, TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    print(f"day_mapping={day_mapping}, feature_dim={tasks[0]['features'].shape[1]}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    scopes = build_scope_histograms(tasks, benign_label)
    rows = compute_pairwise_report(scopes)

    import csv
    out_path = os.path.join(args.out_dir, "distribution_drift_report.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_a", "task_b", "scope", "mean_js_divergence", "mean_wasserstein"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} pairwise rows to {out_path}")

    print_consecutive_summary(rows)


if __name__ == "__main__":
    main()
