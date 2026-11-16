"""
madar_pocket_pipeline_meta_detect.py

Same poisoning, test-time attack, unlearning mechanics (dropped_rows /
amnesiac / opposite_class), buffers, and lineage set as
madar_pocket_pipeline.py -- imported directly from it below, not
reimplemented. The ONLY thing that changes is HOW `detected_poison_idx` is
produced each task: instead of one xgboost/logistic classifier trained on a
balanced oracle-labeled batch (`run_detector`), this uses a meta-feature
detector originally ported from task8_pipeline-Copy1-withmeta.ipynb's
"META-LEARNED POISON DETECTOR (checkpoint1 only, episodic Reptile)" cell
(the notebook's earlier "BAD" oracle-leaking cell is still not ported).

NOTE ON THE DETECTOR CORE: the notebook's own separator was a linear
logistic model fit via Reptile (inner-loop SGD + meta-parameter blending
across episodes) -- the first version of this file ported that faithfully.
That linear separator measurably underperformed (see the per-task recall
comparison against madar_pocket_pipeline.py's oracle xgboost detector: ~50-
68% precision vs. ~85-100%, over-flagging clean rows by 1.5-2x every task).
As a first ablation at improving it, the SEPARATOR was swapped from that
linear/Reptile-SGD model to a single small xgboost classifier (`base.
train_detector("xgboost", ...)`, i.e. the SAME XGBOOST_PARAMS as the oracle
detector: n_estimators=20, max_depth=2) fit ONCE on the episodically-sampled
touched set. The episodic SAMPLING procedure (budget, oracle-grounded
poison/clean pools, held-out split) is unchanged -- only the model that
learns from those samples changed. This is no longer literally "Reptile"
(there is no per-episode gradient step or meta-parameter blend left), so
--meta_inner_lr/--meta_lr are accepted for CLI compatibility but are UNUSED.

SAME FIVE lineages as madar_pocket_pipeline.py: clean, poisoned_baseline,
dropped_rows, amnesiac, opposite_class -- `apply_dropped_rows`/
`apply_amnesiac`/`apply_opposite_class` are reused UNCHANGED from `base`,
so a fix's mechanism is byte-identical between the two files; only the
index array it's handed differs. Because the lineage set and log section
shapes are identical to madar_pocket_pipeline.py, the existing analysis
scripts (plot_pipeline_metrics.py / summarize_pipeline_runs.py /
build_latex_tables.py) need no changes to read this file's logs -- but if
you want to compare THIS file's dropped_rows/amnesiac/opposite_class
against the SAME-NAMED lineages in a madar_pocket_pipeline.py run on one
plot, you'll need --split-lineage/--rename (see plot_pipeline_metrics.py's
docstring) to tell the two apart, since the names collide by design.

THE META-DETECTOR ITSELF, faithful to the notebook's scope apart from the
xgboost-separator ablation above (deliberately, per design discussion -- not
an oversight):

  * SINGLE-TASK episodic sampling: every episode's samples are drawn from
    THIS task's own poisoned training batch only. There is no cross-task
    transfer of a meta-learned initialization between tasks, even though
    this pipeline (unlike the notebook's one-off diagnostic) has 9 poisoned
    tasks it could in principle learn across. That would be a materially
    different, more ambitious design (meta-train on tasks 1..t-1, fast-
    adapt to task t with a small probe) -- deliberately out of scope here,
    kept for a possible later file.

  * PER-SAMPLE FEATURES (7-dim, computed from `poisoned_baseline`'s adapted
    model + its own replay buffer, mostly unsupervised): two IsolationForest
    anomaly scores (raw feature space, and a latent space pulled via a
    forward hook), per-sample cross-entropy loss / entropy / top-2 softmax
    margin from the model's own forward pass, a logistic density-ratio
    score discriminating "replay buffer" from "this task's data", and mean
    distance to each point's k-nearest same-class neighbors in the replay
    buffer. The latent-feature hook is GENERALIZED from the notebook's
    hardcoded `model.fc4_bn` (128-dim) to `model.bns[-1]` (whatever width
    the last configured --hidden_sizes block has), so this works for any
    architecture depth, matching the same generalization pattern already
    used for DEDUCE's GUM.

  * EPISODIC SAMPLING budget matches the notebook's original Reptile budget
    exactly: 8 outer episodes x 3 inner steps x 10 samples/step (5 poisoned +
    5 clean per step, drawn from the ORACLE poison index -- same as the
    existing detector's own oracle-labeled training set, just consumed as
    small episodic draws instead of one flat balanced batch). This is NOT a
    smaller labeled budget than the existing detector's ~240-label
    convention -- it touches ~235-240 unique points, essentially the same
    order. Pass --meta_outer_episodes/--meta_inner_steps/--meta_samples_per_step
    to make it smaller if you want that comparison instead.

  * HELD-OUT EVALUATION: every point NOT touched by any episode is scored
    by the xgboost classifier fit on the touched set, with no further
    fitting on held-out data -- a genuine few-shot generalization test
    within the task, not a train/test split of a bigger labeled set.

A note on optimizers: as in every other lineage in this project's
pipelines, dropped_rows/amnesiac/opposite_class/poisoned_baseline/clean all
train via AdaptableClassifier's Adam optimizer -- unaffected by any of this,
since the meta-detector only decides WHICH indices those fixes are handed.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import madar_pocket_pipeline as base

LINEAGE_NAMES = ["clean", "poisoned_baseline", "dropped_rows", "amnesiac", "opposite_class"]
FIX_NAMES = ["dropped_rows", "amnesiac", "opposite_class"]


def _buffer_ids_only(label_buffers):
    """Drops each replay-buffer entry's feature row (index 0), keeping only
    (label, category, sample_id) -- used ONLY when writing checkpoints, since
    MEM_SIZE-scale feature rows dominate checkpoint file size. NOT used at
    runtime: flatten_replay()/update_shared_buffer() need the real feature
    rows, so this only ever touches the serialized copy going into the
    checkpoint dict, never the live in-memory buffers."""
    return {lbl: [tuple(e[1:]) for e in entries] for lbl, entries in label_buffers.items()}


def _replay_ids_only(replay_buffer):
    """Same as _buffer_ids_only, for the flattened list form."""
    return [tuple(e[1:]) for e in replay_buffer]


# ---------------------------------------------------------------------------
# Timestamped step-by-step timing log ("meta_log.txt"), separate from the
# human-readable pipeline_log.txt -- for diagnosing which step of a real run
# is actually slow. Module-level so every helper function below can log
# without threading a logger through every call signature. main() points
# these at <out_dir>/logs/meta_log.txt before the task loop starts.
# ---------------------------------------------------------------------------
_META_LOG_PATH = None
_META_LOG_T0 = None


def _tlog(msg):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    elapsed = time.perf_counter() - _META_LOG_T0 if _META_LOG_T0 is not None else 0.0
    line = f"[{now} | +{elapsed:8.1f}s] {msg}"
    print(line)
    if _META_LOG_PATH is not None:
        with open(_META_LOG_PATH, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Episodic meta-detector -- feature extraction ported from
# task8_pipeline-Copy1-withmeta.ipynb's "META-LEARNED POISON DETECTOR
# (checkpoint1 only, episodic Reptile)" cell; the separator itself is now a
# small xgboost classifier (see module docstring for the ablation).
# ---------------------------------------------------------------------------
def get_latent_features(model, X, batch_size=512):
    """Runs X through `model` and returns the last hidden block's activation
    (post-BN, pre-ReLU captured via a forward hook, ReLU applied manually --
    matches the real forward pass exactly). Hooks model.bns[-1] rather than
    a hardcoded layer name, so this works for any --hidden_sizes depth."""
    _tlog(f"  [get_latent_features] start, {len(X)} rows, batch_size={batch_size}")
    model.eval()
    captured = {}

    def _hook(_module, _inp, out):
        captured["z"] = out.detach()

    handle = model.bns[-1].register_forward_hook(_hook)
    latents = []
    with torch.no_grad():
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        for i in range(0, len(Xt), batch_size):
            _ = model(Xt[i:i + batch_size])
            latents.append(torch.relu(captured["z"]).numpy())
    handle.remove()
    out = np.concatenate(latents, axis=0)
    _tlog(f"  [get_latent_features] done, latent_dim={out.shape[1]}")
    return out


def replay_knn_distance(X, y, replay_X_np, replay_y_np, k=5):
    """Mean distance (raw feature space) from each point to its k nearest
    same-class neighbors in the replay buffer -- poisoned points, shifted
    toward the opposite class, should sit unusually far from their own
    class's established prior-task samples."""
    if replay_X_np is None or len(replay_X_np) == 0:
        # No replay buffer yet (e.g. right after --resume_from, which starts
        # both buffers fresh rather than persisting their feature rows) --
        # leave this feature NaN for every row; xgboost treats NaN as a
        # missing value natively, so this column is just uninformative
        # rather than a crash.
        _tlog("  [replay_knn_distance] skipped -- no replay buffer yet")
        return np.full(len(X), np.nan)
    _tlog(f"  [replay_knn_distance] start, {len(X)} rows vs replay buffer of {len(replay_X_np)}")
    dist = np.full(len(X), np.nan)
    for c in np.unique(y):
        mask = y == c
        replay_mask = replay_y_np == c
        if replay_mask.sum() == 0:
            continue
        kk = min(k, replay_mask.sum())
        nn_ = NearestNeighbors(n_neighbors=kk).fit(replay_X_np[replay_mask])
        d, _ = nn_.kneighbors(X[mask])
        dist[mask] = d.mean(axis=1)
        _tlog(f"    [replay_knn_distance] class {c} done ({int(mask.sum())} rows vs "
              f"{int(replay_mask.sum())} replay points)")
    _tlog("  [replay_knn_distance] done")
    return dist


def fit_density_ratio_model(X_current, replay_X_np):
    """Discriminator separating replay-buffer ('old distribution', label 0)
    from the current task's points ('new', label 1). p/(1-p) is the density
    ratio: values >> 1 mean a point looks much more like the current task
    than like anything in the established prior-task distribution."""
    if replay_X_np is None or len(replay_X_np) == 0:
        # No replay buffer yet (see replay_knn_distance's same guard) --
        # nothing to discriminate against; density_ratio_score handles clf=None.
        _tlog("  [fit_density_ratio_model] skipped -- no replay buffer yet")
        return None
    _tlog(f"  [fit_density_ratio_model] start, {len(replay_X_np)} replay + {len(X_current)} current rows")
    Xd = np.vstack([replay_X_np, X_current])
    yd = np.concatenate([np.zeros(len(replay_X_np)), np.ones(len(X_current))])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0).fit(Xd, yd)
    _tlog("  [fit_density_ratio_model] done")
    return clf


def density_ratio_score(clf, X):
    if clf is None:
        return np.full(len(X), np.nan)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return p / (1 - p)


def extract_meta_features(adaptable_model, X, y, replay_X_np, replay_y_np, k=5):
    """Builds the 7-dim per-sample feature table, fitting fresh
    IsolationForests + density-ratio discriminator on this task's own
    unlabeled data + replay buffer (no oracle labels used in this step)."""
    _tlog(f"[extract_meta_features] start, {len(X)} rows, feature_dim={X.shape[1]}")
    net = adaptable_model.model
    net.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)

    latent = get_latent_features(net, X)

    _tlog(f"[extract_meta_features] fitting raw-space IsolationForest ({len(X)} rows, dim={X.shape[1]})")
    iso_raw = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
    _tlog(f"[extract_meta_features] fitting latent-space IsolationForest (dim={latent.shape[1]})")
    iso_latent = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(latent)
    density_clf = fit_density_ratio_model(X, replay_X_np)
    _tlog("[extract_meta_features] scoring IsolationForests + density-ratio model")
    iso_raw_score = -iso_raw.score_samples(X)
    iso_latent_score = -iso_latent.score_samples(latent)
    density_ratio = density_ratio_score(density_clf, X)

    _tlog("[extract_meta_features] forward pass for loss/entropy/margin")
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

    _tlog("[extract_meta_features] done")
    return np.column_stack([
        iso_raw_score, iso_latent_score, per_sample_loss, entropy, top2_margin,
        density_ratio, dist_to_replay,
    ])


def run_meta_detector(poisoned_baseline, X_train_poisoned, y_train, idx_poison_ben, idx_poison_mal,
                      replay_X_np, replay_y_np, seed, n_outer_episodes, n_inner_steps,
                      samples_per_step, inner_lr, meta_lr, knn_k):
    """Episodic-budget detector: samples a touched set via the same episodic
    procedure the original Reptile port used (n_outer_episodes x
    n_inner_steps x samples_per_step draws from the oracle poison/clean
    pools), then fits ONE small xgboost classifier on that touched set
    (base.train_detector's "xgboost" path -- same XGBOOST_PARAMS as the
    oracle detector, n_estimators=20/max_depth=2) instead of the original
    linear separator trained via Reptile SGD + meta-parameter blending. See
    the module docstring for why this ablation was made.

    inner_lr/meta_lr are accepted for CLI/backward compatibility but are
    UNUSED here -- there's no gradient loop left to apply them to."""
    _tlog(f"[run_meta_detector] start, {len(y_train)} rows, {n_outer_episodes} episodes x "
          f"{n_inner_steps} inner steps x {samples_per_step} samples/step")
    poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])
    feats = extract_meta_features(poisoned_baseline, X_train_poisoned, y_train,
                                  replay_X_np, replay_y_np, k=knn_k)
    is_poisoned = np.zeros(len(y_train), dtype=int)
    is_poisoned[poison_idx] = 1

    feats_std = StandardScaler().fit_transform(feats)

    poison_pool = poison_idx
    clean_pool = np.setdiff1d(np.arange(len(y_train)), poison_idx)

    rng = np.random.default_rng(seed)
    touched = set()
    n_pos = samples_per_step // 2
    n_neg = samples_per_step - n_pos

    _tlog("[run_meta_detector] sampling episodic touched set")
    for ep in range(n_outer_episodes):
        for _ in range(n_inner_steps):
            idx_p = rng.choice(poison_pool, min(n_pos, len(poison_pool)), replace=False)
            idx_c = rng.choice(clean_pool, min(n_neg, len(clean_pool)), replace=False)
            touched.update(idx_p.tolist())
            touched.update(idx_c.tolist())
        _tlog(f"  [run_meta_detector] episode {ep + 1}/{n_outer_episodes} done "
              f"({len(touched)} unique touched so far)")
    _tlog("[run_meta_detector] episodic sampling done")

    touched_idx = np.array(sorted(touched))
    Xb = feats_std[touched_idx]
    yb = is_poisoned[touched_idx]

    _tlog(f"[run_meta_detector] fitting small xgboost detector on {len(touched_idx)} touched points")
    clf, detector_type_used = base.train_detector("xgboost", Xb, yb, seed)

    query_mask = np.ones(len(is_poisoned), dtype=bool)
    query_mask[touched_idx] = False
    Xq, yq = feats_std[query_mask], is_poisoned[query_mask]
    scores = clf.predict_proba(Xq)[:, 1]
    preds = (scores > 0.5).astype(int)

    _tlog(f"[run_meta_detector] scoring {int(query_mask.sum())} held-out points")
    if len(np.unique(yq)) > 1:
        # average=None, labels=[0,1] -- per-class (clean vs perturbed) precision/recall,
        # not just the perturbed/positive-class numbers "binary" would give.
        (prec_clean, prec_pert), (rec_clean, rec_pert), (f1_clean, f1_pert), _ = \
            precision_recall_fscore_support(yq, preds, average=None, labels=[0, 1], zero_division=0)
        auc = roc_auc_score(yq, scores)
    else:
        prec_clean = prec_pert = rec_clean = rec_pert = f1_clean = f1_pert = auc = float("nan")

    meta_detected_poison_idx = np.where(query_mask)[0][preds == 1]
    _tlog(f"[run_meta_detector] done, flagged {len(meta_detected_poison_idx)}")

    metrics = {
        "detector_type": f"episodic_meta ({detector_type_used}, single-task episodic sampling)",
        "n_outer_episodes": n_outer_episodes, "n_inner_steps": n_inner_steps,
        "samples_per_step": samples_per_step,
        "nominal_budget": n_outer_episodes * n_inner_steps * samples_per_step,
        "n_touched": len(touched), "n_query": int(query_mask.sum()),
        "held_out_precision_clean": float(prec_clean), "held_out_recall_clean": float(rec_clean),
        "held_out_precision_perturbed": float(prec_pert), "held_out_recall_perturbed": float(rec_pert),
        "held_out_f1": float(f1_pert), "held_out_auc": float(auc),
        "n_detected": len(meta_detected_poison_idx), "n_oracle": len(poison_idx),
    }
    return meta_detected_poison_idx, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _META_LOG_PATH, _META_LOG_T0
    start_time = time.perf_counter()
    _META_LOG_T0 = start_time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_meta_detect_run")
    ap.add_argument("--h5-path", type=str, default=base.H5_DATASET_PATH)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--hidden_sizes", type=str,
                     default=",".join(str(h) for h in base.DEFAULT_HIDDEN_SIZES),
                     help="Comma-separated hidden-layer widths for ClassifierNN. "
                          "Same meaning/default as in madar_pocket_pipeline.py.")
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline.py. "
                          "Off by default.")
    ap.add_argument("--meta_outer_episodes", type=int, default=8,
                     help="Episodic-sampling outer-loop count (notebook's original Reptile default: 8).")
    ap.add_argument("--meta_inner_steps", type=int, default=3,
                     help="Sampling steps per episode (notebook's original Reptile default: 3).")
    ap.add_argument("--meta_samples_per_step", type=int, default=10,
                     help="Samples per step, split evenly poisoned/clean (notebook default: "
                          "10 -- 5+5). Total nominal labeled budget = episodes * steps * this value; "
                          "lower this to test the meta-detector at a genuinely SMALLER labeled "
                          "budget than the existing oracle detector's ~240-label convention.")
    ap.add_argument("--meta_inner_lr", type=float, default=0.5,
                     help="UNUSED by the current xgboost-separator variant -- kept for CLI "
                          "compatibility with the original linear/Reptile-SGD separator.")
    ap.add_argument("--meta_lr", type=float, default=0.3,
                     help="UNUSED by the current xgboost-separator variant -- kept for CLI "
                          "compatibility with the original linear/Reptile-SGD separator.")
    ap.add_argument("--meta_knn_k", type=int, default=5,
                     help="k for the replay-buffer same-class nearest-neighbor distance feature.")
    ap.add_argument("--adapt_epochs", type=int, default=base.ADAPT_EPOCHS,
                     help="Epochs each lineage trains for when adapting to poisoned data (step 4) "
                          f"and again when applying its fix (step 8). Base default: {base.ADAPT_EPOCHS}. "
                          "Overrides base.ADAPT_EPOCHS for this run only.")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}.")
    ap.add_argument("--resume_from", type=str, default=None,
                     help="Path to a per-task checkpoint (classifier_checkpoint_task<t>.pt) to resume "
                          "from. Restores lineage weights and results, then continues from the task "
                          "AFTER the checkpoint's own task_id. Both replay buffers start FRESH/empty "
                          "(refilling naturally over the next few tasks) and task_test_splits starts "
                          "empty too (so the 'Adversarial test-set breakdown' table won't show spillover "
                          "rows for tasks before the resume point) -- neither's feature rows are "
                          "persisted in the checkpoint (only each entry's label/category/sample_id is, "
                          "to keep checkpoint files small), so there's nothing to restore them from. "
                          "--seed/--hidden_sizes/--poison_fraction/--per_feature_epsilon are restored "
                          "from the checkpoint itself (a mismatch with what you passed is a warning, "
                          "not an error) since they're baked into the saved weights/history; "
                          "--adapt_epochs and the --meta_* flags are still taken fresh from the command "
                          "line, since they only affect tasks not yet run.")
    args = ap.parse_args()
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=base.DEVICE, weights_only=False)
        if tuple(resume_ckpt["hidden_sizes"]) != hidden_sizes:
            print(f"[resume] --hidden_sizes {hidden_sizes} ignored -- using checkpoint's "
                  f"{tuple(resume_ckpt['hidden_sizes'])} instead (weights were trained with it).")
        hidden_sizes = tuple(resume_ckpt["hidden_sizes"])
        if resume_ckpt["seed"] != args.seed:
            print(f"[resume] --seed {args.seed} ignored -- using checkpoint's {resume_ckpt['seed']} instead.")
        args.seed = resume_ckpt["seed"]
        if resume_ckpt["poison_fraction"] != args.poison_fraction:
            print(f"[resume] --poison_fraction {args.poison_fraction} ignored -- using checkpoint's "
                  f"{resume_ckpt['poison_fraction']} instead.")
        poison_fraction = resume_ckpt["poison_fraction"]
        if resume_ckpt["per_feature_epsilon"] != args.per_feature_epsilon:
            print(f"[resume] --per_feature_epsilon {args.per_feature_epsilon} ignored -- using "
                  f"checkpoint's {resume_ckpt['per_feature_epsilon']} instead.")
        args.per_feature_epsilon = resume_ckpt["per_feature_epsilon"]
    else:
        poison_fraction = args.poison_fraction

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    base.ADAPT_EPOCHS = args.adapt_epochs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_meta_detect", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    meta_log_path = os.path.join(out_dir, "logs", "meta_log.txt")

    def checkpoint_path_for(task_id):
        return os.path.join(out_dir, "logs", f"classifier_checkpoint_task{task_id}.pt")

    _META_LOG_PATH = meta_log_path
    resuming_same_log = bool(resume_ckpt) and os.path.exists(meta_log_path)
    with open(meta_log_path, "a" if resuming_same_log else "w") as f:
        if resuming_same_log:
            f.write(f"\n--- RESUMED from {args.resume_from} (after task {resume_ckpt['task_id']}) "
                    f"at {datetime.datetime.now().isoformat()} -- args: {vars(args)}\n\n")
        else:
            f.write(f"META-DETECT TIMING LOG -- run started {datetime.datetime.now().isoformat()}\n"
                    f"args: {vars(args)}\n\n")
    _tlog("Run starting" if not resume_ckpt else f"Run resuming from {args.resume_from}")

    with open(log_path, "a" if resuming_same_log else "w") as f:
        if resuming_same_log:
            f.write(f"\n--- RESUMED from {args.resume_from} (after task {resume_ckpt['task_id']}) "
                    f"at {datetime.datetime.now().isoformat()} ---\n")
        else:
            f.write(
                "MADAR POCKET-PIPELINE LOG (episodic meta-detector: xgboost separator)\n"
                "======================================================================\n"
                "5 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
                "dropped_rows, amnesiac, opposite_class. Same poisoning/attack/unlearning\n"
                "mechanics as madar_pocket_pipeline.py -- see that file. The ONLY difference\n"
                "is the detector: a single-task episodic sampling procedure (originally ported\n"
                "from task8_pipeline-Copy1-withmeta.ipynb as a Reptile-trained linear\n"
                "separator) feeding a small xgboost classifier, in place of the oracle\n"
                "detector's balanced-batch xgboost/logistic classifier.\n"
                f"Classifier hidden layer sizes: {hidden_sizes}\n"
                f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
                f"Episodic sampling: {args.meta_outer_episodes} episodes x {args.meta_inner_steps} inner "
                f"steps x {args.meta_samples_per_step} samples/step, knn_k={args.meta_knn_k} "
                f"(--meta_inner_lr/--meta_lr are unused by this variant)\n"
                f"Adapt epochs (per-task fix/poison training): {args.adapt_epochs}\n"
            )

    print(f"Loading {args.h5_path} and building {base.NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = base.load_pooled_chronological_tasks(args.h5_path, base.TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    if resume_ckpt:
        if resume_ckpt["feature_dim"] != feature_dim:
            raise ValueError(f"Checkpoint's feature_dim ({resume_ckpt['feature_dim']}) doesn't match "
                              f"this --h5-path's feature_dim ({feature_dim}) -- wrong dataset?")
        scaler = resume_ckpt["scaler"]
        lineages = {}
        for name in LINEAGE_NAMES:
            m = base.ClassifierNN(feature_dim, 2, hidden_sizes=hidden_sizes).to(base.DEVICE)
            m.load_state_dict(resume_ckpt["lineages"][name])
            lineages[name] = base.AdaptableClassifier(m)
        # Replay buffers and task_test_splits are NOT restored from the
        # checkpoint -- their feature rows are the expensive part and are
        # intentionally not persisted (see _buffer_ids_only/_replay_ids_only
        # on the save side). Both start fresh: the replay buffers refill
        # naturally over the next few tasks via update_shared_buffer(), same
        # as a brand-new run's task 0; task_test_splits simply won't have
        # entries for tasks before the resume point, so the "Adversarial
        # test-set breakdown" table's spillover rows for those tasks are
        # silently absent going forward rather than reconstructed.
        baseline_label_buffers, baseline_replay_buffer = {}, []
        joint_label_buffers, joint_replay_buffer = {}, []
        task_test_splits = {}
        task_test_gids = resume_ckpt["task_test_gids"]
        results = resume_ckpt["results"]
        start_task = resume_ckpt["task_id"] + 1
        _tlog(f"Resumed: restored 5 lineages from task {resume_ckpt['task_id']} (replay buffers and "
              f"task_test_splits start fresh -- not persisted in checkpoints), continuing at task "
              f"{start_task}")
    else:
        scaler = None
        lineages = {}
        baseline_label_buffers, baseline_replay_buffer = {}, []
        joint_label_buffers, joint_replay_buffer = {}, []
        task_test_splits, task_test_gids = {}, {}
        results = []
        start_task = 0

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -base.FEATURE_CLIP,
                       base.FEATURE_CLIP).astype(np.float32)

    for t in range(start_task, base.NUM_TASKS):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        _tlog(f"=== Task {t}: start ===")
        task = tasks[t]
        X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y_all = task["labels"].astype(np.int64)
        gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

        X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
            X_raw, y_all, gid_all, test_size=base.TASK_TEST_FRAC, random_state=args.seed, stratify=y_all,
        )
        _tlog(f"Task {t}: train/test split done (train={len(y_train)}, test={len(y_test)})")

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = to_scaled(X_train_raw)
            X_test_scaled = to_scaled(X_test_raw)

            base_model = base.ClassifierNN(feature_dim, 2, hidden_sizes=hidden_sizes).to(base.DEVICE)
            Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
            yt = torch.tensor(y_train, dtype=torch.long)
            opt0 = torch.optim.Adam(base_model.parameters(), lr=base.TASK0_LR)
            loss_fn0 = nn.CrossEntropyLoss()
            base_model.train()
            n = len(Xt)
            _tlog(f"Task 0: pretraining for {base.TASK0_EPOCHS} epochs on {n} rows")
            for epoch in range(base.TASK0_EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, base.TASK0_BATCH_SIZE):
                    idx = perm[i:i + base.TASK0_BATCH_SIZE]
                    if len(idx) < 2:
                        continue
                    opt0.zero_grad()
                    loss = loss_fn0(base_model(Xt[idx]), yt[idx])
                    loss.backward()
                    opt0.step()
                _tlog(f"  Task 0: pretraining epoch {epoch + 1}/{base.TASK0_EPOCHS} done")
            base_model.eval()
            _tlog("Task 0: pretraining done")

            for name in LINEAGE_NAMES:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            category = np.where(y_train == benign_label, "benign", "malicious_clean")
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )
            joint_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], joint_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )
            _tlog("Task 0: replay buffers filled")

            train_section = (
                f"malicious: {int((y_train == mal_label).sum())}, benign: {int((y_train == benign_label).sum())}\n"
                f"malicious_perturbed: 0, benign_perturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = (
                f"malicious: {int((y_test == mal_label).sum())}, benign: {int((y_test == benign_label).sum())}\n"
                f"genuine pockets: N/A (task 0 -- no poisoning yet)\n"
            )
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            unlearn_section = "N/A -- task 0 has no prior model to poison against."

            base.write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Unlearning step", unlearn_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
                "label_mapping": label_mapping,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                # Feature rows intentionally NOT persisted (they dominate checkpoint file size) --
                # only each entry's (label, category, sample_id) survives; see _buffer_ids_only.
                "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
                "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
                "joint_label_buffers": _buffer_ids_only(joint_label_buffers),
                "joint_replay_buffer": _replay_ids_only(joint_replay_buffer),
                # task_test_splits (X_test_raw, y_test per task) intentionally NOT persisted, same
                # reason -- task_test_gids (just the row IDs) still is.
                "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
                "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path_for(t))
            _tlog(f"=== Task {t}: done (log + checkpoint written) ===")
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> detect (episodic meta-detector) -> unlearn.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        replay_X, replay_y = base.flatten_replay(joint_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)

        # Step 2: clean lineage adapts on clean data only.
        _tlog(f"Task {t}: step 2 -- adapting clean lineage")
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=replay_X, replay_y=replay_y,
                                epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison (shared across all poisoned lineages).
        _tlog(f"Task {t}: step 3 -- crafting poison (poison_fraction={poison_fraction})")
        X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = base.craft_task_poison(
            lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
        )
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])
        _tlog(f"Task {t}: step 3 done ({len(poison_idx)} poisoned rows)")

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix"), using
        # its OWN separate replay buffer.
        _tlog(f"Task {t}: step 4 -- adapting poisoned_baseline")
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                            replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                            epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )

        # Step 5: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean).
        _tlog(f"Task {t}: step 5 -- crafting this task's adversarial test attack")
        eps_this_task = base.typical_class_gap(X_test_scaled, y_test, benign_label,
                                               mal_label) * base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, per_feature_epsilon=args.per_feature_epsilon,
        )
        _tlog(f"Task {t}: step 5 done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())})")

        # Step 6: spillover check -- re-attack every PRIOR task's test set.
        _tlog(f"Task {t}: step 6 -- spillover re-attack over {len(task_test_splits)} prior task(s)")
        historical_adv = {}
        for s, (Xs_raw, ys) in task_test_splits.items():
            Xs_scaled = to_scaled(Xs_raw)
            eps_s = base.typical_class_gap(Xs_scaled, ys, benign_label, mal_label)
            eps_s = eps_s * base.ATTACK_EPS_MULTIPLIER if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = base.adversarial_attack_pocket(
                lineages["poisoned_baseline"], lineages["clean"], Xs_scaled, ys, epsilon_max=eps_s,
                per_feature_epsilon=args.per_feature_epsilon,
            )
            historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)
            _tlog(f"  Task {t}: step 6 -- re-attacked source task {s} ({len(ys)} rows)")
        _tlog(f"Task {t}: step 6 done")

        all_clean_sets = {s: (to_scaled(Xs_raw), ys) for s, (Xs_raw, ys) in task_test_splits.items()}
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pocket_info_by_source = {s: (v[2], v[3]) for s, v in historical_adv.items()}
        pocket_info_by_source[t] = (succ_pocket, eps_this_task)

        # Step 7: episodic meta-detector, trained from poisoned_baseline's own
        # (episodic, oracle-few-shot) poison pool + its own replay buffer.
        _tlog(f"Task {t}: step 7 -- running episodic meta-detector")
        meta_detected_poison_idx, det_metrics = run_meta_detector(
            lineages["poisoned_baseline"], X_train_poisoned, y_train, idx_poison_ben, idx_poison_mal,
            baseline_replay_X, baseline_replay_y, args.seed,
            args.meta_outer_episodes, args.meta_inner_steps, args.meta_samples_per_step,
            args.meta_inner_lr, args.meta_lr, args.meta_knn_k,
        )
        _tlog(f"Task {t}: step 7 done (flagged {det_metrics['n_detected']}, "
              f"oracle={det_metrics['n_oracle']})")

        # Step 8: each fix lineage adapts on poisoned data from ITS OWN prior
        # weights, snapshot pre-unlearning metrics, then applies its own fix
        # to the SAME meta_detected_poison_idx.
        _tlog(f"Task {t}: step 8 -- adapting fix lineages + snapshotting pre-unlearning metrics")
        pre_unlearn_metrics = {}
        for name in FIX_NAMES:
            lineages[name].adapt(X_train_poisoned, y_train, replay_X=replay_X, replay_y=replay_y,
                                 epochs=base.ADAPT_EPOCHS)
            pre_task_acc = lineages[name].score(X_test_scaled, y_test)
            pre_pooled_acc, pre_mean_acc, _ = base.pooled_and_per_task_accuracy(lineages[name], all_test_sets_full)
            pre_unlearn_metrics[name] = {
                "task_acc": pre_task_acc, "pooled_acc": pre_pooled_acc, "mean_acc": pre_mean_acc,
            }
            _tlog(f"  Task {t}: step 8 -- {name} pre-unlearning snapshot done")
        _tlog(f"Task {t}: step 8 -- applying fixes (dropped_rows/amnesiac/opposite_class)")
        base.apply_dropped_rows(lineages["dropped_rows"], X_train_poisoned, y_train, meta_detected_poison_idx,
                                replay_X, replay_y)
        base.apply_amnesiac(lineages["amnesiac"], X_train_poisoned, y_train, meta_detected_poison_idx,
                            replay_X, replay_y, benign_label, mal_label)
        base.apply_opposite_class(lineages["opposite_class"], X_train_poisoned, y_train, meta_detected_poison_idx,
                                  replay_X, replay_y, benign_label, mal_label)
        _tlog(f"Task {t}: step 8 done")

        # Step 9: pooled/mean/per-class accuracy across all clean + adversarial
        # test sets seen so far, for every lineage.
        _tlog(f"Task {t}: step 9 -- computing pooled/mean/per-class accuracy for all lineages")
        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = base.pooled_and_per_task_accuracy(lineages[name], all_test_sets_full)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test)
            _tlog(f"  Task {t}: step 9 -- {name} done")

        still_evades = {}
        for name in FIX_NAMES:
            pred = lineages[name].predict(X_test_adv)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")
        _tlog(f"Task {t}: step 9 done")

        # Step 10: update both replay buffers -- only now, after every
        # lineage's adaptation/unlearning for this task is fully done.
        _tlog(f"Task {t}: step 10 -- updating replay buffers")
        category_all = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        category_all[idx_poison_ben] = "benign_perturbed"
        category_all[idx_poison_mal] = "malicious_perturbed"

        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers,
            X_train_poisoned, y_train, category_all, gid_train, benign_label, mal_label,
        )

        clean_mask = np.ones(len(X_train_poisoned), dtype=bool)
        clean_mask[meta_detected_poison_idx] = False
        joint_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], joint_label_buffers,
            X_train_poisoned[clean_mask], y_train[clean_mask], category_all[clean_mask], gid_train[clean_mask],
            benign_label, mal_label,
        )
        _tlog(f"Task {t}: step 10 done")

        task_test_splits[t] = (X_test_raw, y_test)
        task_test_gids[t] = gid_test

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        n_ben_train = int((y_train == benign_label).sum())
        n_mal_train = int((y_train == mal_label).sum())
        train_section = (
            f"malicious: {n_mal_train}, benign: {n_ben_train}\n"
            f"malicious_perturbed (oracle): {len(idx_poison_mal)}, "
            f"benign_perturbed (oracle): {len(idx_poison_ben)}\n"
            f"meta-detector-flagged: {det_metrics['n_detected']} (of {det_metrics['n_query']} held-out points)\n"
            f"poison_fraction used: {poison_fraction}\n"
            f"poisoned_baseline accuracy on poisoned points' forced labels: {acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
        )

        n_ben_test = int((y_test == benign_label).sum())
        n_mal_test = int((y_test == mal_label).sum())
        succ_ben = int(succ_pocket[y_test == benign_label].sum())
        succ_mal = int(succ_pocket[y_test == mal_label].sum())
        test_section = (
            f"malicious: {n_mal_test}, benign: {n_ben_test}\n"
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})\n"
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        baseline_dist = base.buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution (post-update, this task): "
                            f"{baseline_dist}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
            adapt_lines.append(f"[{name}] classification report (this task's adversarial test):")
            adapt_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        adapt_section = "\n".join(adapt_lines)

        joint_dist = base.buffer_distribution(joint_label_buffers)
        unlearn_lines = [
            f"detector type: {det_metrics['detector_type']}",
            f"meta-training: {det_metrics['n_outer_episodes']} episodes x {det_metrics['n_inner_steps']} inner "
            f"steps x {det_metrics['samples_per_step']} samples/step ({det_metrics['nominal_budget']} nominal, "
            f"{det_metrics['n_touched']} unique touched)",
            f"held-out (episode-untouched) vs oracle -- clean: P={det_metrics['held_out_precision_clean']:.3f} "
            f"R={det_metrics['held_out_recall_clean']:.3f}, perturbed: "
            f"P={det_metrics['held_out_precision_perturbed']:.3f} "
            f"R={det_metrics['held_out_recall_perturbed']:.3f} (f1={det_metrics['held_out_f1']:.3f} "
            f"auc={det_metrics['held_out_auc']:.3f}, on {det_metrics['n_query']} held-out points)",
            f"flagged {det_metrics['n_detected']} as poisoned; oracle poisoned this task = "
            f"{det_metrics['n_oracle']}",
            f"JOINT replay buffer distribution (dropped_rows/amnesiac/opposite_class share this one; "
            f"meta-detector-clean only; post-update, this task): {joint_dist}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}",
        ]
        unlearn_lines.append("")
        unlearn_lines.append("Pre-unlearning (poison-adapted, before any fix) accuracy:")
        unlearn_lines.append(f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}")
        for name in FIX_NAMES:
            pre = pre_unlearn_metrics[name]
            unlearn_lines.append(
                f"{name:<18} {pre['task_acc']:>10.3f} {pre['pooled_acc']:>12.3f} {pre['mean_acc']:>10.3f}"
            )
        unlearn_lines.append("")
        unlearn_lines.append("Post-unlearning accuracy:")
        unlearn_lines.append(
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}"
        )
        for name in FIX_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adv_acc_name = lineages[name].score(X_test_adv, y_test)
            unlearn_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        unlearn_lines.append("")
        for name in FIX_NAMES:
            unlearn_lines.append(f"[{name}] classification report (this task's clean test):")
            unlearn_lines.append(per_class_reports[name])
            unlearn_lines.append(f"[{name}] classification report (this task's adversarial test):")
            unlearn_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        unlearn_section = "\n".join(unlearn_lines)

        breakdown_lines = [
            "Genuine-pocket rate per source task's adv-test-set, attacked FRESH this task",
            "(against THIS task's poisoned_baseline/clean -- not the rate recorded when",
            "that set was first created at its own task):",
            "",
            f"{'source task':<12} {'n':>8} {'genuine pockets':>18} {'rate':>8} {'eps':>8}",
        ]
        for s in sorted(pocket_info_by_source.keys()):
            succ_s, eps_s = pocket_info_by_source[s]
            n_s = len(succ_s)
            breakdown_lines.append(
                f"{s:<12} {n_s:>8} {int(succ_s.sum()):>10}/{n_s:<7} {base._fmt_pct(succ_s.mean()):>8} {eps_s:>8.4f}"
            )
        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-unlearning) classifier accuracy on each source task's adv-test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-unlearning) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-unlearning) classifier COMBINED (clean+adversarial, pooled) "
            f"accuracy on each source task's test-set:"
        )
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            Xs_clean, ys_clean = all_clean_sets[s]
            Xs_adv, ys_adv = all_test_sets_full[f"{s}_adversarial"]
            X_comb = np.vstack([Xs_clean, Xs_adv])
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb):>18.3f}"
            breakdown_lines.append(row)
        breakdown_section = "\n".join(breakdown_lines)

        # ---------------------------------------------------------------
        # Debug: pocket send/recovery summary -- how many test points were
        # sent into a genuine pocket this task, how many of those are no
        # longer evading (recovered) after each fix, how healthy each fix
        # looks on ordinary (non-attacked) traffic right now, and whether
        # fixing THIS task accidentally reopened any EARLIER task's pockets.
        # Always printed+logged; followed by an interactive breakpoint()
        # unless --no_breakpoint. Written in plain language on purpose --
        # this is meant to be read live, task by task, not computed from.
        # ---------------------------------------------------------------
        n_pocketed = int(succ_pocket.sum())
        pocket_lines = [
            f"Sent into pockets (genuine pockets found, this task's test set): "
            f"{n_pocketed}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})",
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}",
            "",
            f"Recovered after unlearning (of the {n_pocketed} pocketed points, no longer "
            f"evading post-fix):",
        ]
        fix_still_evading = {}
        for name in FIX_NAMES:
            pred_name = lineages[name].predict(X_test_adv)
            still_evading_mask = (pred_name != y_test) & succ_pocket
            fix_still_evading[name] = still_evading_mask
            n_recovered = n_pocketed - int(still_evading_mask.sum())
            pct_recovered = base._fmt_pct(n_recovered / n_pocketed) if n_pocketed else "N/A"
            pocket_lines.append(f"  {name:<18}: {n_recovered}/{n_pocketed} recovered ({pct_recovered})")

        pocket_lines.append("")
        pocket_lines.append("How well each fix reads NORMAL (non-attacked) traffic right now:")
        for name in FIX_NAMES:
            pred_clean = lineages[name].predict(X_test_scaled)
            catch_ben = (pred_clean[y_test == benign_label] == benign_label).mean() if n_ben_test else float("nan")
            catch_mal = (pred_clean[y_test == mal_label] == mal_label).mean() if n_mal_test else float("nan")
            pocket_lines.append(
                f"  {name:<18}: catches {base._fmt_pct(catch_ben)} of benign, "
                f"{base._fmt_pct(catch_mal)} of malicious"
            )

        pocket_lines.append("")
        pockets_to_check = [
            (s, historical_adv[s]) for s in sorted(historical_adv.keys()) if int(historical_adv[s][2].sum()) > 0
        ]
        if pockets_to_check:
            pocket_lines.append(
                "Checking back on earlier tasks' pockets (did fixing THIS task "
                "accidentally reopen any of them?):"
            )
            for s, (Xs_adv, ys, succ_s, eps_s) in pockets_to_check:
                n_pocketed_s = int(succ_s.sum())
                pocket_lines.append(f"  Task {s}'s pockets ({n_pocketed_s} total):")
                for name in FIX_NAMES:
                    pred_s = lineages[name].predict(Xs_adv)
                    n_still_closed = int((succ_s & (pred_s == ys)).sum())
                    pocket_lines.append(
                        f"    {name:<18}: {n_still_closed}/{n_pocketed_s} still closed "
                        f"({base._fmt_pct(n_still_closed / n_pocketed_s)})"
                    )
        else:
            pocket_lines.append("No earlier tasks with pockets to check yet.")

        pocket_summary = "\n".join(pocket_lines)

        print(f"\n--- Task {t}: pocket recovery summary ---\n{pocket_summary}")
        _tlog(f"Task {t}: pocket recovery summary\n{pocket_summary}")

        _tlog(f"Task {t}: writing pipeline_log.txt")
        base.write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Unlearning step", unlearn_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
            ("Pocket recovery summary (debug)", pocket_summary),
        ])

        if not args.no_breakpoint:
            print(f"\n[breakpoint] Task {t}: pocket recovery summary above -- inspect `succ_pocket`, "
                  f"`fix_still_evading`, `X_test_adv`, `y_test`, `lineages`. Continue with `c`.")
            breakpoint()

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()), "detector_metrics": det_metrics,
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            # Feature rows intentionally NOT persisted (they dominate checkpoint file size) --
            # only each entry's (label, category, sample_id) survives; see _buffer_ids_only.
            "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
            "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
            "joint_label_buffers": _buffer_ids_only(joint_label_buffers),
            "joint_replay_buffer": _replay_ids_only(joint_replay_buffer),
            # task_test_splits (X_test_raw, y_test per task) intentionally NOT persisted, same
            # reason -- task_test_gids (just the row IDs) still is.
            "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
            "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path_for(t))
        _tlog(f"Task {t}: checkpoint saved")

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")
        _tlog(f"=== Task {t}: done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())}) ===")

        if t == base.NUM_TASKS - 1:
            _tlog(f"Task {t}: rendering final PCA correctness-grid plot")
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled)
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels)
            _tlog(f"Task {t}: plot saved")

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`baseline_label_buffers`, `joint_label_buffers`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    _tlog(f"Run done. Total runtime: {time.perf_counter() - start_time:.1f}s")
    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
