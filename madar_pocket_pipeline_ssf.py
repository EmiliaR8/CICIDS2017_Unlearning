"""
madar_pocket_pipeline_ssf.py

Same poisoning, test-time attack, and pocket-targeting criterion as
madar_pocket_pipeline.py -- imported directly from it below, not
reimplemented, so the two files cannot drift apart. Adds SSF (Strategic
Selection and Forgetting; Zhang et al., INFOCOM 2025), ported from
xinchen930/SSF-Strategic-Selection-and-Forgetting (ssf.py + utils.py, the
UNSW-NB15 code path -- the one with a real binary classifier head; the
NSL-KDD path classifies via a two-Gaussian fit over cosine similarities
and has no head at all).

THREE lineages:

  clean             -- never poisoned; reference. Own replay buffer filled
                       from its own clean data each task (same simplified
                       design as madar_pocket_pipeline_si_agem.py's clean).
  poisoned_baseline -- poisoned every task, never fixed. IDENTICAL to
                       madar_pocket_pipeline.py's poisoned_baseline.
  ssf               -- poisoned every task; adapts with SSF's own model,
                       loss, memory and selection/forgetting machinery (see
                       below). Does NOT start from the shared task-0
                       ClassifierNN -- it has its own task-0 pretraining of
                       SSF's AE_classifier on the SAME task-0 training data.

WHAT IS FAITHFUL TO THE REFERENCE (per design discussion, deliberately):

  * Model: AE_classifier (encoder -> decoder -> sigmoid head on the
    reconstruction; layer widths derived from the nearest power of 2 of the
    input dim), trained with SSF's loss -- InfoNCE-style contrastive loss on
    the reconstruction (temperature 0.02) + BCE on the sigmoid output.
  * Label budget: SSF is an ACTIVE-LEARNING method. Each round, only
    --ssf_num_labeled rows of the incoming chunk get their TRUE label
    (default 200, the reference's UNSW setting); every other chunk row that
    enters memory does so with a PSEUDO-label (the model's own prediction).
    ssf therefore never sees most of each task's labels, unlike every other
    lineage in this project -- that is SSF's premise, not an oversight.
  * Per round: KS-test drift detection (model outputs on memory vs. on the
    new chunk, p < 0.05), the two KL-histogram mask optimizations (M_c over
    memory, M_t over the chunk; SGD lr 24 / 50, 100 steps, 10 bins, inits
    0.5-1 / 0-0.5 -- the UNSW settings), strategic selection of which chunk
    rows to label, strategic forgetting of memory rows (random
    non-representative removal without drift; remove ALL non-representative
    rows and refill memory with pseudo-labeled chunk rows under drift),
    new-sample loss weighting (x60), and LwF distillation (MSE vs. the
    previous round's model, lambda 0.5) on no-drift rounds only.
  * Memory = the training set. SSF trains each round on its memory ONLY
    (never memory + the whole chunk) -- the chunk only reaches training
    through the rows selection admits into memory.

WHAT DIFFERS FROM THE REFERENCE, AND WHY:

  * Memory size (--ssf_buffer_percent). The reference's memory is
    x_train.shape[0] * (1 - percent) = 20% of the WHOLE training set, and
    that same 20% subset is its initial training data. Here the percentage
    is a CLI flag, and "the whole training set" is the union of every
    task's training split (tasks 0..NUM_TASKS-1, post train/test split), so
    0.2 reproduces the reference's sizing literally. Memory is initialized
    as a random subset of task 0's training split of that size (task 0 is
    this project's analogue of the reference's initial labeled data). If
    the requested size exceeds task 0's training split, it is capped there
    and a warning is logged.
  * Pretraining data. The reference pretrains ONLY on its initial 20%
    subset (the memory). Here pretraining uses task 0's FULL training
    split -- the same data every other lineage pretrains on -- so that
    sweeping --ssf_buffer_percent changes memory size ONLY, not the amount
    of pretraining data as well (which would confound a buffer-size sweep).
  * Optimizer/epochs: Adam, project convention (task 0: base.TASK0_LR /
    base.TASK0_EPOCHS; adaptation: base.ADAPT_LR / base.ADAPT_WEIGHT_DECAY /
    --ssf_epochs, default base.ADAPT_EPOCHS), not the reference's SGD lr
    0.001 with 200 / 180 epochs -- same rationale as the SI/A-GEM/DEDUCE
    ports: the question is whether SSF's mechanism helps within this
    project's training setup.
  * Stream: each task's poisoned training split is the incoming stream,
    split into --ssf_rounds_per_task chunks (default 1: one SSF round per
    task, so evaluation lines up with every other lineage's per-task
    checkpoints). The reference instead streams a random 80% of its
    training set plus its whole test set; here test sets stay held out and
    are scored exactly like every other lineage's (clean + genuine-pocket
    adversarial + spillover), replacing the reference's own stream-scoring.
  * Feature space: ssf consumes the same task-0 StandardScaler space
    (clipped to +/-FEATURE_CLIP) as every other lineage, not a per-split
    MinMaxScaler, so it can be scored on the shared adversarial test points
    (crafted in that space). SSF's histograms/KS test operate on model
    OUTPUTS (sigmoid, in [0,1]), so input scaling does not touch them.

REFERENCE BUGS FIXED IN THE PORT (none change SSF's intended behavior):

  * detect_drift skips any window shorter than sample_interval, so a chunk
    smaller than that could never register drift. Here the KS test always
    runs over the whole chunk.
  * InfoNCE anchors only on benign rows; a batch with none returns an empty
    loss whose mean is NaN. This project's tasks run up to ~97% malicious,
    so such batches happen -- the contrastive term is treated as 0 there.
  * optimize_old_mask takes log(0) whenever memory has no rows in an
    output-histogram bin the new chunk does occupy (common: memory outputs
    pile up near 0 and 1), which turns every mask entry NaN -- and a NaN
    M_c makes a drift round forget the ENTIRE memory. Its bins are now
    clamped to >= 1e-10, exactly as the reference's own optimize_new_mask
    already clamps its bins.
  * Selected-row bookkeeping: the reference records positions WITHIN the
    representative subset rather than within the chunk (only affected its
    own final "exclude labeled rows" scoring, which isn't used here). Rows
    are tracked by chunk index here, which is also what makes the
    poison-vs-label-source diagnostics below possible.

LOGGING/CHECKPOINTS match madar_pocket_pipeline_meta_detect.py: a
timestamped step-by-step timing log (logs/meta_log.txt, also printed),
classification reports on each task's ADVERSARIAL test set alongside the
clean one, a plain-language "Pocket recovery summary (debug)" section every
task (printed + logged, then a breakpoint() unless --no_breakpoint), and one
checkpoint per task (logs/classifier_checkpoint_task<t>.pt) with replay
buffers/ssf memory stored as ids/labels only and test splits as row ids
only. --resume_from is NOT ported: meta_detect resumes with EMPTY buffers,
and ssf's memory IS its training set, so an empty-memory resume would not be
the same method.

DIAGNOSTICS SPECIFIC TO THIS FILE (logged under "Continual-learning
baselines step"): drift decision + KS p-value per round, how many chunk rows
were true-labeled vs. pseudo-labeled, how many of each were oracle-poisoned,
pseudo-label accuracy overall and on poisoned rows (poisoned rows are
shifted toward the other class, so a wrong pseudo-label on them is
effectively a label flip), and memory composition by category x label
source after the task.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import madar_pocket_pipeline as base

LINEAGE_NAMES = ["clean", "poisoned_baseline", "ssf"]
BASELINE_NAMES = ["ssf"]  # the one under test; gets the full metrics table

# Reference defaults (ssf.py module-level constants + its UNSW README command).
SSF_TEMPERATURE = 0.02
SSF_BATCH_SIZE = 128
SSF_DRIFT_THRESHOLD = 0.05
SSF_LWF_LAMBDA = 0.5
SSF_OLD_INIT = "0.5-1"
SSF_NEW_INIT = "0-0.5"
SSF_MASK_BINS = 10
SSF_MASK_STEPS = 100


def _buffer_ids_only(label_buffers):
    """Drops each replay-buffer entry's feature row (index 0), keeping only
    (label, category, sample_id) -- used ONLY when writing checkpoints, since
    MEM_SIZE-scale feature rows dominate checkpoint file size. Same helper as
    madar_pocket_pipeline_meta_detect.py's."""
    return {lbl: [tuple(e[1:]) for e in entries] for lbl, entries in label_buffers.items()}


def _replay_ids_only(replay_buffer):
    """Same as _buffer_ids_only, for the flattened list form."""
    return [tuple(e[1:]) for e in replay_buffer]


# ---------------------------------------------------------------------------
# Timestamped step-by-step timing log ("meta_log.txt"), separate from the
# human-readable pipeline_log.txt -- same mechanism and file name as
# madar_pocket_pipeline_meta_detect.py's, for diagnosing which step of a real
# run is actually slow. main() points these at <out_dir>/logs/meta_log.txt
# before the task loop starts.
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
# SSF model + loss -- ported from the reference utils.py.
# ---------------------------------------------------------------------------
class AE_classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))
        second_fourth_layer_size = nearest_power_of_2 // 2
        third_layer_size = nearest_power_of_2 // 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify


class InfoNCELoss(nn.Module):
    """Returns the per-pair (n_benign x n_benign) loss matrix, like the
    reference -- callers weight and .mean() it themselves."""

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        logits_without_ii = logits * logits_mask

        normal = (labels == 0).squeeze(1)
        abnormal = (labels > 0).squeeze(1)
        logits_normal = logits_without_ii[normal]
        logits_normal_normal = logits_normal[:, normal]
        logits_normal_abnormal = logits_normal[:, abnormal]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), dim=1, keepdim=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        loss = -(logits_normal_normal - torch.log(denominator))
        if self.scale_by_temperature:
            loss = loss * self.temperature
        return loss


class SSFLineage:
    """Wraps AE_classifier + its persistent optimizer, exposing the same
    predict_proba()/predict()/score()/.model interface AdaptableClassifier
    gives every other lineage, so base.pooled_and_per_task_accuracy /
    base.plot_correctness_grid / the still-evades computation score it
    unchanged. predict_proba returns 2 columns [P(benign), P(malicious)]
    from the single sigmoid output; predict thresholds at 0.5 exactly like
    the reference's evaluate_classifier."""

    def __init__(self, torch_model, lr, weight_decay):
        self.model = torch_model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def malicious_proba(self, X, batch_size=base.EMBED_BATCH_SIZE):
        self.model.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        parts = [self.model(Xt[i:i + batch_size])[2].squeeze(1) for i in range(0, len(Xt), batch_size)]
        return torch.cat(parts).numpy() if parts else np.empty((0,), dtype=np.float32)

    def predict_proba(self, X):
        p = self.malicious_proba(X)
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X):
        return (self.malicious_proba(X) > 0.5).astype(np.int64)

    def score(self, X, y):
        return (self.predict(X) == np.asarray(y)).mean()


def ssf_loss(model, contrastive, xb, yb, new_mask_b, new_sample_weight, teacher=None, lwf_lambda=0.0):
    """Reference training loss (UNSW path): new-sample-weighted contrastive +
    BCE, plus LwF MSE against the teacher's sigmoid output when a teacher is
    passed (no-drift rounds only)."""
    _, recon_vec, classifications = model(xb)
    classifications = classifications.squeeze(1)

    normal_new_mask = new_mask_b[yb == 0]
    if len(normal_new_mask) > 0:
        con_loss = contrastive(recon_vec, yb)
        # Broadcasts the per-benign-row weight across the loss matrix's
        # columns -- kept exactly as the reference writes it.
        weighted_con = (con_loss * ((1 - normal_new_mask) + normal_new_mask * new_sample_weight)).mean()
    else:
        # Reference bug fix: no benign anchor in the batch -> empty loss
        # matrix -> NaN mean. Treat the contrastive term as 0 instead.
        weighted_con = torch.zeros(())

    cls_loss = F.binary_cross_entropy(classifications, yb.float(), reduction="none")
    weighted_cls = (cls_loss * ((1 - new_mask_b) + new_mask_b * new_sample_weight)).mean()
    loss = weighted_con + weighted_cls

    if teacher is not None:
        with torch.no_grad():
            teacher_out = teacher(xb)[2].squeeze(1)
        loss = loss + lwf_lambda * F.mse_loss(classifications, teacher_out)
    return loss


def ssf_train(lineage, X, y, new_mask, epochs, batch_size, contrastive, new_sample_weight,
              teacher=None, lwf_lambda=0.0):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    mt = torch.as_tensor(np.asarray(new_mask), dtype=torch.float32)
    n = len(Xt)
    if teacher is not None:
        teacher.eval()
    lineage.model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            lineage.opt.zero_grad()
            loss = ssf_loss(lineage.model, contrastive, Xt[idx], yt[idx], mt[idx], new_sample_weight,
                            teacher=teacher, lwf_lambda=lwf_lambda)
            loss.backward()
            lineage.opt.step()
    lineage.model.eval()


# ---------------------------------------------------------------------------
# SSF drift detection + mask optimization -- ported from the reference.
# ---------------------------------------------------------------------------
def detect_drift(new_outputs, memory_outputs, drift_threshold):
    """Two-sample KS test over the whole chunk (see module docstring for the
    reference's window-length bug this sidesteps). Returns (drift, p_value)."""
    _, p_value = ks_2samp(memory_outputs, new_outputs)
    return bool(p_value < drift_threshold), float(p_value)


def _initialize_mask(size, initialization):
    if initialization == "0-1":
        init = torch.rand(size)
    elif initialization == "0-0.5":
        init = torch.rand(size) * 0.5
    elif initialization == "0.5-1":
        init = torch.rand(size) * 0.5 + 0.5
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")
    return torch.nn.Parameter(init, requires_grad=True)


def optimize_old_mask(control_res, treatment_res, initialization, lr,
                      num_bins=SSF_MASK_BINS, steps=SSF_MASK_STEPS):
    """M_c over memory rows: reweight memory so its output histogram matches
    the new chunk's (KL over num_bins bins on [0, 1])."""
    control_res = torch.as_tensor(control_res, dtype=torch.float32)
    treatment_res = torch.as_tensor(treatment_res, dtype=torch.float32)
    M_c = _initialize_mask(control_res.size(0), initialization)
    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4
    bin_edges = torch.linspace(0., 1., num_bins + 1)
    treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)
    bin_masks_c = [((control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])).float()
                   for i in range(num_bins)]

    for _ in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)
        optimizer.zero_grad()
        bin_obs_c = torch.stack([torch.sum(M_c * m) / torch.sum(M_c) for m in bin_masks_c])
        bin_tgt_c = treatment_hist / len(treatment_res)
        # Reference bug fix: a bin with no memory rows gives log(0) = -inf and
        # a NaN gradient that turns EVERY M_c entry NaN. Clamped exactly the
        # way the reference's own optimize_new_mask already clamps its bins.
        bin_obs_c = torch.clamp(bin_obs_c / bin_obs_c.sum(), min=1e-10)
        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()
        loss = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction="sum")
        loss.backward()
        optimizer.step()
    return M_c.detach()


def optimize_new_mask(control_res, treatment_res, M_c, initialization, lr,
                      num_bins=SSF_MASK_BINS, steps=SSF_MASK_STEPS):
    """M_t over chunk rows: pick the chunk rows whose addition makes the
    combined (weighted memory + weighted chunk) histogram match the chunk's."""
    control_res = torch.as_tensor(control_res, dtype=torch.float32)
    treatment_res = torch.as_tensor(treatment_res, dtype=torch.float32)
    M_c = M_c.detach()
    M_t = _initialize_mask(treatment_res.size(0), initialization)
    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4
    bin_edges = torch.linspace(0., 1., num_bins + 1)
    treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)
    bin_masks_c = [((control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])).float()
                   for i in range(num_bins)]
    bin_masks_t = [((treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])).float()
                   for i in range(num_bins)]

    for _ in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)
        optimizer.zero_grad()
        bin_tgt_t = treatment_hist / len(treatment_res)
        total = torch.sum(M_t) + torch.sum(M_c)
        bin_combined = torch.stack([
            (torch.sum(M_t * mt) + torch.sum(M_c * mc)) / total
            for mt, mc in zip(bin_masks_t, bin_masks_c)
        ])
        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()
        loss = F.kl_div(bin_combined.log(), bin_tgt_t, reduction="sum")
        loss.backward()
        optimizer.step()
    return M_t.detach()


# ---------------------------------------------------------------------------
# SSF strategic selection + forgetting -- the reference's
# select_and_update_representative_samples(_when_drift), re-expressed over
# row INDICES (memory index / chunk index) instead of feature tensors, so
# each admitted row's global id / oracle category / label source can be
# carried along for diagnostics. Selection logic is unchanged.
# ---------------------------------------------------------------------------
def ssf_select(M_c, M_t, num_labeled, drift, memory_size, rng):
    """Returns (keep_mem_idx, labeled_chunk_idx, pseudo_chunk_idx).

    keep_mem_idx      -- memory rows that survive this round's forgetting.
    labeled_chunk_idx -- chunk rows admitted with their TRUE label (the
                         round's label budget).
    pseudo_chunk_idx  -- chunk rows admitted with a PSEUDO label (drift
                         rounds only, to refill memory up to memory_size).
    """
    M_c_np = M_c.numpy()
    M_t_np = M_t.numpy()
    n_mem, n_chunk = len(M_c_np), len(M_t_np)
    num_labeled = min(num_labeled, n_chunk)

    # --- Forgetting (old memory) ---
    # Non-representative = complement of representative, as in the
    # reference (so a NaN mask entry counts as non-representative).
    rep_mask_old = M_c_np >= 0.5
    rep_old = np.where(rep_mask_old)[0]
    non_rep_old = np.where(~rep_mask_old)[0]
    if len(non_rep_old) < num_labeled:
        # Remove every non-representative row, then the lowest-M_c
        # representative rows to make up the difference (both variants).
        extra = num_labeled - len(non_rep_old)
        extra_idx = rep_old[np.argsort(M_c_np[rep_old], kind="stable")[:extra]]
        remove = np.concatenate([non_rep_old, extra_idx])
    elif drift:
        # Drift: forget ALL non-representative memory rows.
        remove = non_rep_old
    else:
        # No drift: forget num_labeled random non-representative rows.
        remove = rng.choice(non_rep_old, size=num_labeled, replace=False)
    keep = np.ones(n_mem, dtype=bool)
    keep[remove] = False
    keep_mem_idx = np.where(keep)[0]

    # --- Selection (new chunk rows to label) ---
    rep_new = np.where(M_t_np >= 0.5)[0]
    rep_new_sorted = rep_new[np.argsort(-M_t_np[rep_new], kind="stable")]
    if len(rep_new) < num_labeled:
        available = np.setdiff1d(np.arange(n_chunk), rep_new)
        fallback = rng.permutation(available)[:num_labeled - len(rep_new)]
        labeled_chunk_idx = np.concatenate([rep_new, fallback])
    else:
        labeled_chunk_idx = rep_new_sorted[:num_labeled]

    # --- Drift-only memory refill with pseudo-labeled chunk rows ---
    pseudo_chunk_idx = np.array([], dtype=np.int64)
    if drift:
        n_after = len(keep_mem_idx) + len(labeled_chunk_idx)
        needed = memory_size - n_after
        if needed > 0:
            if len(rep_new) > num_labeled:
                remaining = rep_new_sorted[num_labeled:]
                if len(remaining) >= needed:
                    pseudo_chunk_idx = remaining[:needed]
                else:
                    # Reference tops up with random rows drawn from the WHOLE
                    # chunk (may repeat rows already admitted) -- kept as-is.
                    extra = rng.permutation(n_chunk)[:needed - len(remaining)]
                    pseudo_chunk_idx = np.concatenate([remaining, extra])
            else:
                pseudo_chunk_idx = rng.permutation(n_chunk)[:needed]
    return keep_mem_idx, labeled_chunk_idx.astype(np.int64), pseudo_chunk_idx.astype(np.int64)


class SSFMemory:
    """SSF's memory (= its training set), plus per-row metadata used only
    for logging: global sample id, oracle category, and label source
    ('true' or 'pseudo'). new_mask marks rows admitted as newly labeled THIS
    round (the reference's new_sample_mask, which up-weights them in the
    loss); it's reset every round, like the reference's."""

    def __init__(self, X, y, gid, category):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.gid = np.asarray(gid, dtype=np.int64)
        self.category = np.asarray(category, dtype=object)
        self.label_source = np.full(len(self.y), "true", dtype=object)
        self.new_mask = np.zeros(len(self.y), dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def update(self, keep_idx, add_X, add_y, add_gid, add_cat, add_source, add_new_mask):
        self.X = np.concatenate([self.X[keep_idx], add_X]).astype(np.float32)
        self.y = np.concatenate([self.y[keep_idx], add_y]).astype(np.int64)
        self.gid = np.concatenate([self.gid[keep_idx], add_gid]).astype(np.int64)
        self.category = np.concatenate([self.category[keep_idx], add_cat])
        self.label_source = np.concatenate([self.label_source[keep_idx], add_source])
        self.new_mask = np.concatenate([np.zeros(len(keep_idx), dtype=np.float32), add_new_mask])

    def composition(self):
        dist = {}
        for cat, src in zip(self.category, self.label_source):
            key = f"{cat}/{src}"
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items()))


def ssf_round(lineage, memory, X_chunk, y_chunk, gid_chunk, cat_chunk, args, contrastive, teacher,
              memory_size, rng):
    """One SSF round on one chunk: drift check -> masks -> select/forget ->
    retrain on memory. Returns this round's diagnostics dict."""
    _tlog(f"  [ssf_round] start, memory={len(memory)} rows, chunk={len(y_chunk)} rows")
    mem_out = lineage.malicious_proba(memory.X)
    chunk_out = lineage.malicious_proba(X_chunk)
    drift, p_value = detect_drift(chunk_out, mem_out, SSF_DRIFT_THRESHOLD)
    _tlog(f"  [ssf_round] drift check done (drift={drift}, KS p={p_value:.3g})")

    M_c = optimize_old_mask(mem_out, chunk_out, SSF_OLD_INIT, args.ssf_opt_old_lr)
    _tlog(f"  [ssf_round] memory mask M_c optimized ({int((M_c.numpy() >= 0.5).sum())}/{len(M_c)} "
          f"representative)")
    M_t = optimize_new_mask(mem_out, chunk_out, M_c, SSF_NEW_INIT, args.ssf_opt_new_lr)
    _tlog(f"  [ssf_round] chunk mask M_t optimized ({int((M_t.numpy() >= 0.5).sum())}/{len(M_t)} "
          f"representative)")

    keep_idx, labeled_idx, pseudo_idx = ssf_select(
        M_c, M_t, args.ssf_num_labeled, drift, memory_size, rng)
    _tlog(f"  [ssf_round] selection done (kept {len(keep_idx)}/{len(memory)} memory rows, "
          f"labeled {len(labeled_idx)}, pseudo-labeled {len(pseudo_idx)})")

    # Pseudo-labels come from the model as it stands BEFORE this round's
    # training, exactly as in the reference.
    pseudo_y = (chunk_out[pseudo_idx] > 0.5).astype(np.int64) if len(pseudo_idx) else \
        np.array([], dtype=np.int64)

    n_mem_before = len(memory)
    removed = np.ones(n_mem_before, dtype=bool)
    removed[keep_idx] = False
    n_forgotten_poisoned = int(np.char.endswith(memory.category[removed].astype(str), "_perturbed").sum())
    add_idx = np.concatenate([labeled_idx, pseudo_idx])
    memory.update(
        keep_idx,
        X_chunk[add_idx],
        np.concatenate([y_chunk[labeled_idx], pseudo_y]),
        gid_chunk[add_idx],
        cat_chunk[add_idx],
        np.array(["true"] * len(labeled_idx) + ["pseudo"] * len(pseudo_idx), dtype=object),
        np.concatenate([np.ones(len(labeled_idx), dtype=np.float32),
                        np.zeros(len(pseudo_idx), dtype=np.float32)]),
    )

    ssf_train(lineage, memory.X, memory.y, memory.new_mask, epochs=args.ssf_epochs,
              batch_size=SSF_BATCH_SIZE, contrastive=contrastive,
              new_sample_weight=args.ssf_new_sample_weight,
              teacher=None if drift else teacher, lwf_lambda=SSF_LWF_LAMBDA)
    _tlog(f"  [ssf_round] trained {args.ssf_epochs} epoch(s) on memory ({len(memory)} rows, "
          f"LwF {'off -- drift' if drift else 'on'})")

    is_poisoned = np.char.endswith(cat_chunk.astype(str), "_perturbed")
    pseudo_correct = (pseudo_y == y_chunk[pseudo_idx]) if len(pseudo_idx) else np.array([], dtype=bool)
    pseudo_pois = is_poisoned[pseudo_idx] if len(pseudo_idx) else np.array([], dtype=bool)
    return {
        "chunk_size": int(len(y_chunk)),
        "drift": drift,
        "ks_p_value": p_value,
        "n_mem_representative": int((M_c.numpy() >= 0.5).sum()),
        "n_chunk_representative": int((M_t.numpy() >= 0.5).sum()),
        "memory_before": n_mem_before,
        "n_forgotten": int(n_mem_before - len(keep_idx)),
        "n_forgotten_poisoned": n_forgotten_poisoned,
        "n_labeled": int(len(labeled_idx)),
        "n_labeled_poisoned": int(is_poisoned[labeled_idx].sum()),
        "n_pseudo": int(len(pseudo_idx)),
        "n_pseudo_poisoned": int(pseudo_pois.sum()),
        "pseudo_acc": float(pseudo_correct.mean()) if len(pseudo_correct) else float("nan"),
        "pseudo_acc_poisoned": float(pseudo_correct[pseudo_pois].mean()) if pseudo_pois.any() else float("nan"),
        "memory_after": int(len(memory)),
    }


def ssf_pretrain(feature_dim, X, y, contrastive, epochs, batch_size, lr):
    """Task-0 pretraining of SSF's own AE_classifier with SSF's own loss
    (contrastive + BCE, no new-sample weighting, no teacher) on the same
    task-0 training data every other lineage pretrains on."""
    model = AE_classifier(feature_dim).to(base.DEVICE)
    opt0 = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    zeros = torch.zeros(len(yt))
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt0.zero_grad()
            loss = ssf_loss(model, contrastive, Xt[idx], yt[idx], zeros[idx], new_sample_weight=1.0)
            loss.backward()
            opt0.step()
    model.eval()
    return model


def _fmt_rounds(rounds):
    lines = [f"{'round':<6} {'chunk':>7} {'drift':>6} {'KS p':>10} {'mem before':>11} {'forgot':>7} "
             f"{'(poisoned)':>11} {'labeled':>8} {'(poisoned)':>11} {'pseudo':>7} {'(poisoned)':>11} "
             f"{'pseudo acc':>11} {'pseudo acc on poisoned':>23} {'mem after':>10}"]
    for r_i, r in enumerate(rounds):
        lines.append(
            f"{r_i:<6} {r['chunk_size']:>7} {str(r['drift']):>6} {r['ks_p_value']:>10.3g} "
            f"{r['memory_before']:>11} {r['n_forgotten']:>7} {r['n_forgotten_poisoned']:>11} "
            f"{r['n_labeled']:>8} {r['n_labeled_poisoned']:>11} {r['n_pseudo']:>7} "
            f"{r['n_pseudo_poisoned']:>11} {r['pseudo_acc']:>11.3f} {r['pseudo_acc_poisoned']:>23.3f} "
            f"{r['memory_after']:>10}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _META_LOG_PATH, _META_LOG_T0
    start_time = time.perf_counter()
    _META_LOG_T0 = start_time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_ssf_run")
    ap.add_argument("--h5-path", type=str, default=base.H5_DATASET_PATH)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--hidden_sizes", type=str,
                     default=",".join(str(h) for h in base.DEFAULT_HIDDEN_SIZES),
                     help="Comma-separated hidden-layer widths for ClassifierNN (clean / "
                          "poisoned_baseline only -- ssf uses SSF's own AE_classifier, whose widths "
                          "are derived from the input dim). Same meaning/default as in "
                          "madar_pocket_pipeline.py.")
    ap.add_argument("--per_feature_epsilon", type=float, default=None,
                     help="Same per-feature (L-infinity) attack cap as madar_pocket_pipeline.py. "
                          "Off by default.")
    ap.add_argument("--ssf_buffer_percent", type=float, default=0.2,
                     help="SSF memory size as a fraction of the WHOLE training pool (every task's "
                          "training split, summed). 0.2 = the reference's memory sizing "
                          "(x_train.shape[0] * (1 - percent), percent=0.8). Memory is initialized "
                          "as a random subset of task 0's training split, capped at its size.")
    ap.add_argument("--ssf_num_labeled", type=int, default=200,
                     help="True labels queried per round (reference UNSW setting: 200).")
    ap.add_argument("--ssf_new_sample_weight", type=float, default=60.0,
                     help="Loss weight on newly labeled rows (reference UNSW setting: 60).")
    ap.add_argument("--ssf_opt_old_lr", type=float, default=24.0,
                     help="SGD lr for the memory mask M_c (reference UNSW setting: 24).")
    ap.add_argument("--ssf_opt_new_lr", type=float, default=50.0,
                     help="SGD lr for the chunk mask M_t (reference UNSW setting: 50).")
    ap.add_argument("--ssf_epochs", type=int, default=base.ADAPT_EPOCHS,
                     help=f"Training epochs over memory per SSF round (project convention, default "
                          f"{base.ADAPT_EPOCHS}; the reference used 180 on UNSW).")
    ap.add_argument("--ssf_rounds_per_task", type=int, default=1,
                     help="Split each task's poisoned training batch into this many SSF rounds "
                          "(chunks). Each round gets its own --ssf_num_labeled label budget.")
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pauses: after every task's pocket "
                          "recovery summary, and at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}. Needed for a non-interactive/headless run.")
    args = ap.parse_args()
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))
    if not 0 < args.ssf_buffer_percent <= 1:
        raise ValueError(f"--ssf_buffer_percent must be in (0, 1], got {args.ssf_buffer_percent}")
    if args.ssf_rounds_per_task < 1:
        raise ValueError(f"--ssf_rounds_per_task must be >= 1, got {args.ssf_rounds_per_task}")

    base.SEED = args.seed  # update_shared_buffer reads this module-level global
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ssf_rng = np.random.default_rng(args.seed)
    poison_fraction = args.poison_fraction
    contrastive = InfoNCELoss(temperature=SSF_TEMPERATURE)

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_ssf", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    meta_log_path = os.path.join(out_dir, "logs", "meta_log.txt")

    def checkpoint_path_for(task_id):
        return os.path.join(out_dir, "logs", f"classifier_checkpoint_task{task_id}.pt")

    _META_LOG_PATH = meta_log_path
    with open(meta_log_path, "w") as f:
        f.write(f"SSF TIMING LOG -- run started {datetime.datetime.now().isoformat()}\n"
                f"args: {vars(args)}\n\n")
    _tlog("Run starting")

    print(f"Loading {args.h5_path} and building {base.NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = base.load_pooled_chronological_tasks(
        args.h5_path, base.TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    # Every task's training split, re-derived with the SAME train_test_split
    # call the loop below makes, so the memory size is sized against the
    # exact rows that make up the training pool.
    splits = []
    for t in range(base.NUM_TASKS):
        y_t = tasks[t]["labels"].astype(np.int64)
        splits.append(train_test_split(
            np.arange(len(y_t)), test_size=base.TASK_TEST_FRAC, random_state=args.seed, stratify=y_t,
        )[0])
    total_train = sum(len(s) for s in splits)
    task0_train = len(splits[0])
    memory_size_requested = int(math.floor(total_train * args.ssf_buffer_percent))
    memory_size = min(memory_size_requested, task0_train)
    memory_note = (
        f"SSF memory: {memory_size} rows = {args.ssf_buffer_percent} x {total_train} total training "
        f"rows ({memory_size / task0_train:.1%} of task 0's {task0_train} training rows)"
    )
    if memory_size < memory_size_requested:
        memory_note += (f"  <-- CAPPED: requested {memory_size_requested} exceeds task 0's training "
                        f"split, so memory starts as ALL of task 0")
    _tlog(memory_note)

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (SSF continual-learning baseline)\n"
            "=====================================================================\n"
            "3 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "ssf (Strategic Selection and Forgetting, own AE_classifier + own task-0\n"
            "pretraining on the same data). Same poisoning/attack/pocket-targeting as\n"
            "madar_pocket_pipeline.py -- see that file for those mechanics. ssf is an\n"
            "active-learning method: it only sees --ssf_num_labeled TRUE labels per\n"
            "round; other admitted chunk rows are pseudo-labeled.\n"
            f"Classifier hidden layer sizes (clean/poisoned_baseline): {hidden_sizes}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
            f"ssf_buffer_percent={args.ssf_buffer_percent}, ssf_num_labeled={args.ssf_num_labeled}, "
            f"ssf_new_sample_weight={args.ssf_new_sample_weight}, ssf_opt_old_lr={args.ssf_opt_old_lr}, "
            f"ssf_opt_new_lr={args.ssf_opt_new_lr}, ssf_epochs={args.ssf_epochs}, "
            f"ssf_rounds_per_task={args.ssf_rounds_per_task}\n"
            f"{memory_note}\n"
        )

    scaler = None
    lineages = {}
    clean_label_buffers, clean_replay_buffer = {}, []
    baseline_label_buffers, baseline_replay_buffer = {}, []
    ssf_memory = None
    ssf_teacher = None
    task_test_splits, task_test_gids = {}, {}
    results = []

    def to_scaled(X_raw):
        return np.clip(scaler.transform(X_raw.astype(np.float32)), -base.FEATURE_CLIP,
                       base.FEATURE_CLIP).astype(np.float32)

    for t in range(base.NUM_TASKS):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        task = tasks[t]
        X_raw = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
        y_all = task["labels"].astype(np.int64)
        gid_all = task_offsets[t] + np.arange(len(y_all), dtype=np.int64)

        _tlog(f"=== Task {t}: start ===")
        X_train_raw, X_test_raw, y_train, y_test, gid_train, gid_test = train_test_split(
            X_raw, y_all, gid_all, test_size=base.TASK_TEST_FRAC, random_state=args.seed,
            stratify=y_all,
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
            _tlog(f"Task 0: pretraining shared ClassifierNN for {base.TASK0_EPOCHS} epochs on {n} rows")
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
            _tlog("Task 0: shared ClassifierNN pretraining done")

            for name in ["clean", "poisoned_baseline"]:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            # ssf: its OWN model, pretrained with its OWN loss on the same data.
            _tlog(f"Task 0: pretraining ssf's AE_classifier for {base.TASK0_EPOCHS} epochs on {n} rows")
            ssf_model = ssf_pretrain(feature_dim, X_train_scaled, y_train, contrastive,
                                     epochs=base.TASK0_EPOCHS, batch_size=SSF_BATCH_SIZE, lr=base.TASK0_LR)
            lineages["ssf"] = SSFLineage(ssf_model, lr=base.ADAPT_LR, weight_decay=base.ADAPT_WEIGHT_DECAY)
            ssf_teacher = copy.deepcopy(ssf_model).eval()
            _tlog("Task 0: ssf pretraining done")

            category = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
            mem_idx = ssf_rng.choice(len(y_train), size=memory_size, replace=False)
            ssf_memory = SSFMemory(X_train_scaled[mem_idx], y_train[mem_idx], gid_train[mem_idx],
                                   category[mem_idx])

            task_acc = {name: lineages[name].score(X_test_scaled, y_test) for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_raw, y_test)
            task_test_gids[0] = gid_test

            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, gid_train,
                benign_label, mal_label,
            )
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, gid_train, benign_label, mal_label,
            )
            _tlog("Task 0: replay buffers + ssf memory filled")

            train_section = (
                f"malicious: {int((y_train == mal_label).sum())}, benign: {int((y_train == benign_label).sum())}\n"
                f"malicious_perturbed: 0, benign_perturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = (
                f"malicious: {int((y_test == mal_label).sum())}, benign: {int((y_test == benign_label).sum())}\n"
                f"genuine pockets: N/A (task 0 -- no poisoning yet)\n"
            )
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            cl_section = (
                "N/A -- task 0 has no prior model to poison against.\n"
                f"{memory_note}\n"
                f"ssf memory composition (initial): {ssf_memory.composition()}"
            )

            base.write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Continual-learning baselines step", cl_section),
            ])

            results.append({"task": t, "task_acc": task_acc, "ssf_memory_size": memory_size})
            torch.save({
                "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
                "label_mapping": label_mapping,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                # Feature rows intentionally NOT persisted (they dominate checkpoint file size) --
                # only each entry's (label, category, sample_id) survives; see _buffer_ids_only.
                "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
                "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
                "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
                "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
                "ssf_memory": {"gid": ssf_memory.gid, "y": ssf_memory.y, "category": ssf_memory.category,
                               "label_source": ssf_memory.label_source},
                # task_test_splits (X_test_raw, y_test per task) intentionally NOT persisted, same
                # reason -- task_test_gids (just the row IDs) still is.
                "task_test_gids": task_test_gids,
                "results": results, "poison_fraction": poison_fraction,
                "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
                "ssf_args": {k: v for k, v in vars(args).items() if k.startswith("ssf_")},
            }, checkpoint_path_for(t))
            _tlog(f"=== Task {t}: done (log + checkpoint written) ===")
            continue

        # -------------------------------------------------------------
        # Tasks 1..NUM_TASKS-1: poison -> attack, every task.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        _tlog(f"Task {t}: step 2 -- adapting clean lineage")
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison, shared by poisoned_baseline/ssf.
        _tlog(f"Task {t}: step 3 -- crafting poison (poison_fraction={poison_fraction})")
        X_train_poisoned, idx_poison_ben, idx_poison_mal, _ = base.craft_task_poison(
            lineages["clean"], X_train_scaled, y_train, benign_label, mal_label, poison_fraction,
        )
        poison_idx = np.concatenate([idx_poison_ben, idx_poison_mal])
        category_all = np.where(y_train == benign_label, "benign", "malicious_clean").astype(object)
        category_all[idx_poison_ben] = "benign_perturbed"
        category_all[idx_poison_mal] = "malicious_perturbed"
        _tlog(f"Task {t}: step 3 done ({len(poison_idx)} poisoned rows)")

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix"),
        # identical to madar_pocket_pipeline.py.
        _tlog(f"Task {t}: step 4 -- adapting poisoned_baseline")
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                            replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                            epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(X_train_poisoned[poison_idx], y_train[poison_idx])
            if len(poison_idx) else float("nan")
        )

        # Step 5: ssf streams this task's poisoned training batch as
        # --ssf_rounds_per_task unlabeled chunks (true labels only reach it
        # through its per-round label budget). Teacher = the model at the
        # end of the previous round, updated after every round (reference).
        _tlog(f"Task {t}: step 5 -- ssf adapting ({args.ssf_rounds_per_task} round(s))")
        chunk_order = ssf_rng.permutation(len(y_train))
        ssf_rounds = []
        for r_i, chunk in enumerate(np.array_split(chunk_order, args.ssf_rounds_per_task)):
            if len(chunk) == 0:
                continue
            _tlog(f"  Task {t}: step 5 -- ssf round {r_i + 1}/{args.ssf_rounds_per_task}")
            ssf_rounds.append(ssf_round(
                lineages["ssf"], ssf_memory, X_train_poisoned[chunk], y_train[chunk], gid_train[chunk],
                category_all[chunk], args, contrastive, ssf_teacher, memory_size, ssf_rng,
            ))
            ssf_teacher = copy.deepcopy(lineages["ssf"].model).eval()
        _tlog(f"Task {t}: step 5 done")

        # Step 6: craft this task's genuine-pocket test attack, ONCE, against
        # poisoned_baseline (reference = clean) -- same points re-scored under
        # every lineage below.
        _tlog(f"Task {t}: step 6 -- crafting this task's adversarial test attack")
        eps_this_task = base.typical_class_gap(X_test_scaled, y_test, benign_label,
                                               mal_label) * base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, per_feature_epsilon=args.per_feature_epsilon,
        )
        _tlog(f"Task {t}: step 6 done (genuine pocket rate {base._fmt_pct(succ_pocket.mean())})")

        # Step 7: spillover check -- re-attack every PRIOR task's test set.
        _tlog(f"Task {t}: step 7 -- spillover re-attack over {len(task_test_splits)} prior task(s)")
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
            _tlog(f"  Task {t}: step 7 -- re-attacked source task {s} ({len(ys)} rows)")
        _tlog(f"Task {t}: step 7 done")

        all_clean_sets = {s: (to_scaled(Xs_raw), ys) for s, (Xs_raw, ys) in task_test_splits.items()}
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pocket_info_by_source = {s: (v[2], v[3]) for s, v in historical_adv.items()}
        pocket_info_by_source[t] = (succ_pocket, eps_this_task)

        _tlog(f"Task {t}: computing pooled/mean/per-class accuracy for all lineages")
        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = base.pooled_and_per_task_accuracy(
                lineages[name], all_test_sets_full)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test)
            _tlog(f"  Task {t}: accuracy -- {name} done")

        still_evades = {}
        for name in BASELINE_NAMES:
            pred = lineages[name].predict(X_test_adv)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 8: update clean's and poisoned_baseline's buffers, AFTER every
        # lineage's adaptation this task is fully done. ssf's memory was
        # already updated inside its own rounds (SSF curates memory BEFORE
        # training on it, by design).
        _tlog(f"Task {t}: step 8 -- updating replay buffers")
        clean_category = np.where(y_train == benign_label, "benign", "malicious_clean")
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, gid_train,
            benign_label, mal_label,
        )
        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers, X_train_poisoned, y_train,
            category_all, gid_train, benign_label, mal_label,
        )

        _tlog(f"Task {t}: step 8 done")
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

        clean_dist = base.buffer_distribution(clean_label_buffers)
        baseline_dist = base.buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"clean's OWN replay buffer distribution (post-update, this task): {clean_dist}")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution (post-update, this task): "
                            f"{baseline_dist}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
            adapt_lines.append(f"[{name}] classification report (this task's adversarial test):")
            adapt_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        adapt_section = "\n".join(adapt_lines)

        cl_lines = [
            f"ssf rounds this task ({args.ssf_rounds_per_task} requested; poisoned = oracle "
            f"*_perturbed rows; pseudo acc = pseudo-label vs. true label):",
            _fmt_rounds(ssf_rounds),
            "",
            f"ssf memory composition (category/label source, post-task): {ssf_memory.composition()}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}",
        ]
        for name in BASELINE_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test)
            adv_acc_name = lineages[name].score(X_test_adv, y_test)
            cl_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        cl_lines.append("")
        for name in BASELINE_NAMES:
            cl_lines.append(f"[{name}] classification report (this task's clean test):")
            cl_lines.append(per_class_reports[name])
            cl_lines.append(f"[{name}] classification report (this task's adversarial test):")
            cl_lines.append(base._fmt_report(lineages[name], X_test_adv, y_test))
        cl_section = "\n".join(cl_lines)

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
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's adv-test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][f"{s}_adversarial"]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(f"Task {t}'s (post-adaptation) classifier accuracy on each source task's CLEAN test-set:")
        breakdown_lines.append(f"{'source task':<12} " + "".join(f"{name:>18}" for name in LINEAGE_NAMES))
        for s in sorted(pocket_info_by_source.keys()):
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                acc_s, _n = per_task_by_lineage[name][s]
                row += f"{acc_s:>18.3f}"
            breakdown_lines.append(row)

        breakdown_lines.append("")
        breakdown_lines.append(
            f"Task {t}'s (post-adaptation) classifier COMBINED (clean+adversarial, pooled) "
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
        # Debug: pocket send/recovery summary -- same block as
        # madar_pocket_pipeline_meta_detect.py's, for ssf (the lineage under
        # test) instead of the fix variants: how many test points were sent
        # into a genuine pocket this task, how many of those ssf does NOT
        # fall for, how healthy ssf looks on ordinary (non-attacked) traffic
        # right now, and whether this task's adaptation reopened any EARLIER
        # task's pockets. Always printed+logged; followed by an interactive
        # breakpoint() unless --no_breakpoint. Written in plain language on
        # purpose -- this is meant to be read live, task by task.
        # ---------------------------------------------------------------
        n_pocketed = int(succ_pocket.sum())
        pocket_lines = [
            f"Sent into pockets (genuine pockets found, this task's test set): "
            f"{n_pocketed}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})",
            f"  benign side: {succ_ben}/{n_ben_test}, malicious side: {succ_mal}/{n_mal_test}",
            "",
            f"Recovered after adaptation (of the {n_pocketed} pocketed points, no longer evading):",
        ]
        lineage_still_evading = {}
        for name in BASELINE_NAMES:
            pred_name = lineages[name].predict(X_test_adv)
            still_evading_mask = (pred_name != y_test) & succ_pocket
            lineage_still_evading[name] = still_evading_mask
            n_recovered = n_pocketed - int(still_evading_mask.sum())
            pct_recovered = base._fmt_pct(n_recovered / n_pocketed) if n_pocketed else "N/A"
            pocket_lines.append(f"  {name:<18}: {n_recovered}/{n_pocketed} recovered ({pct_recovered})")
        pocket_lines.append("")
        pocket_lines.append("How well each lineage under test reads NORMAL (non-attacked) traffic right now:")
        for name in BASELINE_NAMES:
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
                "Checking back on earlier tasks' pockets (did adapting to THIS task "
                "accidentally reopen any of them?):"
            )
            for s, (Xs_adv, ys, succ_s, eps_s) in pockets_to_check:
                n_pocketed_s = int(succ_s.sum())
                pocket_lines.append(f"  Task {s}'s pockets ({n_pocketed_s} total):")
                for name in BASELINE_NAMES:
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
            ("Continual-learning baselines step", cl_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
            ("Pocket recovery summary (debug)", pocket_summary),
        ])

        if not args.no_breakpoint:
            print(f"\n[breakpoint] Task {t}: pocket recovery summary above -- inspect `succ_pocket`, "
                  f"`lineage_still_evading`, `X_test_adv`, `y_test`, `lineages`. Continue with `c`.")
            breakpoint()

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()),
            "ssf_rounds": ssf_rounds, "ssf_memory_composition": ssf_memory.composition(),
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "feature_dim": feature_dim, "scaler": scaler,
            "label_mapping": label_mapping,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            # Feature rows intentionally NOT persisted (they dominate checkpoint file size) --
            # only each entry's (label, category, sample_id) survives; see _buffer_ids_only.
            "clean_label_buffers": _buffer_ids_only(clean_label_buffers),
            "clean_replay_buffer": _replay_ids_only(clean_replay_buffer),
            "baseline_label_buffers": _buffer_ids_only(baseline_label_buffers),
            "baseline_replay_buffer": _replay_ids_only(baseline_replay_buffer),
            "ssf_memory": {"gid": ssf_memory.gid, "y": ssf_memory.y, "category": ssf_memory.category,
                           "label_source": ssf_memory.label_source},
            # task_test_splits (X_test_raw, y_test per task) intentionally NOT persisted, same
            # reason -- task_test_gids (just the row IDs) still is.
            "task_test_gids": task_test_gids,
            "results": results, "poison_fraction": poison_fraction,
            "hidden_sizes": hidden_sizes, "per_feature_epsilon": args.per_feature_epsilon,
            "ssf_args": {k: v for k, v in vars(args).items() if k.startswith("ssf_")},
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
                  f"`ssf_memory`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    _tlog(f"Run done. Total runtime: {time.perf_counter() - start_time:.1f}s")
    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
