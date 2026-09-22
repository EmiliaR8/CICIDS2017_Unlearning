"""
madar_pocket_pipeline_deduce_tinyimagenet.py

Image/class-incremental port of madar_pocket_pipeline_deduce.py -- same
relationship as madar_pocket_pipeline_tinyimagenet.py is to
madar_pocket_pipeline.py: same poisoning, test-time attack, and
pocket-targeting criterion, imported directly from
madar_pocket_pipeline_tinyimagenet below rather than reimplemented. Adds a
third continual-learning baseline: DEDUCE (Wang, Benavides-Prado & Koh, ICLR
2026), same standalone (no STAR, no X-DER) scope as the network-flow port --
see that file's module docstring for why.

THREE lineages, all starting from the same task-0 model (see
madar_pocket_pipeline_tinyimagenet.py for task 0's plain-pretraining
details):

  clean             -- never poisoned; reference. Own replay buffer filled
                       from its own clean data each task.
  poisoned_baseline -- poisoned every task, never fixed. IDENTICAL to
                       madar_pocket_pipeline_tinyimagenet.py's
                       poisoned_baseline.
  deduce            -- poisoned every task; adapts with DEDUCE's detect /
                       decide / unlearn wrapper around plain cross-entropy.

DiagonalFisher / LocalUnlearning / gradient_conflict / fisher_update /
deduce_adapt are ported VERBATIM from the network-flow file -- all operate
purely on model.named_parameters()/flat gradients, with no assumption about
architecture, so they work unchanged on ClassifierCNN.

GUM (Eq. 12 / Algorithm 2) is the one piece that is NOT architecture-neutral:
the reference's neuron-reinit machinery is defined for a stack of
Linear+BatchNorm1d MLP blocks (this project's ClassifierNN: fc1..fc4, each
paired with a BatchNorm1d, each block's "units" being that Linear's output
features). ClassifierCNN's hidden representation is genuinely conv-structured
for 3 of its 4 blocks -- a conv output CHANNEL is not the same kind of "unit"
a Linear layer's output feature is (channels have spatial extent; a channel's
outgoing connections are a whole (out_ch, kH, kW) slice of the next conv
layer's weight, not one column), and the reference paper's Eq. (12) says
nothing about how to score or reinitialize a convolutional channel.

CHOICE (GUM scope, this port): rather than invent a conv-channel reinit
scheme the paper does not specify, GUM here is scoped to the ONE genuinely
Linear+activation hidden block ClassifierCNN actually has: latent_fc (which
bridges the flattened conv trunk to fc_last). The conv trunk itself is left
out of GUM's reinit scope entirely -- exactly the same kind of choice the
network-flow DEDUCE port already made for STAR/X-DER ("rather than build
inert class-incremental machinery... layered directly onto this pipeline's
plain cross-entropy adaptation instead").

Mechanically this means _gum_blocks(model) returns a single block instead of
one per hidden layer, and that block has no paired BatchNorm1d (latent_fc's
only downstream op is a plain nn.ReLU) -- GlobalUnlearning.attach/step/
_reinit are adjusted to tolerate bn=None (skip the BN-specific reset lines)
and to hook model.relu directly for activation recording, rather than a
BatchNorm module. Hooking model.relu is safe here even though the reference
avoids hooking ReLU for ClassifierNN (whose single nn.ReLU instance is reused
by every block, so a hook on it cannot tell blocks apart): ClassifierCNN's
model.relu is used in exactly ONE place in the whole forward pass (right
after latent_fc), so there is no such ambiguity for this one block.

Fisher/LUM/detection all still cover the WHOLE model (every parameter,
conv trunk included) via model.named_parameters() -- only GUM's neuron-level
reinit is scoped down.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import madar_pocket_pipeline_tinyimagenet as base
from madar_pocket_pipeline_si_agem_tinyimagenet import ReservoirBuffer
from tinyimagenet_data_loader import DEFAULT_TASK_SETUP, build_schedule, load_tinyimagenet_tasks

LINEAGE_NAMES = ["clean", "poisoned_baseline", "deduce"]
BASELINE_NAMES = ["deduce"]  # the one under test; gets the full metrics table


# ---------------------------------------------------------------------------
# DEDUCE -- ported from madar_pocket_pipeline_deduce.py. DiagonalFisher,
# LocalUnlearning, gradient_conflict, fisher_update, deduce_adapt are
# UNCHANGED (architecture-agnostic). GlobalUnlearning/_gum_blocks are
# adapted for ClassifierCNN -- see module docstring's CHOICE.
# ---------------------------------------------------------------------------
class DiagonalFisher:
    """Diagonal empirical FIM over the trainable parameters. Operates purely
    on model.named_parameters(), no architecture assumption."""

    def __init__(self, model):
        self.F = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.P = {}
        self.n_batches = 0
        self.raw_mean = 0.0

    def is_empty(self):
        return self.n_batches == 0

    def estimate(self, model, batches, device, max_batches=32):
        new = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        used = 0
        was = model.training
        model.eval()
        for xb, yb in batches:
            if used >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(xb.to(device))
            F.nll_loss(F.log_softmax(logits, dim=1), yb.to(device)).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new[n] += p.grad.detach() ** 2
            used += 1
        model.zero_grad(set_to_none=True)
        model.train(was)
        if not used:
            return {"fim_batches": 0, "fim_raw_mean": 0.0}
        for n in new:
            new[n] /= used
        count = sum(v.numel() for v in new.values())
        self.raw_mean = sum(float(v.sum()) for v in new.values()) / max(1, count)
        scale = self.raw_mean if self.raw_mean > 0 else 1.0
        self.F = {n: v / scale for n, v in new.items()}
        self.P = {}
        self.n_batches = used
        return {"fim_batches": used, "fim_raw_mean": self.raw_mean}

    def preconditioner(self, damping):
        if self.P.get("_key") == damping:
            return self.P
        raw = {n: 1.0 / (v + damping) for n, v in self.F.items()}
        count = sum(v.numel() for v in raw.values())
        mean = sum(float(v.sum()) for v in raw.values()) / max(1, count)
        self.P = {n: v / mean for n, v in raw.items()}
        self.P["_key"] = damping
        return self.P

    def quadratic(self, model, anchor):
        total = None
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in anchor:
                continue
            term = (self.F[n] * (p - anchor[n]) ** 2).sum()
            total = term if total is None else total + term
        return total if total is not None else torch.zeros((), device=next(model.parameters()).device)

    def neuron_importance(self, weight_name, model):
        """F_{l,i} = sum over the outgoing connections of neuron i, where
        `weight_name` names the NEXT layer's weight. Unchanged from the
        network-flow version: fc_last.weight is shape (num_classes,
        latent_dim) in BOTH ClassifierNN and ClassifierCNN."""
        f = self.F[weight_name]
        return f.sum(dim=0)


def _flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters() if p.requires_grad])


def gradient_conflict(model, cur, mem, epsilon=0.0):
    ce = nn.CrossEntropyLoss()

    def grad_of(x, y):
        model.zero_grad(set_to_none=True)
        ce(model(x), y).backward()
        return _flat_grad(model).clone()

    g_mem = grad_of(*mem)
    g_cur = grad_of(*cur)
    model.zero_grad(set_to_none=True)

    denom = float(g_mem.norm() * g_cur.norm())
    if denom <= 0:
        return {"conflict": 0.0, "cosine": 0.0}
    cos = float(torch.dot(g_mem, g_cur)) / denom
    return {"conflict": float(cos <= epsilon), "cosine": cos}


class LocalUnlearning:
    """Eq. (9)-(10): one FIM-scaled ascent step on the current batch's CE."""

    def __init__(self, delta, alpha, fim_damping, grad_clip):
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.damping = float(fim_damping)
        self.clip = float(grad_clip)

    @torch.no_grad()
    def _apply(self, model, fisher, anchor):
        precond = fisher.preconditioner(self.damping)
        steps, moved = {}, 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            prox = 2.0 * (p.detach() - anchor[n]) if anchor else torch.zeros_like(p)
            step = self.delta * precond[n] * (self.alpha * prox - p.grad.detach())
            steps[n] = step
            moved += float(step.norm() ** 2)
        moved = math.sqrt(moved)
        cap = self.clip * self.delta
        scale = min(1.0, cap / moved) if moved > 0 else 1.0
        for n, p in model.named_parameters():
            if n in steps:
                p.add_(steps[n] * scale)
        return moved * scale

    def step(self, model, xb, yb, *, fisher, anchor):
        model.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(xb), yb).backward()
        moved = self._apply(model, fisher, anchor)
        model.zero_grad(set_to_none=True)
        return {"lum_fired": 1.0, "lum_step_norm": moved}


def _gum_blocks(model):
    """ONE block: latent_fc (see module docstring's GUM CHOICE). bn=None
    (latent_fc has no paired BatchNorm1d); hook_target=model.relu is what
    attach() hooks for activations instead."""
    return [(0, model.latent_fc, None, model.fc_last, "fc_last.weight", model.relu)]


class GlobalUnlearning:
    """Eq. (12) plus Algorithm 2: reinitialize low-contribution mature
    neurons. Adapted from the network-flow version for ClassifierCNN --
    operates on the single latent_fc block _gum_blocks returns; tolerates
    bn=None throughout (skips the BatchNorm reset lines for that block)."""

    def __init__(self, model, eta, phi, maturity, seed=0):
        self.eta = float(eta)
        self.phi = float(phi)
        self.maturity = int(maturity)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 5501)
        self.blocks = _gum_blocks(model)
        self.C, self.age, self.credit = {}, {}, {}
        for i, linear, _, _, _, _ in self.blocks:
            n_units = linear.out_features
            self.C[i] = torch.zeros(n_units)
            self.age[i] = torch.zeros(n_units)
            self.credit[i] = 0.0
        self._h = {}
        self._handles = []

    def attach(self, model):
        self.detach()
        self.blocks = _gum_blocks(model)
        for i, _, bn, _, _, hook_target in self.blocks:
            def hook(_m, _inp, out, key=i):
                if key not in self._h:
                    self._h[key] = out.detach().relu().abs().mean(dim=0).cpu()
            self._handles.append(hook_target.register_forward_hook(hook))

    def arm(self):
        self._h = {}

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def step(self, model, fisher):
        if not self._h:
            return {}
        resets = 0
        for i, linear, bn, nxt, nxt_name, _hook in self.blocks:
            h = self._h.get(i)
            if h is None:
                continue
            outgoing = nxt.weight.abs().sum(dim=0).detach().cpu()
            imp = fisher.neuron_importance(nxt_name, model).detach().cpu()
            span = float(imp.max() - imp.min())
            norm = (imp - imp.min()) / span if span > 0 else torch.zeros_like(imp)
            self.C[i] = ((1 - self.eta) * h * outgoing + self.eta * self.C[i]) * torch.sigmoid(norm)
            self.age[i] += 1

            eligible = self.age[i] > self.maturity
            n_eligible = int(eligible.sum())
            if not n_eligible:
                continue
            self.credit[i] += self.phi * n_eligible
            while self.credit[i] >= 1.0:
                scores = self.C[i].clone()
                scores[~eligible] = float("inf")
                r = int(torch.argmin(scores))
                if not bool(eligible[r]):
                    break
                self._reinit(linear, bn, nxt, r)
                self.age[i][r] = 0
                self.C[i][r] = 0.0
                eligible[r] = False
                self.credit[i] -= 1.0
                resets += 1
        return {"gum_resets": float(resets)}

    @torch.no_grad()
    def _reinit(self, linear, bn, nxt, r):
        bound = 1.0 / math.sqrt(linear.in_features)
        shape = tuple(linear.weight.shape[1:])
        fresh = (torch.rand(shape, generator=self.gen) * 2 - 1) * bound
        linear.weight[r].copy_(fresh.to(linear.weight.dtype))
        if linear.bias is not None:
            linear.bias[r] = float((torch.rand(1, generator=self.gen) * 2 - 1) * bound)
        if bn is not None:
            bn.weight[r] = 1.0
            bn.bias[r] = 0.0
            if bn.running_mean is not None:
                bn.running_mean[r] = 0.0
                bn.running_var[r] = 1.0
        nxt.weight[:, r] = 0.0  # zero every outgoing connection this unit has


def _batches(Xt, yt, batch_size):
    return [(Xt[i:i + batch_size], yt[i:i + batch_size]) for i in range(0, len(Xt), batch_size)]


def fisher_update(fisher, model, memory, X_task, y_task, device, batch_size, max_batches):
    Xt = torch.as_tensor(np.asarray(X_task), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y_task), dtype=torch.long)
    batches = _batches(Xt, yt, batch_size)
    if not memory.is_empty():
        Xm = torch.as_tensor(np.stack(memory.X), dtype=torch.float32)
        ym = torch.as_tensor(np.array(memory.y), dtype=torch.long)
        batches += _batches(Xm, ym, batch_size)
    return fisher.estimate(model, batches, device, max_batches=max_batches)


def deduce_adapt(lineage, memory, fisher, lum, gum, prev_task_params, X, y, epochs, batch_size,
                  detect_every, epsilon, k, beta):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    n = len(Xt)
    ce = nn.CrossEntropyLoss()
    model = lineage.model
    model.train()
    gum.attach(model)
    anchor = {}
    step = 0
    conflict = 0.0
    n_detected = n_lum_fired = 0
    gum_resets = 0.0
    try:
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                xb, yb = Xt[idx], yt[idx]
                gum.arm()

                if step % detect_every == 0 and not memory.is_empty():
                    mx, my = memory.sample(batch_size)
                    if mx is not None:
                        out = gradient_conflict(model, (xb, yb), (mx, my), epsilon)
                        conflict = out["conflict"]
                        n_detected += 1

                if conflict and not fisher.is_empty():
                    lum.step(model, xb, yb, fisher=fisher, anchor=anchor)
                    n_lum_fired += 1

                step += 1
                if step == k:
                    anchor = {n_: p.detach().clone() for n_, p in model.named_parameters()
                              if p.requires_grad}

                lineage.opt.zero_grad()
                loss = ce(model(xb), yb)
                if beta > 0 and prev_task_params and not fisher.is_empty():
                    loss = loss + beta * fisher.quadratic(model, prev_task_params)
                loss.backward()
                lineage.opt.step()

                gum_info = gum.step(model, fisher)
                gum_resets += gum_info.get("gum_resets", 0.0)
    finally:
        gum.detach()
    model.eval()
    return {"n_steps": step, "n_detections": n_detected, "n_lum_fired": n_lum_fired,
            "lum_fire_rate": n_lum_fired / max(1, step), "gum_resets": gum_resets}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="madar_pocket_deduce_tinyimagenet_run")
    ap.add_argument("--data-root", type=str, default=base.DATA_ROOT)
    ap.add_argument("--cache-root", type=str, default=None)
    ap.add_argument("--task_setup", type=str, default=DEFAULT_TASK_SETUP)
    ap.add_argument("--poison_fraction", type=float, default=base.POISON_FRACTION)
    ap.add_argument("--cnn_channels", type=str, default=",".join(str(c) for c in base.DEFAULT_CNN_CHANNELS))
    ap.add_argument("--latent_dim", type=int, default=base.DEFAULT_LATENT_DIM)
    ap.add_argument("--per_feature_epsilon", type=float, default=None)
    ap.add_argument("--deduce_mem_size", type=int, default=base.MEM_SIZE)
    ap.add_argument("--deduce_detect_every", type=int, default=1)
    ap.add_argument("--deduce_epsilon", type=float, default=0.0)
    ap.add_argument("--deduce_delta", type=float, default=0.001)
    ap.add_argument("--deduce_alpha", type=float, default=1.0)
    ap.add_argument("--deduce_k", type=int, default=10)
    ap.add_argument("--deduce_fim_damping", type=float, default=1e-3)
    ap.add_argument("--deduce_fim_batches", type=int, default=32)
    ap.add_argument("--deduce_beta", type=float, default=0.1)
    ap.add_argument("--deduce_eta", type=float, default=0.99)
    ap.add_argument("--deduce_phi", type=float, default=1e-5)
    ap.add_argument("--deduce_maturity", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no_breakpoint", action="store_true",
                     help="Disable the interactive breakpoint() pause at the end of tasks "
                          f">= {base.BREAKPOINT_FROM_TASK}.")
    args = ap.parse_args()
    cnn_channels = tuple(int(c) for c in args.cnn_channels.split(","))
    schedule = build_schedule(args.task_setup)

    base.SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    poison_fraction = args.poison_fraction

    out_dir = os.path.join(base.RUNS_BASE_DIR, "madar_pocket_deduce_tinyimagenet", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(out_dir, "logs", "pipeline_log.txt")
    checkpoint_path = os.path.join(out_dir, "logs", "classifier_checkpoint.pt")

    with open(log_path, "w") as f:
        f.write(
            "MADAR POCKET-PIPELINE LOG (TinyImageNet, DEDUCE continual-learning baseline)\n"
            "=============================================================================\n"
            "3 lineages per task: clean (reference), poisoned_baseline (no fix),\n"
            "deduce (detect / LUM / GUM). Same poisoning/attack/pocket-targeting as\n"
            "madar_pocket_pipeline_tinyimagenet.py. DEDUCE is standalone here (no STAR,\n"
            "no X-DER); GUM is scoped to latent_fc only -- see module docstring.\n"
            f"Task schedule: {schedule.spec} ({schedule.n_tasks} tasks, {schedule.n_classes} classes)\n"
            f"CNN channels: {cnn_channels}, latent_dim: {args.latent_dim}\n"
            f"Per-feature epsilon cap: {args.per_feature_epsilon}\n"
            f"deduce_mem_size={args.deduce_mem_size}, detect_every={args.deduce_detect_every}, "
            f"epsilon={args.deduce_epsilon}\n"
            f"delta={args.deduce_delta}, alpha={args.deduce_alpha}, k={args.deduce_k}, "
            f"beta={args.deduce_beta}\n"
            f"fim_damping={args.deduce_fim_damping}, fim_batches={args.deduce_fim_batches}\n"
            f"eta={args.deduce_eta}, phi={args.deduce_phi}, maturity={args.deduce_maturity}, "
            f"grad_clip={args.grad_clip}\n"
        )

    print(f"Loading TinyImageNet from {args.data_root} under schedule {schedule.spec}...")
    tasks, wnids = load_tinyimagenet_tasks(args.data_root, schedule, cache_root=args.cache_root)
    print(f"n_tasks={schedule.n_tasks}, n_classes={schedule.n_classes}, "
          f"task sizes(train)={[len(t['y_train']) for t in tasks]}")

    scaler_mean, scaler_std = None, None
    lineages = {}
    clean_label_buffers, clean_replay_buffer = {}, []
    baseline_label_buffers, baseline_replay_buffer = {}, []
    deduce_memory = None
    fisher = None
    lum = None
    gum = None
    prev_task_params = {}
    task_test_splits = {}
    class_centroids = {}
    results = []

    def to_scaled(X_raw_uint8):
        X = X_raw_uint8.astype(np.float32) / 255.0
        return ((X - scaler_mean[None, :, None, None]) / scaler_std[None, :, None, None]).astype(np.float32)

    for t in range(schedule.n_tasks):
        print(f"\n{'#' * 60}\n# TASK {t}\n{'#' * 60}")
        task = tasks[t]
        new_classes = schedule.classes_for(t)
        n_active = schedule.active_count(t)
        X_train_raw, y_train = task["X_train"], task["y_train"]
        X_test_raw, y_test = task["X_test"], task["y_test"]

        # -------------------------------------------------------------
        # Task 0: plain supervised pretraining only, no poisoning yet.
        # -------------------------------------------------------------
        if t == 0:
            X_f = X_train_raw.astype(np.float32) / 255.0
            scaler_mean = X_f.mean(axis=(0, 2, 3))
            scaler_std = X_f.std(axis=(0, 2, 3))
            scaler_std[scaler_std <= 0] = 1.0
            X_train_scaled = to_scaled(X_train_raw)
            X_test_scaled = to_scaled(X_test_raw)

            base_model = base.ClassifierCNN(schedule.n_classes, channels=cnn_channels,
                                             latent_dim=args.latent_dim).to(base.DEVICE)
            Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
            yt = torch.tensor(y_train, dtype=torch.long)
            opt0 = torch.optim.Adam(base_model.parameters(), lr=base.TASK0_LR)
            loss_fn0 = nn.CrossEntropyLoss()
            base_model.train()
            n = len(Xt)
            for _ in range(base.TASK0_EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, base.TASK0_BATCH_SIZE):
                    idx = perm[i:i + base.TASK0_BATCH_SIZE]
                    if len(idx) < 2:
                        continue
                    opt0.zero_grad()
                    loss = loss_fn0(base_model(Xt[idx]), yt[idx])
                    loss.backward()
                    opt0.step()
            base_model.eval()

            for name in LINEAGE_NAMES:
                lineages[name] = base.AdaptableClassifier(copy.deepcopy(base_model))

            deduce_memory = ReservoirBuffer(mem_size=args.deduce_mem_size, seed=args.seed)
            fisher = DiagonalFisher(lineages["deduce"].model)
            lum = LocalUnlearning(args.deduce_delta, args.deduce_alpha, args.deduce_fim_damping,
                                   args.grad_clip)
            gum = GlobalUnlearning(lineages["deduce"].model, args.deduce_eta, args.deduce_phi,
                                    args.deduce_maturity, seed=args.seed)

            task_acc = {name: lineages[name].score(X_test_scaled, y_test, n_active=n_active)
                        for name in LINEAGE_NAMES}

            task_test_splits[0] = (X_test_scaled, y_test)
            base.update_class_centroids(class_centroids, X_train_scaled, y_train, new_classes)

            sample_id = np.arange(len(y_train))
            category = np.array(["clean"] * len(y_train), dtype=object)
            clean_replay_buffer = base.update_shared_buffer(
                lineages["clean"], clean_label_buffers, X_train_scaled, y_train, category, sample_id,
                new_classes,
            )
            baseline_replay_buffer = base.update_shared_buffer(
                lineages["poisoned_baseline"], baseline_label_buffers, X_train_scaled, y_train,
                category, sample_id, new_classes,
            )
            deduce_memory.add_stream(X_train_scaled, y_train)
            fisher_info = fisher_update(fisher, lineages["deduce"].model, deduce_memory,
                                         X_train_scaled, y_train, base.DEVICE,
                                         base.ADAPT_BATCH_SIZE, args.deduce_fim_batches)
            prev_task_params = {n: p.detach().clone()
                                 for n, p in lineages["deduce"].model.named_parameters()
                                 if p.requires_grad}

            train_section = (
                f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
                f"n_train: {len(y_train)}\nperturbed: 0 (task 0 -- no poisoning yet)\n"
            )
            test_section = f"n_test: {len(y_test)}\ngenuine pockets: N/A (task 0 -- no poisoning yet)\n"
            adapt_section = "\n".join(f"{name}: task test acc = {task_acc[name]:.3f}" for name in LINEAGE_NAMES)
            cl_section = (
                "N/A -- task 0 has no prior model to poison against.\n"
                f"deduce Fisher initialized: {fisher_info}"
            )

            base.write_task_log(log_path, t, [
                ("Training Data information", train_section),
                ("Testing Data information", test_section),
                ("Adaptation step", adapt_section),
                ("Continual-learning baselines step", cl_section),
            ])

            results.append({"task": t, "task_acc": task_acc})
            torch.save({
                "task_id": t, "seed": args.seed, "schedule": schedule.as_dict(), "wnids": wnids,
                "scaler_mean": scaler_mean, "scaler_std": scaler_std,
                "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
                "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
                "baseline_label_buffers": baseline_label_buffers,
                "baseline_replay_buffer": baseline_replay_buffer,
                "deduce_memory": {"X": deduce_memory.X, "y": deduce_memory.y, "n_seen": deduce_memory.n_seen},
                "deduce_fisher": {"F": fisher.F, "raw_mean": fisher.raw_mean},
                "deduce_gum": {"C": gum.C, "age": gum.age, "credit": gum.credit},
                "deduce_prev_task_params": prev_task_params,
                "task_test_splits": task_test_splits, "class_centroids": class_centroids,
                "results": results, "poison_fraction": poison_fraction,
                "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
                "per_feature_epsilon": args.per_feature_epsilon,
            }, checkpoint_path)
            continue

        # -------------------------------------------------------------
        # Tasks 1..N-1: poison -> attack, every task's own new classes.
        # -------------------------------------------------------------
        X_train_scaled = to_scaled(X_train_raw)
        X_test_scaled = to_scaled(X_test_raw)
        clean_replay_X, clean_replay_y = base.flatten_replay(clean_replay_buffer)
        baseline_replay_X, baseline_replay_y = base.flatten_replay(baseline_replay_buffer)

        # Step 2: clean lineage adapts on clean data + its own clean buffer.
        lineages["clean"].adapt(X_train_scaled, y_train, replay_X=clean_replay_X,
                                 replay_y=clean_replay_y, epochs=base.CLEAN_ADAPT_EPOCHS)

        # Step 3: craft this task's poison, shared by poisoned_baseline/deduce.
        X_train_poisoned, poison_idx, poison_target = base.craft_task_poison_images(
            lineages["clean"], X_train_scaled, y_train, new_classes, class_centroids,
            poison_fraction, n_active,
        )

        # Step 4: poisoned_baseline adapts on poisoned data ("no fix"),
        # identical to madar_pocket_pipeline_tinyimagenet.py.
        lineages["poisoned_baseline"].adapt(X_train_poisoned, y_train,
                                             replay_X=baseline_replay_X, replay_y=baseline_replay_y,
                                             epochs=base.ADAPT_EPOCHS)
        acc_on_forced_labels = (
            lineages["poisoned_baseline"].score(
                X_train_poisoned[poison_idx],
                np.array([poison_target[int(i)] for i in poison_idx]), n_active=n_active,
            ) if len(poison_idx) else float("nan")
        )

        # Step 5: deduce adapts on poisoned data via detect/LUM/GUM.
        deduce_info = deduce_adapt(
            lineages["deduce"], deduce_memory, fisher, lum, gum, prev_task_params,
            X_train_poisoned, y_train, epochs=base.ADAPT_EPOCHS, batch_size=base.ADAPT_BATCH_SIZE,
            detect_every=args.deduce_detect_every, epsilon=args.deduce_epsilon, k=args.deduce_k,
            beta=args.deduce_beta,
        )

        # Step 6: craft this task's genuine-pocket test attack.
        eps_this_task = base.typical_class_gap_multiclass(class_centroids, new_classes)
        if eps_this_task is None:
            eps_this_task = 1.0
        eps_this_task *= base.ATTACK_EPS_MULTIPLIER
        X_test_adv, succ_pocket, norms_pocket = base.adversarial_attack_pocket(
            lineages["poisoned_baseline"], lineages["clean"], X_test_scaled, y_test,
            epsilon_max=eps_this_task, n_active=n_active, per_feature_epsilon=args.per_feature_epsilon,
        )

        # Step 7: spillover check -- re-attack every PRIOR task's test set.
        historical_adv = {}
        for s, (Xs_scaled, ys) in task_test_splits.items():
            eps_s = base.typical_class_gap_multiclass(class_centroids, schedule.classes_for(s))
            eps_s = (eps_s * base.ATTACK_EPS_MULTIPLIER) if eps_s is not None else eps_this_task
            Xs_adv, succ_s, norms_s = base.adversarial_attack_pocket(
                lineages["poisoned_baseline"], lineages["clean"], Xs_scaled, ys,
                epsilon_max=eps_s, n_active=n_active, per_feature_epsilon=args.per_feature_epsilon,
            )
            historical_adv[s] = (Xs_adv, ys, succ_s, eps_s)

        all_clean_sets = dict(task_test_splits)
        all_clean_sets[t] = (X_test_scaled, y_test)
        all_test_sets_full = dict(all_clean_sets)
        for s, (Xs_adv, ys, succ_s, eps_s) in historical_adv.items():
            all_test_sets_full[f"{s}_adversarial"] = (Xs_adv, ys)
        all_test_sets_full[f"{t}_adversarial"] = (X_test_adv, y_test)

        pocket_info_by_source = {s: (v[2], v[3]) for s, v in historical_adv.items()}
        pocket_info_by_source[t] = (succ_pocket, eps_this_task)

        pooled_results, mean_results, per_class_reports, per_task_by_lineage = {}, {}, {}, {}
        for name in LINEAGE_NAMES:
            pooled_acc, mean_acc, per_task = base.pooled_and_per_task_accuracy(
                lineages[name], all_test_sets_full, n_active)
            pooled_results[name] = pooled_acc
            mean_results[name] = mean_acc
            per_task_by_lineage[name] = per_task
            per_class_reports[name] = base._fmt_report(lineages[name], X_test_scaled, y_test, n_active)

        still_evades = {}
        for name in BASELINE_NAMES:
            pred = lineages[name].predict(X_test_adv, n_active=n_active)
            wrong = (pred != y_test)
            still_evades[name] = float(wrong[succ_pocket].mean()) if succ_pocket.any() else float("nan")

        # Step 8: update clean's and poisoned_baseline's buffers, and
        # deduce's own memory + Fisher, AFTER every lineage's adaptation.
        category_all = np.array(["clean"] * len(y_train), dtype=object)
        category_all[poison_idx] = "perturbed"
        clean_category = np.array(["clean"] * len(y_train), dtype=object)
        sample_id = np.arange(len(y_train))
        clean_replay_buffer = base.update_shared_buffer(
            lineages["clean"], clean_label_buffers, X_train_scaled, y_train, clean_category, sample_id,
            new_classes,
        )
        baseline_replay_buffer = base.update_shared_buffer(
            lineages["poisoned_baseline"], baseline_label_buffers, X_train_poisoned, y_train,
            category_all, sample_id, new_classes,
        )
        deduce_mem_update = deduce_memory.add_stream(X_train_poisoned, y_train)
        fisher_info = fisher_update(fisher, lineages["deduce"].model, deduce_memory,
                                     X_train_poisoned, y_train, base.DEVICE,
                                     base.ADAPT_BATCH_SIZE, args.deduce_fim_batches)
        prev_task_params = {n: p.detach().clone()
                             for n, p in lineages["deduce"].model.named_parameters()
                             if p.requires_grad}

        task_test_splits[t] = (X_test_scaled, y_test)

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        train_section = (
            f"classes introduced: {len(new_classes)} ({new_classes[0]}..{new_classes[-1]}), "
            f"n_train: {len(y_train)}\n"
            f"perturbed (oracle): {len(poison_idx)}\n"
            f"poison_fraction used: {poison_fraction}\n"
            f"poisoned_baseline accuracy on poisoned points' forced (target-class) labels: "
            f"{acc_on_forced_labels:.3f}"
            f"{'  <-- LOW, poisoning may not have taken hold' if acc_on_forced_labels < 0.7 else ''}\n"
        )

        test_section = (
            f"n_test: {len(y_test)}\n"
            f"genuine pockets found: {int(succ_pocket.sum())}/{len(y_test)} ({base._fmt_pct(succ_pocket.mean())})\n"
            f"mean perturbation norm among successes: "
            f"{norms_pocket[succ_pocket].mean() if succ_pocket.any() else float('nan'):.4f}\n"
            f"epsilon used this task: {eps_this_task:.4f}\n"
        )

        clean_dist = base.buffer_distribution(clean_label_buffers)
        baseline_dist = base.buffer_distribution(baseline_label_buffers)
        adapt_lines = [f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10}"]
        for name in ["clean", "poisoned_baseline"]:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adapt_lines.append(f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} "
                                f"{mean_results[name]:>10.3f}")
        adapt_lines.append("")
        adapt_lines.append(f"clean's OWN replay buffer distribution (post-update, this task, "
                            f"{len(clean_dist)} classes buffered): total={sum(v['total'] for v in clean_dist.values())}")
        adapt_lines.append(f"poisoned_baseline's OWN replay buffer distribution (post-update, this task, "
                            f"{len(baseline_dist)} classes buffered): "
                            f"total={sum(v['total'] for v in baseline_dist.values())}")
        adapt_lines.append("")
        for name in ["clean", "poisoned_baseline"]:
            adapt_lines.append(f"[{name}] classification report summary (this task's clean test):")
            adapt_lines.append(per_class_reports[name])
        adapt_section = "\n".join(adapt_lines)

        cl_lines = [
            f"deduce detections: {deduce_info['n_detections']}/{deduce_info['n_steps']} steps checked; "
            f"LUM fired {deduce_info['n_lum_fired']}/{deduce_info['n_steps']} "
            f"({deduce_info['lum_fire_rate'] * 100:.1f}%) -- near 0% means the detector never found "
            f"interference and this row is plain fine-tuning with idle machinery",
            f"deduce GUM resets this task: {deduce_info['gum_resets']:.0f}",
            f"deduce memory (post-update, this task): size={deduce_mem_update['size']} "
            f"n_seen={deduce_mem_update['n_seen']} admitted={deduce_mem_update['admitted']} "
            f"replaced={deduce_mem_update['replaced']}",
            f"deduce Fisher re-estimated on {fisher_info['fim_batches']} batches, "
            f"raw_mean={fisher_info['fim_raw_mean']:.4g}",
            "",
            f"{'lineage':<18} {'task acc':>10} {'pooled acc':>12} {'mean acc':>10} "
            f"{'adv acc':>10} {'still-evades %':>16}",
        ]
        for name in BASELINE_NAMES:
            task_acc_name = lineages[name].score(X_test_scaled, y_test, n_active=n_active)
            adv_acc_name = lineages[name].score(X_test_adv, y_test, n_active=n_active)
            cl_lines.append(
                f"{name:<18} {task_acc_name:>10.3f} {pooled_results[name]:>12.3f} {mean_results[name]:>10.3f} "
                f"{adv_acc_name:>10.3f} {still_evades[name] * 100:>15.1f}%"
            )
        cl_lines.append("")
        for name in BASELINE_NAMES:
            cl_lines.append(f"[{name}] classification report summary (this task's clean test):")
            cl_lines.append(per_class_reports[name])
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
            X_comb = np.concatenate([Xs_clean, Xs_adv], axis=0)
            y_comb = np.concatenate([ys_clean, ys_adv])
            row = f"{s:<12} "
            for name in LINEAGE_NAMES:
                row += f"{lineages[name].score(X_comb, y_comb, n_active=n_active):>18.3f}"
            breakdown_lines.append(row)
        breakdown_section = "\n".join(breakdown_lines)

        base.write_task_log(log_path, t, [
            ("Training Data information", train_section),
            ("Testing Data information", test_section),
            ("Adaptation step", adapt_section),
            ("Continual-learning baselines step", cl_section),
            ("Adversarial test-set breakdown (per source task)", breakdown_section),
        ])

        spillover_summary = {s: float(v[2].mean()) for s, v in historical_adv.items()}
        results.append({
            "task": t, "pooled_acc": pooled_results, "mean_acc": mean_results,
            "genuine_pocket_rate": float(succ_pocket.mean()),
            "deduce": deduce_info,
            "spillover_genuine_pocket_rate_by_prior_task": spillover_summary,
        })

        torch.save({
            "task_id": t, "seed": args.seed, "schedule": schedule.as_dict(), "wnids": wnids,
            "scaler_mean": scaler_mean, "scaler_std": scaler_std,
            "lineages": {name: lineages[name].model.state_dict() for name in LINEAGE_NAMES},
            "clean_label_buffers": clean_label_buffers, "clean_replay_buffer": clean_replay_buffer,
            "baseline_label_buffers": baseline_label_buffers,
            "baseline_replay_buffer": baseline_replay_buffer,
            "deduce_memory": {"X": deduce_memory.X, "y": deduce_memory.y, "n_seen": deduce_memory.n_seen},
            "deduce_fisher": {"F": fisher.F, "raw_mean": fisher.raw_mean},
            "deduce_gum": {"C": gum.C, "age": gum.age, "credit": gum.credit},
            "deduce_prev_task_params": prev_task_params,
            "task_test_splits": task_test_splits, "class_centroids": class_centroids,
            "results": results, "poison_fraction": poison_fraction,
            "cnn_channels": cnn_channels, "latent_dim": args.latent_dim,
            "per_feature_epsilon": args.per_feature_epsilon,
        }, checkpoint_path)

        print(f"Task {t} done. Genuine pocket rate: {base._fmt_pct(succ_pocket.mean())}. "
              f"Log written to {log_path}")

        if t == schedule.n_tasks - 1:
            pca_fit = PCA(n_components=2, random_state=args.seed).fit(X_train_scaled.reshape(len(X_train_scaled), -1))
            panels = [(name, lineages[name], X_test_scaled, y_test) for name in LINEAGE_NAMES]
            base.plot_correctness_grid(
                os.path.join(out_dir, "plots", f"task{t}_correctness.png"), pca_fit, panels, n_active)

        if not args.no_breakpoint and t >= base.BREAKPOINT_FROM_TASK:
            print(f"\n[breakpoint] Task {t} finished -- inspect `results`, `lineages`, "
                  f"`fisher`, `gum`, `deduce_memory`, or the log at {log_path}. Continue with `c`.")
            breakpoint()

    print(f"\nDone. Total runtime: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
