"""
pocket_illustration.py

A SELF-CONTAINED, SYNTHETIC illustration of the pocket-poisoning/attack/
unlearning mechanic this project's real pipelines implement -- not derived
from any real run or log. Two classes in a toy 2D feature space, closed-form
decision boundaries (no trained model), so the "pocket" bulge this project's
real classifiers learn under poisoning can be drawn exactly rather than
approximated from a real (noisier) decision surface.

THE MECHANIC BEING ILLUSTRATED (mirrors craft_boundary_pocket_poison /
adversarial_attack_pocket in madar_pocket_pipeline.py):

  1. C1 (pre-adaptation): a clean linear boundary separating two natural
     clusters -- the previous task's classifier.
  2. Training-time poisoning: a handful of near-boundary points from EACH
     class are shifted toward the OPPOSITE class's territory, keeping their
     TRUE label (clean-label poisoning). Adapting to this data pulls the
     boundary into a small local bulge around each shifted point, so it can
     still classify them correctly -- that bulge IS the pocket: a region of
     one class's prediction carved out of the other class's territory.
  3. C2 (post-adaptation): C1's line plus one Gaussian "bump" per poisoned
     training point, each pulling the local decision value toward that
     point's true label.
  4. Test-time attack: FRESH points (never part of training) are perturbed
     into the SAME pocket regions the training poison carved out -- this is
     what a real adversary exploits: the pocket is a property of the trained
     model, not of the specific points that created it. These are what
     "genuine pockets" are in the real pipeline: poisoned-model-wrong,
     clean-model-right.
  5. C3 (post-unlearning): the bumps are attenuated (not necessarily to
     zero), so the boundary retracts most of the way back toward C1 -- most
     adversarial test points fall back on their true label's side, without
     the boundary being pixel-identical to C1 (a small residual bump and a
     small global drift are kept, since that is what the real pipeline's
     unlearning mechanisms actually achieve: reduction, not exact reversal).

Produces 4 PNGs to --out-dir:
  1_pre_adaptation.png       C1 + clean train data only
  2_adaptation.png           C2 + clean+poisoned TRAIN data, pockets circled
  3_pre_unlearning_eval.png  C2 + clean+adversarial TEST data (pockets circled),
                             adversarial points marked correct/WRONG against
                             their true label
  4_post_unlearning.png      C3 + the SAME test data, most now correct

Usage:
    python pocket_illustration.py --out-dir illustrations/ --seed 0
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

BENIGN, MALICIOUS = 0, 1
COLOR = {BENIGN: "#2a78d6", MALICIOUS: "#c94141"}
NAME = {BENIGN: "Benign", MALICIOUS: "Malicious"}
XLIM, YLIM = (-6, 6), (-4, 4)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(seed=0, n_train=120, n_test=40, n_poison=8):
    rng = np.random.default_rng(seed)
    centers = {BENIGN: np.array([-2.5, 0.0]), MALICIOUS: np.array([2.5, 0.0])}

    def draw(label, n):
        return rng.normal(loc=centers[label], scale=[1.1, 1.0], size=(n, 2))

    X_train = {c: draw(c, n_train) for c in (BENIGN, MALICIOUS)}
    X_test_clean = {c: draw(c, n_test) for c in (BENIGN, MALICIOUS)}

    # C1: a clean linear boundary, decision(x,y) = x (>0 -> malicious side).
    c1 = dict(w=np.array([1.0, 0.0]), b=0.0)

    def decision_c1(X):
        return X @ c1["w"] + c1["b"]

    # --- craft_boundary_pocket_poison equivalent: near-boundary points of
    # each class, shifted toward the OPPOSITE class's centroid, true label
    # kept. Separately for train (creates the pocket) and test (exploits it).
    separating_direction = centers[MALICIOUS] - centers[BENIGN]

    def craft_poison(X, label, n, shift_frac, jitter, rng):
        margin = np.abs(decision_c1(X))
        idx = np.argsort(margin)[:n]
        direction = separating_direction if label == BENIGN else -separating_direction
        shifted = X[idx] + shift_frac * direction + rng.normal(scale=jitter, size=(n, 2))
        return shifted, idx

    X_poison_train, poison_train_idx = {}, {}
    for c in (BENIGN, MALICIOUS):
        X_poison_train[c], poison_train_idx[c] = craft_poison(
            X_train[c], c, n_poison, shift_frac=0.85, jitter=0.15, rng=rng)

    # Fresh test-time perturbations exploiting the pockets: EACH point stays
    # on ITS OWN native (correct-per-C1) side -- a benign point never
    # crosses to x>0 -- and is nudged toward the NEAREST bulge that lives on
    # that same side. That bulge is the OPPOSITE class's poison (a benign
    # point's native territory, x<0, is where the MALICIOUS-poison bulge
    # sits, since those points were shifted FROM x>0 INTO x<0). This is what
    # makes it a genuine pocket: the clean model (no bulges) still sees a
    # native, unperturbed-looking point and gets it right; only the
    # poisoned model, whose bulge now occupies that exact spot, gets it
    # wrong. Shifting a point across x=0 entirely (as V1 of this script
    # did) fails that criterion -- the CLEAN model would misclassify it
    # too, so it would not be a pocket, just an ordinary adversarial example.
    X_adv_test = {}
    for c in (BENIGN, MALICIOUS):
        fresh = draw(c, n_poison)
        opposite = MALICIOUS if c == BENIGN else BENIGN
        targets = X_poison_train[opposite]
        d2 = ((fresh[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
        nearest = targets[np.argmin(d2, axis=1)]
        X_adv_test[c] = fresh + 0.9 * (nearest - fresh) + rng.normal(scale=0.1, size=fresh.shape)

    # --- C2: C1 plus one Gaussian bump per TRAIN poison point, pulling the
    # local decision value toward that point's TRUE label. Amplitude is
    # ADAPTIVE per point (not a fixed constant): a point that landed far
    # from the boundary after its shift needs a proportionally bigger bump
    # to actually flip the local sign there, or its "pocket" would be a
    # no-op that still misclassifies its own poisoned training point --
    # not what a real classifier does (it always fits the training point
    # given enough capacity/epochs, which is the whole reason a pocket
    # forms at all).
    target_margin = 2.2
    bumps = []  # (center, sign, amplitude, sigma)
    for c in (BENIGN, MALICIOUS):
        sign = -1.0 if c == BENIGN else 1.0
        for p in X_poison_train[c]:
            base_val = float(decision_c1(p[None, :])[0])
            amp = max(1.0, target_margin - sign * base_val)
            bumps.append((p, sign, amp, 0.55))

    def decision_c2(X):
        out = decision_c1(X)
        for center, sign, amp, sigma in bumps:
            out = out + sign * amp * np.exp(-((X - center) ** 2).sum(axis=1) / (2 * sigma ** 2))
        return out

    # --- C3: bumps attenuated (not zeroed) + a small global drift, so the
    # boundary is close to but not identical to C1.
    c3_global = dict(w=np.array([0.92, 0.12]), b=0.15)

    def decision_c3(X):
        out = X @ c3_global["w"] + c3_global["b"]
        for center, sign, amp, sigma in bumps:
            out = out + sign * (0.12 * amp) * np.exp(-((X - center) ** 2).sum(axis=1) / (2 * sigma ** 2))
        return out

    return dict(
        X_train=X_train, X_test_clean=X_test_clean,
        X_poison_train=X_poison_train, X_adv_test=X_adv_test,
        bumps=bumps, decision_c1=decision_c1, decision_c2=decision_c2, decision_c3=decision_c3,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _shade_and_boundary(ax, decision_fn, resolution=300):
    xx, yy = np.meshgrid(np.linspace(*XLIM, resolution), np.linspace(*YLIM, resolution))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = decision_fn(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9],
               colors=[COLOR[BENIGN], COLOR[MALICIOUS]], alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0], colors="#333333", linewidths=2.2)


def _scatter(ax, X, label, marker="o", size=26, edgecolor="none", lw=0, alpha=1.0, name=None):
    ax.scatter(X[:, 0], X[:, 1], s=size, c=COLOR[label], marker=marker,
              edgecolors=edgecolor, linewidths=lw, alpha=alpha,
              label=name, zorder=3)


def _highlight_pockets(ax, bumps):
    for center, _sign, _amp, sigma in bumps:
        ax.add_patch(Ellipse(center, width=4 * sigma, height=4 * sigma, fill=False,
                             linestyle="--", edgecolor="#555555", linewidth=1.3, zorder=4))


def _finish(ax, title):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")


def plot_pre_adaptation(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _shade_and_boundary(ax, data["decision_c1"])
    for c in (BENIGN, MALICIOUS):
        _scatter(ax, data["X_train"][c], c, name=f"{NAME[c]} (clean train)")
    _finish(ax, "1. Pre-adaptation: previous task's classifier (C1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_adaptation(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _shade_and_boundary(ax, data["decision_c2"])
    for c in (BENIGN, MALICIOUS):
        _scatter(ax, data["X_train"][c], c, alpha=0.5, name=f"{NAME[c]} (clean train)")
        _scatter(ax, data["X_poison_train"][c], c, marker="D", size=70,
                edgecolor="black", lw=1.1, name=f"{NAME[c]} (poisoned train, true label kept)")
    _highlight_pockets(ax, data["bumps"])
    ax.annotate("pocket", xy=data["bumps"][0][0], xytext=(0, 2.6),
               textcoords="data", fontsize=10, color="#333333",
               arrowprops=dict(arrowstyle="->", color="#333333"))
    _finish(ax, "2. Adaptation: boundary shifts to fit poisoned data (C2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_eval(data, decision_fn, out_path, title, show_correctness):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _shade_and_boundary(ax, decision_fn)
    for c in (BENIGN, MALICIOUS):
        _scatter(ax, data["X_train"][c], c, alpha=0.15, size=16, name=None)
    for c in (BENIGN, MALICIOUS):
        _scatter(ax, data["X_test_clean"][c], c, alpha=0.7, name=f"{NAME[c]} (clean test)")
    for c in (BENIGN, MALICIOUS):
        X_adv = data["X_adv_test"][c]
        if show_correctness:
            pred = (decision_fn(X_adv) > 0).astype(int)
            correct = pred == c
            ax.scatter(X_adv[correct, 0], X_adv[correct, 1], s=90, c=COLOR[c], marker="*",
                      edgecolors="black", linewidths=1.0, zorder=5,
                      label=f"{NAME[c]} (adversarial test, correct)")
            ax.scatter(X_adv[~correct, 0], X_adv[~correct, 1], s=90, facecolors="none",
                      edgecolors=COLOR[c], marker="*", linewidths=2.0, zorder=5,
                      label=f"{NAME[c]} (adversarial test, WRONG)")
        else:
            ax.scatter(X_adv[:, 0], X_adv[:, 1], s=90, c=COLOR[c], marker="*",
                      edgecolors="black", linewidths=1.0, zorder=5,
                      label=f"{NAME[c]} (adversarial test)")
    _highlight_pockets(ax, data["bumps"])
    _finish(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="pocket_illustration_plots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-poison", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = generate_data(seed=args.seed, n_train=args.n_train, n_test=args.n_test, n_poison=args.n_poison)

    plot_pre_adaptation(data, os.path.join(args.out_dir, "1_pre_adaptation.png"))
    plot_adaptation(data, os.path.join(args.out_dir, "2_adaptation.png"))
    _plot_eval(data, data["decision_c2"], os.path.join(args.out_dir, "3_pre_unlearning_eval.png"),
              "3. Pre-unlearning eval: test set lands in the pockets (C2)", show_correctness=True)
    _plot_eval(data, data["decision_c3"], os.path.join(args.out_dir, "4_post_unlearning.png"),
              "4. Post-unlearning: boundary retracts, pockets mostly closed (C3)", show_correctness=True)


if __name__ == "__main__":
    main()
