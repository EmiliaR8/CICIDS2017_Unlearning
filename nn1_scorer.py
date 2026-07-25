"""
NN-1: a tiny per-sample scorer for learned forget-set selection.

Deliberately small so Evolution Strategies can train it from scalar rewards.
Default is LINEAR (7 feature weights + bias = 8 parameters): well within
small-population ES reach, and directly interpretable - each learned weight
answers "does this property make a sample worth unlearning?". A one-hidden-
layer MLP is available via hidden>0 once the linear version shows signal
(expect to need a substantially larger evaluation budget for it).

Features per current-task sample (computed in the inner loop, z-scored
within the task so scales are comparable across tasks/datasets):
    0. Isolation-Forest score in LATENT space (joint over the task)
    1. Isolation-Forest score in RAW feature space (joint over the task)
    2. Per-sample cross-entropy loss under the current model
    3. Prediction entropy (normalized by log #active classes)
    4. Margin (top-1 prob minus top-2 prob)
    5. Latent L2 distance to own family centroid
    6. log10(family size)

Selection: score every sample, forget the TOP `ratio` fraction by score.
(The sign convention is learned; ES is free to invert any feature.)
"""

import numpy as np

N_FEATURES = 7          # mode A: current-task selection
N_FEATURES_B = 9        # mode B: buffer selection vs the new distribution
FEATURE_NAMES = ['iso_latent', 'iso_raw', 'ce_loss', 'entropy',
                 'margin', 'centroid_dist', 'log_family_size']
FEATURE_NAMES_B = FEATURE_NAMES + ['grad_conflict', 'density_ratio']


def n_params(hidden=0, n_features=N_FEATURES):
    if hidden == 0:
        return n_features + 1                      # linear + bias
    return (n_features * hidden + hidden) + (hidden + 1)


def init_params(hidden=0, seed=0, scale=0.3, n_features=N_FEATURES):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_params(hidden, n_features)) * scale).astype(np.float64)


def score(params, features, hidden=0):
    """features: (n_samples, n_features) z-scored. Returns (n_samples,)."""
    p = np.asarray(params, dtype=np.float64)
    nf = features.shape[1]
    assert p.shape == (n_params(hidden, nf),), \
        f"expected {n_params(hidden, nf)} params for hidden={hidden}, n_features={nf}, got {p.shape}"
    if hidden == 0:
        return features @ p[:nf] + p[nf]
    i = 0
    W1 = p[i:i + nf * hidden].reshape(nf, hidden); i += nf * hidden
    b1 = p[i:i + hidden]; i += hidden
    W2 = p[i:i + hidden]; i += hidden
    b2 = p[i]
    return np.tanh(features @ W1 + b1) @ W2 + b2


def select_forget(params, features, ratio, hidden=0):
    """Return (forget_idx, retain_idx) - forget the top `ratio` by score."""
    s = score(params, features, hidden)
    n_forget = int(len(s) * ratio)
    order = np.argsort(s)
    forget_idx = order[-n_forget:] if n_forget > 0 else np.array([], dtype=int)
    retain_idx = order[:-n_forget] if n_forget > 0 else order
    return forget_idx, retain_idx


def save(path, params, hidden=0, n_features=N_FEATURES):
    np.savez(path, params=np.asarray(params), hidden=np.array(hidden),
             n_features=np.array(n_features))


def load(path):
    z = np.load(path)
    hidden = int(z['hidden']) if 'hidden' in z else 0
    nf = int(z['n_features']) if 'n_features' in z else N_FEATURES
    return z['params'], hidden, nf


def zscore(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + eps)