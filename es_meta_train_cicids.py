"""
run with python es_meta_train_cicids.py --pop 4 --generations 2 --window_rows 15000 \
    --n_tasks 6 --cl_iters 200 --unlearn_epochs 5 --tag smoketest2
Evolution Strategies driver for meta-learning NN-1 on CICIDS (forget-set selection).

Same OpenAI-style ES as es_meta_train.py (mirrored/antithetic sampling, rank-shaped
fitness, common random numbers within a generation), pointed at mu_inner_loop_cicids.py
instead of the EMBER inner loop. Two differences from the EMBER driver:

  * No GPUs assumed. clmd_cicids_*.py forces CPU (newer GPUs can crash the installed
    torch build -- see those files), so candidates run as parallel CPU subprocesses
    instead of one-per-GPU; --workers caps how many run at once.
  * Outer-loop sliding window: each generation gets its own --window_rows-sized, time-
    ordered slice of the dataset. Every candidate IN a generation shares that window
    (fair comparison -- same data, only the NN-1 weights differ). The window then slides
    forward by (1 - --window_overlap) * --window_rows rows each generation, so
    consecutive generations share --window_overlap of their data (default 45%) rather
    than each claiming a fresh chunk -- this stretches a fixed-size dataset across many
    more generations before it needs to wrap back to row 0. Task-0 training doesn't
    depend on the selector, so it's cached once per (window, seed) and reused by every
    candidate in that generation.

MINI run (quick signal):
    python es_meta_train_cicids.py --pop 8 --generations 10 --window_rows 15000 \
        --n_tasks 5 --cl_iters 500 --unlearn_epochs 1 --tag mini

FULL run (after the mini looks sane):
    python es_meta_train_cicids.py --pop 16 --generations 40 --window_rows 20000 \
        --n_tasks 6 --cl_iters 2000 --unlearn_epochs 3 --tag full
"""

import argparse
import json
import os
import pickle
import queue
import subprocess
import sys
import threading

import numpy as np

import nn1_scorer

parser = argparse.ArgumentParser(description="ES meta-training for NN-1 forget-set selection (CICIDS)")
parser.add_argument('--dataset_path', type=str, default="./Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset.pkl")
parser.add_argument('--workers', type=int, default=4, help='Max CICIDS inner-loop subprocesses running at once')
parser.add_argument('--pop', type=int, default=8, help='Population size (must be even; mirrored pairs)')
parser.add_argument('--generations', type=int, default=10)
parser.add_argument('--sigma', type=float, default=0.1, help='Perturbation std')
parser.add_argument('--lr', type=float, default=0.05, help='ES learning rate')
parser.add_argument('--n_tasks', type=int, default=10, help='Total tasks including task 0')
parser.add_argument('--cl_iters', type=int, default=2000)
parser.add_argument('--unlearn_epochs', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.2, help='Weight of the forget loss during unlearning')
parser.add_argument('--window_rows', type=int, default=None,
                    help='Rows per outer-loop window. Default: the whole dataset every generation '
                         '(sliding disabled) -- pass this to actually slide across the dataset.')
parser.add_argument('--window_overlap', type=float, default=0.45,
                    help='Fraction of a window shared with the previous one (0.4-0.5 typical)')
parser.add_argument('--inner_seed', type=int, default=42, help='Base inner-loop seed (rotates per generation)')
parser.add_argument('--tag', type=str, default='es_cicids')
parser.add_argument('--hidden', type=int, default=0,
                    help='NN-1 hidden units. 0 = linear (8 params, recommended for small budgets)')
parser.add_argument('--resume', type=str, default=None, help='npz with key "params" to resume from')
parser.add_argument('--skip_baselines', action='store_true')
args = parser.parse_args()

assert args.pop % 2 == 0, "--pop must be even (mirrored sampling)"
assert 0.0 <= args.window_overlap < 1.0, "--window_overlap must be in [0, 1)"
WORK = f'es_work_{args.tag}'
os.makedirs(WORK, exist_ok=True)


def count_original_rows(path):
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    return int((blob["data_original_scale"]["row_type"] == "original").sum())


N_TOTAL = count_original_rows(args.dataset_path)
WINDOW_ROWS = args.window_rows if args.window_rows is not None else N_TOTAL
assert WINDOW_ROWS <= N_TOTAL, f"--window_rows {WINDOW_ROWS} exceeds the dataset ({N_TOTAL} original rows)"
if args.window_rows is None:
    print(f"No --window_rows given: using the full dataset ({N_TOTAL} rows) every generation "
          f"(outer-loop sliding disabled).")
WINDOW_STEP = max(1, int(round(WINDOW_ROWS * (1.0 - args.window_overlap))))
print(f"Dataset has {N_TOTAL} original rows. window_rows={WINDOW_ROWS}  "
      f"window_step={WINDOW_STEP} ({int((1 - args.window_overlap) * 100)}% of a window, "
      f"{int(args.window_overlap * 100)}% overlap with the previous one)")

# --- Subprocess evaluation with a worker-slot queue (CPU-only, no GPU pinning) ---
worker_q = queue.Queue()
for w in range(args.workers):
    worker_q.put(w)


def ensure_checkpoint(window_start, seed, checkpoint_path):
    if os.path.exists(checkpoint_path):
        return
    print(f"Building task-0 checkpoint: window_start={window_start} seed={seed} -> {checkpoint_path}")
    cmd = [sys.executable, 'mu_inner_loop_cicids.py',
           '--dataset_path', args.dataset_path,
           '--window_start', str(window_start), '--window_rows', str(WINDOW_ROWS),
           '--checkpoint', checkpoint_path, '--seed', str(seed),
           '--n_tasks', str(args.n_tasks), '--make_checkpoint', '--quiet']
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"failed to build checkpoint for window_start={window_start}, seed={seed}")


def run_inner(selector, out_path, seed, window_start, checkpoint_path, scorer_path=None):
    slot = worker_q.get()
    try:
        cmd = [sys.executable, 'mu_inner_loop_cicids.py',
               '--dataset_path', args.dataset_path,
               '--selector', selector, '--out', out_path, '--seed', str(seed),
               '--window_start', str(window_start), '--window_rows', str(WINDOW_ROWS),
               '--checkpoint', checkpoint_path,
               '--n_tasks', str(args.n_tasks), '--cl_iters', str(args.cl_iters),
               '--unlearn_epochs', str(args.unlearn_epochs), '--alpha', str(args.alpha),
               '--quiet']
        if scorer_path:
            cmd += ['--scorer_weights', scorer_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"inner loop failed ({selector}, seed {seed})")
        with open(out_path) as f:
            return json.load(f)['mean_acc']
    finally:
        worker_q.put(slot)


def run_parallel(jobs):
    """jobs: list of (selector, out_path, seed, window_start, checkpoint_path, scorer_path). Returns rewards in order."""
    rewards = [None] * len(jobs)
    threads = []

    def work(i, job):
        rewards[i] = run_inner(*job)

    for i, job in enumerate(jobs):
        t = threading.Thread(target=work, args=(i, job)); t.start(); threads.append(t)
    for t in threads:
        t.join()
    return rewards


# --- Baselines under the SAME mini config, window_start=0 (crucial for fair comparison) ---
BASELINE_SELECTORS = ['donut', 'random']
baselines = {}
if not args.skip_baselines:
    baseline_checkpoint = f'{WORK}/ckpt_win0_seed{args.inner_seed}.pt'
    ensure_checkpoint(0, args.inner_seed, baseline_checkpoint)
    print(f"Evaluating baselines {BASELINE_SELECTORS} under the current config...")
    jobs = [(b, f'{WORK}/base_{b}.json', args.inner_seed, 0, baseline_checkpoint, None) for b in BASELINE_SELECTORS]
    r = run_parallel(jobs)
    baselines = dict(zip(BASELINE_SELECTORS, r))
    for k, v in baselines.items():
        print(f"  {k:14s} {v:.2f}%")

# --- ES main loop ---
N_FEAT = nn1_scorer.N_FEATURES
if args.resume:
    theta, resumed_hidden, resumed_nf = nn1_scorer.load(args.resume)
    assert resumed_hidden == args.hidden and resumed_nf == N_FEAT, "resume file architecture mismatch"
else:
    theta = nn1_scorer.init_params(hidden=args.hidden, seed=0, scale=0.3, n_features=N_FEAT)
D = nn1_scorer.n_params(args.hidden, N_FEAT)
rng = np.random.default_rng(123)
log_path = f'{WORK}/es_log.csv'
with open(log_path, 'a') as f:
    f.write('generation,window_start,theta_reward,pop_mean,pop_max,sigma,lr\n')

window_start = 0
for gen in range(args.generations):
    if gen > 0:
        window_start = window_start + WINDOW_STEP
        if window_start + WINDOW_ROWS > N_TOTAL:
            window_start = 0          # wrap back to the start of the (time-ordered) dataset

    gen_seed = args.inner_seed + gen          # common random numbers within a generation
    checkpoint_path = f'{WORK}/ckpt_win{window_start}_seed{gen_seed}.pt'
    ensure_checkpoint(window_start, gen_seed, checkpoint_path)

    eps = rng.standard_normal((args.pop // 2, D))
    eps = np.concatenate([eps, -eps])          # mirrored pairs

    jobs = []
    for i in range(args.pop):
        cand = theta + args.sigma * eps[i]
        cpath = f'{WORK}/cand_g{gen}_i{i}.npz'
        nn1_scorer.save(cpath, cand, hidden=args.hidden, n_features=N_FEAT)
        jobs.append(('nn1', f'{WORK}/r_g{gen}_i{i}.json', gen_seed, window_start, checkpoint_path, cpath))
    rewards = np.array(run_parallel(jobs), dtype=float)

    # Rank-shaped fitness (robust to reward scale/outliers)
    ranks = np.empty(args.pop); ranks[np.argsort(rewards)] = np.arange(args.pop)
    shaped = (ranks / (args.pop - 1)) - 0.5
    grad = (shaped[:, None] * eps).mean(axis=0) / args.sigma
    theta = theta + args.lr * grad

    # Evaluate current theta itself (same seed/window) for a clean learning curve
    tpath = f'{WORK}/theta_g{gen}.npz'
    nn1_scorer.save(tpath, theta, hidden=args.hidden, n_features=N_FEAT)
    theta_reward = run_inner('nn1', f'{WORK}/r_theta_g{gen}.json', gen_seed, window_start, checkpoint_path, tpath)

    with open(log_path, 'a') as f:
        f.write(f'{gen},{window_start},{theta_reward:.4f},{rewards.mean():.4f},{rewards.max():.4f},'
                f'{args.sigma},{args.lr}\n')
    base_str = ' | '.join(f'{k} {v:.2f}' for k, v in baselines.items())
    print(f"Gen {gen:3d} | window_start {window_start:6d} | theta {theta_reward:.2f}% | "
          f"pop mean {rewards.mean():.2f}% max {rewards.max():.2f}% | baselines: {base_str}")

nn1_scorer.save(f'{WORK}/theta_final.npz', theta, hidden=args.hidden, n_features=N_FEAT)
if args.hidden == 0:
    print('Learned linear feature weights (positive => more likely forgotten):')
    for name, w in zip(nn1_scorer.FEATURE_NAMES, theta[:N_FEAT]):
        print(f'    {name:18s} {w:+.3f}')
    print(f'    {"bias":18s} {theta[N_FEAT]:+.3f}')
print(f"\nDone. Final NN-1 weights: {WORK}/theta_final.npz | log: {log_path}")
print("Evaluate on a held-out window with, e.g.:")
print(f"  python mu_inner_loop_cicids.py --selector nn1 --scorer_weights {WORK}/theta_final.npz "
      f"--window_start 0 --n_tasks {args.n_tasks} --cl_iters {args.cl_iters} --seed 7 --out eval_full.json")
