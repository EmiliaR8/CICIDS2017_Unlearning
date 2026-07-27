"""
naive_cl_pipeline.py

NEW naive continual-learning pipeline over the pooled/chronological 10-task
split of subsampled_dataset.h5 / full_dataset.h5 (see h5_data_loader.py's
load_pooled_chronological_tasks). Two learners run task-by-task:

  - Blue (XGBoostIDSWrapper, model.py): naive CL strategy -- sequential
    fine-tuning only. No replay buffer, no distillation, no regularization.
    Task 0 gets a full fit (train_model); tasks 1-9 get an incremental update
    (classifier.incremental_step) on that task's data alone. This is the
    naive baseline the project's phasing plan (joint -> MADAR -> MADAR+
    Unlearning) is meant to be compared against.
  - Red (SAC, adversary_env.NetworkAttackEnv): a FRESH agent is trained from
    scratch at every task boundary against a snapshot of the blue classifier
    as it stood at the end of the previous task, then attacks the current
    task's malicious training samples. This is the retraining cadence fixed
    in the prior handoff (decision #1) specifically to keep perturbation
    strategies from going stale as the classifier drifts across tasks.
    Diversity across tasks' agents is encouraged by
    RecencyWeightedContrastiveBank below: an extension of adversary_env's
    ContrastiveBank that reweights the "stay dissimilar to prior prototypes"
    term toward the immediately-previous task via geometric decay, rather
    than a flat average over every earlier task.

Task 0 is structurally clean (no red agent exists yet to poison it -- the
first classifier snapshot to attack doesn't exist until task 0 finishes
training), so there's no separate "clean warm-start" flag to set; poisoning
only starts at task 1.

This is a first-cut validation run: the goal is to confirm the red agent
actually produces the perturbations we expect (evasion rate that responds to
training, cumulative-perturbation signatures that diverge across tasks
rather than collapsing onto one strategy) before building the next phases
(joint, MADAR, MADAR+Unlearning) on top. See the run's plots/prototype_heatmap.png
and plots/episode_clouds.png for the direct read on that question.

Usage:
    python naive_cl_pipeline.py --seed 42 --log_name naive_10task_run \\
        --h5-path /mnt/processed_data/subsampled_dataset.h5
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from train_classifer import train_model
from adversary_env import NetworkAttackEnv, ContrastiveBank
from h5_data_loader import load_pooled_chronological_tasks

# ---------------------------------------------------------------------------
# Top-of-file config. Project convention (matches Static_Data_Pipeline's
# clmd_cicids_*_cd.py scripts): hyperparameters live here, not as CLI flags.
# Only --seed and --log_name are CLI.
# ---------------------------------------------------------------------------
H5_DATASET_PATH = "/mnt/processed_data/subsampled_dataset.h5"

NUM_TASKS = 10
# task0 = 30% warm start; tasks1-9 taper 9.18% -> 6.38% of the pooled dataset.
TASK_FRACTIONS = [0.3000, 0.0918, 0.0883, 0.0848, 0.0813, 0.0778, 0.0743, 0.0708, 0.0673, 0.0638]
TASK_TEST_FRAC = 0.20

POISON_FRACTION = 0.3
REQUIRE_EVASION_SUCCESS = True  # only successfully-evasive perturbations poison training data

RED_EPSILON = 0.25
RED_MAX_STEPS = 25
RED_TIMESTEPS_PER_TASK = 10_000  # fresh SAC agent trained this many steps per task boundary
ALPHA_CONTRAST = 0.5
CONTRASTIVE_EMA = 0.95
CONTRASTIVE_RECENCY_DECAY = 0.5  # weight of task (k-1) vs (k-2) vs ... in the diversity reward

# Caps evaluate_agent_on_batch's per-task episode count for runtime; None = every malicious sample.
MAX_EVAL_SAMPLES_PER_TASK = 1000

SAC_POLICY_KWARGS = dict(net_arch=[512, 256, 128], optimizer_kwargs={"weight_decay": 1e-4})
SAC_KWARGS = dict(
    learning_rate=1e-3, buffer_size=50_000, batch_size=128, gamma=0.95,
    ent_coef="auto_0.01", device="auto",
)

INCREMENTAL_NUM_NEW_TREES = 16
INCREMENTAL_MAX_TREES_TOTAL = 800


# ---------------------------------------------------------------------------
# Recency-weighted contrastive diversity
# ---------------------------------------------------------------------------
class RecencyWeightedContrastiveBank(ContrastiveBank):
    """
    Same EMA-prototype machinery as adversary_env.ContrastiveBank, but
    contrastive_reward compares against a geometrically recency-weighted
    average of every prior task's prototype instead of a flat mean, so the
    immediately-previous task's strategy is weighted most heavily while
    older tasks still contribute (decay -> 0 degrades to "ignore everything
    but the previous task"; decay -> 1 degrades to the flat-mean behavior of
    the base class).
    """

    def __init__(self, dim, ema=0.95, recency_decay=0.5):
        super().__init__(dim, ema=ema)
        self.recency_decay = recency_decay
        self.task_order = []  # ids in the order their prototypes were first created

    def update(self, agent_id, emb):
        super().update(agent_id, emb)
        if agent_id not in self.task_order:
            self.task_order.append(agent_id)

    def contrastive_reward(self, agent_id, emb):
        others_oldest_first = [pid for pid in self.task_order if pid != agent_id and pid in self.protos]
        if not others_oldest_first:
            return 0.0

        emb = self._norm(emb)
        others_recent_first = list(reversed(others_oldest_first))
        weights = np.array([self.recency_decay ** i for i in range(len(others_recent_first))])
        weights = weights / weights.sum()
        neg_sims = np.array([float(np.dot(emb, self.protos[pid])) for pid in others_recent_first])
        neg_mean = float(np.dot(weights, neg_sims))

        pos_sim = float(np.dot(emb, self.protos[agent_id])) if agent_id in self.protos else 0.0
        return pos_sim - neg_mean


# ---------------------------------------------------------------------------
# Red agent training / evaluation (adapted from test_red.py, single agent
# per task rather than N simultaneous agents, and extended to also return
# which specific samples were successfully evaded for poisoning selection).
# ---------------------------------------------------------------------------
class EpisodeTimerCallback(BaseCallback):
    """Measures wall-clock duration of each environment episode during model.learn()."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_times = []
        self._start_times = None
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("# episode_index, seconds\n")

    def _on_training_start(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        now = time.perf_counter()
        self._start_times = [now for _ in range(n_envs)]

    def _on_step(self):
        dones = self.locals.get("dones", None)
        if dones is None:
            return True
        dones_list = list(dones) if isinstance(dones, (list, tuple, np.ndarray)) else [bool(dones)]

        now = time.perf_counter()
        if self._start_times is None:
            self._start_times = [now for _ in range(len(dones_list))]

        for env_idx, done in enumerate(dones_list):
            if self._start_times[env_idx] is None:
                self._start_times[env_idx] = now
            if done:
                elapsed = now - self._start_times[env_idx]
                self.episode_times.append(elapsed)
                with open(self.log_path, "a") as f:
                    f.write(f"{len(self.episode_times)},{elapsed:.6f}\n")
                self._start_times[env_idx] = now
        return True

    def _on_training_end(self):
        if not self.episode_times:
            return
        avg = sum(self.episode_times) / len(self.episode_times)
        with open(self.log_path, "a") as f:
            f.write(f"# total_episodes={len(self.episode_times)}, avg_seconds={avg:.6f}\n")


def run_single_agent_attack(env, model, start_index, benign_label, deterministic=True):
    """Runs one episode attacking one sample index. Returns
    (final_obs, total_reward, pred_label, cum_perturbation)."""
    obs, _ = env.reset(options={"index": start_index})
    total_reward = 0.0
    cum_pert = np.zeros(env.action_space.shape[0], dtype=np.float32)

    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        a = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)
        a = np.clip(a, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(a)
        total_reward += float(reward)
        cum_pert += a
        done = bool(terminated) or bool(truncated)

    return obs, total_reward, info["prediction"], cum_pert


def evaluate_agent_on_batch(env, model, X, y, benign_label, only_malicious=True,
                             deterministic=True, max_test=None):
    """
    Runs the agent over (up to max_test of) the malicious samples. Returns
    (X_pert, evasion_rate, rewards, avg_pert_norm, n_attacked, evaded_mask).
    evaded_mask[i] is True only for samples where the attack succeeded --
    this is the exact set poisoning should draw from, so callers don't need
    to infer success by diffing X_pert against X.
    """
    X_pert = X.copy()
    evaded_mask = np.zeros(X.shape[0], dtype=bool)
    rewards, pert_norms = [], []
    evasion = 0
    attacked = 0

    # Filter to the samples actually eligible for attack BEFORE capping by
    # max_test, not after -- capping a raw range(X.shape[0]) first and only
    # then filtering out benign rows made max_test cap "rows scanned" rather
    # than "malicious samples attacked", starving poisoning on any task where
    # malicious samples aren't already in the first max_test rows.
    if only_malicious:
        indices = np.where(y != benign_label)[0]
    else:
        indices = np.arange(X.shape[0])
    if max_test is not None:
        indices = indices[:max_test]

    for i in indices:
        final_obs, total_reward, pred_label, cum_pert = run_single_agent_attack(
            env, model, start_index=i, benign_label=benign_label, deterministic=deterministic
        )
        rewards.append(total_reward)
        pert_norms.append(float(np.linalg.norm(cum_pert, ord=2)))
        attacked += 1

        if pred_label == benign_label and y[i] != benign_label:
            evasion += 1
            X_pert[i] = final_obs
            evaded_mask[i] = True

    evasion_rate = evasion / max(attacked, 1)
    avg_pert_norm = float(np.mean(pert_norms)) if pert_norms else 0.0
    return X_pert, evasion_rate, rewards, avg_pert_norm, attacked, evaded_mask


def train_red_agent_for_task(task_id, classifier, X_train, y_train, benign_label, bank, seed, out_dir):
    def _thunk():
        return NetworkAttackEnv(
            classifier, X_train, y_train,
            benign_label=benign_label,
            max_steps=RED_MAX_STEPS,
            epsilon=RED_EPSILON,
            agent_id=task_id,
            contrastive_bank=bank,
            alpha_contrast=ALPHA_CONTRAST,
        )

    vec_env = make_vec_env(_thunk, n_envs=1, seed=seed + task_id)
    agent = SAC(
        "MlpPolicy", vec_env, verbose=0, policy_kwargs=SAC_POLICY_KWARGS,
        seed=seed + task_id, **SAC_KWARGS,
    )
    timer_cb = EpisodeTimerCallback(
        log_path=os.path.join(out_dir, "logs", f"red_episode_times_task{task_id}.txt")
    )
    agent.learn(total_timesteps=RED_TIMESTEPS_PER_TASK, callback=timer_cb)
    return vec_env.envs[0], agent


# ---------------------------------------------------------------------------
# Blue classifier evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(classifier, X, y):
    y_pred = classifier.predict(X)
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_task_metrics(results, out_path):
    task_ids = [r["task_id"] for r in results]
    pooled_bal_acc = [r["pooled_eval"]["balanced_accuracy"] for r in results]
    mean_per_task_bal_acc = [r["mean_per_task_balanced_accuracy"] for r in results]
    train_evasion = [r["red_agent"]["train_evasion_rate"] if r["red_agent"] else None for r in results]
    test_evasion = [r["red_agent"]["test_evasion_rate"] if r["red_agent"] else None for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(task_ids, pooled_bal_acc, marker="o", label="pooled balanced accuracy")
    ax1.plot(task_ids, mean_per_task_bal_acc, marker="s", label="mean per-task balanced accuracy")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("Naive CL: classifier accuracy per task boundary")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xs = [t for t, v in zip(task_ids, train_evasion) if v is not None]
    ys_train = [v for v in train_evasion if v is not None]
    ys_test = [v for v in test_evasion if v is not None]
    ax2.plot(xs, ys_train, marker="o", color="tab:red", label="evasion rate (train samples)")
    ax2.plot(xs, ys_test, marker="s", color="tab:orange", label="evasion rate (held-out test samples)")
    ax2.set_xlabel("Task id")
    ax2.set_ylabel("Evasion rate")
    ax2.set_title("Red agent evasion rate per task boundary (task 0 has no red agent)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prototype_heatmap(bank, out_path):
    agent_ids, S = bank.cosine_matrix(agent_ids=sorted(bank.protos.keys()))
    if S is None or len(agent_ids) < 2:
        print("[Plot] Not enough task prototypes to plot a heatmap yet.")
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(S, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(agent_ids)), agent_ids)
    plt.yticks(range(len(agent_ids)), agent_ids)
    plt.title("Red agent policy diversity across tasks\n(prototype cosine similarity)")
    plt.xlabel("Task id")
    plt.ylabel("Task id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_episode_clouds(bank, out_path):
    task_ids = sorted(bank.episode_embs.keys())
    X, y = [], []
    for tid in task_ids:
        for e in bank.episode_embs.get(tid, []):
            X.append(e)
            y.append(tid)

    if len(X) < 5:
        print("[Plot] Not enough episode embeddings to plot.")
        return

    X = np.stack(X, axis=0)
    y = np.array(y)

    X0 = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    Z = X0 @ Vt[:2].T

    cmap = matplotlib.colormaps["tab10"]
    plt.figure(figsize=(7, 6))
    for tid in task_ids:
        mask = y == tid
        if mask.sum() == 0:
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.6, color=cmap(tid % 10), label=f"task {tid}")

    plt.title("Episode embedding clouds (PCA-2D of cumulative perturbations)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_name", type=str, default="naive_cl_run")
    ap.add_argument("--h5-path", type=str, default=H5_DATASET_PATH)
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = os.path.join("runs", args.log_name)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    print(f"Loading {args.h5_path} and building {NUM_TASKS} pooled chronological tasks...")
    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(args.h5_path, TASK_FRACTIONS)
    benign_label = label_mapping["Benign"]
    mal_label = 1 - benign_label
    feature_dim = tasks[0]["features"].shape[1]
    print(f"day_mapping={day_mapping}, feature_dim={feature_dim}, "
          f"task sizes={[len(t['labels']) for t in tasks]}")

    bank = RecencyWeightedContrastiveBank(
        dim=feature_dim, ema=CONTRASTIVE_EMA, recency_decay=CONTRASTIVE_RECENCY_DECAY
    )
    # ContrastiveBank.step() calls log_cosine_snapshot() once >=2 task prototypes
    # exist; it requires this to have been called first or it AttributeErrors.
    bank.init_distance_logger(
        path=os.path.join(out_dir, "logs", "prototype_cosine_over_time.txt"),
        agent_ids=list(range(NUM_TASKS)),
    )

    classifier = None
    task_test_splits = {}
    results = []
    warnings_log = []

    for t, task in enumerate(tasks):
        X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)  # data is documented [0,1]-scaled already
        y = task["labels"].astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TASK_TEST_FRAC, random_state=args.seed, stratify=y
        )
        task_test_splits[t] = (X_test, y_test)

        mal_idx_train = np.where(y_train == mal_label)[0]
        red_report = None
        n_poisoned = 0

        if t == 0:
            print(f"\n===== Task 0 (warm start, no red agent): "
                  f"{len(y_train)} train / {len(y_test)} test =====")
            classifier, _ = train_model(X_train, y_train)
            X_train_for_classifier = X_train

        else:
            print(f"\n===== Task {t}: training fresh red agent against pre-task-{t} classifier "
                  f"({len(y_train)} train / {len(y_test)} test, {len(mal_idx_train)} malicious train) =====")

            if len(mal_idx_train) == 0:
                warnings_log.append(f"Task {t}: no malicious training samples, skipping red agent.")
                X_train_for_classifier = X_train
            else:
                env, agent = train_red_agent_for_task(
                    t, classifier, X_train, y_train, benign_label, bank, args.seed, out_dir
                )

                X_train_pert, train_evasion_rate, train_rewards, train_avg_norm, train_attacked, evaded_mask = \
                    evaluate_agent_on_batch(
                        env, agent, X_train, y_train, benign_label,
                        only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                    )
                _, test_evasion_rate, test_rewards, test_avg_norm, test_attacked, _ = evaluate_agent_on_batch(
                    env, agent, X_test, y_test, benign_label,
                    only_malicious=True, deterministic=True, max_test=MAX_EVAL_SAMPLES_PER_TASK,
                )

                red_report = {
                    "train_evasion_rate": train_evasion_rate, "train_attacked": train_attacked,
                    "train_avg_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
                    "train_avg_pert_l2": train_avg_norm,
                    "test_evasion_rate": test_evasion_rate, "test_attacked": test_attacked,
                    "test_avg_reward": float(np.mean(test_rewards)) if test_rewards else 0.0,
                    "test_avg_pert_l2": test_avg_norm,
                }
                print(f"[Task {t} red agent] train_evasion={train_evasion_rate:.3f} "
                      f"test_evasion={test_evasion_rate:.3f} avg_pert_L2={train_avg_norm:.3f}")

                poison_pool = np.where(evaded_mask)[0] if REQUIRE_EVASION_SUCCESS else mal_idx_train
                n_to_poison = min(int(round(POISON_FRACTION * len(mal_idx_train))), len(poison_pool))
                rng = np.random.RandomState(args.seed + t)
                poison_idx = rng.choice(poison_pool, size=n_to_poison, replace=False) if n_to_poison > 0 else \
                    np.array([], dtype=int)
                n_poisoned = len(poison_idx)

                X_train_for_classifier = X_train.copy()
                X_train_for_classifier[poison_idx] = X_train_pert[poison_idx]
                print(f"[Task {t}] poisoned {n_poisoned}/{len(mal_idx_train)} malicious train samples "
                      f"(poison_fraction={POISON_FRACTION})")

            classifier.incremental_step(
                X_train_for_classifier, y_train,
                num_new_trees=INCREMENTAL_NUM_NEW_TREES, max_trees_total=INCREMENTAL_MAX_TREES_TOTAL,
            )

        per_task_eval = {j: evaluate_classifier(classifier, *task_test_splits[j]) for j in range(t + 1)}
        pooled_X = np.concatenate([task_test_splits[j][0] for j in range(t + 1)])
        pooled_y = np.concatenate([task_test_splits[j][1] for j in range(t + 1)])
        pooled_eval = evaluate_classifier(classifier, pooled_X, pooled_y)
        mean_per_task_bal_acc = float(np.mean([per_task_eval[j]["balanced_accuracy"] for j in range(t + 1)]))

        print(f"[Task {t} classifier] this-task acc={per_task_eval[t]['balanced_accuracy']:.4f} "
              f"pooled acc={pooled_eval['balanced_accuracy']:.4f} "
              f"mean-per-task acc={mean_per_task_bal_acc:.4f}")

        results.append({
            "task_id": t,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "n_malicious_train": int(len(mal_idx_train)),
            "n_poisoned": n_poisoned,
            "red_agent": red_report,
            "per_task_eval": per_task_eval,
            "pooled_eval": pooled_eval,
            "mean_per_task_balanced_accuracy": mean_per_task_bal_acc,
        })

    config = {
        "h5_path": args.h5_path, "seed": args.seed, "num_tasks": NUM_TASKS,
        "task_fractions": TASK_FRACTIONS, "task_test_frac": TASK_TEST_FRAC,
        "poison_fraction": POISON_FRACTION, "require_evasion_success": REQUIRE_EVASION_SUCCESS,
        "red_epsilon": RED_EPSILON, "red_max_steps": RED_MAX_STEPS,
        "red_timesteps_per_task": RED_TIMESTEPS_PER_TASK, "alpha_contrast": ALPHA_CONTRAST,
        "contrastive_ema": CONTRASTIVE_EMA, "contrastive_recency_decay": CONTRASTIVE_RECENCY_DECAY,
        "max_eval_samples_per_task": MAX_EVAL_SAMPLES_PER_TASK, "feature_dim": feature_dim,
        "day_mapping": day_mapping,
    }
    with open(os.path.join(out_dir, f"{args.log_name}.json"), "w") as f:
        json.dump({"config": config, "warnings": warnings_log, "results": results}, f, indent=2)

    plot_task_metrics(results, os.path.join(out_dir, "plots", "task_metrics.png"))
    plot_prototype_heatmap(bank, os.path.join(out_dir, "plots", "prototype_heatmap.png"))
    plot_episode_clouds(bank, os.path.join(out_dir, "plots", "episode_clouds.png"))

    print(f"\nDone. Results + plots written to {out_dir}/")


if __name__ == "__main__":
    main()
