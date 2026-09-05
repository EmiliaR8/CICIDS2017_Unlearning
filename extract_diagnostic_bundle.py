"""
extract_diagnostic_bundle.py

Standalone utility for the tasks-8/9 retrain-vs-unlearn diagnostic
(madar_unlearning_cl_pipeline.py's DIAGNOSTIC_TASKS / --diagnostic_method).
Takes a dataset path and a checkpoint from a run that stopped after some
task (normally task 7, right before DIAGNOSTIC_TASKS), and writes ONE file
containing everything needed to inspect that diagnostic's inputs without
re-running the pipeline:

  - The FULL checkpoint dict as-is (replay buffer, SI omega/p_old_task,
    contrastive bank state, results, and -- since the checkpoint already
    carries task_test_splits for every task up through the checkpoint's own
    task_id -- the test sets for those prior tasks too).
  - The train/test splits for the two tasks AFTER the checkpoint (tasks 8
    and 9, for a task-7 checkpoint) -- these are NOT in the checkpoint
    (they haven't run yet), so they're rebuilt here via the exact same
    load_pooled_chronological_tasks(...) + train_test_split(...,
    random_state=seed) calls the real pipeline uses, so they match a real
    `--resume_from` run bit-for-bit given the same seed.

Usage:
    python extract_diagnostic_bundle.py \\
        --dataset /path/to/dataset(.h5 file or a directory containing one) \\
        --checkpoint /path/to/checkpoint(.pt file, or a run's out_dir/logs) \\
        --out diagnostic_bundle.pt

Output is a single torch.save() bundle (pickle-based, like the pipeline's
own checkpoints) -- load it back with:
    bundle = torch.load("diagnostic_bundle.pt", weights_only=False)
"""
import argparse
import glob
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from h5_data_loader import load_pooled_chronological_tasks
import madar_unlearning_cl_pipeline as pipeline


def _resolve_dataset_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        matches = sorted(glob.glob(os.path.join(path, "*.h5")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(f"No .h5 file found in directory: {path}")
        raise ValueError(
            f"Multiple .h5 files found in {path}, pass one directly with --dataset: {matches}"
        )
    raise FileNotFoundError(f"--dataset path does not exist: {path}")


def _resolve_checkpoint_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "classifier_checkpoint.pt"),
            os.path.join(path, "logs", "classifier_checkpoint.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"No classifier_checkpoint.pt found in {path} or {path}/logs"
        )
    raise FileNotFoundError(f"--checkpoint path does not exist: {path}")


def _build_task_split(tasks, task_offsets, task_id, seed):
    """Exact mirror of madar_unlearning_cl_pipeline.py's main() loop body:
    same X/y/gid construction, same train_test_split call (same seed,
    same test_size, same stratify) -- so this reproduces that task's
    train/test rows bit-for-bit for a run using this seed."""
    task = tasks[task_id]
    X = np.clip(task["features"].astype(np.float32), 0.0, 1.0)
    y = task["labels"].astype(np.int64)
    gid = task_offsets[task_id] + np.arange(len(y), dtype=np.int64)

    X_train, X_test, y_train, y_test, gid_train, gid_test = train_test_split(
        X, y, gid, test_size=pipeline.TASK_TEST_FRAC, random_state=seed, stratify=y
    )
    return {
        "X_train": X_train, "y_train": y_train, "gid_train": gid_train,
        "X_test": X_test, "y_test": y_test, "gid_test": gid_test,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Path to the .h5 dataset file, or a directory containing one.")
    ap.add_argument("--checkpoint", required=True,
                     help="Path to a classifier_checkpoint.pt, or a run's out_dir (or out_dir/logs).")
    ap.add_argument("--out", default="diagnostic_bundle.pt", help="Output bundle path.")
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_task_id = checkpoint["task_id"]
    seed = checkpoint["seed"]
    next_task_ids = [checkpoint_task_id + 1, checkpoint_task_id + 2]
    print(f"Checkpoint is at task {checkpoint_task_id} (seed={seed}); "
          f"regenerating tasks {next_task_ids} from {dataset_path}...")

    if any(t >= pipeline.NUM_TASKS for t in next_task_ids):
        raise ValueError(
            f"Checkpoint task_id={checkpoint_task_id} leaves fewer than 2 tasks remaining "
            f"(NUM_TASKS={pipeline.NUM_TASKS})."
        )

    tasks, day_mapping, label_mapping = load_pooled_chronological_tasks(dataset_path, pipeline.TASK_FRACTIONS)
    task_offsets = np.concatenate([[0], np.cumsum([len(t["labels"]) for t in tasks])[:-1]])

    regenerated_tasks = {
        t: _build_task_split(tasks, task_offsets, t, seed) for t in next_task_ids
    }
    for t in next_task_ids:
        d = regenerated_tasks[t]
        print(f"  task {t}: {len(d['y_train'])} train / {len(d['y_test'])} test rows")

    prior_test_task_ids = sorted(checkpoint.get("task_test_splits", {}).keys())
    print(f"Checkpoint already carries test splits for prior tasks: {prior_test_task_ids}")
    print(f"Replay buffer size in checkpoint: {len(checkpoint.get('replay_buffer', []))}")

    bundle = {
        "meta": {
            "source_dataset_path": dataset_path,
            "source_checkpoint_path": checkpoint_path,
            "checkpoint_task_id": checkpoint_task_id,
            "seed": seed,
            "regenerated_task_ids": next_task_ids,
            "prior_test_task_ids": prior_test_task_ids,
            "day_mapping": day_mapping,
            "label_mapping": label_mapping,
            "note": (
                "regenerated_tasks are NOT part of the checkpoint (the checkpoint only "
                "covers tasks up to checkpoint_task_id) -- rebuilt here via the exact same "
                "load_pooled_chronological_tasks + train_test_split(seed=...) calls the "
                "real pipeline uses, so they match a real --resume_from run bit-for-bit. "
                "Everything else (replay buffer, SI omega/p_old_task, contrastive bank "
                "state, and the test sets for prior_test_task_ids) is the checkpoint "
                "itself, included below unmodified under 'checkpoint'."
            ),
        },
        "regenerated_tasks": regenerated_tasks,
        "checkpoint": checkpoint,
    }

    torch.save(bundle, args.out)
    print(f"\nWrote bundle to {args.out}")
    print("Load it back with: torch.load(path, weights_only=False)")


if __name__ == "__main__":
    main()
