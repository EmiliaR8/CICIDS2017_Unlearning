"""
tinyimagenet_data_loader.py

TinyImageNet-200 loading + class-incremental task scheduling for
madar_pocket_pipeline_tinyimagenet.py.

Two pieces are ported (with attribution) from a sister project,
EmiliaR8/Meta-Unlearning (code/core/tasks.py and code/core/data/{base,tinyimagenet}.py),
which already runs class-incremental continual learning on this exact dataset
(no poisoning/attack layer there -- that's what this repo's pipeline adds on top):

  - TaskSchedule / build_schedule: the "<task0>+<step>x<n_increments>" spec
    parser. Class ids follow PRESENTATION order (wnids.txt order, not directory
    listing order, which is filesystem-dependent) -- classes_for(t) is which
    classes are INTRODUCED by task t, seen_classes(t) is every class
    introduced up to and including task t.
  - The JPEG decode/uint8-cache logic in _decode/_build_train/_build_val/_cached.
    Kept uint8 end-to-end (float32 would be ~5GB resident for the training
    split); ClassifierCNN converts to float and normalizes per batch, not here.

Adapted from the original: no `paths` object (this repo's pipelines take flat
--h5-path-style CLI args), and load_tinyimagenet_tasks() returns per-task
(X_train, y_train, X_test, y_test) splits directly by filtering on
schedule.classes_for(t)/seen_classes(t), matching how h5_data_loader.py's
load_pooled_chronological_tasks() hands madar_pocket_pipeline.py one dict per
task -- so the rest of the pipeline can stay structurally close to the
network-flow version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIRNAME = "tiny-imagenet-200"
IMAGE_SHAPE = (3, 64, 64)


# ---------------------------------------------------------------------------
# TaskSchedule (ported from Meta-Unlearning's core/tasks.py)
# ---------------------------------------------------------------------------
_SPEC = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)

# 20+20x9: 20 base classes in task 0, +20 new classes/task for 9 more tasks ->
# 10 tasks total, all 200 TinyImageNet classes used, 20 new classes/task
# throughout (task 0 included) so poisoning has the same "this task's own new
# classes" shape at every task, not just tasks 1+.
DEFAULT_TASK_SETUP = "20+20x9"


@dataclass(frozen=True)
class TaskSchedule:
    task0: int
    step: int
    n_increments: int

    @property
    def n_tasks(self) -> int:
        return self.n_increments + 1

    @property
    def n_classes(self) -> int:
        return self.task0 + self.step * self.n_increments

    @property
    def spec(self) -> str:
        return f"{self.task0}+{self.step}x{self.n_increments}"

    def classes_for(self, tid: int) -> list:
        """Classes introduced BY task `tid` (task 0 introduces the base set)."""
        self._check(tid)
        if tid == 0:
            return list(range(self.task0))
        lo = self.task0 + (tid - 1) * self.step
        return list(range(lo, lo + self.step))

    def seen_classes(self, tid: int) -> list:
        """Every class introduced up to and including task `tid`."""
        return list(range(self.active_count(tid)))

    def active_count(self, tid: int) -> int:
        self._check(tid)
        return self.task0 + tid * self.step

    def _check(self, tid: int) -> None:
        if not 0 <= tid < self.n_tasks:
            raise IndexError(f"task {tid} outside 0..{self.n_tasks - 1} for {self.spec}")

    def as_dict(self) -> dict:
        return {"spec": self.spec, "task0_classes": self.task0,
                "step_classes": self.step, "n_increments": self.n_increments,
                "n_tasks": self.n_tasks, "n_classes": self.n_classes}


def build_schedule(spec: str) -> TaskSchedule:
    m = _SPEC.match(str(spec))
    if not m:
        raise ValueError(f"task setup {spec!r} not understood. Use '<task0>+<step>x<n>', e.g. '20+20x9'.")
    task0, step, n_inc = (int(g) for g in m.groups())
    if task0 <= 0 or step <= 0 or n_inc <= 0:
        raise ValueError(f"task setup {spec!r}: all three components must be positive")
    return TaskSchedule(task0=task0, step=step, n_increments=n_inc)


# ---------------------------------------------------------------------------
# Decode + cache (ported from Meta-Unlearning's core/data/tinyimagenet.py)
# ---------------------------------------------------------------------------
def _wnids(root: Path) -> list:
    """Class order is wnids.txt as distributed, NOT directory listing order --
    directory order varies by filesystem, and a label space that depends on
    the machine would make two servers' task schedules silently disagree on
    which classes are 0..19, 20..39, etc."""
    f = root / "wnids.txt"
    if not f.exists():
        raise SystemExit(f"{f} missing; the archive looks incomplete")
    ids = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    if len(ids) != 200:
        raise SystemExit(f"{f} lists {len(ids)} classes, expected 200")
    return ids


def _decode(files) -> np.ndarray:
    from PIL import Image
    out = np.empty((len(files),) + IMAGE_SHAPE, dtype=np.uint8)
    for i, f in enumerate(files):
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)  # greyscale files: RGB replicates the channel
        out[i] = arr.transpose(2, 0, 1)
    return out


def _build_train(root: Path, wnids: list):
    files, labels = [], []
    for idx, wnid in enumerate(wnids):
        d = root / "train" / wnid / "images"
        if not d.is_dir():
            d = root / "train" / wnid
        got = sorted(p for p in d.iterdir() if p.suffix.upper() == ".JPEG")
        if not got:
            raise SystemExit(f"no JPEGs for class {wnid} under {d}")
        files.extend(got)
        labels.extend([idx] * len(got))
    return _decode(files), np.asarray(labels, dtype=np.int64)


def _build_val(root: Path, wnids: list):
    """The distributed test/ split has no labels; val (50 labelled images/class)
    is used as the test set, same convention every CL paper on this benchmark
    uses."""
    ann = root / "val" / "val_annotations.txt"
    if not ann.exists():
        raise SystemExit(f"{ann} missing; cannot label the validation split")
    index = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    for line in ann.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, wnid = parts[0], parts[1]
        if wnid not in index:
            raise SystemExit(f"{ann} references unknown class {wnid}")
        files.append(root / "val" / "images" / name)
        labels.append(index[wnid])
    if len(files) != 10000:
        raise SystemExit(f"{ann} lists {len(files)} images, expected 10,000")
    return _decode(files), np.asarray(labels, dtype=np.int64)


def _cached(cache_root: Path, split: str, build):
    xs = cache_root / f"tinyimagenet_{split}_x.npy"
    ys = cache_root / f"tinyimagenet_{split}_y.npy"
    if xs.exists() and ys.exists():
        return np.load(xs), np.load(ys)
    X, y = build()
    cache_root.mkdir(parents=True, exist_ok=True)
    np.save(xs, X)
    np.save(ys, y)
    return np.asarray(X), y


def load_tinyimagenet_raw(data_root: str, cache_root: str = None):
    """Returns (X_train, y_train, X_test, y_test, wnids) as uint8 arrays,
    labels in wnids.txt presentation order (0..199)."""
    root = Path(data_root) / DIRNAME
    if not root.exists():
        raise SystemExit(
            f"TinyImageNet not found at {root}.\nDownload and unzip it there:\n"
            f"  cd {data_root}\n  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            f"  unzip -q tiny-imagenet-200.zip")
    wnids = _wnids(root)
    cache = Path(cache_root) if cache_root else Path(data_root) / "_cache"
    X_train, y_train = _cached(cache, "train", lambda: _build_train(root, wnids))
    X_test, y_test = _cached(cache, "val", lambda: _build_val(root, wnids))
    return X_train, y_train, X_test, y_test, wnids


# ---------------------------------------------------------------------------
# Per-task split, matching h5_data_loader.load_pooled_chronological_tasks'
# contract of handing the pipeline one dict per task.
# ---------------------------------------------------------------------------
def load_tinyimagenet_tasks(data_root: str, schedule: TaskSchedule, cache_root: str = None):
    """Returns (tasks, wnids) where tasks[t] = {"X_train","y_train","X_test","y_test"}
    holds ONLY the rows of classes_for(t) -- this task's own newly-introduced
    classes -- matching how the network-flow pipeline treats "this task's own
    train/test data" as what poisoning/detection/logging act on. Use
    schedule.seen_classes(t) against task_test_splits (accumulated by the
    caller across tasks, same pattern as the existing pipeline) for
    pooled/mean accuracy over everything seen so far.
    """
    X_train_all, y_train_all, X_test_all, y_test_all, wnids = load_tinyimagenet_raw(data_root, cache_root)
    tasks = []
    for t in range(schedule.n_tasks):
        cls = np.asarray(schedule.classes_for(t))
        tr_mask = np.isin(y_train_all, cls)
        te_mask = np.isin(y_test_all, cls)
        tasks.append({
            "X_train": X_train_all[tr_mask], "y_train": y_train_all[tr_mask],
            "X_test": X_test_all[te_mask], "y_test": y_test_all[te_mask],
        })
    return tasks, wnids
