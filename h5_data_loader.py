"""
Loader for the HDF5 datasets produced by build_datasets.py (subsampled_dataset.h5 /
full_dataset.h5), replacing the CSV-only loaders in data_loader.py for the new
JSON/HDF5 data path described in the CL handoff.

Expected on-disk schema (per build_datasets.py, HANDOFF_1.md section 3):
  - "features"      float32 dataset, shape (N, D)
  - "label"         int8 dataset, shape (N,)      -- 0 = benign, 1 = malicious
  - "source_day"    int8 dataset, shape (N,)      -- CL task boundary id
  - "timestamp"     float64 dataset, shape (N,)   -- unix epoch
  - "attack_family" UTF-8 string dataset, shape (N,)
  - file attr "day_mapping"                       -- source_day id -> day name

day_mapping is read defensively since its exact on-disk encoding (JSON string vs.
string array vs. dict-like) wasn't pinned down by the session that wrote
build_datasets.py. If the attr is absent or unparseable, this falls back to the
fixed {0: "tuesday", 1: "wednesday", 2: "friday"} mapping documented in the handoff
and prints a warning -- silent fallback here would risk mislabeling task order.
"""

import ast
import json

import h5py
import numpy as np

# Matches the {'Benign': 0, 'Malicious': 1} convention used throughout data_loader.py
# and consumed downstream (test_red.py, main.py) via label_mapping['Benign'].
LABEL_MAPPING = {"Benign": 0, "Malicious": 1}

DEFAULT_DAY_MAPPING = {0: "tuesday", 1: "wednesday", 2: "friday"}

# Chronological order of CL task boundaries -- day names not present in a given
# file's day_mapping are simply skipped (e.g. a subsampled fixture with one day).
_TASK_ORDER = ["tuesday", "wednesday", "friday"]


def _parse_day_mapping(raw_attr):
    """Best-effort parse of the day_mapping file attribute into {int: str}."""
    if raw_attr is None:
        print("[h5_data_loader] WARNING: 'day_mapping' attr missing, "
              f"falling back to default {DEFAULT_DAY_MAPPING}.")
        return dict(DEFAULT_DAY_MAPPING)

    # Already dict-like (h5py can round-trip some mapping-ish types via numpy void arrays)
    if isinstance(raw_attr, dict):
        return {int(k): str(v) for k, v in raw_attr.items()}

    # JSON or Python-literal string, e.g. '{"0": "tuesday", ...}' or "{0: 'tuesday', ...}"
    if isinstance(raw_attr, (str, bytes)):
        text = raw_attr.decode("utf-8") if isinstance(raw_attr, bytes) else raw_attr
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                return {int(k): str(v) for k, v in parsed.items()}
            except Exception:
                continue
        print(f"[h5_data_loader] WARNING: could not parse day_mapping attr {text!r}, "
              f"falling back to default {DEFAULT_DAY_MAPPING}.")
        return dict(DEFAULT_DAY_MAPPING)

    # Array/list of names, positionally indexed by day id: ["tuesday", "wednesday", "friday"]
    try:
        return {
            i: (name.decode("utf-8") if isinstance(name, bytes) else str(name))
            for i, name in enumerate(raw_attr)
        }
    except TypeError:
        print(f"[h5_data_loader] WARNING: unrecognized day_mapping attr type "
              f"{type(raw_attr)}, falling back to default {DEFAULT_DAY_MAPPING}.")
        return dict(DEFAULT_DAY_MAPPING)


def load_h5_dataset(path):
    """
    Loads an HDF5 dataset built by build_datasets.py and splits it by source_day
    (CL task boundary).

    Returns:
      - day_data: {day_id: {"features": (N,D) float32, "labels": (N,) int8,
                             "timestamp": (N,) float64, "attack_family": (N,) object}}
      - day_mapping: {day_id: day_name}, e.g. {0: "tuesday", 1: "wednesday", 2: "friday"}
      - label_mapping: {"Benign": 0, "Malicious": 1}
    """
    with h5py.File(path, "r") as f:
        for required in ("features", "label", "source_day"):
            if required not in f:
                raise KeyError(
                    f"'{required}' dataset missing from {path}. "
                    f"Found datasets: {list(f.keys())}"
                )

        day_mapping = _parse_day_mapping(f.attrs.get("day_mapping"))

        features = f["features"][:].astype(np.float32)
        labels = f["label"][:].astype(np.int8)
        source_day = f["source_day"][:].astype(np.int8)
        timestamp = f["timestamp"][:] if "timestamp" in f else None
        # h5py returns fixed-length/vlen string datasets as bytes regardless of the
        # UTF-8 encoding used on write; decode here so callers get plain str.
        attack_family = (
            np.array([v.decode("utf-8") if isinstance(v, bytes) else v
                       for v in f["attack_family"][:]])
            if "attack_family" in f else None
        )

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels {{0, 1}} (benign/malicious) in {path}, "
            f"found {sorted(unique_labels)}."
        )

    day_data = {}
    for day_id in sorted(set(source_day.tolist())):
        mask = source_day == day_id
        day_data[day_id] = {
            "features": features[mask],
            "labels": labels[mask],
            "timestamp": timestamp[mask] if timestamp is not None else None,
            "attack_family": attack_family[mask] if attack_family is not None else None,
        }

    missing = set(day_data) - set(day_mapping)
    if missing:
        raise ValueError(
            f"source_day values {sorted(missing)} present in {path} have no entry "
            f"in day_mapping {day_mapping}."
        )

    return day_data, day_mapping, LABEL_MAPPING


def get_task_order(day_mapping):
    """CL task boundary order = chronological day order (tuesday, wednesday, friday),
    restricted to whichever days are actually present in day_mapping."""
    name_to_id = {name: day_id for day_id, name in day_mapping.items()}
    return [name_to_id[name] for name in _TASK_ORDER if name in name_to_id]


def get_balanced_batch_from_index_h5(features, labels, batch_size, start_index,
                                     class_0_sampling_prob=0.1):
    """
    Numpy-native equivalent of data_loader.get_balanced_batch_from_index, for use
    against the (N,D) arrays returned by load_h5_dataset instead of a features
    DataFrame. Same selection semantics (all malicious up to quota, benign sampled
    probabilistically, batch kept in original temporal order), but indexes a numpy
    array directly instead of features_df.iloc[i] since h5-loaded features have no
    DataFrame wrapper (and no column names to preserve).

    Returns (batch_features, batch_labels, batch_indices, end_index), matching the
    CSV-loader function's contract -- batch_features is already a plain ndarray
    (no `.values` conversion needed downstream, unlike the DataFrame-returning
    original).
    """
    class_1_quota = batch_size
    class_1_count = 0
    batch_data = []

    i = start_index
    n = len(labels)
    while i < n:
        label = labels[i]

        if label == 1 and class_1_count < class_1_quota:
            batch_data.append((i, features[i], label))
            class_1_count += 1
        elif label == 0 and np.random.rand() < class_0_sampling_prob:
            batch_data.append((i, features[i], label))

        if class_1_count == class_1_quota and len(batch_data) >= batch_size:
            break

        i += 1

    if class_1_count < class_1_quota:
        print("Malicious samples left: ", class_1_count)
        return None, None, None, i

    batch_data.sort(key=lambda x: x[0])

    batch_indices = [idx for idx, _, _ in batch_data]
    batch_features = np.stack([row for _, row, _ in batch_data])
    batch_labels = np.array([label for _, _, label in batch_data], dtype=np.int8)
    end_index = i + 1

    return batch_features, batch_labels, batch_indices, end_index


def load_pooled_chronological_tasks(path, task_fractions):
    """
    Alternative to the day-boundary split in load_h5_dataset: pools every day's
    records together, sorts the pool by timestamp, and partitions it into
    len(task_fractions) synthetic CL tasks by row-count fraction (fractions need
    not be equal -- e.g. a large warm-start task 0 followed by a taper). Day
    identity is retained per-row as metadata only, NOT as the task boundary.

    Args:
      path: path to an HDF5 file matching the load_h5_dataset schema.
      task_fractions: sequence of positive floats describing each task's share
        of the pooled row count, in task order. Renormalized to sum to 1.0 (so
        callers don't have to hand-tune values to cancel out rounding).

    Returns:
      - tasks: list of dicts, one per task index (0..len(task_fractions)-1):
          {"features": (n,D) float32, "labels": (n,) int8, "timestamp": (n,) float64,
           "source_day": (n,) int8, "attack_family": (n,) object or None}
      - day_mapping: {day_id: day_name}, unchanged from load_h5_dataset (informational).
      - label_mapping: {"Benign": 0, "Malicious": 1}
    """
    day_data, day_mapping, label_mapping = load_h5_dataset(path)

    fractions = np.asarray(task_fractions, dtype=np.float64)
    if len(fractions) == 0 or (fractions <= 0).any():
        raise ValueError(f"task_fractions must be all-positive, got {task_fractions}")
    fractions = fractions / fractions.sum()

    day_ids_sorted = sorted(day_data.keys())
    features = np.concatenate([day_data[d]["features"] for d in day_ids_sorted])
    labels = np.concatenate([day_data[d]["labels"] for d in day_ids_sorted])
    source_day = np.concatenate(
        [np.full(len(day_data[d]["labels"]), d, dtype=np.int8) for d in day_ids_sorted]
    )

    has_ts = all(day_data[d]["timestamp"] is not None for d in day_ids_sorted)
    if not has_ts:
        raise ValueError(
            f"{path} is missing a 'timestamp' dataset for at least one day; "
            f"load_pooled_chronological_tasks requires timestamps to sort the pool."
        )
    timestamp = np.concatenate([day_data[d]["timestamp"] for d in day_ids_sorted])

    has_fam = all(day_data[d]["attack_family"] is not None for d in day_ids_sorted)
    attack_family = (
        np.concatenate([day_data[d]["attack_family"] for d in day_ids_sorted])
        if has_fam else None
    )

    order = np.argsort(timestamp, kind="stable")
    features, labels, source_day, timestamp = (
        features[order], labels[order], source_day[order], timestamp[order]
    )
    if attack_family is not None:
        attack_family = attack_family[order]

    n_total = len(labels)
    counts = np.round(fractions * n_total).astype(int)
    counts[-1] = n_total - counts[:-1].sum()  # last task absorbs rounding drift
    if (counts <= 0).any():
        raise ValueError(
            f"Computed a non-positive task size {counts.tolist()} for n_total={n_total} "
            f"rows and task_fractions={task_fractions}; dataset is too small for this "
            f"many tasks, or fractions need adjusting."
        )
    boundaries = np.cumsum(counts)
    starts = np.concatenate([[0], boundaries[:-1]])

    tasks = []
    for s, e in zip(starts, boundaries):
        tasks.append({
            "features": features[s:e],
            "labels": labels[s:e],
            "timestamp": timestamp[s:e],
            "source_day": source_day[s:e],
            "attack_family": attack_family[s:e] if attack_family is not None else None,
        })
    return tasks, day_mapping, label_mapping


def day_boundary_batch_iterator(day_data, day_mapping, batch_size,
                                class_0_sampling_prob=0.1):
    """
    Generator walking every CL task (day) in chronological order, yielding balanced
    batches within each day via get_balanced_batch_from_index_h5. Signals task
    boundaries via `is_new_task` so a driver loop (e.g. a future multi-day main.py)
    can snapshot/retrain the red agent per decision #1 in the handoff.

    Yields dicts:
      {"day_id": int, "day_name": str, "is_new_task": bool,
       "features": (batch,D) float32, "labels": (batch,) int8,
       "indices": list[int], "start_index": int, "end_index": int}

    A day with too few malicious samples left to fill one more batch is skipped
    (moves on to the next day) rather than stopping iteration entirely, since day
    boundaries -- not exhausting every last sample -- are what matters for CL.
    """
    for day_id in get_task_order(day_mapping):
        features = day_data[day_id]["features"]
        labels = day_data[day_id]["labels"]
        day_name = day_mapping[day_id]

        start_index = 0
        is_new_task = True
        while start_index < len(labels):
            batch_features, batch_labels, batch_indices, end_index = \
                get_balanced_batch_from_index_h5(
                    features, labels, batch_size, start_index, class_0_sampling_prob
                )

            if batch_features is None:
                break

            yield {
                "day_id": day_id,
                "day_name": day_name,
                "is_new_task": is_new_task,
                "features": batch_features,
                "labels": batch_labels,
                "indices": batch_indices,
                "start_index": start_index,
                "end_index": end_index,
            }

            is_new_task = False
            start_index = end_index
