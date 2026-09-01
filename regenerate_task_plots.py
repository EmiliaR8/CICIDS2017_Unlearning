#!/usr/bin/env python3
"""
Standalone: regenerate task_metrics.png and unlearning_metrics.png from an
already-saved results JSON, without re-running the pipeline.

Use this after a crash that happens AFTER the results JSON is written but
during/after plotting -- e.g. the plot_task_metrics KeyError this script's
sibling commit fixed. Results are written to <log_name>.json before any
plotting runs, so a crash there never loses the underlying per-task data,
only the plots.

LIMITATION: this can only regenerate the two plots built purely from
`results` (task_metrics.png, unlearning_metrics.png). The other three
end-of-run plots -- prototype_heatmap.png, episode_clouds.png, and
decision_boundary_evolution.png -- are built from runtime objects (the
contrastive bank, and the per-checkpoint decision-boundary grids) that are
never written to the results JSON, so they cannot be recovered once the
process that held them has exited. The per-task decision_boundary_task*.png
files ARE already saved individually during the run itself, well before
this final summary step -- only the run-wide *_evolution.png montage and
the two bank-based plots are actually unrecoverable without a fresh run.

Usage:
    python regenerate_task_plots.py /path/to/madar_unlearn_cl_run.json [--out-dir /path/to/plots]

If --out-dir is omitted, writes into a "plots" subdirectory next to the
JSON file (matching where the original run would have put them).
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", help="Path to the saved <log_name>.json from a madar_unlearning_cl_pipeline.py run")
    ap.add_argument("--out-dir", default=None,
                     help="Directory to write task_metrics.png/unlearning_metrics.png into "
                          "(default: a 'plots' folder next to the JSON file)")
    args = ap.parse_args()

    if not os.path.isfile(args.results_json):
        print(f"error: {args.results_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.results_json) as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded {len(results)} tasks from {args.results_json}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.results_json)), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Import the pipeline module directly -- its main() is guarded behind
    # `if __name__ == "__main__":`, so importing it only defines functions/
    # constants and doesn't run a training pipeline. This keeps the plot
    # logic byte-identical to what the real run would have produced,
    # instead of a separate reimplementation that could drift out of sync.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import madar_unlearning_cl_pipeline as pipeline

    task_metrics_path = os.path.join(out_dir, "task_metrics.png")
    pipeline.plot_task_metrics(results, task_metrics_path)
    print(f"Wrote {task_metrics_path}")

    unlearning_metrics_path = os.path.join(out_dir, "unlearning_metrics.png")
    pipeline.plot_unlearning_metrics(results, unlearning_metrics_path)
    print(f"Wrote {unlearning_metrics_path}")

    print(
        "\nNote: prototype_heatmap.png, episode_clouds.png, and "
        "decision_boundary_evolution.png cannot be regenerated from the "
        "results JSON -- they're built from runtime objects (the "
        "contrastive bank, per-checkpoint boundary grids) that were never "
        "serialized. The per-task decision_boundary_task*.png files from "
        "during the run should already be on disk from earlier in the run, "
        "though -- only the run-wide summary/montage versions are lost."
    )


if __name__ == "__main__":
    main()
