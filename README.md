# L2U: Learning What to Unlearn for Continually Adapting Cybersecurity Threat Detectors

Code for the CICIDS 2017 and Tiny ImageNet **poisoned** continual-learning experiments, and for the
rebuttal's informed-baseline comparisons.

## Setup

```bash
pip install numpy scipy scikit-learn xgboost h5py matplotlib pandas torch
```

Data (not included):
- **CICIDS 2017:** the preprocessed HDF5 file `subsampled_dataset.h5` (datasets `features`, `label`,
  `source_day`, `timestamp`). The default path is `./data/subsampled_dataset.h5`; override it with `--h5-path`.
- **Tiny ImageNet:** a directory containing `tiny-imagenet-200/`, passed with `--data-root`.

All pipelines run on CPU. Seeds used for the reported CICIDS results: **130, 131, 132** (`--seed`).
Pass `--no_breakpoint` for non-interactive runs. Outputs are written to `./runs/<pipeline>/<log_name>/`
under the directory you run from (`./imagenet_runs/` for Tiny ImageNet). Pipelines that accept
`--runs_dir` can write elsewhere.

```bash
export H5=./data/subsampled_dataset.h5 RUNS=./runs SEED=130
```

## CICIDS 2017, poisoned setting (10 chronological tasks, 30% of each class poisoned per task)

| Method | Script | Command | Lineage name in the logs |
|---|---|---|---|
| MADAR; Heuristic-L2U (Dropped Rows, Approximate Unlearning, Flip Label) | `madar_pocket_pipeline.py` | `python madar_pocket_pipeline.py --seed $SEED --h5-path $H5 --runs_dir $RUNS --log_name l2u_s$SEED --no_breakpoint` | `poisoned_baseline`; `dropped_rows`, `amnesiac`, `opposite_class` |
| Random forget-set selection | `madar_pocket_pipeline.py` | add `--detector_type random --random_forget_size budget` | as above |
| Logistic-regression detector | `madar_pocket_pipeline.py` | add `--detector_type logistic` | as above |
| SI, A-GEM | `madar_pocket_pipeline_si_agem.py` | `python madar_pocket_pipeline_si_agem.py --seed $SEED --h5-path $H5 --log_name si_agem_s$SEED --no_breakpoint` | `si`, `agem` |
| Naive, Joint | `madar_pocket_pipeline_naive_joint.py` | `python madar_pocket_pipeline_naive_joint.py --seed $SEED --h5-path $H5 --log_name naive_joint_s$SEED --no_breakpoint` | `naive`, `joint` |
| DEDUCE | `madar_pocket_pipeline_deduce.py` | `python madar_pocket_pipeline_deduce.py --seed $SEED --h5-path $H5 --runs_dir $RUNS --log_name deduce_s$SEED --no_breakpoint` | `deduce` |
| SSF | `madar_pocket_pipeline_ssf.py` | `python madar_pocket_pipeline_ssf.py --seed $SEED --h5-path $H5 --runs_dir $RUNS --log_name ssf_s$SEED --no_breakpoint` | `ssf` |
| Replay buffer: clean (oracle-filtered) | `madar_pocket_pipeline_baseline_filtered.py` | `python madar_pocket_pipeline_baseline_filtered.py --seed $SEED --h5-path $H5 --runs_dir $RUNS --log_name buf_clean_s$SEED --no_breakpoint` | `poisoned_baseline_filtered` |
| Replay buffer: unfilled (oracle purge, no refill) | `madar_pocket_pipeline_baseline_purge.py` | `python madar_pocket_pipeline_baseline_purge.py --seed $SEED --h5-path $H5 --log_name buf_purge_s$SEED --no_breakpoint` | `poisoned_baseline_purged` |
| Replay buffer: dirty / purged after fill | `madar_pocket_pipeline.py` | add `--joint_buffer_allow_perturbed` / `--joint_buffer_purge_after_fill` | fix lineages |

Notes:
- `--random_forget_size budget` flags `min(60, class size)` rows per class at random. That is the sizing the
  random-selection ablation was run with. The later default, `matched`, sizes the random set like the
  trained detector's flagged set instead.
- `amnesiac` is reported as *Approximate Unlearning* and `opposite_class` as *Flip Label*.

### Model capacity (non-Meta methods)

Add `--hidden_sizes` to any CICIDS pocket pipeline above:

| Hidden layers | Capacity | `--hidden_sizes` |
|---|---|---|
| 4 | 100% (default) | `1024,512,256,128` |
| 3 | 75% | `512,256,128` |
| 2 | 50% | `64,32` |
| 1 | 25% | `128` |

### Meta-L2U (reference implementation)

`notebooks/task0-9_pipeline_4arch-updatedmeta.ipynb` is the reference Meta-L2U implementation
(meta-learned active-learning detector `meta_al` + test-time adaptation, 4 architectures). The reported
Meta-L2U results were produced by a co-author's run of this notebook outside this repository.
`notebooks/task0_9_pipeline_4arch.py` is the same code as a script, with a CLI and a timing report:

```bash
python notebooks/task0_9_pipeline_4arch.py --h5 $H5 --logs_dir <dir with classifier_checkpoint_task{0..9}.pt> --depth 4 --num_tasks 10 --seed <SEED>
```

- **Seeds:** _TBD_.
- The notebook reads per-task checkpoints (test split, scaler, replay buffer) from an earlier pocket-pipeline
  run, e.g. `madar_pocket_pipeline_ssf.py`, which writes `logs/classifier_checkpoint_task{t}.pt`.
- Its architectures (`--depth 1..4`) are `(512,)`, `(512, 256)`, `(512, 256, 128)`, `(1024, 512, 256, 128)`.

### Tables

```bash
python build_latex_tables.py --logs runs/<pipeline>/<name>_s130/logs/pipeline_log.txt runs/<pipeline>/<name>_s131/logs/pipeline_log.txt runs/<pipeline>/<name>_s132/logs/pipeline_log.txt --seeds 130 131 132 --lineage <lineage> --method-name "<row label>"
```

Columns: Mean Acc (mean over all clean and adversarial test sets at the final task), Task 10 and Task 1
(clean + adversarial accuracy averaged, on that task's test set; Task 1 is task index 1, the first
poisoned task), and macro P/R/F1. `summarize_pipeline_runs.py` and `plot_pipeline_metrics.py` read the
same logs.

## Tiny ImageNet, poisoned setting

```bash
python madar_pocket_pipeline_tinyimagenet.py --seed $SEED --data-root <dir> --log_name tin_l2u_s$SEED --no_breakpoint
```
```bash
python madar_pocket_pipeline_si_agem_tinyimagenet.py --seed $SEED --data-root <dir> --log_name tin_si_agem_s$SEED --no_breakpoint
```
```bash
python madar_pocket_pipeline_deduce_tinyimagenet.py --seed $SEED --data-root <dir> --log_name tin_deduce_s$SEED --no_breakpoint
```

The class-incremental schedule is set with `--task_setup`; use `--cache-root` to cache decoded images.

## Rebuttal: informed baselines on a shared poison stream

The informed variants give a baseline the same poison supervision as L2U: the same detector, the same
240-row annotated set (60 per class × poison-status group), the same flagged set U_t and the same
buffer filtering. The shared stream makes every method train and be scored on identical data.

1. **Producer** (also gives the L2U and MADAR rows). Writes the stream to `$STREAM/seed<seed>/`, about
   6 GB per seed at full data:
   ```bash
   python madar_pocket_pipeline.py --seed $SEED --h5-path $H5 --runs_dir $RUNS --dump_stream $STREAM --log_name fulldata_l2u_s$SEED --no_breakpoint
   ```
   Add `--data_fraction 0.5` for the 50%-data runs (stratified, timestamp-ordered, fixed subset seed).
2. **Baselines on the shared stream:**
   ```bash
   python madar_pocket_pipeline_deduce.py --seed $SEED --runs_dir $RUNS --load_stream $STREAM --log_name fulldata_deduce_stream_s$SEED --no_breakpoint
   ```
   ```bash
   python madar_pocket_pipeline_deduce.py --seed $SEED --runs_dir $RUNS --load_stream $STREAM --informed_forget --deduce_delta 0.003 --log_name fulldata_informed_stream_s$SEED --no_breakpoint
   ```
   ```bash
   python madar_pocket_pipeline_ssf.py --seed $SEED --runs_dir $RUNS --load_stream $STREAM --log_name fulldata_ssf_stream_s$SEED --no_breakpoint
   ```
   ```bash
   python madar_pocket_pipeline_ssf.py --seed $SEED --runs_dir $RUNS --load_stream $STREAM --informed_forget --log_name fulldata_issf_stream_s$SEED --no_breakpoint
   ```
   ```bash
   python madar_pocket_pipeline_baseline_filtered.py --seed $SEED --runs_dir $RUNS --load_stream $STREAM --filter_source detector --log_name fulldata_imadar_stream_s$SEED --no_breakpoint
   ```
3. **Independent poison:** the same DEDUCE and SSF commands with `--h5-path $H5` instead of
   `--load_stream $STREAM`, and `*_indep_*` log names. Each run then crafts its own poison and attacks;
   informed runs train the same detector with the same budget on their own data.
4. **δ for informed DEDUCE**, chosen on a validation split carved from training rows (seed 130):
   ```bash
   python madar_pocket_pipeline_deduce.py --seed 130 --runs_dir $RUNS --load_stream $STREAM --informed_forget --val_fraction 0.1 --deduce_delta 0.003 --log_name val_inf_d0.003_s130 --no_breakpoint
   ```
   The grid was {0.001, 0.003, 0.01, 0.05}; 0.003 was selected.
5. **Tables.** The script checks that all runs share the same poison and U_t (sha256) before printing.
   Add `--no_stream_check` for the independent-poison runs.
   ```bash
   python build_rebuttal_tables.py --seeds 130 131 132 --l2u "$RUNS/madar_pocket_pipeline/fulldata_l2u_s{seed}/logs/rebuttal_metrics.json" --madar_filtered "$RUNS/madar_pocket_baseline_filtered/fulldata_imadar_stream_s{seed}/logs/rebuttal_metrics.json" --deduce "$RUNS/madar_pocket_deduce/fulldata_deduce_stream_s{seed}/logs/rebuttal_metrics.json" --informed "$RUNS/madar_pocket_deduce/fulldata_informed_stream_s{seed}/logs/rebuttal_metrics.json" --ssf "$RUNS/madar_pocket_ssf/fulldata_ssf_stream_s{seed}/logs/rebuttal_metrics.json" --informed_ssf "$RUNS/madar_pocket_ssf/fulldata_issf_stream_s{seed}/logs/rebuttal_metrics.json" --out_csv rebuttal.csv
   ```

`--num_tasks 2` runs a 2-task smoke test in `madar_pocket_pipeline.py`, `_deduce.py`, `_ssf.py` and
`_baseline_filtered.py`.

## Legacy code

`madar_cl_pipeline.py`, `madar_unlearning_cl_pipeline.py`, `naive_cl_pipeline.py`, `joint_cl_pipeline.py`,
`adversary_env.py` (earlier red-agent pipeline) and `Static_Data_Pipeline/` (earlier static pipeline and
meta-trained forget-set selector) are kept for reference. `runs/` and `comparative_plots/` hold outputs
of those earlier pipelines.

## Not in this repository

- Natural-drift experiments (EMBER, LAMDA, Tiny ImageNet class-incremental natural drift).
- The GNB buffer-exclusion experiment.
- The EMBER 2018 → 2024 Meta-L2U transfer experiment.
