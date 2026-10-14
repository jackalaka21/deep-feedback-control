# DDPG Baseline and Reward-Shaping Sweep: Design Spec

Date: 2026-10-15
Extends: `2026-10-14-ai-engineering-portfolio-design.md` (Project 1, `deep-feedback-control`). Project 1 only.

## 1. Goal and success criteria

Add a DDPG baseline next to TD3, and a reward-shaping sweep that measures how the *training reward* design affects learning stability, across algorithms and seeds.

Success:
- `train.py` can train either `td3` or `ddpg` with identical hyperparameters (lr 1e-3, batch 256, action-noise sigma 0.1).
- `scripts/sweep.py` runs the full grid, in parallel and resumably, and writes per-run evaluation curves plus TensorBoard logs.
- Analysis produces learning-curve plots and a stability table (CSV), and `notebooks/03_reward_shaping_sweep.ipynb` presents them.
- The README reports what the sweep actually found, including negative results, with the limitations below.
- All tests pass; the full sweep is actually run.

Non-goals: new tasks (push, slide, 7-DOF), extending the IPQC stress harness to DDPG, HER, GPU training for the sweep.

## 2. Algorithms

- Config gains `algo: td3` (default `td3` when absent, so old configs keep working) and `ddpg_model_path: models/reacher_ddpg`.
- `train(cfg, ..., algo=None)` and `--algo {td3,ddpg}` select the SB3 class. Both use `MlpPolicy` and receive the same lr, batch size, buffer size, `learning_starts`, action noise and seed.
- Unchanged: the IPQC evaluation stays TD3-only.

## 3. Reward shaping (`envs/reward_shaping.py`)

`RewardShaping(env, variant)` is a `gym.Wrapper` that replaces the *training* reward using the base env's `info["reward_dist"]` (= minus distance to target, so `d = -reward_dist >= 0`) and `info["reward_ctrl"]` (= minus squared action). It raises a clear error if those keys are missing (non-Reacher env).

| Variant | Training reward per step | What it probes |
|---|---|---|
| `default` | `reward_dist + reward_ctrl` (unchanged) | the reference |
| `dist_only` | `reward_dist` | removing the control penalty |
| `sparse` | `0` if `d <= 0.05` else `-1` | sparse reward, no shaping signal |
| `progress` | `10 * (d_prev - d) + reward_ctrl`; `d_prev` is the previous step's distance, and the first step of an episode gets progress 0 | potential-based progress shaping (gamma = 1) |
| `scaled_x10` | `10 * (reward_dist + reward_ctrl)` | reward magnitude |

The wrapper changes only what the agent trains on. Observations, termination and `info` pass through.

## 4. Fair evaluation

Every run is evaluated on a separate **unshaped** environment using the task metric `distance_return` (= sum of `reward_dist`, i.e. minus summed distance; higher is better). This avoids the control-cost trap found in the IPQC analysis (spec amendment 7.6 of the portfolio spec): shaped training returns are not comparable across variants.

Evaluation protocol: at step 0 and then every `eval_every` steps (2000), run `eval_episodes` (10) deterministic episodes; the first episode of each evaluation is seeded with a fixed eval seed so the targets are identical across evaluations and runs. Record `timesteps`, `mean_distance_return`, `std_distance_return`, `mean_total_return`.

## 5. Sweep runner (`scripts/sweep.py`, `configs/sweep_config.yaml`)

Config: `algos: [td3, ddpg]`, `variants: [default, dist_only, sparse, progress, scaled_x10]`, `seeds: [0, 1, 2]`, `timesteps: 40000`, `learning_starts: 5000`, `eval_every: 2000`, `eval_episodes: 10`, `workers: 6`, `solve_threshold: -4.0`, `runs_dir: runs`.
(`-4.0` corresponds to about 0.87 retention against the random policy `-9.78` and the trained TD3 `-3.11` measured in the IPQC analysis; it is configurable.)

- Grid = algos x variants x seeds (30 runs).
- Each run trains on **CPU, single-threaded** (measured: a gradient step costs about 13 ms on CPU and on CUDA alike, because these tiny networks are overhead-bound, so the GPU gives no speedup). Runs execute in a `ProcessPoolExecutor` (spawn context, `workers` processes).
- Outputs per run in `runs/<algo>_<variant>_seed<k>/`: `eval.csv`, TensorBoard event files (SB3 `tensorboard_log`), and `done.json` written last. A run with `done.json` is skipped on re-launch, so an interrupted sweep resumes.
- A random-policy baseline over the same eval seeds is computed once and stored in `runs/random_baseline.json`.
- CLI: `python scripts/sweep.py [--config] [--workers N] [--quick]`; `--quick` shrinks the grid and budget for smoke tests and writes to `runs/quick/` (git-ignored). Prints one progress line per finished run.
- W&B is not used by the sweep (TensorBoard + CSV only).

## 6. Analysis (`scripts/sweep_analysis.py`, notebook 03)

Per run, from `eval.csv`:
- `final_return`: mean of the last 3 evaluations.
- `auc`: mean return over all evaluations (sample efficiency).
- `steps_to_threshold`: first evaluation step with mean return >= `solve_threshold` (NaN if never).
- `max_drawdown`: largest drop from a running maximum (instability).

Per (algo, variant), across seeds: mean and std of each metric, and the number of seeds that reached the threshold.

Outputs: `plots/sweep_summary.csv` (the table), `plots/sweep_eval_curves.csv` (all evaluations from all runs, small, committed so notebook 03 renders without re-running), and PNGs: learning curves per variant (mean ± std over seeds, both algorithms overlaid, random-baseline line) and a final-return/drawdown comparison. Raw `runs/` are git-ignored.

Notebook 03 loads `runs/` if present, else the committed `plots/sweep_eval_curves.csv`, and shows the table and plots. It does **not** launch the multi-hour sweep in full mode (it says to run `python scripts/sweep.py`); with `DFC_QUICK=1` it runs a tiny sweep for smoke-testing.

## 7. Tests

- Reward variants: exact reward values for each variant on a stub env that supplies `info` keys, including the first-step rule of `progress` and reset behaviour; missing-key error.
- DDPG: construction and a short CPU training smoke test; `algo` config default and CLI selection; invalid algo name error.
- Evaluation callback: records the expected number of evaluations at the expected steps, and uses the unshaped task metric even when training with a shaped reward.
- Sweep: grid enumeration; resumability (a finished run is skipped, an unfinished one is re-run); a tiny end-to-end sweep (2 algos x 2 variants x 1 seed, ~1,500 steps) through the process pool.
- Metrics on synthetic curves: final return, AUC, steps-to-threshold (reached and never), drawdown; table aggregation across seeds.

## 8. Honest limitations (go in the README)

- 3 seeds and 40k steps per run: enough to see large differences, not small ones; no significance testing.
- One task (Reacher). Conclusions about reward shaping are about this task.
- The variants are my choices, and `sparse` may simply fail to learn in 40k steps; that is a result, not a bug.
- Any explanation of *why* a variant behaves as it does is a hypothesis unless an experiment tested it.
- Throughput numbers are for this laptop; the GPU is not used in the sweep.

## 9. Verification plan

- `pytest` passes (existing 39 plus the new tests).
- Benchmark parallel throughput first (workers vs steps/s) to confirm the time estimate (about 1.5-2 h wall-clock for 30 runs; unverified until measured).
- Quick sweep smoke test, then the full 30-run sweep, resumable.
- Report exactly what was and was not run; do not push without asking.
