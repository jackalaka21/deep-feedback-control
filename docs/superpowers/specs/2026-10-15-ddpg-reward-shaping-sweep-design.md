# DDPG Baseline and Reward-Shaping Sweep: Design Spec

Date: 2026-10-15 (revised the same day after the user chose extra scope: robustness link, statistics, success rate, housekeeping)
Extends: `2026-10-14-ai-engineering-portfolio-design.md` (Project 1, `deep-feedback-control`). Project 1 only.

## 1. Goal and success criteria

Add a DDPG baseline next to TD3 and a reward-shaping sweep that measures how the *training reward* design affects (a) learning stability and (b) **robustness to the IPQC failure modes**, across algorithms and seeds, with honest statistics.

Success:
- `train.py` trains either `td3` or `ddpg` with identical hyperparameters (lr 1e-3, batch 256, action-noise sigma 0.1).
- `scripts/sweep.py` runs the full grid (2 algos x 5 reward variants x 5 seeds = 50 runs), in parallel and resumably, writing per-run evaluation curves, final models and TensorBoard logs.
- Analysis produces per-seed learning curves with bootstrap confidence bands, a stability table with bootstrap CIs, a success-rate metric, and an algorithm-difference table.
- `scripts/shaping_robustness.py` stress-tests every solved final policy and answers: does the training reward change how the policy copes with deadband, degradation and latency?
- `notebooks/03_reward_shaping_sweep.ipynb` presents all of it. The README reports what was actually found, including negative results.
- The deferred minor review findings from the first review (section 8) are fixed.
- All tests pass; the full sweep and the robustness test are actually run.

Non-goals: new tasks (push, slide, 7-DOF), SAC, HER, TD3-component ablations, learning-rate or exploration-noise sweeps, GPU training for the sweep, git-history rewriting.

## 2. Algorithms

- Config gains `algo: td3` (defaults to `td3` when absent, so old configs keep working) and `ddpg_model_path: models/reacher_ddpg`; `success_radius: 0.05` (metres, shared by the `sparse` variant and the success-rate metric).
- `train(cfg, ..., algo=None)` and `--algo {td3,ddpg}` select the SB3 class. Both use `MlpPolicy` and receive the same lr, batch size, buffer size, `learning_starts`, action noise and seed. An unknown algo name raises a clear error.
- `evaluate_ipqc.load_policy(model_path, device="cpu", algo="td3")` can load either algorithm, so the stress harness works for DDPG policies too.

## 3. Reward shaping (`envs/reward_shaping.py`)

`RewardShaping(env, variant, success_radius=0.05)` is a `gym.Wrapper` that replaces the *training* reward using the base env's `info["reward_dist"]` (= minus distance to target, so `d = -reward_dist >= 0`) and `info["reward_ctrl"]` (= minus squared action). It raises a clear error if those keys are missing (non-Reacher env) or if the variant name is unknown.

| Variant | Training reward per step | What it probes |
|---|---|---|
| `default` | `reward_dist + reward_ctrl` (unchanged) | the reference |
| `dist_only` | `reward_dist` | removing the control penalty |
| `sparse` | `0` if `d <= success_radius` else `-1` | sparse reward, no shaping signal |
| `progress` | `10 * (distance before the step - distance after the step) + reward_ctrl`, with the distance read from the simulator state (`get_body_com`), starting from the reset state | potential-based progress shaping (gamma = 1). *Amended after review:* the first version used `info["reward_dist"]`, which Reacher-v4 computes before the simulation step, so each action was credited with the previous action's movement. |
| `scaled_x10` | `10 * (reward_dist + reward_ctrl)` | reward magnitude |

The wrapper changes only what the agent trains on. Observations, termination and `info` pass through.

## 4. Fair evaluation

Every run is evaluated on a separate **unshaped** environment using the task metric `distance_return` (= sum of `reward_dist`, minus summed distance; higher is better). This avoids the control-cost trap found in the IPQC analysis (portfolio spec amendment 7.6): shaped training returns are not comparable across variants.

`run_episode_parts` additionally returns `final_distance` (distance to target at the episode's last step). **Success** = `final_distance <= success_radius`; `success_rate` = fraction of evaluation episodes that succeed. (The IPQC sweep DataFrame gains the extra `final_distance` column; scores are unaffected.)

Evaluation protocol: at step 0 and then every `eval_every` steps (2000), run `eval_episodes` (10) deterministic episodes; the first episode of each evaluation is seeded with a fixed eval seed so targets are identical across evaluations and runs. Record `timesteps`, `mean_distance_return`, `std_distance_return`, `mean_total_return`, `success_rate`.

## 5. Sweep runner (`scripts/sweep.py`, `configs/sweep_config.yaml`)

Config: `algos: [td3, ddpg]`, `variants: [default, dist_only, sparse, progress, scaled_x10]`, `seeds: [0, 1, 2, 3, 4]`, `timesteps: 40000`, `learning_starts: 5000`, `eval_every: 2000`, `eval_episodes: 10`, `workers: 6`, `solve_threshold: -4.0`, `runs_dir: runs`, plus a `robustness:` block (section 7).
(`-4.0` corresponds to about 0.87 retention against the random policy `-9.78` and the trained TD3 `-3.11` measured in the IPQC analysis; configurable.)

- Grid = algos x variants x seeds (50 runs).
- Each run trains on **CPU, single-threaded** (measured: a gradient step costs about 13 ms on CPU and on CUDA alike, because these tiny networks are overhead-bound, so the GPU gives no speedup). Runs execute in a `ProcessPoolExecutor` (spawn context, `workers` processes).
- Outputs per run in `runs/<algo>_<variant>_seed<k>/`: `eval.csv`, `final_model.zip`, TensorBoard event files (SB3 `tensorboard_log`), and `done.json` written last, containing algo, variant, seed, steps, wall-clock seconds, library versions (python, torch, stable-baselines3, gymnasium, mujoco) and the git commit. A run with `done.json` is skipped on re-launch, so an interrupted sweep resumes.
- A random-policy baseline over the eval seeds is computed once and stored in `runs/random_baseline.json`.
- CLI: `python scripts/sweep.py [--config] [--workers N] [--quick]`; `--quick` shrinks the grid and budget for smoke tests and writes to `runs/quick/` (git-ignored). Prints one progress line per finished run.
- Budget estimate (unverified until the throughput benchmark): 50 runs x 40k steps x ~22 ms = ~12 CPU-hours, so ~2-2.5 hours wall-clock with 6 workers.
- W&B is not used by the sweep (TensorBoard + CSV only).

## 6. Learning-stability analysis (`scripts/sweep_analysis.py`)

Per run, from `eval.csv`:
- `final_return`: mean of the last 3 evaluations. `final_success`: same for `success_rate`.
- `auc`: mean return over all evaluations (sample efficiency).
- `steps_to_threshold`: first evaluation step with mean return >= `solve_threshold` (NaN if never).
- `max_drawdown`: largest drop from a running maximum (instability), computed only over evaluations after `learning_starts` (*amended after review:* evaluations before learning starts score the same untrained policy, and the first evaluation after learning begins often dips, which is not instability during learning).

Per (algo, variant), across seeds: mean of each metric with a **95% bootstrap CI** (`bootstrap_ci`: resample seeds with replacement, 10,000 draws, percentile interval; crude with 5 seeds, and said so), and the number of seeds that reached the threshold.
`compare_algos` gives, per variant, the TD3 minus DDPG difference in `final_return` with a bootstrap CI (unpaired resampling of each algo's seeds).

Outputs: `plots/sweep_summary.csv`, `plots/sweep_algo_differences.csv`, `plots/sweep_eval_curves.csv` (all evaluations from all runs, small, committed so notebook 03 renders without re-running) and PNGs: learning curves per variant with **thin per-seed lines plus mean and bootstrap band** (both algorithms overlaid, random-baseline line), and a final-return / success-rate / drawdown comparison. Raw `runs/` are git-ignored.

## 7. Robustness of shaped policies (`scripts/shaping_robustness.py`)

Question: does the reward a policy was trained on change its safety boundaries?

- Loads each run's `final_model.zip` (algo from `done.json`) and evaluates it under a reduced IPQC battery from the `robustness:` config block: `episodes: 10`, `eval_seed: 2000`, `degradation: [1.0, 0.6, 0.4, 0.2]`, `latency_steps: [0, 2, 4]`, `deadband: [0.0, 0.1, 0.2, 0.3]` (levels chosen around the boundaries measured for the default TD3 policy), reusing `make_stressed_env` and `run_episode_parts`. Evaluation uses the unshaped task metric.
- Retention uses the same formula as the IPQC analysis, with each policy's *own* nominal return and the random-policy distance return. A policy is **eligible** only if its own nominal return is at least `solve_threshold`; ineligible policies (for example, a `sparse` run that never learned) are listed and excluded, never scored.
- Parallelised with the same process pool; about 12 conditions x 10 episodes per policy (a few minutes in total).
- Outputs: `plots/shaping_robustness.csv` (per policy x condition), `plots/shaping_robustness_summary.csv` (per algo x variant x stress x level: mean retention with bootstrap CI over eligible seeds, `n_eligible`), `plots/shaping_boundaries.csv` (safety boundary per eligible policy via `safety_boundary`, summarised per algo x variant) and PNGs of retention versus stress level (rows = algo, columns = stress type, lines = variant).

## 8. Housekeeping (the deferred minor findings from the first final review)

- `--config` resolves against the project root like the other paths.
- The CUDA-unavailable message also mentions setting `device: cpu` in the config.
- The IPQC sweep runs the random and nominal conditions first and fails fast if nominal is not better than random (instead of after the whole sweep); `summarize` raises a clear error when a baseline row is missing.
- `load_config` validates: stress lists non-empty, `learning_rate` is a float, `algo` is `td3` or `ddpg`.
- `--wandb` becomes `--wandb/--no-wandb` (`BooleanOptionalAction`), so it can switch W&B off.
- `requirements.txt` pins versions known to work (as installed now), and adds a Jupyter frontend and `tensorboard`.
- README: cross-platform setup commands; states that `training_curve.png` is produced by notebook 01; notebook 01's "10-20 minutes" text corrected to the measured time.
- Not included: confidence intervals for the original IPQC analysis, latency-level print format, deprecation warnings stored in notebook outputs, git-history rewrite.

## 9. Tests

- Reward variants: exact reward values per variant on a stub env that supplies `info` keys, including the `progress` first-step rule and reset behaviour; missing-key and unknown-variant errors.
- DDPG: construction and a short CPU training smoke test; `algo` default and CLI selection; invalid algo error; `load_policy` round trip for DDPG.
- Evaluation callback: expected number of evaluations at expected steps, uses the unshaped metric even when training with a shaped reward, `success_rate` and `final_distance` correct.
- Sweep: grid enumeration; resumability (finished run skipped, unfinished re-run); `done.json` contents; a tiny end-to-end sweep (2 algos x 2 variants x 1 seed, ~1,500 steps) through the process pool.
- Metrics on synthetic curves: final return, AUC, steps-to-threshold (reached and never), drawdown; `bootstrap_ci` (deterministic with a seed, contains the mean for a constant sample); table aggregation; `compare_algos`.
- Robustness: eligibility filtering (ineligible policy excluded, not scored); retention uses the policy's own nominal; a tiny end-to-end run on two quick-sweep models.
- Housekeeping: each new validation and error message above.

## 10. Honest limitations (go in the README)

- 5 seeds and 40k steps per run: enough to see large differences, not small ones. Bootstrap CIs from 5 seeds are crude; there is no formal significance testing.
- One task (Reacher). Conclusions about reward shaping are about this task.
- The variants are my choices, and `sparse` may simply fail to learn in 40k steps; that is a result, not a bug, and such policies are excluded from the robustness comparison.
- The robustness battery uses a reduced set of stress levels and 10 episodes per condition.
- Any explanation of *why* a variant behaves as it does is a hypothesis unless an experiment tested it.
- Throughput numbers are for this laptop; the GPU is not used in the sweep.

## 11. Verification plan

- `pytest` passes (existing 39 plus the new tests).
- Benchmark parallel throughput first (workers vs steps/s) to confirm or correct the time estimate.
- Quick sweep smoke test, then the full 50-run sweep, resumable, then the robustness run, then regenerate the analysis and notebook 03.
- Check README claims against the CSVs before committing. Report exactly what was and was not run. Do not push without asking.

## 12. Amendments after the final review (2026-10-15)

1. `progress` reward measured from the simulator state after the step (section 3); the 10 `progress` runs were re-run. It still failed, so the original misalignment was not the main cause.
2. Drawdown computed after `learning_starts` (section 6). The earlier all-evaluation version reversed the TD3-vs-DDPG stability comparison.
3. `done.json` and `random_baseline.json` record the evaluation/training protocol; the sweep refuses to resume next to finished runs with a different protocol, and recomputes a baseline whose protocol changed. Runs finished before protocols were recorded are accepted.
4. Boundary summaries report the median (levels are ordinal and censored) next to the mean.
5. Scope note: all results use `Reacher-v4`. Its reward is computed before the action is applied, so the distance metrics and the success check use the pre-action distance. A port to `Reacher-v5` (reward after the action) is a planned follow-up, with results kept separate.
