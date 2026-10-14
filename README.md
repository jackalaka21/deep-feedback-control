# Deep Feedback Control

A TD3 policy learns to drive a 2-joint MuJoCo arm (`Reacher`) to a random target, and an **IPQC-style stress-testing harness** then breaks it on purpose: delayed sensors, sticky actuators and worn motors. The result is a measured *safety boundary* for each failure mode, not just a reward curve.

Everything runs headless on one laptop GPU (RTX 3070 Ti, 8 GB). The work is available as scripts and as two Jupyter notebooks that call the same code.

## Results

Trained for 150k steps (about 37 minutes wall-clock on the RTX 3070 Ti). Deterministic evaluation over 60 episodes (3 seeds x 20) per condition. Reacher's reward is `-distance to target` plus `-(action)^2` (control cost); both parts are reported:

| Policy | Total return | Task part (-distance) | Control-cost part |
|---|---|---|---|
| Random actions | -43.35 | -9.78 | -33.57 |
| **Trained TD3, no stress** | **-3.82** (std 1.56) | **-3.11** | -0.72 |

![Training curve](plots/training_curve.png)

The training curve is noisier and lower than the evaluation score because training adds exploration noise to every action.

### Safety boundaries

Each stress level gets a **retention score** on the *task* measure: `(R - R_random) / (R_nominal - R_random)`, where `R` is the summed negative distance to the target. 1.0 means as good as the healthy robot, 0.0 means no better than random at reaching the target, and negative means worse than random. The **safety boundary** is the most severe level reached before the score first drops below 0.9.

| Failure mode | What it simulates | Safety boundary | Retention at the edge |
|---|---|---|---|
| Observation latency | slow or delayed sensors | **safe up to 3 steps**, fails at 4 | 0.981 at 3 steps, 0.778 at 4 |
| Actuator deadband | static friction: commands below a threshold do nothing | **safe up to 0.1**, fails at 0.2 | 0.922 at 0.1, 0.675 at 0.2 |
| Motor degradation | delivered torque is a fraction of the command | **safe down to 60% torque**, fails at 40% | 0.911 at 60%, 0.807 at 40% |

What the measurements show:
- **Latency:** the arm still reaches the target through 3 steps of delay (0.981), but its control effort rises sharply: the control-cost part goes from -0.72 (no delay) to -3.12 at 3 steps and -7.57 at 4. Past 3 steps reaching degrades (0.778 at 4, 0.553 at 5).
- **Deadband:** the most fragile mode at the levels tested. The playbook's 0.05 costs little (0.984), but 0.2 already drops retention to 0.675, and at 0.5 the arm is worse than a random policy at reaching the target (-0.201).
- **Degradation:** graceful. 0.911 at 60% torque, 0.595 even at 20%.
- Levels are in different units for each stress, so the boundaries are not a ranking of which failure is "worse".

| Latency | Deadband | Degradation |
|---|---|---|
| ![latency](plots/latency_curve.png) | ![deadband](plots/deadband_curve.png) | ![degradation](plots/degradation_curve.png) |

The full table, including total-return scores and mean control cost for every level, is in [`plots/ipqc_results.csv`](plots/ipqc_results.csv).

### Why the score ignores control cost

An earlier version of this analysis scored the full reward and concluded the policy was far more robust than it is. The control-cost term is charged on the action the motors actually receive, so deadband and degradation *reduce* it, and a random policy's deficit is 77% control cost. Scoring total return flatters those stresses badly:

| Condition | Retention on total return | Retention on task (used here) |
|---|---|---|
| Deadband 0.3 | 0.895 | 0.302 |
| Deadband 0.5 | 0.815 | **-0.201** (worse than random) |
| Degradation 20% torque | 0.947 | 0.595 |

Both scores are kept in the CSV (`total_score` and `score`); the boundaries use `score`.

## How it works

| Piece | File |
|---|---|
| Stress wrappers (`ObservationLatency`, `ActuatorDeadband`, `MotorDegradation`) and env factories | [`envs/stress_wrappers.py`](envs/stress_wrappers.py) |
| TD3 training, YAML config, optional Weights & Biases logging | [`scripts/train.py`](scripts/train.py), [`configs/td3_config.yaml`](configs/td3_config.yaml) |
| Stress sweep, retention scoring, safety boundaries, plots | [`scripts/evaluate_ipqc.py`](scripts/evaluate_ipqc.py) |
| Walkthrough: train and inspect | [`notebooks/01_train_td3.ipynb`](notebooks/01_train_td3.ipynb) |
| Walkthrough: stress test and boundaries | [`notebooks/02_ipqc_stress_test.ipynb`](notebooks/02_ipqc_stress_test.ipynb) |

The commanded action meets the deadband first (static friction acts on the command), then degradation scales the torque actually delivered. Latency returns the observation from *k* steps earlier, seeded with the first observation at the start of each episode.

Hyperparameters (learning rate 1e-3, batch 256, action-noise sigma 0.1, CUDA) and every sweep level live in the config file.

## Reproduce

Python 3.12.0 has a `zipfile` bug that stops Stable-Baselines3 reloading saved models, so use a newer Python (developed on 3.14).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name ai_portfolio
```

Then either run the notebooks (kernel `ai_portfolio`), or the scripts:

```bash
python scripts/train.py                  # trains and writes models/reacher_td3.zip
python scripts/evaluate_ipqc.py          # writes the stress plots and plots/ipqc_results.csv
python scripts/train.py --timesteps 1000 --learning-starts 100 --model-path models/dryrun   # GPU smoke test
pytest                                   # 39 tests
```

`--timesteps` must exceed `--learning-starts`, otherwise training would run no gradient updates and the script refuses.
Set `DFC_QUICK=1` before launching a notebook for a tiny smoke-test run (outputs go to `plots/quick/`).

## Limitations

- Reacher is a simple benchmark. The stress curves demonstrate the method; they do not certify a real robot.
- One trained policy and one training seed: boundaries would shift for other seeds. Evaluation uses 3 seeds x 20 episodes per condition and reports no confidence intervals. Levels close to the 0.9 threshold (deadband 0.1 at 0.922, degradation 60% at 0.911) are within episode-to-episode noise.
- The retention score uses the random policy on the unstressed environment as its floor.
- `Reacher-v4` still works in Gymnasium 1.4 but is flagged out of date; the code falls back to `v5` automatically if v4 is removed.
- Weights & Biases logging is implemented and unit-tested with a stub, but has not been run against a live W&B account.
