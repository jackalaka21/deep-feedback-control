> **Repo note (2026-10-14):** this is the portfolio-level spec covering both projects. Project 1 (`deep-feedback-control`, sections 1, 2, 3 and 7) lives in this repo; Project 2 (`unified-risk-engine`, section 4) will get its own repo and a copy of this spec.

# AI Engineering Portfolio: Design Spec

Date: 2026-10-14
Source: `AI_Engineering_Portfolio_Plan.docx` + pasted "Master AI Engineering Portfolio Playbook" README.
Where they differ, the pasted playbook wins (Hetero-GAT + GNNExplainer, TD3 only), but the backbone keeps a SAGEConv option.

## 1. Goal and success criteria

Two portfolio-grade projects that run end to end on one HP OMEN 16 (RTX 3070 Ti, 8 GB), laid out as in the playbook:

1. **deep-feedback-control**: TD3 on Reacher, plus an "IPQC" stress-testing harness (latency, deadband, motor degradation) that plots policy safety boundaries.
2. **unified-risk-engine**: a Hetero-GAT multi-task model on IEEE-CIS (fraud per transaction, default per user) with Focal Loss and GNNExplainer.

Success:
- A 1,000-step TD3 dry-run trains on CUDA with no errors.
- The full 150k-step run produces `plots/degradation_curve.png`.
- The risk-engine pipeline passes its tests on a generated IEEE-CIS-shaped fixture, and runs on the real data once the user places it in `data/`.
- The README explains setup, usage and honest limitations.

Non-goals: hackathon matrix content, hosting/deployment, remote pushes, any Kaggle login or W&B login on the user's behalf.

## 2. Environment and repository

- Repo root: `<repo root>`, with a local `git init` (no remote, no push).
- Python 3.14 venv at `<venv>` (outside OneDrive). Python 3.12.0 was rejected: its `zipfile` bug makes SB3 model loading fail. PyTorch CUDA wheel from an index matching the installed driver (not necessarily `cu121`), then `gymnasium[mujoco]`, `stable-baselines3`, `torch-geometric`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `wandb`, `pytest`.
- `requirements.txt` at the root lists all of these; the torch CUDA index URL is documented in the README.
- `.gitignore`: `models/*.zip`, `*.pt`, `data/*.csv`, `plots/`, `wandb/`, `__pycache__`, venvs.
- The dataset directory is configurable through `DATA_DIR` (default `unified-risk-engine/data/`) so the large CSVs can live outside OneDrive.
- Layout is exactly the playbook tree, plus `tests/` in each project and `docs/superpowers/` for this spec and the plan.

## 3. Project 1: deep-feedback-control

**Config** (`configs/td3_config.yaml`): env id, total_timesteps 150000, learning_rate 1e-3, batch_size 256, device cuda, action-noise sigma 0.1, seed, save path, W&B flag (default off).

**Env id:** `Reacher-v4`. If the installed Gymnasium no longer provides it, use `Reacher-v5` and record that in the README.

**`scripts/train.py`**
- Loads the YAML, builds the env and `NormalActionNoise`, trains SB3 TD3.
- CLI overrides: `--timesteps` (used for the 1,000-step dry-run), `--config`, `--device`.
- Saves to `models/reacher_td3`. W&B is optional and offline-safe.

**`envs/stress_wrappers.py`** (composable `gym.Wrapper`s)
- `ObservationLatency(delay_steps)`: returns the observation from k steps earlier. The buffer is seeded with the first observation on reset and cleared on every reset.
- `ActuatorDeadband(threshold=0.05)`: zeroes action components with |a| < threshold.
- `MotorDegradation(scale)`: multiplies the action by `scale` (1.0 = nominal).
- `make_stressed_env(...)` helper composing all three.

**`scripts/evaluate_ipqc.py`**
- Loads the saved model. Sweeps:
  - degradation: 1.0, 0.8, 0.6, 0.4, 0.2
  - latency: 0 to 5 steps
  - deadband: several thresholds
- Each setting runs N episodes per seed over several seeds, giving mean and standard deviation of episode return.
- Outputs `plots/degradation_curve.png`, `plots/latency_curve.png`, `plots/deadband_curve.png`, `plots/ipqc_results.csv`.
- Safety boundary: per stress type, the most severe level at which mean return is at least 90% of nominal. It is printed and annotated on the plots.

**Tests:** wrapper unit tests (delay exactness, reset behaviour, deadband boundary at exactly 0.05, scaling), a config-loading test, and a smoke test of train + evaluate with tiny budgets.

## 4. Project 2: unified-risk-engine (IEEE-CIS)

**Data:** `train_transaction.csv` is required; `train_identity.csv` is optional. The user downloads them from Kaggle (login and rule acceptance required, so not automated). Missing files give a clear error naming the expected path.

**`pipeline/feature_engineering.py`**
- Transaction features: log-amount, time-of-day and day-of-week from `TransactionDT`, frequency encodings for categorical columns, a selected subset of `V*` columns, NaN indicators.
- User features: aggregates over each user's transactions (count, mean/std amount, fraction by product code, etc.).
- Fit statistics come from the training split only.

**`pipeline/graph_builder.py`** builds `HeteroData`:
- Node types: `transaction`, `user` (key = `card1` + `addr1`), `email_domain` (`P_emaildomain`), `device` (`DeviceInfo`, only if the identity file is present).
- Edges: `user→transaction`, `transaction→email_domain`, `transaction→device`, and reverse edges.
- Labels and masks, built as follows:
  - Fraud label: `isFraud` on transaction nodes.
  - Default label, a documented proxy: a user is "defaulted" if their fraud rate among training-period transactions is ≥ a configurable threshold and they have ≥ a configurable minimum number of transactions.
  - Users are split first into train/val/test. Time is split into train/val/test periods by `TransactionDT`.
  - A transaction is in the fraud-train mask only if its user is a train user and it falls in the training period.
  - The fraud val and test masks are defined from the time split and the user split, so that no label derived from val/test users' transactions leaks into training.
  - The default masks are the user splits.

**`models/gnn_backbone.py`:** 2-layer `HeteroConv` with GATv2 per relation (configurable heads and hidden size, default 64×4), plus a SAGEConv option, with residual connections and dropout.

**`models/multi_task_heads.py`:** a transaction head (fraud logit) and a user head (default logit), each a small MLP on backbone embeddings.

**`scripts/train.py`**
- Loads data, builds the graph, trains with `NeighborLoader` mini-batches (full-batch option for small data and tests).
- Loss: Focal Loss per head (configurable alpha and gamma), combined as a weighted sum.
- Metrics: PR-AUC, ROC-AUC and recall at 1% FPR for each head, plus a gradient-boosting tabular baseline on the same features.
- Saves the best checkpoint by validation PR-AUC, logs to console and optionally W&B.

**`scripts/explain.py`**
- PyG `Explainer` with `GNNExplainer` on chosen or top-risk transaction and user nodes.
- Outputs a feature-importance bar chart, the key neighbouring edges, and a short plain-English report in `reports/`.

**Tests:** run on a generated fixture with IEEE-CIS-shaped columns:
- graph schema and edge counts
- no split leakage (asserted explicitly)
- Focal Loss correctness (reduces to cross-entropy at gamma 0)
- forward-pass shapes
- a short training smoke test
- explainer smoke test

## 5. Honest limitations (go in the README)

- The default label is a proxy derived from fraud history, not real credit data. The multi-task claim is "joint risk modelling on a shared graph", not "validated credit scoring".
- Reacher is a simple benchmark, so its stress curves illustrate the method more than they certify a real robot.
- Results on the real IEEE-CIS data are not measured until the user supplies the data; only the fixture-based tests are verified by the builder.

## 6. Verification plan

- `pytest` passes in both projects.
- TD3 1,000-step dry-run on CUDA shows `device=cuda` and completes.
- Full 150k-step training run and IPQC evaluation, if the user wants the real plots.
- GNN smoke training on the fixture on GPU.
- Report exactly what was and was not run.

## 7. Amendments (2026-10-14, after spec approval)

1. **Build order:** Project 1 first, with its own implementation plan. Project 2 gets a separate plan afterwards. The root `requirements.txt` starts with the shared and Project 1 dependencies; PyG and friends are added in the Project 2 plan.
2. **Jupyter notebooks are required** (user request). Project 1 gets `notebooks/01_train_td3.ipynb` and `notebooks/02_ipqc_stress_test.ipynb`. They import the same functions as `scripts/train.py` and `scripts/evaluate_ipqc.py` (no duplicated logic), show training curves and stress plots inline, and are committed with outputs from the full run. Both notebooks honour `DFC_QUICK=1` (tiny budgets) so they can be smoke-tested headlessly.
3. **Safety boundary metric (replaces "90% of nominal"):** Reacher returns are negative (about -5 when good, about -43 for a random policy), so a percentage of nominal return is ill-defined. Instead, each stress level gets a retention score `(R - R_random) / (R_nominal - R_random)`. The safety boundary is the most severe level reached before the score first drops below the threshold (0.9), scanning from mildest to most severe.
4. **Plots:** the final `degradation_curve.png`, `latency_curve.png`, `deadband_curve.png`, `training_curve.png` and `ipqc_results.csv` are committed so the README can show them. Quick-mode outputs go to `plots/quick/` (ignored). Trained model files stay ignored.
5. **Evaluation device:** IPQC episodes run the policy on CPU (single-observation inference is faster there); training uses CUDA.
6. **Retention is scored on task performance, not total reward (supersedes the metric in 7.3; found in final review).** Reacher's reward is `reward_dist + reward_ctrl`, and the control cost is charged on the action the motors actually receive. Deadband and degradation shrink actions, so they also shrink the control cost, and 77% of a random policy's deficit is control cost. Scoring total reward therefore flattered those stresses (for example, deadband 0.5 scored 0.815 while the arm was worse than random at reaching the target). The retention score `(R - R_random) / (R_nominal - R_random)` now uses `R = -sum(distance to target)` (`score`) and drives the safety boundaries; the full-reward version is kept as `total_score`, with `mean_control_return` per level, in `ipqc_results.csv`.
