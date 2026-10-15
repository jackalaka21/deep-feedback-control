# DDPG Baseline and Reward-Shaping Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DDPG baseline, a 50-run reward-shaping sweep (2 algorithms x 5 reward variants x 5 seeds) with bootstrap statistics and a success-rate metric, and a stress test of every solved final policy, all presented in a new notebook and the README.

**Architecture:** New modules sit beside the existing ones: `envs/reward_shaping.py` (training-reward wrapper), `scripts/sweep.py` (grid, single-run executor, resumable process pool), `scripts/sweep_analysis.py` (metrics, bootstrap, tables, plots), `scripts/shaping_robustness.py` (IPQC battery on final models). `scripts/train.py` gains `build_model`, an `algo` option and config validation; `scripts/evaluate_ipqc.py` is generalised to both algorithms. Every run is evaluated on a separate unshaped environment using the task metric (summed distance), so reward variants stay comparable.

**Tech Stack:** Python 3.14 venv, PyTorch, gymnasium[mujoco], stable-baselines3 (TD3, DDPG), TensorBoard, pandas, matplotlib, PyYAML, pytest, ipykernel / nbformat / nbconvert.

**Spec:** `docs/superpowers/specs/2026-10-15-ddpg-reward-shaping-sweep-design.md` (extends `docs/superpowers/specs/2026-10-14-ai-engineering-portfolio-design.md`). Executors read both.

## Global Constraints

- Repo root is the git repo `deep-feedback-control`, branch `main`. Project 1 only. All paths used by scripts resolve against the project root, never the working directory.
- `<venv>` is the Python 3.14 venv created earlier, outside OneDrive. In bash calls use `PY="<venv>/Scripts/python.exe"`, set again in every call. Python 3.12.0 must not be used (its `zipfile` bug breaks SB3 model loading).
- Hyperparameters for BOTH algorithms: learning rate `1e-3` (YAML `1.0e-3`), batch size `256`, action-noise sigma `0.1`, buffer size `200000`, seed from the run.
- Sweep grid: algos `[td3, ddpg]`; variants `[default, dist_only, sparse, progress, scaled_x10]`; seeds `[0, 1, 2, 3, 4]`; `40000` timesteps; `learning_starts` `5000`; `eval_every` `2000`; `eval_episodes` `10`; `solve_threshold` `-4.0`; `success_radius` `0.05`.
- Sweep runs train on CPU, single-threaded (measured: a gradient step is about 13 ms on CPU and on CUDA alike, so the GPU gives no speedup). W&B is not used by the sweep.
- Evaluation always uses the unshaped environment and the task metric `distance_return` (sum of `reward_dist`; higher is better). Never compare shaped training returns.
- Statistics: 95% bootstrap CIs over seeds, 10,000 draws, percentile interval.
- Do NOT push to a remote without asking the user first. Never log in to W&B or any service.
- End every commit message with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (an extra `-m` argument). Commit with the repo-local identity already configured (personal email).
- Test files are named uniquely; tests run from the repo root with `"$PY" -m pytest -q`. A passing suite is required at the end of every task.
- Do not weaken the existing tests. Where a task changes behaviour covered by an existing test (the IPQC sweep columns), update that test in the same task and say so.

## Review Focus

Failure modes the spec implies but a first implementation would likely miss, most likely first. Each has a test in the task that owns the code.

1. **Shaped training reward leaking into evaluation.** The evaluation callback must report the unshaped task metric even when the agent trains on a shaped reward. (Task 6: `test_eval_callback_records_expected_steps_and_uses_unshaped_metric_while_training_on_shaped_reward`)
2. **A crashed run must not look finished.** A run directory with `eval.csv` but no `done.json` must be re-run on resume and must not appear in the analysis. (Task 7: `test_partial_run_without_done_json_is_rerun`; Task 8: `test_load_eval_curves_ignores_runs_without_done_json`)
3. **Policies that never learned must not be scored in the robustness comparison.** A `sparse` run whose own nominal return is below the solve threshold must be excluded and counted, never given a retention score. (Task 10: `test_ineligible_policy_is_excluded_not_scored`)
4. **One crashing worker must not destroy the sweep.** The other runs must complete and be kept, and the error must name the failed runs. (Task 7: `test_worker_failure_is_isolated_and_reported`)
5. **Empty or degenerate analysis inputs.** No completed runs, a single seed, a run with fewer evaluations than the averaging window, or a threshold never reached must give clear errors or well-defined values, not empty plots or `NaN` crashes. (Task 8: `test_load_eval_curves_with_no_completed_runs_raises`, `test_bootstrap_ci_single_value_is_degenerate`, `test_run_metrics_short_curve`)

---

## File Structure

```
deep-feedback-control/
├── .gitignore                         (Task 1: add runs/)
├── requirements.txt                   (Task 1: pin versions, add tensorboard, jupyterlab)
├── README.md                          (Task 13)
├── configs/td3_config.yaml            (Task 2: algo, ddpg_model_path, success_radius)
├── configs/sweep_config.yaml          (Task 6)
├── envs/__init__.py                   (Task 4: export RewardShaping, VARIANTS)
├── envs/reward_shaping.py             (Task 4)
├── scripts/train.py                   (Tasks 2, 3)
├── scripts/evaluate_ipqc.py           (Tasks 2, 3, 5)
├── scripts/sweep.py                   (Tasks 6, 7)
├── scripts/sweep_analysis.py          (Tasks 8, 9)
├── scripts/shaping_robustness.py      (Task 10)
├── notebooks/01_train_td3.ipynb       (Task 11: text fix only)
├── notebooks/03_reward_shaping_sweep.ipynb (Task 11, executed in Task 12)
├── plots/sweep_*.csv|png|json, plots/shaping_*.csv|png   (Task 12, committed)
└── tests/
    ├── sweep_fakes.py                 (Task 8: shared helper that fabricates run directories)
    ├── test_train_config.py           (Task 2: extended)
    ├── test_dfc_smoke.py              (Task 3: extended)
    ├── test_reward_shaping.py         (Task 4)
    ├── test_ipqc_eval.py              (Task 5: extended)
    ├── test_sweep.py                  (Tasks 6, 7)
    ├── test_sweep_analysis.py         (Tasks 8, 9)
    └── test_shaping_robustness.py     (Task 10)
```

---

### Task 1: Dependencies, pins and a parallel-throughput benchmark

**Files:**
- Modify: `requirements.txt`, `.gitignore`
- Scratch (not committed): `<scratchpad>/bench_parallel.py`

**Interfaces:**
- Produces: TensorBoard and JupyterLab installed in the venv; `requirements.txt` with exact pins; a measured `workers` value for `configs/sweep_config.yaml` (consumed by Task 6).

- [ ] **Step 1: Install the new dependencies**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -m pip install tensorboard jupyterlab
"$PY" -m pip show tensorboard jupyterlab | grep -E "^(Name|Version)"
```
Expected: both install; note the two versions printed.

- [ ] **Step 2: Write `requirements.txt` with exact pins**

Use the versions installed at planning time, plus the two versions from Step 1 (replace `<TB>` and `<JL>`):

```text
# PyTorch must be installed first from a CUDA index (see README), e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
torch==2.11.0
gymnasium[mujoco]==1.4.0
mujoco==3.16.0
stable-baselines3==2.9.0
tensorboard==<TB>
numpy==2.5.4
pandas==3.0.6
matplotlib==3.11.2
pyyaml==6.0.3
wandb==0.30.0
jupyterlab==<JL>
ipykernel==7.4.0
nbformat==5.11.1
nbconvert==7.17.2
pytest==9.1.1
```

- [ ] **Step 3: Add `runs/` to `.gitignore`**

Append these lines to `.gitignore`:

```gitignore

# sweep outputs (raw runs and TensorBoard logs; aggregated CSV/PNG results live in plots/)
runs/
```

- [ ] **Step 4: Check the pinned file installs cleanly and the suite still passes**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" -m pip install --dry-run -r requirements.txt 2>&1 | tail -3
"$PY" -m pytest -q 2>&1 | tail -2
```
Expected: the dry run reports nothing to install or only already-satisfied requirements (torch resolves to the installed `+cu128` build); `39 passed`.

- [ ] **Step 5: Benchmark parallel throughput**

Write `<scratchpad>/bench_parallel.py`:

```python
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path.cwd()  # run this from the repo root
sys.path.insert(0, str(ROOT))


def work(n_steps):
    import torch

    torch.set_num_threads(1)
    from scripts.train import load_config, train

    cfg = load_config()
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        train(cfg, timesteps=n_steps, learning_starts=500, model_path=os.path.join(d, "m"),
              device="cpu", verbose=0)
        return n_steps - 500, time.time() - t0


if __name__ == "__main__":
    for n in (1, 4, 6, 8, 12):
        with mp.get_context("spawn").Pool(n) as pool:
            res = pool.map(work, [2500] * n)
        wall = max(r[1] for r in res)
        total = sum(r[0] for r in res)
        print(f"workers={n:2d}  aggregate {total / wall:6.0f} update-steps/s  (per worker {total / wall / n:5.1f})", flush=True)
```

Run (about 5-8 minutes): `"$PY" <scratchpad>/bench_parallel.py`

Expected: a table of aggregate steps/s per worker count. Choose `workers` as the smallest count reaching at least 90% of the best aggregate throughput, at most 12. Record the table and the choice in the ledger. Estimated sweep wall-clock = `50 runs x 40000 steps / aggregate steps/s`; compare with the spec's 2-2.5 h and tell the user if it differs by more than 30%.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: pin dependencies, add tensorboard and jupyterlab, ignore sweep runs" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Config validation, algo/success-radius keys and CLI housekeeping

**Files:**
- Modify: `configs/td3_config.yaml`, `scripts/train.py`, `scripts/evaluate_ipqc.py`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Produces in `scripts.train`: `VALID_ALGOS = ("td3", "ddpg")`; `load_config(path=None)` (now resolves relative paths against the project root, defaults `algo="td3"`, `ddpg_model_path="models/reacher_ddpg"`, `success_radius=0.05`, and validates); `build_parser() -> argparse.ArgumentParser` with `--config --timesteps --learning-starts --device --model-path --algo --wandb/--no-wandb`.
- `resolve_device` error message mentions `device: cpu`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_train_config.py`)

```python
import yaml

from scripts.train import build_parser


def write_config(tmp_path, **overrides):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_config_defaults_to_td3_with_success_radius():
    cfg = load_config()
    assert cfg["algo"] == "td3"
    assert cfg["success_radius"] == 0.05
    assert cfg["ddpg_model_path"] == "models/reacher_ddpg"


def test_load_config_rejects_unknown_algo(tmp_path):
    with pytest.raises(ValueError, match="algo"):
        load_config(write_config(tmp_path, algo="sac"))


def test_load_config_rejects_string_learning_rate(tmp_path):
    path = tmp_path / "cfg.yaml"
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("learning_rate: 1.0e-3", "learning_rate: 1e-3")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(path)


def test_load_config_rejects_empty_stress_list(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["ipqc"]["latency_steps"] = []
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="latency_steps"):
        load_config(path)


def test_relative_config_path_resolves_against_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/td3_config.yaml")["env_id"].startswith("Reacher")


def test_cuda_error_mentions_config_option(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device: cpu"):
        resolve_device("cuda")


def test_parser_wandb_tristate_and_algo():
    parser = build_parser()
    assert parser.parse_args([]).wandb is None
    assert parser.parse_args(["--wandb"]).wandb is True
    assert parser.parse_args(["--no-wandb"]).wandb is False
    assert parser.parse_args(["--algo", "ddpg"]).algo == "ddpg"
    assert parser.parse_args([]).algo is None
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_train_config.py -q`
Expected: collection error `ImportError: cannot import name 'build_parser'`.

- [ ] **Step 3: Add the new keys to `configs/td3_config.yaml`**

Insert after the `model_path:` line:

```yaml
algo: td3                   # td3 or ddpg
ddpg_model_path: models/reacher_ddpg
success_radius: 0.05        # metres: an episode "succeeds" if the fingertip ends this close to the target
```

- [ ] **Step 4: Implement in `scripts/train.py`**

Add near the other constants:

```python
VALID_ALGOS = ("td3", "ddpg")
STRESS_LIST_KEYS = ("degradation", "latency_steps", "deadband", "seeds")
```

Replace `load_config` with:

```python
def load_config(path=None) -> dict:
    path = resolve_path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    missing_ipqc = [k for k in REQUIRED_IPQC_KEYS if k not in cfg["ipqc"]]
    if missing_ipqc:
        raise ValueError(f"{path} is missing required ipqc keys: {missing_ipqc}")
    cfg.setdefault("algo", "td3")
    cfg.setdefault("ddpg_model_path", "models/reacher_ddpg")
    cfg.setdefault("success_radius", 0.05)
    if cfg["algo"] not in VALID_ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {cfg['algo']!r}")
    if not isinstance(cfg["learning_rate"], float):
        raise ValueError(
            f"learning_rate must be a float, got {cfg['learning_rate']!r}; "
            "write 1.0e-3 (PyYAML reads 1e-3 as a string)"
        )
    for key in STRESS_LIST_KEYS:
        values = cfg["ipqc"][key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"ipqc.{key} must be a non-empty list, got {values!r}")
    return cfg
```

In `resolve_device`, change the message to:

```python
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Re-run with --device cpu, set `device: cpu` in configs/td3_config.yaml, "
            "or fix the PyTorch CUDA install."
        )
```

Move the argument parsing out of `main` into a function and use it:

```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TD3 or DDPG on Reacher.")
    p.add_argument("--config", default=None, help="config path (relative paths resolve against the project root)")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--algo", choices=VALID_ALGOS, default=None)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                   help="force W&B logging on or off (default: the config value)")
    return p
```

and in `main`, replace the old parser block with `args = build_parser().parse_args(argv)`. (`--algo` is stored but only acted on in Task 3.)

- [ ] **Step 5: Make the IPQC CLI resolve `--config` the same way**

`evaluate_ipqc.main` already passes `args.config` to `load_config`, which now resolves it against the project root; update its `--config` help text to say so. No other change.

- [ ] **Step 6: Run to verify pass**

Run: `"$PY" -m pytest -q`
Expected: all tests pass (39 existing + 7 new).

- [ ] **Step 7: Commit**

```bash
git add configs/td3_config.yaml scripts/train.py scripts/evaluate_ipqc.py tests/test_train_config.py
git commit -m "feat: validate config, add algo and success_radius keys, tri-state --wandb" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: DDPG support

**Files:**
- Modify: `scripts/train.py`, `scripts/evaluate_ipqc.py`
- Test: `tests/test_dfc_smoke.py`

**Interfaces:**
- Consumes: `VALID_ALGOS`, `build_parser` (Task 2).
- Produces in `scripts.train`: `ALGOS = {"td3": TD3, "ddpg": DDPG}`; `default_model_path(cfg, algo) -> str`; `build_model(cfg, env, algo, device, learning_starts, seed=None, verbose=1, tensorboard_log=None)`; `train(cfg, timesteps=None, learning_starts=None, model_path=None, device=None, use_wandb=None, verbose=1, algo=None) -> TrainResult`.
- Produces in `scripts.evaluate_ipqc`: `load_policy(model_path, device="cpu", algo="td3")`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dfc_smoke.py`)

```python
from stable_baselines3 import DDPG, TD3

from scripts.evaluate_ipqc import load_policy
from scripts.train import default_model_path


def test_train_ddpg_cpu_smoke(tmp_path):
    pytest.importorskip("mujoco")
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m",
                device="cpu", verbose=0, algo="ddpg")
    assert isinstance(res.model, DDPG)
    assert model_file(res.model_path).exists()


def test_train_default_algo_comes_from_config(tmp_path):
    pytest.importorskip("mujoco")
    cfg = {**load_config(), "algo": "ddpg"}
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m", device="cpu", verbose=0)
    assert isinstance(res.model, DDPG)


def test_train_rejects_unknown_algo(tmp_path):
    with pytest.raises(ValueError, match="algo"):
        train(load_config(), timesteps=300, learning_starts=100, model_path=tmp_path / "m",
              device="cpu", verbose=0, algo="sac")


def test_default_model_path_depends_on_algo():
    cfg = load_config()
    assert default_model_path(cfg, "td3") == cfg["model_path"]
    assert default_model_path(cfg, "ddpg") == cfg["ddpg_model_path"]


def test_ddpg_save_load_roundtrip(tmp_path):
    pytest.importorskip("mujoco")
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m",
                device="cpu", verbose=0, algo="ddpg")
    assert isinstance(load_policy(res.model_path, algo="ddpg"), DDPG)


def test_load_policy_rejects_unknown_algo(tmp_path):
    with pytest.raises(ValueError, match="algo"):
        load_policy(tmp_path / "m", algo="sac")


def test_td3_still_trains_as_td3(tmp_path):
    pytest.importorskip("mujoco")
    res = train(load_config(), timesteps=300, learning_starts=100, model_path=tmp_path / "m",
                device="cpu", verbose=0)
    assert isinstance(res.model, TD3)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_dfc_smoke.py -q`
Expected: `ImportError: cannot import name 'default_model_path'`.

- [ ] **Step 3: Implement in `scripts/train.py`**

Change the SB3 import to `from stable_baselines3 import DDPG, TD3` and add after `VALID_ALGOS`:

```python
ALGOS = {"td3": TD3, "ddpg": DDPG}


def default_model_path(cfg: dict, algo: str) -> str:
    return cfg["ddpg_model_path"] if algo == "ddpg" else cfg["model_path"]


def build_model(cfg, env, algo, device, learning_starts, seed=None, verbose=1, tensorboard_log=None):
    """One place that builds TD3 or DDPG with identical hyperparameters."""
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {algo!r}")
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=cfg["action_noise_sigma"] * np.ones(n_actions)
    )
    return ALGOS[algo](
        "MlpPolicy",
        env,
        action_noise=noise,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        learning_starts=learning_starts,
        device=device,
        seed=cfg["seed"] if seed is None else seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
```

Update `TrainResult.model` annotation to `object` (either algorithm), and rewrite `train`:

```python
def train(
    cfg: dict,
    timesteps: int | None = None,
    learning_starts: int | None = None,
    model_path=None,
    device: str | None = None,
    use_wandb: bool | None = None,
    verbose: int = 1,
    algo: str | None = None,
) -> TrainResult:
    algo = algo or cfg["algo"]
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {algo!r}")
    device = resolve_device(device or cfg["device"])
    timesteps = int(timesteps or cfg["total_timesteps"])
    start = learning_starts if learning_starts is not None else cfg["learning_starts"]
    if start >= timesteps:
        raise ValueError(
            f"timesteps ({timesteps}) must exceed learning_starts ({start}), otherwise no gradient "
            "updates run and the saved model is untrained. Raise --timesteps or lower --learning-starts."
        )
    model_path = resolve_path(model_path or default_model_path(cfg, algo))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = model_path.with_name(model_path.name + ".monitor.csv")

    env = Monitor(make_base_env(cfg["env_id"]), filename=str(monitor_path))
    model = build_model(cfg, env, algo, device, start, verbose=verbose)

    callback, run = None, None
    if use_wandb if use_wandb is not None else cfg.get("wandb", False):
        import wandb

        run = wandb.init(project=cfg.get("wandb_project", "deep-feedback-control"), config=cfg)
        callback = WandbEpisodeLogger(wandb)
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
    finally:
        if run is not None:
            run.finish()
        env.close()  # flushes the monitor CSV
    model.save(str(model_path))
    return TrainResult(model, model_path, monitor_path)
```

In `main`, pass `algo=args.algo` to `train(...)`.

- [ ] **Step 4: Implement `load_policy` in `scripts/evaluate_ipqc.py`**

Replace it with:

```python
def load_policy(model_path, device: str = "cpu", algo: str = "td3"):
    from scripts.train import ALGOS

    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {tuple(ALGOS)}, got {algo!r}")
    f = model_file(resolve_path(model_path))
    if not f.exists():
        raise FileNotFoundError(
            f"No trained model at {f}. Train one first: `python scripts/train.py` "
            "(or run notebook 01_train_td3.ipynb)."
        )
    return ALGOS[algo].load(str(f), device=device)
```

Also add `p.add_argument("--algo", choices=("td3", "ddpg"), default=None)` to `evaluate_ipqc.main`, and use `algo = args.algo or cfg["algo"]` with `load_policy(args.model_path or default_model_path(cfg, algo), algo=algo)` (import `default_model_path` from `scripts.train`).

- [ ] **Step 5: Run to verify pass**

Run: `"$PY" -m pytest -q`
Expected: all tests pass (the earlier tests, including `test_train_save_load_evaluate_roundtrip`, still pass).

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py scripts/evaluate_ipqc.py tests/test_dfc_smoke.py
git commit -m "feat: add DDPG baseline with identical hyperparameters via build_model" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Reward-shaping wrapper

**Files:**
- Create: `envs/reward_shaping.py`
- Modify: `envs/__init__.py`
- Test: `tests/test_reward_shaping.py`

**Interfaces:**
- Produces in `envs`: `VARIANTS = ("default", "dist_only", "sparse", "progress", "scaled_x10")`; `RewardShaping(env, variant="default", success_radius=0.05)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_reward_shaping.py`)

```python
import gymnasium as gym
import numpy as np
import pytest

from envs import VARIANTS, RewardShaping

DISTANCES = [0.2, 0.1, 0.04]
ACTION = np.array([0.3, 0.4])  # control cost = -(0.09 + 0.16) = -0.25


class StubEnv(gym.Env):
    """Reacher-like: reports reward_dist / reward_ctrl in info, like MuJoCo Reacher."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float64)

    def __init__(self, with_info=True):
        self.t = 0
        self.with_info = with_info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0]), {}

    def step(self, action):
        d = DISTANCES[self.t % len(DISTANCES)]
        ctrl = -float(np.sum(np.square(action)))
        self.t += 1
        info = {"reward_dist": -d, "reward_ctrl": ctrl} if self.with_info else {}
        return np.array([float(self.t)]), -d + ctrl, False, self.t >= 3, info


def rewards(variant, **kw):
    env = RewardShaping(StubEnv(), variant, **kw)
    env.reset()
    return [env.step(ACTION)[1] for _ in DISTANCES]


def test_variant_names():
    assert VARIANTS == ("default", "dist_only", "sparse", "progress", "scaled_x10")


def test_default_is_unchanged():
    assert rewards("default") == pytest.approx([-0.45, -0.35, -0.29])


def test_dist_only_drops_control_cost():
    assert rewards("dist_only") == pytest.approx([-0.2, -0.1, -0.04])


def test_sparse_is_zero_only_within_radius():
    assert rewards("sparse", success_radius=0.05) == pytest.approx([-1.0, -1.0, 0.0])
    assert rewards("sparse", success_radius=0.15) == pytest.approx([-1.0, 0.0, 0.0])


def test_progress_rewards_getting_closer_and_first_step_has_none():
    # step 1: no previous distance -> progress 0, only control cost
    # step 2: 10 * (0.2 - 0.1) - 0.25 ; step 3: 10 * (0.1 - 0.04) - 0.25
    assert rewards("progress") == pytest.approx([-0.25, 0.75, 0.35])


def test_progress_state_resets_between_episodes():
    env = RewardShaping(StubEnv(), "progress")
    env.reset()
    for _ in DISTANCES:
        env.step(ACTION)
    env.reset()
    assert env.step(ACTION)[1] == pytest.approx(-0.25)


def test_scaled_x10_multiplies_reward():
    assert rewards("scaled_x10") == pytest.approx([-4.5, -3.5, -2.9])


def test_observations_and_termination_pass_through():
    env = RewardShaping(StubEnv(), "sparse")
    env.reset()
    obs, _, terminated, truncated, info = env.step(ACTION)
    assert obs[0] == 1.0 and terminated is False and truncated is False
    assert info["reward_dist"] == pytest.approx(-0.2)


def test_missing_info_keys_raise_clear_error():
    env = RewardShaping(StubEnv(with_info=False), "dist_only")
    env.reset()
    with pytest.raises(KeyError, match="reward_dist"):
        env.step(ACTION)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        RewardShaping(StubEnv(), "bogus")


def test_wrapper_works_on_real_reacher():
    pytest.importorskip("mujoco")
    from envs import make_base_env

    env = RewardShaping(make_base_env("Reacher-v4"), "dist_only")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert reward == pytest.approx(info["reward_dist"])
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_reward_shaping.py -q`
Expected: `ImportError: cannot import name 'VARIANTS' from 'envs'`.

- [ ] **Step 3: Implement** (`envs/reward_shaping.py`)

```python
"""Training-reward variants for MuJoCo Reacher, used by the reward-shaping sweep.

Only the reward the agent trains on changes. Observations, termination and `info`
pass through, and evaluation always uses the unshaped environment.
"""
from __future__ import annotations

import gymnasium as gym

VARIANTS = ("default", "dist_only", "sparse", "progress", "scaled_x10")


class RewardShaping(gym.Wrapper):
    def __init__(self, env: gym.Env, variant: str = "default", success_radius: float = 0.05):
        super().__init__(env)
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.success_radius = float(success_radius)
        self._prev_distance = None

    def reset(self, **kwargs):
        self._prev_distance = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            dist_term = float(info["reward_dist"])  # minus distance to the target
            ctrl_term = float(info["reward_ctrl"])  # minus squared action
        except KeyError as exc:
            raise KeyError(
                f"RewardShaping needs info['reward_dist'] and info['reward_ctrl'] "
                f"(MuJoCo Reacher provides both); missing {exc}"
            ) from exc
        distance = -dist_term

        if self.variant == "default":
            shaped = reward
        elif self.variant == "dist_only":
            shaped = dist_term
        elif self.variant == "sparse":
            shaped = 0.0 if distance <= self.success_radius else -1.0
        elif self.variant == "progress":
            progress = 0.0 if self._prev_distance is None else self._prev_distance - distance
            shaped = 10.0 * progress + ctrl_term
        else:  # scaled_x10
            shaped = 10.0 * reward
        self._prev_distance = distance
        return obs, shaped, terminated, truncated, info
```

Update `envs/__init__.py` to also export them:

```python
from .reward_shaping import VARIANTS, RewardShaping
from .stress_wrappers import (
    ActuatorDeadband,
    MotorDegradation,
    ObservationLatency,
    make_base_env,
    make_stressed_env,
)

__all__ = [
    "ActuatorDeadband",
    "MotorDegradation",
    "ObservationLatency",
    "RewardShaping",
    "VARIANTS",
    "make_base_env",
    "make_stressed_env",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add envs tests/test_reward_shaping.py
git commit -m "feat: add reward-shaping wrapper with five training-reward variants" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: IPQC evaluation changes (final distance, fail-fast baselines, clear errors)

**Files:**
- Modify: `scripts/evaluate_ipqc.py`
- Test: `tests/test_ipqc_eval.py`

**Interfaces:**
- `run_episode_parts(env, policy, seed=None) -> dict` now also returns `final_distance` (distance to target at the last step; `-info["reward_dist"]`). Keys: `episode_return`, `distance_return`, `control_return`, `final_distance`.
- `run_ipqc_sweep` DataFrame columns become `stress_type, level, order, seed, episode, episode_return, distance_return, control_return, final_distance`, and it raises `ValueError` as soon as the random and nominal baselines are done if nominal is not better than random.
- `summarize` raises `ValueError` naming any missing baseline rows.

- [ ] **Step 1: Write the failing tests** (in `tests/test_ipqc_eval.py`)

Update the expected columns in `test_sweep_covers_every_condition` to:

```python
    assert list(df.columns) == ["stress_type", "level", "order", "seed", "episode",
                                "episode_return", "distance_return", "control_return",
                                "final_distance"]
```

Append:

```python
def test_final_distance_matches_last_step_distance():
    pytest.importorskip("mujoco")
    from envs import make_base_env
    from scripts.evaluate_ipqc import run_episode_parts

    cfg = load_config()
    env = make_base_env(cfg["env_id"])
    parts = run_episode_parts(env, ZeroModel(), seed=0)
    env2 = make_base_env(cfg["env_id"])
    env2.reset(seed=0)
    done = False
    while not done:
        _, _, te, tr, info = env2.step(np.zeros(2, dtype=np.float32))
        done = te or tr
    assert parts["final_distance"] == pytest.approx(-info["reward_dist"])
    assert parts["final_distance"] >= 0


def test_sweep_fails_fast_when_nominal_is_not_better_than_random(monkeypatch):
    import scripts.evaluate_ipqc as ev

    calls = []

    def fake_parts(env, policy, seed=None):
        calls.append(type(policy).__name__)
        d = -5.0 if isinstance(policy, ev.RandomPolicy) else -20.0  # model is worse than random
        return {"episode_return": d, "distance_return": d, "control_return": 0.0, "final_distance": 0.1}

    monkeypatch.setattr(ev, "run_episode_parts", fake_parts)
    cfg = load_config()
    with pytest.raises(ValueError, match="Nominal return"):
        ev.run_ipqc_sweep(ZeroModel(), cfg, episodes_per_seed=2, seeds=[0])
    assert len(calls) == 4  # random (2) + nominal (2) only; none of the stress levels ran


def test_summarize_names_missing_baseline_rows():
    df = episode_df({("nominal", 0.0, 0): -5.0, ("degradation", 1.0, 0): -5.0})
    with pytest.raises(ValueError, match="random"):
        summarize(df)
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_ipqc_eval.py -q`
Expected: failures: `KeyError: 'final_distance'`, the fail-fast test does not raise before running every condition, and `summarize` raises `IndexError` instead of `ValueError`.

- [ ] **Step 3: Implement in `scripts/evaluate_ipqc.py`**

In `run_episode_parts`, track the last-step distance and return it:

```python
    obs, _ = env.reset(seed=seed)
    total = distance = control = 0.0
    final_distance = float("nan")
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        distance += float(info["reward_dist"])
        control += float(info["reward_ctrl"])
        final_distance = -float(info["reward_dist"])
        done = terminated or truncated
    return {"episode_return": total, "distance_return": distance,
            "control_return": control, "final_distance": final_distance}
```

Add the baseline check and call it in the sweep right after the nominal condition:

```python
def _check_baselines(rows) -> None:
    df = pd.DataFrame(rows)
    r_rand = df.loc[df.stress_type == "random", "distance_return"].mean()
    r_nom = df.loc[df.stress_type == "nominal", "distance_return"].mean()
    if r_nom <= r_rand:
        raise ValueError(
            f"Nominal return ({r_nom:.2f}, task/distance) is not better than the random policy "
            f"({r_rand:.2f}); the model is not trained well enough to stress-test. Train longer."
        )
```

and in `run_ipqc_sweep`, after the inner `for seed in seeds:` loop finishes for a condition (i.e., at the end of the outer loop body):

```python
        if stress_type == "nominal":
            _check_baselines(rows)  # fail fast, before the long stress sweep
```

At the top of `summarize`, before grouping:

```python
    missing = [name for name in ("random", "nominal") if name not in set(df["stress_type"])]
    if missing:
        raise ValueError(f"summarize needs 'random' and 'nominal' baseline rows; missing: {missing}")
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest -q`
Expected: all pass. (Note in the ledger that the existing `test_sweep_covers_every_condition` column list was updated, per Global Constraints.)

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_ipqc.py tests/test_ipqc_eval.py
git commit -m "feat: record final distance, fail fast on weak baselines, clearer IPQC errors" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Sweep config, task evaluation callback and single-run executor

**Files:**
- Create: `configs/sweep_config.yaml`, `scripts/sweep.py` (part 1), `tests/test_sweep.py`

**Interfaces:**
- Consumes: `scripts.train.{ALGOS, VALID_ALGOS, build_model, load_config, resolve_path, PROJECT_ROOT}`; `scripts.evaluate_ipqc.{RandomPolicy, run_episode_parts}`; `envs.{RewardShaping, VARIANTS, make_base_env}`.
- Produces in `scripts.sweep`:
  - `load_sweep_config(path=None) -> dict`
  - `run_name(spec) -> str` (`f"{algo}_{variant}_seed{seed}"`), `run_dir(runs_dir, spec) -> Path`, `is_done(runs_dir, spec) -> bool`
  - `evaluate_policy(policy, env_id, episodes, eval_seed, success_radius) -> dict` with keys `mean_distance_return, std_distance_return, mean_total_return, success_rate`
  - `TaskEvalCallback(env_id, eval_every, episodes, eval_seed, success_radius)` with `.records` (list of dicts including `timesteps`)
  - `run_metadata() -> dict` with keys `python, torch, stable_baselines3, gymnasium, mujoco, git_commit`
  - `make_spec(sweep_cfg, train_cfg, item) -> dict` (keys: `algo, variant, seed, timesteps, learning_starts, eval_every, eval_episodes, eval_seed, success_radius, runs_dir, train_cfg`)
  - `run_one(spec) -> dict` (the `done.json` content; writes `eval.csv`, `final_model.zip`, `tb/`, `done.json`)

- [ ] **Step 1: Write `configs/sweep_config.yaml`**

Set `workers` to the value chosen in Task 1 Step 5 (shown here as 6):

```yaml
# Reward-shaping sweep: DDPG vs TD3 across training-reward variants and seeds.
algos: [td3, ddpg]
variants: [default, dist_only, sparse, progress, scaled_x10]
seeds: [0, 1, 2, 3, 4]
timesteps: 40000
learning_starts: 5000
eval_every: 2000
eval_episodes: 10
eval_seed: 1000
random_baseline_episodes: 100
workers: 6                  # parallel single-threaded CPU processes (benchmarked in the plan)
solve_threshold: -4.0       # task return (-sum distance) that counts as "solved"; about 0.87 retention
runs_dir: runs

robustness:                 # reduced IPQC battery applied to every solved final policy
  episodes: 10
  eval_seed: 2000
  degradation: [1.0, 0.6, 0.4, 0.2]
  latency_steps: [0, 2, 4]
  deadband: [0.0, 0.1, 0.2, 0.3]
  retention_threshold: 0.9
```

- [ ] **Step 2: Write the failing tests** (`tests/test_sweep.py`)

```python
import json

import pandas as pd
import pytest

from envs import RewardShaping, make_base_env
from scripts.sweep import (
    TaskEvalCallback,
    evaluate_policy,
    is_done,
    load_sweep_config,
    make_spec,
    run_dir,
    run_metadata,
    run_name,
    run_one,
)
from scripts.train import build_model, load_config


def tiny_spec(tmp_path, algo="td3", variant="default", seed=0, timesteps=600, **over):
    sweep = load_sweep_config()
    spec = make_spec(sweep, load_config(), {"algo": algo, "variant": variant, "seed": seed})
    spec.update(timesteps=timesteps, learning_starts=100, eval_every=200, eval_episodes=2,
                runs_dir=str(tmp_path / "runs"))
    spec.update(over)
    return spec


def test_sweep_config_loads_with_expected_grid():
    cfg = load_sweep_config()
    assert cfg["algos"] == ["td3", "ddpg"]
    assert cfg["variants"] == ["default", "dist_only", "sparse", "progress", "scaled_x10"]
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["timesteps"] == 40000 and cfg["eval_every"] == 2000


def test_sweep_config_rejects_unknown_variant(tmp_path):
    import yaml

    from scripts.train import resolve_path

    cfg = yaml.safe_load(resolve_path("configs/sweep_config.yaml").read_text(encoding="utf-8"))
    cfg["variants"] = ["default", "bogus"]
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        load_sweep_config(path)


def test_sweep_config_rejects_learning_starts_not_below_timesteps(tmp_path):
    import yaml

    from scripts.train import resolve_path

    cfg = yaml.safe_load(resolve_path("configs/sweep_config.yaml").read_text(encoding="utf-8"))
    cfg["learning_starts"] = cfg["timesteps"]
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="learning_starts"):
        load_sweep_config(path)


def test_run_naming(tmp_path):
    spec = {"algo": "ddpg", "variant": "sparse", "seed": 3}
    assert run_name(spec) == "ddpg_sparse_seed3"
    assert run_dir(tmp_path, spec) == tmp_path / "ddpg_sparse_seed3"
    assert not is_done(tmp_path, spec)
    (tmp_path / "ddpg_sparse_seed3").mkdir()
    (tmp_path / "ddpg_sparse_seed3" / "done.json").write_text("{}", encoding="utf-8")
    assert is_done(tmp_path, spec)


def test_evaluate_policy_returns_expected_fields_and_is_reproducible():
    pytest.importorskip("mujoco")
    cfg = load_config()
    env = make_base_env(cfg["env_id"])
    model = build_model(cfg, env, "td3", "cpu", 100, verbose=0)
    a = evaluate_policy(model, cfg["env_id"], 3, 1000, 0.05)
    b = evaluate_policy(model, cfg["env_id"], 3, 1000, 0.05)
    assert set(a) == {"mean_distance_return", "std_distance_return", "mean_total_return", "success_rate"}
    assert a == b  # same fixed targets, deterministic policy
    assert a["mean_distance_return"] <= 0 and 0.0 <= a["success_rate"] <= 1.0


def test_eval_callback_records_expected_steps_and_uses_unshaped_metric_while_training_on_shaped_reward():
    pytest.importorskip("mujoco")
    cfg = load_config()
    env = RewardShaping(make_base_env(cfg["env_id"]), "sparse", 0.05)  # agent sees rewards in {0, -1}
    model = build_model(cfg, env, "td3", "cpu", 100, verbose=0)
    cb = TaskEvalCallback(cfg["env_id"], eval_every=200, episodes=2, eval_seed=1000, success_radius=0.05)
    model.learn(total_timesteps=600, callback=cb)
    assert [r["timesteps"] for r in cb.records] == [0, 200, 400, 600]
    # The last record must equal an independent evaluation on the UNSHAPED env at the same model state.
    independent = evaluate_policy(model, cfg["env_id"], 2, 1000, 0.05)
    last = {k: v for k, v in cb.records[-1].items() if k != "timesteps"}
    assert last == pytest.approx(independent)
    assert last["mean_distance_return"] != int(last["mean_distance_return"])  # continuous, not {0,-1} shaped


def test_run_metadata_has_versions_and_commit():
    meta = run_metadata()
    assert set(meta) == {"python", "torch", "stable_baselines3", "gymnasium", "mujoco", "git_commit"}
    assert all(isinstance(v, str) and v for v in meta.values())


@pytest.mark.parametrize("algo", ["td3", "ddpg"])
def test_run_one_writes_all_outputs(tmp_path, algo):
    pytest.importorskip("mujoco")
    spec = tiny_spec(tmp_path, algo=algo, variant="progress")
    done = run_one(spec)
    rd = tmp_path / "runs" / f"{algo}_progress_seed0"
    assert is_done(tmp_path / "runs", spec)
    saved = json.loads((rd / "done.json").read_text(encoding="utf-8"))
    assert saved["algo"] == algo and saved["variant"] == "progress" and saved["seed"] == 0
    assert saved["timesteps"] == 600 and saved["final_model"] == "final_model.zip"
    assert "git_commit" in saved and "torch" in saved and saved == done
    curve = pd.read_csv(rd / "eval.csv")
    assert list(curve["timesteps"]) == [0, 200, 400, 600]
    assert {"mean_distance_return", "success_rate"} <= set(curve.columns)
    assert (rd / "final_model.zip").exists()
    assert any((rd / "tb").rglob("events.out.tfevents.*"))
```

- [ ] **Step 3: Run to verify failure**

Run: `"$PY" -m pytest tests/test_sweep.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.sweep'`.

- [ ] **Step 4: Implement** (`scripts/sweep.py`, part 1; Task 7 appends to it)

```python
"""Reward-shaping sweep: DDPG vs TD3 across training-reward variants and seeds."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from envs import VARIANTS, RewardShaping, make_base_env  # noqa: E402
from scripts.evaluate_ipqc import RandomPolicy, run_episode_parts  # noqa: E402
from scripts.train import VALID_ALGOS, build_model, load_config, resolve_path  # noqa: E402

DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / "configs" / "sweep_config.yaml"
REQUIRED_SWEEP_KEYS = (
    "algos", "variants", "seeds", "timesteps", "learning_starts", "eval_every", "eval_episodes",
    "eval_seed", "random_baseline_episodes", "workers", "solve_threshold", "runs_dir", "robustness",
)


def load_sweep_config(path=None) -> dict:
    path = resolve_path(path) if path else DEFAULT_SWEEP_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_SWEEP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    for key in ("algos", "variants", "seeds"):
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f"{key} must be a non-empty list, got {cfg[key]!r}")
    for algo in cfg["algos"]:
        if algo not in VALID_ALGOS:
            raise ValueError(f"unknown algo {algo!r}; valid: {VALID_ALGOS}")
    for variant in cfg["variants"]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; valid: {VARIANTS}")
    if cfg["learning_starts"] >= cfg["timesteps"]:
        raise ValueError(
            f"learning_starts ({cfg['learning_starts']}) must be below timesteps ({cfg['timesteps']})"
        )
    if cfg["timesteps"] % cfg["eval_every"] != 0:
        raise ValueError("timesteps must be a multiple of eval_every so the last step is evaluated")
    return cfg


def run_name(spec: dict) -> str:
    return f"{spec['algo']}_{spec['variant']}_seed{spec['seed']}"


def run_dir(runs_dir, spec: dict) -> Path:
    return resolve_path(runs_dir) / run_name(spec)


def is_done(runs_dir, spec: dict) -> bool:
    return (run_dir(runs_dir, spec) / "done.json").exists()


def evaluate_policy(policy, env_id: str, episodes: int, eval_seed: int, success_radius: float) -> dict:
    """Deterministic evaluation on the UNSHAPED env with fixed targets (task metric = -sum distance)."""
    env = make_base_env(env_id)
    parts = [
        run_episode_parts(env, policy, seed=eval_seed if i == 0 else None) for i in range(episodes)
    ]
    env.close()
    dist = np.array([p["distance_return"] for p in parts])
    total = np.array([p["episode_return"] for p in parts])
    final = np.array([p["final_distance"] for p in parts])
    return {
        "mean_distance_return": float(dist.mean()),
        "std_distance_return": float(dist.std(ddof=1)) if len(dist) > 1 else 0.0,
        "mean_total_return": float(total.mean()),
        "success_rate": float((final <= success_radius).mean()),
    }


class TaskEvalCallback(BaseCallback):
    """Evaluate on the unshaped task at step 0 and then every `eval_every` steps."""

    def __init__(self, env_id, eval_every, episodes, eval_seed, success_radius):
        super().__init__()
        self.env_id, self.eval_every, self.episodes = env_id, eval_every, episodes
        self.eval_seed, self.success_radius = eval_seed, success_radius
        self.records: list[dict] = []

    def _evaluate(self) -> None:
        rec = evaluate_policy(self.model, self.env_id, self.episodes, self.eval_seed, self.success_radius)
        for key, value in rec.items():
            self.logger.record(f"eval/{key}", value)
        self.records.append({"timesteps": int(self.num_timesteps), **rec})

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._evaluate()
        return True


def run_metadata() -> dict:
    import gymnasium
    import mujoco
    import stable_baselines3

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "git_commit": commit,
    }


def make_spec(sweep_cfg: dict, train_cfg: dict, item: dict) -> dict:
    return {
        "algo": item["algo"], "variant": item["variant"], "seed": item["seed"],
        "timesteps": sweep_cfg["timesteps"], "learning_starts": sweep_cfg["learning_starts"],
        "eval_every": sweep_cfg["eval_every"], "eval_episodes": sweep_cfg["eval_episodes"],
        "eval_seed": sweep_cfg["eval_seed"], "success_radius": train_cfg["success_radius"],
        "runs_dir": sweep_cfg["runs_dir"], "train_cfg": train_cfg,
    }


def run_one(spec: dict) -> dict:
    """Train one (algo, variant, seed) run on CPU and write eval.csv, final_model.zip, tb/, done.json."""
    torch.set_num_threads(1)
    rd = run_dir(spec["runs_dir"], spec)
    rd.mkdir(parents=True, exist_ok=True)
    for stale in ("eval.csv", "final_model.zip", "done.json"):  # restart cleanly after a crash
        (rd / stale).unlink(missing_ok=True)

    cfg = {**spec["train_cfg"], "seed": spec["seed"]}
    env = RewardShaping(make_base_env(cfg["env_id"]), spec["variant"], spec["success_radius"])
    model = build_model(cfg, env, spec["algo"], "cpu", spec["learning_starts"], seed=spec["seed"],
                        verbose=0, tensorboard_log=str(rd / "tb"))
    callback = TaskEvalCallback(cfg["env_id"], spec["eval_every"], spec["eval_episodes"],
                                spec["eval_seed"], spec["success_radius"])
    t0 = time.time()
    try:
        model.learn(total_timesteps=spec["timesteps"], callback=callback, tb_log_name="run")
    finally:
        env.close()
    seconds = time.time() - t0

    pd.DataFrame(callback.records).to_csv(rd / "eval.csv", index=False)
    model.save(str(rd / "final_model"))
    done = {
        "algo": spec["algo"], "variant": spec["variant"], "seed": spec["seed"],
        "timesteps": spec["timesteps"], "seconds": round(seconds, 1),
        "final_model": "final_model.zip", **run_metadata(),
    }
    (rd / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")  # written last
    return done
```

- [ ] **Step 5: Run to verify pass**

Run: `"$PY" -m pytest tests/test_sweep.py -q` then `"$PY" -m pytest -q`
Expected: all pass. (`test_run_one_writes_all_outputs` trains two tiny runs; allow about a minute.)

- [ ] **Step 6: Commit**

```bash
git add configs/sweep_config.yaml scripts/sweep.py tests/test_sweep.py
git commit -m "feat: add sweep config, unshaped task evaluation callback and single-run executor" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Sweep orchestration (grid, random baseline, resumable pool, CLI)

**Files:**
- Modify: `scripts/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: everything produced in Task 6.
- Produces in `scripts.sweep`: `build_grid(sweep_cfg) -> list[dict]` (items `{"algo","variant","seed"}` in algos, variants, seeds nesting order); `ensure_random_baseline(sweep_cfg, train_cfg) -> dict` (writes/reads `<runs_dir>/random_baseline.json` with `mean_distance_return, std_distance_return, episodes`); `quick_config(sweep_cfg) -> dict`; `run_sweep(sweep_cfg, train_cfg, workers=None, log=print) -> list[dict]` (raises `RuntimeError` naming failed runs after finishing the others); `main(argv=None)` with `--config --train-config --workers --quick`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sweep.py`)

```python
import scripts.sweep as sweep
from scripts.sweep import build_grid, ensure_random_baseline, quick_config, run_sweep


def one_run_cfg(tmp_path, variants=("default",)):
    cfg = quick_config(load_sweep_config())
    cfg.update(algos=["td3"], variants=list(variants), seeds=[0], runs_dir=str(tmp_path / "runs"),
               workers=1)
    return cfg


def test_grid_enumerates_algos_variants_seeds_in_order():
    cfg = load_sweep_config()
    grid = build_grid(cfg)
    assert len(grid) == 2 * 5 * 5
    assert grid[0] == {"algo": "td3", "variant": "default", "seed": 0}
    assert grid[1] == {"algo": "td3", "variant": "default", "seed": 1}
    assert grid[-1] == {"algo": "ddpg", "variant": "scaled_x10", "seed": 4}
    assert len({(g["algo"], g["variant"], g["seed"]) for g in grid}) == 50


def test_quick_config_is_small_and_isolated():
    q = quick_config(load_sweep_config())
    assert len(build_grid(q)) <= 4 and q["timesteps"] <= 1000 and q["runs_dir"].endswith("quick")
    assert q["learning_starts"] < q["timesteps"] and q["timesteps"] % q["eval_every"] == 0


def test_random_baseline_is_computed_once_and_cached(tmp_path):
    pytest.importorskip("mujoco")
    cfg = one_run_cfg(tmp_path)
    cfg["random_baseline_episodes"] = 3
    first = ensure_random_baseline(cfg, load_config())
    assert first["mean_distance_return"] < -3 and first["episodes"] == 3
    path = tmp_path / "runs" / "random_baseline.json"
    stamp = path.stat().st_mtime_ns
    assert ensure_random_baseline(cfg, load_config()) == first
    assert path.stat().st_mtime_ns == stamp


def test_finished_run_is_skipped_on_relaunch(tmp_path):
    pytest.importorskip("mujoco")
    cfg = one_run_cfg(tmp_path)
    cfg["random_baseline_episodes"] = 2
    logs = []
    run_sweep(cfg, load_config(), workers=1, log=logs.append)
    done_path = tmp_path / "runs" / "td3_default_seed0" / "done.json"
    stamp = done_path.stat().st_mtime_ns
    logs.clear()
    results = run_sweep(cfg, load_config(), workers=1, log=logs.append)
    assert results == [] and done_path.stat().st_mtime_ns == stamp
    assert any("1 already done" in line for line in logs)


def test_partial_run_without_done_json_is_rerun(tmp_path):
    pytest.importorskip("mujoco")
    cfg = one_run_cfg(tmp_path)
    cfg["random_baseline_episodes"] = 2
    run_sweep(cfg, load_config(), workers=1, log=lambda *_: None)
    rd = tmp_path / "runs" / "td3_default_seed0"
    (rd / "done.json").unlink()  # simulate a crash after eval.csv was written
    (rd / "eval.csv").write_text("garbage", encoding="utf-8")
    results = run_sweep(cfg, load_config(), workers=1, log=lambda *_: None)
    assert len(results) == 1 and (rd / "done.json").exists()
    assert "garbage" not in (rd / "eval.csv").read_text(encoding="utf-8")


def test_worker_failure_is_isolated_and_reported(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    cfg = one_run_cfg(tmp_path, variants=("default", "dist_only"))
    cfg["random_baseline_episodes"] = 2
    real_run_one = sweep.run_one

    def flaky(spec):
        if spec["variant"] == "dist_only":
            raise RuntimeError("boom")
        return real_run_one(spec)

    monkeypatch.setattr(sweep, "run_one", flaky)
    with pytest.raises(RuntimeError, match="td3_dist_only_seed0"):
        run_sweep(cfg, load_config(), workers=1, log=lambda *_: None)
    assert (tmp_path / "runs" / "td3_default_seed0" / "done.json").exists()  # the other run was kept


def test_pool_runs_every_grid_cell_end_to_end(tmp_path):
    pytest.importorskip("mujoco")
    cfg = quick_config(load_sweep_config())
    cfg.update(runs_dir=str(tmp_path / "runs"), workers=2, random_baseline_episodes=2)
    results = run_sweep(cfg, load_config(), workers=2, log=lambda *_: None)
    assert len(results) == len(build_grid(cfg)) == 4
    for item in build_grid(cfg):
        assert is_done(tmp_path / "runs", item)
    assert (tmp_path / "runs" / "random_baseline.json").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_sweep.py -q`
Expected: `ImportError: cannot import name 'build_grid' from 'scripts.sweep'`.

- [ ] **Step 3: Implement** (append to `scripts/sweep.py`)

Add to the imports at the top: `import argparse`, `import multiprocessing`, `import itertools`, and `from concurrent.futures import ProcessPoolExecutor, as_completed`. Then append:

```python
def build_grid(sweep_cfg: dict) -> list[dict]:
    return [
        {"algo": a, "variant": v, "seed": s}
        for a, v, s in itertools.product(sweep_cfg["algos"], sweep_cfg["variants"], sweep_cfg["seeds"])
    ]


def ensure_random_baseline(sweep_cfg: dict, train_cfg: dict) -> dict:
    """Random-policy task return over the eval protocol; computed once and cached in the runs dir."""
    runs_dir = resolve_path(sweep_cfg["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "random_baseline.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    env = make_base_env(train_cfg["env_id"])
    policy = RandomPolicy(env.action_space, seed=sweep_cfg["eval_seed"])
    n = sweep_cfg["random_baseline_episodes"]
    dist = [
        run_episode_parts(env, policy, seed=sweep_cfg["eval_seed"] if i == 0 else None)["distance_return"]
        for i in range(n)
    ]
    env.close()
    result = {
        "mean_distance_return": float(np.mean(dist)),
        "std_distance_return": float(np.std(dist, ddof=1)) if n > 1 else 0.0,
        "episodes": n,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def quick_config(sweep_cfg: dict) -> dict:
    """A tiny grid and budget for smoke tests; writes under <runs_dir>/quick (git-ignored)."""
    q = dict(sweep_cfg)
    q.update(
        variants=sweep_cfg["variants"][:2], seeds=sweep_cfg["seeds"][:1], timesteps=800,
        learning_starts=200, eval_every=400, eval_episodes=2, random_baseline_episodes=10,
        runs_dir=str(Path(sweep_cfg["runs_dir"]) / "quick"), workers=min(2, sweep_cfg["workers"]),
    )
    return q


def run_sweep(sweep_cfg: dict, train_cfg: dict, workers: int | None = None, log=print) -> list[dict]:
    """Run every unfinished grid cell; finished cells (with done.json) are skipped."""
    runs_dir = sweep_cfg["runs_dir"]
    grid = build_grid(sweep_cfg)
    pending = [g for g in grid if not is_done(runs_dir, g)]
    log(f"{len(grid)} runs in the grid, {len(grid) - len(pending)} already done, {len(pending)} to run")
    ensure_random_baseline(sweep_cfg, train_cfg)
    specs = [make_spec(sweep_cfg, train_cfg, g) for g in pending]
    workers = workers or sweep_cfg["workers"]

    results, failures = [], []

    def finished(i, spec, done=None, error=None):
        if error is not None:
            failures.append(f"{run_name(spec)}: {error}")
            log(f"[{i}/{len(specs)}] {run_name(spec)} FAILED: {error}")
        else:
            results.append(done)
            log(f"[{i}/{len(specs)}] {run_name(spec)} finished in {done['seconds'] / 60:.1f} min")

    if workers <= 1 or len(specs) <= 1:
        for i, spec in enumerate(specs, 1):
            try:
                finished(i, spec, done=run_one(spec))
            except Exception as exc:  # keep going; report at the end
                finished(i, spec, error=exc)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_one, s): s for s in specs}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    finished(i, futures[fut], done=fut.result())
                except Exception as exc:
                    finished(i, futures[fut], error=exc)
    if failures:
        raise RuntimeError("some runs failed (finished runs were kept; re-launch to retry): " + "; ".join(failures))
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run the DDPG/TD3 reward-shaping sweep (resumable).")
    p.add_argument("--config", default=None, help="sweep config (default configs/sweep_config.yaml)")
    p.add_argument("--train-config", default=None, help="training config (default configs/td3_config.yaml)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="tiny grid and budget, writes to runs/quick")
    args = p.parse_args(argv)
    sweep_cfg = load_sweep_config(args.config)
    if args.quick:
        sweep_cfg = quick_config(sweep_cfg)
    run_sweep(sweep_cfg, load_config(args.train_config), workers=args.workers)
    print("Done. Next: python scripts/sweep_analysis.py ; python scripts/shaping_robustness.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest tests/test_sweep.py -q` then `"$PY" -m pytest -q`
Expected: all pass (the pool test spawns two worker processes and takes about a minute).

- [ ] **Step 5: Smoke-test the CLI**

```bash
"$PY" scripts/sweep.py --quick
```
Expected: prints `4 runs in the grid, 0 already done, 4 to run`, four `finished` lines, then the "Done" hint; `runs/quick/` holds four run directories. Run it again: `4 already done, 0 to run`. (`runs/` is git-ignored.)

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep.py tests/test_sweep.py
git commit -m "feat: add resumable parallel sweep orchestration with failure isolation" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Analysis metrics, bootstrap statistics and tables

**Files:**
- Create: `scripts/sweep_analysis.py` (part 1), `tests/sweep_fakes.py`, `tests/test_sweep_analysis.py`

**Interfaces:**
- Produces in `scripts.sweep_analysis`:
  - `list_completed_runs(runs_dir) -> list[Path]` (run directories containing `done.json`, sorted)
  - `load_eval_curves(runs_dir) -> DataFrame` columns `algo, variant, seed, timesteps, mean_distance_return, std_distance_return, mean_total_return, success_rate` (raises `ValueError("no completed runs ...")` when empty)
  - `run_metrics(curve, threshold, last_k=3) -> dict` keys `final_return, final_success, auc, steps_to_threshold, max_drawdown`
  - `per_run_table(curves, threshold) -> DataFrame` (one row per algo/variant/seed)
  - `bootstrap_ci(values, n_boot=10_000, seed=0, alpha=0.05) -> (mean, lo, hi)`
  - `summarize_runs(per_run) -> DataFrame` per algo/variant with `<metric>_mean/_lo/_hi`, `n_seeds`, `n_reached`
  - `compare_algos(per_run, metric="final_return", a="td3", b="ddpg") -> DataFrame` per variant: `diff_mean, diff_lo, diff_hi` for `a - b`
- Produces in `tests/sweep_fakes.py`: `make_fake_runs(runs_dir, runs, baseline=-10.0)` where `runs` is a list of `(algo, variant, seed, values, success)` tuples (`success` may be `None`); writes `eval.csv` and `done.json` per run plus `random_baseline.json`.

- [ ] **Step 1: Write the shared test helper** (`tests/sweep_fakes.py`)

```python
import json
from pathlib import Path

import pandas as pd


def make_fake_runs(runs_dir, runs, baseline=-10.0, eval_every=2000):
    """Fabricate completed run directories. runs: [(algo, variant, seed, values, success_or_None)]."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for algo, variant, seed, values, success in runs:
        rd = runs_dir / f"{algo}_{variant}_seed{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        success = success if success is not None else [0.0] * len(values)
        pd.DataFrame({
            "timesteps": [i * eval_every for i in range(len(values))],
            "mean_distance_return": values,
            "std_distance_return": [0.5] * len(values),
            "mean_total_return": [v - 1.0 for v in values],
            "success_rate": success,
        }).to_csv(rd / "eval.csv", index=False)
        (rd / "done.json").write_text(json.dumps({
            "algo": algo, "variant": variant, "seed": seed,
            "timesteps": eval_every * (len(values) - 1), "seconds": 1.0, "final_model": "final_model.zip",
        }), encoding="utf-8")
    (runs_dir / "random_baseline.json").write_text(json.dumps({
        "mean_distance_return": baseline, "std_distance_return": 1.0, "episodes": 100,
    }), encoding="utf-8")
    return runs_dir
```

- [ ] **Step 2: Write the failing tests** (`tests/test_sweep_analysis.py`)

```python
import numpy as np
import pandas as pd
import pytest
from sweep_fakes import make_fake_runs

from scripts.sweep_analysis import (
    bootstrap_ci,
    compare_algos,
    list_completed_runs,
    load_eval_curves,
    per_run_table,
    run_metrics,
    summarize_runs,
)


def curve(values, success=None):
    n = len(values)
    return pd.DataFrame({
        "timesteps": [2000 * i for i in range(n)],
        "mean_distance_return": values,
        "success_rate": success if success is not None else [0.0] * n,
    })


def test_run_metrics_final_auc_threshold_and_drawdown():
    m = run_metrics(curve([-8, -5, -3, -6, -3.5], [0, 0.2, 0.8, 0.4, 0.7]), threshold=-4.0)
    assert m["final_return"] == pytest.approx(np.mean([-3, -6, -3.5]))
    assert m["final_success"] == pytest.approx(np.mean([0.8, 0.4, 0.7]))
    assert m["auc"] == pytest.approx(np.mean([-8, -5, -3, -6, -3.5]))
    assert m["steps_to_threshold"] == 4000  # first eval at or above -4.0 is index 2
    assert m["max_drawdown"] == pytest.approx(3.0)  # peak -3 down to -6


def test_run_metrics_threshold_never_reached_and_monotone_curve():
    m = run_metrics(curve([-9, -8, -7, -6]), threshold=-4.0)
    assert np.isnan(m["steps_to_threshold"]) and m["max_drawdown"] == 0.0


def test_run_metrics_short_curve():
    m = run_metrics(curve([-5.0, -4.0]), threshold=-4.5)  # fewer evaluations than last_k=3
    assert m["final_return"] == pytest.approx(-4.5) and m["steps_to_threshold"] == 2000


def test_bootstrap_ci_constant_sample_is_exact():
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    a = bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    assert a == bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    mean, lo, hi = a
    assert mean == 3.0 and lo <= mean <= hi and lo >= 1 and hi <= 5


def test_bootstrap_ci_single_value_is_degenerate():
    assert bootstrap_ci([-3.0]) == (-3.0, -3.0, -3.0)


def test_bootstrap_ci_drops_nan_and_handles_empty():
    assert bootstrap_ci([1.0, float("nan"), 1.0]) == (1.0, 1.0, 1.0)
    mean, lo, hi = bootstrap_ci([float("nan")])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_load_eval_curves_with_no_completed_runs_raises(tmp_path):
    (tmp_path / "td3_default_seed0").mkdir()
    with pytest.raises(ValueError, match="no completed runs"):
        load_eval_curves(tmp_path)


def test_load_eval_curves_ignores_runs_without_done_json(tmp_path):
    make_fake_runs(tmp_path, [("td3", "default", 0, [-8, -4], None), ("td3", "default", 1, [-8, -5], None)])
    (tmp_path / "td3_default_seed1" / "done.json").unlink()  # crashed run
    curves = load_eval_curves(tmp_path)
    assert list(curves["seed"].unique()) == [0]
    assert [p.name for p in list_completed_runs(tmp_path)] == ["td3_default_seed0"]
    assert {"algo", "variant", "seed", "timesteps", "mean_distance_return", "success_rate"} <= set(curves.columns)


def fake_grid(tmp_path):
    runs = []
    for seed, (a, b) in enumerate([(-3.0, -7.0), (-3.2, -6.0), (-3.1, -5.0)]):
        runs.append(("td3", "default", seed, [-9, -6, -4, a, a, a], [0, 0.2, 0.6, 0.9, 0.9, 0.9]))
        runs.append(("ddpg", "default", seed, [-9, -8, -7, b, b, b], None))
    make_fake_runs(tmp_path, runs)
    return load_eval_curves(tmp_path)


def test_per_run_and_summary_tables(tmp_path):
    per_run = per_run_table(fake_grid(tmp_path), threshold=-4.0)
    assert len(per_run) == 6
    summary = summarize_runs(per_run).set_index(["algo", "variant"])
    td3, ddpg = summary.loc[("td3", "default")], summary.loc[("ddpg", "default")]
    assert td3["n_seeds"] == 3 and td3["n_reached"] == 3
    assert ddpg["n_reached"] == 0 and np.isnan(ddpg["steps_to_threshold_mean"])
    assert td3["final_return_mean"] == pytest.approx(np.mean([-3.0, -3.2, -3.1]))
    assert td3["final_return_lo"] <= td3["final_return_mean"] <= td3["final_return_hi"]


def test_compare_algos_reports_td3_minus_ddpg_with_interval(tmp_path):
    per_run = per_run_table(fake_grid(tmp_path), threshold=-4.0)
    diff = compare_algos(per_run).set_index("variant").loc["default"]
    assert diff["diff_mean"] == pytest.approx(np.mean([-3.0, -3.2, -3.1]) - np.mean([-7.0, -6.0, -5.0]))
    assert diff["diff_lo"] > 0  # TD3 clearly better in this fabricated data
```

- [ ] **Step 3: Run to verify failure**

Run: `"$PY" -m pytest tests/test_sweep_analysis.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.sweep_analysis'`.

- [ ] **Step 4: Implement** (`scripts/sweep_analysis.py`, part 1; Task 9 appends)

```python
"""Learning-stability analysis for the reward-shaping sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.train import resolve_path  # noqa: E402

CURVE_COLUMNS = ["algo", "variant", "seed", "timesteps", "mean_distance_return",
                 "std_distance_return", "mean_total_return", "success_rate"]
METRICS = ["final_return", "final_success", "auc", "steps_to_threshold", "max_drawdown"]


def list_completed_runs(runs_dir) -> list[Path]:
    """Run directories that finished (have done.json), sorted. Crashed runs are not listed."""
    runs_dir = resolve_path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "done.json").exists())


def load_eval_curves(runs_dir) -> pd.DataFrame:
    paths = list_completed_runs(runs_dir)
    if not paths:
        raise ValueError(
            f"no completed runs found in {resolve_path(runs_dir)}; run `python scripts/sweep.py` first"
        )
    frames = []
    for path in paths:
        done = json.loads((path / "done.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(path / "eval.csv")
        frame.insert(0, "seed", done["seed"])
        frame.insert(0, "variant", done["variant"])
        frame.insert(0, "algo", done["algo"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[CURVE_COLUMNS]


def run_metrics(curve: pd.DataFrame, threshold: float, last_k: int = 3) -> dict:
    c = curve.sort_values("timesteps")
    r = c["mean_distance_return"].to_numpy(dtype=float)
    t = c["timesteps"].to_numpy()
    s = c["success_rate"].to_numpy(dtype=float)
    k = min(last_k, len(r))
    reached = np.nonzero(r >= threshold)[0]
    running_max = np.maximum.accumulate(r)
    return {
        "final_return": float(r[-k:].mean()),
        "final_success": float(s[-k:].mean()),
        "auc": float(r.mean()),
        "steps_to_threshold": float(t[reached[0]]) if reached.size else float("nan"),
        "max_drawdown": float((running_max - r).max()),
    }


def per_run_table(curves: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (algo, variant, seed), group in curves.groupby(["algo", "variant", "seed"], sort=True):
        rows.append({"algo": algo, "variant": variant, "seed": seed, **run_metrics(group, threshold)})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """(mean, lo, hi): percentile bootstrap over the sample. NaNs are dropped; crude for few seeds."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return (float(v[0]),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, variant), group in per_run.groupby(["algo", "variant"], sort=True):
        row = {"algo": algo, "variant": variant, "n_seeds": int(len(group)),
               "n_reached": int(group["steps_to_threshold"].notna().sum())}
        for metric in METRICS:
            mean, lo, hi = bootstrap_ci(group[metric].to_numpy())
            row[f"{metric}_mean"], row[f"{metric}_lo"], row[f"{metric}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def compare_algos(per_run: pd.DataFrame, metric: str = "final_return", a: str = "td3", b: str = "ddpg",
                  n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Per variant: mean(a) - mean(b) with an unpaired bootstrap interval (each algo's seeds resampled)."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, group in per_run.groupby("variant", sort=True):
        va = group.loc[group["algo"] == a, metric].dropna().to_numpy(dtype=float)
        vb = group.loc[group["algo"] == b, metric].dropna().to_numpy(dtype=float)
        if va.size == 0 or vb.size == 0:
            continue
        ma = va[rng.integers(0, va.size, size=(n_boot, va.size))].mean(axis=1)
        mb = vb[rng.integers(0, vb.size, size=(n_boot, vb.size))].mean(axis=1)
        diff = ma - mb
        rows.append({"variant": variant, "metric": metric, "diff_mean": float(va.mean() - vb.mean()),
                     "diff_lo": float(np.quantile(diff, alpha / 2)),
                     "diff_hi": float(np.quantile(diff, 1 - alpha / 2)),
                     f"n_{a}": int(va.size), f"n_{b}": int(vb.size)})
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run to verify pass**

Run: `"$PY" -m pytest tests/test_sweep_analysis.py -q` then `"$PY" -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_analysis.py tests/sweep_fakes.py tests/test_sweep_analysis.py
git commit -m "feat: add sweep metrics, bootstrap confidence intervals and algorithm comparison" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Analysis plots and committed outputs

**Files:**
- Modify: `scripts/sweep_analysis.py`
- Test: `tests/test_sweep_analysis.py`

**Interfaces:**
- Consumes: Task 8 functions.
- Produces in `scripts.sweep_analysis`:
  - `bootstrap_band(matrix, n_boot=2000, seed=0, alpha=0.05) -> (mean, lo, hi)` per column of a `(n_seeds, T)` matrix
  - `plot_learning_curves(curves, random_baseline, threshold) -> Figure` (one panel per variant; thin per-seed lines, mean, bootstrap band; algorithms overlaid)
  - `plot_metric_comparison(summary) -> Figure` (final return, final success, max drawdown; grouped by variant, colour = algo, error bars = bootstrap CI)
  - `write_sweep_outputs(runs_dir, out_dir, threshold) -> dict[str, Path]` writing `sweep_eval_curves.csv`, `sweep_per_run.csv`, `sweep_summary.csv`, `sweep_algo_differences.csv`, `sweep_random_baseline.json`, `sweep_learning_curves.png`, `sweep_metric_comparison.png`
  - `main(argv=None)` with `--runs-dir --out`, reading `solve_threshold` from `configs/sweep_config.yaml`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sweep_analysis.py`)

```python
import json

import matplotlib.pyplot as plt

from scripts.sweep_analysis import (
    bootstrap_band,
    plot_learning_curves,
    plot_metric_comparison,
    write_sweep_outputs,
)


def test_bootstrap_band_shapes_and_ordering():
    m = np.array([[-5.0, -4.0], [-6.0, -3.0], [-4.0, -3.5]])
    mean, lo, hi = bootstrap_band(m)
    assert mean.shape == lo.shape == hi.shape == (2,)
    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9)
    assert mean == pytest.approx(m.mean(axis=0))


def grid_runs():
    runs = []
    for algo, base in (("td3", -3.0), ("ddpg", -6.0)):
        for variant in ("default", "sparse"):
            for seed in range(3):
                shift = (1.0 if variant == "sparse" else 0.0) + 0.1 * seed
                runs.append((algo, variant, seed, [-9, -7, base - shift, base - shift], [0, 0.1, 0.5, 0.6]))
    return runs


def test_plots_render_without_error(tmp_path):
    make_fake_runs(tmp_path, grid_runs())
    curves = load_eval_curves(tmp_path)
    fig = plot_learning_curves(curves, -10.0, -4.0)
    assert len(fig.axes) >= 2  # one panel per variant (+ legend)
    plt.close(fig)
    summary = summarize_runs(per_run_table(curves, -4.0))
    fig = plot_metric_comparison(summary)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_write_sweep_outputs_creates_every_file(tmp_path):
    runs = make_fake_runs(tmp_path / "runs", grid_runs())
    paths = write_sweep_outputs(runs, tmp_path / "out", -4.0)
    expected = {"sweep_eval_curves.csv", "sweep_per_run.csv", "sweep_summary.csv",
                "sweep_algo_differences.csv", "sweep_random_baseline.json",
                "sweep_learning_curves.png", "sweep_metric_comparison.png"}
    assert {p.name for p in paths.values()} == expected
    assert all(p.stat().st_size > 0 for p in paths.values())
    curves = pd.read_csv(tmp_path / "out" / "sweep_eval_curves.csv")
    assert set(curves["algo"]) == {"td3", "ddpg"} and len(curves) == 12 * 4
    summary = pd.read_csv(tmp_path / "out" / "sweep_summary.csv")
    assert {"final_return_mean", "final_return_lo", "final_return_hi", "n_reached"} <= set(summary.columns)
    base = json.loads((tmp_path / "out" / "sweep_random_baseline.json").read_text(encoding="utf-8"))
    assert base["mean_distance_return"] == -10.0
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_sweep_analysis.py -q`
Expected: `ImportError: cannot import name 'bootstrap_band'`.

- [ ] **Step 3: Implement** (append to `scripts/sweep_analysis.py`)

Add imports near the top: `import argparse`, `import matplotlib.pyplot as plt`. Then append:

```python
ALGO_COLORS = {"td3": "#1f77b4", "ddpg": "#d62728"}


def bootstrap_band(matrix, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Per-column mean and percentile bootstrap band over the rows (seeds) of an (n_seeds, T) matrix."""
    m = np.asarray(matrix, dtype=float)
    mean = m.mean(axis=0)
    if m.shape[0] == 1:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    boots = m[idx].mean(axis=1)  # (n_boot, T)
    return mean, np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def plot_learning_curves(curves: pd.DataFrame, random_baseline: float, threshold: float):
    variants = list(dict.fromkeys(curves["variant"]))
    ncols = 3
    nrows = int(np.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False, sharey=True)
    for ax in axes.ravel()[len(variants):]:
        ax.axis("off")
    for ax, variant in zip(axes.ravel(), variants):
        for algo in sorted(curves["algo"].unique()):
            sub = curves[(curves["variant"] == variant) & (curves["algo"] == algo)]
            if sub.empty:
                continue
            wide = sub.pivot(index="seed", columns="timesteps", values="mean_distance_return").sort_index(axis=1)
            steps = wide.columns.to_numpy()
            for _, row in wide.iterrows():  # thin per-seed lines
                ax.plot(steps, row.to_numpy(), color=ALGO_COLORS.get(algo), alpha=0.25, lw=0.8)
            mean, lo, hi = bootstrap_band(wide.to_numpy())
            ax.fill_between(steps, lo, hi, color=ALGO_COLORS.get(algo), alpha=0.18)
            ax.plot(steps, mean, color=ALGO_COLORS.get(algo), lw=2, label=f"{algo.upper()} (n={len(wide)})")
        ax.axhline(random_baseline, color="gray", ls=":", label="random policy")
        ax.axhline(threshold, color="orange", ls="--", lw=1, label="solve threshold")
        ax.set_title(variant)
        ax.set_xlabel("environment steps")
    axes[0, 0].set_ylabel("task return: -sum distance\n(higher is better)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves on the unshaped task (thin = seeds, thick = mean, band = 95% bootstrap CI)")
    fig.tight_layout()
    return fig


def plot_metric_comparison(summary: pd.DataFrame):
    panels = [("final_return", "final task return (higher is better)"),
              ("final_success", "final success rate"),
              ("max_drawdown", "max drawdown (lower is better)")]
    variants = list(dict.fromkeys(summary["variant"]))
    algos = sorted(summary["algo"].unique())
    width = 0.8 / len(algos)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(variants))
    for ax, (metric, title) in zip(axes, panels):
        for j, algo in enumerate(algos):
            sub = summary[summary["algo"] == algo].set_index("variant").reindex(variants)
            mean = sub[f"{metric}_mean"].to_numpy(dtype=float)
            err = np.vstack([mean - sub[f"{metric}_lo"].to_numpy(dtype=float),
                             sub[f"{metric}_hi"].to_numpy(dtype=float) - mean])
            ax.bar(x + j * width - 0.4 + width / 2, mean, width, yerr=err, capsize=3,
                   color=ALGO_COLORS.get(algo), label=algo.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20)
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    return fig


def write_sweep_outputs(runs_dir, out_dir, threshold: float) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_eval_curves(runs_dir)
    baseline_path = resolve_path(runs_dir) / "random_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    per_run = per_run_table(curves, threshold)
    summary = summarize_runs(per_run)
    diffs = compare_algos(per_run)

    paths = {
        "curves": out_dir / "sweep_eval_curves.csv",
        "per_run": out_dir / "sweep_per_run.csv",
        "summary": out_dir / "sweep_summary.csv",
        "diffs": out_dir / "sweep_algo_differences.csv",
        "baseline": out_dir / "sweep_random_baseline.json",
        "curves_png": out_dir / "sweep_learning_curves.png",
        "comparison_png": out_dir / "sweep_metric_comparison.png",
    }
    curves.to_csv(paths["curves"], index=False)
    per_run.to_csv(paths["per_run"], index=False)
    summary.to_csv(paths["summary"], index=False)
    diffs.to_csv(paths["diffs"], index=False)
    paths["baseline"].write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    fig = plot_learning_curves(curves, baseline["mean_distance_return"], threshold)
    fig.savefig(paths["curves_png"], dpi=130)
    plt.close(fig)
    fig = plot_metric_comparison(summary)
    fig.savefig(paths["comparison_png"], dpi=130)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scripts.sweep import load_sweep_config

    p = argparse.ArgumentParser(description="Analyse a finished reward-shaping sweep.")
    p.add_argument("--runs-dir", default=None, help="default: runs_dir from the sweep config")
    p.add_argument("--out", default="plots")
    args = p.parse_args(argv)
    cfg = load_sweep_config()
    paths = write_sweep_outputs(args.runs_dir or cfg["runs_dir"], resolve_path(args.out), cfg["solve_threshold"])
    summary = pd.read_csv(paths["summary"])
    cols = ["algo", "variant", "n_seeds", "n_reached", "final_return_mean", "final_success_mean",
            "max_drawdown_mean"]
    print(summary[cols].round(3).to_string(index=False))
    print("Wrote:", ", ".join(p.name for p in paths.values()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Smoke the CLI against the quick sweep from Task 7**

```bash
"$PY" scripts/sweep_analysis.py --runs-dir runs/quick --out plots/quick
```
Expected: prints a summary table (4 rows) and the list of written files; files appear under `plots/quick/` (git-ignored). Open `plots/quick/sweep_learning_curves.png` and check it renders sensibly.

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_analysis.py tests/test_sweep_analysis.py
git commit -m "feat: add sweep learning-curve and comparison plots and output writer" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Robustness of shaped policies

**Files:**
- Create: `scripts/shaping_robustness.py`, `tests/test_shaping_robustness.py`

**Interfaces:**
- Consumes: `scripts.sweep_analysis.{list_completed_runs, bootstrap_ci}`; `scripts.evaluate_ipqc.{STRESS_KWARG, STRESS_XLABEL, load_policy, run_episode_parts, safety_boundary}`; `envs.make_stressed_env`; `scripts.sweep.{load_sweep_config}`; `scripts.train.{load_config, resolve_path}`.
- Produces in `scripts.shaping_robustness`:
  - `robustness_conditions(rob_cfg) -> list[(stress_type, level, order, kwargs)]` (nominal first)
  - `evaluate_policy_under_stress(policy, env_id, conditions, episodes, eval_seed, success_radius) -> list[dict]` rows `stress_type, level, order, mean_distance_return, mean_total_return, success_rate`
  - `evaluate_run(args) -> list[dict]` (top-level and picklable; `args` keys `run_dir, env_id, conditions, episodes, eval_seed, success_radius`; adds `algo, variant, seed`)
  - `score_rows(df, random_distance, solve_threshold) -> DataFrame` adding `nominal_distance, eligible, score`
  - `eligibility_table(scored) -> DataFrame` (`algo, variant, n_policies, n_eligible`)
  - `summarize_robustness(scored) -> DataFrame` (`algo, variant, stress_type, level, order, retention_mean/_lo/_hi, n_eligible`)
  - `policy_boundaries(scored, retention_threshold) -> DataFrame` (`algo, variant, seed, stress_type, boundary_level, reached`) and `summarize_boundaries(boundaries) -> DataFrame`
  - `plot_robustness(summary) -> Figure` (rows = algo, columns = stress type, lines = variant)
  - `run_robustness(sweep_cfg, train_cfg, out_dir, workers=None) -> dict[str, Path]` writing `shaping_robustness.csv`, `shaping_eligibility.csv`, `shaping_robustness_summary.csv`, `shaping_boundaries.csv`, `shaping_boundaries_summary.csv`, `shaping_robustness.png`
  - `main(argv=None)` with `--out --workers --quick`

- [ ] **Step 1: Write the failing tests** (`tests/test_shaping_robustness.py`)

```python
import numpy as np
import pandas as pd
import pytest

from scripts.shaping_robustness import (
    eligibility_table,
    policy_boundaries,
    robustness_conditions,
    score_rows,
    summarize_boundaries,
    summarize_robustness,
)

ROB = {"episodes": 2, "eval_seed": 2000, "degradation": [1.0, 0.6], "latency_steps": [0, 2],
       "deadband": [0.0, 0.1], "retention_threshold": 0.9}


def rows_for(algo, variant, seed, nominal, stressed):
    """Raw rows for one policy: nominal return plus stressed returns {(stress, level, order): value}."""
    rows = [dict(algo=algo, variant=variant, seed=seed, stress_type="nominal", level=0.0, order=0,
                 mean_distance_return=nominal, mean_total_return=nominal, success_rate=1.0)]
    for (stress, level, order), value in stressed.items():
        rows.append(dict(algo=algo, variant=variant, seed=seed, stress_type=stress, level=level,
                         order=order, mean_distance_return=value, mean_total_return=value, success_rate=0.5))
    return rows


def test_conditions_start_with_nominal_then_every_level():
    conds = robustness_conditions(ROB)
    assert conds[0] == ("nominal", 0.0, 0, {})
    assert len(conds) == 1 + 2 + 2 + 2
    assert ("latency", 2.0, 1, {"delay_steps": 2}) in conds
    assert ("degradation", 0.6, 1, {"degradation": 0.6}) in conds
    assert ("deadband", 0.1, 1, {"deadband": 0.1}) in conds


def test_retention_uses_the_policys_own_nominal_return():
    raw = pd.DataFrame(rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -5.0}))
    scored = score_rows(raw, random_distance=-10.0, solve_threshold=-4.0)
    row = scored[scored.stress_type == "latency"].iloc[0]
    assert row["eligible"] and row["score"] == pytest.approx((-5.0 + 10.0) / (-3.0 + 10.0))


def test_ineligible_policy_is_excluded_not_scored():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -4.0})
        + rows_for("td3", "sparse", 0, -9.0, {("latency", 2.0, 1): -9.5})  # never learned
    )
    scored = score_rows(raw, random_distance=-10.0, solve_threshold=-4.0)
    bad = scored[scored.variant == "sparse"]
    assert not bad["eligible"].any() and bad["score"].isna().all()
    summary = summarize_robustness(scored)
    assert set(summary["variant"]) == {"default"}  # excluded from the comparison
    table = eligibility_table(scored).set_index("variant")
    assert table.loc["default", "n_eligible"] == 1 and table.loc["sparse", "n_eligible"] == 0
    assert table.loc["sparse", "n_policies"] == 1  # but still counted, never silently dropped


def test_summary_aggregates_over_eligible_seeds_with_interval():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -4.0})
        + rows_for("td3", "default", 1, -3.0, {("latency", 2.0, 1): -5.0})
    )
    s = summarize_robustness(score_rows(raw, -10.0, -4.0))
    row = s[(s.stress_type == "latency")].iloc[0]
    scores = [(-4 + 10) / 7, (-5 + 10) / 7]
    assert row["n_eligible"] == 2 and row["retention_mean"] == pytest.approx(np.mean(scores))
    assert row["retention_lo"] <= row["retention_mean"] <= row["retention_hi"]


def test_boundaries_use_first_failure_and_skip_ineligible():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 0.0, 0): -3.0, ("latency", 2.0, 1): -3.2,
                                              ("latency", 4.0, 2): -9.0})
        + rows_for("td3", "sparse", 0, -9.0, {("latency", 0.0, 0): -9.0, ("latency", 2.0, 1): -9.0})
    )
    b = policy_boundaries(score_rows(raw, -10.0, -4.0), retention_threshold=0.9)
    assert set(b["variant"]) == {"default"}
    row = b[b.stress_type == "latency"].iloc[0]
    assert row["boundary_level"] == 2.0 and bool(row["reached"])
    summary = summarize_boundaries(b)
    assert summary.iloc[0]["n_eligible"] == 1


def test_end_to_end_on_tiny_trained_policies(tmp_path):
    pytest.importorskip("mujoco")
    from scripts.shaping_robustness import run_robustness
    from scripts.sweep import load_sweep_config, make_spec, quick_config, run_one
    from scripts.train import load_config

    train_cfg = load_config()
    cfg = quick_config(load_sweep_config())
    cfg.update(runs_dir=str(tmp_path / "runs"), solve_threshold=-1000.0, random_baseline_episodes=3,
               robustness={**ROB, "latency_steps": [0, 2]})
    for algo in ("td3", "ddpg"):
        run_one(make_spec(cfg, train_cfg, {"algo": algo, "variant": "default", "seed": 0}))
    import json

    (tmp_path / "runs" / "random_baseline.json").write_text(json.dumps(
        {"mean_distance_return": -1500.0, "std_distance_return": 1.0, "episodes": 3}), encoding="utf-8")
    paths = run_robustness(cfg, train_cfg, tmp_path / "out", workers=1)
    raw = pd.read_csv(paths["raw"])
    assert set(raw["algo"]) == {"td3", "ddpg"} and "nominal" in set(raw["stress_type"])
    assert len(raw) == 2 * (1 + 2 + 2 + 2)
    assert paths["png"].stat().st_size > 0 and paths["boundaries"].exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `"$PY" -m pytest tests/test_shaping_robustness.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.shaping_robustness'`.

- [ ] **Step 3: Implement** (`scripts/shaping_robustness.py`)

```python
"""Do reward-shaped policies differ in how they cope with the IPQC failure modes?

Every solved final policy from the sweep is evaluated under a reduced stress battery on the
unshaped task. Policies that never learned (own nominal return below the solve threshold) are
excluded from the comparison and counted, never scored.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from envs import make_stressed_env  # noqa: E402
from scripts.evaluate_ipqc import (  # noqa: E402
    STRESS_KWARG,
    STRESS_XLABEL,
    load_policy,
    run_episode_parts,
    safety_boundary,
)
from scripts.sweep_analysis import bootstrap_ci, list_completed_runs  # noqa: E402
from scripts.train import load_config, resolve_path  # noqa: E402

CONFIG_KEY = {"degradation": "degradation", "latency": "latency_steps", "deadband": "deadband"}
VARIANT_COLORS = {"default": "#1f77b4", "dist_only": "#ff7f0e", "sparse": "#7f7f7f",
                  "progress": "#2ca02c", "scaled_x10": "#9467bd"}


def robustness_conditions(rob_cfg: dict) -> list:
    conds = [("nominal", 0.0, 0, {})]
    for stress_type, key in CONFIG_KEY.items():
        for order, level in enumerate(rob_cfg[key]):
            conds.append((stress_type, float(level), order, {STRESS_KWARG[stress_type]: level}))
    return conds


def evaluate_policy_under_stress(policy, env_id, conditions, episodes, eval_seed, success_radius) -> list[dict]:
    rows = []
    for stress_type, level, order, kwargs in conditions:
        env = make_stressed_env(env_id, **kwargs)
        parts = [run_episode_parts(env, policy, seed=eval_seed if i == 0 else None) for i in range(episodes)]
        env.close()
        rows.append({
            "stress_type": stress_type, "level": level, "order": order,
            "mean_distance_return": float(np.mean([p["distance_return"] for p in parts])),
            "mean_total_return": float(np.mean([p["episode_return"] for p in parts])),
            "success_rate": float(np.mean([p["final_distance"] <= success_radius for p in parts])),
        })
    return rows


def evaluate_run(args: dict) -> list[dict]:
    """Top-level (picklable) worker: load one run's final model and stress-test it."""
    import torch

    torch.set_num_threads(1)
    run_dir = Path(args["run_dir"])
    done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    policy = load_policy(run_dir / done["final_model"], algo=done["algo"])
    rows = evaluate_policy_under_stress(policy, args["env_id"], args["conditions"], args["episodes"],
                                        args["eval_seed"], args["success_radius"])
    return [{"algo": done["algo"], "variant": done["variant"], "seed": done["seed"], **r} for r in rows]


def score_rows(df: pd.DataFrame, random_distance: float, solve_threshold: float) -> pd.DataFrame:
    """Retention (R - R_random)/(R_own_nominal - R_random); ineligible policies get NaN."""
    if solve_threshold <= random_distance:
        raise ValueError(
            f"solve_threshold ({solve_threshold}) must be above the random-policy return ({random_distance})"
        )
    keys = ["algo", "variant", "seed"]
    nominal = (df[df.stress_type == "nominal"].set_index(keys)["mean_distance_return"]
               .rename("nominal_distance").reset_index())
    out = df.merge(nominal, on=keys, how="left")
    out["eligible"] = out["nominal_distance"] >= solve_threshold
    retention = (out["mean_distance_return"] - random_distance) / (out["nominal_distance"] - random_distance)
    out["score"] = np.where(out["eligible"], retention, np.nan)
    return out


def eligibility_table(scored: pd.DataFrame) -> pd.DataFrame:
    per_policy = scored.drop_duplicates(["algo", "variant", "seed"])
    return (per_policy.groupby(["algo", "variant"], as_index=False)
            .agg(n_policies=("seed", "count"), n_eligible=("eligible", "sum")))


def summarize_robustness(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stressed = scored[(scored.stress_type != "nominal") & scored["eligible"]]
    for (algo, variant, stress, level, order), g in stressed.groupby(
            ["algo", "variant", "stress_type", "level", "order"], sort=True):
        mean, lo, hi = bootstrap_ci(g["score"].to_numpy())
        rows.append({"algo": algo, "variant": variant, "stress_type": stress, "level": level, "order": order,
                     "retention_mean": mean, "retention_lo": lo, "retention_hi": hi,
                     "n_eligible": int(len(g))})
    return pd.DataFrame(rows)


def policy_boundaries(scored: pd.DataFrame, retention_threshold: float) -> pd.DataFrame:
    rows = []
    stressed = scored[(scored.stress_type != "nominal") & scored["eligible"]]
    for (algo, variant, seed), g in stressed.groupby(["algo", "variant", "seed"], sort=True):
        for stress_type in sorted(g.stress_type.unique()):
            b = safety_boundary(g, stress_type, retention_threshold)
            rows.append({"algo": algo, "variant": variant, "seed": seed, "stress_type": stress_type,
                         "boundary_level": b["level"], "reached": b["reached"]})
    return pd.DataFrame(rows)


def summarize_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame(columns=["algo", "variant", "stress_type", "boundary_mean", "n_eligible",
                                     "n_not_reached"])
    return (boundaries.groupby(["algo", "variant", "stress_type"], as_index=False)
            .agg(boundary_mean=("boundary_level", "mean"), n_eligible=("seed", "count"),
                 n_not_reached=("reached", lambda s: int((~s.astype(bool)).sum()))))


def plot_robustness(summary: pd.DataFrame):
    algos = sorted(summary["algo"].unique())
    stresses = ["degradation", "latency", "deadband"]
    fig, axes = plt.subplots(len(algos), 3, figsize=(15, 3.6 * len(algos)), squeeze=False, sharey=True)
    for i, algo in enumerate(algos):
        for j, stress in enumerate(stresses):
            ax = axes[i, j]
            for variant in dict.fromkeys(summary["variant"]):
                g = summary[(summary.algo == algo) & (summary.stress_type == stress)
                            & (summary.variant == variant)].sort_values("order")
                if g.empty:
                    continue
                c = VARIANT_COLORS.get(variant)
                ax.plot(g["level"], g["retention_mean"], marker="o", color=c,
                        label=f"{variant} (n={int(g['n_eligible'].min())})")
                ax.fill_between(g["level"], g["retention_lo"], g["retention_hi"], color=c, alpha=0.15)
            ax.axhline(0.9, color="orange", ls="--", lw=1)
            ax.axhline(0.0, color="red", ls=":", lw=1)
            levels = summary[summary.stress_type == stress]["level"]
            if stress == "degradation" and len(levels):
                ax.invert_xaxis()
            ax.set_title(f"{algo.upper()}: {stress}")
            ax.set_xlabel(STRESS_XLABEL[stress])
    axes[0, 0].set_ylabel("task retention\n(1 = healthy robot, 0 = random)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Robustness of solved policies by training reward (mean over eligible seeds, 95% bootstrap CI)")
    fig.tight_layout()
    return fig


def run_robustness(sweep_cfg: dict, train_cfg: dict, out_dir, workers: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = resolve_path(sweep_cfg["runs_dir"])
    runs = list_completed_runs(runs_dir)
    if not runs:
        raise ValueError(f"no completed runs in {runs_dir}; run `python scripts/sweep.py` first")
    rob = sweep_cfg["robustness"]
    conditions = robustness_conditions(rob)
    random_distance = json.loads((runs_dir / "random_baseline.json").read_text(encoding="utf-8"))["mean_distance_return"]
    jobs = [{"run_dir": str(p), "env_id": train_cfg["env_id"], "conditions": conditions,
             "episodes": rob["episodes"], "eval_seed": rob["eval_seed"],
             "success_radius": train_cfg["success_radius"]} for p in runs]
    workers = workers or sweep_cfg["workers"]
    rows: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            rows.extend(evaluate_run(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(evaluate_run, jobs):
                rows.extend(result)

    scored = score_rows(pd.DataFrame(rows), random_distance, sweep_cfg["solve_threshold"])
    summary = summarize_robustness(scored)
    boundaries = policy_boundaries(scored, rob["retention_threshold"])
    paths = {
        "raw": out_dir / "shaping_robustness.csv",
        "eligibility": out_dir / "shaping_eligibility.csv",
        "summary": out_dir / "shaping_robustness_summary.csv",
        "boundaries": out_dir / "shaping_boundaries.csv",
        "boundaries_summary": out_dir / "shaping_boundaries_summary.csv",
        "png": out_dir / "shaping_robustness.png",
    }
    scored.to_csv(paths["raw"], index=False)
    eligibility_table(scored).to_csv(paths["eligibility"], index=False)
    summary.to_csv(paths["summary"], index=False)
    boundaries.to_csv(paths["boundaries"], index=False)
    summarize_boundaries(boundaries).to_csv(paths["boundaries_summary"], index=False)
    if summary.empty:
        raise ValueError("no policy reached the solve threshold, so there is nothing to compare")
    fig = plot_robustness(summary)
    fig.savefig(paths["png"], dpi=130)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scripts.sweep import load_sweep_config, quick_config

    p = argparse.ArgumentParser(description="Stress-test the final policies of a finished sweep.")
    p.add_argument("--out", default="plots")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="read runs/quick (written by sweep.py --quick)")
    args = p.parse_args(argv)
    sweep_cfg = load_sweep_config()
    if args.quick:
        sweep_cfg = quick_config(sweep_cfg)
    out = resolve_path(args.out if not args.quick else "plots/quick")
    paths = run_robustness(sweep_cfg, load_config(), out, workers=args.workers)
    print(pd.read_csv(paths["eligibility"]).to_string(index=False))
    print(pd.read_csv(paths["boundaries_summary"]).round(3).to_string(index=False))
    print("Wrote:", ", ".join(p.name for p in paths.values()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `"$PY" -m pytest tests/test_shaping_robustness.py -q` then `"$PY" -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Smoke the CLI against the quick sweep from Task 7**

```bash
"$PY" scripts/shaping_robustness.py --quick
```
Expected: prints an eligibility table and a boundaries summary (quick-trained policies are unlikely to be "solved", so with the real `solve_threshold` of `-4.0` the command may raise `no policy reached the solve threshold`; that is the intended message, not a bug). Either outcome proves the pipeline runs; any other traceback is a bug.

- [ ] **Step 6: Commit**

```bash
git add scripts/shaping_robustness.py tests/test_shaping_robustness.py
git commit -m "feat: add stress test of solved policies by training reward" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Notebook 03 and the notebook-01 text fix

**Files:**
- Create: `notebooks/03_reward_shaping_sweep.ipynb`
- Modify: `notebooks/01_train_td3.ipynb` (markdown text only)
- Generator (throwaway, not committed): `<scratchpad>/build_notebook3.py`

**Interfaces:**
- Consumes: `scripts.train.load_config`; `scripts.sweep.{load_sweep_config, quick_config, run_sweep}`; `scripts.sweep_analysis.{write_sweep_outputs, plot_learning_curves, plot_metric_comparison}`; `scripts.shaping_robustness.{run_robustness, plot_robustness}`.
- Behaviour: with `DFC_QUICK=1` the notebook runs a tiny sweep into `runs/quick`, writes outputs to `plots/quick/`, and renders them. Without it, the notebook does NOT launch the multi-hour sweep; it loads the committed CSVs in `plots/` and tells the reader to run `python scripts/sweep.py` if they are missing.

- [ ] **Step 1: Write the generator** (run it from the repo root; it writes relative to `notebooks/`)

```python
import nbformat as nbf
from pathlib import Path

NB_DIR = Path("notebooks")

SETUP = '''import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "envs" / "stress_wrappers.py").exists())
sys.path.insert(0, str(ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

QUICK = os.environ.get("DFC_QUICK", "0") == "1"  # DFC_QUICK=1 -> tiny sweep, for smoke-testing only
PLOTS = ROOT / "plots" / ("quick" if QUICK else "")
PLOTS.mkdir(parents=True, exist_ok=True)
print("project root :", ROOT.name)
print("quick mode   :", QUICK)'''

LOAD = '''from scripts.train import load_config
from scripts.sweep import load_sweep_config, quick_config, run_sweep
from scripts.sweep_analysis import plot_learning_curves, plot_metric_comparison, write_sweep_outputs
from scripts.shaping_robustness import plot_robustness, run_robustness

train_cfg = load_config()
sweep_cfg = load_sweep_config()
threshold = sweep_cfg["solve_threshold"]

if QUICK:
    sweep_cfg = quick_config(sweep_cfg)           # 2 algos x 2 variants x 1 seed, ~800 steps
    run_sweep(sweep_cfg, train_cfg, log=print)    # resumable
    write_sweep_outputs(sweep_cfg["runs_dir"], PLOTS, threshold)
    try:
        run_robustness(sweep_cfg, train_cfg, PLOTS, workers=1)
    except ValueError as err:                     # quick policies rarely reach the solve threshold
        print("robustness skipped in quick mode:", err)
else:
    needed = ["sweep_eval_curves.csv", "sweep_summary.csv", "sweep_random_baseline.json"]
    missing = [n for n in needed if not (PLOTS / n).exists()]
    assert not missing, f"missing {missing} in plots/: run `python scripts/sweep.py` then `python scripts/sweep_analysis.py`"

curves = pd.read_csv(PLOTS / "sweep_eval_curves.csv")
summary = pd.read_csv(PLOTS / "sweep_summary.csv")
random_baseline = json.loads((PLOTS / "sweep_random_baseline.json").read_text())["mean_distance_return"]
print(f"{curves.groupby(['algo', 'variant', 'seed']).ngroups} runs loaded; random policy = {random_baseline:.2f}")'''

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python (ai_portfolio)", "language": "python", "name": "ai_portfolio"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 03 · Reward shaping and learning stability: TD3 vs DDPG\n\n"
       "How does the reward the agent *trains on* change learning stability, and does it change how the finished "
       "policy copes with failures? We train **TD3 and DDPG** with identical hyperparameters on five reward variants "
       "(`default`, `dist_only`, `sparse`, `progress`, `scaled_x10`) across 5 seeds, then stress-test every solved policy.\n\n"
       "Everything is judged on the **unshaped task**: summed negative distance to the target (higher is better) and "
       "success rate (fingertip within 5 cm at the end). Shaped training returns are never compared, because each "
       "variant uses a different reward scale.\n\n"
       "The full sweep takes hours, so this notebook does not run it: it loads the committed results in `plots/`. "
       "To reproduce, run `python scripts/sweep.py`, `python scripts/sweep_analysis.py`, `python scripts/shaping_robustness.py`. "
       "Set `DFC_QUICK=1` for a tiny smoke-test sweep."),
    code(SETUP),
    md("## Load results"),
    code(LOAD),
    md("## Stability table\nMeans over seeds with 95% bootstrap CIs (crude with 5 seeds). `n_reached` = seeds that reached the solve threshold."),
    code('''cols = ["algo", "variant", "n_seeds", "n_reached", "final_return_mean", "final_return_lo", "final_return_hi",
        "final_success_mean", "auc_mean", "steps_to_threshold_mean", "max_drawdown_mean"]
display(summary[cols].round(3))'''),
    md("## Learning curves\nThin lines are individual seeds, thick lines the mean, bands the 95% bootstrap CI."),
    code('''fig = plot_learning_curves(curves, random_baseline, threshold)
plt.show()'''),
    md("## Metric comparison"),
    code('''fig = plot_metric_comparison(summary)
plt.show()
diffs_path = PLOTS / "sweep_algo_differences.csv"
if diffs_path.exists():
    display(pd.read_csv(diffs_path).round(3))'''),
    md("## Robustness of the solved policies\nEach solved final policy is stress-tested with a reduced IPQC battery. "
       "Policies that never reached the solve threshold are excluded (see the eligibility table), never scored."),
    code('''rob_paths = {n: PLOTS / n for n in ["shaping_eligibility.csv", "shaping_robustness_summary.csv", "shaping_boundaries_summary.csv"]}
if all(p.exists() for p in rob_paths.values()):
    display(pd.read_csv(rob_paths["shaping_eligibility.csv"]))
    rob_summary = pd.read_csv(rob_paths["shaping_robustness_summary.csv"])
    plot_robustness(rob_summary)
    plt.show()
    display(pd.read_csv(rob_paths["shaping_boundaries_summary.csv"]).round(3))
else:
    print("no robustness results yet: run `python scripts/shaping_robustness.py`")'''),
    md("## Findings\n\n*(written after the full run; see the README for the discussion)*"),
]
nbf.write(nb, NB_DIR / "03_reward_shaping_sweep.ipynb")
print("notebook 03 written")
```

- [ ] **Step 2: Fix the notebook-01 timing text**

In `notebooks/01_train_td3.ipynb` change the markdown sentence `150k steps takes on the order of 10-20 minutes on the RTX 3070 Ti.` to `150k steps took about 37 minutes on an RTX 3070 Ti laptop.` (edit the JSON cell source with a small Python script or the notebook editor; do not re-execute the notebook, its stored outputs stay).

- [ ] **Step 3: Generate and smoke-execute notebook 03 in quick mode (copy to scratch, original stays clean)**

```bash
PY="<venv>/Scripts/python.exe"
"$PY" <scratchpad>/build_notebook3.py
DFC_QUICK=1 "$PY" -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=ai_portfolio \
  --output-dir <scratchpad>/nbout notebooks/03_reward_shaping_sweep.ipynb
```
Expected: converts without error (it runs a 4-run quick sweep, about 1 minute; the robustness cell prints `robustness skipped in quick mode: ...` if no quick policy is solved). `plots/quick/` holds the sweep CSV/PNG outputs.

- [ ] **Step 4: Commit the output-free notebooks**

```bash
git add notebooks
git commit -m "feat: add reward-shaping sweep notebook; correct notebook 01 timing text" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: The real run (sweep, robustness, analysis) and results

**Files:**
- Commit: `plots/sweep_*.csv|png|json`, `plots/shaping_*.csv|png`; `notebooks/03_reward_shaping_sweep.ipynb` (executed in place)

- [ ] **Step 1: Full test suite and quick end-to-end**

```bash
"$PY" -m pytest -q
rm -rf runs/quick plots/quick
"$PY" scripts/sweep.py --quick
```
Expected: all tests pass; the quick sweep finishes.

- [ ] **Step 2: Launch the full sweep in the background**

```bash
"$PY" scripts/sweep.py > runs_sweep.log 2>&1
```
Run it with `run_in_background`. Expected: `50 runs in the grid, 0 already done, 50 to run`, then 50 `finished in N min` lines over roughly the estimate from Task 1 Step 5. It is resumable: if interrupted, re-run the same command. While waiting, do not touch `runs/`. Do not poll; the harness notifies on completion.

- [ ] **Step 3: Check the sweep finished cleanly**

```bash
ls runs | head -60
tail -5 runs_sweep.log
```
Expected: 50 run directories plus `random_baseline.json`, each run directory containing `done.json`, `eval.csv`, `final_model.zip`, `tb/`. If any run failed, the log names it: re-launch the same command to retry only the unfinished runs.

- [ ] **Step 4: Analyse**

```bash
"$PY" scripts/sweep_analysis.py
```
Expected: a 10-row summary table (2 algos x 5 variants, `n_seeds` 5) and the files written to `plots/`. Open `plots/sweep_learning_curves.png` and `plots/sweep_metric_comparison.png`.

- [ ] **Step 5: Run the robustness test**

```bash
"$PY" scripts/shaping_robustness.py
```
Expected: an eligibility table (how many of the 5 seeds per algo/variant reached the solve threshold) and a boundaries summary; files written to `plots/shaping_*`. Record which variants (if any) had zero eligible policies.

- [ ] **Step 6: Sanity-check the results before writing anything**

- Check `plots/sweep_per_run.csv` for 50 rows and plausible values (the unshaped task return should lie between about -43 and 0 in total terms and between about -10 and 0 for distance).
- The `default` TD3 variant should land near the earlier trained policy (about -3 on the task measure by 40k steps). If it is far off, investigate before proceeding.
- Check the random baseline in `plots/sweep_random_baseline.json` is near -9.8.
- Do not write explanations for *why* a variant behaves as it does unless an experiment tested it; label them hypotheses.

- [ ] **Step 7: Write the notebook's findings and execute it in place**

Replace the placeholder "Findings" markdown cell of notebook 03 with 4-8 plain statements that are directly supported by the tables and plots (numbers copied from the CSVs, including negative results and any variant that never solved). Then:

```bash
"$PY" -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=ai_portfolio \
  notebooks/03_reward_shaping_sweep.ipynb
```
Then remove machine-specific paths from the stored outputs (replace the venv `site-packages` path prefix in warning text with `site-packages\` and the repo absolute path with `deep-feedback-control`), and verify with a grep that no `C:\Users` or `AppData` string remains in the notebook.

- [ ] **Step 8: Commit**

```bash
git add plots notebooks
git status --short   # confirm no runs/, models/, *.zip or the log file is staged
git commit -m "docs: add DDPG and reward-shaping sweep results, robustness analysis and notebook 03" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
(`runs_sweep.log` is a scratch file: delete it, or add it to `.gitignore`, rather than committing it.)

---

### Task 13: README and final checks

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`**

- Add a section "Reward shaping and learning stability (DDPG vs TD3)" after the IPQC results, with: a one-paragraph description of the five variants and the fair-evaluation rule; the learning-curve and comparison figures (`plots/sweep_learning_curves.png`, `plots/sweep_metric_comparison.png`); the stability table (final return with CI, final success rate, seeds that reached the threshold, drawdown) copied from `plots/sweep_summary.csv`; the TD3-minus-DDPG differences from `plots/sweep_algo_differences.csv`; and the robustness result (figure `plots/shaping_robustness.png`, the eligibility table, and the boundary summary), stating explicitly which variants were excluded for not learning.
- Report only what the CSVs show; label every "why" as an untested hypothesis. Include negative results (for example a variant that failed to learn).
- Update the "How it works" table with the new files (`envs/reward_shaping.py`, `scripts/sweep.py`, `scripts/sweep_analysis.py`, `scripts/shaping_robustness.py`, notebook 03).
- Replace the Reproduce section with cross-platform commands (`python -m venv .venv`, a one-line activation note for Windows and macOS/Linux, then `pip install torch --index-url ...`, `pip install -r requirements.txt`, `python -m ipykernel install --user --name ai_portfolio`) and add the sweep commands (`python scripts/sweep.py [--quick]`, `python scripts/sweep_analysis.py`, `python scripts/shaping_robustness.py`); state that `training_curve.png` comes from notebook 01; update the test count.
- Extend the Limitations list: 5 seeds and 40k steps, crude bootstrap CIs, one task, my choice of variants, the robustness battery's reduced levels and 10 episodes per condition, the CPU-only sweep, and W&B still untested live.

- [ ] **Step 2: Verify every README number against the CSVs**

For each number written in the README, open the source CSV and compare. Fix any mismatch.

- [ ] **Step 3: Final verification**

```bash
"$PY" -m pytest -q
git status --short
grep -rnE "C:[/\\]Users|AppData|dca33f4e" --exclude-dir=.git --exclude-dir=runs --exclude-dir=models --exclude="*.png" -I . | head
```
Expected: all tests pass; nothing unstaged except ignored files; the grep prints nothing.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: report the DDPG and reward-shaping sweep and update the README" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Ask the user before pushing**

Report what ran (including the measured sweep wall-clock versus the estimate and any run that had to be retried), what the headline results are, and any negative results. Then ask whether to push to the remote.