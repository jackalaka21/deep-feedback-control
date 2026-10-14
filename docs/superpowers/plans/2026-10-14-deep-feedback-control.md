> **Repo note (2026-10-14):** this plan was written and executed while both portfolio projects shared one repo. Project 1 now lives in its own repo, so every `deep-feedback-control/` path prefix below means the repo root; `docs/` here is `docs/` at the root. Project 2 (`unified-risk-engine`) gets its own repo, spec copy and plan.

# Deep Feedback Control (Project 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a TD3 policy on MuJoCo Reacher, stress-test it with latency / deadband / motor-degradation wrappers, and deliver the whole workflow as scripts plus two Jupyter notebooks with plotted results.

**Architecture:** Reusable logic lives in `envs/stress_wrappers.py` (Gymnasium wrappers + env factories), `scripts/train.py` (config, training, optional W&B) and `scripts/evaluate_ipqc.py` (sweep, normalised scoring, safety boundary, plots). The two notebooks import those functions, so scripts and notebooks never duplicate logic. A `DFC_QUICK=1` switch shrinks every budget so the notebooks can be smoke-tested headlessly before the full run.

**Tech Stack:** Python 3.14 venv, PyTorch (CUDA), gymnasium[mujoco], stable-baselines3 (TD3), pandas, matplotlib, PyYAML, wandb (optional), pytest, ipykernel / nbformat / nbconvert.

**Spec:** `docs/superpowers/specs/2026-10-14-ai-engineering-portfolio-design.md` (sections 2, 3 and 7 apply to this plan).

## Global Constraints

- Repo root: `C:\Users\User\OneDrive\Documents\Coding Projects\ai-engineering-portfolio`. Project 1 lives in `deep-feedback-control/`.
- Python 3.14 venv at `C:\Users\User\.venvs\ai_portfolio` (outside OneDrive; 3.12.0 was rejected because its `zipfile` bug breaks SB3 model loading). In bash calls use `PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"`, set again in every call.
- TD3 config values: learning rate `1e-3`, batch size `256`, device `cuda`, `150000` timesteps, action-noise sigma `0.1`.
- Env id `Reacher-v4`; if the installed Gymnasium no longer provides it, fall back to `Reacher-v5` with a warning and record that in the README.
- Actuator deadband threshold `0.05` (components with |a| < 0.05 are zeroed). Motor degradation levels `1.0, 0.8, 0.6, 0.4` (plus `0.2`). Latency `0` to `5` steps.
- Safety boundary = most severe stress level reached before the retention score `(R - R_random) / (R_nominal - R_random)` first drops below `0.9` (spec section 7.3).
- W&B is optional and OFF by default. Never log in to W&B or any service on the user's behalf.
- Plots written by scripts and notebooks: `plots/degradation_curve.png`, `plots/latency_curve.png`, `plots/deadband_curve.png`, `plots/ipqc_results.csv`, `plots/training_curve.png`. Quick-mode outputs go to `plots/quick/` and `models/*_quick*` (git-ignored).
- Do NOT push to GitHub without asking the user first (the remote is `jackalaka21/ai-engineering-portfolio`, private; pushing needs the `jackalaka21` token, not the default gh account).
- End every commit message with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (second `-m` argument).
- All paths used by scripts are resolved relative to the project root, never the current working directory.

## Review Focus

Failure modes the spec implies but a first implementation would likely miss, most likely first. Each has a test in the task that owns the code.

1. **Negative returns break "percent of nominal".** If nominal is not better than random (e.g. a barely-trained model), scoring must fail loudly instead of producing nonsense. (Task 4: `test_summarize_raises_when_nominal_not_better_than_random`)
2. **SB3 hands wrappers float32 arrays of shape (2,) and may reuse them.** Wrappers must preserve dtype and must not mutate the caller's array. (Task 2: `test_deadband_preserves_dtype_and_does_not_mutate_input`)
3. **Evaluating before training.** A missing model file must produce an error telling the user to train first, not a raw SB3 traceback. (Task 4: `test_load_policy_missing_model_points_to_training`)
4. **`device: cuda` on a machine where CUDA is unavailable.** Must raise a clear message naming the `--device cpu` escape hatch. (Task 3: `test_resolve_device_cuda_unavailable`)
5. **Running from a different working directory** (notebooks run from `notebooks/`, scripts from anywhere). Config and output paths must still resolve. (Task 3: `test_paths_do_not_depend_on_cwd`)

---

## File Structure

```
ai-engineering-portfolio/
├── .gitignore                         (Task 1)
├── requirements.txt                   (Task 1)
├── README.md                          (Task 6)
└── deep-feedback-control/
    ├── conftest.py                    (Task 1: makes project root importable, forces Agg backend)
    ├── pytest.ini                     (Task 1)
    ├── README.md                      (Task 6)
    ├── configs/td3_config.yaml        (Task 3)
    ├── envs/__init__.py               (Task 2)
    ├── envs/stress_wrappers.py        (Task 2)
    ├── models/.gitkeep                (Task 1)
    ├── scripts/__init__.py            (Task 3)
    ├── scripts/train.py               (Task 3)
    ├── scripts/evaluate_ipqc.py       (Task 4)
    ├── notebooks/01_train_td3.ipynb   (Task 5, executed in Task 6)
    ├── notebooks/02_ipqc_stress_test.ipynb (Task 5, executed in Task 6)
    ├── plots/                         (final PNG/CSV committed in Task 6)
    └── tests/
        ├── test_stress_wrappers.py    (Task 2)
        ├── test_train_config.py       (Task 3)
        ├── test_ipqc_eval.py          (Task 4)
        └── test_dfc_smoke.py          (Task 3)
```

---

### Task 1: Environment and scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `deep-feedback-control/{conftest.py,pytest.ini,models/.gitkeep}`

**Interfaces:**
- Produces: a working interpreter at `C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe` with CUDA PyTorch, MuJoCo Reacher, SB3, pytest and a registered Jupyter kernel named `ai_portfolio`.

- [ ] **Step 1: Create the venv**

```bash
py -3.14 -m venv "C:/Users/User/.venvs/ai_portfolio"
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
"$PY" -m pip install --upgrade pip
"$PY" --version
```
Expected: `Python 3.14.x`.

- [ ] **Step 2: Install CUDA PyTorch**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
nvidia-smi | head -5   # note the "CUDA Version" the driver supports
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```
If no matching wheel is found, try `cu126`, then `cu130`, choosing one not newer than the driver's CUDA version.

- [ ] **Step 3: Verify the GPU is visible**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `... True NVIDIA GeForce RTX 3070 Ti Laptop GPU`. If `False`, stop and fix the install (wrong index or old driver) before continuing.

- [ ] **Step 4: Write `requirements.txt`**

```text
# PyTorch must be installed first from the CUDA index, see README:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
torch
gymnasium[mujoco]
stable-baselines3
numpy
pandas
matplotlib
pyyaml
scikit-learn
wandb
ipykernel
nbformat
nbconvert
pytest
```

- [ ] **Step 5: Install the rest and register the kernel**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
"$PY" -m pip install -r requirements.txt
"$PY" -m ipykernel install --user --name ai_portfolio --display-name "Python (ai_portfolio)"
```
Expected: installs without error; `Installed kernelspec ai_portfolio ...`.

- [ ] **Step 6: Verify MuJoCo Reacher works headlessly**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
"$PY" -c "
import gymnasium as gym
for env_id in ('Reacher-v4', 'Reacher-v5'):
    try:
        e = gym.make(env_id); o, _ = e.reset(seed=0)
        print(env_id, 'OK obs', o.shape, 'act', e.action_space)
    except Exception as ex:
        print(env_id, 'FAILED', type(ex).__name__)
print('gymnasium', gym.__version__)
"
```
Expected: at least one of the two prints `OK obs (10,) act Box(-1.0, 1.0, (2,), float32)`. Note which ones work; if only v5, the fallback in Task 2 will be exercised.

- [ ] **Step 7: Write `.gitignore`**

```gitignore
# python
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.venv/
venv/

# project 1
deep-feedback-control/models/*
!deep-feedback-control/models/.gitkeep
deep-feedback-control/plots/quick/

# project 2 (data and checkpoints)
unified-risk-engine/data/*
!unified-risk-engine/data/.gitkeep
*.pt

# tooling
wandb/
```

- [ ] **Step 8: Write `deep-feedback-control/conftest.py` and `pytest.ini`**

`conftest.py` (its presence at the project root also puts the root on `sys.path` for tests):
```python
import matplotlib

matplotlib.use("Agg")
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 9: Create the empty models dir marker and commit**

```bash
mkdir -p deep-feedback-control/models deep-feedback-control/tests
touch deep-feedback-control/models/.gitkeep
git add .gitignore requirements.txt deep-feedback-control
git commit -m "chore: scaffold project 1 environment and test config" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Stress wrappers

**Files:**
- Create: `deep-feedback-control/envs/__init__.py`, `deep-feedback-control/envs/stress_wrappers.py`
- Test: `deep-feedback-control/tests/test_stress_wrappers.py`

**Interfaces:**
- Produces (all in `envs`):
  - `ObservationLatency(env, delay_steps: int = 0)` returns the observation from `delay_steps` steps earlier, clamped to the first observation of the episode.
  - `ActuatorDeadband(env, threshold: float = 0.05)` zeroes action components with `abs(a) < threshold`.
  - `MotorDegradation(env, scale: float = 1.0)` multiplies actions by `scale` (valid range `0..1`).
  - `make_base_env(env_id: str, **kwargs) -> gym.Env` (falls back `-v4` to `-v5`).
  - `make_stressed_env(env_id: str, delay_steps: int = 0, deadband: float = 0.0, degradation: float = 1.0, **kwargs) -> gym.Env`. Order: the policy's action hits the deadband first, then degradation scales what is delivered.

- [ ] **Step 1: Write the failing tests** (`tests/test_stress_wrappers.py`)

```python
import gymnasium as gym
import numpy as np
import pytest

from envs import (
    ActuatorDeadband,
    MotorDegradation,
    ObservationLatency,
    make_base_env,
    make_stressed_env,
)


class CounterEnv(gym.Env):
    """Observation at time t is [t]; records every action it receives."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float64)

    def __init__(self):
        self.t = 0
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0]), {}

    def step(self, action):
        self.actions.append(np.array(action))
        self.t += 1
        return np.array([float(self.t)]), 0.0, False, False, {}


def delivered(env):
    return env.unwrapped.actions[-1]


def test_latency_zero_is_passthrough():
    env = ObservationLatency(CounterEnv(), delay_steps=0)
    env.reset()
    assert [env.step(np.zeros(2))[0][0] for _ in range(3)] == [1.0, 2.0, 3.0]


def test_latency_delays_by_k_and_clamps_to_first_obs():
    env = ObservationLatency(CounterEnv(), delay_steps=2)
    first, _ = env.reset()
    assert first[0] == 0.0
    seen = [env.step(np.zeros(2))[0][0] for _ in range(4)]
    assert seen == [0.0, 0.0, 1.0, 2.0]


def test_latency_reset_clears_buffer():
    env = ObservationLatency(CounterEnv(), delay_steps=2)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(2))
    env.reset()
    assert [env.step(np.zeros(2))[0][0] for _ in range(3)] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_latency_rejects_invalid_delay(bad):
    with pytest.raises(ValueError):
        ObservationLatency(CounterEnv(), delay_steps=bad)


def test_deadband_zeroes_only_small_components():
    env = ActuatorDeadband(CounterEnv(), threshold=0.05)
    env.reset()
    env.step(np.array([0.3, 0.04]))
    assert np.allclose(delivered(env), [0.3, 0.0])
    env.step(np.array([-0.049, -0.5]))
    assert np.allclose(delivered(env), [0.0, -0.5])


def test_deadband_boundary_value_passes_through():
    env = ActuatorDeadband(CounterEnv(), threshold=0.05)
    env.reset()
    env.step(np.array([0.05, -0.05]))
    assert np.array_equal(delivered(env), np.array([0.05, -0.05]))


def test_deadband_preserves_dtype_and_does_not_mutate_input():
    env = ActuatorDeadband(CounterEnv(), threshold=0.05)
    env.reset()
    action = np.array([0.01, 0.5], dtype=np.float32)
    original = action.copy()
    env.step(action)
    assert np.array_equal(action, original)
    assert delivered(env).dtype == np.float32


def test_degradation_scales_action():
    env = MotorDegradation(CounterEnv(), scale=0.6)
    env.reset()
    env.step(np.array([1.0, -0.5]))
    assert np.allclose(delivered(env), [0.6, -0.3])


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_degradation_rejects_out_of_range_scale(bad):
    with pytest.raises(ValueError):
        MotorDegradation(CounterEnv(), scale=bad)


def test_deadband_applies_before_degradation(monkeypatch):
    monkeypatch.setattr("envs.stress_wrappers.make_base_env", lambda env_id, **kw: CounterEnv())
    env = make_stressed_env("ignored", deadband=0.05, degradation=0.5)
    env.reset()
    env.step(np.array([0.04, 0.08]))
    # 0.04 is zeroed by the deadband first; 0.08 survives and is then halved.
    assert np.allclose(delivered(env), [0.0, 0.04])


def test_make_base_env_unknown_id_raises():
    with pytest.raises(gym.error.Error):
        make_base_env("DefinitelyNotAnEnv-v0")


def test_nominal_stressed_env_matches_base_env():
    pytest.importorskip("mujoco")
    base = make_base_env("Reacher-v4")
    stressed = make_stressed_env("Reacher-v4")
    o_a, _ = base.reset(seed=3)
    o_b, _ = stressed.reset(seed=3)
    assert np.allclose(o_a, o_b)
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        o_a, r_a, *_ = base.step(a)
        o_b, r_b, *_ = stressed.step(a)
        assert np.allclose(o_a, o_b) and r_a == r_b
```

- [ ] **Step 2: Run to verify failure**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest tests/test_stress_wrappers.py -q
```
Expected: collection error `ImportError: cannot import name 'ActuatorDeadband' from 'envs'` (or `No module named 'envs'`).

- [ ] **Step 3: Implement** (`envs/stress_wrappers.py`)

```python
"""Wrappers that simulate physical failure modes for IPQC-style stress testing."""
from __future__ import annotations

import warnings
from collections import deque

import gymnasium as gym
import numpy as np


class ObservationLatency(gym.Wrapper):
    """Sensor latency: the agent sees the observation from `delay_steps` steps ago.

    At the start of an episode the buffer is filled with the first observation, so the
    agent sees the initial state until real delayed data becomes available.
    """

    def __init__(self, env: gym.Env, delay_steps: int = 0):
        super().__init__(env)
        if int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError(f"delay_steps must be a non-negative integer, got {delay_steps!r}")
        self.delay_steps = int(delay_steps)
        self._buffer: deque = deque(maxlen=self.delay_steps + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer.clear()
        self._buffer.extend([obs] * (self.delay_steps + 1))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs)
        return self._buffer[0], reward, terminated, truncated, info


class ActuatorDeadband(gym.ActionWrapper):
    """Static friction: action components with |a| < threshold produce no torque."""

    def __init__(self, env: gym.Env, threshold: float = 0.05):
        super().__init__(env)
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold!r}")
        self.threshold = float(threshold)

    def action(self, action):
        action = np.asarray(action)
        return np.where(np.abs(action) < self.threshold, 0, action).astype(action.dtype, copy=False)


class MotorDegradation(gym.ActionWrapper):
    """Motor wear: delivered torque is `scale` times the commanded torque (1.0 = healthy)."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"scale must be within [0, 1], got {scale!r}")
        self.scale = float(scale)

    def action(self, action):
        action = np.asarray(action)
        return (action * self.scale).astype(action.dtype, copy=False)


def make_base_env(env_id: str, **kwargs) -> gym.Env:
    """gym.make with a -v4 -> -v5 fallback for Gymnasium versions that dropped v4."""
    try:
        return gym.make(env_id, **kwargs)
    except gym.error.Error as exc:
        if not env_id.endswith("-v4"):
            raise
        fallback = env_id[:-3] + "-v5"
        warnings.warn(f"{env_id} is unavailable ({exc}); falling back to {fallback}", stacklevel=2)
        return gym.make(fallback, **kwargs)


def make_stressed_env(
    env_id: str,
    delay_steps: int = 0,
    deadband: float = 0.0,
    degradation: float = 1.0,
    **kwargs,
) -> gym.Env:
    """Build `env_id` with the requested stresses. Defaults are the nominal (unstressed) env.

    The commanded action meets the deadband first (static friction on the command),
    then degradation scales the torque actually delivered.
    """
    env = make_base_env(env_id, **kwargs)
    env = ObservationLatency(env, delay_steps)
    env = MotorDegradation(env, degradation)
    env = ActuatorDeadband(env, deadband)
    return env
```

`envs/__init__.py`:
```python
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
    "make_base_env",
    "make_stressed_env",
]
```

- [ ] **Step 4: Run to verify pass**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest tests/test_stress_wrappers.py -q
```
Expected: all pass (the Reacher test is skipped only if `mujoco` is missing, which it should not be).

- [ ] **Step 5: Commit**

```bash
git add deep-feedback-control/envs deep-feedback-control/tests/test_stress_wrappers.py
git commit -m "feat: add latency, deadband and degradation stress wrappers" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Config and training script

**Files:**
- Create: `deep-feedback-control/configs/td3_config.yaml`, `deep-feedback-control/scripts/__init__.py` (empty), `deep-feedback-control/scripts/train.py`
- Test: `deep-feedback-control/tests/test_train_config.py`, `deep-feedback-control/tests/test_dfc_smoke.py`

**Interfaces:**
- Consumes: `envs.make_base_env`.
- Produces (all in `scripts.train`):
  - `PROJECT_ROOT: Path`, `DEFAULT_CONFIG: Path`
  - `load_config(path=None) -> dict` (raises `ValueError` naming missing keys)
  - `resolve_path(p) -> Path` (relative paths resolve against `PROJECT_ROOT`)
  - `resolve_device(requested: str) -> str` (raises `RuntimeError` when CUDA is requested but unavailable)
  - `model_file(model_path) -> Path` (the `.zip` SB3 writes)
  - `WandbEpisodeLogger(wandb_module, every=1000)` (SB3 callback)
  - `TrainResult` dataclass: `model: TD3`, `model_path: Path`, `monitor_path: Path`
  - `train(cfg, timesteps=None, learning_starts=None, model_path=None, device=None, use_wandb=None, verbose=1) -> TrainResult`
  - `main(argv=None)` CLI: `--config --timesteps --learning-starts --device --model-path --wandb`

- [ ] **Step 1: Write the config** (`configs/td3_config.yaml`)

```yaml
# TD3 on Reacher: training hyperparameters and IPQC sweep settings.
env_id: Reacher-v4          # falls back to Reacher-v5 if the installed Gymnasium dropped v4
seed: 0
total_timesteps: 150000
learning_rate: 1.0e-3       # note: PyYAML needs the "1.0e-3" form to parse a float
batch_size: 256
buffer_size: 200000
learning_starts: 5000
device: cuda
action_noise_sigma: 0.1
model_path: models/reacher_td3
wandb: false                # optional; needs `wandb login` done by the user
wandb_project: deep-feedback-control

ipqc:
  # Each list is ordered from mildest to most severe.
  degradation: [1.0, 0.8, 0.6, 0.4, 0.2]
  latency_steps: [0, 1, 2, 3, 4, 5]
  deadband: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
  episodes_per_seed: 20
  seeds: [0, 1, 2]
  retention_threshold: 0.9
```

- [ ] **Step 2: Write the failing config tests** (`tests/test_train_config.py`)

```python
from types import SimpleNamespace

import pytest
import torch

from scripts.train import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    WandbEpisodeLogger,
    load_config,
    model_file,
    resolve_device,
    resolve_path,
)


def test_default_config_has_expected_hyperparameters():
    cfg = load_config()
    assert isinstance(cfg["learning_rate"], float) and cfg["learning_rate"] == 1e-3
    assert cfg["batch_size"] == 256
    assert cfg["device"] == "cuda"
    assert cfg["total_timesteps"] == 150_000
    assert cfg["action_noise_sigma"] == 0.1
    assert cfg["ipqc"]["degradation"][:4] == [1.0, 0.8, 0.6, 0.4]
    assert 0.05 in cfg["ipqc"]["deadband"]


def test_load_config_names_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("env_id: Reacher-v4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size"):
        load_config(bad)


def test_resolve_device_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cpu"):
        resolve_device("cuda")
    assert resolve_device("cpu") == "cpu"


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config()["env_id"].startswith("Reacher")
    assert resolve_path("models/x") == PROJECT_ROOT / "models" / "x"
    assert DEFAULT_CONFIG.is_absolute()


def test_model_file_adds_zip_suffix(tmp_path):
    assert model_file(tmp_path / "m").name == "m.zip"
    assert model_file(tmp_path / "m.zip").name == "m.zip"


def test_wandb_logger_logs_mean_episode_return():
    logged = []
    stub = SimpleNamespace(log=lambda data, step: logged.append((data, step)))
    cb = WandbEpisodeLogger(stub, every=1)
    cb.model = SimpleNamespace(ep_info_buffer=[{"r": -5.0}, {"r": -3.0}])
    cb.n_calls, cb.num_timesteps = 1, 100
    assert cb._on_step() is True
    assert logged == [({"train/ep_rew_mean": -4.0}, 100)]
```

- [ ] **Step 3: Write the smoke tests** (`tests/test_dfc_smoke.py`)

```python
import pytest
import torch

from scripts.train import load_config, model_file, train


def test_train_cpu_smoke(tmp_path):
    pytest.importorskip("mujoco")
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m", device="cpu", verbose=0)
    assert model_file(res.model_path).exists()
    assert res.monitor_path.exists()
    assert res.model.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_cuda_smoke(tmp_path):
    pytest.importorskip("mujoco")
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m", verbose=0)
    assert res.model.device.type == "cuda"
```

- [ ] **Step 4: Run to verify failure**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest tests/test_train_config.py tests/test_dfc_smoke.py -q
```
Expected: `ModuleNotFoundError: No module named 'scripts'` (or import error for `scripts.train`).

- [ ] **Step 5: Implement** (`scripts/__init__.py` empty; `scripts/train.py`)

```python
"""Train a TD3 policy on a MuJoCo manipulator (headless)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402

from envs import make_base_env  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "td3_config.yaml"
REQUIRED_KEYS = (
    "env_id", "seed", "total_timesteps", "learning_rate", "batch_size", "buffer_size",
    "learning_starts", "device", "action_noise_sigma", "model_path", "ipqc",
)
REQUIRED_IPQC_KEYS = (
    "degradation", "latency_steps", "deadband", "episodes_per_seed", "seeds", "retention_threshold",
)


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    missing_ipqc = [k for k in REQUIRED_IPQC_KEYS if k not in cfg["ipqc"]]
    if missing_ipqc:
        raise ValueError(f"{path} is missing required ipqc keys: {missing_ipqc}")
    return cfg


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Re-run with --device cpu, or fix the PyTorch CUDA install."
        )
    return requested


def model_file(model_path) -> Path:
    """The file SB3 actually writes: `<model_path>.zip`."""
    p = Path(model_path)
    return p if p.suffix == ".zip" else p.with_name(p.name + ".zip")


class WandbEpisodeLogger(BaseCallback):
    """Log the rolling mean training return to Weights & Biases every `every` steps."""

    def __init__(self, wandb_module, every: int = 1000):
        super().__init__()
        self.wandb = wandb_module
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0 and len(self.model.ep_info_buffer) > 0:
            mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.wandb.log({"train/ep_rew_mean": mean}, step=self.num_timesteps)
        return True


@dataclass
class TrainResult:
    model: TD3
    model_path: Path
    monitor_path: Path


def train(
    cfg: dict,
    timesteps: int | None = None,
    learning_starts: int | None = None,
    model_path=None,
    device: str | None = None,
    use_wandb: bool | None = None,
    verbose: int = 1,
) -> TrainResult:
    device = resolve_device(device or cfg["device"])
    timesteps = int(timesteps or cfg["total_timesteps"])
    model_path = resolve_path(model_path or cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = model_path.with_name(model_path.name + ".monitor.csv")

    env = Monitor(make_base_env(cfg["env_id"]), filename=str(monitor_path))
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=cfg["action_noise_sigma"] * np.ones(n_actions)
    )
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=noise,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        learning_starts=learning_starts if learning_starts is not None else cfg["learning_starts"],
        device=device,
        seed=cfg["seed"],
        verbose=verbose,
    )

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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train TD3 on Reacher.")
    p.add_argument("--config", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    res = train(
        cfg,
        timesteps=args.timesteps,
        learning_starts=args.learning_starts,
        model_path=args.model_path,
        device=args.device,
        use_wandb=args.wandb,
    )
    print(f"device={res.model.device}  saved={model_file(res.model_path)}")
    if res.model.device.type == "cuda":
        print(f"cuda memory allocated: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run to verify pass**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest tests/test_train_config.py tests/test_dfc_smoke.py -q
```
Expected: all pass, including the CUDA smoke test.

- [ ] **Step 7: The playbook's dry run: 1,000 steps on the GPU**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" scripts/train.py --timesteps 1000 --learning-starts 100 --model-path models/dryrun
```
Expected final lines: `device=cuda:0  saved=...models\dryrun.zip` and a `cuda memory allocated: ...` line. (`models/dryrun*` is git-ignored.)

- [ ] **Step 8: Commit**

```bash
git add deep-feedback-control/configs deep-feedback-control/scripts deep-feedback-control/tests
git commit -m "feat: add TD3 config and training script with optional W&B" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: IPQC evaluation

**Files:**
- Create: `deep-feedback-control/scripts/evaluate_ipqc.py`
- Test: `deep-feedback-control/tests/test_ipqc_eval.py`

**Interfaces:**
- Consumes: `envs.make_base_env`, `envs.make_stressed_env`; from `scripts.train`: `load_config`, `model_file`, `resolve_path`, `PROJECT_ROOT`.
- Produces (all in `scripts.evaluate_ipqc`):
  - `STRESS_FILE: dict[str, str]` mapping `"degradation" | "latency" | "deadband"` to the PNG file name
  - `RandomPolicy(action_space, seed=0)` with `.predict(obs, deterministic=True) -> (action, None)`
  - `run_episode(env, policy, seed=None) -> float`
  - `load_policy(model_path, device="cpu")` (raises `FileNotFoundError` mentioning training if missing)
  - `run_ipqc_sweep(model, cfg, episodes_per_seed=None, seeds=None) -> DataFrame` with columns `stress_type, level, order, seed, episode, episode_return`. `stress_type` also includes the baselines `"random"` and `"nominal"`.
  - `summarize(df) -> DataFrame` with columns `stress_type, level, order, mean_return, std_return, n_episodes, score`
  - `safety_boundary(summary, stress_type, threshold=0.9) -> {"level": float | None, "reached": bool, "first_failure": float | None}`
  - `describe_boundary(stress_type, boundary, threshold) -> str`
  - `plot_stress_curve(summary, stress_type, boundary, threshold) -> Figure`
  - `save_report(summary, out_dir, threshold) -> dict[str, dict]` (writes the three PNGs and `ipqc_results.csv`)
  - `main(argv=None)` CLI: `--config --model-path --episodes --seeds --out`

- [ ] **Step 1: Write the failing tests** (`tests/test_ipqc_eval.py`)

```python
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_ipqc import (
    STRESS_FILE,
    describe_boundary,
    load_policy,
    run_ipqc_sweep,
    safety_boundary,
    save_report,
    summarize,
)
from scripts.train import load_config


def episode_df(means):
    """Build a raw-episode frame from {(stress_type, level, order): mean_return} (2 episodes each)."""
    rows = []
    for (stress_type, level, order), mean in means.items():
        for ep, delta in enumerate((-1.0, 1.0)):
            rows.append(dict(stress_type=stress_type, level=level, order=order, seed=0,
                             episode=ep, episode_return=mean + delta))
    return pd.DataFrame(rows)


BASELINES = {("random", 0.0, 0): -40.0, ("nominal", 0.0, 0): -5.0}


def test_summarize_normalizes_between_random_and_nominal():
    df = episode_df({**BASELINES, ("degradation", 1.0, 0): -5.0, ("degradation", 0.6, 1): -22.5})
    s = summarize(df).set_index(["stress_type", "level"])
    assert s.loc[("degradation", 1.0), "score"] == pytest.approx(1.0)
    assert s.loc[("degradation", 0.6), "score"] == pytest.approx(0.5)
    assert s.loc[("random", 0.0), "score"] == pytest.approx(0.0)


def test_summarize_raises_when_nominal_not_better_than_random():
    df = episode_df({("random", 0.0, 0): -5.0, ("nominal", 0.0, 0): -20.0})
    with pytest.raises(ValueError, match="nominal"):
        summarize(df)


def boundary_summary(scores):
    return pd.DataFrame(
        [dict(stress_type="latency", level=float(i), order=i, score=s) for i, s in enumerate(scores)]
    )


def test_safety_boundary_stops_at_first_failure_even_if_later_levels_recover():
    b = safety_boundary(boundary_summary([1.0, 0.95, 0.6, 0.95]), "latency", 0.9)
    assert b == {"level": 1.0, "reached": True, "first_failure": 2.0}


def test_safety_boundary_all_levels_pass():
    b = safety_boundary(boundary_summary([1.0, 0.99, 0.95]), "latency", 0.9)
    assert b == {"level": 2.0, "reached": False, "first_failure": None}


def test_safety_boundary_mildest_level_fails():
    b = safety_boundary(boundary_summary([0.5, 0.4]), "latency", 0.9)
    assert b == {"level": None, "reached": True, "first_failure": 0.0}


def test_describe_boundary_mentions_level():
    text = describe_boundary("latency", {"level": 2.0, "reached": True, "first_failure": 3.0}, 0.9)
    assert "latency" in text and "2.0" in text and "3.0" in text


def test_load_policy_missing_model_points_to_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_policy(tmp_path / "nope")


def test_save_report_writes_plots_and_csv(tmp_path):
    means = dict(BASELINES)
    for i, lvl in enumerate([1.0, 0.8, 0.6]):
        means[("degradation", lvl, i)] = -5.0 - 5 * i
    for i, lvl in enumerate([0.0, 1.0, 2.0]):
        means[("latency", lvl, i)] = -5.0 - 2 * i
    for i, lvl in enumerate([0.0, 0.05, 0.1]):
        means[("deadband", lvl, i)] = -5.0 - i
    summary = summarize(episode_df(means))
    boundaries = save_report(summary, tmp_path, 0.9)
    assert set(boundaries) == set(STRESS_FILE)
    for name in STRESS_FILE.values():
        assert (tmp_path / name).stat().st_size > 0
    assert (tmp_path / "ipqc_results.csv").exists()


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_sweep_covers_every_condition():
    pytest.importorskip("mujoco")
    cfg = load_config()
    ip = cfg["ipqc"]
    df = run_ipqc_sweep(ZeroModel(), cfg, episodes_per_seed=1, seeds=[0])
    expected = 2 + len(ip["degradation"]) + len(ip["latency_steps"]) + len(ip["deadband"])
    assert len(df) == expected
    assert set(df.stress_type) == {"random", "nominal", "degradation", "latency", "deadband"}
    assert list(df.columns) == ["stress_type", "level", "order", "seed", "episode", "episode_return"]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest tests/test_ipqc_eval.py -q
```
Expected: `ModuleNotFoundError: No module named 'scripts.evaluate_ipqc'`.

- [ ] **Step 3: Implement** (`scripts/evaluate_ipqc.py`)

```python
"""IPQC stress test: sweep latency / deadband / motor degradation and find safety boundaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from envs import make_stressed_env  # noqa: E402
from scripts.train import load_config, model_file, resolve_path  # noqa: E402

STRESS_KWARG = {"degradation": "degradation", "latency": "delay_steps", "deadband": "deadband"}
CONFIG_KEY = {"degradation": "degradation", "latency": "latency_steps", "deadband": "deadband"}
STRESS_FILE = {
    "degradation": "degradation_curve.png",
    "latency": "latency_curve.png",
    "deadband": "deadband_curve.png",
}
STRESS_XLABEL = {
    "degradation": "motor torque scale (1.0 = healthy)",
    "latency": "observation delay (steps)",
    "deadband": "actuator deadband threshold",
}


class RandomPolicy:
    """Uniform-random actions: the 'no skill' reference for the retention score."""

    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.action_space.sample(), None


def run_episode(env, policy, seed=None) -> float:
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    return total


def load_policy(model_path, device: str = "cpu"):
    from stable_baselines3 import TD3

    f = model_file(resolve_path(model_path))
    if not f.exists():
        raise FileNotFoundError(
            f"No trained model at {f}. Train one first: `python scripts/train.py` "
            "(or run notebook 01_train_td3.ipynb)."
        )
    return TD3.load(str(f), device=device)


def _conditions(ipqc: dict):
    conds = [("random", 0.0, 0, {}), ("nominal", 0.0, 0, {})]
    for stress_type in STRESS_KWARG:
        for order, level in enumerate(ipqc[CONFIG_KEY[stress_type]]):
            conds.append((stress_type, level, order, {STRESS_KWARG[stress_type]: level}))
    return conds


def run_ipqc_sweep(model, cfg: dict, episodes_per_seed=None, seeds=None) -> pd.DataFrame:
    ipqc = cfg["ipqc"]
    n_eps = episodes_per_seed or ipqc["episodes_per_seed"]
    seeds = seeds if seeds is not None else ipqc["seeds"]
    rows = []
    for stress_type, level, order, kwargs in _conditions(ipqc):
        for seed in seeds:
            env = make_stressed_env(cfg["env_id"], **kwargs)
            policy = RandomPolicy(env.action_space, seed) if stress_type == "random" else model
            for ep in range(n_eps):
                ret = run_episode(env, policy, seed=seed if ep == 0 else None)
                rows.append(
                    dict(stress_type=stress_type, level=float(level), order=order,
                         seed=seed, episode=ep, episode_return=ret)
                )
            env.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["stress_type", "level", "order"], as_index=False)["episode_return"].agg(
        mean_return="mean", std_return="std", n_episodes="count"
    )
    r_rand = g.loc[g.stress_type == "random", "mean_return"].iloc[0]
    r_nom = g.loc[g.stress_type == "nominal", "mean_return"].iloc[0]
    if r_nom <= r_rand:
        raise ValueError(
            f"Nominal return ({r_nom:.2f}) is not better than the random policy ({r_rand:.2f}); "
            "the model is not trained well enough to stress-test. Train longer."
        )
    g["score"] = (g["mean_return"] - r_rand) / (r_nom - r_rand)
    return g.sort_values(["stress_type", "order"]).reset_index(drop=True)


def safety_boundary(summary: pd.DataFrame, stress_type: str, threshold: float = 0.9) -> dict:
    """Most severe level reached before the score first drops below `threshold`."""
    rows = summary[summary.stress_type == stress_type].sort_values("order")
    last_ok = None
    for _, r in rows.iterrows():
        if r["score"] >= threshold:
            last_ok = float(r["level"])
        else:
            return {"level": last_ok, "reached": True, "first_failure": float(r["level"])}
    return {"level": last_ok, "reached": False, "first_failure": None}


def describe_boundary(stress_type: str, boundary: dict, threshold: float) -> str:
    pct = f"{threshold:.0%}"
    if boundary["level"] is None:
        return f"{stress_type}: policy already falls below {pct} retention at the mildest level tested."
    if not boundary["reached"]:
        return (f"{stress_type}: policy keeps >= {pct} retention at every level tested "
                f"(up to {boundary['level']}); the boundary was not reached.")
    return (f"{stress_type}: safe up to {boundary['level']}; "
            f"retention drops below {pct} at {boundary['first_failure']}.")


def plot_stress_curve(summary: pd.DataFrame, stress_type: str, boundary: dict, threshold: float):
    rows = summary[summary.stress_type == stress_type].sort_values("order")
    r_rand = summary.loc[summary.stress_type == "random", "mean_return"].iloc[0]
    r_nom = summary.loc[summary.stress_type == "nominal", "mean_return"].iloc[0]
    floor = r_rand + threshold * (r_nom - r_rand)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(rows["level"], rows["mean_return"], yerr=rows["std_return"],
                marker="o", capsize=3, label="TD3 policy (mean ± std)")
    ax.axhline(r_nom, ls="--", color="gray", label="nominal")
    ax.axhline(floor, ls="--", color="orange", label=f"{threshold:.0%} retention floor")
    ax.axhline(r_rand, ls=":", color="red", label="random policy")
    if boundary["level"] is not None:
        ax.axvline(boundary["level"], color="green", alpha=0.6,
                   label=f"safety boundary = {boundary['level']}")
    if len(rows) > 1 and rows["level"].iloc[0] > rows["level"].iloc[-1]:
        ax.invert_xaxis()  # degradation: severity grows to the right
    ax.set_xlabel(STRESS_XLABEL[stress_type])
    ax.set_ylabel("episode return (higher is better)")
    ax.set_title(f"Reacher TD3: {stress_type} stress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_report(summary: pd.DataFrame, out_dir, threshold: float) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = {}
    for stress_type, filename in STRESS_FILE.items():
        boundary = safety_boundary(summary, stress_type, threshold)
        boundaries[stress_type] = boundary
        fig = plot_stress_curve(summary, stress_type, boundary, threshold)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    summary.to_csv(out_dir / "ipqc_results.csv", index=False)
    return boundaries


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="IPQC stress test for a trained Reacher TD3 policy.")
    p.add_argument("--config", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--episodes", type=int, default=None, help="episodes per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", default="plots")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = load_policy(args.model_path or cfg["model_path"])
    raw = run_ipqc_sweep(model, cfg, episodes_per_seed=args.episodes, seeds=args.seeds)
    summary = summarize(raw)
    threshold = cfg["ipqc"]["retention_threshold"]
    out_dir = resolve_path(args.out)
    boundaries = save_report(summary, out_dir, threshold)
    print(summary.to_string(index=False))
    for stress_type, b in boundaries.items():
        print(describe_boundary(stress_type, b, threshold))
    print(f"Wrote plots and ipqc_results.csv to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest -q
```
Expected: the whole suite passes (wrappers, config, smoke, eval).

- [ ] **Step 5: End-to-end CLI check on the dry-run model**

The 1,000-step dry-run model is far from trained, so scoring may legitimately raise the "not better than random" error. Either outcome is informative:

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" scripts/evaluate_ipqc.py --model-path models/dryrun --episodes 2 --seeds 0 --out plots/quick
```
Expected: either a printed summary plus `Wrote plots ...` (and three PNGs in `plots/quick/`), or `ValueError: Nominal return (...) is not better than the random policy (...)`. Both prove the pipeline runs; a Python traceback of any other kind is a bug to fix.

- [ ] **Step 6: Commit**

```bash
git add deep-feedback-control/scripts/evaluate_ipqc.py deep-feedback-control/tests/test_ipqc_eval.py
git commit -m "feat: add IPQC sweep, retention scoring, safety boundary and plots" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Notebooks

**Files:**
- Create: `deep-feedback-control/notebooks/01_train_td3.ipynb`, `deep-feedback-control/notebooks/02_ipqc_stress_test.ipynb`
- Generator (throwaway, not committed): `C:\Users\User\AppData\Local\Temp\claude\C--Users-User-OneDrive-Documents-Coding-Projects\dca33f4e-a625-4047-b9d5-f0ead89624b3\scratchpad\build_notebooks.py`

**Interfaces:**
- Consumes: `scripts.train.{load_config, train, model_file, resolve_device}`, `scripts.evaluate_ipqc.{run_episode, RandomPolicy, load_policy, run_ipqc_sweep, summarize, save_report, describe_boundary, STRESS_FILE}`, `envs.make_base_env`.
- Behaviour contract: both notebooks find the project root by walking up from the working directory until `envs/stress_wrappers.py` exists; `DFC_QUICK=1` switches to tiny budgets and `quick`-suffixed outputs.

- [ ] **Step 1: Write the generator script**

```python
import nbformat as nbf
from pathlib import Path

NB_DIR = Path(r"C:\Users\User\OneDrive\Documents\Coding Projects\ai-engineering-portfolio\deep-feedback-control\notebooks")
NB_DIR.mkdir(exist_ok=True)

SETUP = '''import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "envs" / "stress_wrappers.py").exists())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

QUICK = os.environ.get("DFC_QUICK", "0") == "1"  # DFC_QUICK=1 -> tiny budgets, for smoke-testing only
PLOTS = ROOT / "plots" / ("quick" if QUICK else "")
PLOTS.mkdir(parents=True, exist_ok=True)

print("project root :", ROOT)
print("CUDA         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (cpu only)")
print("quick mode   :", QUICK)'''


def nb(cells, path):
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (ai_portfolio)", "language": "python", "name": "ai_portfolio",
    }
    notebook.cells = cells
    nbf.write(notebook, path)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb([
    md("# 01 · Train a TD3 policy on MuJoCo Reacher\n\n"
       "A 2-joint arm must touch a random target. We train **TD3** (Stable-Baselines3 + PyTorch) "
       "headless on the GPU, then compare the learned policy to a random one.\n\n"
       "All logic lives in `scripts/train.py`; this notebook just drives it and plots the results. "
       "Set the environment variable `DFC_QUICK=1` before launching Jupyter for a tiny smoke-test run."),
    code(SETUP),
    md("## Configuration\nHyperparameters come from `configs/td3_config.yaml`."),
    code('''from scripts.train import load_config, train, model_file
from scripts.evaluate_ipqc import run_episode, RandomPolicy
from envs import make_base_env

cfg = load_config()
if QUICK:
    TIMESTEPS, LEARNING_STARTS, MODEL_PATH = 3000, 500, "models/reacher_td3_quick"
else:
    TIMESTEPS, LEARNING_STARTS, MODEL_PATH = None, None, None  # use the values in the config file

pd.Series({k: v for k, v in cfg.items() if k != "ipqc"}, name="td3_config").to_frame()'''),
    md("## Train\n150k steps takes on the order of 10-20 minutes on the RTX 3070 Ti. Training is silent; "
       "the curve below is read back from the Monitor log."),
    code('''result = train(cfg, timesteps=TIMESTEPS, learning_starts=LEARNING_STARTS, model_path=MODEL_PATH, verbose=0)
print("device:", result.model.device, "| saved:", model_file(result.model_path))'''),
    md("## Training curve"),
    code('''log = pd.read_csv(result.monitor_path, skiprows=1)  # columns: r = return, l = length, t = wall time
log["timesteps"] = log["l"].cumsum()
log["rolling_return"] = log["r"].rolling(50, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(log["timesteps"], log["r"], alpha=0.25, label="episode return")
ax.plot(log["timesteps"], log["rolling_return"], label="rolling mean (50 episodes)")
ax.set_xlabel("environment steps")
ax.set_ylabel("episode return (higher is better)")
ax.set_title("TD3 on Reacher: training curve")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "training_curve.png", dpi=150)
plt.show()'''),
    md("## Trained vs. random policy\nTwenty evaluation episodes each, deterministic policy, no stress."),
    code('''env = make_base_env(cfg["env_id"])
trained = [run_episode(env, result.model, seed=0 if i == 0 else None) for i in range(20)]
rand_policy = RandomPolicy(env.action_space, seed=0)
random_ = [run_episode(env, rand_policy, seed=0 if i == 0 else None) for i in range(20)]

print(f"trained policy : {np.mean(trained):7.2f} ± {np.std(trained):.2f}")
print(f"random policy  : {np.mean(random_):7.2f} ± {np.std(random_):.2f}")'''),
    md("**Next:** `02_ipqc_stress_test.ipynb` loads this model and measures how far each failure mode can go "
       "before performance collapses."),
], NB_DIR / "01_train_td3.ipynb")

nb([
    md("# 02 · IPQC stress test: where does the policy stop being safe?\n\n"
       "We inject three physical failure modes into the trained Reacher policy and measure the damage:\n\n"
       "| Stress | What it simulates | Wrapper |\n|---|---|---|\n"
       "| Observation latency | slow / delayed sensors | `ObservationLatency` |\n"
       "| Actuator deadband | static friction: tiny commands do nothing | `ActuatorDeadband` |\n"
       "| Motor degradation | worn motors deliver a fraction of the commanded torque | `MotorDegradation` |\n\n"
       "Returns are negative, so each level gets a **retention score** "
       "`(R - R_random) / (R_nominal - R_random)`: 1.0 = as good as the healthy robot, 0.0 = no better than random. "
       "The **safety boundary** is the most severe level reached before the score first drops below 0.9.\n\n"
       "Run `01_train_td3.ipynb` first (same `DFC_QUICK` setting)."),
    code(SETUP),
    code('''from IPython.display import Image, display

from scripts.train import load_config
from scripts.evaluate_ipqc import (
    STRESS_FILE, describe_boundary, load_policy, run_ipqc_sweep, save_report, summarize,
)

cfg = load_config()
model = load_policy("models/reacher_td3_quick" if QUICK else cfg["model_path"])  # inference runs on CPU
threshold = cfg["ipqc"]["retention_threshold"]
print("stress levels:", {k: v for k, v in cfg["ipqc"].items() if k in ("degradation", "latency_steps", "deadband")})'''),
    md("## Run the sweep"),
    code('''EPISODES, SEEDS = (3, [0]) if QUICK else (None, None)  # None -> values from the config file
raw = run_ipqc_sweep(model, cfg, episodes_per_seed=EPISODES, seeds=SEEDS)
summary = summarize(raw)
summary'''),
    md("## Safety boundaries and plots"),
    code('''boundaries = save_report(summary, PLOTS, threshold)
for stress_type, boundary in boundaries.items():
    print(describe_boundary(stress_type, boundary, threshold))

for filename in STRESS_FILE.values():
    display(Image(filename=PLOTS / filename))'''),
    md("## Reading the plots\n"
       "Each plot shows mean episode return (± std over episodes and seeds) as the stress grows. "
       "The orange line is the 90% retention floor, the red dotted line is a random policy, and the green "
       "vertical line marks the safety boundary printed above. A policy that tolerates a failure mode "
       "keeps its curve above the orange line for longer."),
], NB_DIR / "02_ipqc_stress_test.ipynb")
print("notebooks written")
```

- [ ] **Step 2: Generate the notebooks**

```bash
"C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" "C:/Users/User/AppData/Local/Temp/claude/C--Users-User-OneDrive-Documents-Coding-Projects/dca33f4e-a625-4047-b9d5-f0ead89624b3/scratchpad/build_notebooks.py"
```
Expected: `notebooks written`.

- [ ] **Step 3: Smoke-execute both notebooks in quick mode (copies go to scratch, originals stay clean)**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
OUT="C:/Users/User/AppData/Local/Temp/claude/C--Users-User-OneDrive-Documents-Coding-Projects/dca33f4e-a625-4047-b9d5-f0ead89624b3/scratchpad/nbout"
cd deep-feedback-control
for nb in 01_train_td3 02_ipqc_stress_test; do
  DFC_QUICK=1 "$PY" -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=ai_portfolio \
    --output-dir "$OUT" notebooks/$nb.ipynb || break
done
```
Expected: both convert without error. A 3,000-step model should still beat the random policy (a near-zero-action policy scores far better than random on Reacher), so notebook 02's `summarize` should succeed. If it raises the intended `ValueError: Nominal return ... not better than the random policy`, retrain the quick model with `python scripts/train.py --timesteps 25000 --model-path models/reacher_td3_quick` and re-run the loop.

- [ ] **Step 4: Check the quick outputs exist**

```bash
ls deep-feedback-control/plots/quick/
```
Expected: `training_curve.png`, `degradation_curve.png`, `latency_curve.png`, `deadband_curve.png`, `ipqc_results.csv`.

- [ ] **Step 5: Commit the (output-free) notebooks**

```bash
git add deep-feedback-control/notebooks
git commit -m "feat: add training and IPQC stress-test notebooks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Full run, README, final verification

**Files:**
- Modify (executed in place): `deep-feedback-control/notebooks/01_train_td3.ipynb`, `deep-feedback-control/notebooks/02_ipqc_stress_test.ipynb`
- Create: `deep-feedback-control/README.md`, `README.md` (repo root)
- Commit: `deep-feedback-control/plots/{training_curve,degradation_curve,latency_curve,deadband_curve}.png`, `deep-feedback-control/plots/ipqc_results.csv`

- [ ] **Step 1: Full test suite**

```bash
cd deep-feedback-control && "C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe" -m pytest -q
```
Expected: all pass.

- [ ] **Step 2: Execute both notebooks for real (150k steps, full sweep), in place**

```bash
PY="C:/Users/User/.venvs/ai_portfolio/Scripts/python.exe"
cd deep-feedback-control
for nb in 01_train_td3 02_ipqc_stress_test; do
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=ai_portfolio \
    notebooks/$nb.ipynb || break
done
```
Run in the background; expect roughly 10-30 minutes in total. Expected: `models/reacher_td3.zip`, `models/reacher_td3.monitor.csv`, and the five files under `plots/` exist.

- [ ] **Step 3: Sanity-check the results**

Read `plots/ipqc_results.csv` and the notebook 01 printout:
- Trained policy mean return is clearly better than the random policy (expect roughly -4 to -7 vs about -40).
- The `degradation 1.0`, `latency 0` and `deadband 0.0` rows have `score == 1.0` (identical to nominal).
- Every plot opens and looks sensible.
If the trained policy is poor (worse than about -10), report that honestly and retrain with more timesteps in the config instead of publishing weak results.

- [ ] **Step 4: Write `deep-feedback-control/README.md`**

Sections, with real numbers copied from `plots/ipqc_results.csv` and the printed boundaries (state the Gymnasium env id actually used, v4 or v5):
1. What it is (two sentences) and the three stress modes.
2. Results: embed `plots/training_curve.png` and the three stress plots; one sentence per stress type stating its measured safety boundary.
3. How the safety boundary is defined (retention score formula).
4. Reproduce: venv + torch CUDA install line, `pip install -r requirements.txt`, `python scripts/train.py`, `python scripts/evaluate_ipqc.py`, or run the two notebooks (kernel `ai_portfolio`); `DFC_QUICK=1` for a smoke run; `pytest`.
5. Limitations: Reacher is a simple benchmark; stress curves illustrate the method rather than certify a real robot; evaluation uses a fixed seed set.

Root `README.md`: one paragraph on the portfolio, a table of the two projects with status (Project 1 done, Project 2 planned), and a link to `deep-feedback-control/README.md`.

- [ ] **Step 5: Final verification before claiming done**

```bash
git status --short
git diff --stat HEAD
```
Confirm: no model `.zip`, no `plots/quick/`, no `wandb/` or data files are staged; notebooks contain outputs.

- [ ] **Step 6: Commit**

```bash
git add README.md deep-feedback-control/README.md deep-feedback-control/notebooks deep-feedback-control/plots
git commit -m "docs: add Project 1 results, executed notebooks and READMEs" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ask the user before pushing**

Report what ran and what the numbers are, then ask whether to push to `jackalaka21/ai-engineering-portfolio` (use `GH_TOKEN="$(gh auth token --user jackalaka21)" git push`).
