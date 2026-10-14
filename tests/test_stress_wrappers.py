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
