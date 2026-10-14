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
