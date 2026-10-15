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
