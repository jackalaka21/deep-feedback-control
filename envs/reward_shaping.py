"""Training-reward variants for MuJoCo Reacher, used by the reward-shaping sweep.

Only the reward the agent trains on changes. Observations, termination and `info`
pass through, and evaluation always uses the unshaped environment.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

VARIANTS = ("default", "dist_only", "sparse", "progress", "scaled_x10")


class RewardShaping(gym.Wrapper):
    def __init__(self, env: gym.Env, variant: str = "default", success_radius: float = 0.05):
        super().__init__(env)
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.success_radius = float(success_radius)
        self._prev_distance = None

    def _fingertip_distance(self) -> float:
        """Distance from the live simulator state, so it is the post-step distance after env.step().

        (Reacher-v4's info["reward_dist"] is computed BEFORE the simulation step and v5's after, so it
        cannot be used to measure the effect of the action just taken.)
        """
        base = self.env.unwrapped
        return float(np.linalg.norm(base.get_body_com("fingertip") - base.get_body_com("target")))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._prev_distance = self._fingertip_distance() if self.variant == "progress" else None
        return result

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
            now = self._fingertip_distance()  # after this action
            shaped = 10.0 * (self._prev_distance - now) + ctrl_term
            self._prev_distance = now
        else:  # scaled_x10
            shaped = 10.0 * reward
        return obs, shaped, terminated, truncated, info
