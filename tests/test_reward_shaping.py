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
