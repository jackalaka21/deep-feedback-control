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

    class SpyCallback(TaskEvalCallback):
        """After each callback evaluation, re-evaluate the SAME model state on the unshaped env."""

        independent: list = []

        def _evaluate(self):
            super()._evaluate()
            self.independent.append(
                evaluate_policy(self.model, self.env_id, self.episodes, self.eval_seed, self.success_radius)
            )

    env = RewardShaping(make_base_env(cfg["env_id"]), "sparse", 0.05)  # agent sees rewards in {0, -1}
    model = build_model(cfg, env, "td3", "cpu", 100, verbose=0)
    cb = SpyCallback(cfg["env_id"], eval_every=200, episodes=2, eval_seed=1000, success_radius=0.05)
    cb.independent = []
    model.learn(total_timesteps=600, callback=cb)
    # Evaluations happen at step 0 and then every eval_every steps, including the last step.
    assert [r["timesteps"] for r in cb.records] == [0, 200, 400, 600]
    assert len(cb.independent) == 4
    for record, independent in zip(cb.records, cb.independent):
        measured = {k: v for k, v in record.items() if k != "timesteps"}
        assert measured == pytest.approx(independent)  # identical unshaped evaluation of the same model
        assert measured["mean_distance_return"] != int(measured["mean_distance_return"])  # not {0,-1}


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
