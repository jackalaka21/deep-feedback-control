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


import yaml  # noqa: E402

from scripts.train import build_parser  # noqa: E402


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
