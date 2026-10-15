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


def test_train_save_load_evaluate_roundtrip(tmp_path, monkeypatch):
    """The path a user actually takes: train, save, reload from disk, run the stress sweep."""
    pytest.importorskip("mujoco")
    import scripts.evaluate_ipqc as ev
    from scripts.evaluate_ipqc import load_policy, run_ipqc_sweep

    # A 300-step model is not expected to beat random; the fail-fast guard has its own test.
    monkeypatch.setattr(ev, "_check_baselines", lambda rows: None)
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m", device="cpu", verbose=0)
    reloaded = load_policy(res.model_path)
    df = run_ipqc_sweep(reloaded, cfg, episodes_per_seed=1, seeds=[0])
    ip = cfg["ipqc"]
    assert len(df) == 2 + len(ip["degradation"]) + len(ip["latency_steps"]) + len(ip["deadband"])
    assert df["episode_return"].notna().all()


def test_train_rejects_timesteps_that_never_start_learning(tmp_path):
    cfg = load_config()
    with pytest.raises(ValueError, match="learning_starts"):
        train(cfg, timesteps=300, learning_starts=300, model_path=tmp_path / "m", device="cpu", verbose=0)
    assert not (tmp_path / "m.zip").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_cuda_smoke(tmp_path):
    pytest.importorskip("mujoco")
    cfg = load_config()
    res = train(cfg, timesteps=300, learning_starts=100, model_path=tmp_path / "m", verbose=0)
    assert res.model.device.type == "cuda"


from stable_baselines3 import DDPG, TD3  # noqa: E402

from scripts.evaluate_ipqc import load_policy  # noqa: E402
from scripts.train import default_model_path  # noqa: E402


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
