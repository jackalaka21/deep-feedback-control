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


def test_train_save_load_evaluate_roundtrip(tmp_path):
    """The path a user actually takes: train, save, reload from disk, run the stress sweep."""
    pytest.importorskip("mujoco")
    from scripts.evaluate_ipqc import load_policy, run_ipqc_sweep

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
