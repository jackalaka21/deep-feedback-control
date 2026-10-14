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
