"""Train a TD3 policy on a MuJoCo manipulator (headless)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402

from envs import make_base_env  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "td3_config.yaml"
REQUIRED_KEYS = (
    "env_id", "seed", "total_timesteps", "learning_rate", "batch_size", "buffer_size",
    "learning_starts", "device", "action_noise_sigma", "model_path", "ipqc",
)
REQUIRED_IPQC_KEYS = (
    "degradation", "latency_steps", "deadband", "episodes_per_seed", "seeds", "retention_threshold",
)


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    missing_ipqc = [k for k in REQUIRED_IPQC_KEYS if k not in cfg["ipqc"]]
    if missing_ipqc:
        raise ValueError(f"{path} is missing required ipqc keys: {missing_ipqc}")
    return cfg


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Re-run with --device cpu, or fix the PyTorch CUDA install."
        )
    return requested


def model_file(model_path) -> Path:
    """The file SB3 actually writes: `<model_path>.zip`."""
    p = Path(model_path)
    return p if p.suffix == ".zip" else p.with_name(p.name + ".zip")


class WandbEpisodeLogger(BaseCallback):
    """Log the rolling mean training return to Weights & Biases every `every` steps."""

    def __init__(self, wandb_module, every: int = 1000):
        super().__init__()
        self.wandb = wandb_module
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0 and len(self.model.ep_info_buffer) > 0:
            mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.wandb.log({"train/ep_rew_mean": mean}, step=self.num_timesteps)
        return True


@dataclass
class TrainResult:
    model: TD3
    model_path: Path
    monitor_path: Path


def train(
    cfg: dict,
    timesteps: int | None = None,
    learning_starts: int | None = None,
    model_path=None,
    device: str | None = None,
    use_wandb: bool | None = None,
    verbose: int = 1,
) -> TrainResult:
    device = resolve_device(device or cfg["device"])
    timesteps = int(timesteps or cfg["total_timesteps"])
    start = learning_starts if learning_starts is not None else cfg["learning_starts"]
    if start >= timesteps:
        raise ValueError(
            f"timesteps ({timesteps}) must exceed learning_starts ({start}), otherwise no gradient "
            "updates run and the saved model is untrained. Raise --timesteps or lower --learning-starts."
        )
    model_path = resolve_path(model_path or cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = model_path.with_name(model_path.name + ".monitor.csv")

    env = Monitor(make_base_env(cfg["env_id"]), filename=str(monitor_path))
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=cfg["action_noise_sigma"] * np.ones(n_actions)
    )
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=noise,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        learning_starts=learning_starts if learning_starts is not None else cfg["learning_starts"],
        device=device,
        seed=cfg["seed"],
        verbose=verbose,
    )

    callback, run = None, None
    if use_wandb if use_wandb is not None else cfg.get("wandb", False):
        import wandb

        run = wandb.init(project=cfg.get("wandb_project", "deep-feedback-control"), config=cfg)
        callback = WandbEpisodeLogger(wandb)
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
    finally:
        if run is not None:
            run.finish()
        env.close()  # flushes the monitor CSV
    model.save(str(model_path))
    return TrainResult(model, model_path, monitor_path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train TD3 on Reacher.")
    p.add_argument("--config", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    res = train(
        cfg,
        timesteps=args.timesteps,
        learning_starts=args.learning_starts,
        model_path=args.model_path,
        device=args.device,
        use_wandb=args.wandb,
    )
    print(f"device={res.model.device}  saved={model_file(res.model_path)}")
    if res.model.device.type == "cuda":
        print(f"cuda memory allocated: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
