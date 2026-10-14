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
from stable_baselines3 import DDPG, TD3  # noqa: E402
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
VALID_ALGOS = ("td3", "ddpg")
ALGOS = {"td3": TD3, "ddpg": DDPG}
STRESS_LIST_KEYS = ("degradation", "latency_steps", "deadband", "seeds")


def load_config(path=None) -> dict:
    path = resolve_path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    missing_ipqc = [k for k in REQUIRED_IPQC_KEYS if k not in cfg["ipqc"]]
    if missing_ipqc:
        raise ValueError(f"{path} is missing required ipqc keys: {missing_ipqc}")
    cfg.setdefault("algo", "td3")
    cfg.setdefault("ddpg_model_path", "models/reacher_ddpg")
    cfg.setdefault("success_radius", 0.05)
    if cfg["algo"] not in VALID_ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {cfg['algo']!r}")
    if not isinstance(cfg["learning_rate"], float):
        raise ValueError(
            f"learning_rate must be a float, got {cfg['learning_rate']!r}; "
            "write 1.0e-3 (PyYAML reads 1e-3 as a string)"
        )
    for key in STRESS_LIST_KEYS:
        values = cfg["ipqc"][key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"ipqc.{key} must be a non-empty list, got {values!r}")
    return cfg


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests device '{requested}' but CUDA is not available. "
            "Re-run with --device cpu, set `device: cpu` in configs/td3_config.yaml, "
            "or fix the PyTorch CUDA install."
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


def default_model_path(cfg: dict, algo: str) -> str:
    return cfg["ddpg_model_path"] if algo == "ddpg" else cfg["model_path"]


def build_model(cfg, env, algo, device, learning_starts, seed=None, verbose=1, tensorboard_log=None):
    """One place that builds TD3 or DDPG with identical hyperparameters."""
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {algo!r}")
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=cfg["action_noise_sigma"] * np.ones(n_actions)
    )
    return ALGOS[algo](
        "MlpPolicy",
        env,
        action_noise=noise,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        learning_starts=learning_starts,
        device=device,
        seed=cfg["seed"] if seed is None else seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )


@dataclass
class TrainResult:
    model: object  # a stable-baselines3 TD3 or DDPG
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
    algo: str | None = None,
) -> TrainResult:
    algo = algo or cfg["algo"]
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}, got {algo!r}")
    device = resolve_device(device or cfg["device"])
    timesteps = int(timesteps or cfg["total_timesteps"])
    start = learning_starts if learning_starts is not None else cfg["learning_starts"]
    if start >= timesteps:
        raise ValueError(
            f"timesteps ({timesteps}) must exceed learning_starts ({start}), otherwise no gradient "
            "updates run and the saved model is untrained. Raise --timesteps or lower --learning-starts."
        )
    model_path = resolve_path(model_path or default_model_path(cfg, algo))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = model_path.with_name(model_path.name + ".monitor.csv")

    env = Monitor(make_base_env(cfg["env_id"]), filename=str(monitor_path))
    model = build_model(cfg, env, algo, device, start, verbose=verbose)

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TD3 or DDPG on Reacher.")
    p.add_argument("--config", default=None, help="config path (relative paths resolve against the project root)")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--algo", choices=VALID_ALGOS, default=None)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                   help="force W&B logging on or off (default: the config value)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    res = train(
        cfg,
        timesteps=args.timesteps,
        learning_starts=args.learning_starts,
        model_path=args.model_path,
        device=args.device,
        use_wandb=args.wandb,
        algo=args.algo,
    )
    print(f"device={res.model.device}  saved={model_file(res.model_path)}")
    if res.model.device.type == "cuda":
        print(f"cuda memory allocated: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
