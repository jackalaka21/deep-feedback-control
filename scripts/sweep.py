"""Reward-shaping sweep: DDPG vs TD3 across training-reward variants and seeds."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from envs import VARIANTS, RewardShaping, make_base_env  # noqa: E402
from scripts.evaluate_ipqc import RandomPolicy, run_episode_parts  # noqa: E402
from scripts.train import VALID_ALGOS, build_model, load_config, resolve_path  # noqa: E402

DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / "configs" / "sweep_config.yaml"
REQUIRED_SWEEP_KEYS = (
    "algos", "variants", "seeds", "timesteps", "learning_starts", "eval_every", "eval_episodes",
    "eval_seed", "random_baseline_episodes", "workers", "solve_threshold", "runs_dir", "robustness",
)


def load_sweep_config(path=None) -> dict:
    path = resolve_path(path) if path else DEFAULT_SWEEP_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_SWEEP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    for key in ("algos", "variants", "seeds"):
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f"{key} must be a non-empty list, got {cfg[key]!r}")
    for algo in cfg["algos"]:
        if algo not in VALID_ALGOS:
            raise ValueError(f"unknown algo {algo!r}; valid: {VALID_ALGOS}")
    for variant in cfg["variants"]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; valid: {VARIANTS}")
    if cfg["learning_starts"] >= cfg["timesteps"]:
        raise ValueError(
            f"learning_starts ({cfg['learning_starts']}) must be below timesteps ({cfg['timesteps']})"
        )
    if cfg["timesteps"] % cfg["eval_every"] != 0:
        raise ValueError("timesteps must be a multiple of eval_every so the last step is evaluated")
    return cfg


def run_name(spec: dict) -> str:
    return f"{spec['algo']}_{spec['variant']}_seed{spec['seed']}"


def run_dir(runs_dir, spec: dict) -> Path:
    return resolve_path(runs_dir) / run_name(spec)


def is_done(runs_dir, spec: dict) -> bool:
    return (run_dir(runs_dir, spec) / "done.json").exists()


def evaluate_policy(policy, env_id: str, episodes: int, eval_seed: int, success_radius: float) -> dict:
    """Deterministic evaluation on the UNSHAPED env with fixed targets (task metric = -sum distance)."""
    env = make_base_env(env_id)
    parts = [
        run_episode_parts(env, policy, seed=eval_seed if i == 0 else None) for i in range(episodes)
    ]
    env.close()
    dist = np.array([p["distance_return"] for p in parts])
    total = np.array([p["episode_return"] for p in parts])
    final = np.array([p["final_distance"] for p in parts])
    return {
        "mean_distance_return": float(dist.mean()),
        "std_distance_return": float(dist.std(ddof=1)) if len(dist) > 1 else 0.0,
        "mean_total_return": float(total.mean()),
        "success_rate": float((final <= success_radius).mean()),
    }


class TaskEvalCallback(BaseCallback):
    """Evaluate on the unshaped task at step 0 and then every `eval_every` steps."""

    def __init__(self, env_id, eval_every, episodes, eval_seed, success_radius):
        super().__init__()
        self.env_id, self.eval_every, self.episodes = env_id, eval_every, episodes
        self.eval_seed, self.success_radius = eval_seed, success_radius
        self.records: list[dict] = []

    def _evaluate(self) -> None:
        rec = evaluate_policy(self.model, self.env_id, self.episodes, self.eval_seed, self.success_radius)
        for key, value in rec.items():
            self.logger.record(f"eval/{key}", value)
        self.records.append({"timesteps": int(self.num_timesteps), **rec})

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._evaluate()
        return True


def run_metadata() -> dict:
    import gymnasium
    import mujoco
    import stable_baselines3

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "git_commit": commit,
    }


def make_spec(sweep_cfg: dict, train_cfg: dict, item: dict) -> dict:
    return {
        "algo": item["algo"], "variant": item["variant"], "seed": item["seed"],
        "timesteps": sweep_cfg["timesteps"], "learning_starts": sweep_cfg["learning_starts"],
        "eval_every": sweep_cfg["eval_every"], "eval_episodes": sweep_cfg["eval_episodes"],
        "eval_seed": sweep_cfg["eval_seed"], "success_radius": train_cfg["success_radius"],
        "runs_dir": sweep_cfg["runs_dir"], "train_cfg": train_cfg,
    }


PROTOCOL_KEYS = ("timesteps", "learning_starts", "eval_every", "eval_episodes", "eval_seed", "success_radius")
TRAIN_PROTOCOL_KEYS = ("env_id", "learning_rate", "batch_size", "buffer_size", "action_noise_sigma")


def protocol_of(spec: dict) -> dict:
    """Everything that must match for two runs to be comparable (recorded in done.json)."""
    out = {k: spec[k] for k in PROTOCOL_KEYS}
    out.update({k: spec["train_cfg"][k] for k in TRAIN_PROTOCOL_KEYS})
    return out


def check_resume_protocols(sweep_cfg: dict, train_cfg: dict) -> None:
    """Refuse to resume next to finished runs that were produced with a different protocol.

    Runs finished before protocols were recorded (no "protocol" in done.json) are accepted.
    """
    for item in build_grid(sweep_cfg):
        path = run_dir(sweep_cfg["runs_dir"], item) / "done.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding="utf-8")).get("protocol")
        if saved is None:
            continue
        current = protocol_of(make_spec(sweep_cfg, train_cfg, item))
        diffs = [f"{k}: {saved.get(k)} -> {current[k]}" for k in current if saved.get(k) != current[k]]
        if diffs:
            raise ValueError(
                f"{run_name(item)} was produced with a different protocol ({'; '.join(diffs)}). "
                f"Delete {path.parent} (or use another runs_dir) to re-run it; mixing protocols "
                "would silently corrupt the comparison."
            )


def run_one(spec: dict) -> dict:
    """Train one (algo, variant, seed) run on CPU and write eval.csv, final_model.zip, tb/, done.json."""
    torch.set_num_threads(1)
    rd = run_dir(spec["runs_dir"], spec)
    rd.mkdir(parents=True, exist_ok=True)
    for stale in ("eval.csv", "final_model.zip", "done.json"):  # restart cleanly after a crash
        (rd / stale).unlink(missing_ok=True)

    cfg = {**spec["train_cfg"], "seed": spec["seed"]}
    env = RewardShaping(make_base_env(cfg["env_id"]), spec["variant"], spec["success_radius"])
    model = build_model(cfg, env, spec["algo"], "cpu", spec["learning_starts"], seed=spec["seed"],
                        verbose=0, tensorboard_log=str(rd / "tb"))
    callback = TaskEvalCallback(cfg["env_id"], spec["eval_every"], spec["eval_episodes"],
                                spec["eval_seed"], spec["success_radius"])
    t0 = time.time()
    try:
        model.learn(total_timesteps=spec["timesteps"], callback=callback, tb_log_name="run")
    finally:
        env.close()
    seconds = time.time() - t0

    pd.DataFrame(callback.records).to_csv(rd / "eval.csv", index=False)
    model.save(str(rd / "final_model"))
    done = {
        "algo": spec["algo"], "variant": spec["variant"], "seed": spec["seed"],
        "timesteps": spec["timesteps"], "seconds": round(seconds, 1),
        "final_model": "final_model.zip", "protocol": protocol_of(spec), **run_metadata(),
    }
    (rd / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")  # written last
    return done


def build_grid(sweep_cfg: dict) -> list[dict]:
    return [
        {"algo": a, "variant": v, "seed": s}
        for a, v, s in itertools.product(sweep_cfg["algos"], sweep_cfg["variants"], sweep_cfg["seeds"])
    ]


def ensure_random_baseline(sweep_cfg: dict, train_cfg: dict) -> dict:
    """Random-policy task return over the eval protocol; cached in the runs dir, recomputed if the
    protocol (eval seed, episode count, env) changed since it was written."""
    runs_dir = resolve_path(sweep_cfg["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "random_baseline.json"
    n = sweep_cfg["random_baseline_episodes"]
    protocol = {"eval_seed": sweep_cfg["eval_seed"], "episodes": n, "env_id": train_cfg["env_id"]}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("protocol", protocol) == protocol:  # a legacy file without a protocol is reused
            return cached
    env = make_base_env(train_cfg["env_id"])
    policy = RandomPolicy(env.action_space, seed=sweep_cfg["eval_seed"])
    dist = [
        run_episode_parts(env, policy, seed=sweep_cfg["eval_seed"] if i == 0 else None)["distance_return"]
        for i in range(n)
    ]
    env.close()
    result = {
        "mean_distance_return": float(np.mean(dist)),
        "std_distance_return": float(np.std(dist, ddof=1)) if n > 1 else 0.0,
        "episodes": n,
        "protocol": protocol,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def quick_config(sweep_cfg: dict) -> dict:
    """A tiny grid and budget for smoke tests; writes under <runs_dir>/quick (git-ignored)."""
    q = dict(sweep_cfg)
    q.update(
        variants=sweep_cfg["variants"][:2], seeds=sweep_cfg["seeds"][:1], timesteps=800,
        learning_starts=200, eval_every=400, eval_episodes=2, random_baseline_episodes=10,
        runs_dir=str(Path(sweep_cfg["runs_dir"]) / "quick"), workers=min(2, sweep_cfg["workers"]),
    )
    return q


def run_sweep(sweep_cfg: dict, train_cfg: dict, workers: int | None = None, log=print) -> list[dict]:
    """Run every unfinished grid cell; finished cells (with done.json) are skipped."""
    runs_dir = sweep_cfg["runs_dir"]
    check_resume_protocols(sweep_cfg, train_cfg)
    grid = build_grid(sweep_cfg)
    pending = [g for g in grid if not is_done(runs_dir, g)]
    log(f"{len(grid)} runs in the grid, {len(grid) - len(pending)} already done, {len(pending)} to run")
    ensure_random_baseline(sweep_cfg, train_cfg)
    specs = [make_spec(sweep_cfg, train_cfg, g) for g in pending]
    workers = workers or sweep_cfg["workers"]

    results, failures = [], []

    def finished(i, spec, done=None, error=None):
        if error is not None:
            failures.append(f"{run_name(spec)}: {error}")
            log(f"[{i}/{len(specs)}] {run_name(spec)} FAILED: {error}")
        else:
            results.append(done)
            log(f"[{i}/{len(specs)}] {run_name(spec)} finished in {done['seconds'] / 60:.1f} min")

    if workers <= 1 or len(specs) <= 1:
        for i, spec in enumerate(specs, 1):
            try:
                finished(i, spec, done=run_one(spec))
            except Exception as exc:  # keep going; report at the end
                finished(i, spec, error=exc)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_one, s): s for s in specs}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    finished(i, futures[fut], done=fut.result())
                except Exception as exc:
                    finished(i, futures[fut], error=exc)
    if failures:
        raise RuntimeError(
            "some runs failed (finished runs were kept; re-launch to retry): " + "; ".join(failures)
        )
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run the DDPG/TD3 reward-shaping sweep (resumable).")
    p.add_argument("--config", default=None, help="sweep config (default configs/sweep_config.yaml)")
    p.add_argument("--train-config", default=None, help="training config (default configs/td3_config.yaml)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="tiny grid and budget, writes to runs/quick")
    args = p.parse_args(argv)
    sweep_cfg = load_sweep_config(args.config)
    if args.quick:
        sweep_cfg = quick_config(sweep_cfg)
    run_sweep(sweep_cfg, load_config(args.train_config), workers=args.workers)
    print("Done. Next: python scripts/sweep_analysis.py ; python scripts/shaping_robustness.py")


if __name__ == "__main__":
    main()
