"""IPQC stress test: sweep latency / deadband / motor degradation and find safety boundaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from envs import make_stressed_env  # noqa: E402
from scripts.train import default_model_path, load_config, model_file, resolve_path  # noqa: E402

STRESS_KWARG = {"degradation": "degradation", "latency": "delay_steps", "deadband": "deadband"}
CONFIG_KEY = {"degradation": "degradation", "latency": "latency_steps", "deadband": "deadband"}
STRESS_FILE = {
    "degradation": "degradation_curve.png",
    "latency": "latency_curve.png",
    "deadband": "deadband_curve.png",
}
STRESS_XLABEL = {
    "degradation": "motor torque scale (1.0 = healthy)",
    "latency": "observation delay (steps)",
    "deadband": "actuator deadband threshold",
}


class RandomPolicy:
    """Uniform-random actions: the 'no skill' reference for the retention score."""

    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.action_space.sample(), None


def run_episode_parts(env, policy, seed=None) -> dict:
    """Run one episode and split the return into its task and control-cost parts.

    Reacher's reward is `reward_dist + reward_ctrl`. The control cost is charged on the action the
    motors actually receive, so stresses that shrink actions (deadband, degradation) also shrink
    the cost. `distance_return` (-sum of distance to target) is therefore the clean task measure.
    """
    obs, _ = env.reset(seed=seed)
    total = distance = control = 0.0
    final_distance = float("nan")
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        distance += float(info["reward_dist"])
        control += float(info["reward_ctrl"])
        final_distance = -float(info["reward_dist"])
        done = terminated or truncated
    return {"episode_return": total, "distance_return": distance,
            "control_return": control, "final_distance": final_distance}


def run_episode(env, policy, seed=None) -> float:
    return run_episode_parts(env, policy, seed)["episode_return"]


def load_policy(model_path, device: str = "cpu", algo: str = "td3"):
    from scripts.train import ALGOS

    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {tuple(ALGOS)}, got {algo!r}")
    f = model_file(resolve_path(model_path))
    if not f.exists():
        raise FileNotFoundError(
            f"No trained model at {f}. Train one first: `python scripts/train.py` "
            "(or run notebook 01_train_td3.ipynb)."
        )
    return ALGOS[algo].load(str(f), device=device)


def _conditions(ipqc: dict):
    conds = [("random", 0.0, 0, {}), ("nominal", 0.0, 0, {})]
    for stress_type in STRESS_KWARG:
        for order, level in enumerate(ipqc[CONFIG_KEY[stress_type]]):
            conds.append((stress_type, level, order, {STRESS_KWARG[stress_type]: level}))
    return conds


def _check_baselines(rows) -> None:
    df = pd.DataFrame(rows)
    r_rand = df.loc[df.stress_type == "random", "distance_return"].mean()
    r_nom = df.loc[df.stress_type == "nominal", "distance_return"].mean()
    if r_nom <= r_rand:
        raise ValueError(
            f"Nominal return ({r_nom:.2f}, task/distance) is not better than the random policy "
            f"({r_rand:.2f}); the model is not trained well enough to stress-test. Train longer."
        )


def run_ipqc_sweep(model, cfg: dict, episodes_per_seed=None, seeds=None) -> pd.DataFrame:
    ipqc = cfg["ipqc"]
    n_eps = episodes_per_seed or ipqc["episodes_per_seed"]
    seeds = seeds if seeds is not None else ipqc["seeds"]
    rows = []
    for stress_type, level, order, kwargs in _conditions(ipqc):
        for seed in seeds:
            env = make_stressed_env(cfg["env_id"], **kwargs)
            policy = RandomPolicy(env.action_space, seed) if stress_type == "random" else model
            for ep in range(n_eps):
                parts = run_episode_parts(env, policy, seed=seed if ep == 0 else None)
                rows.append(
                    dict(stress_type=stress_type, level=float(level), order=order,
                         seed=seed, episode=ep, **parts)
                )
            env.close()
        if stress_type == "nominal":
            _check_baselines(rows)  # fail fast, before the long stress sweep
    return pd.DataFrame(rows)


def _retention(g: pd.DataFrame, column: str, label: str) -> pd.Series:
    """(R - R_random) / (R_nominal - R_random) for the mean-return column `column`."""
    r_rand = g.loc[g.stress_type == "random", column].iloc[0]
    r_nom = g.loc[g.stress_type == "nominal", column].iloc[0]
    if r_nom <= r_rand:
        raise ValueError(
            f"Nominal return ({r_nom:.2f}, {label}) is not better than the random policy ({r_rand:.2f}); "
            "the model is not trained well enough to stress-test. Train longer."
        )
    return (g[column] - r_rand) / (r_nom - r_rand)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition means plus retention scores.

    `score` is retention on the task measure (distance only, no control cost) and drives the safety
    boundaries; it can go negative when the arm is worse than random at reaching the target.
    `total_score` is retention on the full reward, which also includes the control-cost term that
    stresses artificially shrink; it is reported for transparency but flatters deadband/degradation.
    """
    missing = [name for name in ("random", "nominal") if name not in set(df["stress_type"])]
    if missing:
        raise ValueError(f"summarize needs 'random' and 'nominal' baseline rows; missing: {missing}")
    g = df.groupby(["stress_type", "level", "order"], as_index=False).agg(
        mean_return=("episode_return", "mean"),
        std_return=("episode_return", "std"),
        mean_distance_return=("distance_return", "mean"),
        std_distance_return=("distance_return", "std"),
        mean_control_return=("control_return", "mean"),
        n_episodes=("episode_return", "count"),
    )
    g["score"] = _retention(g, "mean_distance_return", "task/distance")
    g["total_score"] = _retention(g, "mean_return", "total")
    return g.sort_values(["stress_type", "order"]).reset_index(drop=True)


def safety_boundary(summary: pd.DataFrame, stress_type: str, threshold: float = 0.9) -> dict:
    """Most severe level reached before the score first drops below `threshold`."""
    rows = summary[summary.stress_type == stress_type].sort_values("order")
    last_ok = None
    for _, r in rows.iterrows():
        if r["score"] >= threshold:
            last_ok = float(r["level"])
        else:
            return {"level": last_ok, "reached": True, "first_failure": float(r["level"])}
    return {"level": last_ok, "reached": False, "first_failure": None}


def describe_boundary(stress_type: str, boundary: dict, threshold: float) -> str:
    pct = f"{threshold:.0%}"
    if boundary["level"] is None:
        return f"{stress_type}: policy already falls below {pct} retention at the mildest level tested."
    if not boundary["reached"]:
        return (f"{stress_type}: policy keeps >= {pct} retention at every level tested "
                f"(up to {boundary['level']}); the boundary was not reached.")
    return (f"{stress_type}: safe up to {boundary['level']}; "
            f"retention drops below {pct} at {boundary['first_failure']}.")


def plot_stress_curve(summary: pd.DataFrame, stress_type: str, boundary: dict, threshold: float):
    rows = summary[summary.stress_type == stress_type].sort_values("order")
    r_rand = summary.loc[summary.stress_type == "random", "mean_distance_return"].iloc[0]
    r_nom = summary.loc[summary.stress_type == "nominal", "mean_distance_return"].iloc[0]
    floor = r_rand + threshold * (r_nom - r_rand)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(rows["level"], rows["mean_distance_return"], yerr=rows["std_distance_return"],
                marker="o", capsize=3, label="TD3 policy (mean ± std)")
    ax.axhline(r_nom, ls="--", color="gray", label="nominal")
    ax.axhline(floor, ls="--", color="orange", label=f"{threshold:.0%} retention floor")
    ax.axhline(r_rand, ls=":", color="red", label="random policy")
    if boundary["level"] is not None:
        if boundary["reached"]:
            boundary_label = f"safety boundary = {boundary['level']}"
        else:
            boundary_label = f"boundary not reached (safe up to {boundary['level']})"
        ax.axvline(boundary["level"], color="green", alpha=0.6, label=boundary_label)
    if len(rows) > 1 and rows["level"].iloc[0] > rows["level"].iloc[-1]:
        ax.invert_xaxis()  # degradation: severity grows to the right
    ax.set_xlabel(STRESS_XLABEL[stress_type])
    ax.set_ylabel("task return: -sum of distance to target\n(higher is better)")
    ax.set_title(f"Reacher TD3: {stress_type} stress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_report(summary: pd.DataFrame, out_dir, threshold: float) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = {}
    for stress_type, filename in STRESS_FILE.items():
        boundary = safety_boundary(summary, stress_type, threshold)
        boundaries[stress_type] = boundary
        fig = plot_stress_curve(summary, stress_type, boundary, threshold)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    summary.to_csv(out_dir / "ipqc_results.csv", index=False)
    return boundaries


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="IPQC stress test for a trained Reacher TD3 or DDPG policy.")
    p.add_argument("--config", default=None, help="config path (relative paths resolve against the project root)")
    p.add_argument("--model-path", default=None)
    p.add_argument("--algo", choices=("td3", "ddpg"), default=None)
    p.add_argument("--episodes", type=int, default=None, help="episodes per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", default="plots")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    algo = args.algo or cfg["algo"]
    model = load_policy(args.model_path or default_model_path(cfg, algo), algo=algo)
    raw = run_ipqc_sweep(model, cfg, episodes_per_seed=args.episodes, seeds=args.seeds)
    summary = summarize(raw)
    threshold = cfg["ipqc"]["retention_threshold"]
    out_dir = resolve_path(args.out)
    boundaries = save_report(summary, out_dir, threshold)
    print(summary.to_string(index=False))
    for stress_type, b in boundaries.items():
        print(describe_boundary(stress_type, b, threshold))
    print(f"Wrote plots and ipqc_results.csv to {out_dir}")


if __name__ == "__main__":
    main()
