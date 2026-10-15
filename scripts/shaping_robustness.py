"""Do reward-shaped policies differ in how they cope with the IPQC failure modes?

Every solved final policy from the sweep is evaluated under a reduced stress battery on the
unshaped task. Policies that never learned (own nominal return below the solve threshold) are
excluded from the comparison and counted, never scored.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from envs import make_stressed_env  # noqa: E402
from scripts.evaluate_ipqc import (  # noqa: E402
    STRESS_KWARG,
    STRESS_XLABEL,
    load_policy,
    run_episode_parts,
    safety_boundary,
)
from scripts.sweep_analysis import bootstrap_ci, list_completed_runs  # noqa: E402
from scripts.train import load_config, resolve_path  # noqa: E402

CONFIG_KEY = {"degradation": "degradation", "latency": "latency_steps", "deadband": "deadband"}
VARIANT_COLORS = {"default": "#1f77b4", "dist_only": "#ff7f0e", "sparse": "#7f7f7f",
                  "progress": "#2ca02c", "scaled_x10": "#9467bd"}


def robustness_conditions(rob_cfg: dict) -> list:
    conds = [("nominal", 0.0, 0, {})]
    for stress_type, key in CONFIG_KEY.items():
        for order, level in enumerate(rob_cfg[key]):
            conds.append((stress_type, float(level), order, {STRESS_KWARG[stress_type]: level}))
    return conds


def evaluate_policy_under_stress(policy, env_id, conditions, episodes, eval_seed, success_radius) -> list[dict]:
    rows = []
    for stress_type, level, order, kwargs in conditions:
        env = make_stressed_env(env_id, **kwargs)
        parts = [run_episode_parts(env, policy, seed=eval_seed if i == 0 else None) for i in range(episodes)]
        env.close()
        rows.append({
            "stress_type": stress_type, "level": level, "order": order,
            "mean_distance_return": float(np.mean([p["distance_return"] for p in parts])),
            "mean_total_return": float(np.mean([p["episode_return"] for p in parts])),
            "success_rate": float(np.mean([p["final_distance"] <= success_radius for p in parts])),
        })
    return rows


def evaluate_run(args: dict) -> list[dict]:
    """Top-level (picklable) worker: load one run's final model and stress-test it."""
    import torch

    torch.set_num_threads(1)
    run_dir = Path(args["run_dir"])
    done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    policy = load_policy(run_dir / done["final_model"], algo=done["algo"])
    rows = evaluate_policy_under_stress(policy, args["env_id"], args["conditions"], args["episodes"],
                                        args["eval_seed"], args["success_radius"])
    return [{"algo": done["algo"], "variant": done["variant"], "seed": done["seed"], **r} for r in rows]


def score_rows(df: pd.DataFrame, random_distance: float, solve_threshold: float) -> pd.DataFrame:
    """Retention (R - R_random)/(R_own_nominal - R_random); ineligible policies get NaN."""
    if solve_threshold <= random_distance:
        raise ValueError(
            f"solve_threshold ({solve_threshold}) must be above the random-policy return ({random_distance})"
        )
    keys = ["algo", "variant", "seed"]
    nominal = (df[df.stress_type == "nominal"].set_index(keys)["mean_distance_return"]
               .rename("nominal_distance").reset_index())
    out = df.merge(nominal, on=keys, how="left")
    out["eligible"] = out["nominal_distance"] >= solve_threshold
    retention = (out["mean_distance_return"] - random_distance) / (out["nominal_distance"] - random_distance)
    out["score"] = np.where(out["eligible"], retention, np.nan)
    return out


def eligibility_table(scored: pd.DataFrame) -> pd.DataFrame:
    per_policy = scored.drop_duplicates(["algo", "variant", "seed"])
    return (per_policy.groupby(["algo", "variant"], as_index=False)
            .agg(n_policies=("seed", "count"), n_eligible=("eligible", "sum")))


def summarize_robustness(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stressed = scored[(scored.stress_type != "nominal") & scored["eligible"]]
    for (algo, variant, stress, level, order), g in stressed.groupby(
            ["algo", "variant", "stress_type", "level", "order"], sort=True):
        mean, lo, hi = bootstrap_ci(g["score"].to_numpy())
        rows.append({"algo": algo, "variant": variant, "stress_type": stress, "level": level, "order": order,
                     "retention_mean": mean, "retention_lo": lo, "retention_hi": hi,
                     "n_eligible": int(len(g))})
    return pd.DataFrame(rows)


def policy_boundaries(scored: pd.DataFrame, retention_threshold: float) -> pd.DataFrame:
    rows = []
    stressed = scored[(scored.stress_type != "nominal") & scored["eligible"]]
    for (algo, variant, seed), g in stressed.groupby(["algo", "variant", "seed"], sort=True):
        for stress_type in sorted(g.stress_type.unique()):
            b = safety_boundary(g, stress_type, retention_threshold)
            rows.append({"algo": algo, "variant": variant, "seed": seed, "stress_type": stress_type,
                         "boundary_level": b["level"], "reached": b["reached"]})
    return pd.DataFrame(rows)


def summarize_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame(columns=["algo", "variant", "stress_type", "boundary_mean", "n_eligible",
                                     "n_not_reached"])
    return (boundaries.groupby(["algo", "variant", "stress_type"], as_index=False)
            .agg(boundary_mean=("boundary_level", "mean"), n_eligible=("seed", "count"),
                 n_not_reached=("reached", lambda s: int((~s.astype(bool)).sum()))))


def plot_robustness(summary: pd.DataFrame):
    algos = sorted(summary["algo"].unique())
    stresses = ["degradation", "latency", "deadband"]
    fig, axes = plt.subplots(len(algos), 3, figsize=(15, 3.6 * len(algos)), squeeze=False, sharey=True)
    for i, algo in enumerate(algos):
        for j, stress in enumerate(stresses):
            ax = axes[i, j]
            for variant in dict.fromkeys(summary["variant"]):
                g = summary[(summary.algo == algo) & (summary.stress_type == stress)
                            & (summary.variant == variant)].sort_values("order")
                if g.empty:
                    continue
                c = VARIANT_COLORS.get(variant)
                ax.plot(g["level"], g["retention_mean"], marker="o", color=c,
                        label=f"{variant} (n={int(g['n_eligible'].min())})")
                ax.fill_between(g["level"], g["retention_lo"], g["retention_hi"], color=c, alpha=0.15)
            ax.axhline(0.9, color="orange", ls="--", lw=1)
            ax.axhline(0.0, color="red", ls=":", lw=1)
            levels = summary[summary.stress_type == stress]["level"]
            if stress == "degradation" and len(levels):
                ax.invert_xaxis()
            ax.set_title(f"{algo.upper()}: {stress}")
            ax.set_xlabel(STRESS_XLABEL[stress])
    axes[0, 0].set_ylabel("task retention (1 = healthy robot, 0 = random)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Robustness of solved policies by training reward (mean over eligible seeds, 95% bootstrap CI)")
    fig.tight_layout()
    return fig


def run_robustness(sweep_cfg: dict, train_cfg: dict, out_dir, workers: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = resolve_path(sweep_cfg["runs_dir"])
    runs = list_completed_runs(runs_dir)
    if not runs:
        raise ValueError(f"no completed runs in {runs_dir}; run `python scripts/sweep.py` first")
    rob = sweep_cfg["robustness"]
    conditions = robustness_conditions(rob)
    baseline = json.loads((runs_dir / "random_baseline.json").read_text(encoding="utf-8"))
    random_distance = baseline["mean_distance_return"]
    jobs = [{"run_dir": str(p), "env_id": train_cfg["env_id"], "conditions": conditions,
             "episodes": rob["episodes"], "eval_seed": rob["eval_seed"],
             "success_radius": train_cfg["success_radius"]} for p in runs]
    workers = workers or sweep_cfg["workers"]
    rows: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            rows.extend(evaluate_run(job))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for result in pool.map(evaluate_run, jobs):
                rows.extend(result)

    scored = score_rows(pd.DataFrame(rows), random_distance, sweep_cfg["solve_threshold"])
    summary = summarize_robustness(scored)
    boundaries = policy_boundaries(scored, rob["retention_threshold"])
    paths = {
        "raw": out_dir / "shaping_robustness.csv",
        "eligibility": out_dir / "shaping_eligibility.csv",
        "summary": out_dir / "shaping_robustness_summary.csv",
        "boundaries": out_dir / "shaping_boundaries.csv",
        "boundaries_summary": out_dir / "shaping_boundaries_summary.csv",
        "png": out_dir / "shaping_robustness.png",
    }
    scored.to_csv(paths["raw"], index=False)
    eligibility_table(scored).to_csv(paths["eligibility"], index=False)
    summary.to_csv(paths["summary"], index=False)
    boundaries.to_csv(paths["boundaries"], index=False)
    summarize_boundaries(boundaries).to_csv(paths["boundaries_summary"], index=False)
    if summary.empty:
        raise ValueError("no policy reached the solve threshold, so there is nothing to compare")
    fig = plot_robustness(summary)
    fig.savefig(paths["png"], dpi=130)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scripts.sweep import load_sweep_config, quick_config

    p = argparse.ArgumentParser(description="Stress-test the final policies of a finished sweep.")
    p.add_argument("--out", default="plots")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="read runs/quick (written by sweep.py --quick)")
    args = p.parse_args(argv)
    sweep_cfg = load_sweep_config()
    if args.quick:
        sweep_cfg = quick_config(sweep_cfg)
    out = resolve_path("plots/quick" if args.quick else args.out)
    paths = run_robustness(sweep_cfg, load_config(), out, workers=args.workers)
    print(pd.read_csv(paths["eligibility"]).to_string(index=False))
    print(pd.read_csv(paths["boundaries_summary"]).round(3).to_string(index=False))
    print("Wrote:", ", ".join(p.name for p in paths.values()))


if __name__ == "__main__":
    main()
