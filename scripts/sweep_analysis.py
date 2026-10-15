"""Learning-stability analysis for the reward-shaping sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.train import resolve_path  # noqa: E402

CURVE_COLUMNS = ["algo", "variant", "seed", "timesteps", "mean_distance_return",
                 "std_distance_return", "mean_total_return", "success_rate"]
METRICS = ["final_return", "final_success", "auc", "steps_to_threshold", "max_drawdown"]


def list_completed_runs(runs_dir) -> list[Path]:
    """Run directories that finished (have done.json), sorted. Crashed runs are not listed."""
    runs_dir = resolve_path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "done.json").exists())


def load_eval_curves(runs_dir) -> pd.DataFrame:
    paths = list_completed_runs(runs_dir)
    if not paths:
        raise ValueError(
            f"no completed runs found in {resolve_path(runs_dir)}; run `python scripts/sweep.py` first"
        )
    frames = []
    for path in paths:
        done = json.loads((path / "done.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(path / "eval.csv")
        frame.insert(0, "seed", done["seed"])
        frame.insert(0, "variant", done["variant"])
        frame.insert(0, "algo", done["algo"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[CURVE_COLUMNS]


def run_metrics(curve: pd.DataFrame, threshold: float, last_k: int = 3) -> dict:
    c = curve.sort_values("timesteps")
    r = c["mean_distance_return"].to_numpy(dtype=float)
    t = c["timesteps"].to_numpy()
    s = c["success_rate"].to_numpy(dtype=float)
    k = min(last_k, len(r))
    reached = np.nonzero(r >= threshold)[0]
    running_max = np.maximum.accumulate(r)
    return {
        "final_return": float(r[-k:].mean()),
        "final_success": float(s[-k:].mean()),
        "auc": float(r.mean()),
        "steps_to_threshold": float(t[reached[0]]) if reached.size else float("nan"),
        "max_drawdown": float((running_max - r).max()),
    }


def per_run_table(curves: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (algo, variant, seed), group in curves.groupby(["algo", "variant", "seed"], sort=True):
        rows.append({"algo": algo, "variant": variant, "seed": seed, **run_metrics(group, threshold)})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """(mean, lo, hi): percentile bootstrap over the sample. NaNs are dropped; crude for few seeds."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return (float(v[0]),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, variant), group in per_run.groupby(["algo", "variant"], sort=True):
        row = {"algo": algo, "variant": variant, "n_seeds": int(len(group)),
               "n_reached": int(group["steps_to_threshold"].notna().sum())}
        for metric in METRICS:
            mean, lo, hi = bootstrap_ci(group[metric].to_numpy())
            row[f"{metric}_mean"], row[f"{metric}_lo"], row[f"{metric}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def compare_algos(per_run: pd.DataFrame, metric: str = "final_return", a: str = "td3", b: str = "ddpg",
                  n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Per variant: mean(a) - mean(b) with an unpaired bootstrap interval (each algo's seeds resampled)."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, group in per_run.groupby("variant", sort=True):
        va = group.loc[group["algo"] == a, metric].dropna().to_numpy(dtype=float)
        vb = group.loc[group["algo"] == b, metric].dropna().to_numpy(dtype=float)
        if va.size == 0 or vb.size == 0:
            continue
        ma = va[rng.integers(0, va.size, size=(n_boot, va.size))].mean(axis=1)
        mb = vb[rng.integers(0, vb.size, size=(n_boot, vb.size))].mean(axis=1)
        diff = ma - mb
        rows.append({"variant": variant, "metric": metric, "diff_mean": float(va.mean() - vb.mean()),
                     "diff_lo": float(np.quantile(diff, alpha / 2)),
                     "diff_hi": float(np.quantile(diff, 1 - alpha / 2)),
                     f"n_{a}": int(va.size), f"n_{b}": int(vb.size)})
    return pd.DataFrame(rows)
