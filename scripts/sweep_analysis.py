"""Learning-stability analysis for the reward-shaping sweep."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
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


ALGO_COLORS = {"td3": "#1f77b4", "ddpg": "#d62728"}


def bootstrap_band(matrix, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Per-column mean and percentile bootstrap band over the rows (seeds) of an (n_seeds, T) matrix."""
    m = np.asarray(matrix, dtype=float)
    mean = m.mean(axis=0)
    if m.shape[0] == 1:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    boots = m[idx].mean(axis=1)  # (n_boot, T)
    return mean, np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def plot_learning_curves(curves: pd.DataFrame, random_baseline: float, threshold: float):
    variants = list(dict.fromkeys(curves["variant"]))
    ncols = 3
    nrows = int(np.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False, sharey=True)
    for ax in axes.ravel()[len(variants):]:
        ax.axis("off")
    for ax, variant in zip(axes.ravel(), variants):
        for algo in sorted(curves["algo"].unique()):
            sub = curves[(curves["variant"] == variant) & (curves["algo"] == algo)]
            if sub.empty:
                continue
            wide = sub.pivot(index="seed", columns="timesteps", values="mean_distance_return").sort_index(axis=1)
            steps = wide.columns.to_numpy()
            for _, row in wide.iterrows():  # thin per-seed lines
                ax.plot(steps, row.to_numpy(), color=ALGO_COLORS.get(algo), alpha=0.25, lw=0.8)
            mean, lo, hi = bootstrap_band(wide.to_numpy())
            ax.fill_between(steps, lo, hi, color=ALGO_COLORS.get(algo), alpha=0.18)
            ax.plot(steps, mean, color=ALGO_COLORS.get(algo), lw=2, label=f"{algo.upper()} (n={len(wide)})")
        ax.axhline(random_baseline, color="gray", ls=":", label="random policy")
        ax.axhline(threshold, color="orange", ls="--", lw=1, label="solve threshold")
        ax.set_title(variant)
        ax.set_xlabel("environment steps")
    axes[0, 0].set_ylabel("task return: -sum distance (higher is better)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves on the unshaped task (thin = seeds, thick = mean, band = 95% bootstrap CI)")
    fig.tight_layout()
    return fig


def plot_metric_comparison(summary: pd.DataFrame):
    panels = [("final_return", "final task return (higher is better)"),
              ("final_success", "final success rate"),
              ("max_drawdown", "max drawdown (lower is better)")]
    variants = list(dict.fromkeys(summary["variant"]))
    algos = sorted(summary["algo"].unique())
    width = 0.8 / len(algos)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(variants))
    for ax, (metric, title) in zip(axes, panels):
        for j, algo in enumerate(algos):
            sub = summary[summary["algo"] == algo].set_index("variant").reindex(variants)
            mean = sub[f"{metric}_mean"].to_numpy(dtype=float)
            err = np.vstack([mean - sub[f"{metric}_lo"].to_numpy(dtype=float),
                             sub[f"{metric}_hi"].to_numpy(dtype=float) - mean])
            ax.bar(x + j * width - 0.4 + width / 2, mean, width, yerr=err, capsize=3,
                   color=ALGO_COLORS.get(algo), label=algo.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20)
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    return fig


def write_sweep_outputs(runs_dir, out_dir, threshold: float) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_eval_curves(runs_dir)
    baseline_path = resolve_path(runs_dir) / "random_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    per_run = per_run_table(curves, threshold)
    summary = summarize_runs(per_run)
    diffs = compare_algos(per_run)

    paths = {
        "curves": out_dir / "sweep_eval_curves.csv",
        "per_run": out_dir / "sweep_per_run.csv",
        "summary": out_dir / "sweep_summary.csv",
        "diffs": out_dir / "sweep_algo_differences.csv",
        "baseline": out_dir / "sweep_random_baseline.json",
        "curves_png": out_dir / "sweep_learning_curves.png",
        "comparison_png": out_dir / "sweep_metric_comparison.png",
    }
    curves.to_csv(paths["curves"], index=False)
    per_run.to_csv(paths["per_run"], index=False)
    summary.to_csv(paths["summary"], index=False)
    diffs.to_csv(paths["diffs"], index=False)
    paths["baseline"].write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    fig = plot_learning_curves(curves, baseline["mean_distance_return"], threshold)
    fig.savefig(paths["curves_png"], dpi=130)
    plt.close(fig)
    fig = plot_metric_comparison(summary)
    fig.savefig(paths["comparison_png"], dpi=130)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scripts.sweep import load_sweep_config

    p = argparse.ArgumentParser(description="Analyse a finished reward-shaping sweep.")
    p.add_argument("--runs-dir", default=None, help="default: runs_dir from the sweep config")
    p.add_argument("--out", default="plots")
    args = p.parse_args(argv)
    cfg = load_sweep_config()
    paths = write_sweep_outputs(args.runs_dir or cfg["runs_dir"], resolve_path(args.out), cfg["solve_threshold"])
    summary = pd.read_csv(paths["summary"])
    cols = ["algo", "variant", "n_seeds", "n_reached", "final_return_mean", "final_success_mean",
            "max_drawdown_mean"]
    print(summary[cols].round(3).to_string(index=False))
    print("Wrote:", ", ".join(p.name for p in paths.values()))


if __name__ == "__main__":
    main()
