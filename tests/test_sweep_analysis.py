import numpy as np
import pandas as pd
import pytest
from sweep_fakes import make_fake_runs

from scripts.sweep_analysis import (
    bootstrap_ci,
    compare_algos,
    list_completed_runs,
    load_eval_curves,
    per_run_table,
    run_metrics,
    summarize_runs,
)


def curve(values, success=None):
    n = len(values)
    return pd.DataFrame({
        "timesteps": [2000 * i for i in range(n)],
        "mean_distance_return": values,
        "success_rate": success if success is not None else [0.0] * n,
    })


def test_run_metrics_final_auc_threshold_and_drawdown():
    m = run_metrics(curve([-8, -5, -3, -6, -3.5], [0, 0.2, 0.8, 0.4, 0.7]), threshold=-4.0)
    assert m["final_return"] == pytest.approx(np.mean([-3, -6, -3.5]))
    assert m["final_success"] == pytest.approx(np.mean([0.8, 0.4, 0.7]))
    assert m["auc"] == pytest.approx(np.mean([-8, -5, -3, -6, -3.5]))
    assert m["steps_to_threshold"] == 4000  # first eval at or above -4.0 is index 2
    assert m["max_drawdown"] == pytest.approx(3.0)  # peak -3 down to -6


def test_run_metrics_threshold_never_reached_and_monotone_curve():
    m = run_metrics(curve([-9, -8, -7, -6]), threshold=-4.0)
    assert np.isnan(m["steps_to_threshold"]) and m["max_drawdown"] == 0.0


def test_run_metrics_short_curve():
    m = run_metrics(curve([-5.0, -4.0]), threshold=-4.5)  # fewer evaluations than last_k=3
    assert m["final_return"] == pytest.approx(-4.5) and m["steps_to_threshold"] == 2000


def test_run_metrics_drawdown_ignores_evaluations_before_learning_starts():
    # Evaluations at 0, 2000, 4000 precede learning_starts=5000 (an untrained, flat segment); the
    # first evaluation after learning begins can dip. That dip is not instability during learning.
    c = curve([-9.0, -9.0, -9.0, -10.5, -6.0, -4.0])
    assert run_metrics(c, -4.0)["max_drawdown"] == pytest.approx(1.5)  # legacy: whole curve
    assert run_metrics(c, -4.0, learning_starts=5000)["max_drawdown"] == 0.0


def test_run_metrics_drawdown_still_detects_instability_after_learning_starts():
    c = curve([-9.0, -9.0, -9.0, -5.0, -3.0, -6.0])
    assert run_metrics(c, -4.0, learning_starts=5000)["max_drawdown"] == pytest.approx(3.0)


def test_run_metrics_drawdown_is_nan_without_post_learning_evaluations():
    m = run_metrics(curve([-9.0, -9.0]), -4.0, learning_starts=5000)
    assert np.isnan(m["max_drawdown"]) and m["final_return"] == pytest.approx(-9.0)


def test_per_run_table_passes_learning_starts_to_drawdown(tmp_path):
    make_fake_runs(tmp_path, [("td3", "default", 0, [-9.0, -9.0, -9.0, -10.5, -6.0, -4.0], None)])
    curves = load_eval_curves(tmp_path)
    assert per_run_table(curves, -4.0)["max_drawdown"].iloc[0] == pytest.approx(1.5)
    assert per_run_table(curves, -4.0, learning_starts=5000)["max_drawdown"].iloc[0] == 0.0


def test_bootstrap_ci_constant_sample_is_exact():
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    a = bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    assert a == bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    mean, lo, hi = a
    assert mean == 3.0 and lo <= mean <= hi and lo >= 1 and hi <= 5


def test_bootstrap_ci_single_value_is_degenerate():
    assert bootstrap_ci([-3.0]) == (-3.0, -3.0, -3.0)


def test_bootstrap_ci_drops_nan_and_handles_empty():
    assert bootstrap_ci([1.0, float("nan"), 1.0]) == (1.0, 1.0, 1.0)
    mean, lo, hi = bootstrap_ci([float("nan")])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_load_eval_curves_with_no_completed_runs_raises(tmp_path):
    (tmp_path / "td3_default_seed0").mkdir()
    with pytest.raises(ValueError, match="no completed runs"):
        load_eval_curves(tmp_path)


def test_load_eval_curves_ignores_runs_without_done_json(tmp_path):
    make_fake_runs(tmp_path, [("td3", "default", 0, [-8, -4], None), ("td3", "default", 1, [-8, -5], None)])
    (tmp_path / "td3_default_seed1" / "done.json").unlink()  # crashed run
    curves = load_eval_curves(tmp_path)
    assert list(curves["seed"].unique()) == [0]
    assert [p.name for p in list_completed_runs(tmp_path)] == ["td3_default_seed0"]
    assert {"algo", "variant", "seed", "timesteps", "mean_distance_return", "success_rate"} <= set(curves.columns)


def fake_grid(tmp_path):
    runs = []
    for seed, (a, b) in enumerate([(-3.0, -7.0), (-3.2, -6.0), (-3.1, -5.0)]):
        runs.append(("td3", "default", seed, [-9, -6, -4, a, a, a], [0, 0.2, 0.6, 0.9, 0.9, 0.9]))
        runs.append(("ddpg", "default", seed, [-9, -8, -7, b, b, b], None))
    make_fake_runs(tmp_path, runs)
    return load_eval_curves(tmp_path)


def test_per_run_and_summary_tables(tmp_path):
    per_run = per_run_table(fake_grid(tmp_path), threshold=-4.0)
    assert len(per_run) == 6
    summary = summarize_runs(per_run).set_index(["algo", "variant"])
    td3, ddpg = summary.loc[("td3", "default")], summary.loc[("ddpg", "default")]
    assert td3["n_seeds"] == 3 and td3["n_reached"] == 3
    assert ddpg["n_reached"] == 0 and np.isnan(ddpg["steps_to_threshold_mean"])
    assert td3["final_return_mean"] == pytest.approx(np.mean([-3.0, -3.2, -3.1]))
    assert td3["final_return_lo"] <= td3["final_return_mean"] <= td3["final_return_hi"]


def test_compare_algos_reports_td3_minus_ddpg_with_interval(tmp_path):
    per_run = per_run_table(fake_grid(tmp_path), threshold=-4.0)
    diff = compare_algos(per_run).set_index("variant").loc["default"]
    assert diff["diff_mean"] == pytest.approx(np.mean([-3.0, -3.2, -3.1]) - np.mean([-7.0, -6.0, -5.0]))
    assert diff["diff_lo"] > 0  # TD3 clearly better in this fabricated data


import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from scripts.sweep_analysis import (  # noqa: E402
    bootstrap_band,
    plot_learning_curves,
    plot_metric_comparison,
    write_sweep_outputs,
)


def test_bootstrap_band_shapes_and_ordering():
    m = np.array([[-5.0, -4.0], [-6.0, -3.0], [-4.0, -3.5]])
    mean, lo, hi = bootstrap_band(m)
    assert mean.shape == lo.shape == hi.shape == (2,)
    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9)
    assert mean == pytest.approx(m.mean(axis=0))


def grid_runs():
    runs = []
    for algo, base in (("td3", -3.0), ("ddpg", -6.0)):
        for variant in ("default", "sparse"):
            for seed in range(3):
                shift = (1.0 if variant == "sparse" else 0.0) + 0.1 * seed
                runs.append((algo, variant, seed, [-9, -7, base - shift, base - shift], [0, 0.1, 0.5, 0.6]))
    return runs


def test_plots_render_without_error(tmp_path):
    make_fake_runs(tmp_path, grid_runs())
    curves = load_eval_curves(tmp_path)
    fig = plot_learning_curves(curves, -10.0, -4.0)
    assert len(fig.axes) >= 2  # one panel per variant (+ legend)
    plt.close(fig)
    summary = summarize_runs(per_run_table(curves, -4.0))
    fig = plot_metric_comparison(summary)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_write_sweep_outputs_creates_every_file(tmp_path):
    runs = make_fake_runs(tmp_path / "runs", grid_runs())
    paths = write_sweep_outputs(runs, tmp_path / "out", -4.0)
    expected = {"sweep_eval_curves.csv", "sweep_per_run.csv", "sweep_summary.csv",
                "sweep_algo_differences.csv", "sweep_random_baseline.json",
                "sweep_learning_curves.png", "sweep_metric_comparison.png"}
    assert {p.name for p in paths.values()} == expected
    assert all(p.stat().st_size > 0 for p in paths.values())
    curves = pd.read_csv(tmp_path / "out" / "sweep_eval_curves.csv")
    assert set(curves["algo"]) == {"td3", "ddpg"} and len(curves) == 12 * 4
    summary = pd.read_csv(tmp_path / "out" / "sweep_summary.csv")
    assert {"final_return_mean", "final_return_lo", "final_return_hi", "n_reached"} <= set(summary.columns)
    base = json.loads((tmp_path / "out" / "sweep_random_baseline.json").read_text(encoding="utf-8"))
    assert base["mean_distance_return"] == -10.0
