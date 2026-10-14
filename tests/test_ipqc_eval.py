import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_ipqc import (
    STRESS_FILE,
    describe_boundary,
    load_policy,
    run_ipqc_sweep,
    safety_boundary,
    save_report,
    summarize,
)
from scripts.train import load_config


def episode_df(means):
    """Build a raw-episode frame from {(stress_type, level, order): mean_return} (2 episodes each)."""
    rows = []
    for (stress_type, level, order), mean in means.items():
        for ep, delta in enumerate((-1.0, 1.0)):
            rows.append(dict(stress_type=stress_type, level=level, order=order, seed=0,
                             episode=ep, episode_return=mean + delta))
    return pd.DataFrame(rows)


BASELINES = {("random", 0.0, 0): -40.0, ("nominal", 0.0, 0): -5.0}


def test_summarize_normalizes_between_random_and_nominal():
    df = episode_df({**BASELINES, ("degradation", 1.0, 0): -5.0, ("degradation", 0.6, 1): -22.5})
    s = summarize(df).set_index(["stress_type", "level"])
    assert s.loc[("degradation", 1.0), "score"] == pytest.approx(1.0)
    assert s.loc[("degradation", 0.6), "score"] == pytest.approx(0.5)
    assert s.loc[("random", 0.0), "score"] == pytest.approx(0.0)


def test_summarize_raises_when_nominal_not_better_than_random():
    df = episode_df({("random", 0.0, 0): -5.0, ("nominal", 0.0, 0): -20.0})
    with pytest.raises(ValueError, match="Nominal return"):
        summarize(df)


def boundary_summary(scores):
    return pd.DataFrame(
        [dict(stress_type="latency", level=float(i), order=i, score=s) for i, s in enumerate(scores)]
    )


def test_safety_boundary_stops_at_first_failure_even_if_later_levels_recover():
    b = safety_boundary(boundary_summary([1.0, 0.95, 0.6, 0.95]), "latency", 0.9)
    assert b == {"level": 1.0, "reached": True, "first_failure": 2.0}


def test_safety_boundary_all_levels_pass():
    b = safety_boundary(boundary_summary([1.0, 0.99, 0.95]), "latency", 0.9)
    assert b == {"level": 2.0, "reached": False, "first_failure": None}


def test_safety_boundary_mildest_level_fails():
    b = safety_boundary(boundary_summary([0.5, 0.4]), "latency", 0.9)
    assert b == {"level": None, "reached": True, "first_failure": 0.0}


def test_describe_boundary_mentions_level():
    text = describe_boundary("latency", {"level": 2.0, "reached": True, "first_failure": 3.0}, 0.9)
    assert "latency" in text and "2.0" in text and "3.0" in text


def test_load_policy_missing_model_points_to_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_policy(tmp_path / "nope")


def test_save_report_writes_plots_and_csv(tmp_path):
    means = dict(BASELINES)
    for i, lvl in enumerate([1.0, 0.8, 0.6]):
        means[("degradation", lvl, i)] = -5.0 - 5 * i
    for i, lvl in enumerate([0.0, 1.0, 2.0]):
        means[("latency", lvl, i)] = -5.0 - 2 * i
    for i, lvl in enumerate([0.0, 0.05, 0.1]):
        means[("deadband", lvl, i)] = -5.0 - i
    summary = summarize(episode_df(means))
    boundaries = save_report(summary, tmp_path, 0.9)
    assert set(boundaries) == set(STRESS_FILE)
    for name in STRESS_FILE.values():
        assert (tmp_path / name).stat().st_size > 0
    assert (tmp_path / "ipqc_results.csv").exists()


def _plot_legend_labels(boundary):
    from scripts.evaluate_ipqc import plot_stress_curve

    means = {**BASELINES, ("latency", 0.0, 0): -5.0, ("latency", 1.0, 1): -5.5}
    summary = summarize(episode_df(means))
    fig = plot_stress_curve(summary, "latency", boundary, 0.9)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    import matplotlib.pyplot as plt

    plt.close(fig)
    return labels


def test_plot_labels_unreached_boundary_as_not_reached():
    labels = _plot_legend_labels({"level": 1.0, "reached": False, "first_failure": None})
    assert any("not reached" in label for label in labels)


def test_plot_labels_reached_boundary_as_safety_boundary():
    labels = _plot_legend_labels({"level": 1.0, "reached": True, "first_failure": 2.0})
    assert any(label == "safety boundary = 1.0" for label in labels)
    assert not any("not reached" in label for label in labels)


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_sweep_covers_every_condition():
    pytest.importorskip("mujoco")
    cfg = load_config()
    ip = cfg["ipqc"]
    df = run_ipqc_sweep(ZeroModel(), cfg, episodes_per_seed=1, seeds=[0])
    expected = 2 + len(ip["degradation"]) + len(ip["latency_steps"]) + len(ip["deadband"])
    assert len(df) == expected
    assert set(df.stress_type) == {"random", "nominal", "degradation", "latency", "deadband"}
    assert list(df.columns) == ["stress_type", "level", "order", "seed", "episode", "episode_return"]
