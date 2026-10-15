import json

import numpy as np
import pandas as pd
import pytest

from scripts.shaping_robustness import (
    eligibility_table,
    policy_boundaries,
    robustness_conditions,
    score_rows,
    summarize_boundaries,
    summarize_robustness,
)

ROB = {"episodes": 2, "eval_seed": 2000, "degradation": [1.0, 0.6], "latency_steps": [0, 2],
       "deadband": [0.0, 0.1], "retention_threshold": 0.9}


def rows_for(algo, variant, seed, nominal, stressed):
    """Raw rows for one policy: nominal return plus stressed returns {(stress, level, order): value}."""
    rows = [dict(algo=algo, variant=variant, seed=seed, stress_type="nominal", level=0.0, order=0,
                 mean_distance_return=nominal, mean_total_return=nominal, success_rate=1.0)]
    for (stress, level, order), value in stressed.items():
        rows.append(dict(algo=algo, variant=variant, seed=seed, stress_type=stress, level=level,
                         order=order, mean_distance_return=value, mean_total_return=value, success_rate=0.5))
    return rows


def test_conditions_start_with_nominal_then_every_level():
    conds = robustness_conditions(ROB)
    assert conds[0] == ("nominal", 0.0, 0, {})
    assert len(conds) == 1 + 2 + 2 + 2
    assert ("latency", 2.0, 1, {"delay_steps": 2}) in conds
    assert ("degradation", 0.6, 1, {"degradation": 0.6}) in conds
    assert ("deadband", 0.1, 1, {"deadband": 0.1}) in conds


def test_retention_uses_the_policys_own_nominal_return():
    raw = pd.DataFrame(rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -5.0}))
    scored = score_rows(raw, random_distance=-10.0, solve_threshold=-4.0)
    row = scored[scored.stress_type == "latency"].iloc[0]
    assert row["eligible"] and row["score"] == pytest.approx((-5.0 + 10.0) / (-3.0 + 10.0))


def test_ineligible_policy_is_excluded_not_scored():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -4.0})
        + rows_for("td3", "sparse", 0, -9.0, {("latency", 2.0, 1): -9.5})  # never learned
    )
    scored = score_rows(raw, random_distance=-10.0, solve_threshold=-4.0)
    bad = scored[scored.variant == "sparse"]
    assert not bad["eligible"].any() and bad["score"].isna().all()
    summary = summarize_robustness(scored)
    assert set(summary["variant"]) == {"default"}  # excluded from the comparison
    table = eligibility_table(scored).set_index("variant")
    assert table.loc["default", "n_eligible"] == 1 and table.loc["sparse", "n_eligible"] == 0
    assert table.loc["sparse", "n_policies"] == 1  # but still counted, never silently dropped


def test_summary_aggregates_over_eligible_seeds_with_interval():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 2.0, 1): -4.0})
        + rows_for("td3", "default", 1, -3.0, {("latency", 2.0, 1): -5.0})
    )
    s = summarize_robustness(score_rows(raw, -10.0, -4.0))
    row = s[(s.stress_type == "latency")].iloc[0]
    scores = [(-4 + 10) / 7, (-5 + 10) / 7]
    assert row["n_eligible"] == 2 and row["retention_mean"] == pytest.approx(np.mean(scores))
    assert row["retention_lo"] <= row["retention_mean"] <= row["retention_hi"]


def test_boundaries_use_first_failure_and_skip_ineligible():
    raw = pd.DataFrame(
        rows_for("td3", "default", 0, -3.0, {("latency", 0.0, 0): -3.0, ("latency", 2.0, 1): -3.2,
                                              ("latency", 4.0, 2): -9.0})
        + rows_for("td3", "sparse", 0, -9.0, {("latency", 0.0, 0): -9.0, ("latency", 2.0, 1): -9.0})
    )
    b = policy_boundaries(score_rows(raw, -10.0, -4.0), retention_threshold=0.9)
    assert set(b["variant"]) == {"default"}
    row = b[b.stress_type == "latency"].iloc[0]
    assert row["boundary_level"] == 2.0 and bool(row["reached"])
    summary = summarize_boundaries(b)
    assert summary.iloc[0]["n_eligible"] == 1


def test_end_to_end_on_tiny_trained_policies(tmp_path):
    pytest.importorskip("mujoco")
    from scripts.shaping_robustness import run_robustness
    from scripts.sweep import load_sweep_config, make_spec, quick_config, run_one
    from scripts.train import load_config

    train_cfg = load_config()
    cfg = quick_config(load_sweep_config())
    cfg.update(runs_dir=str(tmp_path / "runs"), solve_threshold=-1000.0, random_baseline_episodes=3,
               robustness={**ROB, "latency_steps": [0, 2]})
    for algo in ("td3", "ddpg"):
        run_one(make_spec(cfg, train_cfg, {"algo": algo, "variant": "default", "seed": 0}))
    (tmp_path / "runs" / "random_baseline.json").write_text(json.dumps(
        {"mean_distance_return": -1500.0, "std_distance_return": 1.0, "episodes": 3}), encoding="utf-8")
    paths = run_robustness(cfg, train_cfg, tmp_path / "out", workers=1)
    raw = pd.read_csv(paths["raw"])
    assert set(raw["algo"]) == {"td3", "ddpg"} and "nominal" in set(raw["stress_type"])
    assert len(raw) == 2 * (1 + 2 + 2 + 2)
    assert paths["png"].stat().st_size > 0 and paths["boundaries"].exists()


def test_empty_results_keep_their_columns_and_round_trip_through_csv(tmp_path):
    # No policy is eligible: the outputs must still be readable CSVs with headers, not empty files.
    raw = pd.DataFrame(rows_for("td3", "sparse", 0, -9.0, {("latency", 2.0, 1): -9.5}))
    scored = score_rows(raw, random_distance=-10.0, solve_threshold=-4.0)
    summary = summarize_robustness(scored)
    boundaries = policy_boundaries(scored, retention_threshold=0.9)
    assert summary.empty and {"algo", "variant", "stress_type", "level", "order", "retention_mean",
                              "retention_lo", "retention_hi", "n_eligible"} <= set(summary.columns)
    assert boundaries.empty and {"algo", "variant", "seed", "stress_type", "boundary_level",
                                 "reached"} <= set(boundaries.columns)
    for name, frame in (("summary", summary), ("boundaries", boundaries)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        assert list(pd.read_csv(path).columns) == list(frame.columns)
