import json
from pathlib import Path

import pandas as pd


def make_fake_runs(runs_dir, runs, baseline=-10.0, eval_every=2000):
    """Fabricate completed run directories. runs: [(algo, variant, seed, values, success_or_None)]."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for algo, variant, seed, values, success in runs:
        rd = runs_dir / f"{algo}_{variant}_seed{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        success = success if success is not None else [0.0] * len(values)
        pd.DataFrame({
            "timesteps": [i * eval_every for i in range(len(values))],
            "mean_distance_return": values,
            "std_distance_return": [0.5] * len(values),
            "mean_total_return": [v - 1.0 for v in values],
            "success_rate": success,
        }).to_csv(rd / "eval.csv", index=False)
        (rd / "done.json").write_text(json.dumps({
            "algo": algo, "variant": variant, "seed": seed,
            "timesteps": eval_every * (len(values) - 1), "seconds": 1.0, "final_model": "final_model.zip",
        }), encoding="utf-8")
    (runs_dir / "random_baseline.json").write_text(json.dumps({
        "mean_distance_return": baseline, "std_distance_return": 1.0, "episodes": 100,
    }), encoding="utf-8")
    return runs_dir
