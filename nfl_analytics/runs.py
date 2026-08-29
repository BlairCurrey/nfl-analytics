"""
Manages training run artifacts.

Each training run produces a self-contained directory under assets/runs/:

    assets/runs/<run_id>/
        manifest.json           # written last; marks the run as complete
        model.joblib
        scaler.joblib
        running_average.csv.gz
        matchups.json           # written by predict-upcoming
        predictions.json        # written by predict-upcoming

The model, scaler, and running average dataframe are only ever loaded
together from the same run, so they can never be mismatched. A run without
a manifest.json is incomplete (e.g. a crashed training run) and is ignored.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from nfl_analytics.config import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    PREDICTIONS_FILENAME,
    RUNNING_AVG_FILENAME,
    SCALER_FILENAME,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(THIS_DIR, "assets", "runs")


class RunNotFoundError(Exception):
    pass


def save_run(
    model: LinearRegression,
    scaler: StandardScaler,
    df_running_avg: pd.DataFrame,
    metrics: dict[str, float],
    runs_dir: str = DEFAULT_RUNS_DIR,
) -> str:
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%d%H%M%S")
    run_dir = get_run_dir(run_id, runs_dir)
    os.makedirs(run_dir, exist_ok=True)

    dump(model, os.path.join(run_dir, MODEL_FILENAME))
    dump(scaler, os.path.join(run_dir, SCALER_FILENAME))
    df_running_avg.to_csv(
        os.path.join(run_dir, RUNNING_AVG_FILENAME), index=False, compression="gzip"
    )

    # Manifest goes last so an interrupted run is never mistaken for a complete one
    manifest = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, MANIFEST_FILENAME), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved training run {run_id} to {run_dir}")
    return run_id


def get_run_dir(run_id: str, runs_dir: str = DEFAULT_RUNS_DIR) -> str:
    return os.path.join(runs_dir, run_id)


def find_latest_run_id(runs_dir: str = DEFAULT_RUNS_DIR) -> Optional[str]:
    if not os.path.isdir(runs_dir):
        return None

    complete_runs = [
        entry
        for entry in os.listdir(runs_dir)
        if os.path.isfile(os.path.join(runs_dir, entry, MANIFEST_FILENAME))
    ]

    if not complete_runs:
        return None

    return sorted(complete_runs)[-1]


def load_run(
    run_id: Optional[str] = None, runs_dir: str = DEFAULT_RUNS_DIR
) -> tuple[LinearRegression, StandardScaler, pd.DataFrame, dict[str, Any]]:
    if run_id is None:
        run_id = find_latest_run_id(runs_dir)

    if run_id is None:
        raise RunNotFoundError(
            "No trained model found. Run `nfl update` to download data and train one."
        )

    run_dir = get_run_dir(run_id, runs_dir)
    manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)

    if not os.path.isfile(manifest_path):
        raise RunNotFoundError(f"No complete training run found at {run_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading training run {run_id} from {run_dir}")
    model = load(os.path.join(run_dir, MODEL_FILENAME))
    scaler = load(os.path.join(run_dir, SCALER_FILENAME))
    df_running_avg = pd.read_csv(
        os.path.join(run_dir, RUNNING_AVG_FILENAME), low_memory=False
    )

    return model, scaler, df_running_avg, manifest


def save_run_json(
    run_id: str, filename: str, records: list, runs_dir: str = DEFAULT_RUNS_DIR
) -> str:
    """Save a list of dataclasses (e.g. matchups, predictions) into a run directory."""
    filepath = os.path.join(get_run_dir(run_id, runs_dir), filename)
    with open(filepath, "w") as f:
        json.dump([asdict(record) for record in records], f, indent=2)

    print(f"Saved {filepath}")
    return filepath


def has_predictions(run_id: str, runs_dir: str = DEFAULT_RUNS_DIR) -> bool:
    return os.path.isfile(
        os.path.join(get_run_dir(run_id, runs_dir), PREDICTIONS_FILENAME)
    )
