import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from nfl_analytics import runs
from nfl_analytics.config import MANIFEST_FILENAME
from nfl_analytics.model import Prediction


def make_fitted_model_and_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    scaler = StandardScaler().fit(X)
    model = LinearRegression().fit(scaler.transform(X), y)

    return model, scaler


def test_save_and_load_run_roundtrip(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    df = pd.DataFrame({"team": ["KC", "SF"], "rushing_avg": [120.0, 110.0]})
    metrics = {"mean_squared_error": 1.0, "mean_absolute_error": 0.5}

    run_id = runs.save_run(model, scaler, df, metrics, runs_dir=runs_dir)

    assert runs.find_latest_run_id(runs_dir) == run_id

    loaded_model, loaded_scaler, loaded_df, manifest = runs.load_run(
        runs_dir=runs_dir
    )

    assert manifest["run_id"] == run_id
    assert manifest["metrics"] == metrics
    assert list(loaded_df["team"]) == ["KC", "SF"]
    assert np.allclose(loaded_model.coef_, model.coef_)
    assert np.allclose(loaded_scaler.mean_, scaler.mean_)


def test_incomplete_run_is_ignored(tmp_path):
    runs_dir = str(tmp_path)

    # A run directory without a manifest (e.g. crashed mid-training)
    incomplete_dir = os.path.join(runs_dir, "20990101000000")
    os.makedirs(incomplete_dir)

    assert runs.find_latest_run_id(runs_dir) is None

    model, scaler = make_fitted_model_and_scaler()
    df = pd.DataFrame({"team": ["KC"]})
    run_id = runs.save_run(model, scaler, df, {}, runs_dir=runs_dir)

    # The complete run wins even though the incomplete one sorts later
    assert runs.find_latest_run_id(runs_dir) == run_id


def test_load_run_without_any_runs_raises(tmp_path):
    with pytest.raises(runs.RunNotFoundError):
        runs.load_run(runs_dir=str(tmp_path))


def test_save_run_json_and_has_predictions(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    run_id = runs.save_run(model, scaler, pd.DataFrame({"a": [1]}), {}, runs_dir=runs_dir)

    assert not runs.has_predictions(run_id, runs_dir=runs_dir)

    predictions = [Prediction("KC", "SF", 3.5)]
    runs.save_run_json(run_id, "predictions.json", predictions, runs_dir=runs_dir)

    assert runs.has_predictions(run_id, runs_dir=runs_dir)


def test_manifest_written(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    run_id = runs.save_run(model, scaler, pd.DataFrame({"a": [1]}), {}, runs_dir=runs_dir)

    assert os.path.isfile(os.path.join(runs_dir, run_id, MANIFEST_FILENAME))
