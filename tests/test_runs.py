import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from nfl_analytics import runs
from nfl_analytics.config import FEATURES, MANIFEST_FILENAME, MODEL_FILENAME
from nfl_analytics.model import Prediction


def make_fitted_model_and_scaler():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (30, len(FEATURES))), columns=FEATURES)
    y = rng.normal(0, 1, 30)

    scaler = StandardScaler().fit(X)
    model = LinearRegression().fit(scaler.transform(X), y)

    return model, scaler


def test_save_and_load_run_roundtrip(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    df = pd.DataFrame({"team": ["KC", "SF"], "rushing_avg": [120.0, 110.0]})
    details = {"training": {"n_games": 30}}

    run_id = runs.save_run(model, scaler, df, details, runs_dir=runs_dir)

    assert runs.find_latest_run_id(runs_dir) == run_id

    loaded_model, loaded_scaler, loaded_df, manifest = runs.load_run(
        runs_dir=runs_dir
    )

    assert manifest["run_id"] == run_id
    assert manifest["training"] == {"n_games": 30}
    assert list(loaded_df["team"]) == ["KC", "SF"]
    assert np.allclose(loaded_model.coef_, model.coef_)
    assert np.allclose(loaded_scaler.mean_, scaler.mean_)


def test_json_model_predicts_like_the_original(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    run_id = runs.save_run(model, scaler, pd.DataFrame({"a": [1]}), {}, runs_dir=runs_dir)

    loaded_model, loaded_scaler, _, _ = runs.load_run(run_id, runs_dir=runs_dir)

    X = pd.DataFrame(np.ones((2, len(FEATURES))), columns=FEATURES)
    expected = model.predict(scaler.transform(X))
    actual = loaded_model.predict(loaded_scaler.transform(X))
    assert np.allclose(actual, expected)


def test_old_pickle_format_run_raises_clear_error(tmp_path):
    runs_dir = str(tmp_path)
    model, scaler = make_fitted_model_and_scaler()
    run_id = runs.save_run(model, scaler, pd.DataFrame({"a": [1]}), {}, runs_dir=runs_dir)
    os.remove(os.path.join(runs_dir, run_id, MODEL_FILENAME))

    with pytest.raises(runs.RunNotFoundError, match="JSON model format"):
        runs.load_run(run_id, runs_dir=runs_dir)


def test_model_with_different_features_is_rejected():
    model, scaler = make_fitted_model_and_scaler()
    params = runs.model_to_dict(model, scaler)
    params["features"] = params["features"][:-1]

    with pytest.raises(runs.RunNotFoundError, match="FEATURES"):
        runs.model_from_dict(params)


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
