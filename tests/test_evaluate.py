import numpy as np
import pandas as pd
import pytest

from nfl_analytics.config import FEATURES
from nfl_analytics.evaluate import (
    evaluate_spread_model,
    extract_vegas_lines,
    format_report,
)


def make_training_df(n_games_per_year, years, seed=0):
    """Synthetic training data: home_spread loosely driven by one feature."""
    rng = np.random.default_rng(seed)
    records = []
    game_num = 0

    for year in years:
        for _ in range(n_games_per_year):
            game_num += 1
            game_id = f"{year}_{game_num:04d}"
            features = {col: rng.normal(0, 1) for col in FEATURES}
            spread = 2.0 + 3.0 * features["home_mean_epa_avg"] + rng.normal(0, 5)

            # two identical rows per game, mirroring build_training_dataframe
            for _ in range(2):
                records.append(
                    {
                        "game_id": game_id,
                        "week": 2 + game_num % 15,
                        "year": year,
                        "home_spread": spread,
                        **features,
                    }
                )

    return pd.DataFrame(records)


def make_vegas_df(df_training, noise=1.0, seed=1):
    rng = np.random.default_rng(seed)
    games = df_training.drop_duplicates(subset="game_id")
    return pd.DataFrame(
        {
            "game_id": games["game_id"],
            "spread_line": games["home_spread"] + rng.normal(0, noise, len(games)),
        }
    )


def test_evaluate_returns_expected_shape():
    df = make_training_df(60, [2020, 2021, 2022, 2023])
    vegas = make_vegas_df(df)

    results = evaluate_spread_model(df, vegas, test_since=2023)

    assert results["n_test_games"] == 60
    assert results["n_train_games"] == 180
    assert results["n_games_with_vegas_line"] == 60
    assert results["model"]["mae"] > 0
    # model uses a real feature, so it should beat the constant baseline
    assert results["model"]["mae"] < results["naive"]["mae"]
    # these vegas lines are near-perfect by construction
    assert results["vegas"]["mae"] < results["model"]["mae"]
    assert 0.0 <= results["ats_accuracy"] <= 1.0
    assert results["ats_ci95"] > 0

    # new diagnostics
    assert abs(results["model"]["bias"]) < 5
    assert results["model"]["mae_ci95"] > 0
    assert 0.5 < results["calibration"]["slope"] < 1.5
    assert 0.0 < results["gap_closed"] <= 1.5
    assert results["vs_vegas"]["mae_to_line"] > 0
    assert -1.0 <= results["vs_vegas"]["corr"] <= 1.0
    # vegas is near-perfect here, so the model should trail it significantly
    assert results["vs_vegas"]["delta_mae"] > 0
    assert results["vs_vegas"]["delta_mae_ci95"] > 0


def test_evaluate_without_vegas_lines():
    df = make_training_df(40, [2021, 2022, 2023])
    empty_vegas = pd.DataFrame({"game_id": [], "spread_line": []})

    results = evaluate_spread_model(df, empty_vegas, test_since=2023)

    assert results["n_games_with_vegas_line"] == 0
    assert "vegas" not in results
    assert "No Vegas lines" in format_report(results)


def test_evaluate_raises_when_split_is_empty():
    df = make_training_df(20, [2023])
    vegas = make_vegas_df(df)

    with pytest.raises(ValueError, match="Not enough data"):
        evaluate_spread_model(df, vegas, test_since=2023)


def test_extract_vegas_lines_dedupes_games():
    df_raw = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "spread_line": [3.0, 3.0, -7.0],
            "other_col": [1, 2, 3],
        }
    )

    vegas = extract_vegas_lines(df_raw)

    assert len(vegas) == 2
    assert list(vegas.columns) == ["game_id", "spread_line"]


def test_format_report_contains_key_numbers():
    df = make_training_df(60, [2021, 2022, 2023])
    vegas = make_vegas_df(df)

    report = format_report(evaluate_spread_model(df, vegas, test_since=2023))

    assert "naive (constant HFA)" in report
    assert "vegas closing line" in report
    assert "Against-the-spread" in report
    assert "calibration" in report
    assert "gap closed" in report
    assert "paired" in report
