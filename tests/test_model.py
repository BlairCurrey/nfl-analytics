import numpy as np
import pandas as pd
import pytest

from nfl_analytics.config import FEATURES
from nfl_analytics.model import make_matchup, train_model

AVG_COLS = [
    "rushing_avg",
    "passing_avg",
    "yards_gained_avg",
    "sack_yards_avg",
    "passing_yards_defense_avg",
    "rushing_yards_defense_avg",
    "yards_gained_defense_avg",
    "sack_yards_defense_avg",
    "score_differential_post_avg",
    "points_scored_avg",
    "points_allowed_avg",
    "mean_epa_avg",
]


def make_running_avg_df(rows):
    """rows: list of (team, year, week, fill_value)"""
    records = []
    for team, year, week, value in rows:
        record = {"team": team, "year": year, "week": week}
        record.update({col: value for col in AVG_COLS})
        records.append(record)
    return pd.DataFrame(records)


def test_uses_each_teams_own_latest_week():
    # KC last played week 22 (Super Bowl), SF only through week 18.
    # Historically this crashed: the global max week (22) had no SF row.
    df = make_running_avg_df(
        [
            ("KC", 2025, 18, 1.0),
            ("KC", 2025, 22, 2.0),
            ("SF", 2025, 17, 3.0),
            ("SF", 2025, 18, 4.0),
        ]
    )

    matchup = make_matchup(df, "SF", "KC")

    assert len(matchup) == 1
    assert matchup["home_rushing_avg"].iloc[0] == 4.0  # SF week 18
    assert matchup["away_rushing_avg"].iloc[0] == 2.0  # KC week 22


def test_defaults_to_latest_year():
    df = make_running_avg_df(
        [
            ("KC", 2024, 18, 1.0),
            ("SF", 2024, 18, 1.0),
            ("KC", 2025, 5, 2.0),
            ("SF", 2025, 5, 3.0),
        ]
    )

    matchup = make_matchup(df, "KC", "SF")

    assert matchup["home_rushing_avg"].iloc[0] == 2.0
    assert matchup["away_rushing_avg"].iloc[0] == 3.0


def test_explicit_week_and_year():
    df = make_running_avg_df(
        [
            ("KC", 2024, 5, 1.0),
            ("KC", 2024, 6, 2.0),
            ("SF", 2024, 5, 3.0),
            ("SF", 2024, 6, 4.0),
        ]
    )

    matchup = make_matchup(df, "KC", "SF", week=5, year=2024)

    assert matchup["home_rushing_avg"].iloc[0] == 1.0
    assert matchup["away_rushing_avg"].iloc[0] == 3.0


def test_missing_team_raises_clear_error():
    df = make_running_avg_df([("KC", 2025, 18, 1.0)])

    with pytest.raises(ValueError, match="No stats found for SF in the 2025 season"):
        make_matchup(df, "KC", "SF")


def test_train_model_uses_complete_rows_including_week_1():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURES})
    df["home_spread"] = rng.normal(2, 10, n)
    df["week"] = [1] * 10 + [2] * 30

    # a week-1 row without features (team with no prior season) must be dropped,
    # not imputed; complete week-1 rows train fine
    df.loc[0, FEATURES] = np.nan

    model, scaler, metrics = train_model(df)

    assert model.coef_.shape == (len(FEATURES),)
    assert "mean_absolute_error" in metrics
    assert not np.isnan(metrics["mean_absolute_error"])


def test_missing_week_raises_clear_error():
    df = make_running_avg_df([("KC", 2025, 18, 1.0), ("SF", 2025, 18, 1.0)])

    with pytest.raises(ValueError, match="No stats found for KC in week 3 of 2025"):
        make_matchup(df, "KC", "SF", week=3)
