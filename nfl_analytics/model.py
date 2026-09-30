from typing import Any, Tuple, Optional, Union
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import spmatrix
from numpy import ndarray

from nfl_analytics.config import FEATURES


@dataclass
class Prediction:
    home_team: str
    away_team: str
    spread: float
    # nflverse season (start year) and week of the game, when known
    season: Optional[int] = None
    week: Optional[int] = None


def train_model(
    df_training: pd.DataFrame,
) -> Tuple[LinearRegression, StandardScaler, dict[str, Any]]:
    """Fit on every usable game. Nothing is held out: the shipped model should
    learn from all the data. Measure accuracy with `nfl benchmark` / `nfl
    evaluate`, which hold out whole seasons."""
    target = "home_spread"

    # The training dataframe has two identical rows per game (one per team
    # perspective); use one row per game. Keep any game with complete features
    # and a target. Week 1 rows qualify when the team has a prior-season
    # blend; rows without full features (e.g. a team's first-ever season) are
    # dropped rather than imputed, since prediction can't impute either.
    df_train = df_training.drop_duplicates(subset="game_id").dropna(
        subset=FEATURES + [target]
    )

    # Note: the scaler is fit here. Must re-use the same scaler for prediction.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[FEATURES])

    model = LinearRegression()
    model.fit(X_scaled, df_train[target])

    details = {
        "n_games": len(df_train),
        "seasons": [int(df_train["year"].min()), int(df_train["year"].max())],
    }
    print(f"Trained on {details['n_games']} games from seasons {details['seasons']}")

    return model, scaler, details


def predict(
    model: LinearRegression,
    scaler: StandardScaler,
    df_running_avg: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> float:
    matchup = make_matchup(df_running_avg, home_team, away_team)
    matchup_input = get_matchup_input(scaler, matchup)

    return model.predict(matchup_input)[0]


def make_matchup(
    df_running_avg: pd.DataFrame,
    home_team: str,
    away_team: str,
    week: Optional[int] = None,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """Merge given team/week/years stats into a single row.
    To be used for predicting spreads for future games."""

    df = df_running_avg.copy()

    if year is None:
        year = df["year"].max()

    # df_running_avg include running averages prior to that week, and data about
    # that week itself: teams, final scores, etc.). Basically (and literally at
    # the time of writing) anything not suffixed with `_avg`. The data about the
    # week itself are necessary for training the model but dont make sense in
    # the context of predicting future games so they are not included here.
    cols = [
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

    # Each team's averages come from its own latest available week (unless a
    # specific week is given). Teams don't all have rows for the same weeks:
    # byes, and playoff rounds where only some teams play.
    home_data = (
        _team_week_row(df, home_team, year, week, cols)
        .add_prefix("home_")
        .reset_index(drop=True)
    )
    away_data = (
        _team_week_row(df, away_team, year, week, cols)
        .add_prefix("away_")
        .reset_index(drop=True)
    )

    return pd.concat([home_data, away_data], axis=1)


def _team_week_row(
    df: pd.DataFrame,
    team: str,
    year: int,
    week: Optional[int],
    cols: list[str],
) -> pd.DataFrame:
    team_rows = df[(df["year"] == year) & (df["team"] == team)]

    if team_rows.empty:
        raise ValueError(f"No stats found for {team} in the {year} season.")

    if week is None:
        week = team_rows["week"].max()

    row = team_rows[team_rows["week"] == week]

    if row.empty:
        raise ValueError(f"No stats found for {team} in week {week} of {year}.")

    print(f"Using {team} averages through {year} week {week}")
    return row[cols]


def get_matchup_input(
    scaler: StandardScaler, matchup: pd.DataFrame
) -> Union[ndarray, spmatrix]:
    # Keep feature names so the scaler sees the same columns it was fit with
    return scaler.transform(matchup[FEATURES])


if __name__ == "__main__":
    from nfl_analytics.dataframes import (
        build_running_avg_dataframe,
        build_training_dataframe,
    )

    df_running_avg = build_running_avg_dataframe()
    df_training = build_training_dataframe()
    model, scaler, _ = train_model(df_training)
    print(make_matchup(df_running_avg, "KC", "SF").tail())
    # first team is home but this is superbowl so neither is technically home
    # week 22 (? its the superbowl) 2023 (2023 SEASON, year is 2024)
    kc_sf = predict(model, scaler, df_running_avg, "KC", "SF")
    print(f"Prediction: {kc_sf}")
    sf_kc = predict(model, scaler, df_running_avg, "SF", "KC")
    print(f"Prediction: {sf_kc}")
