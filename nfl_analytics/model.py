import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump

from nfl_analytics.config import FEATURES, ASSET_DIR


def train_model(df_training: pd.DataFrame) -> Tuple[LinearRegression, StandardScaler]:
    # Drop week 1 because is all NaN
    df_train = df_training[df_training["week"] > 1]

    # Dont use unnecessary columns like 'game_id', 'week', 'year', 'team', 'home_team', 'away_team'
    # Keep only relevant columns for prediction
    target = "home_spread"
    select_columns = FEATURES + [target]

    df_train = df_train[select_columns]

    # TODO: why are there missing values?
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns)

    X = df_imputed.drop(target, axis=1)
    y = df_imputed[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Note: scaler is transformed by fit_transform. Must re-use the same scaler for prediction.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    return model, scaler


def save_model_and_scaler(
    model: LinearRegression, scaler: StandardScaler, timestamp: str
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(script_dir, ASSET_DIR)
    os.makedirs(asset_dir, exist_ok=True)

    model_filename = f"trained_model-{timestamp}.joblib"
    scaler_filename = f"trained_scaler-{timestamp}.joblib"

    dump(model, os.path.join(asset_dir, model_filename))
    dump(scaler, os.path.join(asset_dir, scaler_filename))
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")


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
    week: int = None,
    year: int = None,
) -> pd.DataFrame:
    """Merge given team/week/years stats into a single row.
    To be used for predicting spreads for future games."""

    df = df_running_avg.copy()

    if year is None:
        year = df["year"].max()

    if week is None:
        last_week = df[df["year"] == year]["week"].max()
        week = last_week

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

    # Select data for the specified week, home team, and away team in the specified year
    home_data = (
        df[(df["year"] == year) & (df["week"] == week) & (df["team"] == home_team)][
            cols
        ]
        .add_prefix("home_")
        .reset_index(drop=True)
    )
    away_data = (
        df[(df["year"] == year) & (df["week"] == week) & (df["team"] == away_team)][
            cols
        ]
        .add_prefix("away_")
        .reset_index(drop=True)
    )

    return pd.concat([home_data, away_data], axis=1)


def get_matchup_input(scaler: StandardScaler, matchup: pd.DataFrame) -> pd.DataFrame:
    reshaped_matchup = matchup[FEATURES].values.reshape(1, -1)
    return scaler.transform(reshaped_matchup)


if __name__ == "__main__":
    from nfl_analytics.dataframes import (
        build_running_avg_dataframe,
        build_training_dataframe,
    )

    df_running_avg = build_running_avg_dataframe()
    df_training = build_training_dataframe()
    model, scaler = train_model(df_training)
    print(make_matchup(df_running_avg, "KC", "SF").tail())
    # first team is home but this is superbowl so neither is technically home
    # week 22 (? its the superbowl) 2023 (2023 SEASON, year is 2024)
    kc_sf = predict(model, scaler, df_running_avg, "KC", "SF")
    print(f"Prediction: {kc_sf}")
    sf_kc = predict(model, scaler, df_running_avg, "SF", "KC")
    print(f"Prediction: {sf_kc}")
