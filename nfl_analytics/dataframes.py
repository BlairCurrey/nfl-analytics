"""
Builds the dataframes used for training and prediction.
Handles everything between getting the data and training/using the model.
"""

from typing import Optional

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from nfl_analytics.data import load_dataframe_from_raw


def build_training_dataframe(
    df_running_avg: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_running_avg is None:
        df_running_avg = build_running_avg_dataframe()

    # Create a new column 'is_home' to indicate whether the team is playing at home
    df_running_avg["is_home"] = df_running_avg.apply(
        lambda row: True if row["team"] == row["home_team"] else False, axis=1
    )

    # Group by game_id and is_home and aggregate using the first value
    squashed_df = (
        df_running_avg.groupby(["game_id", "is_home"])[
            [
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
        ]
        .first()
        .unstack()
    )

    squashed_df.columns = [
        f"{'home' if is_home else 'away'}_{col}" for col, is_home in squashed_df.columns
    ]
    squashed_df.reset_index(inplace=True)

    # Merge with the original DataFrame to get the rest of the columns
    return pd.merge(
        df_running_avg[
            [
                "game_id",
                "week",
                "year",
                "team",
                "home_team",
                "away_team",
                "home_spread",
            ]
        ],
        squashed_df,
        on="game_id",
    )


def build_running_avg_dataframe(df_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Builds a dataframe with weakly running averages for each team by year.
    Used to create prediction inputs and build the training dataset
    """
    if df_raw is None:
        df_raw = load_dataframe_from_raw()

    df_sacks = add_sack_yards(df_raw)
    # df_game is team games stats by team: week 1, DET, 250 pass, 120 run, etc.
    df_game_posteam = df_sacks.groupby(["game_id", "posteam"])
    df_game = aggregate_game_stats(df_sacks, df_game_posteam)
    df_game = adjust_game_dataframe(df_game, df_game_posteam)
    df_running_avg = df_game[
        [
            "game_id",
            "team",
            "week",
            "year",
            "home_team",
            "away_team",
            "score_differential_post",
        ]
    ].copy()

    # Set the home_spread
    # This will be our target variable. It's the spread relative to the home team. We want this because we need to predict a single spread value (which we can then invert for the away team's spread).
    df_running_avg["home_spread"] = df_game.apply(
        lambda row: (
            -row["score_differential_post"]
            if row["team"] != row["home_team"]
            else row["score_differential_post"]
        ),
        axis=1,
    )

    # Get the running average for each team by team and year
    # Uses lambda and shift to not include current row in running average
    # Expand is an expanding window function that gets everything from the first to current row
    df_running_avg[
        [
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
    ] = (
        df_game.groupby(["team", "year"])[
            [
                "rushing_yards",
                "passing_yards",
                "yards_gained",
                "sack_yards",
                "passing_yards_defense",
                "rushing_yards_defense",
                "yards_gained_defense",
                "sack_yards_defense",
                "score_differential_post",
                "points_scored",
                "points_allowed",
                "mean_epa",
            ]
        ]
        .apply(lambda x: x.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    return df_running_avg


def add_sack_yards(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # Sack yards would be necessary to get accurate TEAM passing stats.
    # Team passing yards are sum(passing_yards) - sum(sack_yards)
    # Player passing stats are simply sum(passing_yards).
    df["sack_yards"] = pd.NA

    # Set sack_yards to yards_gained for rows where sack is not equal to 0
    df.loc[df["sack"] != 0, "sack_yards"] = df["yards_gained"]

    return df


def aggregate_game_stats(
    df_sacks: pd.DataFrame, df_game_posteam: DataFrameGroupBy
) -> pd.DataFrame:
    # Group by game and team and combine offensive and defensive stats into single record

    # Separate offensive and defensive stats
    offensive_stats = (
        df_game_posteam[
            ["passing_yards", "rushing_yards", "yards_gained", "sack_yards"]
        ]
        .sum()
        .reset_index()
    )
    defensive_stats = (
        df_sacks.groupby(["game_id", "defteam"])[
            ["passing_yards", "rushing_yards", "yards_gained", "sack_yards"]
        ]
        .sum()
        .reset_index()
    )

    # Rename columns for defensive stats to distinguish them
    defensive_stats.rename(
        columns={
            "defteam": "team",
            "passing_yards": "passing_yards_defense",
            "rushing_yards": "rushing_yards_defense",
            "yards_gained": "yards_gained_defense",
            "sack_yards": "sack_yards_defense",
        },
        inplace=True,
    )

    return pd.merge(
        offensive_stats,
        defensive_stats,
        left_on=["game_id", "posteam"],
        right_on=["game_id", "team"],
    )


def adjust_game_dataframe(df_game: pd.DataFrame, df_game_posteam: DataFrameGroupBy):
    df = df_game.copy()

    # Add home_team, away_team, home_score, away_score
    df[["home_team", "away_team", "home_score", "away_score"]] = (
        df_game_posteam[["home_team", "away_team", "home_score", "away_score"]]
        .first()
        .reset_index(drop=True)
    )

    df["points_scored"] = df.apply(
        lambda row: (
            row["home_score"]
            if row["posteam"] == row["home_team"]
            else row["away_score"]
        ),
        axis=1,
    )
    df["points_allowed"] = df.apply(
        lambda row: (
            row["away_score"]
            if row["posteam"] == row["home_team"]
            else row["home_score"]
        ),
        axis=1,
    )

    df.drop(["posteam"], axis=1, inplace=True)

    # sets score differential to last value for each game and team
    df[["score_differential_post", "week", "year"]] = (
        df_game_posteam[["score_differential_post", "week", "year"]]
        .last()
        .reset_index(drop=True)
    )

    df["mean_epa"] = df_game_posteam["epa"].mean().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df_running_avg = build_running_avg_dataframe()
    print(df_running_avg.tail())
    df_train = build_training_dataframe()
    print(df_train.tail())
