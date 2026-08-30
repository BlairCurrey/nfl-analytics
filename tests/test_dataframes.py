import numpy as np
import pandas as pd
import pytest

from nfl_analytics.dataframes import build_running_avg_dataframe


def make_play(game_id, year, week, posteam, defteam, home, away,
              home_score, away_score, yards):
    """One offensive play with enough columns for the aggregation pipeline."""
    return {
        "game_id": game_id,
        "year": year,
        "week": week,
        "posteam": posteam,
        "defteam": defteam,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "passing_yards": yards,
        "rushing_yards": yards,
        "yards_gained": yards,
        "sack": 0,
        "epa": 0.1,
        "score_differential_post": (
            home_score - away_score if posteam == home else away_score - home_score
        ),
    }


def make_raw_df(games):
    """games: list of (game_id, year, week, home, away, home_score, away_score,
    home_yards, away_yards). Two plays per game, one per possession team."""
    plays = []
    for game_id, year, week, home, away, hs, as_, hy, ay in games:
        plays.append(make_play(game_id, year, week, home, away, home, away, hs, as_, hy))
        plays.append(make_play(game_id, year, week, away, home, home, away, hs, as_, ay))
    return pd.DataFrame(plays)


def test_averages_exclude_current_game():
    df_raw = make_raw_df(
        [
            ("2020_01_AB", 2020, 1, "AAA", "BBB", 20, 10, 300, 200),
            ("2020_02_AB", 2020, 2, "AAA", "BBB", 30, 0, 400, 100),
        ]
    )

    ra = build_running_avg_dataframe(df_raw, prior_weight=0)

    week1 = ra[(ra["team"] == "AAA") & (ra["week"] == 1)]
    week2 = ra[(ra["team"] == "AAA") & (ra["week"] == 2)]

    assert np.isnan(week1["passing_avg"].iloc[0])  # no prior games, no prior season
    assert week2["passing_avg"].iloc[0] == 300.0  # only week 1, not week 2 itself


def test_prior_season_blending():
    # 2020: AAA passes for 300 and 400 (season mean 350). 2021: passes for 100 in week 1.
    df_raw = make_raw_df(
        [
            ("2020_01_AB", 2020, 1, "AAA", "BBB", 20, 10, 300, 200),
            ("2020_02_AB", 2020, 2, "AAA", "BBB", 30, 0, 400, 200),
            ("2021_01_AB", 2021, 1, "AAA", "BBB", 10, 3, 100, 200),
            ("2021_02_AB", 2021, 2, "AAA", "BBB", 10, 3, 100, 200),
        ]
    )

    ra = build_running_avg_dataframe(df_raw, prior_weight=4)

    week1 = ra[(ra["team"] == "AAA") & (ra["year"] == 2021) & (ra["week"] == 1)]
    week2 = ra[(ra["team"] == "AAA") & (ra["year"] == 2021) & (ra["week"] == 2)]

    # week 1 with no games played: average IS the prior-season mean
    assert week1["passing_avg"].iloc[0] == pytest.approx(350.0)
    # week 2: (1 real game of 100 + 4 pseudo-games of 350) / 5
    assert week2["passing_avg"].iloc[0] == pytest.approx((100 + 4 * 350) / 5)

    # first season has no prior: unblended, so week 1 is NaN
    first = ra[(ra["team"] == "AAA") & (ra["year"] == 2020) & (ra["week"] == 1)]
    assert np.isnan(first["passing_avg"].iloc[0])


def test_home_spread_sign_convention():
    df_raw = make_raw_df([("2020_01_AB", 2020, 1, "AAA", "BBB", 20, 10, 300, 200)])

    ra = build_running_avg_dataframe(df_raw, prior_weight=0)

    home_row = ra[ra["team"] == "AAA"]
    away_row = ra[ra["team"] == "BBB"]

    # both rows carry the margin relative to the home team
    assert home_row["home_spread"].iloc[0] == 10
    assert away_row["home_spread"].iloc[0] == 10
