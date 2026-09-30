"""
Current Vegas lines for upcoming games, from nflverse's schedule file.

This is the same source the play-by-play `spread_line` (the closing line)
comes from, but it's updated through the week, so reading it when predictions
are published captures the line at that moment. Comparing that with the
closing line shows whether the market moved toward the model afterwards.
"""

import pandas as pd

from nfl_analytics.utils import normalize_team_abbr

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def lines_for_week(
    df_games: pd.DataFrame, season: int, week: int
) -> dict[tuple[str, str], float]:
    """(home, away) -> spread line as a home margin (positive = home favored)."""
    games = df_games[
        (df_games["season"] == season) & (df_games["week"] == week)
    ].dropna(subset=["spread_line"])

    return {
        (normalize_team_abbr(g.home_team), normalize_team_abbr(g.away_team)): float(
            g.spread_line
        )
        for g in games.itertuples()
    }


def fetch_lines(season: int, week: int) -> dict[tuple[str, str], float]:
    df_games = pd.read_csv(
        GAMES_URL, usecols=["season", "week", "home_team", "away_team", "spread_line"]
    )
    return lines_for_week(df_games, season, week)
