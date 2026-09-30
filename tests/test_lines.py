import pandas as pd

from nfl_analytics.lines import lines_for_week


def test_lines_for_week_keys_by_home_and_away():
    df = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2025],
            "week": [4, 4, 5, 4],
            "home_team": ["CLE", "LAR", "BUF", "CLE"],
            "away_team": ["PIT", "SF", "MIA", "PIT"],
            "spread_line": [-2.5, 3.0, 6.5, 1.0],
        }
    )

    assert lines_for_week(df, 2026, 4) == {("CLE", "PIT"): -2.5, ("LA", "SF"): 3.0}


def test_games_without_a_line_yet_are_skipped():
    df = pd.DataFrame(
        {
            "season": [2026],
            "week": [6],
            "home_team": ["DEN"],
            "away_team": ["SEA"],
            "spread_line": [None],
        }
    )

    assert lines_for_week(df, 2026, 6) == {}
