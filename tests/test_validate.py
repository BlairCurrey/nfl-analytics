from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_analytics.model import Prediction
from nfl_analytics.schedule import Matchup
from nfl_analytics.validate import (
    REQUIRED_COLUMNS,
    validate_predictions,
    validate_raw_data,
)

NOW = datetime(2026, 9, 29, 16, 0, tzinfo=timezone.utc)


def make_raw(games_per_season, current_weeks, current_season=2026):
    """One play per game. games_per_season covers 1999..current_season-1."""
    rows = []
    for year, n_games in games_per_season.items():
        for i in range(n_games):
            rows.append({"year": year, "week": 1, "game_id": f"{year}_{i}",
                         "home_team": "KC", "away_team": "SF"})
    for week in current_weeks:
        rows.append({"year": current_season, "week": week,
                     "game_id": f"{current_season}_{week}",
                     "home_team": "KC", "away_team": "SF"})
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df


def full_history(n=260):
    return {year: n for year in range(1999, 2026)}


def test_complete_data_passes():
    df = make_raw(full_history(), current_weeks=[1, 2, 3])
    assert validate_raw_data(df, 2026, upcoming_week=4) == []


def test_missing_columns_reported_alone():
    df = make_raw(full_history(), [1]).drop(columns=["spread_line"])
    assert validate_raw_data(df, 2026, 2) == ["missing columns: ['spread_line']"]


def test_short_or_missing_season_is_reported():
    history = full_history()
    history[2010] = 100
    del history[2011]

    problems = validate_raw_data(make_raw(history, [1, 2, 3]), 2026, 4)

    assert any("season 2010 has 100 games" in p for p in problems)
    assert any("season 2011 has 0 games" in p for p in problems)


def test_stale_current_season_is_reported():
    df = make_raw(full_history(), current_weeks=[1, 2])

    problems = validate_raw_data(df, 2026, upcoming_week=4)

    assert problems == [
        "season 2026 data only reaches week 2, but week 4 is next: "
        "play-by-play data is stale"
    ]


def test_week_1_needs_no_current_season_data():
    df = make_raw(full_history(), current_weeks=[])
    assert validate_raw_data(df, 2026, upcoming_week=1) == []


def test_unknown_team_is_reported():
    df = make_raw(full_history(), current_weeks=[1])
    df.loc[df["year"] == 2026, "home_team"] = "XYZ"

    problems = validate_raw_data(df, 2026, 2)

    assert any("unknown team abbreviations" in p and "XYZ" in p for p in problems)


def matchup(home, away, kickoff=NOW + timedelta(days=2), season=2026, week=4):
    return Matchup(home, away, season, week, kickoff.strftime("%Y-%m-%dT%H:%MZ"))


def test_valid_predictions_pass():
    matchups = [matchup("KC", "SF"), matchup("DAL", "PHI")]
    predictions = [Prediction("KC", "SF", 3.5), Prediction("DAL", "PHI", -2.0)]

    assert validate_predictions(matchups, predictions, NOW) == []


def test_team_playing_twice_is_reported():
    matchups = [matchup("KC", "SF"), matchup("KC", "PHI")]
    predictions = [Prediction("KC", "SF", 1.0), Prediction("KC", "PHI", 1.0)]

    assert "teams playing twice in one week: ['KC']" in validate_predictions(
        matchups, predictions, NOW
    )


def test_past_or_distant_kickoff_is_reported():
    matchups = [
        matchup("KC", "SF", kickoff=NOW - timedelta(days=1)),
        matchup("DAL", "PHI", kickoff=NOW + timedelta(days=20)),
    ]
    predictions = [Prediction("KC", "SF", 1.0), Prediction("DAL", "PHI", 1.0)]

    problems = validate_predictions(matchups, predictions, NOW)

    assert len([p for p in problems if "not within the next 8 days" in p]) == 2


def test_implausible_spread_is_reported():
    matchups = [matchup("KC", "SF")]

    problems = validate_predictions(matchups, [Prediction("KC", "SF", 45.0)], NOW)

    assert problems == ["KC vs SF: implausible spread 45.0"]


def test_mixed_weeks_are_reported():
    matchups = [matchup("KC", "SF", week=4), matchup("DAL", "PHI", week=5)]
    predictions = [Prediction("KC", "SF", 1.0), Prediction("DAL", "PHI", 1.0)]

    assert "matchups span more than one season/week" in validate_predictions(
        matchups, predictions, NOW
    )


def test_predictions_far_from_the_market_are_reported():
    matchups = [matchup(h, a) for h, a in [("KC", "SF"), ("DAL", "PHI"), ("BUF", "MIA"), ("GB", "CHI")]]
    # every prediction has the sign flipped relative to the line
    predictions = [Prediction(m.home_team, m.away_team, -7.0, vegas_line=7.0) for m in matchups]

    problems = validate_predictions(matchups, predictions, NOW)

    assert problems == ["predictions are 14.0 points from the Vegas line on average (max 7.0)"]


def test_market_check_needs_enough_lines():
    matchups = [matchup("KC", "SF"), matchup("DAL", "PHI")]
    predictions = [
        Prediction("KC", "SF", -7.0, vegas_line=7.0),
        Prediction("DAL", "PHI", 1.0),
    ]

    assert validate_predictions(matchups, predictions, NOW) == []
