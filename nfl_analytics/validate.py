"""
Checks that stop the weekly pipeline before anything is published.

Each check returns a list of problems (empty means OK) so a failure reports
everything wrong at once. These guard against the ways a correct recipe still
produces bad output: incomplete or stale data, a changed data schema, and a
broken matchup fetch.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from nfl_analytics.config import START_YEAR, TEAMS
from nfl_analytics.model import Prediction
from nfl_analytics.schedule import Matchup

# Columns the dataframe builders, evaluation, and grading read from the raw
# play-by-play data
REQUIRED_COLUMNS = [
    "game_id",
    "game_date",
    "year",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "score_differential_post",
    "passing_yards",
    "rushing_yards",
    "yards_gained",
    "sack",
    "epa",
    "spread_line",
]

# Fewest games in any completed season since 1999 (31 teams, 16 games) plus
# playoffs is 259; well below that means a truncated or partial file.
MIN_GAMES_PER_COMPLETED_SEASON = 240

MAX_GAMES_PER_WEEK = 16
MAX_ABS_SPREAD = 30.0

# Replaying 2004-2025 (each season predicted by a model trained on earlier
# seasons), the weekly average distance from the Vegas line has median 2.4,
# 99th percentile 4.9, and max 5.7 (final regular-season weeks, when starters
# rest). Far beyond that means something broke, e.g. a flipped sign puts it
# around 11. Single games can legitimately be far off (injuries the model
# can't see), so only the weekly average is checked.
MAX_MEAN_DISTANCE_FROM_MARKET = 7.0
MIN_LINES_FOR_MARKET_CHECK = 4


class ValidationError(Exception):
    def __init__(self, what: str, problems: List[str]):
        super().__init__(f"{what} failed validation:\n  " + "\n  ".join(problems))
        self.problems = problems


def raise_if_problems(what: str, problems: List[str]) -> None:
    if problems:
        raise ValidationError(what, problems)


def validate_raw_data(
    df_raw: pd.DataFrame, season: int, upcoming_week: Optional[int]
) -> List[str]:
    """`season` and `upcoming_week` are the nflverse season and week about to
    be predicted."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing_columns:
        # nothing else can be checked reliably
        return [f"missing columns: {missing_columns}"]

    problems = []
    games_per_season = df_raw.groupby("year")["game_id"].nunique()

    for year in range(START_YEAR, season):
        n_games = games_per_season.get(year, 0)
        if n_games < MIN_GAMES_PER_COMPLETED_SEASON:
            problems.append(
                f"season {year} has {n_games} games "
                f"(expected at least {MIN_GAMES_PER_COMPLETED_SEASON})"
            )

    current = df_raw[df_raw["year"] == season]

    # Before week 1 there are no games yet; after that, last week's games must
    # be in the data or predictions would silently use stale averages.
    if upcoming_week is not None and upcoming_week > 1:
        latest_week = int(current["week"].max()) if not current.empty else 0
        if latest_week < upcoming_week - 1:
            problems.append(
                f"season {season} data only reaches week {latest_week}, but "
                f"week {upcoming_week} is next: play-by-play data is stale"
            )

    unknown_teams = sorted(
        set(current["home_team"]).union(current["away_team"]) - set(TEAMS)
    )
    if unknown_teams:
        problems.append(
            f"unknown team abbreviations in season {season}: {unknown_teams}. "
            "Update TEAMS / TEAM_ABBR_MAP in config.py."
        )

    return problems


def validate_predictions(
    matchups: List[Matchup],
    predictions: List[Prediction],
    now: Optional[datetime] = None,
) -> List[str]:
    if now is None:
        now = datetime.now(timezone.utc)

    problems = []

    if len(predictions) != len(matchups):
        problems.append(
            f"{len(predictions)} predictions for {len(matchups)} matchups"
        )

    if len(matchups) > MAX_GAMES_PER_WEEK:
        problems.append(
            f"{len(matchups)} games in one week (max {MAX_GAMES_PER_WEEK})"
        )

    teams = [t for m in matchups for t in (m.home_team, m.away_team)]
    duplicates = sorted({t for t in teams if teams.count(t) > 1})
    if duplicates:
        problems.append(f"teams playing twice in one week: {duplicates}")

    if len({(m.season, m.week) for m in matchups}) > 1:
        problems.append("matchups span more than one season/week")

    for m in matchups:
        label = f"{m.home_team} vs {m.away_team}"

        if m.season is None or m.week is None:
            problems.append(f"{label}: missing season/week")

        if m.kickoff is None:
            problems.append(f"{label}: missing kickoff time")
        else:
            kickoff = datetime.strptime(m.kickoff, "%Y-%m-%dT%H:%MZ").replace(
                tzinfo=timezone.utc
            )
            if not now < kickoff <= now + timedelta(days=8):
                problems.append(
                    f"{label}: kickoff {m.kickoff} is not within the next 8 days"
                )

    for p in predictions:
        if not math.isfinite(p.spread) or abs(p.spread) > MAX_ABS_SPREAD:
            problems.append(
                f"{p.home_team} vs {p.away_team}: implausible spread {p.spread}"
            )

    distances = [
        abs(p.spread - p.vegas_line) for p in predictions if p.vegas_line is not None
    ]
    if len(distances) >= MIN_LINES_FOR_MARKET_CHECK:
        mean_distance = sum(distances) / len(distances)
        if mean_distance > MAX_MEAN_DISTANCE_FROM_MARKET:
            problems.append(
                f"predictions are {mean_distance:.1f} points from the Vegas line on "
                f"average (max {MAX_MEAN_DISTANCE_FROM_MARKET})"
            )

    return problems
