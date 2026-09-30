"""
Handles getting the data for upcoming matchups.
"""

import json
from datetime import datetime, timezone
import urllib.request
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from nfl_analytics.utils import normalize_team_abbr

BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"


@dataclass
class Matchup:
    home_team: str
    away_team: str
    # nflverse season (start year) and week, and ESPN's kickoff time (UTC ISO)
    season: Optional[int] = None
    week: Optional[int] = None
    kickoff: Optional[str] = None


class SeasonType(Enum):
    PRESEASON = 1
    REGULAR = 2
    POSTSEASON = 3
    OFFSEASON = 4


@dataclass
class SeasonPosition:
    type: SeasonType
    week: str


# ESPN numbers postseason weeks 1-5 (week 4 is the Pro Bowl, which has no real
# teams); nflverse continues the regular-season numbering.
ESPN_POSTSEASON_TO_NFLVERSE_WEEK = {1: 19, 2: 20, 3: 21, 5: 22}


# The Pro Bowl is ESPN postseason week 4 and is skipped (see
# ESPN_POSTSEASON_TO_NFLVERSE_WEEK). If it ever shares a week with real games,
# its AFC/NFC teams fail matchup validation and the pipeline stops loudly.


def load_matchups(filepath: str) -> List[Matchup]:
    with open(filepath, "r") as file:
        matchups_data = json.load(file)

    matchups_list = []
    for matchup_data in matchups_data:
        matchup = Matchup(
            home_team=matchup_data["home_team"],
            away_team=matchup_data["away_team"],
            season=matchup_data.get("season"),
            week=matchup_data.get("week"),
            kickoff=matchup_data.get("kickoff"),
        )
        matchups_list.append(matchup)

    return matchups_list


def get_upcoming_matchups(now: Optional[datetime] = None) -> List[Matchup]:
    """Matchups for the next week that still has games to play.

    ESPN's calendar weeks run Wednesday to Wednesday, so on a Tuesday the
    "current" week is the one that just finished. Picking by kickoff time
    instead guarantees every returned game is still in the future.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    calendar_data = _get_calendar_data()

    season_url = calendar_data.get("season", {}).get("$ref")
    if season_url is None:
        raise ValueError("Season url not found.")
    season = _get_season_data(season_url)["year"]

    for calendar_week in _current_and_next_week(calendar_data, now):
        # only regular season and postseason have games worth predicting.
        # Week 1 is predictable too: running averages blend in the prior
        # season, and for week 1 that blend is (approximately) the prior-season mean.
        if calendar_week.type not in [SeasonType.REGULAR, SeasonType.POSTSEASON]:
            continue

        week = to_nflverse_week(calendar_week.type, int(calendar_week.week))
        if week is None:
            print("Skipping the Pro Bowl week.")
            continue

        matchups = [
            matchup
            for matchup in _get_week_matchups(season, calendar_week, week)
            if matchup.kickoff is not None and _parse_date(matchup.kickoff) > now
        ]

        if matchups:
            return matchups

    return []


def to_nflverse_week(season_type: SeasonType, espn_week: int) -> Optional[int]:
    if season_type == SeasonType.REGULAR:
        return espn_week
    return ESPN_POSTSEASON_TO_NFLVERSE_WEEK.get(espn_week)


def _parse_date(value: str) -> datetime:
    # ESPN dates look like 2026-10-02T00:15Z
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _current_and_next_week(
    calendar_data: Dict[str, Any], now: datetime
) -> List[SeasonPosition]:
    """The calendar week containing `now` and the one after it."""
    weeks = [
        (SeasonPosition(SeasonType(int(section["value"])), entry["value"]), entry)
        for section in calendar_data["sections"]
        for entry in section.get("entries", [])
    ]

    for i, (_, entry) in enumerate(weeks):
        if _parse_date(entry["startDate"]) <= now <= _parse_date(entry["endDate"]):
            return [position for position, _ in weeks[i : i + 2]]

    return []


def _get_week_matchups(
    season: int, calendar_week: SeasonPosition, week: int
) -> List[Matchup]:
    events_url = (
        f"{BASE_URL}seasons/{season}/types/{calendar_week.type.value}"
        f"/weeks/{calendar_week.week}/events"
    )
    event_data = _get_event_data(events_url)

    matchups = []

    for item in event_data.get("items", []):
        with urllib.request.urlopen(item["$ref"]) as response:
            event = json.load(response)

        competitions = event.get("competitions", [])
        competitionCount = len(competitions)

        if competitionCount != 1:
            raise ValueError(
                f"Get upcoming matchup failed. Expected 1 competition, got {competitionCount}."
            )

        competitors = competitions[0].get("competitors", [])
        home_team = None
        away_team = None

        for competitor in competitors:
            home_away = competitor.get("homeAway")

            if home_away == "home":
                home_team = _get_team_abbreviation(competitor["team"]["$ref"])
            elif home_away == "away":
                away_team = _get_team_abbreviation(competitor["team"]["$ref"])

        if home_team is None or away_team is None:
            raise ValueError(
                "Get upcoming matchup failed. Home or away team not found."
            )

        matchups.append(
            Matchup(home_team, away_team, season, week, event.get("date"))
        )

    return matchups


def _get_team_abbreviation(team_url: str) -> str:
    with urllib.request.urlopen(team_url) as response:
        team = json.load(response)

    return normalize_team_abbr(team["abbreviation"])


def _get_event_data(events_url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(events_url) as response:
        events = json.load(response)

    return events


def _get_calendar_data() -> Dict[str, Any]:
    blacklist_url = BASE_URL + "calendar/blacklist"

    with urllib.request.urlopen(blacklist_url) as response:
        calendar = json.load(response)

    return calendar


def _get_season_data(season_url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(season_url) as response:
        season = json.load(response)

    return season


if __name__ == "__main__":
    matchups = get_upcoming_matchups()
    print(matchups)
