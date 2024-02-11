"""
Handles getting the data for upcoming matchups.
"""

import json
import urllib.request
from typing import Any, Dict, List
from dataclasses import dataclass

BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"


@dataclass
class Matchup:
    home_team: str
    away_team: str


# TODO: how to handle when its probowl?
# it will be an entry in the calendar among the playoffs in postseason.
# I guess it will also be in the events and we might have trouble forming the machtup (not in the list of team codes)
# Not sure I simply want to skip the week in case its not its OWN week sometime.


def get_upcoming_matchups() -> List[Matchup]:
    upcoming_event_urls = _get_upcoming_event_urls()

    matchups = []

    for url in upcoming_event_urls:
        with urllib.request.urlopen(url) as response:
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

        matchups.append(Matchup(home_team, away_team))
    return matchups


def _get_upcoming_event_urls():
    # Get calendar to find what the current season is
    calendar_data = _get_calendar_data()

    # Get's the current season including the current week
    season_url = calendar_data.get("season", {}).get("$ref")
    if season_url is None:
        raise ValueError("Season url not found.")
    season_data = _get_season_data(season_url)

    # Get events for the current week. The top level "type" field is the current week/season type.
    events_url = (
        season_data.get("type", {}).get("week", {}).get("events", {}).get("$ref")
    )
    if events_url is None:
        raise ValueError("Events url not found.")
    event_data = _get_event_data(events_url)

    return [item.get("$ref") for item in event_data.get("items", [])]


def _get_team_abbreviation(team_url: str) -> str:
    with urllib.request.urlopen(team_url) as response:
        team = json.load(response)

    return team["abbreviation"]


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
