"""
Handles getting the data for upcoming matchups.
"""

import json
import os
import time
from datetime import datetime
import urllib.request
from typing import Any, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum
from nfl_analytics.utils import ASSET_DIR as ASSET_DIR_
from nfl_analytics.config import MATCHUPS_FILENAME

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, ASSET_DIR_)
BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"


@dataclass
class Matchup:
    home_team: str
    away_team: str


class SeasonType(Enum):
    PRESEASON = 1
    REGULAR = 2
    POSTSEASON = 3
    OFFSEASON = 4


@dataclass
class SeasonPosition:
    type: SeasonType
    week: str


# TODO: how to handle when its probowl?
# it will be an entry in the calendar among the playoffs in postseason.
# I guess it will also be in the events and we might have trouble forming the machtup (not in the list of team codes)
# Not sure I simply want to skip the week in case its not its OWN week sometime.


def save_upcoming_matchups(matchups: List[Matchup]) -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)

    filepath = os.path.join(ASSET_DIR, f"{MATCHUPS_FILENAME}-{int(time.time())}.json")
    with open(filepath, "w") as json_file:
        matchups_dict = [asdict(p) for p in matchups]
        json.dump(matchups_dict, json_file)

    print(f"Upcoming matchups saved to {filepath}")


def load_matchups(filepath: str) -> List[Matchup]:
    with open(filepath, "r") as file:
        matchups_data = json.load(file)

    matchups_list = []
    for matchup_data in matchups_data:
        matchup = Matchup(
            home_team=matchup_data["home_team"], away_team=matchup_data["away_team"]
        )
        matchups_list.append(matchup)

    return matchups_list


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

            # TODO: normalize team abbreviations. should match what Im using
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


def _get_season_position(calendar_data: Dict[str, Any]) -> SeasonPosition | None:
    current_date = datetime.utcnow()

    for section in calendar_data["sections"]:
        section_start_date = datetime.strptime(section["startDate"], "%Y-%m-%dT%H:%MZ")
        section_end_date = datetime.strptime(section["endDate"], "%Y-%m-%dT%H:%MZ")

        if section_start_date <= current_date <= section_end_date:
            for entry in section["entries"]:
                entry_start_date = datetime.strptime(
                    entry["startDate"], "%Y-%m-%dT%H:%MZ"
                )
                entry_end_date = datetime.strptime(entry["endDate"], "%Y-%m-%dT%H:%MZ")

                if entry_start_date <= current_date <= entry_end_date:
                    season_type = SeasonType(int(section["value"]))
                    week = entry["value"]
                    return SeasonPosition(type=season_type, week=week)

    raise ValueError("Could not find current season position")


def _get_upcoming_event_urls() -> List[str]:
    # Get calendar to find what the current season is
    calendar_data = _get_calendar_data()

    season_position = _get_season_position(calendar_data)

    # only get events for regular season and postseason, but not regular season week 1
    if season_position not in [SeasonType.REGULAR, SeasonType.POSTSEASON]:
        return []
    if season_position is SeasonType.REGULAR and season_position.week == "1":
        return []

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
        print("Events url not found from season data.")
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
