import json
import urllib.request
from enum import Enum
from typing import Any, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

BASE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"


class SeasonType(Enum):
    PRESEASON = "1"
    REGULAR_SEASON = "2"
    POST_SEASON = "3"
    OFF_SEASON = "4"


@dataclass
class SeasonInfo:
    year: int
    season_type: SeasonType
    week: Optional[str]


def get_calendar_data() -> Dict[str, Any]:
    blacklist_url = BASE_URL + "calendar/blacklist"

    with urllib.request.urlopen(blacklist_url) as response:
        calendar = json.load(response)

    return calendar


# TODO: how to handle when its probowl?
# it will be an entry in the calendar among the playoffs in postseason.
# I guess it will also be in the events and we might have trouble forming the machtup (not in the list of team codes)
# Not sure I simply want to skip the week in case its not its OWN week sometime.


def get_current_season_info(calendar: Dict[str, Any]) -> SeasonInfo:

    # Did (and probably should) match the preseason start date and postseason end date
    calendar_start_date, calendar_end_date = get_start_and_end_date(calendar)
    season_year = datetime.fromisoformat(calendar_start_date).year

    now = datetime.now(timezone.utc).timestamp()

    if is_outside_date_range(calendar_start_date, calendar_end_date, now):
        return SeasonInfo(season_year, SeasonType.OFF_SEASON, None)

    # find the current season type by its top level date range
    for season_type_section in calendar.get("sections", []):
        section_start_date, section_end_date = get_start_and_end_date(
            season_type_section
        )

        if is_outside_date_range(section_start_date, section_end_date, now):
            continue

        # find the current week for the found season type
        for week in season_type_section.get("entries", []):
            # entries should pretty much correspond to week, but may include events like probowl (in postseason)
            week_start_date, week_end_date = get_start_and_end_date(week)
            if is_outside_date_range(week_start_date, week_end_date, now):
                continue
            # TODO: Is this a normal python pattern? destructuring inside fn call to condense into 1 line
            # if is_outside_date_range(*(get_start_and_end_date(week)), now):
            #     continue
            # TODO: If not destructuring, maybe I should just call get_start_and_end_date in is_outside_date_range.

            return SeasonInfo(
                season_year, SeasonType(season_type_section["value"]), week["value"]
            )

    raise ValueError("No current season type and/or week found")


def is_outside_date_range(
    start_date_str: str,
    end_date_str: str,
    timestamp: float = datetime.now(timezone.utc).timestamp(),
) -> bool:
    """
    Expects date strings like: 2024-02-15T07:59Z.
    Compares current time if none provided.
    """
    start_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)

    return timestamp < start_date.timestamp() or timestamp > end_date.timestamp()


def get_start_and_end_date(json: Dict[str, Any]) -> Tuple[str, str]:
    start_date = json.get("startDate")
    end_date = json.get("endDate")

    if start_date is None or end_date is None:
        raise ValueError("Start date and end date not found.")

    return start_date, end_date


if __name__ == "__main__":
    # is_outside_date_range
    current_datetime = datetime.now(timezone.utc)
    future_datetime = current_datetime + timedelta(
        weeks=52 * 50
    )  # 50 years into the future
    future_timestamp = future_datetime.timestamp()

    assert is_outside_date_range(
        "2023-08-01T07:00Z", "2024-02-15T07:59Z", timestamp=future_timestamp
    )
    assert not is_outside_date_range("2023-08-01T07:00Z", "2074-02-15T07:59Z")

    calendar_data = get_calendar_data()
    current_season_info = get_current_season_info(calendar_data)
    print(current_season_info)
