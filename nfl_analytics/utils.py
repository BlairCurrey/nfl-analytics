import datetime

from nfl_analytics.config import (
    START_YEAR,
    TEAM_ABBR_MAP,
)


def is_valid_year(year: int) -> bool:
    current_year = datetime.datetime.now().year
    return START_YEAR <= year <= current_year


def normalize_team_abbr(team_abbr: str) -> str:
    team_abbr = team_abbr.upper()
    return TEAM_ABBR_MAP.get(team_abbr, team_abbr)
