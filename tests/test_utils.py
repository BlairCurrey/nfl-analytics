import datetime

from nfl_analytics.config import START_YEAR
from nfl_analytics.utils import is_valid_year, normalize_team_abbr


def test_is_valid_year_bounds():
    current_year = datetime.datetime.now().year

    assert not is_valid_year(START_YEAR - 1)
    assert is_valid_year(START_YEAR)
    assert is_valid_year(current_year)
    assert not is_valid_year(current_year + 1)


def test_normalize_team_abbr():
    assert normalize_team_abbr("LAR") == "LA"
    assert normalize_team_abbr("WSH") == "WAS"
    assert normalize_team_abbr("NYG") == "NYG"
    assert normalize_team_abbr("kc") == "KC"
