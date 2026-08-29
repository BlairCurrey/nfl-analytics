from nfl_analytics.config import START_YEAR
from nfl_analytics.data import (
    default_years,
    get_year_from_filename,
    latest_season_year,
)


def test_get_year_from_filename():
    assert get_year_from_filename("play_by_play_2020.csv.gz") == 2020
    assert get_year_from_filename("play_by_play_1999.csv.gz") == 1999


def test_default_years_covers_start_through_latest_season():
    years = default_years()

    assert years.start == START_YEAR
    assert years.stop == latest_season_year() + 1
