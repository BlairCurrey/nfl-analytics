import pytest

from nfl_analytics.main import build_parser


@pytest.fixture
def parser():
    return build_parser()


def test_requires_a_command(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_download_accepts_years(parser):
    args = parser.parse_args(["download", "2020", "2021"])
    assert args.command == "download"
    assert args.years == [2020, 2021]


def test_download_years_optional(parser):
    args = parser.parse_args(["download"])
    assert args.years == []


def test_predict_takes_home_and_away(parser):
    args = parser.parse_args(["predict", "KC", "SF"])
    assert args.command == "predict"
    assert args.home_team == "KC"
    assert args.away_team == "SF"
    assert args.run is None


def test_predict_requires_both_teams(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "KC"])


def test_predict_upcoming_options(parser):
    args = parser.parse_args(
        ["predict-upcoming", "--matchups", "some/path.json", "--run", "20260101000000"]
    )
    assert args.command == "predict-upcoming"
    assert args.matchups == "some/path.json"
    assert args.run == "20260101000000"


def test_simple_commands(parser):
    for command in ["train", "update", "run-pipeline"]:
        assert parser.parse_args([command]).command == command
