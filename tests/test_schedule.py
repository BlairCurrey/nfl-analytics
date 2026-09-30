from datetime import datetime, timezone

from nfl_analytics import schedule
from nfl_analytics.schedule import Matchup, SeasonType

CALENDAR = {
    "season": {"$ref": "season-url"},
    "sections": [
        {"value": "2", "entries": [
            {"value": "3", "startDate": "2026-09-23T07:00Z", "endDate": "2026-09-30T06:59Z"},
            {"value": "4", "startDate": "2026-09-30T07:00Z", "endDate": "2026-10-07T06:59Z"},
            {"value": "18", "startDate": "2027-01-06T07:00Z", "endDate": "2027-01-13T07:59Z"},
        ]},
        {"value": "3", "entries": [
            {"value": "1", "startDate": "2027-01-13T08:00Z", "endDate": "2027-01-20T07:59Z"},
            {"value": "4", "startDate": "2027-02-03T08:00Z", "endDate": "2027-02-10T07:59Z"},
            {"value": "5", "startDate": "2027-02-10T08:00Z", "endDate": "2027-02-16T07:59Z"},
        ]},
        {"value": "4", "entries": [
            {"value": "1", "startDate": "2027-02-16T08:00Z", "endDate": "2027-08-01T06:59Z"},
        ]},
    ],
}

# kickoffs per (season type, espn week)
EVENTS = {
    (2, "3"): ["2026-09-25T00:15Z", "2026-09-29T00:15Z"],
    (2, "4"): ["2026-10-02T00:15Z", "2026-10-04T17:00Z"],
    (2, "18"): ["2027-01-10T18:00Z"],
    (3, "1"): ["2027-01-16T21:30Z"],
    (3, "5"): ["2027-02-14T23:30Z"],
}


def fake_espn(monkeypatch):
    monkeypatch.setattr(schedule, "_get_calendar_data", lambda: CALENDAR)
    monkeypatch.setattr(schedule, "_get_season_data", lambda url: {"year": 2026})

    def week_matchups(season, calendar_week, week):
        kickoffs = EVENTS[(calendar_week.type.value, calendar_week.week)]
        return [Matchup("KC", "SF", season, week, k) for k in kickoffs]

    monkeypatch.setattr(schedule, "_get_week_matchups", week_matchups)


def upcoming(monkeypatch, now):
    fake_espn(monkeypatch)
    return schedule.get_upcoming_matchups(
        datetime.fromisoformat(now).replace(tzinfo=timezone.utc)
    )


def test_tuesday_after_monday_night_predicts_next_week(monkeypatch):
    # ESPN still calls this week 3; its games are all played
    matchups = upcoming(monkeypatch, "2026-09-29T16:00")

    assert [m.week for m in matchups] == [4, 4]


def test_mid_week_only_returns_unplayed_games(monkeypatch):
    matchups = upcoming(monkeypatch, "2026-10-03T12:00")

    assert [m.kickoff for m in matchups] == ["2026-10-04T17:00Z"]


def test_regular_season_rolls_into_postseason_week_numbers(monkeypatch):
    matchups = upcoming(monkeypatch, "2027-01-12T16:00")

    assert [m.week for m in matchups] == [19]


def test_pro_bowl_week_skips_to_super_bowl(monkeypatch):
    matchups = upcoming(monkeypatch, "2027-02-04T16:00")

    assert [m.week for m in matchups] == [22]


def test_offseason_has_no_matchups(monkeypatch):
    assert upcoming(monkeypatch, "2027-03-01T16:00") == []


def test_to_nflverse_week():
    assert schedule.to_nflverse_week(SeasonType.REGULAR, 7) == 7
    assert schedule.to_nflverse_week(SeasonType.POSTSEASON, 1) == 19
    assert schedule.to_nflverse_week(SeasonType.POSTSEASON, 4) is None
    assert schedule.to_nflverse_week(SeasonType.POSTSEASON, 5) == 22
