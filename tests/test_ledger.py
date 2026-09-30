import pandas as pd

from nfl_analytics import ledger
from nfl_analytics.model import Prediction
from nfl_analytics.schedule import Matchup


def week_4():
    matchups = [
        Matchup("KC", "SF", 2026, 4, "2026-10-04T20:25Z"),
        Matchup("DAL", "PHI", 2026, 4, "2026-10-05T00:20Z"),
    ]
    predictions = [
        Prediction("KC", "SF", 3.456, 2026, 4, vegas_line=1.5),
        Prediction("DAL", "PHI", -2.0, 2026, 4, vegas_line=-3.0),
    ]
    return matchups, predictions


def results(rows):
    """rows: (season, week, home, away, vegas_line, home_score, away_score)"""
    raw = pd.DataFrame(
        [
            {"year": s, "week": w, "home_team": h, "away_team": a,
             "game_id": f"{s}_{w:02d}_{a}_{h}", "spread_line": line,
             "home_score": hs, "away_score": as_}
            for s, w, h, a, line, hs, as_ in rows
        ]
    )
    # several plays per game in real data
    return ledger.game_results(pd.concat([raw, raw]))


def test_missing_ledger_file_starts_empty(tmp_path):
    df = ledger.load_ledger(str(tmp_path / "nope.csv"))
    assert df.empty
    assert list(df.columns) == ledger.COLUMNS


def test_add_predictions_appends_ungraded_rows():
    matchups, predictions = week_4()

    df = ledger.add_predictions(
        ledger.load_ledger(""), matchups, predictions, "run1"
    )

    assert len(df) == 2
    assert df.loc[0, "predicted_spread"] == 3.46
    assert df["game_id"].isna().all()


def test_add_predictions_keeps_the_originally_published_prediction():
    matchups, predictions = week_4()
    df = ledger.add_predictions(ledger.load_ledger(""), matchups, predictions, "run1")

    later = [Prediction("KC", "SF", 10.0), Prediction("DAL", "PHI", 10.0)]
    df = ledger.add_predictions(df, matchups, later, "run2")

    assert len(df) == 2
    assert list(df["run_id"]) == ["run1", "run1"]


def test_grade_fills_played_games_only():
    matchups, predictions = week_4()
    df = ledger.add_predictions(ledger.load_ledger(""), matchups, predictions, "run1")

    graded = ledger.grade(df, results([(2026, 4, "KC", "SF", 2.5, 27, 20)]))

    kc = graded[graded["home_team"] == "KC"].iloc[0]
    assert kc["game_id"] == "2026_04_SF_KC"
    assert kc["actual_margin"] == 7
    assert kc["vegas_line"] == 2.5
    assert pd.isna(graded[graded["home_team"] == "DAL"].iloc[0]["game_id"])


def test_ledger_roundtrip_then_grade(tmp_path):
    path = str(tmp_path / "ledger.csv")
    matchups, predictions = week_4()
    df = ledger.add_predictions(ledger.load_ledger(""), matchups, predictions, "run1")
    ledger.save_ledger(df, path)

    graded = ledger.grade(
        ledger.load_ledger(path),
        results([(2026, 4, "KC", "SF", 2.5, 27, 20), (2026, 4, "DAL", "PHI", -3.0, 10, 24)]),
    )

    assert graded["game_id"].notna().all()


def test_summarize_scores_model_and_vegas():
    matchups, predictions = week_4()
    df = ledger.add_predictions(ledger.load_ledger(""), matchups, predictions, "run1")
    df = ledger.grade(
        df,
        results([(2026, 4, "KC", "SF", 2.5, 27, 20), (2026, 4, "DAL", "PHI", -3.0, 10, 24)]),
    )

    summary = ledger.summarize(df)

    # KC: predicted 3.46, actual 7, line 2.5 -> error 3.54, vegas 4.5, pick home, covered: win
    # DAL: predicted -2.0, actual -14, line -3 -> error 12.0, vegas 11.0, pick home, lost: loss
    # line moves: KC opened 1.5, closed 2.5 -> moved toward the model's 3.46;
    # DAL opened -3.0, closed -3.0 -> didn't move, so it doesn't count
    assert "| 2026 season | 2 | 7.77 | 7.75 | 1-1 | 1 of 1 |" in summary


def test_add_predictions_records_line_at_publish():
    matchups, predictions = week_4()

    df = ledger.add_predictions(ledger.load_ledger(""), matchups, predictions, "run1")

    assert list(df["line_at_publish"]) == [1.5, -3.0]


def test_ledger_written_before_line_at_publish_still_loads(tmp_path):
    path = tmp_path / "old.csv"
    old_columns = [c for c in ledger.COLUMNS if c != "line_at_publish"]
    pd.DataFrame([[2026, 3, "KC", "SF", "2026-09-27T17:00Z", 3.0, "r", "", "", ""]],
                 columns=old_columns).to_csv(path, index=False)

    df = ledger.load_ledger(str(path))

    assert list(df.columns) == ledger.COLUMNS
    assert df["line_at_publish"].isna().all()


def test_summarize_without_graded_rows():
    assert ledger.summarize(ledger.load_ledger("")) == "No graded predictions yet."
