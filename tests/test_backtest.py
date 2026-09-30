import numpy as np
import pandas as pd

from nfl_analytics import backtest as backtest_module
from nfl_analytics.backtest import backtest
from nfl_analytics.config import FEATURES
from nfl_analytics.ledger import COLUMNS


def make_games(weeks_by_season, games_per_week=4, seed=0):
    """Training rows (two per game, like build_training_dataframe) and the
    matching game results."""
    rng = np.random.default_rng(seed)
    rows, results = [], []
    for season, weeks in weeks_by_season.items():
        for week in weeks:
            for g in range(games_per_week):
                game_id = f"{season}_{week:02d}_A{g}_H{g}"
                features = {col: rng.normal() for col in FEATURES}
                margin = float(rng.integers(-14, 15))
                for _ in range(2):
                    rows.append({"game_id": game_id, "year": season, "week": week,
                                 "home_team": f"H{g}", "away_team": f"A{g}",
                                 "home_spread": margin, **features})
                results.append({"game_id": game_id, "vegas_line": 2.5,
                                "actual_margin": margin, "game_date": f"{season}-09-{week:02d}"})
    return pd.DataFrame(rows), pd.DataFrame(results)


def test_each_week_trains_only_on_earlier_games(monkeypatch):
    df, results = make_games({2020: [1, 2, 3], 2021: [1, 2, 3]})
    seen = []
    real_train = backtest_module.train_model

    def recording_train(df_train, verbose=True):
        seen.append(max(zip(df_train["year"], df_train["week"])))
        return real_train(df_train, verbose=False)

    monkeypatch.setattr(backtest_module, "train_model", recording_train)

    out = backtest(df, results, since=2021)

    predicted_weeks = sorted(set(zip(out["season"], out["week"])))
    assert predicted_weeks == [(2021, 1), (2021, 2), (2021, 3)]
    # the latest game each model saw was the week before the one it predicted
    assert seen == [(2020, 3), (2021, 1), (2021, 2)]


def test_before_and_exclude_limit_the_weeks():
    df, results = make_games({2020: [1, 2], 2021: [1, 2, 3, 4]})

    out = backtest(df, results, since=2021, before=(2021, 4), exclude={(2021, 2)})

    assert sorted(set(zip(out["season"], out["week"]))) == [(2021, 1), (2021, 3)]


def test_output_matches_the_ledger_format_and_is_graded():
    df, results = make_games({2020: [1, 2], 2021: [1]})

    out = backtest(df, results, since=2021)

    assert list(out.columns) == COLUMNS
    assert len(out) == 4
    assert (out["run_id"] == "backtest").all()
    assert out["line_at_publish"].isna().all()
    assert out["vegas_line"].eq(2.5).all()
    assert out["actual_margin"].notna().all()
    assert out["kickoff"].iloc[0] == "2021-09-01"
