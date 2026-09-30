"""
Backtest: what the current model would have predicted in past weeks.

Each week is replayed the way the weekly run works: the model is trained only
on games played before that week, then predicts that week's games from their
pre-game running averages. So every prediction uses only information that
existed at the time.

These predictions were not published before kickoff, so they're kept out of
the ledger (the record of real, published predictions) and saved separately
in the same format. Vegas lines at publish time don't exist historically, so
`line_at_publish` is empty; only closing lines are known.
"""

import pandas as pd

from nfl_analytics.config import FEATURES
from nfl_analytics.ledger import COLUMNS
from nfl_analytics.model import train_model


def backtest(
    df_training: pd.DataFrame,
    results: pd.DataFrame,
    since: int,
    before: tuple[int, int] | None = None,
    exclude: set[tuple[int, int]] = frozenset(),
) -> pd.DataFrame:
    """Predict every (season, week) from `since` up to, not including,
    `before`, skipping weeks in `exclude` (e.g. weeks that were published).

    `results` comes from ledger.game_results and supplies each game's final
    margin, closing line, and date."""
    games = df_training.drop_duplicates(subset="game_id").dropna(
        subset=FEATURES + ["home_spread"]
    )
    game_weeks = list(zip(games["year"], games["week"]))

    weeks = sorted(
        {
            (int(season), int(week))
            for season, week in game_weeks
            if season >= since
            and (before is None or (season, week) < before)
            and (season, week) not in exclude
        }
    )

    info = results.set_index("game_id")[["vegas_line", "actual_margin", "game_date"]]
    predicted = []

    for season, week in weeks:
        played_before = (games["year"] < season) | (
            (games["year"] == season) & (games["week"] < week)
        )
        model, scaler, _ = train_model(games[played_before], verbose=False)

        this_week = games[(games["year"] == season) & (games["week"] == week)]
        spreads = model.predict(scaler.transform(this_week[FEATURES]))

        for game, spread in zip(this_week.itertuples(), spreads):
            game_info = info.loc[game.game_id]
            predicted.append(
                {
                    "season": season,
                    "week": week,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    # only the date is known historically, not the kickoff time
                    "kickoff": game_info["game_date"],
                    "predicted_spread": round(float(spread), 2),
                    "run_id": "backtest",
                    "game_id": game.game_id,
                    "vegas_line": game_info["vegas_line"],
                    "actual_margin": game_info["actual_margin"],
                }
            )

    print(f"Backtested {len(weeks)} weeks, {len(predicted)} games")
    return pd.DataFrame(predicted, columns=COLUMNS)
