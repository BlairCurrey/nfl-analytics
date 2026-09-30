"""
The ledger: every published prediction, graded once its game is played.

This is the model's real scorecard. Each weekly run carries the previous
release's ledger forward, grades any rows whose games now have results, and
appends the new week's predictions (ungraded until next time). Rows are only
ever added or filled in, never rewritten, so the ledger reflects exactly what
was published before kickoff.

All spreads are home margins: positive means the home team wins by that many.
`line_at_publish` is the Vegas line when the prediction was published;
`vegas_line` is the closing line from the play-by-play data. If the line tends
to move toward the model's prediction between the two, the model knew
something the market later learned: the clearest sign of a real edge, and one
that shows up in far fewer games than a betting record does.
"""

import os
from typing import List

import numpy as np
import pandas as pd

from nfl_analytics.model import Prediction
from nfl_analytics.schedule import Matchup

KEY = ["season", "week", "home_team", "away_team"]
COLUMNS = KEY + [
    "kickoff",
    "predicted_spread",
    "line_at_publish",
    "run_id",
    "game_id",
    "vegas_line",
    "actual_margin",
]


def load_ledger(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame(columns=COLUMNS)

    # reindex so ledgers written before a column existed still load
    return pd.read_csv(path).reindex(columns=COLUMNS)


def save_ledger(ledger: pd.DataFrame, path: str) -> None:
    ledger.to_csv(path, index=False)
    print(f"Saved {path}")


def add_predictions(
    ledger: pd.DataFrame,
    matchups: List[Matchup],
    predictions: List[Prediction],
    run_id: str,
) -> pd.DataFrame:
    new_rows = pd.DataFrame(
        [
            {
                "season": m.season,
                "week": m.week,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "kickoff": m.kickoff,
                "predicted_spread": round(p.spread, 2),
                "line_at_publish": p.vegas_line,
                "run_id": run_id,
            }
            for m, p in zip(matchups, predictions)
        ],
        columns=COLUMNS,
    )

    if ledger.empty:
        return new_rows

    # A game already in the ledger was already published; keep that original
    # prediction rather than overwriting it with a later one.
    existing = ledger.set_index(KEY).index
    new_rows = new_rows[~new_rows.set_index(KEY).index.isin(existing)]

    return pd.concat([ledger, new_rows], ignore_index=True)


def game_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per played game: final margin, closing line, and date."""
    games = df_raw.drop_duplicates(subset="game_id")

    return pd.DataFrame(
        {
            "season": games["year"],
            "week": games["week"],
            "home_team": games["home_team"],
            "away_team": games["away_team"],
            "game_id": games["game_id"],
            "vegas_line": games["spread_line"],
            "actual_margin": games["home_score"] - games["away_score"],
            "game_date": games["game_date"],
        }
    )


def grade(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Fill in results for ungraded rows whose games have been played."""
    ledger = ledger.copy()
    ungraded = ledger["game_id"].isna()

    if not ungraded.any():
        return ledger

    matched = (
        ledger.loc[ungraded, KEY]
        .astype({"season": int, "week": int})
        .reset_index()
        .merge(results, on=KEY, how="inner")
        .set_index("index")
    )

    # an all-empty game_id column loads from CSV as float
    ledger["game_id"] = ledger["game_id"].astype(object)
    for col in ["game_id", "vegas_line", "actual_margin"]:
        ledger.loc[matched.index, col] = matched[col].to_numpy()

    print(f"Graded {len(matched)} of {int(ungraded.sum())} ungraded predictions")
    return ledger


def _score(graded: pd.DataFrame) -> dict[str, float]:
    predicted = graded["predicted_spread"].astype(float)
    line = graded["vegas_line"].astype(float)
    actual = graded["actual_margin"].astype(float)

    # Against the spread: the model "picks" the side of the line it leans to;
    # a win means the actual margin landed on that side. Pushes don't count.
    pick = np.sign(predicted - line)
    outcome = np.sign(actual - line)
    decided = (outcome != 0) & (pick != 0)

    # Line movement: of the games where the line moved after publishing (and
    # the model disagreed with the line at publish), how often it moved
    # toward the model
    opened = graded["line_at_publish"].astype(float)
    move = np.sign(line - opened)
    lean = np.sign(predicted - opened)
    moved = opened.notna() & (move != 0) & (lean != 0)

    return {
        "games": len(graded),
        "model_mae": float((predicted - actual).abs().mean()),
        "vegas_mae": float((line - actual).abs().mean()),
        "ats_wins": int((pick[decided] == outcome[decided]).sum()),
        "ats_losses": int((pick[decided] != outcome[decided]).sum()),
        "line_moves": int(moved.sum()),
        "line_moves_toward": int((move[moved] == lean[moved]).sum()),
    }


def summarize(ledger: pd.DataFrame) -> str:
    """Markdown scorecard for release notes."""
    graded = ledger.dropna(subset=["actual_margin", "vegas_line"])

    if graded.empty:
        return "No graded predictions yet."

    season = int(graded["season"].max())
    rows = [
        (f"{season} season", _score(graded[graded["season"] == season])),
        ("All time", _score(graded)),
    ]

    lines = [
        "| | games | model MAE | Vegas MAE | against the spread | line moved toward model |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, s in rows:
        toward = (
            f"{s['line_moves_toward']} of {s['line_moves']}" if s["line_moves"] else "—"
        )
        lines.append(
            f"| {label} | {s['games']} | {s['model_mae']:.2f} | "
            f"{s['vegas_mae']:.2f} | {s['ats_wins']}-{s['ats_losses']} | {toward} |"
        )

    return "\n".join(lines)
