"""
Do rest-day features help? (Targets the confirmed short-week weakness.)

Candidates, fixed before running (rest days come from nflverse schedules):

  A. rest_diff: home rest minus away rest, capped at +/-7 days
  B. rest_flags: short week (<= 5 days) and off a bye (>= 13 days), each team
  C. rest_diff + short_week: A plus a flag for games where either team is on
     a short week (mostly Thursday games, where both teams usually are)

Protocol:

  1. Walk-forward (each season predicted by a model trained on earlier
     seasons). Pick the best candidate by paired MAE change on 2004-2019.
  2. Judge only that candidate on 2020-2025, which played no part in picking.
     Ship it if the paired MAE change there is negative with a 95% CI below
     zero, the short-week gap shrinks, and the committed benchmark window
     (fit before 2023, test 2023-2025) doesn't get worse.

Run: uv run python -m nfl_analytics.scripts.experiment_rest
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from nfl_analytics.config import FEATURES
from nfl_analytics.data import load_dataframe_from_raw
from nfl_analytics.dataframes import (
    build_running_avg_dataframe,
    build_training_dataframe,
)
from nfl_analytics.evaluate import extract_vegas_lines
from nfl_analytics.lines import GAMES_URL

FIRST_TEST_SEASON = 2004
CONFIRM_SINCE = 2020
rng = np.random.default_rng(0)


def load() -> pd.DataFrame:
    raw = load_dataframe_from_raw()
    lines = extract_vegas_lines(raw)
    df = (
        build_training_dataframe(build_running_avg_dataframe(raw))
        .drop_duplicates(subset="game_id")
        .dropna(subset=FEATURES + ["home_spread"])
        .merge(lines, on="game_id")
        .dropna(subset=["spread_line"])
    )
    games = pd.read_csv(GAMES_URL).set_index("game_id")[["home_rest", "away_rest"]]
    df = df.join(games, on="game_id").dropna(subset=["home_rest", "away_rest"])

    df["rest_diff"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7)
    df["home_short"] = (df["home_rest"] <= 5).astype(float)
    df["away_short"] = (df["away_rest"] <= 5).astype(float)
    df["home_bye"] = (df["home_rest"] >= 13).astype(float)
    df["away_bye"] = (df["away_rest"] >= 13).astype(float)
    df["short_week"] = ((df[["home_rest", "away_rest"]].min(axis=1) <= 5) & (df["week"] <= 18)).astype(float)
    return df


CANDIDATES = {
    "base": FEATURES,
    "A. rest_diff": FEATURES + ["rest_diff"],
    "B. rest_flags": FEATURES + ["home_short", "away_short", "home_bye", "away_bye"],
    "C. rest_diff + short_week": FEATURES + ["rest_diff", "short_week"],
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    scaler = StandardScaler().fit(train[features])
    model = LinearRegression().fit(scaler.transform(train[features]), train["home_spread"])
    return model.predict(scaler.transform(test[features]))


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    seasons = []
    for season in range(FIRST_TEST_SEASON, int(df["year"].max()) + 1):
        train, test = df[df["year"] < season], df[df["year"] == season].copy()
        for name, features in CANDIDATES.items():
            test[name] = fit_predict(train, test, features)
        seasons.append(test)
    return pd.concat(seasons)


def paired(actual: np.ndarray, base: np.ndarray, cand: np.ndarray) -> tuple[float, float, float]:
    """MAE change (candidate - base) with a bootstrap 95% CI."""
    d = np.abs(cand - actual) - np.abs(base - actual)
    boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi


def report(name: str, d: pd.DataFrame, candidates: list[str]) -> None:
    y = d["home_spread"].to_numpy()
    short = d["short_week"] == 1
    print(f"\n{name}: {len(d)} games ({int(short.sum())} short-week)")
    print(f"  {'candidate':<28}{'MAE':>8}{'change':>9}{'95% CI':>18}{'short-week gap':>17}")
    for c in ["base"] + candidates:
        p = d[c].to_numpy()
        mae = np.abs(p - y).mean()
        gap = (np.abs(p - y) - np.abs(d["spread_line"] - y))[short].mean()
        if c == "base":
            print(f"  {c:<28}{mae:>8.3f}{'':>9}{'':>18}{gap:>+17.2f}")
        else:
            delta, lo, hi = paired(y, d["base"].to_numpy(), p)
            print(f"  {c:<28}{mae:>8.3f}{delta:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}{gap:>+17.2f}")


def main() -> None:
    df = load()
    wf = walk_forward(df)
    candidates = [c for c in CANDIDATES if c != "base"]

    discover = wf[wf["year"] < CONFIRM_SINCE]
    report("Step 1, discover (2004-2019)", discover, candidates)
    y = discover["home_spread"].to_numpy()
    best = min(candidates, key=lambda c: np.abs(discover[c] - y).mean())
    print(f"\nPicked: {best}")

    report("Step 2, confirm (2020-2025)", wf[wf["year"] >= CONFIRM_SINCE], [best])

    # the committed benchmark window: fit before 2023, test 2023-2025
    train, test = df[df["year"] < 2023], df[df["year"].between(2023, 2025)].copy()
    for c in ["base", best]:
        test[c] = fit_predict(train, test, CANDIDATES[c])
    report("Benchmark window (fit before 2023, test 2023-2025)", test, [best])


if __name__ == "__main__":
    main()
