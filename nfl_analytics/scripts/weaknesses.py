"""
Where is the model weakest relative to Vegas?

Walk-forward backtest: each season from 2004 on is predicted by a model trained
only on earlier seasons. Every game gets a paired score,

    gap = |model - actual| - |vegas closing line - actual|

(positive = the model missed by more). A group of games is a weakness when its
average gap is clearly larger than the gap on all other games.

Hypotheses were fixed before looking at results, each with a reason:

  1. early season (weeks 1-4): averages lean on last season's priors
  2. final regular-season week: starters rest; the model can't see it
  3. big favorites: the model looked more extreme than Vegas on them
  4. big disagreement with Vegas: usually something the model can't see
  5. home-field advantage by era: the model weights all seasons equally
  6. rest differences (byes, short weeks): the model has no rest data
  7. starting QB change: team averages still reflect the previous QB
  8. division games
  9. extreme weather (wind >= 15 mph or temp <= 32F, outdoors)

To avoid finding patterns in noise, groups are discovered on 2004-2019 and
then confirmed (or not) on 2020-2025 without changes.

Run: uv run python -m nfl_analytics.scripts.weaknesses
"""

import numpy as np
import pandas as pd

from nfl_analytics.config import FEATURES
from nfl_analytics.data import load_dataframe_from_raw
from nfl_analytics.dataframes import (
    build_running_avg_dataframe,
    build_training_dataframe,
)
from nfl_analytics.evaluate import extract_vegas_lines
from nfl_analytics.lines import GAMES_URL
from nfl_analytics.model import train_model

FIRST_TEST_SEASON = 2004
CONFIRM_SINCE = 2020
BOOTSTRAPS = 2000
rng = np.random.default_rng(0)


def walk_forward_predictions() -> pd.DataFrame:
    raw = load_dataframe_from_raw()
    lines = extract_vegas_lines(raw)
    df = (
        build_training_dataframe(build_running_avg_dataframe(raw))
        .drop_duplicates(subset="game_id")
        .dropna(subset=FEATURES + ["home_spread"])
        .merge(lines, on="game_id")
        .dropna(subset=["spread_line"])
    )

    seasons = []
    for season in range(FIRST_TEST_SEASON, int(df["year"].max()) + 1):
        model, scaler, _ = train_model(df[df["year"] < season])
        test = df[df["year"] == season].copy()
        test["model"] = model.predict(scaler.transform(test[FEATURES]))
        seasons.append(test)

    out = pd.concat(seasons)
    out = out.rename(columns={"spread_line": "vegas", "home_spread": "actual"})
    return out[["game_id", "year", "week", "home_team", "away_team", "model", "vegas", "actual"]]


def add_game_context(df: pd.DataFrame) -> pd.DataFrame:
    games = pd.read_csv(GAMES_URL)
    games = games.sort_values(["season", "gameday", "gametime"])

    # A team's starting QB changed if it differs from that team's previous
    # game in the same season (unknown for each team's first game)
    starts = pd.concat(
        [
            games[["game_id", "season", "gameday", "home_team", "home_qb_id"]].set_axis(
                ["game_id", "season", "gameday", "team", "qb"], axis=1
            ),
            games[["game_id", "season", "gameday", "away_team", "away_qb_id"]].set_axis(
                ["game_id", "season", "gameday", "team", "qb"], axis=1
            ),
        ]
    ).sort_values(["season", "gameday"])
    starts["prev_qb"] = starts.groupby(["season", "team"])["qb"].shift()
    starts["qb_changed"] = np.where(
        starts["prev_qb"].isna() | starts["qb"].isna(),
        np.nan,
        (starts["qb"] != starts["prev_qb"]).astype(float),
    )
    qb_change = starts.groupby("game_id")["qb_changed"].max()  # either team

    context = games.set_index("game_id")[
        ["home_rest", "away_rest", "div_game", "roof", "temp", "wind", "game_type"]
    ].join(qb_change)

    return df.join(context, on="game_id")


def label_groups(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Hypothesis -> {group name: boolean mask}. Masks are NaN-safe."""
    final_week = df.groupby("year")["week"].transform(
        lambda w: w[w <= 18].max()
    )  # 17 before 2021, 18 since
    abs_vegas = df["vegas"].abs()
    disagreement = (df["model"] - df["vegas"]).abs()
    rest_diff = df["home_rest"] - df["away_rest"]
    outdoors = df["roof"].isin(["outdoors", "open"])
    regular = df["week"] <= 18

    return {
        "1. early season": {"weeks 1-4": df["week"] <= 4},
        "2. final regular-season week": {"final week": df["week"] == final_week},
        "3. favorite size (Vegas)": {
            "0-3": abs_vegas <= 3,
            "3.5-7": (abs_vegas > 3) & (abs_vegas <= 7),
            "7.5-10": (abs_vegas > 7) & (abs_vegas <= 10),
            "10.5+": abs_vegas > 10,
        },
        "4. disagreement with Vegas": {
            "under 2": disagreement < 2,
            "2-4": (disagreement >= 2) & (disagreement < 4),
            "4-6": (disagreement >= 4) & (disagreement < 6),
            "6+": disagreement >= 6,
        },
        "5. era": {
            "2004-2011": df["year"] <= 2011,
            "2012-2019": (df["year"] >= 2012) & (df["year"] <= 2019),
            "2020-2025": df["year"] >= 2020,
        },
        "6. rest": {
            "home 3+ days more": rest_diff >= 3,
            "away 3+ days more": rest_diff <= -3,
            "short week (either <= 5 days)": (df[["home_rest", "away_rest"]].min(axis=1) <= 5) & regular,
        },
        "7. starting QB change": {"either team changed QB": df["qb_changed"] == 1},
        "8. division game": {"division game": df["div_game"] == 1},
        "9. extreme weather": {
            "wind 15+ or temp <= 32": outdoors & ((df["wind"] >= 15) | (df["temp"] <= 32)),
        },
    }


def compare(gap: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Group gap minus the rest's gap, with a bootstrap 95% interval."""
    inside, outside = gap[mask], gap[~mask]
    diff = inside.mean() - outside.mean()
    boot = (
        inside[rng.integers(0, len(inside), (BOOTSTRAPS, len(inside)))].mean(axis=1)
        - outside[rng.integers(0, len(outside), (BOOTSTRAPS, len(outside)))].mean(axis=1)
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff, lo, hi


def summarize(df: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    g = df[mask]
    # bias toward the Vegas favorite: positive = the model rates the favorite
    # higher than the result did
    fav = np.sign(g["vegas"]).replace(0, 1)
    return {
        "n": len(g),
        "model_mae": (g["model"] - g["actual"]).abs().mean(),
        "vegas_mae": (g["vegas"] - g["actual"]).abs().mean(),
        "model_home_bias": (g["model"] - g["actual"]).mean(),
        "vegas_home_bias": (g["vegas"] - g["actual"]).mean(),
        "model_fav_bias": ((g["model"] - g["actual"]) * fav).mean(),
        "vegas_fav_bias": ((g["vegas"] - g["actual"]) * fav).mean(),
    }


def main() -> None:
    df = add_game_context(walk_forward_predictions())
    df["gap"] = (df["model"] - df["actual"]).abs() - (df["vegas"] - df["actual"]).abs()

    periods = {
        "discover": df["year"] < CONFIRM_SINCE,
        "confirm": df["year"] >= CONFIRM_SINCE,
    }
    for name, period in periods.items():
        d = df[period]
        print(
            f"{name}: {len(d)} games, model MAE {(d.model - d.actual).abs().mean():.2f}, "
            f"Vegas MAE {(d.vegas - d.actual).abs().mean():.2f}, gap {d.gap.mean():+.2f}"
        )

    rows = []
    for hypothesis, groups in label_groups(df).items():
        for group, mask in groups.items():
            mask = mask.fillna(False).astype(bool)
            row = {"hypothesis": hypothesis, "group": group}
            for name, period in periods.items():
                d, m = df[period], mask[period].to_numpy()
                if m.sum() < 30 or (~m).sum() < 30:
                    continue
                diff, lo, hi = compare(d["gap"].to_numpy(), m)
                s = summarize(d, m)
                row |= {
                    f"{name}_n": s["n"],
                    f"{name}_gap": d["gap"].to_numpy()[m].mean(),
                    f"{name}_vs_rest": diff,
                    f"{name}_ci": f"[{lo:+.2f}, {hi:+.2f}]",
                    f"{name}_clear": lo > 0 or hi < 0,
                    f"{name}_model_fav_bias": s["model_fav_bias"],
                    f"{name}_vegas_fav_bias": s["vegas_fav_bias"],
                    f"{name}_model_home_bias": s["model_home_bias"],
                    f"{name}_vegas_home_bias": s["vegas_home_bias"],
                }
            rows.append(row)

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print("\nGap vs the rest of the games (positive = model relatively worse here)")
    cols = ["hypothesis", "group"] + [
        f"{p}_{c}" for p in periods for c in ["n", "gap", "vs_rest", "ci", "clear"]
    ]
    print(results[cols].round(2).to_string(index=False))

    print("\nBias (mean of prediction - actual). fav = toward the Vegas favorite; home = toward the home team")
    cols = ["hypothesis", "group"] + [
        f"{p}_{c}"
        for p in periods
        for c in ["model_fav_bias", "vegas_fav_bias", "model_home_bias", "vegas_home_bias"]
    ]
    print(results[cols].round(2).to_string(index=False))

    print("\nHome-field edge by season (actual mean home margin vs what each predicted)")
    by_year = df.groupby("year")[["actual", "model", "vegas"]].mean().round(2)
    print(by_year.to_string())


if __name__ == "__main__":
    main()
