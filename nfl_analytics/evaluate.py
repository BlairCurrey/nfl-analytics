"""
Honest evaluation of the spread model against fixed benchmarks.

Unlike training (which fits on all data), evaluation holds out entire seasons:
the model is fit only on seasons before `test_since` and scored on the rest.
Three predictors are compared on the same held-out games:

- naive: always predicts the training-set mean home margin (home-field advantage)
- model: the linear regression, fit on training seasons only
- vegas: the closing spread from the play-by-play data (`spread_line`,
  positive = home favored)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from nfl_analytics.config import FEATURES


def extract_vegas_lines(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One closing spread per game from raw play-by-play data."""
    return df_raw[["game_id", "spread_line"]].drop_duplicates(subset="game_id")


def evaluate_spread_model(
    df_training: pd.DataFrame,
    vegas_lines: pd.DataFrame,
    test_since: int = 2023,
) -> dict[str, Any]:
    # The training dataframe has two identical rows per game (one per team
    # perspective); use one clean row per game. Rows without complete features
    # (e.g. week 1 for a team with no prior season) are dropped, matching what
    # training and live prediction can actually use.
    df = df_training.drop_duplicates(subset="game_id").dropna(
        subset=FEATURES + ["home_spread"]
    )

    train = df[df["year"] < test_since]
    test = df[df["year"] >= test_since].merge(vegas_lines, on="game_id", how="left")

    if train.empty or test.empty:
        raise ValueError(
            f"Not enough data to split at {test_since}: "
            f"{len(train)} train games, {len(test)} test games. "
            "Make sure data through the test seasons is downloaded."
        )

    scaler = StandardScaler().fit(train[FEATURES])
    model = LinearRegression().fit(
        scaler.transform(train[FEATURES]), train["home_spread"]
    )

    y = test["home_spread"].to_numpy()
    model_pred = model.predict(scaler.transform(test[FEATURES]))
    naive_pred = np.full(len(test), train["home_spread"].mean())

    def scores(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        return {
            "mae": mean_absolute_error(actual, pred),
            "rmse": mean_squared_error(actual, pred) ** 0.5,
            "bias": float(np.mean(pred - actual)),
        }

    results: dict[str, Any] = {
        "test_since": test_since,
        "n_train_games": len(train),
        "n_test_games": len(test),
        "home_field_advantage": train["home_spread"].mean(),
        "naive": scores(naive_pred, y),
        "model": scores(model_pred, y),
    }

    # calibration: regress actual on predicted; ideal is intercept 0, slope 1.
    # slope > 1 means predictions are too timid, < 1 too bold.
    slope, intercept = np.polyfit(model_pred, y, 1)
    results["calibration"] = {"slope": float(slope), "intercept": float(intercept)}

    # bootstrap: how much of the headline MAE is noise? (paired deltas below
    # are much tighter than this absolute interval)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), (5000, len(y)))
    model_abs_err = np.abs(model_pred - y)
    results["model"]["mae_ci95"] = float(1.96 * model_abs_err[idx].mean(axis=1).std())

    # Vegas comparison only on games that have a line
    has_line = test["spread_line"].notna().to_numpy()
    results["n_games_with_vegas_line"] = int(has_line.sum())

    if has_line.any():
        vegas_pred = test.loc[has_line, "spread_line"].to_numpy()
        y_lined = y[has_line]
        results["vegas"] = scores(vegas_pred, y_lined)

        # one-number summary of "how far from ignorance toward the market":
        # 0% = naive, 100% = vegas
        naive_mae_lined = mean_absolute_error(y_lined, naive_pred[has_line])
        model_mae_lined = mean_absolute_error(y_lined, model_pred[has_line])
        results["gap_closed"] = float(
            (naive_mae_lined - model_mae_lined)
            / (naive_mae_lined - results["vegas"]["mae"])
        )

        # distance to the line: removes game-outcome noise, so it moves more
        # decisively than margin MAE when the model genuinely changes
        results["vs_vegas"] = {
            "mae_to_line": mean_absolute_error(vegas_pred, model_pred[has_line]),
            "corr": float(np.corrcoef(model_pred[has_line], vegas_pred)[0, 1]),
        }

        # paired bootstrap of (model - vegas) MAE on the same games
        delta = np.abs(model_pred[has_line] - y_lined) - np.abs(vegas_pred - y_lined)
        idx_l = rng.integers(0, len(y_lined), (5000, len(y_lined)))
        results["vs_vegas"]["delta_mae"] = float(delta.mean())
        results["vs_vegas"]["delta_mae_ci95"] = float(
            1.96 * delta[idx_l].mean(axis=1).std()
        )

        # Against-the-spread: pick the side the model favors vs the closing
        # line; a win means the actual margin landed on the model's side.
        picks = np.sign(model_pred[has_line] - vegas_pred)
        outcomes = np.sign(y_lined - vegas_pred)
        decided = outcomes != 0  # pushes don't count
        ats = float((picks[decided] == outcomes[decided]).mean())
        n_ats = int(decided.sum())
        results["ats_accuracy"] = ats
        results["n_ats_games"] = n_ats
        results["ats_ci95"] = float(1.96 * (ats * (1 - ats) / n_ats) ** 0.5)

    return results


def format_report(results: dict[str, Any]) -> str:
    def row(name: str, s: dict[str, Any]) -> str:
        return f"{name:<22}{s['mae']:>8.3f}{s['rmse']:>8.3f}{s['bias']:>+8.2f}"

    cal = results["calibration"]
    lines = [
        f"Held-out evaluation: trained on seasons before {results['test_since']}, "
        f"tested on {results['n_test_games']} games since "
        f"({results['n_train_games']} training games)",
        f"Home-field advantage (train mean home margin): "
        f"{results['home_field_advantage']:+.2f}",
        "",
        f"{'predictor':<22}{'MAE':>8}{'RMSE':>8}{'bias':>8}",
        row("naive (constant HFA)", results["naive"]),
        row("model", results["model"]),
    ]

    if "vegas" in results:
        lines.append(row("vegas closing line", results["vegas"]))
        lines.append(
            f"  (vegas on {results['n_games_with_vegas_line']} games with a line)"
        )

    lines += [
        "",
        f"model MAE 95% CI: ±{results['model']['mae_ci95']:.2f} (bootstrap; judge "
        "changes by paired deltas, not this)",
        f"calibration: actual ≈ {cal['intercept']:+.2f} + {cal['slope']:.3f} × "
        "predicted (ideal: +0.00 + 1.000×)",
    ]

    if "vegas" in results:
        vv = results["vs_vegas"]
        trails = "trails" if vv["delta_mae"] > 0 else "leads"
        significant = abs(vv["delta_mae"]) > vv["delta_mae_ci95"]
        lines += [
            f"distance to vegas line: MAE {vv['mae_to_line']:.2f}, "
            f"corr {vv['corr']:.3f}",
            f"naive-to-vegas gap closed: {results['gap_closed']:.1%}",
            f"model vs vegas paired ΔMAE: {vv['delta_mae']:+.3f} "
            f"± {vv['delta_mae_ci95']:.3f} — model {trails} vegas"
            f"{'' if significant else ' (within noise)'}",
            f"Against-the-spread pick accuracy vs closing line: "
            f"{results['ats_accuracy']:.1%} ± {results['ats_ci95']:.1%} over "
            f"{results['n_ats_games']} games (betting breakeven is ~52.4%)",
        ]
    else:
        lines.append("No Vegas lines found in the test games; skipping comparison.")

    return "\n".join(lines)
