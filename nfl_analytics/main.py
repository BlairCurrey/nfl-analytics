import argparse
import sys
import time
from typing import List, Optional

from nfl_analytics.data import (
    download_data,
    default_years,
    get_downloaded_years,
    latest_season_year,
    load_dataframe_from_raw,
)
from nfl_analytics.model import (
    train_model,
    predict,
    Prediction,
)
from nfl_analytics.dataframes import (
    build_running_avg_dataframe,
    build_training_dataframe,
)
from nfl_analytics.evaluate import (
    evaluate_spread_model,
    extract_vegas_lines,
    format_report,
)
from nfl_analytics.schedule import (
    Matchup,
    get_upcoming_matchups,
    load_matchups,
)
from nfl_analytics import runs
from nfl_analytics.utils import (
    is_valid_year,
    normalize_team_abbr,
)
from nfl_analytics.config import (
    TEAMS,
    MATCHUPS_FILENAME,
    PREDICTIONS_FILENAME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfl",
        description="Predict NFL spreads.",
        epilog="Typical usage: `nfl update` once, then `nfl predict KC SF`.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download raw play-by-play data."
    )
    download_parser.add_argument(
        "years",
        nargs="*",
        type=int,
        metavar="year",
        help="Season start years to download. Defaults to every season since 1999.",
    )

    subparsers.add_parser(
        "train",
        help="Train a model from downloaded data. Saves a new run to assets/runs/.",
    )

    subparsers.add_parser(
        "update",
        help="Download any missing/current season data and train a fresh model. "
        "Handles first-time setup and weekly refreshes.",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Predict the spread for a single matchup."
    )
    predict_parser.add_argument("home_team", help="Home team abbreviation, e.g. KC")
    predict_parser.add_argument("away_team", help="Away team abbreviation, e.g. SF")
    predict_parser.add_argument(
        "--run",
        metavar="run_id",
        help="Training run to use (defaults to the latest).",
    )

    predict_upcoming_parser = subparsers.add_parser(
        "predict-upcoming",
        help="Fetch this week's matchups and predict every spread. "
        "Saves matchups and predictions to the run directory.",
    )
    predict_upcoming_parser.add_argument(
        "--matchups",
        metavar="path",
        help="Predict from a saved matchups JSON file instead of fetching.",
    )
    predict_upcoming_parser.add_argument(
        "--run",
        metavar="run_id",
        help="Training run to use (defaults to the latest).",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Score the model against naive and Vegas baselines on held-out "
        "seasons. Trains on seasons before --test-since, tests on the rest.",
    )
    evaluate_parser.add_argument(
        "--test-since",
        type=int,
        default=2023,
        metavar="year",
        help="First season of the held-out test set (default: 2023).",
    )

    subparsers.add_parser(
        "run-pipeline",
        help="Full weekly pipeline: fetch matchups, update data, train, and "
        "predict. Exits cleanly when there are no upcoming games (offseason).",
    )

    return parser


def run_download(years: List[int]) -> None:
    if years:
        invalid_years = [year for year in set(years) if not is_valid_year(year)]

        if invalid_years:
            sys.exit(f"Invalid year(s) provided: {invalid_years}. No data downloaded.")

        download_data(set(years))
    else:
        download_data()


def run_train() -> str:
    if not get_downloaded_years():
        sys.exit(
            "No downloaded data found. Run `nfl download` first, "
            "or `nfl update` to download and train in one step."
        )

    start_time = time.time()
    print("Loading dataframe...")
    df_raw = load_dataframe_from_raw()
    print(f"Loaded dataframe in {time.time() - start_time:.1f} seconds")

    print("Training model...")
    df_running_avg = build_running_avg_dataframe(df_raw)
    df_training = build_training_dataframe(df_running_avg)
    model, scaler, metrics = train_model(df_training)

    return runs.save_run(model, scaler, df_running_avg, metrics)


def run_evaluate(test_since: int) -> None:
    if not get_downloaded_years():
        sys.exit(
            "No downloaded data found. Run `nfl download` first, "
            "or `nfl update` to download and train in one step."
        )

    start_time = time.time()
    print("Loading dataframe...")
    df_raw = load_dataframe_from_raw()
    print(f"Loaded dataframe in {time.time() - start_time:.1f} seconds")

    print("Building training data and evaluating...")
    vegas_lines = extract_vegas_lines(df_raw)
    df_running_avg = build_running_avg_dataframe(df_raw)
    df_training = build_training_dataframe(df_running_avg)

    try:
        results = evaluate_spread_model(df_training, vegas_lines, test_since)
    except ValueError as e:
        sys.exit(str(e))

    print()
    print(format_report(results))


def run_update() -> str:
    downloaded = get_downloaded_years()
    missing = set(default_years()) - downloaded
    # Always re-download the current season: its file grows as games are played
    to_download = sorted(missing | {latest_season_year()})

    print(f"Downloading season(s): {to_download}")
    download_data(to_download)

    return run_train()


def _load_run_or_exit(run_id: Optional[str]):
    try:
        return runs.load_run(run_id)
    except runs.RunNotFoundError as e:
        sys.exit(str(e))


def _validate_matchup(home_team: str, away_team: str) -> tuple[str, str]:
    home_team = normalize_team_abbr(home_team)
    away_team = normalize_team_abbr(away_team)

    for team in [home_team, away_team]:
        if team not in TEAMS:
            sys.exit(f"Invalid team: {team}. See TEAMS in nfl_analytics/config.py.")

    if home_team == away_team:
        sys.exit("Home and away team cannot be the same.")

    return home_team, away_team


def run_predict(home_team: str, away_team: str, run_id: Optional[str]) -> None:
    home_team, away_team = _validate_matchup(home_team, away_team)

    model, scaler, df_running_avg, _ = _load_run_or_exit(run_id)
    predicted_spread = predict(model, scaler, df_running_avg, home_team, away_team)

    print(
        f"Predicted spread for {home_team} (home) vs {away_team} (away): "
        f"{predicted_spread}"
    )


def run_predict_upcoming(
    matchups_path: Optional[str],
    run_id: Optional[str],
    matchups: Optional[List[Matchup]] = None,
) -> None:
    if matchups is None:
        if matchups_path:
            print(f"Loading matchups from {matchups_path}")
            try:
                matchups = load_matchups(matchups_path)
            except FileNotFoundError:
                sys.exit(f"No matchup file found at {matchups_path}.")
        else:
            print("Fetching upcoming matchups...")
            matchups = get_upcoming_matchups()

    if not matchups:
        print("No upcoming matchups found.")
        return

    model, scaler, df_running_avg, manifest = _load_run_or_exit(run_id)

    predictions: List[Prediction] = []

    for matchup in matchups:
        home_team, away_team = _validate_matchup(matchup.home_team, matchup.away_team)
        predicted_spread = predict(model, scaler, df_running_avg, home_team, away_team)
        predictions.append(Prediction(home_team, away_team, predicted_spread))
        print(
            f"{home_team} (home) vs {away_team} (away): {predicted_spread:.1f}"
        )

    runs.save_run_json(manifest["run_id"], MATCHUPS_FILENAME, matchups)
    runs.save_run_json(manifest["run_id"], PREDICTIONS_FILENAME, predictions)


def run_pipeline() -> None:
    # Check for matchups first: during the offseason there is nothing to
    # predict, so skip the expensive download/train steps entirely.
    print("Fetching upcoming matchups...")
    matchups = get_upcoming_matchups()

    if not matchups:
        print("No upcoming matchups (offseason?). Nothing to do.")
        return

    print(f"Found {len(matchups)} upcoming matchup(s).")
    run_id = run_update()
    run_predict_upcoming(None, run_id, matchups=matchups)


def main():
    args = build_parser().parse_args()

    if args.command == "download":
        run_download(args.years)
    elif args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate(args.test_since)
    elif args.command == "update":
        run_update()
    elif args.command == "predict":
        run_predict(args.home_team, args.away_team, args.run)
    elif args.command == "predict-upcoming":
        run_predict_upcoming(args.matchups, args.run)
    elif args.command == "run-pipeline":
        run_pipeline()


if __name__ == "__main__":
    main()
