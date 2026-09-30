import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import pandas as pd

from nfl_analytics.data import (
    DATA_DIR,
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
from nfl_analytics.benchmark import (
    BENCHMARK_PATH,
    BenchmarkMismatchError,
    check_benchmark,
    compute_benchmark,
    format_benchmark,
    load_benchmark,
    write_benchmark,
)
from nfl_analytics.validate import (
    ValidationError,
    raise_if_problems,
    validate_predictions,
    validate_raw_data,
)
from nfl_analytics import ledger, runs
from nfl_analytics.lines import fetch_lines
from nfl_analytics.provenance import collect_provenance
from nfl_analytics.utils import (
    is_valid_year,
    normalize_team_abbr,
)
from nfl_analytics.config import (
    BENCHMARK_TEST_SEASONS,
    LEDGER_FILENAME,
    START_YEAR,
    TEAMS,
    MATCHUPS_FILENAME,
    PREDICTIONS_FILENAME,
    RELEASE_NOTES_FILENAME,
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

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the fixed backtest and compare it with the committed "
        "benchmark.json. Fails if they differ.",
    )
    benchmark_parser.add_argument(
        "--write",
        action="store_true",
        help="Update benchmark.json instead of checking it. Do this (and "
        "commit it) when you change the model on purpose.",
    )
    benchmark_parser.add_argument(
        "--report",
        metavar="path",
        help="Also append a markdown comparison table to this file.",
    )

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Full weekly pipeline: fetch matchups, update and validate data, "
        "check the benchmark, train, predict, and grade past predictions. "
        "Exits cleanly when there are no upcoming games (offseason).",
    )
    pipeline_parser.add_argument(
        "--previous",
        metavar="dir",
        help="Directory holding the previous release's files. Its ledger.csv "
        "is graded and carried forward.",
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


def _require_data() -> None:
    if not get_downloaded_years():
        sys.exit(
            "No downloaded data found. Run `nfl download` first, "
            "or `nfl update` to download and train in one step."
        )


def _load_raw() -> pd.DataFrame:
    start_time = time.time()
    print("Loading dataframe...")
    df_raw = load_dataframe_from_raw()
    print(f"Loaded dataframe in {time.time() - start_time:.1f} seconds")
    return df_raw


def _train_and_save(
    df_running_avg: pd.DataFrame,
    df_training: pd.DataFrame,
    details: dict[str, Any],
) -> str:
    print("Training model...")
    model, scaler, training = train_model(df_training)

    details = {
        "training": training,
        **details,
        "provenance": collect_provenance(DATA_DIR),
    }
    return runs.save_run(model, scaler, df_running_avg, details)


def run_train() -> str:
    _require_data()
    df_raw = _load_raw()

    vegas_lines = extract_vegas_lines(df_raw)
    df_running_avg = build_running_avg_dataframe(df_raw)
    df_training = build_training_dataframe(df_running_avg)

    try:
        benchmark = compute_benchmark(df_training, vegas_lines)
    except ValueError as e:
        print(f"Skipping benchmark: {e}")
        benchmark = None

    return _train_and_save(df_running_avg, df_training, {"benchmark": benchmark})


def run_benchmark(write: bool, report_path: Optional[str]) -> None:
    # The benchmark window is fixed, so it can fetch exactly the seasons it needs
    needed = set(range(START_YEAR, BENCHMARK_TEST_SEASONS[1] + 1))
    missing = sorted(needed - get_downloaded_years())
    if missing:
        print(f"Downloading season(s) needed for the benchmark: {missing}")
        download_data(missing)

    df_raw = _load_raw()
    df_raw = df_raw[df_raw["year"].isin(needed)]

    print("Building training data and running the benchmark...")
    vegas_lines = extract_vegas_lines(df_raw)
    df_training = build_training_dataframe(build_running_avg_dataframe(df_raw))
    benchmark = compute_benchmark(df_training, vegas_lines)

    committed = load_benchmark() if os.path.isfile(BENCHMARK_PATH) else None
    table = format_benchmark(benchmark, committed)
    print()
    print(table)

    if report_path:
        with open(report_path, "a") as f:
            f.write(table + "\n")

    if write:
        write_benchmark(benchmark)
        return

    if committed is None:
        sys.exit("No committed benchmark.json. Run `nfl benchmark --write`.")

    try:
        check_benchmark(benchmark)
    except BenchmarkMismatchError as e:
        sys.exit(str(e))

    print("Benchmark matches benchmark.json.")


def run_evaluate(test_since: int) -> None:
    _require_data()
    df_raw = _load_raw()

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


def _refresh_data() -> None:
    downloaded = get_downloaded_years()
    missing = set(default_years()) - downloaded
    # Always re-download the current season: its file grows as games are played
    to_download = sorted(missing | {latest_season_year()})

    print(f"Downloading season(s): {to_download}")
    download_data(to_download)


def run_update() -> str:
    _refresh_data()
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


def _predict_matchups(
    model, scaler, df_running_avg: pd.DataFrame, matchups: List[Matchup]
) -> List[Prediction]:
    predictions: List[Prediction] = []

    for matchup in matchups:
        home_team, away_team = _validate_matchup(matchup.home_team, matchup.away_team)
        predicted_spread = predict(model, scaler, df_running_avg, home_team, away_team)
        predictions.append(
            Prediction(
                home_team, away_team, predicted_spread, matchup.season, matchup.week
            )
        )
        print(
            f"{home_team} (home) vs {away_team} (away): {predicted_spread:.1f}"
        )

    return predictions


def run_predict_upcoming(matchups_path: Optional[str], run_id: Optional[str]) -> None:
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
    predictions = _predict_matchups(model, scaler, df_running_avg, matchups)

    runs.save_run_json(manifest["run_id"], MATCHUPS_FILENAME, matchups)
    runs.save_run_json(manifest["run_id"], PREDICTIONS_FILENAME, predictions)


def run_pipeline(previous_dir: Optional[str]) -> None:
    """Every check runs before anything is written for publishing, so a
    failure leaves nothing to release."""
    # Check for matchups first: during the offseason there is nothing to
    # predict, so skip the expensive download/train steps entirely.
    print("Fetching upcoming matchups...")
    matchups = get_upcoming_matchups()

    if not matchups:
        print("No upcoming matchups (offseason?). Nothing to do.")
        return

    season, week = matchups[0].season, matchups[0].week
    print(f"Found {len(matchups)} upcoming matchup(s) in {season} week {week}.")

    _refresh_data()
    df_raw = _load_raw()

    try:
        raise_if_problems(
            "Play-by-play data", validate_raw_data(df_raw, season, week)
        )
    except ValidationError as e:
        sys.exit(str(e))

    vegas_lines = extract_vegas_lines(df_raw)
    results = ledger.game_results(df_raw)
    df_running_avg = build_running_avg_dataframe(df_raw)
    del df_raw
    df_training = build_training_dataframe(df_running_avg)

    # The committed benchmark was reviewed with the code. If recomputing it
    # gives different numbers, the data or dependencies changed underneath.
    print("Checking the benchmark...")
    benchmark = compute_benchmark(df_training, vegas_lines)
    try:
        check_benchmark(benchmark)
    except BenchmarkMismatchError as e:
        sys.exit(str(e))

    run_id = _train_and_save(
        df_running_avg,
        df_training,
        {"benchmark": benchmark, "target": {"season": season, "week": week}},
    )

    # Predict from the saved run, not the in-memory model, so the published
    # files are exactly what produced the published predictions
    model, scaler, df_running_avg, _ = _load_run_or_exit(run_id)
    predictions = _predict_matchups(model, scaler, df_running_avg, matchups)
    _attach_vegas_lines(predictions, season, week)

    try:
        raise_if_problems(
            "Predictions",
            validate_predictions(matchups, predictions, datetime.now(timezone.utc)),
        )
    except ValidationError as e:
        sys.exit(str(e))

    previous_ledger = (
        os.path.join(previous_dir, LEDGER_FILENAME) if previous_dir else ""
    )
    graded = ledger.grade(ledger.load_ledger(previous_ledger), results)
    updated = ledger.add_predictions(graded, matchups, predictions, run_id)

    run_dir = runs.get_run_dir(run_id)
    ledger.save_ledger(updated, os.path.join(run_dir, LEDGER_FILENAME))
    runs.save_run_json(run_id, MATCHUPS_FILENAME, matchups)
    runs.save_run_json(run_id, PREDICTIONS_FILENAME, predictions)

    notes = _release_notes(matchups, predictions, updated, benchmark)
    with open(os.path.join(run_dir, RELEASE_NOTES_FILENAME), "w") as f:
        f.write(notes)
    print()
    print(notes)


def _attach_vegas_lines(predictions: List[Prediction], season: int, week: int) -> None:
    """Record the current Vegas line on each prediction. Lines are useful but
    not essential, so an unavailable source doesn't block publishing."""
    print("Fetching current Vegas lines...")
    try:
        lines = fetch_lines(season, week)
    except (OSError, ValueError) as e:
        print(f"Couldn't fetch Vegas lines; publishing without them: {e}")
        return

    for p in predictions:
        p.vegas_line = lines.get((p.home_team, p.away_team))

    found = sum(p.vegas_line is not None for p in predictions)
    print(f"Found Vegas lines for {found} of {len(predictions)} games.")


def _line_text(home: str, away: str, home_margin: Optional[float]) -> str:
    """Sportsbook style: the favorite takes the minus sign."""
    if home_margin is None:
        return "—"
    if abs(home_margin) < 0.05:
        return "Even"

    team = home if home_margin > 0 else away
    points = abs(home_margin)
    # Vegas lines are in half points ("3", "2.5"); model margins get one decimal
    shown = f"{points:g}" if (points * 2).is_integer() else f"{points:.1f}"
    return f"{team} −{shown}"


def _release_notes(
    matchups: List[Matchup],
    predictions: List[Prediction],
    ledger_df: pd.DataFrame,
    benchmark: dict[str, Any],
) -> str:
    season, week = matchups[0].season, matchups[0].week
    lines = [
        f"Predicted spreads for {season} week {week}, published before kickoff. "
        "Lines are written like a sportsbook's: the favorite takes the minus sign. "
        "The Vegas column is the line when these predictions were published.",
        "",
        "| game | model | Vegas | kickoff (UTC) |",
        "| --- | --- | --- | --- |",
    ]
    for m, p in zip(matchups, predictions):
        lines.append(
            f"| {m.away_team} @ {m.home_team} "
            f"| {_line_text(m.home_team, m.away_team, p.spread)} "
            f"| {_line_text(m.home_team, m.away_team, p.vegas_line)} "
            f"| {m.kickoff} |"
        )

    lines += [
        "",
        "## Scorecard",
        "",
        "Every published prediction, graded against the final score and the "
        "Vegas closing line (full history in `ledger.csv`).",
        "",
        ledger.summarize(ledger_df),
        "",
        "## Benchmark",
        "",
        format_benchmark(benchmark),
        "",
        "## Files",
        "",
        "- **predictions.json / matchups.json:** this week's predictions and games",
        "- **ledger.csv:** every published prediction, with results once played",
        "- **model.json:** scaler and regression parameters (plain JSON, no pickles)",
        "- **running_average.csv.gz:** team running averages used as model inputs",
        "- **manifest.json:** training info, benchmark, and provenance (git SHA, "
        "`uv.lock` and data hashes, library versions)",
        "",
    ]
    return "\n".join(lines)


def main():
    args = build_parser().parse_args()

    if args.command == "download":
        run_download(args.years)
    elif args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate(args.test_since)
    elif args.command == "benchmark":
        run_benchmark(args.write, args.report)
    elif args.command == "update":
        run_update()
    elif args.command == "predict":
        run_predict(args.home_team, args.away_team, args.run)
    elif args.command == "predict-upcoming":
        run_predict_upcoming(args.matchups, args.run)
    elif args.command == "run-pipeline":
        run_pipeline(args.previous)


if __name__ == "__main__":
    main()
