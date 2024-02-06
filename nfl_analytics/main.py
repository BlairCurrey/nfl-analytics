import argparse
import time

import pandas as pd
from joblib import load

from nfl_analytics.data import (
    download_data,
    load_dataframe_from_raw,
    save_dataframe,
    load_dataframe_from_remote,
)
from nfl_analytics.model import (
    train_model,
    predict,
    save_model_and_scaler,
)
from nfl_analytics.dataframes import (
    build_running_avg_dataframe,
    build_training_dataframe,
)
from nfl_analytics.utils import is_valid_year, get_latest_timestamped_filepath
from nfl_analytics.config import (
    TEAMS,
    RUNNING_AVG_DF_FILENAME,
    TRAINED_MODEL_FILENAME,
    TRAINED_SCALER_FILENAME,
)


# ROUGH CLI docs:
# --download: optional. takes list of years. or if empty, defaults to downloading all years. usage: python main.py --download 2021 2022
# --train: optional. if present, trains the model. usage: python main.py --train
# --predict: optional. takes two arguments, home team and away team. usage: python main.py --predict "CHI" "MIN"


def main():
    parser = argparse.ArgumentParser(description="Manage NFL Spread Predictor Pipeline")
    parser.add_argument(
        "--download",
        nargs="*",
        type=int,
        metavar="year",
        help="Download data for the specified years. The year corresponds to the season start.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model using the downloaded data.",
    )
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("home_team", "away_team"),
        help="Specify the home and away teams for prediction.",
    )
    args = parser.parse_args()

    if args.download is not None:
        if args.download:
            year_set = set(args.download)
            invalid_years = [year for year in year_set if not is_valid_year(year)]

            if invalid_years:
                print(f"Invalid year(s) provided: {invalid_years}. No data downloaded.")
            else:
                download_data(year_set)
        else:
            download_data()

    if args.train:
        start_time = time.time()
        try:
            # df_raw = load_dataframe_from_raw()
            df_raw = load_dataframe_from_remote()
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run with --download first.")
            return
        end_time = time.time()
        print(f"Loaded dataframe in {end_time - start_time} seconds")

        print("Training model...")

        # This wont pick on updated data (downlaoded new data but still have combined, so it will use that)
        # Save combined dataframe to disk
        # save_dir = os.path.join("data", "combined")
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, "play_by_play_combined.parquet.gzip")
        # df_raw.to_parquet(save_path, compression="gzip")

        timestamp = int(time.time())

        df_running_avg = build_running_avg_dataframe(df_raw)
        save_dataframe(df_running_avg, f"{RUNNING_AVG_DF_FILENAME}-{timestamp}")

        df_training = build_training_dataframe(df_running_avg)
        model, scaler = train_model(df_training)

        save_model_and_scaler(model, scaler, timestamp)

    if args.predict:
        # TODO: this will silently predict based off old data if thats all we have.
        # Perhaps I should require the week/year in the predict fn? Or at least log
        # year/week in predict? Or maybe aligning everything by timestamp will resolve this?
        home_team = args.predict[0].upper()
        away_team = args.predict[1].upper()

        for team in [home_team, away_team]:
            if team not in TEAMS:
                print(f"Invalid team: {team}")
                return

        if home_team == away_team:
            print("Home and away team cannot be the same.")
            return

        try:
            latest_model_filepath = get_latest_timestamped_filepath(
                TRAINED_MODEL_FILENAME, ".joblib"
            )
            latest_scaler_filepath = get_latest_timestamped_filepath(
                TRAINED_SCALER_FILENAME, ".joblib"
            )
        except FileNotFoundError:
            print(
                "No trained model and/or scaler found. Please run with --train first."
            )
            return
        model, scaler = load(latest_model_filepath), load(latest_scaler_filepath)

        try:
            latest_running_avg_filename = get_latest_timestamped_filepath(
                RUNNING_AVG_DF_FILENAME, ".csv.gz"
            )
        except FileNotFoundError:
            print("No running average dataframe found. Please run with --train first.")
            return
        df_running_avg = pd.read_csv(latest_running_avg_filename, low_memory=False)

        predicted_spread = predict(model, scaler, df_running_avg, home_team, away_team)

        print(
            f"Predicted spread for {home_team} (home) vs {away_team} (away): {predicted_spread}"
        )


if __name__ == "__main__":
    main()
