"""
Handles fetching and loading the play by play data. Essentially, 
everything before tranforming it.
"""

import urllib.request
from urllib.error import HTTPError
import os
import sqlite3

import pandas as pd

from nfl_analytics.config import (
    DATA_DIR,
    ASSET_DIR as ASSET_DIR_,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(THIS_DIR, ASSET_DIR_)
DATA_DIR_TEST = os.path.join(THIS_DIR, DATA_DIR)


def download_data(years=range(1999, 2024)):
    print("gh actions doesnt like os.makedirs with this: ", DATA_DIR_TEST)
    os.makedirs(DATA_DIR, exist_ok=True)

    for year in years:
        # year gets parsed from this filename and depends on this format
        filename = f"play_by_play_{year}.csv.gz"
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/{filename}"
        save_path = os.path.join(DATA_DIR, filename)

        print(f"Downloading {url} to {save_path}...")

        try:
            urllib.request.urlretrieve(url, save_path)
        except HTTPError as e:
            print(
                f"Error: Failed to download data for {year}. HTTP Error {e.code}: {e.reason}. Season for that year may not exist yet."
            )


def load_dataframe_from_remote(years=range(1999, 2024)):
    combined_df = pd.DataFrame()

    for year in years:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"
        print(f"Reading from remote: {url}")
        df = pd.read_csv(url, low_memory=False)

        # Save year on dataframe
        df["year"] = year
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        raise FileNotFoundError("No data loaded from the remote files.")

    return combined_df


def load_dataframe_from_raw():
    data_directory = DATA_DIR  # os.path.join(THIS_DIR, DATA_DIR)

    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory '{data_directory}' not found.")

    files = os.listdir(data_directory)

    if not files:
        raise FileNotFoundError(f"No data files found in the data directory.")

    # This wont pick on updated data (downlaoded new data but still have combined, so it will use that)
    # # load saved combined from disk if exists
    # combined_file_path = os.path.join(
    #     data_directory, "combined", "play_by_play_combined.parquet.gzip"
    # )
    # if not skip_combined and os.path.exists(combined_file_path):
    #     print(f"Reading combined file {combined_file_path}")
    #     combined_df = pd.read_parquet(combined_file_path)
    #     return combined_df
    # else:
    #     print("Combined file does not exist. Loading individual files.")

    # make combined dataframe from individual files
    combined_df = pd.DataFrame()

    for filename in files:
        if filename.endswith(".csv.gz"):
            print(f"Reading {filename}")
            file_path = os.path.join(data_directory, filename)

            df = pd.read_csv(file_path, compression="gzip", low_memory=False)

            # Save year from filename on dataframe
            year = get_year_from_filename(filename)
            df["year"] = year
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        raise FileNotFoundError("No data loaded from the files.")

    return combined_df


def get_year_from_filename(filename):
    # Expects filename like play_by_play_2020.csv.gz
    return int(filename[-11:-7])


def load_sqlite():
    db_dir = "/tmp/nfl-analytics.db"
    # load into pandas first and use to_sql to infer datatypes
    df = load_dataframe_from_raw()

    print(f"Loading into SQLite database: {db_dir}")

    table_name = "plays"
    db_conn = sqlite3.connect(database=db_dir)
    db_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    df.to_sql(table_name, db_conn, index=False)

    cursor = db_conn.execute(f"SELECT * from {table_name} LIMIT 10")
    print(cursor.fetchall())


def save_dataframe(df, filename_):
    os.makedirs(ASSET_DIR, exist_ok=True)
    filename = f"{filename_}.csv.gz"

    save_path = os.path.join(ASSET_DIR, filename)
    df.to_csv(save_path, index=False, compression="gzip")
    print(f"Running average dataframe saved to {filename}")


if __name__ == "__main__":
    download_data()
    load_sqlite()
