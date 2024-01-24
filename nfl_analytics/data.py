import urllib.request
import os
import pandas as pd
import sqlite3


def get():
    years = range(1999, 2024)

    save_directory = "data"
    os.makedirs(save_directory, exist_ok=True)

    for year in years:
        # year gets parsed from this filename and depends on this format
        filename = f"play_by_play_{year}.csv.gz"
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/{filename}"
        save_path = os.path.join(save_directory, filename)

        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, save_path)


def load_pandas():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, "data")

    combined_df = pd.DataFrame()

    for filename in os.listdir(data_directory):
        if filename.endswith(".csv.gz"):
            print(f"Reading {filename}")
            file_path = os.path.join(data_directory, filename)

            df = pd.read_csv(file_path, compression="gzip", low_memory=False)
            # Save year from filename on dataframe
            year = get_year_from_filename(filename)
            df["year"] = year
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


def get_year_from_filename(filename):
    # Expects filename like play_by_play_2020.csv.gz
    return int(filename[-11:-7])


def load_sqlite():
    # load into pandas first and use to_sql to infer datatypes
    df = load_pandas()

    table_name = "plays"
    db_conn = sqlite3.connect(database="/tmp/nfl-analytics.db")
    # TODO: remove drop table after developing?
    db_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    df.to_sql(table_name, db_conn, index=False)

    cursor = db_conn.execute(f"SELECT * from {table_name} LIMIT 10")
    print(cursor.fetchall())


if __name__ == "__main__":
    get()
    load_sqlite()
