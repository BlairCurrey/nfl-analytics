# Loads csvs into pandas dataframe and sqlite db
import os
import pandas as pd
import sqlite3

data_directory = "nfl_analytics/data"
combined_df = pd.DataFrame()

for filename in os.listdir(data_directory):
    if filename.endswith(".csv.gz"):
        print(f"Reading {filename}")
        file_path = os.path.join(data_directory, filename)

        # Read the CSV file into a DataFrame and concat to combined df
        df = pd.read_csv(file_path, compression="gzip", low_memory=False)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

print(combined_df.head())

table_name = "plays"
db_conn = sqlite3.connect(database="/tmp/my.db")
# TODO: remove drop table after developing?
db_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
num_rows_inserted = combined_df.to_sql(table_name, db_conn, index=False)

cursor = db_conn.execute(f"SELECT * from {table_name} LIMIT 10")
print(cursor.fetchall())
