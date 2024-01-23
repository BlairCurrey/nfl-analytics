import urllib.request
import os

years = range(1999, 2024)

save_directory = "data"
os.makedirs(save_directory, exist_ok=True)

for year in years:
    filename = f"play_by_play_{year}.csv.gz"
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/{filename}"
    save_path = os.path.join(save_directory, filename)

    urllib.request.urlretrieve(url, save_path)
