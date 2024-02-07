import datetime
import os

from nfl_analytics.config import START_YEAR, ASSET_DIR as ASSET_DIR_

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(THIS_DIR, ASSET_DIR_)


def is_valid_year(year):
    current_year = datetime.datetime.now().year
    return START_YEAR <= year <= current_year


def get_latest_timestamped_filepath(starts_with, ends_with):
    matching_files = [
        file
        for file in os.listdir(ASSET_DIR)
        if file.startswith(starts_with) and file.endswith(ends_with)
    ]

    if not matching_files:
        raise FileNotFoundError(f"No matching files found")

    sorted_files = sorted(matching_files)
    latest_filename = sorted_files[-1]

    return os.path.join(ASSET_DIR, latest_filename)


if __name__ == "__main__":
    print(is_valid_year(1998))  # False
    print(is_valid_year(1999))  # True
    print(is_valid_year(2000))  # True
    print(is_valid_year(2023))  # True
    print(is_valid_year(2024))  # True
    print(is_valid_year(2025))  # False (if current year is 2024)
