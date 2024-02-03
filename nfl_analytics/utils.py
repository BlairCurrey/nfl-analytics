import datetime

from nfl_analytics.config import START_YEAR


def is_valid_year(year):
    current_year = datetime.datetime.now().year
    return START_YEAR <= year <= current_year


if __name__ == "__main__":
    print(is_valid_year(1998))  # False
    print(is_valid_year(1999))  # True
    print(is_valid_year(2000))  # True
    print(is_valid_year(2023))  # True
    print(is_valid_year(2024))  # True
    print(is_valid_year(2025))  # False (if current year is 2024)
