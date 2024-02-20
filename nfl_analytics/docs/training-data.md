## Training Data

The model was trained using play-by-play data since 1999 from this [release](https://github.com/nflverse/nflverse-data/releases/tag/pbp) from the [nflfastR project](https://github.com/nflverse/nflfastR). This data is fetched in `./nfl_analytics/data.py`.

I aggregated team stats from all these individual playsto get a snapshot of teams at every week of each season. The resulting dataset is a running total of team stats by team and week and looks something like this:

| TEAM | WEEK | YEAR | SPREAD_ACTUAL | PASSING_YARDS_AVG | RUSHING_YARDS_AVG | PASSING_YARDS_ALLOWED_AVG | RUSHING_YARDS_ALLOWED_AVG |
| ---- | ---- | ---- | ------------- | ----------------- | ----------------- | ------------------------- | ------------------------- |
| CHI  | 1    | 2021 | -4            | NaN               | NaN               | NaN                       | NaN                       |
| CHI  | 2    | 2021 | 3             | 232               | 120               | 300                       | 95                        |
| CHI  | 3    | 2021 | -8            | 245               | 136               | 278                       | 102                       |

This (imagined) chunk of the data shows that in week 3 of 2021, the Chicago Bears averaged 245 pass yards up to that point. That is, they averaged 245 pass yards through the first two games. Hence, the first week average will always be `NaN`. The `SPREAD_ACTUAL` is the actual point difference in the game and is used as the target for the model to predict.

This dataset is built from the raw data in `./nfl_analytics/dataframes.py`. It was originally developed in `./nfl_analytics/dev_notebook.ipynb` before being refactored into the `dataframes` module.
