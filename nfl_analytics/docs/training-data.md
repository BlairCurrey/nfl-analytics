## Training Data

The model was trained using play-by-play data since 1999 from this [release](https://github.com/nflverse/nflverse-data/releases/tag/pbp) from the [nflfastR project](https://github.com/nflverse/nflfastR). This data is fetched in `./nfl_analytics/data.py`. I chose this source because it's free, reliable, not breaking any terms of service, and has all the necessary data. While it leaves substantial work to aggregate the data in a meaningful way, using the most atomic form (a single play) allows me to aggregate it however I need.

I aggregated team stats from all these individual plays to get a snapshot of teams at every week of each season. The resulting dataset is a running total of team stats by team and week and looks something like this:

| TEAM | WEEK | YEAR | SPREAD_ACTUAL | PASSING_YARDS_AVG | RUSHING_YARDS_AVG | PASSING_YARDS_ALLOWED_AVG | RUSHING_YARDS_ALLOWED_AVG |
| ---- | ---- | ---- | ------------- | ----------------- | ----------------- | ------------------------- | ------------------------- |
| CHI  | 1    | 2021 | -4            | NaN               | NaN               | NaN                       | NaN                       |
| CHI  | 2    | 2021 | 3             | 232               | 120               | 300                       | 95                        |
| CHI  | 3    | 2021 | -8            | 245               | 136               | 278                       | 102                       |

This (imagined) chunk of the data shows that in week 3 of 2021, the Chicago Bears averaged 245 pass yards up to that point. That is, they averaged 245 pass yards through the first two games. Early-season averages also blend in a prior of a few pseudo-games at the team's previous-season mean (`PRIOR_PSEUDO_GAMES` in `config.py`), so a week 2 average isn't driven entirely by a single game; the prior's influence decays as real games accumulate. For a team with no previous season under the same abbreviation, the first week average is `NaN`. The `SPREAD_ACTUAL` is the actual point difference in the game and is used as the target for the model to predict. This is an intermediate step towards the final training dataset which is referred to as a "running average" (`df_running_avg` in code, `running_average-[timstamp]` filename, etc.)

This running average dataset is used to make matchups for the training dataset and predictions. The training set consists of this dataset joined by week. The teams are discarded and the team stats are renamed to "HOME_PASSING_YARDS_AVG", "AWAY_RUSHING_YARDS_AVG", etc. The running average dataset is also used to make prediction inputs, which are the matchups for two given teams joined together in the same way. The running average dataset is saved so that it can be used to make predictions in the future.

The `spread_line` column used as the Vegas benchmark in `nfl evaluate` also comes from this play-by-play data. It is inherited verbatim from [Lee Sharpe's nflverse games dataset](https://github.com/nflverse/nfldata) (verified: exact match for all 7,272 games) and behaves like genuine closing lines (full coverage since 1999, home-cover rates near 50%, spot-checks against publicly known Super Bowl lines). Positive values mean the home team was favored.

These running average and training datasets are built from the raw play-by-play data in `./nfl_analytics/dataframes.py`. It was originally developed in `./nfl_analytics/dev_notebook.ipynb` before being refactored into the `dataframes` module. As such, the dev notebook may not work perfectly and may not be exactly up to date.
