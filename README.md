# About

This repository contains a python cli application for predicting nfl spreads. The app trains a model from the latest available data and predicts upcoming matchups. In addition to running locally, a github action runs the full pipeline weekly during the season and publishes the predictions for upcoming games to [this release](https://github.com/BlairCurrey/nfl-analytics/releases).

Visit the docs for [the model](./nfl_analytics/docs/model.md) and [training data](./nfl_analytics/docs/training-data.md) for more details on each.

This project exists for a few reasons:

- I wanted to see how well a simplistic model would do at predicting the spread. I suspected this is a situation where something like 20% of the work could get you 80% of the results (known as the [Pareto Principle](https://en.wikipedia.org/wiki/Pareto_principle)), with "results" being Vegas-like spread prediction accuracy. I think this ended up being the case. See [the model doc](./nfl_analytics/docs/model.md) for more details.
- I wanted to build an end-to-end training and prediction pipeline in github actions.
- I wanted to compile an insightful dataset from atomic NFL play-by-play data. See the [training data doc](./nfl_analytics/docs/training-data.md) for more details on this.

# Using

## Pre-requisites

- [uv](https://docs.astral.sh/uv/) (python 3.12 and dependencies are managed for you)

## Setup

Clone this repository and install dependencies:

    git clone https://github.com/BlairCurrey/nfl-analytics.git
    cd nfl-analytics
    uv sync

Then download the data and train a model with a single command:

    uv run nfl update

This downloads all missing play-by-play data to `./nfl_analytics/data` and trains a model. Every training run is saved as a self-contained directory under `./nfl_analytics/assets/runs/<run_id>/` containing the model, scaler, running averages, and a manifest with the run's error metrics. Commands that need a model always load the latest complete run (or a specific one via `--run`), so artifacts from different training runs are never mixed.

Re-run `nfl update` any time to pick up the latest games and retrain.

## Predicting games

Predict a specific matchup by giving the home and away team (in that order):

    uv run nfl predict kc sf

The prediction returns a float spread relative to the home team. For example, if the `kc sf` prediction returns 1.3, the model favors kc (the home team) by 1.3 points. An exact list of team abbreviations can be found in `./nfl_analytics/config.py`.

Or fetch this week's matchups and predict all of them, saving the predictions to the run directory:

    uv run nfl predict-upcoming

## All commands

Run `uv run nfl --help` for the full CLI reference.

| Command            | What it does                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `update`           | Download missing/current season data and train a fresh model. First-time setup + weekly refresh. |
| `predict HOME AWAY`| Predict the spread for a single matchup.                                                        |
| `predict-upcoming` | Fetch this week's matchups and predict every spread.                                            |
| `download [years]` | Just download raw play-by-play data.                                                            |
| `train`            | Just train from already-downloaded data.                                                        |
| `evaluate`         | Score the model against naive and Vegas baselines on held-out seasons (`--test-since`, default 2023). |
| `run-pipeline`     | Full weekly pipeline used by the github action. Exits cleanly during the offseason.             |

## Evaluating the model

    uv run nfl evaluate

This trains on seasons before a cutoff (default 2023) and scores predictions on the held-out seasons against two fixed benchmarks: a naive constant (always predicting the average home-field advantage) and the Vegas closing spread, which comes from the `spread_line` column already present in the nflverse play-by-play data. Use this to measure whether a model change actually helps — the training MAE printed by `train` mixes eras and is not a fair benchmark.

## Automation

`.github/workflows/train-spread-predictor.yaml` runs `nfl run-pipeline` every Tuesday during the season: it fetches upcoming matchups (exiting early if there are none), downloads the latest data (cached between runs), trains a fresh model, predicts the matchups, and publishes everything to the [spread-predictor release](https://github.com/BlairCurrey/nfl-analytics/releases). It can also be triggered manually from the Actions tab.

## Development

Run the tests with:

    uv run pytest
