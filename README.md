# About

This repository contains a python cli application for predicting nfl spreads. The app trains a model from the latest available data and predicts upcoming matchups. In addition to running locally, a github action uses the cli app to train the model and publish the predictions for upcoming games to [this release](https://github.com/BlairCurrey/nfl-analytics/releases).

Visit the docs for [the model](./nfl_analytics/docs/model.md) and [training data](./nfl_analytics/docs/training-data.md) for more details on each.

This project exists for a few reasons:

- I wanted to see how well a simplistic model would do at predicting the spread. I suspected this is a situation where something like 20% of the work could get you 80% of the results (known as the [Pareto Principle](https://en.wikipedia.org/wiki/Pareto_principle)), with "results" being Vegas-like spread prediction accuracy. I think this ended up being the case. See [the model doc](./nfl_analytics/docs/model.md) for more details.
- I wanted to build an end-to-end training and prediction pipeline in github actions.
- I wanted to compile an insightful dataset from atomic NFL play-by-play data. See the [training data doc](./nfl_analytics/docs/training-data.md) for more details on this.

# Using

## Pre-requisites

- Python 3 (developed on 3.12)
- Poetry (developed on 1.7.1)

## Project Setup

Clone this repository to your local machine:

    git clone https://github.com/BlairCurrey/nfl-analytics.git

Navigate to the repository:

    cd nfl-analytics

Install dependencies with Poetry:

    poetry install

At this point you should be able to run the cli app:

    poetry run python nfl_analytics/main.py --help

## Model Setup

On initial setup, you will need to download all the data for training the model:

    poetry run python nfl_analytics/main.py --download

This downloads all the raw data required to train the model to `./nfl_analytics/data`.

Then you can train the model:

    poetry run python nfl_analytics/main.py --train

This builds and saves the training dataset and then trains the model. The model and scaler used in training and the training dataset are saved to `./nfl_analytics/assets`.

Now you can use the model to predict games.

## Predicting Games

Provide the home and away team (in that order) to `--predict` to predict a specific upcoming game:

    poetry run python nfl_analytics/main.py --predict kc sf

An exact list list of team abbreviations can be found in `./nfl_analytics/config.py`.

Alternatively, you can predict all upcoming games, which fetches upcoming matchups from an external api and saves the predictions to `./nfl_analytics/assets`:

    poetry run python nfl_analytics/main.py --predict-all

## Updating the Model

To update the model, you can re-download the latest year and re-train the model:

    poetry run python nfl_analytics/main.py --download 2024 --train
