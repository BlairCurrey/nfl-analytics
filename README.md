# About

This repository contains a python cli application to train and use a model to predict the spread for upcoming NFL games. The model can be built and used for predictions locally using the cli app. In addition, a github action uses the cli app to train the model and publish the predictions to upcoming games as a [release found here](https://github.com/BlairCurrey/nfl-analytics/releases). The release contains other artifacts of the model building and prediction pipelines as well.

Visit the docs for [the model](./nfl_analytics/docs/model.md) and [training data](./nfl_analytics/docs/training-data.md) for more background.

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
