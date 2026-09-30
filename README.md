# About

This repository contains a python cli application for predicting nfl spreads. The app trains a model from the latest available data and predicts upcoming matchups. In addition to running locally, a github action runs the full pipeline weekly during the season and publishes each week's predictions, before kickoff, as a [release](https://github.com/BlairCurrey/nfl-analytics/releases). The latest predictions are always at https://github.com/BlairCurrey/nfl-analytics/releases/latest/download/predictions.json.

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

This downloads all missing play-by-play data to `./nfl_analytics/data` and trains a model. Every training run is saved as a self-contained directory under `./nfl_analytics/assets/runs/<run_id>/` containing the model (as plain JSON parameters), running averages, and a manifest recording what produced it (git SHA, data hashes, library versions) and its benchmark results. Commands that need a model always load the latest complete run (or a specific one via `--run`), so artifacts from different training runs are never mixed.

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
| `benchmark`        | Run the fixed backtest and check it against the committed `benchmark.json` (`--write` to update it). |
| `run-pipeline`     | Full weekly pipeline used by the github action. Exits cleanly during the offseason.             |

## Evaluating the model

    uv run nfl evaluate

This trains on seasons before a cutoff (default 2023) and scores predictions on the held-out seasons against two fixed benchmarks: a naive constant (always predicting the average home-field advantage) and the Vegas closing spread, which comes from the `spread_line` column already present in the nflverse play-by-play data. Use this to explore whether a model change helps. Judge changes by paired comparisons over many games: a single week is far too few to tell two models apart.

### The committed benchmark

    uv run nfl benchmark

`benchmark` runs the same evaluation on a fixed window (trained on seasons before 2023, tested on 2023–2025) and compares the result with `nfl_analytics/benchmark.json`. Because the window never moves, the same code, data and dependencies always produce the same numbers, so the file works as a snapshot test for the whole recipe:

- When you change the model, features or dependencies on purpose, run `uv run nfl benchmark --write` and commit the updated `benchmark.json`. The change in accuracy then shows up as a diff in review. The benchmark workflow fails any push or pull request where the file is stale.
- The weekly pipeline recomputes the benchmark before publishing. If the numbers differ, the data or environment changed underneath the code, and the run stops.

The model that is actually published is trained on every season, with nothing held out.

## Automation

`.github/workflows/train-spread-predictor.yaml` runs every Tuesday during the season, with a Wednesday retry, and can also be triggered from the Actions tab. It:

1. Runs the tests.
2. Fetches the next week that still has unplayed games (exiting cleanly in the offseason).
3. Downloads the latest play-by-play data and validates it: every season present and complete, the current season up to date, known team names.
4. Recomputes the benchmark and stops if it doesn't match `benchmark.json`.
5. Trains on every season, predicts the matchups, and records the current Vegas line for each game (from nflverse's schedule file).
6. Sanity-checks the predictions: kickoffs in the next 8 days, no team playing twice, plausible spreads, and not unusually far from the Vegas lines on average.
7. Grades previously published predictions against final scores and Vegas closing lines, and carries the full history forward in `ledger.csv`. Comparing the line at publish time with the closing line shows whether the market moved toward the model afterwards.
8. Publishes an immutable release named `predictions-<season>-w<week>`, with checksums and build provenance attestations. An existing release is never replaced, so if Tuesday already published the week, Wednesday's retry does nothing.

If any step fails, nothing is published and the workflow opens (or comments on) a `pipeline-failure` issue.

To redo a bad release, delete it (and its tag) and re-run the workflow before kickoff.

## Website

The GitHub Pages site at https://blaircurrey.github.io/nfl-analytics/ shows this week's predictions, the benchmark, and the graded track record. It's a single static page (`site/index.html`) that reads `data/ledger.csv` and `data/manifest.json`. The Pages workflow copies both from the latest release and redeploys after every weekly run, and whenever `site/` changes.

To preview it locally, copy those two files from a pipeline run (or the latest release) into `site/data/` and serve the folder:

    mkdir -p site/data && cp nfl_analytics/assets/runs/<run_id>/{ledger.csv,manifest.json} site/data/
    python3 -m http.server -d site

## Development

Run the tests with:

    uv run pytest
