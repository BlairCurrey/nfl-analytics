DATA_DIR = "data"
ASSET_DIR = "assets"
START_YEAR = 1999
# Early-season running averages blend in this many pseudo-games at the team's
# previous-season mean, decaying as real games accumulate. Chosen empirically
# via `nfl evaluate` (see .claude/improvements.md).
PRIOR_PSEUDO_GAMES = 4
# Fixed backtest window for `nfl benchmark`: fit on seasons before the first,
# score on these completed seasons. The results are committed to
# BENCHMARK_FILENAME, so any change to the recipe (code, data, dependencies)
# shows up as a diff. Move the window only deliberately, in its own commit.
BENCHMARK_TEST_SEASONS = (2023, 2025)
BENCHMARK_FILENAME = "benchmark.json"
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.json"
RUNNING_AVG_FILENAME = "running_average.csv.gz"
MATCHUPS_FILENAME = "matchups.json"
PREDICTIONS_FILENAME = "predictions.json"
LEDGER_FILENAME = "ledger.csv"
# `nfl backtest` / the weekly run replay every week from this season on
BACKTEST_SINCE = 2021
BACKTEST_FILENAME = "backtest.csv"
RELEASE_NOTES_FILENAME = "release_notes.md"
FEATURES = [
    "away_rushing_avg",
    "home_rushing_avg",
    "away_passing_avg",
    "home_passing_avg",
    "away_sack_yards_avg",
    "home_sack_yards_avg",
    "away_score_differential_post_avg",
    "home_score_differential_post_avg",
    "away_points_scored_avg",
    "home_points_scored_avg",
    "away_points_allowed_avg",
    "home_points_allowed_avg",
    "away_mean_epa_avg",
    "home_mean_epa_avg",
]
TEAMS = [
    "WAS",
    "ARI",
    "BUF",
    "NYJ",
    "ATL",
    "CAR",
    "CIN",
    "CLE",
    "NYG",
    "DAL",
    "DET",
    "KC",
    "CHI",
    "GB",
    "BAL",
    "HOU",
    "IND",
    "JAX",
    "SEA",
    "LA",
    "LV",
    "DEN",
    "MIA",
    "LAC",
    "PHI",
    "NE",
    "PIT",
    "SF",
    "MIN",
    "TB",
    "NO",
    "TEN",
]

TEAM_ABBR_MAP = {
    "LAR": "LA",
    "WSH": "WAS",
}
