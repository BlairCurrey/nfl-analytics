import pytest

from nfl_analytics.benchmark import (
    BenchmarkMismatchError,
    check_benchmark,
    compare_benchmarks,
    format_benchmark,
    write_benchmark,
)

BENCHMARK = {
    "test_seasons": [2023, 2025],
    "n_train_games": 6400,
    "n_test_games": 854,
    "mae": {"naive": 11.1743, "model": 10.2847, "vegas": 9.8179},
    "rmse": {"naive": 14.3595, "model": 13.2209, "vegas": 12.7313},
    "gap_closed": 0.6559,
    "ats_accuracy": 0.4922,
}


def with_changes(**changes):
    benchmark = {**BENCHMARK, "mae": dict(BENCHMARK["mae"])}
    for key, value in changes.items():
        if key.startswith("mae_"):
            benchmark["mae"][key[4:]] = value
        else:
            benchmark[key] = value
    return benchmark


def test_identical_benchmarks_match():
    assert compare_benchmarks(BENCHMARK, with_changes()) == []


def test_float_noise_is_tolerated():
    assert compare_benchmarks(BENCHMARK, with_changes(mae_model=10.2849)) == []


def test_metric_change_is_reported():
    problems = compare_benchmarks(BENCHMARK, with_changes(mae_model=10.31))

    assert len(problems) == 1
    assert problems[0].startswith("mae.model: committed 10.2847, now 10.31")


def test_game_count_change_is_reported():
    problems = compare_benchmarks(BENCHMARK, with_changes(n_test_games=853))

    assert problems == ["n_test_games: committed 854, now 853"]


def test_check_benchmark_raises_on_mismatch(tmp_path):
    path = str(tmp_path / "benchmark.json")
    write_benchmark(BENCHMARK, path)

    check_benchmark(with_changes(), path)

    with pytest.raises(BenchmarkMismatchError, match="nfl benchmark --write"):
        check_benchmark(with_changes(mae_model=10.5), path)


def test_format_benchmark_shows_change_against_committed():
    table = format_benchmark(with_changes(mae_model=10.2), BENCHMARK)

    assert "| model | 10.2000 | 10.2847 | -0.0847 |" in table
    assert "854 games from 2023–2025" in table
