"""
The committed benchmark: a fixed backtest whose results live in
nfl_analytics/benchmark.json.

The window never moves on its own (see BENCHMARK_TEST_SEASONS), so the same
recipe on the same data always produces the same numbers. That makes the
committed file a snapshot test for the whole recipe:

- A pull request that changes the model, features, or dependencies must
  update benchmark.json, so the change in accuracy is reviewed as a diff.
- The weekly pipeline recomputes it before publishing. A mismatch means the
  data or environment changed underneath the code, and the run stops.

This judges the recipe, not one week's model: a single week has too few games
to tell two models apart.
"""

import json
import os
from typing import Any

import pandas as pd

from nfl_analytics.config import BENCHMARK_FILENAME, BENCHMARK_TEST_SEASONS
from nfl_analytics.evaluate import evaluate_spread_model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(THIS_DIR, BENCHMARK_FILENAME)

# Game counts must match exactly; metrics may differ by float noise only.
TOLERANCE = 1e-3


class BenchmarkMismatchError(Exception):
    pass


def compute_benchmark(
    df_training: pd.DataFrame, vegas_lines: pd.DataFrame
) -> dict[str, Any]:
    first, last = BENCHMARK_TEST_SEASONS
    results = evaluate_spread_model(
        df_training, vegas_lines, test_since=first, test_until=last
    )

    if results["test_until"] != last:
        raise ValueError(
            f"Benchmark needs every season through {last}; data only reaches "
            f"{results['test_until']}."
        )

    def r(x: float) -> float:
        return round(float(x), 4)

    return {
        "test_seasons": [first, last],
        "n_train_games": results["n_train_games"],
        "n_test_games": results["n_test_games"],
        "mae": {
            "naive": r(results["naive"]["mae"]),
            "model": r(results["model"]["mae"]),
            "vegas": r(results["vegas"]["mae"]),
        },
        "rmse": {
            "naive": r(results["naive"]["rmse"]),
            "model": r(results["model"]["rmse"]),
            "vegas": r(results["vegas"]["rmse"]),
        },
        "gap_closed": r(results["gap_closed"]),
        "ats_accuracy": r(results["ats_accuracy"]),
    }


def load_benchmark(path: str = BENCHMARK_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_benchmark(benchmark: dict[str, Any], path: str = BENCHMARK_PATH) -> None:
    with open(path, "w") as f:
        json.dump(benchmark, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def compare_benchmarks(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    """Human-readable differences; empty when the benchmarks match."""
    exp, act = _flatten(expected), _flatten(actual)
    problems = []

    for key in sorted(exp.keys() | act.keys()):
        e, a = exp.get(key), act.get(key)

        if isinstance(e, float) and isinstance(a, (int, float)):
            if abs(e - a) > TOLERANCE:
                problems.append(f"{key}: committed {e}, now {a} ({a - e:+.4f})")
        elif e != a:
            problems.append(f"{key}: committed {e}, now {a}")

    return problems


def check_benchmark(actual: dict[str, Any], path: str = BENCHMARK_PATH) -> None:
    problems = compare_benchmarks(load_benchmark(path), actual)

    if problems:
        raise BenchmarkMismatchError(
            "Benchmark doesn't match the committed benchmark.json:\n  "
            + "\n  ".join(problems)
            + "\nIf the recipe changed on purpose, run `nfl benchmark --write` "
            "and commit the result. Otherwise the data or dependencies "
            "changed underneath the code."
        )


def format_benchmark(
    actual: dict[str, Any], committed: dict[str, Any] | None = None
) -> str:
    """Markdown table of the benchmark, with the committed values alongside."""
    first, last = actual["test_seasons"]
    lines = [
        f"Benchmark: fit on seasons before {first}, tested on "
        f"{actual['n_test_games']} games from {first}–{last}",
        "",
    ]

    if committed:
        lines += ["| predictor | MAE | committed | change |", "| --- | --- | --- | --- |"]
        for name in ["naive", "model", "vegas"]:
            now, before = actual["mae"][name], committed["mae"][name]
            lines.append(f"| {name} | {now:.4f} | {before:.4f} | {now - before:+.4f} |")
    else:
        lines += ["| predictor | MAE |", "| --- | --- |"]
        for name in ["naive", "model", "vegas"]:
            lines.append(f"| {name} | {actual['mae'][name]:.4f} |")

    lines += [
        "",
        f"Naive-to-Vegas gap closed: {actual['gap_closed']:.1%}. "
        f"Against-the-spread accuracy: {actual['ats_accuracy']:.1%}.",
    ]
    return "\n".join(lines)
