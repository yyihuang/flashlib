from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("knn_search_benchmark", BENCHMARKS / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _benchmark_module()


def test_knn_search_prepared_api_is_exported():
    from flashlib_cake_knn_search import knn_search, knn_search_prepared, prepare_knn_search

    assert callable(knn_search)
    assert callable(prepare_knn_search)
    assert callable(knn_search_prepared)


@pytest.mark.parametrize("name", list(BENCHMARK.SHAPES))
def test_knn_search_matches_reference(name: str):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    result = BENCHMARK._run_shape(
        name,
        BENCHMARK.SHAPES[name],
        arch=None,
        correctness=True,
        benchmark=False,
    )
    assert result["route_matches_expected"], result
    assert result["correct"], result
