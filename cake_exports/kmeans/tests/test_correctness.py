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
    spec = importlib.util.spec_from_file_location("kmeans_benchmark", BENCHMARKS / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kmeans_matches_reference_on_smoke_shape():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    benchmark = _benchmark_module()
    row = benchmark.FLASH_KMEANS_SHAPES[0]
    result = benchmark._run_shape(
        row,
        arch=None,
        correctness=True,
        benchmark=False,
        reference_chunk_rows=128,
    )
    assert result["match_rate"] == 1.0, result
