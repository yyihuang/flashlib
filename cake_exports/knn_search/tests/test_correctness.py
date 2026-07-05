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

K11_CORRECTNESS_ONLY_SHAPE = {
    "B": 1,
    "Q": 4096,
    "M": 20000,
    "D": 128,
    "K": 11,
    "dtype": "bfloat16",
    "min_recall": 0.999,
    "seed": 610312,
    "self_search": False,
}


def test_knn_search_prepared_api_is_exported():
    from flashlib_cake_knn_search import knn_search, knn_search_prepared, prepare_knn_search

    assert callable(knn_search)
    assert callable(prepare_knn_search)
    assert callable(knn_search_prepared)


def test_prepared_api_rejects_uncaptured_output_copy() -> None:
    import flashlib_cake_knn_search.interface as interface

    owned_distances = object()
    owned_indices = object()
    returned_distances = object()
    returned_indices = object()
    inputs = {"out_distances": owned_distances, "out_indices": owned_indices}

    with pytest.raises(RuntimeError, match="caller-owned outputs"):
        interface._require_owned_outputs(
            {"distances": returned_distances, "indices": returned_indices},
            inputs,
        )


def test_prepared_api_accepts_in_place_dispatcher_outputs() -> None:
    import flashlib_cake_knn_search.interface as interface

    owned_distances = object()
    owned_indices = object()
    interface._require_owned_outputs(
        {"distances": owned_distances, "indices": owned_indices},
        {"out_distances": owned_distances, "out_indices": owned_indices},
    )


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
    assert result["exact_launch_plan"], result
    assert result["correct"], result


def test_exact_k11_public_and_prepared_outputs_match_reference():
    """Correctness-only guard: this shape has no same-session performance baseline."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_knn_search import knn_search, knn_search_prepared, prepare_knn_search

    shape = K11_CORRECTNESS_ONLY_SHAPE
    query, database = BENCHMARK._make_inputs(shape)
    _, reference_indices = BENCHMARK._reference_topk(query, database, int(shape["K"]))
    expected_shape = (int(shape["B"]), int(shape["Q"]), int(shape["K"]))

    def _outputs():
        return (
            torch.full(expected_shape, float("nan"), dtype=torch.float32, device=query.device),
            torch.full(expected_shape, -1, dtype=torch.int32, device=query.device),
        )

    public_owned = _outputs()
    public_outputs, public_info = knn_search(
        query,
        database,
        int(shape["K"]),
        out=public_owned,
        return_info=True,
    )
    prepared_owned = _outputs()
    prepared = prepare_knn_search(query, database, int(shape["K"]), out=prepared_owned)
    prepared_outputs, prepared_info = knn_search_prepared(prepared, return_info=True)
    torch.cuda.synchronize()

    for label, outputs, owned, info in (
        ("public", public_outputs, public_owned, public_info),
        ("prepared", prepared_outputs, prepared_owned, prepared_info),
    ):
        assert outputs[0] is owned[0], f"{label} distances did not preserve out= ownership"
        assert outputs[1] is owned[1], f"{label} indices did not preserve out= ownership"
        assert info["selected_route"] == "k64_prefix_to_k11"
        exact_distances = BENCHMARK._distances_for_indices(query, database, outputs[1])
        recall = BENCHMARK._recall(outputs[1], reference_indices)
        max_abs_dist_error = float((outputs[0] - exact_distances).abs().max().item())
        assert recall >= float(shape["min_recall"]), (label, recall)
        assert max_abs_dist_error <= 1.0e-2, (label, max_abs_dist_error)
