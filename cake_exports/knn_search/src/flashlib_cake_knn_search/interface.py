from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._dispatch import knn_search_dispatch0701_k11_d128_guard_repair_v1 as _dispatcher

SEMANTIC_ENTRYPOINT = "loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:launch_for_eval"


@dataclass(frozen=True)
class PreparedKNNSearch:
    """Reusable validated inputs and output buffers for the hot path."""

    inputs: dict[str, Any]
    selected_route: str


def prepare_knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
) -> PreparedKNNSearch:
    """Validate tensors and prepare outputs and dispatch metadata once."""
    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (query, database)):
        raise TypeError("query and database must be CUDA torch.Tensor objects")
    if query.dtype is not torch.bfloat16 or database.dtype is not torch.bfloat16:
        raise TypeError("query and database dtype must be bfloat16")
    if query.ndim != 3 or database.ndim != 3 or not query.is_contiguous() or not database.is_contiguous():
        raise ValueError("query and database must be contiguous [B, rows, D] tensors")
    bsz, q_rows, dim = map(int, query.shape)
    db_bsz, m_rows, db_dim = map(int, database.shape)
    k = int(k)
    if (bsz, dim) != (db_bsz, db_dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    if not 0 < k <= m_rows:
        raise ValueError(f"k must be in [1, {m_rows}], got {k}")
    expected = (bsz, q_rows, k)
    if out is None:
        out = (
            torch.empty(expected, dtype=torch.float32, device=query.device),
            torch.empty(expected, dtype=torch.int32, device=query.device),
        )
    out_distances, out_indices = out
    if tuple(out_distances.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    inputs = {
        "B": bsz,
        "Q": q_rows,
        "M": m_rows,
        "D": dim,
        "K": k,
        "dtype": "bfloat16",
        "self_search": query.data_ptr() == database.data_ptr(),
        "queries": query,
        "database": database,
        "out_distances": out_distances,
        "out_indices": out_indices,
    }
    return PreparedKNNSearch(inputs=inputs, selected_route=_dispatcher.selected_route(inputs))


def knn_search_prepared(prepared: PreparedKNNSearch, *, return_info: bool = False):
    """Launch a prepared exact KNN search without setup or allocation."""
    if not isinstance(prepared, PreparedKNNSearch):
        raise TypeError("prepared must be returned by prepare_knn_search")
    _dispatcher.launch_for_eval(prepared.inputs)
    out = (prepared.inputs["out_distances"], prepared.inputs["out_indices"])
    info = {"semantic_entrypoint": SEMANTIC_ENTRYPOINT, "selected_route": prepared.selected_route}
    return (out, info) if return_info else out


def knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Run exact squared-L2 kNN with the frozen production dispatcher."""
    del arch, stream, timeout_ms
    prepared = prepare_knn_search(query, database, k, out=out)
    return knn_search_prepared(prepared, return_info=return_info)
