from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._dispatch import knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1 as _dispatcher

SEMANTIC_ENTRYPOINT = (
    "loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:launch_from_contract_inputs"
)


@dataclass(frozen=True)
class PreparedKNNBuild:
    """Reusable inputs for the allocation- and preprocessing-free hot path."""

    inputs: dict[str, Any]
    selected_route: str
    shape_label: str | None


def prepare_knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
) -> PreparedKNNBuild:
    """Validate and prepare norms, outputs, and dispatch metadata once."""
    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    if not isinstance(database, torch.Tensor) or not database.is_cuda:
        raise TypeError("database must be a CUDA torch.Tensor")
    if query.dtype not in (torch.bfloat16, torch.float16) or database.dtype != query.dtype:
        raise TypeError("query and database must have the same bfloat16 or float16 dtype")
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("query must be contiguous with shape [B, Q, D]")
    if database.ndim != 3 or not database.is_contiguous():
        raise ValueError("database must be contiguous with shape [B, N, D]")
    bsz, n_query, dim = map(int, query.shape)
    db_bsz, n_database, db_dim = map(int, database.shape)
    if (db_bsz, db_dim) != (bsz, dim):
        raise ValueError("query and database batch/feature dimensions must match")
    if build and (n_query != n_database or query.data_ptr() != database.data_ptr()):
        raise ValueError("build=True requires query to alias database and Q == M")
    k = int(k)
    if not 0 < k <= n_database:
        raise ValueError(f"k must be in [1, {n_database}], got {k}")
    expected = (bsz, n_query, k)
    if out is None:
        out = (
            torch.empty(expected, dtype=torch.float32, device=database.device),
            torch.empty(expected, dtype=torch.int32, device=database.device),
        )
    out_dists, out_indices = out
    if tuple(out_dists.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    if out_dists.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out must be (float32 distances, int32 indices)")
    query_sq = (query.float() ** 2).sum(-1).contiguous()
    database_sq = query_sq if build else (database.float() ** 2).sum(-1).contiguous()
    inputs = {
        "label": shape_label,
        "B": bsz,
        "Q": n_query,
        "M": n_database,
        "D": dim,
        "K": k,
        "dtype": str(database.dtype).removeprefix("torch."),
        "build": bool(build),
        "query": query,
        "database": database,
        "query_sq": query_sq,
        "database_sq": database_sq,
        "out_dists": out_dists,
        "out_indices": out_indices,
    }
    return PreparedKNNBuild(
        inputs=inputs,
        selected_route=_dispatcher.route_for_contract_inputs(inputs),
        shape_label=shape_label,
    )


def knn_build_prepared(prepared: PreparedKNNBuild, *, return_info: bool = False):
    """Launch a prepared KNN build/search without allocation or norm setup."""
    if not isinstance(prepared, PreparedKNNBuild):
        raise TypeError("prepared must be returned by prepare_knn_build")
    _dispatcher.launch_from_contract_inputs(prepared.inputs)
    out = (prepared.inputs["out_dists"], prepared.inputs["out_indices"])
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "shape_label": prepared.shape_label,
    }
    return (out, info) if return_info else out


def knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Run the frozen exact kNN build/search production dispatcher."""
    del arch, stream, timeout_ms
    prepared = prepare_knn_build(
        query,
        database,
        k,
        build=build,
        shape_label=shape_label,
        out=out,
    )
    return knn_build_prepared(prepared, return_info=return_info)
