from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._direct_plan import PreparedDirectRoute, prepare_route

SEMANTIC_ENTRYPOINT = (
    "loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:launch_from_contract_inputs"
)


@dataclass(frozen=True)
class PreparedKNNBuild:
    """Fixed inputs and a direct route resolved once outside the hot path."""

    inputs: dict[str, Any]
    launch_plan: PreparedDirectRoute
    shape_label: str | None
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.launch_plan.route_id


def _prepare_inputs(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool,
    shape_label: str | None,
    out: tuple[Any, Any] | None,
) -> dict[str, Any]:
    """Validate the public ABI and allocate one fixed set of intermediates."""
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
    if (db_bsz, db_dim) != (bsz, dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
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
    if out_dists.device != database.device or out_indices.device != database.device:
        raise ValueError("out tensors must be on the query/database device")
    if not out_dists.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    query_sq = (query.float() ** 2).sum(-1).contiguous()
    database_sq = query_sq if build else (database.float() ** 2).sum(-1).contiguous()
    return {
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


def prepare_knn_build(
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
) -> PreparedKNNBuild:
    """Prepare norms, outputs, scratch, and a fully marshalled direct leaf."""

    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != int(device_index):
            raise ValueError(
                f"KNN-build stream device {stream_device_index} does not match input device {device_index}"
            )
        with torch.cuda.stream(resolved_stream):
            inputs = _prepare_inputs(query, database, k, build=build, shape_label=shape_label, out=out)
            launch_plan = prepare_route(inputs, arch=arch, stream=resolved_stream)
    return PreparedKNNBuild(
        inputs=inputs,
        launch_plan=launch_plan,
        shape_label=launch_plan.shape_label,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def knn_build_prepared(
    prepared: PreparedKNNBuild,
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Launch a fixed route without re-entering the parent dispatcher or importer."""
    if not isinstance(prepared, PreparedKNNBuild):
        raise TypeError("prepared must be returned by prepare_knn_build")
    plan = prepared.launch_plan
    if arch is not None and str(arch) != plan.arch:
        raise ValueError(f"prepared KNN-build route targets {plan.arch}, requested incompatible arch {arch}")
    plan.launch(
        prepared.inputs,
        stream=stream,
        timeout_ms=prepared.timeout_ms if timeout_ms is None else timeout_ms,
    )
    out = (prepared.inputs["out_dists"], prepared.inputs["out_indices"])
    if not return_info:
        return out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": plan.launch_entrypoint,
        "exact_launch_plan": plan.exact_contract,
        "shape_label": prepared.shape_label,
        "prepared_launch_count": plan.launch_count,
        "arch": plan.arch,
        "device_index": plan.device_index,
        "stream_handle": plan.stream_handle,
    }
    return out, info


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
    """Run one direct frozen KNN build/search route."""
    prepared = prepare_knn_build(
        query,
        database,
        k,
        build=build,
        shape_label=shape_label,
        out=out,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return knn_build_prepared(
        prepared,
        arch=arch,
        stream=prepared.stream,
        timeout_ms=timeout_ms,
        return_info=return_info,
    )
