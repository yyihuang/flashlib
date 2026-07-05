from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from ._dispatch_runtime import capture_kernel_launches, detect_gpu_arch
from ._launch_plan import RouteDecision, resolve_route
from ._runtime import launch_context

SEMANTIC_ENTRYPOINT = "loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:launch_for_eval"
_PREPARE_LOCK = RLock()


@dataclass(frozen=True)
class PreparedKNNSearch:
    """Reusable tensors and a fully marshalled, stream-bound launch sequence."""

    inputs: dict[str, Any]
    launch_plan: RouteDecision
    direct_launcher: Any
    arch: str
    device_index: int
    stream: Any
    stream_handle: int
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.launch_plan.route_id

    @property
    def launch_count(self) -> int:
        return int(self.direct_launcher.launch_count)


def prepare_knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedKNNSearch:
    """Validate tensors and freeze one allocation-free direct launch sequence."""
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
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != device_index:
            raise ValueError(
                f"KNN-search stream device {stream_device_index} does not match input device {device_index}"
            )
        stream_handle = int(resolved_stream.cuda_stream)
        with torch.cuda.stream(resolved_stream), _PREPARE_LOCK:
            detected_arch = detect_gpu_arch()
            resolved_arch = detected_arch if arch is None else str(arch)
            if resolved_arch != detected_arch:
                raise ValueError(
                    "KNN-search launch arch must match the active device: "
                    f"requested {resolved_arch}, detected {detected_arch}"
                )
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
                "_knn_search_prepared_stream_key": (device_index, stream_handle),
            }
            launch_plan = resolve_route(inputs)
            with capture_kernel_launches(stream=resolved_stream, arch=resolved_arch) as captured:
                with launch_context(arch=resolved_arch, stream=resolved_stream, timeout_ms=None):
                    prepared_result = launch_plan.launch(
                        inputs,
                        stream=resolved_stream,
                        timeout_ms=None,
                    )
            _require_owned_outputs(prepared_result, inputs)
            direct_launcher = captured.bind(prepared_result)
    return PreparedKNNSearch(
        inputs=inputs,
        launch_plan=launch_plan,
        direct_launcher=direct_launcher,
        arch=resolved_arch,
        device_index=device_index,
        stream=resolved_stream,
        stream_handle=stream_handle,
        timeout_ms=timeout_ms,
    )


def _require_owned_outputs(outputs: Any, inputs: dict[str, Any]) -> None:
    """Reject prepared routes whose hot path would require an uncaptured copy."""

    owned = (inputs["out_distances"], inputs["out_indices"])
    if outputs is None:
        normalized = owned
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        normalized = (outputs[0], outputs[1])
    elif isinstance(outputs, dict):
        distances = outputs.get("distances", outputs.get("dists", outputs.get("out_distances")))
        indices = outputs.get("indices", outputs.get("idxs", outputs.get("out_indices")))
        if distances is None or indices is None:
            raise TypeError("knn_search dispatcher output dict must contain distances and indices")
        normalized = (distances, indices)
    else:
        raise TypeError("knn_search dispatcher must return (distances, indices), a matching dict, or write outputs")
    if any(source is not destination for destination, source in zip(owned, normalized, strict=True)):
        raise RuntimeError("prepared KNN-search route must write caller-owned outputs through captured launches")


def knn_search_prepared(
    prepared: PreparedKNNSearch,
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Submit a frozen exact KNN search without routing, allocation, or packing."""
    if not isinstance(prepared, PreparedKNNSearch):
        raise TypeError("prepared must be returned by prepare_knn_search")
    import torch

    if arch is not None and str(arch) != prepared.arch:
        raise ValueError(
            f"prepared KNN-search route targets {prepared.arch}; requested incompatible arch {arch}"
        )
    with torch.cuda.device(prepared.device_index):
        requested_stream = prepared.stream if stream is None else stream
        requested_handle = int(requested_stream.cuda_stream)
        if requested_handle != prepared.stream_handle:
            raise RuntimeError(
                "prepared KNN-search route is stream-bound: "
                f"prepared on stream 0x{prepared.stream_handle:x}, requested 0x{requested_handle:x}; "
                "prepare a separate plan inside the target torch.cuda.stream(...) context"
            )
        effective_timeout_ms = prepared.timeout_ms if timeout_ms is None else timeout_ms
        prepared.direct_launcher(
            prepared.inputs,
            stream=None,
            timeout_ms=effective_timeout_ms,
        )
    out = (prepared.inputs["out_distances"], prepared.inputs["out_indices"])
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": prepared.launch_plan.launch_entrypoint,
        "exact_launch_plan": prepared.launch_plan.exact_contract,
        "prepared_launch_count": prepared.launch_count,
        "arch": prepared.arch,
        "device_index": prepared.device_index,
        "stream_handle": prepared.stream_handle,
    }
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
    prepared = prepare_knn_search(
        query,
        database,
        k,
        out=out,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return knn_search_prepared(prepared, arch=arch, return_info=return_info)
