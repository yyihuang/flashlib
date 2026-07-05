from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _dispatcher
from ._runtime import launch_context

SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"


@dataclass(frozen=True)
class PreparedFlashKMeansAssign:
    """Exact tensors, workspace, output, and direct launch plan for KMeans."""

    inputs: dict[str, Any]
    launch_plan: Any
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def out(self) -> Any:
        return self.inputs["out"]

    @property
    def selected_route(self) -> str:
        return self.launch_plan.selected_route


def _require_aux_tensor(
    tensor: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype is not dtype:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def prepare_flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedFlashKMeansAssign:
    """Validate and prepare an allocation-free direct launch for exact tensors."""

    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
        raise TypeError("x and centroids must be CUDA torch.Tensor objects")
    if x.dtype is not torch.bfloat16 or centroids.dtype is not torch.bfloat16:
        raise TypeError("x and centroids dtype must be bfloat16")
    if x.ndim != 3 or centroids.ndim != 3 or not x.is_contiguous() or not centroids.is_contiguous():
        raise ValueError("x and centroids must be contiguous [B, rows, D] tensors")
    bsz, n_points, dim = map(int, x.shape)
    c_bsz, n_clusters, c_dim = map(int, centroids.shape)
    if (bsz, dim) != (c_bsz, c_dim) or x.device != centroids.device:
        raise ValueError("x and centroids batch/feature dimensions and device must match")
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != int(device_index):
            raise ValueError(
                f"Flash-KMeans stream device {stream_device_index} does not match input device {device_index}"
            )
        with torch.cuda.stream(resolved_stream):
            if out is None:
                out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
            _require_aux_tensor(
                out,
                name="out",
                shape=(bsz, n_points),
                dtype=torch.int32,
                device=x.device,
            )
            x_sq = (x.float() ** 2).sum(-1).contiguous() if x_sq is None else x_sq
            c_sq = (centroids.float() ** 2).sum(-1).contiguous() if c_sq is None else c_sq
            _require_aux_tensor(
                x_sq,
                name="x_sq",
                shape=(bsz, n_points),
                dtype=torch.float32,
                device=x.device,
            )
            _require_aux_tensor(
                c_sq,
                name="c_sq",
                shape=(bsz, n_clusters),
                dtype=torch.float32,
                device=x.device,
            )
            inputs = {
                "B": bsz,
                "N": n_points,
                "D": dim,
                "K": n_clusters,
                "dtype": "bfloat16",
                "x": x,
                "centroids": centroids,
                "x_sq": x_sq,
                "c_sq": c_sq,
                "out": out,
            }
            launch_plan = _dispatcher.prepare_launch_plan(
                inputs,
                arch=arch,
                stream=resolved_stream,
                timeout_ms=timeout_ms,
            )
    return PreparedFlashKMeansAssign(
        inputs=inputs,
        launch_plan=launch_plan,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def flash_kmeans_assign_prepared(
    prepared: PreparedFlashKMeansAssign,
    *,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Submit a prepared plan without route selection, allocation, or packing."""

    if not isinstance(prepared, PreparedFlashKMeansAssign):
        raise TypeError("prepared must be returned by prepare_flash_kmeans_assign")
    result = _dispatcher.launch_prepared(
        prepared.launch_plan,
        stream=stream,
        timeout_ms=prepared.timeout_ms if timeout_ms is None else timeout_ms,
    )
    produced = result.get("cluster_ids", prepared.out) if isinstance(result, dict) else prepared.out
    if produced is not prepared.out:
        prepared.out.copy_(produced)
    if not return_info:
        return prepared.out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": prepared.launch_plan.launch_entrypoint,
        "exact_launch_plan": True,
        "prepared_launch_count": prepared.launch_plan.launch_count,
        "arch": prepared.launch_plan.arch,
        "device_index": prepared.launch_plan.device_index,
        "stream_handle": prepared.launch_plan.stream_handle,
    }
    return prepared.out, info


def flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Assign points to centroids through a cold prepared/direct launch."""
    with launch_context(arch=arch, stream=stream, timeout_ms=timeout_ms):
        prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return flash_kmeans_assign_prepared(
            prepared,
            stream=prepared.stream,
            return_info=return_info,
        )
