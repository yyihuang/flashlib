from __future__ import annotations

from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _dispatcher

SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"


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
    """Assign points to centroids with the frozen production dispatcher."""
    import torch

    del arch, stream, timeout_ms
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
    if out is None:
        out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
    if tuple(out.shape) != (bsz, n_points) or out.dtype is not torch.int32:
        raise ValueError(f"out must be int32 with shape {(bsz, n_points)}")
    x_sq = (x.float() ** 2).sum(-1).contiguous() if x_sq is None else x_sq
    c_sq = (centroids.float() ** 2).sum(-1).contiguous() if c_sq is None else c_sq
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
    result = _dispatcher.launch_for_eval(inputs)
    selected_route = _dispatcher.select_route(inputs).route_id
    cluster_ids = result.get("cluster_ids", out) if isinstance(result, dict) else out
    info = {"semantic_entrypoint": SEMANTIC_ENTRYPOINT, "selected_route": selected_route}
    return (cluster_ids, info) if return_info else cluster_ids
