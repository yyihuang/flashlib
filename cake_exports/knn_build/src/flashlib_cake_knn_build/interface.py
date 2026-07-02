from __future__ import annotations

from typing import Any

from ._dispatch import knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1 as _dispatcher

SEMANTIC_ENTRYPOINT = ("loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:"
                       "launch_from_contract_inputs")


def knn_build(database: Any, k: int, *, out: tuple[Any, Any] | None = None,
              arch: str | None = None, stream: Any = None,
              timeout_ms: float | None = None, return_info: bool = False):
    """Build an exact self-kNN graph with the frozen production dispatcher."""
    import torch

    del arch, stream, timeout_ms
    if not isinstance(database, torch.Tensor) or not database.is_cuda:
        raise TypeError("database must be a CUDA torch.Tensor")
    if database.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("database dtype must be bfloat16 or float16")
    if database.ndim != 3 or not database.is_contiguous():
        raise ValueError("database must be contiguous with shape [B, N, D]")
    bsz, rows, dim = map(int, database.shape)
    k = int(k)
    if not 0 < k <= rows:
        raise ValueError(f"k must be in [1, {rows}], got {k}")
    expected = (bsz, rows, k)
    if out is None:
        out = (torch.empty(expected, dtype=torch.float32, device=database.device),
               torch.empty(expected, dtype=torch.int32, device=database.device))
    out_dists, out_indices = out
    if tuple(out_dists.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    if out_dists.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out must be (float32 distances, int32 indices)")
    sq = (database.float() ** 2).sum(-1).contiguous()
    inputs = {"B": bsz, "Q": rows, "M": rows, "D": dim, "K": k,
              "dtype": str(database.dtype).removeprefix("torch."), "build": True,
              "query": database, "database": database, "query_sq": sq,
              "database_sq": sq, "out_dists": out_dists, "out_indices": out_indices}
    selected_route = _dispatcher.route_for_contract_inputs(inputs)
    _dispatcher.launch_from_contract_inputs(inputs)
    info = {"semantic_entrypoint": SEMANTIC_ENTRYPOINT, "selected_route": selected_route}
    return (out, info) if return_info else out
