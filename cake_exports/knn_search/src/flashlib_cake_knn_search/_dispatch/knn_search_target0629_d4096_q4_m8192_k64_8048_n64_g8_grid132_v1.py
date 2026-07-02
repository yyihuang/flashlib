"""Exact D4096/Q4/M8192/K64 N64 tcgen05 seed with an 8-way merge.

Minimum target architecture: sm_100a.  This additive variant keeps the
validated 132-CTA native-N64 tcgen05 producer, but halves the intermediate
group merge fanout before the contract-visible final top-64 merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target0629_d4096_q4_m8192_k64_696a_n64_g16_grid132_v1 as parent
PRODUCER_GRID = parent.PRODUCER_GRID
THREADS = parent.native.THREADS
MERGE_THREADS = 32
BLOCK_Q = parent.native.BLOCK_Q
K64_MAX = parent.native.K64_MAX
HIERMERGE_GROUPS = 8
SMEM_BYTES = parent.native.SMEM_BYTES
MERGE_SMEM_BYTES = parent.native.MERGE_SMEM_BYTES
ROUTE = '8048_target0629_d4096_q4_m8192_k64_native_n64_g8_grid132_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_8048_n64_g8_grid132_v1:launch_for_eval'
TARGET_SHAPES = parent.TARGET_SHAPES
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_8048_n64_g8_grid132_v1:partial_ir"}'))
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_8048_n64_g8_grid132_v1:ir"}'))
_KERNELS: dict[str, Any] | None = None
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
group_merge_ir = _ir_proxy('loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_8048_n64_g8_grid132_v1:group_merge_ir', 256)
final_merge_ir = _ir_proxy('loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_8048_n64_g8_grid132_v1:final_merge_ir', 256)

def _active(inputs: dict[str, Any]) -> bool:
    return parent._active(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return {**parent.route_info(inputs), 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'coverage_class': 'bucket_seed_target0627_d4096_q4_m8192_k64_native_n64_g8_grid132'}
    return parent.route_info(inputs)

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0475"}, "group": {"__kernel__": "dispatch_kernel_0474"}, "partial": {"__kernel__": "dispatch_kernel_0473"}}'))

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _compile_kernels()
    key = (id(inputs['queries']), str(inputs['queries'].dtype))
    partial = _SCRATCH.get(key)
    if partial is None:
        partial = (torch.empty((1, 1, PRODUCER_GRID, BLOCK_Q, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 1, PRODUCER_GRID, BLOCK_Q, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = partial
    groups = _GROUP_SCRATCH.get(key)
    if groups is None:
        groups = (torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = groups
    _KERNELS['partial'].launch(grid=(PRODUCER_GRID, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial[0], partial[1], 1, 4, 8192, PRODUCER_GRID, 1], shared_mem=SMEM_BYTES)
    _KERNELS['group'].launch(grid=(4 * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial[0], partial[1], groups[0], groups[1], 1, 4, PRODUCER_GRID, 1], shared_mem=MERGE_SMEM_BYTES)
    _KERNELS['final'].launch(grid=(4, 1, 1), block=(MERGE_THREADS, 1, 1), args=[groups[0], groups[1], inputs['out_distances'], inputs['out_indices'], 1, 4, 64], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _active(inputs) else parent.launch_for_eval(inputs)
