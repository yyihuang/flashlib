"""Round-5/9a85 Q4096/K64 prefix5 row-flag fused-cert route.

Minimum target architecture: sm_100a. This additive shape kernel retunes the
round-36/e4cb Q4096/K64 route from six local winners plus sentinel to five
local winners plus sentinel per producer list. It keeps the same fused
row-local certification and exact full-K64 Weave fallback on overflow, reducing
partial scratch traffic and merge list work when the prefix5 certificate holds.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from dataclasses import replace
import math
from typing import Any
from . import knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_0615_r36_e4cb_v1 as e4cb_parent
K64_MAX = e4cb_parent.K64_MAX
LOCAL_PREFIX_K = 5
K_PARTIAL_STRIDE = LOCAL_PREFIX_K + 1
Q4096_ROWS = e4cb_parent.Q4096_ROWS
Q4096_M_ROWS = e4cb_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = e4cb_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = e4cb_parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = e4cb_parent.MERGE10_SPLITS_PER_LANE_MAX
THREADS = e4cb_parent.THREADS
BLOCK_Q = e4cb_parent.BLOCK_Q
BLOCK_M = e4cb_parent.BLOCK_M
D_STATIC = e4cb_parent.D_STATIC
MERGE_THREADS = e4cb_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = e4cb_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = e4cb_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = e4cb_parent.MERGE_SMEM_BYTES
ROUTE_Q4096_M20000_K64_PREFIX5_ROWFLAG_FUSEDCERT = 'round5_9a85_q4096_m20000_k64_prefix5_rowflag_fusedcert'
K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_FLAGS: dict[tuple[int, int], Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS: dict[str, int | bool | None] = {'attempts': 0, 'certified_count': 0, 'fallback_count': 0, 'last_overflow': None}
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1:partial_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1:merge_ir"}'))
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1:ir"}'))

def _scratch_prefix5_rowflag_fusedcert(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_SCRATCH[key] = cached
    return cached

def _row_flags(inputs: dict[str, Any], row_count: int):
    import torch
    device_index = int(inputs['queries'].device.index or 0)
    key = (device_index, int(row_count))
    flags = _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_FLAGS.get(key)
    if flags is None:
        flags = torch.empty((int(row_count),), dtype=torch.int32, device=inputs['queries'].device)
        _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_FLAGS[key] = flags
    return flags

def _compile_k64_q4096_prefix5_rowflag_fusedcert_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0334"}, "partial": {"__kernel__": "dispatch_kernel_0333"}}'))

def _use_q4096_k64_prefix5_rowflag_fusedcert(inputs: dict[str, Any]) -> bool:
    return e4cb_parent._use_q4096_k64_rowflag_fusedcert(inputs)

def _launch_q4096_k64_prefix5_rowflag_fusedcert(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_KERNELS:
        _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_KERNELS.update(_compile_k64_q4096_prefix5_rowflag_fusedcert_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch_prefix5_rowflag_fusedcert(inputs, partial_list_count, num_q_tiles)
    overflow_rows = _row_flags(inputs, bsz * q_rows)
    _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], overflow_rows, bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['attempts'] = int(_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['attempts'] or 0) + 1
    overflow = bool(int(overflow_rows.cpu().max().item()))
    _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['last_overflow'] = overflow
    if overflow:
        _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['fallback_count'] = int(_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['fallback_count'] or 0) + 1
        return e4cb_parent.exact_parent._launch_q4096_k64_split79_oddevensort_fastmerge(inputs)
    _KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['certified_count'] = int(_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS['certified_count'] or 0) + 1
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_prefix5_rowflag_fusedcert(inputs):
        return _launch_q4096_k64_prefix5_rowflag_fusedcert(inputs)
    return e4cb_parent.exact_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return e4cb_parent._select_contract_shapes(shape_labels)

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q4096_k64_prefix5_rowflag_fusedcert(inputs):
        return ROUTE_Q4096_M20000_K64_PREFIX5_ROWFLAG_FUSEDCERT
    return e4cb_parent.selected_route_name(inputs)

def stats() -> dict[str, int | bool | None]:
    return dict(_KNN_SEARCH_K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_STATS)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_prefix5_rowflag_fusedcert(inputs):
        return {'route': ROUTE_Q4096_M20000_K64_PREFIX5_ROWFLAG_FUSEDCERT, 'selected_route': ROUTE_Q4096_M20000_K64_PREFIX5_ROWFLAG_FUSEDCERT, 'selected_entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64', 'certification_policy': 'prefix5_plus_sentinel_fused_merge_cert_with_row_flags_and_weave_full_k64_fallback_on_overflow', 'fallback_entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'}
    if hasattr(e4cb_parent, 'route_info'):
        return e4cb_parent.route_info(inputs)
    return {'route': 'parent'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def knn_search_compile_and_launch_k64_q4096_prefix5_rowflag_fusedcert(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_PREFIX5_ROWFLAG_FUSEDCERT_9A85_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
