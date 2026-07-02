"""Round-20/576b Q4096/M20000/K64 fused prefix6-cert route.

Minimum target architecture: sm_100a. This additive shape kernel keeps the
round-20/245d prefix6 certified tcgen05 producer and exact full-K64 Weave
fallback, but folds the omitted-sentinel certification pass into the prefix
merge kernel. The runtime still initializes and reads one device overflow flag
so unsafe rows fall back to the exact split79 route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
import math
from typing import Any
from . import knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1 as prefix6_parent
K64_MAX = prefix6_parent.K64_MAX
LOCAL_PREFIX_K = prefix6_parent.LOCAL_PREFIX_K
K_PARTIAL_STRIDE = prefix6_parent.K_PARTIAL_STRIDE
Q4096_ROWS = prefix6_parent.Q4096_ROWS
Q4096_M_ROWS = prefix6_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = prefix6_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = prefix6_parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = prefix6_parent.MERGE10_SPLITS_PER_LANE_MAX
THREADS = prefix6_parent.THREADS
BLOCK_Q = prefix6_parent.BLOCK_Q
BLOCK_M = prefix6_parent.BLOCK_M
D_STATIC = prefix6_parent.D_STATIC
MERGE_THREADS = prefix6_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = prefix6_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = prefix6_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = prefix6_parent.MERGE_SMEM_BYTES
ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED = 'round20_576b_q4096_m20000_k64_prefix6cert_fused'
K64_Q4096_PREFIXCERT_FUSED_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS: dict[int, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS: dict[str, int | bool | None] = {'attempts': 0, 'certified_count': 0, 'fallback_count': 0, 'last_overflow': None}
knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1 = _ir_proxy('loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1', 256)
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:partial_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:merge_ir"}'))
certflag_init_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:certflag_init_ir"}'))
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:ir"}'))

def _scratch_prefixcert_fused(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH[key] = cached
    return cached

def _cert_flag_fused(inputs: dict[str, Any]):
    import torch
    device_index = int(inputs['queries'].device.index or 0)
    flag = _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS.get(device_index)
    if flag is None:
        flag = torch.empty((1,), dtype=torch.int32, device=inputs['queries'].device)
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS[device_index] = flag
    return flag

def _compile_k64_q4096_prefixcert_fused_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"certflag_init": {"__kernel__": "dispatch_kernel_0355"}, "merge": {"__kernel__": "dispatch_kernel_0354"}, "partial": {"__kernel__": "dispatch_kernel_0353"}}'))

def _use_q4096_k64_prefixcert_fused(inputs: dict[str, Any]) -> bool:
    return prefix6_parent._use_q4096_k64_prefix6cert(inputs)

def _launch_q4096_k64_prefixcert_fused(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS:
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS.update(_compile_k64_q4096_prefixcert_fused_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch_prefixcert_fused(inputs, partial_list_count, num_q_tiles)
    overflow_flag = _cert_flag_fused(inputs)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['certflag_init'].launch(grid=(1, 1, 1), block=(MERGE_THREADS, 1, 1), args=[overflow_flag], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], overflow_flag, bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] or 0) + 1
    overflow = bool(int(overflow_flag.item()))
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['last_overflow'] = overflow
    if overflow:
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['fallback_count'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['fallback_count'] or 0) + 1
        return prefix6_parent.parent.exact_parent._launch_q4096_k64_split79_oddevensort_fastmerge(inputs)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['certified_count'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['certified_count'] or 0) + 1
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_prefixcert_fused(inputs):
        return _launch_q4096_k64_prefixcert_fused(inputs)
    return prefix6_parent.parent.exact_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return prefix6_parent.parent._select_contract_shapes(shape_labels)

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q4096_k64_prefixcert_fused(inputs):
        return ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED
    return prefix6_parent.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route_name(inputs)
    if route == ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64', 'certification_policy': 'prefix6_plus_sentinel_fused_merge_with_weave_full_k64_fallback_on_overflow', 'fallback_entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'}
    return prefix6_parent.route_info(inputs)

def reset_stats() -> None:
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['certified_count'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['fallback_count'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['last_overflow'] = None

def knn_search_compile_and_launch_k64_q4096_prefixcert_fused(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_PREFIXCERT_FUSED_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
