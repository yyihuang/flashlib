"""Round-48 high-Q mid-M K1/K2 plus K5-merge dispatcher for BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 high-Q MMA routes. This
additive dispatcher preserves the round-46 K1/K2 split-9 routes, adds an
explicit ``B=1, Q=4096, M=20000, D=128, K=5`` route, and delegates all guard
misses to the round-46 high-Q dispatcher. The K5 route keeps the inherited
tcgen05 split-4 producer but specializes the final merge to emit five ranks
while reading the producer's stride-10 partial lists.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_highq_midm_k1k2partial_split9_dispatch_0613_r46_48e9_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_PARTIAL_STRIDE = mma.K_MAX
K5_OUT_MAX = 5
Q4096_ROWS = parent.Q4096_ROWS
Q4096_LOWK_M = parent.Q4096_LOWK_M
Q4096_LOWK_K5_SPLIT_M = 4
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:partial_k1_ir"}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:merge_k1_ir"}'))
partial_k2_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:partial_k2_ir"}'))
merge_k2_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:merge_k2_ir"}'))
partial_k5_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:partial_k5_ir"}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:parent_ir"}'))
_LOWK_K5_MERGE_KERNELS: dict[str, Any] = {}
knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1 = _ir_proxy('loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1', 256)
merge_k5_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:merge_k5_ir"}'))
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1:ir"}'))
Q4096_K5_SHAPE: dict[str, Any] = {'label': 'ksweep_q4096_m20000_d128_k5', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 5, 'dtype': 'bfloat16', 'seed': 610314, 'self_search': False, 'min_recall': 0.999}}
Q4096_K5_SHAPES: list[dict[str, Any]] = [Q4096_K5_SHAPE]
Q4096_K1K2_LABELS: tuple[str, ...] = parent.Q4096_LOWK_LABELS
HIGHQ_MIDM_K1K2_LABELS: tuple[str, ...] = parent.HIGHQ_MIDM_K1K2_LABELS
Q4096_K1K2_SHAPES = _decode_capture(_json_loads('[{"label": "ksweep_q4096_m20000_d128_k1", "params": {"B": 1, "D": 128, "K": 1, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610310, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k2", "params": {"B": 1, "D": 128, "K": 2, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610311, "self_search": false}}]'))
HIGHQ_MIDM_K1K2_SHAPES = parent.HIGHQ_MIDM_K1K2_SHAPES
HIGHQ_MIDM_K1K2K5_SHAPES: list[dict[str, Any]] = [*HIGHQ_MIDM_K1K2_SHAPES, *Q4096_K5_SHAPES]
ROUND48_PRESERVE_SHAPES = parent.ROUND46_PRESERVE_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q4096_lowk_k5_stride10_merge', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05', 'route': 'round48_q4096_lowk_k5_stride10_merge'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k5(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K5_OUT_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k5(inputs):
        return 'round48_q4096_lowk_k5_stride10_merge'
    return parent.selected_route(inputs)

def _compile_k5_merge_kernel() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k5": {"__kernel__": "dispatch_kernel_0281"}}'))

def _launch_q4096_k5(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    if not _LOWK_K5_MERGE_KERNELS:
        _LOWK_K5_MERGE_KERNELS.update(_compile_k5_merge_kernel())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_LOWK_K5_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    _LOWK_K5_MERGE_KERNELS['merge_k5'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    torch.cuda.synchronize()
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k5(inputs):
        return _launch_q4096_k5(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_k5(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_K5_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_highq_midm_k1k2k5(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGHQ_MIDM_K1K2K5_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round48_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND48_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
