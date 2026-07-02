"""Selective Q2/Q4/Q7 Block-M640 seed wrapper for exact BF16 kNN.

Minimum target architecture: sm_80 for the Block-M640 tile-reduce route.
Inherited guard-miss paths keep their own architecture requirements. This
additive candidate routes exact ``B=1,Q in {2,4,7},M=131072,D=128,K=10`` rows
through the measured Block-M640 Weave seed and delegates Q3 plus all other
shapes to the incumbent trunk dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as blockm640
from . import knn_search_mma_split_v1 as incumbent
THREADS = blockm640.THREADS
MERGE_THREADS = blockm640.MERGE_THREADS
BLOCK_M = blockm640.BLOCK_M
D_STATIC = blockm640.D_STATIC
K_MAX = blockm640.K_MAX
ROUTED_M = 131072
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:ir"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:partial_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:merge_ir"}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:parent_ir"}'))
ROUTE_LOWQ_Q247_BLOCKM640 = 'round103_6f2a_lowq_q2q4q7_blockm640_exact_m131072'
ROUTE_INCUMBENT = 'current_trunk_knn_search_mma_split_v1'
LOWQ_Q247_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_Q247_SHAPES = _decode_capture(_json_loads('[{"label": "rag_lowq_q2_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 2, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610103, "self_search": false}}, {"label": "rag_lowq_q4_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 4, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610104, "self_search": false}}, {"label": "blind_lowq_q7_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 7, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610606, "self_search": false}}]'))
LOWQ_Q3_GUARDMISS_LABELS: tuple[str, ...] = ('blind_lowq_q3_m131072_d128_k10',)
LOWQ_Q3_GUARDMISS_SHAPES = _decode_capture(_json_loads('[{"label": "blind_lowq_q3_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 3, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610605, "self_search": false}}]'))
LOWQ_Q247_AB_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'blind_lowq_q3_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_Q247_AB_SHAPES = _decode_capture(_json_loads('[{"label": "rag_lowq_q2_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 2, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610103, "self_search": false}}, {"label": "blind_lowq_q3_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 3, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610605, "self_search": false}}, {"label": "rag_lowq_q4_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 4, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610104, "self_search": false}}, {"label": "blind_lowq_q7_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 7, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610606, "self_search": false}}]'))
_BLOCKM640_Q247_ENTRY: dict[str, str] = {'shape_key': 'dispatch0610_r103_6f2a_lowq_q2_q4_q7_blockm640_exact_m131072', 'guard': 'B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q247_BLOCKM640, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:launch_for_eval', 'source_seed': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_BLOCKM640_Q247_ENTRY, {'shape_key': 'dispatch0610_r103_6f2a_guard_miss_incumbent', 'guard': 'otherwise', 'route': ROUTE_INCUMBENT, 'entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'})

def _use_lowq_q247_blockm640(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows in {2, 4, 7} and (int(inputs['M']) == ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q247_blockm640(inputs):
        return ROUTE_LOWQ_Q247_BLOCKM640
    return ROUTE_INCUMBENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q247_blockm640(inputs):
        return {'route': ROUTE_LOWQ_Q247_BLOCKM640, 'selected_route': ROUTE_LOWQ_Q247_BLOCKM640, 'selected_entrypoint': _BLOCKM640_Q247_ENTRY['entrypoint'], 'parent_route': ROUTE_INCUMBENT, 'replaced_route': ROUTE_INCUMBENT, 'route_kind': 'specialized', 'coverage_class': 'performance_route_q2_q4_q7_blockm640_exact_m131072', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _BLOCKM640_Q247_ENTRY['guard'], 'fallback': ROUTE_INCUMBENT, 'missing_weave_route': False}
    return {'route': ROUTE_INCUMBENT, 'selected_route': ROUTE_INCUMBENT, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'parent_route': None, 'replaced_route': None, 'route_kind': 'fallback', 'coverage_class': 'guard_miss_incumbent', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': 'otherwise', 'fallback': None, 'missing_weave_route': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return incumbent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q247_blockm640(inputs):
        return blockm640.launch_for_eval(inputs)
    return incumbent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_q247_blockm640(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_Q247_AB_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
