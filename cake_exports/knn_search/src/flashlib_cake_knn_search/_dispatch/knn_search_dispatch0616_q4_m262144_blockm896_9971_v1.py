"""Round-5 9971 low-Q Q4/M262144 Block-M896 wrapper for exact BF16 kNN.

Minimum target architecture: sm_80 for the Block-M896 tile-reduce route and
sm_100a for inherited 6912 tcgen05/TMEM routes. This additive candidate routes
only exact ``B=1,Q=4,M=262144,D=128,K=10`` rows to the measured Block-M896
Weave seed; every other row delegates to the current round-6912 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0616_seed_bank_6912_v1 as base6912
from . import knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1 as blockm896
THREADS = base6912.THREADS
MERGE_THREADS = base6912.MERGE_THREADS
BLOCK_Q = base6912.BLOCK_Q
BLOCK_M = base6912.BLOCK_M
D_STATIC = base6912.D_STATIC
K_MAX = base6912.K_MAX
SPLIT_M = base6912.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:ir"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:partial_ir"}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:current_ir"}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:base_ir"}'))
blockm896_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:blockm896_ir"}'))
blockm896_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:blockm896_partial_ir"}'))
blockm896_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0616_q4_m262144_blockm896_9971_v1:blockm896_merge_ir"}'))
PROFILE_BASE_6912 = base6912.PROFILE_SELECTED
PROFILE_Q4_M262144_BLOCKM896_9971 = '9971_q4_m262144_blockm896_over_6912'
PROFILE_ALL = PROFILE_Q4_M262144_BLOCKM896_9971
ROUTE_BASE_6912 = 'round6912_seed_bank_selected_dispatcher'
ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971 = 'round5_9971_lowq_q4_m262144_blockm896'
Q4_M262144_SHAPE: dict[str, Any] = {'label': 'probe_lowq_q4_m262144_d128_k10', 'params': {'B': 1, 'Q': 4, 'M': 262144, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 611150, 'self_search': False, 'min_recall': 0.999}}
LOWQ_9971_AUDIT_LABELS: tuple[str, ...] = ('blind_lowq_q3_m131072_d128_k10', 'rag_lowq_q8_m131072_d128_k10', 'rag_lowq_q16_m131072_d128_k10', 'rag_lowq_q32_m131072_d128_k10', 'rag_lowq_q64_m131072_d128_k10')
LOWQ_9971_AUDIT_SHAPES = _decode_capture(_json_loads('[{"label": "blind_lowq_q3_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 3, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610605, "self_search": false}}, {"label": "rag_lowq_q8_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 8, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610105, "self_search": false}}, {"label": "rag_lowq_q16_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 16, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610106, "self_search": false}}, {"label": "rag_lowq_q32_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 32, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610107, "self_search": false}}, {"label": "rag_lowq_q64_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 64, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610108, "self_search": false}}, {"label": "probe_lowq_q4_m262144_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 262144, "Q": 4, "dtype": "bfloat16", "min_recall": 0.999, "seed": 611150, "self_search": false}}]'))
_Q4_M262144_BLOCKM896_ENTRY: dict[str, str] = {'shape_key': 'round5_9971_lowq_q4_m262144_blockm896', 'guard': 'B == 1 and Q == 4 and M == 262144 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-b3b4', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_b3b4_q4_m262144_blockm896.md', 'selected_seed': 'round3_6912_lowq_q4_m262144_blockm896'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q4_M262144_BLOCKM896_ENTRY, *base6912.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base6912._forced_fallback(inputs)

def _use_lowq_q4_m262144_blockm896(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == 4) and (int(inputs['M']) == 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q4_m262144_blockm896(inputs):
        return ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971
    return base6912.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q4_m262144_blockm896(inputs):
        parent_info = dict(base6912.route_info(inputs))
        parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base6912.selected_route(inputs))
        return {'profile': PROFILE_Q4_M262144_BLOCKM896_9971, 'route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'selected_route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'selected_entrypoint': _Q4_M262144_BLOCKM896_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_q4_m262144_blockm896', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _Q4_M262144_BLOCKM896_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _Q4_M262144_BLOCKM896_ENTRY['guard'], 'fallback': ROUTE_BASE_6912, 'missing_weave_route': False, 'source_task': _Q4_M262144_BLOCKM896_ENTRY['source_task'], 'source_round_doc': _Q4_M262144_BLOCKM896_ENTRY['source_round_doc'], 'selected_seed': _Q4_M262144_BLOCKM896_ENTRY['selected_seed']}
    info = dict(base6912.route_info(inputs))
    selected = str(info.get('route') or info.get('selected_route') or base6912.selected_route(inputs))
    info.update({'profile': PROFILE_Q4_M262144_BLOCKM896_9971, 'route': selected, 'selected_route': selected, 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_6912_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base6912.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q4_m262144_blockm896(inputs):
        return blockm896.launch_for_eval(inputs)
    return base6912.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4_m262144_blockm896_9971(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_9971_AUDIT_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
