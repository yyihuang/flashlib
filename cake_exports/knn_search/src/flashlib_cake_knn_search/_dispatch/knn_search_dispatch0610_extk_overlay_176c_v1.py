"""Dispatch-0610 extended-K overlay for exact BF16 kNN search.

Minimum target architecture: sm_100a for the tcgen05-backed extended-K routes.
This additive candidate preserves the current ``knn_search_mma_split_v1``
K<=10 incumbent path and exposes D128/K>10 Weave-only seeds for the expanded
dispatch-slurm contract frontier.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1 as extendedk
from . import knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1 as truek48
from . import knn_search_mma_split_v1 as incumbent
THREADS = incumbent.THREADS
MERGE_THREADS = truek48.MERGE_THREADS
BLOCK_Q = incumbent.BLOCK_Q
BLOCK_M = incumbent.BLOCK_M
D_STATIC = incumbent.D_STATIC
K10_MAX = incumbent.K_MAX
K_MAX = extendedk.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0610_extk_overlay_176c_v1:ir"}'))
incumbent_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0610_extk_overlay_176c_v1:incumbent_ir"}'))
extendedk_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0610_extk_overlay_176c_v1:extendedk_ir"}'))
truek48_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0610_extk_overlay_176c_v1:truek48_ir"}'))
ROUTE_INCUMBENT_K10 = 'round176c_incumbent_mma_split_k_le_10'
ROUTE_TRUEK48 = truek48.ROUTE_Q128_K48_TRUEK48
ROUTE_EXTENDEDK_PORTFOLIO = 'round176c_extendedk_r2_portfolio'
EXTK_OVERLAY_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k10', 'ksweep_q128_m131072_d128_k11', 'ksweep_q128_m131072_d128_k12', 'blind_k16_q128_m131072_d128_k16', 'ksweep_q128_m131072_d128_k20', 'ksweep_q128_m131072_d128_k30', 'blind_k48_q128_m131072_d128_k48', 'ksweep_q128_m131072_d128_k64', 'ksweep_q4096_m20000_d128_k64')
EXTK_OVERLAY_EVAL_SHAPES = _decode_capture(_json_loads('[{"label": "ksweep_q128_m131072_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610305, "self_search": false}}, {"label": "ksweep_q128_m131072_d128_k11", "params": {"B": 1, "D": 128, "K": 11, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610306, "self_search": false}}, {"label": "ksweep_q128_m131072_d128_k12", "params": {"B": 1, "D": 128, "K": 12, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610307, "self_search": false}}, {"label": "blind_k16_q128_m131072_d128_k16", "params": {"B": 1, "D": 128, "K": 16, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610517, "self_search": false}}, {"label": "ksweep_q128_m131072_d128_k20", "params": {"B": 1, "D": 128, "K": 20, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610308, "self_search": false}}, {"label": "ksweep_q128_m131072_d128_k30", "params": {"B": 1, "D": 128, "K": 30, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610309, "self_search": false}}, {"label": "blind_k48_q128_m131072_d128_k48", "params": {"B": 1, "D": 128, "K": 48, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610518, "self_search": false}}, {"label": "ksweep_q128_m131072_d128_k64", "params": {"B": 1, "D": 128, "K": 64, "M": 131072, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610312, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k64", "params": {"B": 1, "D": 128, "K": 64, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610313, "self_search": false}}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': '176c_k_le_10_incumbent', 'guard': 'K <= 10; delegate to knn_search_mma_split_v1 incumbent', 'route': ROUTE_INCUMBENT_K10}, {'shape_key': '176c_exact_q128_m131072_d128_k48_truek48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_TRUEK48}, {'shape_key': '176c_d128_k_gt_10_extendedk_r2', 'guard': 'D == 128 and K > 10; delegate to round-2 extended-K Weave portfolio', 'route': ROUTE_EXTENDEDK_PORTFOLIO})

def selected_route(inputs: dict[str, Any]) -> str:
    if int(inputs['K']) <= K10_MAX:
        return ROUTE_INCUMBENT_K10
    if truek48._use_q128_k48_truek48(inputs):
        return ROUTE_TRUEK48
    return ROUTE_EXTENDEDK_PORTFOLIO

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_INCUMBENT_K10:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'route_kind': 'incumbent', 'coverage_class': 'preserved_k_le_10', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'fallback': None}
    if route == ROUTE_TRUEK48:
        info = dict(truek48.route_info(inputs))
        info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
        return info
    info = dict(extendedk.route_info(inputs))
    info['route'] = ROUTE_EXTENDEDK_PORTFOLIO
    info['selected_route'] = ROUTE_EXTENDEDK_PORTFOLIO
    info['replaced_route'] = extendedk.selected_route(inputs)
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_INCUMBENT_K10:
        return incumbent.launch_for_eval(inputs)
    if route == ROUTE_TRUEK48:
        return truek48.launch_for_eval(inputs)
    return extendedk.launch_for_eval(inputs)

def knn_search_compile_and_launch_extk_overlay_176c(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=EXTK_OVERLAY_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
