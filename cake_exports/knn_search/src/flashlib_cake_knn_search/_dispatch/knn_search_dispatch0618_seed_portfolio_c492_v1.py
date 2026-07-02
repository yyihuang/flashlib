"""Full101 kNN seed-portfolio dispatcher with exact D384 repair.

Minimum target architecture: sm_100a for tcgen05/TMEM seed routes and sm_80
for inherited CUDA-core fallback routes. This wrapper consumes the 1e12 exact
``B=1,Q=32,M=131072,D=384,K=10`` seed before the old 04af a2ab coverage route
and delegates every other guard to the 04af dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_seed_portfolio_04af_v1 as base
from . import knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1 as d384_1e12
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
SPLIT_M = base.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:ir"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:partial_ir"}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:current_ir"}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:base_ir"}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q128_22d9_ir"}'))
blockm640_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm640_ir"}'))
blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm640_partial_ir"}'))
blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm640_merge_ir"}'))
b2_q128_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:b2_q128_ir"}'))
b2_q128_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:b2_q128_merge_ir"}'))
blockm896_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm896_ir"}'))
blockm896_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm896_partial_ir"}'))
blockm896_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:blockm896_merge_ir"}'))
q2_blockm640_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q2_blockm640_ir"}'))
q2_blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q2_blockm640_partial_ir"}'))
q2_blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q2_blockm640_merge_ir"}'))
cc76_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:cc76_partial_ir"}'))
cc76_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:cc76_merge_ir"}'))
q3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q3_partial_ir"}'))
q3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q3_merge_ir"}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:scalar_capacity_ir"}'))
self_q2048_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:self_q2048_ir"}'))
self_q2048_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:self_q2048_merge_ir"}'))
k1_merge8_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k1_merge8_partial_ir"}'))
k1_merge8_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k1_merge8_merge_ir"}'))
q1_m262144_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q1_m262144_ir"}'))
q1_m262144_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:q1_m262144_merge_ir"}'))
b2_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:b2_k64_partial_ir"}'))
b2_k64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:b2_k64_group_merge_ir"}'))
b2_k64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:b2_k64_final_merge_ir"}'))
d384_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:d384_q256_partial_ir"}'))
d384_q256_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:d384_q256_merge_ir"}'))
k64_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_q256_partial_ir"}'))
k64_q256_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_q256_group_merge_ir"}'))
k64_q256_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_q256_final_merge_ir"}'))
lowd_d256_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_d256_ir"}'))
lowd_d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_d256_partial_ir"}'))
lowd_d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_d256_merge_ir"}'))
lowd_d256_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_d256_k64_partial_ir"}'))
lowd_d256_k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_d256_k64_merge_ir"}'))
lowd_dbscan_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:lowd_dbscan_ir"}'))
tinyd_d3_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:tinyd_d3_ir"}'))
tinyd_d3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:tinyd_d3_partial_ir"}'))
tinyd_d3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:tinyd_d3_merge_ir"}'))
tinyd_449d_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:tinyd_449d_ir"}'))
tinyd_449d_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:tinyd_449d_partial_ir"}'))
highd_06f4_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:highd_06f4_ir"}'))
highd_06f4_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:highd_06f4_partial_ir"}'))
k64_f0a3_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_f0a3_ir"}'))
k64_f0a3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_f0a3_partial_ir"}'))
k64_f0a3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:k64_f0a3_merge_ir"}'))
remaining_a2ab_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:remaining_a2ab_ir"}'))
d384_1e12_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:d384_1e12_ir"}'))
d384_1e12_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:d384_1e12_partial_ir"}'))
d384_1e12_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_seed_portfolio_c492_v1:d384_1e12_merge_ir"}'))
PROFILE_BASE_04AF = '04af_seed_portfolio'
PROFILE_ALL = 'c492_seed_portfolio'
ROUTE_BASE_04AF = base.PROFILE_ALL
ROUTE_D384_1E12 = d384_1e12.ROUTE_D384_Q32_EXACT_TCGEN05
CONSUMED_D384_1E12_SEED = 'weave-evolve-knn-search-1e12'
CONSUMED_SEEDS = (*base.CONSUMED_SEEDS, CONSUMED_D384_1E12_SEED)
DYNAMIC_D_C492_LABELS: tuple[str, ...] = base.DYNAMIC_D_04AF_LABELS
DYNAMIC_D_C492_SHAPES = _decode_capture(_json_loads('[{"label": "blind_dyn_d3_q128_m65536_k10", "params": {"B": 1, "D": 3, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610801, "self_search": false}}, {"label": "blind_dyn_d7_q128_m65536_k10", "params": {"B": 1, "D": 7, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610802, "self_search": false}}, {"label": "blind_dyn_d63_q128_m65536_k10", "params": {"B": 1, "D": 63, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610803, "self_search": false}}, {"label": "blind_dyn_d129_q128_m65536_k10", "params": {"B": 1, "D": 129, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610804, "self_search": false}}, {"label": "blind_dyn_d257_q128_m65536_k10", "params": {"B": 1, "D": 257, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610805, "self_search": false}}, {"label": "blind_dyn_d511_q128_m65536_k10", "params": {"B": 1, "D": 511, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610806, "self_search": false}}, {"label": "blind_dyn_d384_q32_m131072_k10", "params": {"B": 1, "D": 384, "K": 10, "M": 131072, "Q": 32, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610807, "self_search": false}}, {"label": "blind_dyn_d257_k64_q64_m65536", "params": {"B": 1, "D": 257, "K": 64, "M": 65536, "Q": 64, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610808, "self_search": false}}, {"label": "blind_dyn_b2_q64_m65536_d129_k10", "params": {"B": 2, "D": 129, "K": 10, "M": 65536, "Q": 64, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610809, "self_search": false}}, {"label": "blind_dyn_self_q2048_m2048_d3_k10", "params": {"B": 1, "D": 3, "K": 10, "M": 2048, "Q": 2048, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610810, "self_search": true}}]'))
_D384_1E12_ENTRY: dict[str, Any] = {'overlay': 'd384_1e12_exact_tcgen05', 'shape_key': 'c492_dynamic_d384_q32_m131072_k10_1e12_exact', 'labels': d384_1e12.D384_Q32_LABELS, 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D384_1E12, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:launch_for_eval', 'selected_seed': CONSUMED_D384_1E12_SEED, 'source_task': CONSUMED_D384_1E12_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_112_5847_d384_q32_exact.md', 'coverage_class': 'bucket_seed_dynamic_d_d384_q32_m131072_k10', 'coverage_only': False}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D384_1E12_ENTRY, *base.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _profile_uses_d384_1e12(profile: str) -> bool:
    return profile == PROFILE_ALL

def _base_profile(profile: str) -> str:
    if profile in {PROFILE_ALL, PROFILE_BASE_04AF}:
        return base.PROFILE_ALL
    return profile

def _use_d384_1e12(inputs: dict[str, Any], profile: str) -> bool:
    if not _profile_uses_d384_1e12(profile):
        return False
    if _shape_key(inputs) != (1, 32, 131072, 384, 10, False):
        return False
    return d384_1e12._use_d384_q32_exact_tcgen05(inputs)

def _guard_order(profile: str) -> list[str]:
    base_order = list(base._guard_order(_base_profile(profile)))
    if _profile_uses_d384_1e12(profile):
        return [str(_D384_1E12_ENTRY['shape_key']), *base_order]
    return base_order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    if _use_d384_1e12(inputs, profile):
        return ROUTE_D384_1E12
    return base.selected_route_for_profile(inputs, _base_profile(profile))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base.route_info_for_profile(inputs, _base_profile(profile)))
    route = str(info.get('route') or info.get('selected_route') or base.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _d384_1e12_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base.selected_route(inputs))
    return {'profile': profile, 'route': _D384_1E12_ENTRY['route'], 'selected_route': _D384_1E12_ENTRY['route'], 'selected_entrypoint': _D384_1E12_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': _D384_1E12_ENTRY['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _D384_1E12_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _D384_1E12_ENTRY['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': _D384_1E12_ENTRY['source_task'], 'source_round_doc': _D384_1E12_ENTRY['source_round_doc'], 'selected_seed': _D384_1E12_ENTRY['selected_seed'], 'selected_seed_task': _D384_1E12_ENTRY['source_task'], 'replaced_seed': parent_info.get('selected_seed')}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if _use_d384_1e12(inputs, profile):
        return _d384_1e12_info(inputs, profile)
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if _use_d384_1e12(inputs, profile):
        return d384_1e12.launch_for_eval(inputs)
    return base.launch_for_profile(inputs, _base_profile(profile))

def launch_base_04af_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_04AF)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return DYNAMIC_D_C492_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0618_seed_portfolio_c492(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_D_C492_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_D_C492_LABELS) -> dict[str, Any]:
    return knn_search_compile_and_launch_dispatch0618_seed_portfolio_c492(benchmark=benchmark, shape_labels=shape_labels)
