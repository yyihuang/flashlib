"""Guarded K11 and D128-tail repair over the exported 8ae1 dispatcher.

Minimum target architecture: sm_100a.  This wrapper leaves all seed schedules
unchanged.  It adapts the validated Q4096/M20000/D128/K64 Weave seed to K11
by returning its exact sorted prefix, uses an explicit +infinity M-tail pack
before the existing K10 Weave MMA seed, and sends the Q tail to that seed
directly.  Forced fallback remains owned by the inherited exported dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
import torch
from . import knn_search_dispatch0701_8ae1_q4096_exported_vertical_slice_consumption_v1 as base
from . import knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1 as k64_seed
from . import knn_search_mma_split_v1 as mma_seed
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as mtail_seed
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:launch_for_eval'
K11_SEED = 'round20-576b-q4096-m20000-d128-k64-prefix-adapter'
MTAIL_SEED = 'round14-4e2c-q4096-split4-mtail'
QTAIL_SEED = 'knn-search-mma-split-v1-qtail'
K64_ENTRYPOINT = 'loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:launch_for_eval'
MMA_ENTRYPOINT = 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'
MTAIL_ENTRYPOINT = 'loom.examples.weave.knn_search_q4096_split4_0611_r14_4e2c_v1:launch_for_eval'
K11_GUARD_ID = 'q4096_m20000_d128_k11_k64_prefix_adapter'
MTAIL_GUARD_ID = 'q4096_m19999_d128_k10_infinity_mtail'
QTAIL_GUARD_ID = 'q4095_m20001_d128_k10_direct_qtail'
_K11 = (1, 4096, 20000, 128, 11)
_MTAIL = (1, 4096, 19999, 128, 10)
_QTAIL = (1, 4095, 20001, 128, 10)
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY = ({'shape_key': K11_GUARD_ID, 'route': 'k64_prefix_to_k11', 'entrypoint': K64_ENTRYPOINT}, {'shape_key': MTAIL_GUARD_ID, 'route': 'split4_k10_mtail', 'entrypoint': MTAIL_ENTRYPOINT}, {'shape_key': QTAIL_GUARD_ID, 'route': 'mma_k10_direct_qtail', 'entrypoint': MMA_ENTRYPOINT}, *base.SHAPE_DISPATCH_REGISTRY)

def _key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')))

def _eligible(inputs: dict[str, Any], key: tuple[int, int, int, int, int]) -> bool:
    return _key(inputs) == key and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _route(inputs: dict[str, Any]) -> str | None:
    if _eligible(inputs, _K11):
        return 'k64_prefix_to_k11'
    if _eligible(inputs, _MTAIL):
        return 'split4_k10_mtail'
    if _eligible(inputs, _QTAIL):
        return 'mma_k10_direct_qtail'
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    return _route(inputs) or base.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = _route(inputs)
    if route is None:
        info = dict(base.route_info(inputs))
        info.update({'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]})
        return info
    guard_id, seed_id, entrypoint, condition = {'k64_prefix_to_k11': (K11_GUARD_ID, K11_SEED, K64_ENTRYPOINT, 'B=1,Q=4096,M=20000,D=128,K=11,nonself,not forced'), 'split4_k10_mtail': (MTAIL_GUARD_ID, MTAIL_SEED, MTAIL_ENTRYPOINT, 'B=1,Q=4096,M=19999,D=128,K=10,nonself,not forced'), 'mma_k10_direct_qtail': (QTAIL_GUARD_ID, QTAIL_SEED, MMA_ENTRYPOINT, 'B=1,Q=4095,M=20001,D=128,K=10,nonself,not forced')}[route]
    return {'route': route, 'selected_route': route, 'selected_entrypoint': entrypoint, 'selected_seed': seed_id, 'expected_seed': seed_id, 'guard_id': guard_id, 'guard_condition': condition, 'selected_guard': condition, 'route_kind': 'specialized', 'route_source': 'generated-variant', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def _launch_k11_prefix(inputs: dict[str, Any]) -> dict[str, Any]:
    seed_inputs = dict(inputs)
    seed_inputs['K'] = 64
    seed_inputs['out_distances'] = torch.empty((*inputs['out_distances'].shape[:-1], 64), dtype=torch.float32, device=inputs['queries'].device)
    seed_inputs['out_indices'] = torch.empty((*inputs['out_indices'].shape[:-1], 64), dtype=torch.int32, device=inputs['queries'].device)
    outputs = k64_seed.launch_for_eval(seed_inputs)
    return {'distances': outputs['distances'][..., :11], 'indices': outputs['indices'][..., :11]}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = _route(inputs)
    if route == 'k64_prefix_to_k11':
        return _launch_k11_prefix(inputs)
    if route == 'split4_k10_mtail':
        return mtail_seed.launch_for_eval(inputs)
    if route == 'mma_k10_direct_qtail':
        return mma_seed.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)
