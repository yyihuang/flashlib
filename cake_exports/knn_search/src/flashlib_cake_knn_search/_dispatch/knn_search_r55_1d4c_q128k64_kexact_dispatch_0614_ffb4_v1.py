"""Round-ffb4 Q128/K64 exact-K dispatcher-consumption wrapper.

Minimum target architecture: sm_100a for the tcgen05 Q128/K64 exact-K64 seed
and inherited 1d4c routes. This wrapper consumes only the exact
``B=1, Q=128, M=131072, D=128, K=64`` round-25/f2f6 exact-K seed on top of
the promoted 1d4c dispatcher. Every guard miss, including Q4096/K64 and forced
fallback, delegates to the 1d4c Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as q128_kexact
from . import knn_search_r55_speedup_repair_dispatch_0614_1d4c_v1 as base
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
SPLIT_M = base.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_r55_1d4c_q128k64_kexact_dispatch_0614_ffb4_v1:ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "cta_group": 1, "threads": 512}'))
q128_kexact_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_r55_1d4c_q128k64_kexact_dispatch_0614_ffb4_v1:q128_kexact_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "cta_group": 1, "threads": 512}'))
q128_kexact_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_r55_1d4c_q128k64_kexact_dispatch_0614_ffb4_v1:q128_kexact_group_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
q128_kexact_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_r55_1d4c_q128k64_kexact_dispatch_0614_ffb4_v1:q128_kexact_final_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
ROUTE_1D4C_DEFAULT = 'round15_1d4c_promoted_dispatcher'
ROUTE_Q128_K64_KEXACT_FFB4 = 'round25_q128_k64_split512_hiermerge32_kexact'
PROFILE_1D4C_BASE = 'current_1d4c'
PROFILE_FFB4 = 'r55_1d4c_plus_q128k64_kexact_ffb4'
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_1D4C_BASE: (), PROFILE_FFB4: ('q128_k64_kexact_ffb4',)}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'roundffb4_q128_m131072_d128_k64_kexact_target', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_Q128_K64_KEXACT_FFB4}, *base.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> set[str]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(f'unknown ffb4 dispatcher-consumption profile: {profile}')
    return set(CANDIDATE_PROFILES[profile])

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base._forced_fallback(inputs)

def _use_q128_k64_kexact(inputs: dict[str, Any]) -> bool:
    return q128_kexact._use_q128_k64_split512_indexfast(inputs)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if _forced_fallback(inputs):
        return base.selected_route(inputs)
    if 'q128_k64_kexact_ffb4' in overlays and _use_q128_k64_kexact(inputs):
        return ROUTE_Q128_K64_KEXACT_FFB4
    return base.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_FFB4)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _target_guard() -> str:
    return str(SHAPE_DISPATCH_REGISTRY[0]['guard'])

def _target_entrypoint() -> str:
    return 'loom.examples.weave.knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1:launch_for_eval'

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    parent_info = dict(base.route_info(inputs))
    forced = _forced_fallback(inputs)
    if not forced and 'q128_k64_kexact_ffb4' in overlays and _use_q128_k64_kexact(inputs):
        return {'profile': profile, 'route': ROUTE_Q128_K64_KEXACT_FFB4, 'selected_route': ROUTE_Q128_K64_KEXACT_FFB4, 'selected_entrypoint': _target_entrypoint(), 'parent_route': parent_info.get('route'), 'replaced_route': parent_info.get('route'), 'route_kind': 'specialized', 'coverage_class': 'performance_route_q128_k64_kexact_ffb4', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _target_guard(), 'fallback': None}
    info = dict(parent_info)
    info['profile'] = profile
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    if forced:
        info['forced_fallback'] = True
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_FFB4)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_FFB4) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if not _forced_fallback(inputs):
        if 'q128_k64_kexact_ffb4' in overlays and _use_q128_k64_kexact(inputs):
            return q128_kexact.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)

def launch_1d4c_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_1D4C_BASE)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_FFB4)
