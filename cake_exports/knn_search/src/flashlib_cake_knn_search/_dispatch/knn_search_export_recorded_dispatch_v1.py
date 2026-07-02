"""Complete recorded-shape kNN search dispatcher for standalone export.

Minimum target architecture: sm_100a.  The current July-1 production
dispatcher owns the normal route set, while three recorded extended-K shapes
retain the previously validated 28ec capacity kernels.  Recorded shapes whose
upstream optimized routes currently fail exact-recall validation use the
source-clean scalar-capacity kernel until those routes are repaired.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0701_k11_d128_guard_repair_v1 as current
from . import knn_search_ext_k_capacity_0618_28ec_v1 as ext_k
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_repair
ENTRYPOINT = 'loom.examples.weave.knn_search_export_recorded_dispatch_v1:launch_for_eval'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
CORRECTNESS_REPAIR_ROUTE = 'export_recorded_scalar_correctness_repair'
_CORRECTNESS_REPAIR_KEYS = {(1, 9, 196608, 128, 10, False), (1, 17, 196608, 128, 10, False), (1, 24, 196608, 128, 10, False), (1, 128, 65536, 64, 10, False), (1, 128, 65536, 96, 10, False), (1, 128, 65536, 192, 10, False), (1, 129, 131073, 128, 10, False), (1, 255, 262143, 128, 10, False), (1, 256, 65536, 128, 10, False), (1, 257, 262145, 128, 10, False), (1, 384, 49152, 128, 10, False), (1, 512, 65536, 128, 10, False), (1, 513, 98304, 128, 10, False), (1, 768, 49152, 128, 10, False), (1, 1024, 1024, 128, 10, True), (1, 1024, 65536, 128, 10, False), (1, 1536, 49152, 128, 10, False), (1, 2048, 65536, 128, 10, False), (1, 3072, 3072, 128, 10, True), (1, 3072, 49152, 128, 10, False), (1, 4096, 4096, 128, 10, True), (1, 4096, 16384, 128, 10, False), (1, 4096, 20000, 128, 7, False), (1, 4096, 32768, 128, 10, False), (1, 4096, 49152, 128, 5, False), (1, 4096, 65536, 128, 10, False), (1, 10000, 100000, 128, 10, False), (2, 64, 262144, 128, 10, False), (2, 128, 65536, 128, 10, False), (2, 257, 65536, 128, 10, False), (2, 4096, 20000, 128, 10, False)}

def _use_correctness_repair(inputs: dict[str, Any]) -> bool:
    key = (*(int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')), bool(inputs.get('self_search', False)))
    return key in _CORRECTNESS_REPAIR_KEYS

def _use_ext_k(inputs: dict[str, Any]) -> bool:
    return bool(ext_k._use_q128_m131072_k40(inputs) or ext_k._use_q128_m65536_k56(inputs) or ext_k._use_q4096_m49152_k64(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_correctness_repair(inputs):
        return CORRECTNESS_REPAIR_ROUTE
    return ext_k.selected_route(inputs) if _use_ext_k(inputs) else current.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_correctness_repair(inputs):
        info = {'route': CORRECTNESS_REPAIR_ROUTE, 'selected_route': CORRECTNESS_REPAIR_ROUTE, 'selected_entrypoint': f'{scalar_repair.__name__}:launch_scalar_capacity_for_eval', 'route_kind': 'correctness-repair', 'route_source': 'validated-generic-weave-fallback'}
    else:
        info = dict(ext_k.route_info(inputs) if _use_ext_k(inputs) else current.route_info(inputs))
    info['dispatcher_entrypoint'] = ENTRYPOINT
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_correctness_repair(inputs):
        return scalar_repair.launch_scalar_capacity_for_eval(inputs)
    return ext_k.launch_for_eval(inputs) if _use_ext_k(inputs) else current.launch_for_eval(inputs)
