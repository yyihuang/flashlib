"""Compose the registered b653 portfolio with the full 0701 public surface.

Minimum architecture: sm_100a for registry shapes delegated to b653.  Other
shapes preserve the architecture requirements and behavior of the 0701 public
dispatcher.  This host-only dispatcher does not change either kernel path; it
prevents registry promotion from dropping the established extended-K and
runtime-coverage surface.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0701_k11_d128_guard_repair_v1 as compat0701
from . import knn_search_residual0705_q64_tail_split152_full_bucket_dispatcher_b653_v1 as registered_b653
ENTRYPOINT = 'loom.examples.weave.knn_search_registry_b653_compat0701_v1:launch_for_eval'
detect_gpu_arch = registered_b653.detect_gpu_arch
_B653_REGISTRY_SHAPES = frozenset({(1, 256, 256, 128, 5, True), (1, 1, 131072, 128, 10, False), (1, 128, 131072, 128, 10, False), (1, 4096, 20000, 128, 10, False), (1, 8, 131072, 128, 10, False), (1, 16, 131072, 128, 10, False), (1, 32, 131072, 128, 10, False), (1, 64, 131072, 128, 10, False), (1, 4096, 20000, 128, 1, False), (1, 4096, 20000, 128, 2, False), (1, 4096, 20000, 128, 64, False), (1, 128, 131072, 256, 10, False), (1, 128, 131072, 256, 64, False), (1, 8, 10, 32, 10, False), (1, 8, 20, 48, 10, False), (1, 1500, 1500, 2, 32, True), (1, 1500, 1500, 2, 64, True), (1, 4096, 20000, 128, 5, False), (1, 4096, 20000, 128, 8, False), (1, 4096, 16384, 128, 8, False), (1, 4096, 32768, 128, 8, False)})
_B653_RESIDUAL_SHAPES = frozenset({(1, 64, 262144, 256, 64, False), (1, 64, 262143, 256, 64, False), (1, 64, 262145, 256, 64, False), (1, 63, 262144, 256, 64, False), (1, 65, 262144, 256, 64, False), (1, 128, 262144, 256, 64, False)})
_B653_DOMAIN_SHAPES = _B653_REGISTRY_SHAPES | _B653_RESIDUAL_SHAPES

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _uses_registered_b653(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in _B653_DOMAIN_SHAPES

def _child(inputs: dict[str, Any]):
    return registered_b653 if _uses_registered_b653(inputs) else compat0701

def selected_route(inputs: dict[str, Any]) -> str:
    return str(_child(inputs).selected_route(inputs))

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    shape_key = _shape_key(inputs)
    uses_registry = shape_key in _B653_DOMAIN_SHAPES
    info = dict(_child(inputs).route_info(inputs))
    info.update({'export_composite_entrypoint': ENTRYPOINT, 'export_composite_branch': 'registered_b653' if uses_registry else 'compat0701', 'registry_contract_shape': shape_key in _B653_REGISTRY_SHAPES, 'residual_convergence_shape': shape_key in _B653_RESIDUAL_SHAPES})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _uses_registered_b653(inputs):
        return registered_b653.launch_361b_f392_d436_split152_for_eval(inputs)
    return compat0701.launch_for_eval(inputs)
launch_for_eval.route_info = route_info
