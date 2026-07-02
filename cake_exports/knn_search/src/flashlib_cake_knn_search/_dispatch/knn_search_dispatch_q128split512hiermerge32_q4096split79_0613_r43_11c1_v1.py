"""Round-43 K64 dispatcher over Q128 split512/group32 and Q4096 split79 routes.

Minimum target architecture: sm_100a for both registered tcgen05 producer
paths. This dispatcher consumes the exact
``B=1,Q=128,M=131072,D=128,K=64`` split512 hierarchical-merge32 route and
preserves the round-35 exact ``B=1,Q=4096,M=20000,D=128,K=64`` split79 route.
All other shapes delegate to the round-38 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from collections.abc import Callable
from typing import Any, NamedTuple
from . import knn_search_dispatch_q128split256_q4096split79_0613_r38_11c1_v1 as parent
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_k64_split512
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:partial_ir"}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:group_merge_ir"}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:final_merge_ir"}'))
q4096_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:q4096_partial_ir"}'))
q4096_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:q4096_merge_ir"}'))
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1:ir"}'))

class DispatchRoute(NamedTuple):
    name: str
    shape_key: str
    guard: str
    kernel_ref: str
    predicate: Callable[[dict[str, Any]], bool]
    launcher: Callable[[dict[str, Any]], dict[str, Any]]
    source_round_doc: str

def _use_q128_k64_split512(inputs: dict[str, Any]) -> bool:
    return q128_k64_split512._use_q128_k64_split512_hiermerge(inputs)

def _launch_q128_k64_split512(inputs: dict[str, Any]) -> dict[str, Any]:
    return q128_k64_split512._launch_q128_k64_split512_hiermerge(inputs)

def _use_q4096_k64_split79(inputs: dict[str, Any]) -> bool:
    return parent._use_q4096_k64_split79(inputs)

def _launch_q4096_k64_split79(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent._launch_q4096_k64_split79(inputs)
DISPATCH_ROUTES: tuple[DispatchRoute, ...] = (DispatchRoute(name='q128_m131072_d128_k64_split512_hiermerge32', shape_key='B1_Q128_M131072_D128_K64', guard='B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch', kernel_ref='loom.examples.weave.knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1:launch_for_eval', predicate=_use_q128_k64_split512, launcher=_launch_q128_k64_split512, source_round_doc='design_doc/active/weave_evolve_knn_search_round_43_11c1_dispatch_hiermerge32.md'), DispatchRoute(name='q4096_m20000_d128_k64_split79_oddeven_fastmerge', shape_key='B1_Q4096_M20000_D128_K64', guard='B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch', kernel_ref='loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval', predicate=_use_q4096_k64_split79, launcher=_launch_q4096_k64_split79, source_round_doc='design_doc/active/weave_evolve_knn_search_round_34_11c1_q4096k64split79.md'))
DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"kernel_ref": "loom.examples.weave.knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1:launch_for_eval", "name": "q128_m131072_d128_k64_split512_hiermerge32", "shape_guard": "B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch", "shape_key": "B1_Q128_M131072_D128_K64", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_43_11c1_dispatch_hiermerge32.md"}, {"kernel_ref": "loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval", "name": "q4096_m20000_d128_k64_split79_oddeven_fastmerge", "shape_guard": "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch", "shape_key": "B1_Q4096_M20000_D128_K64", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_34_11c1_q4096k64split79.md"}]}'))
DISPATCHER_R43_K64_SHAPES: list[dict[str, Any]] = q128_k64_split512.K64_Q128_SPLIT512_HIERMERGE_SHAPES

def selected_route_name(inputs: dict[str, Any]) -> str:
    for route in DISPATCH_ROUTES:
        if route.predicate(inputs):
            return route.name
    return parent.selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    for route in DISPATCH_ROUTES:
        if route.predicate(inputs):
            return route.launcher(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch_r43(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = DISPATCHER_R43_K64_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
