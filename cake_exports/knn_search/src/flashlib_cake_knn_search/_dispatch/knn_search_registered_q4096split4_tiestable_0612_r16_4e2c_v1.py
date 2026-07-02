"""Round-16 registered kNN dispatcher with Q4096 split-4 tie-stable buckets.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA paths. This
wrapper preserves the current round-20 registered dispatcher for all non-Q4096
split-4 bucket shapes, and routes only the measured ``Q=4096,D=128,K<=10`` M
buckets through the round-15 tie-stable split-4 path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from . import knn_search_k20k30_mma_q128register_0611_r20_4e96_v1 as parent
from . import knn_search_q4096_split4_tiestable_0612_r15_4e2c_v1 as q4096_split4
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_registered_q4096split4_tiestable_0612_r16_4e2c_v1:ir"}'))
Q4096_SPLIT4_REGISTERED_SHAPES: list[dict[str, Any]] = q4096_split4.Q4096_SPLIT4_TIESTABLE_SHAPES

def _use_registered_q4096_split4(inputs: dict[str, Any]) -> bool:
    return q4096_split4._use_q4096_split4_tiestable(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_registered_q4096_split4(inputs):
        return q4096_split4.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
