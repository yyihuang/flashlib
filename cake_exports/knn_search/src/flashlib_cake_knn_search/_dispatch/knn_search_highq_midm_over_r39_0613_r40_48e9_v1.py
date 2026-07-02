"""Round-40 high-Q mid-M D128 route over the r39 kNN dispatcher.

Minimum target architecture: sm_100a for the tcgen05 high-Q MMA routes. This
shape-family candidate routes Q256..Q2048/M65536/K10 labels through the
source-clean high-Q Q-bucket split policy, routes Q4096/M16K..32K/K<=10 labels
through the split-8 Q4096 policy, and preserves the round-39 low-Q/Q128/D256
dispatcher for all other contract shapes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_highq_qbucket_registered_0612_r19_4e96split64_v1 as highq_qbucket
from . import knn_search_lowq_d128_dispatch_0613_r39_48e9_v1 as parent
from . import knn_search_q4096_split8_0611_r13_4e2c_v1 as q4096_split8
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:ir"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:partial_ir"}'))
q4096_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:q4096_ir"}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:parent_ir"}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:d256_partial_ir"}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:d256_merge_ir"}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:k64_partial_ir"}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_highq_midm_over_r39_0613_r40_48e9_v1:k64_merge_ir"}'))
HIGHQ_QBUCKET_LABELS: tuple[str, ...] = ('dispatch_q256_m65536_d128_k10', 'dispatch_q512_m65536_d128_k10', 'dispatch_q1024_m65536_d128_k10', 'dispatch_q2048_m65536_d128_k10')
Q4096_SPLIT8_LABELS: tuple[str, ...] = ('rag_q4096_m20000_d128_k10', 'rag_batch_q4096_m20000_d128_k10', 'dispatch_q4096_m16384_d128_k10', 'dispatch_q4096_m32768_d128_k10', 'ksweep_q4096_m20000_d128_k1', 'ksweep_q4096_m20000_d128_k2')
HIGHQ_MIDM_SHAPE_LABELS: tuple[str, ...] = (*HIGHQ_QBUCKET_LABELS, *Q4096_SPLIT8_LABELS)
HIGHQ_QBUCKET_SHAPES = _decode_capture(_json_loads('[{"label": "dispatch_q256_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 256, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610206, "self_search": false}}, {"label": "dispatch_q512_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 512, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610207, "self_search": false}}, {"label": "dispatch_q1024_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 1024, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610208, "self_search": false}}, {"label": "dispatch_q2048_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 2048, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610209, "self_search": false}}]'))
Q4096_SPLIT8_SHAPES = _decode_capture(_json_loads('[{"label": "rag_q4096_m20000_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 20000, "Q": 4096, "benchmark": true, "check_correctness": true, "dtype": "bfloat16", "min_recall": 0.999, "seed": 3, "self_search": false}}, {"label": "rag_batch_q4096_m20000_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610110, "self_search": false}}, {"label": "dispatch_q4096_m16384_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 16384, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610210, "self_search": false}}, {"label": "dispatch_q4096_m32768_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 32768, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610211, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k1", "params": {"B": 1, "D": 128, "K": 1, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610310, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k2", "params": {"B": 1, "D": 128, "K": 2, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610311, "self_search": false}}]'))
HIGHQ_MIDM_SHAPES = _decode_capture(_json_loads('[{"label": "dispatch_q256_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 256, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610206, "self_search": false}}, {"label": "dispatch_q512_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 512, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610207, "self_search": false}}, {"label": "dispatch_q1024_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 1024, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610208, "self_search": false}}, {"label": "dispatch_q2048_m65536_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 2048, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610209, "self_search": false}}, {"label": "rag_q4096_m20000_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 20000, "Q": 4096, "benchmark": true, "check_correctness": true, "dtype": "bfloat16", "min_recall": 0.999, "seed": 3, "self_search": false}}, {"label": "rag_batch_q4096_m20000_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610110, "self_search": false}}, {"label": "dispatch_q4096_m16384_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 16384, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610210, "self_search": false}}, {"label": "dispatch_q4096_m32768_d128_k10", "params": {"B": 1, "D": 128, "K": 10, "M": 32768, "Q": 4096, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610211, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k1", "params": {"B": 1, "D": 128, "K": 1, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610310, "self_search": false}}, {"label": "ksweep_q4096_m20000_d128_k2", "params": {"B": 1, "D": 128, "K": 2, "M": 20000, "Q": 4096, "dtype": "bfloat16", "min_recall": 1.0, "seed": 610311, "self_search": false}}]'))
ROUND40_DISPATCHER_SHAPES = [*HIGHQ_MIDM_SHAPES, *parent.ROUND39_DISPATCHER_SHAPES]
ROUND40_PRESERVE_SHAPES = [*HIGHQ_MIDM_SHAPES, *parent.ROUND39_PRESERVE_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_highq_midm_qbucket_k10', 'guard': 'B == 1 and 256 <= Q <= 2048 and 16384 <= M <= 65536 and D == 128 and K <= 10 and tcgen05', 'route': 'round40_highq_midm_qbucket'}, {'shape_key': 'd128_q4096_midm_lowk_split8', 'guard': 'B == 1 and Q == 4096 and 16384 <= M <= 32768 and D == 128 and K <= 10 and tcgen05', 'route': 'round40_q4096_split8'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_highq_qbucket(inputs: dict[str, Any]) -> bool:
    return highq_qbucket._use_highq_qbucket(inputs)

def _use_q4096_split8(inputs: dict[str, Any]) -> bool:
    return q4096_split8._use_q4096_split8(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_highq_qbucket(inputs):
        return 'round40_highq_midm_qbucket'
    if _use_q4096_split8(inputs):
        return 'round40_q4096_split8'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_qbucket(inputs):
        return highq_qbucket.launch_for_eval(inputs)
    if _use_q4096_split8(inputs):
        return q4096_split8.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_highq_midm(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGHQ_MIDM_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round40_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND40_DISPATCHER_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round40_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND40_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
