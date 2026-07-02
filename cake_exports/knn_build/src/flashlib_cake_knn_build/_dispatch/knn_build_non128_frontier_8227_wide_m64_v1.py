"""kNN non-D128 frontier seed combining wide D192/D320 and M64 D768 routes.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
preserves the round-8199 widecombine routes for D192/D320, keeps D96 on the
validated split-retuned path, and routes only
``rag_microbatch_highd_b1_q16_m50000_d768_k10`` through the round-7ee5 M64/N64
tcgen05/TMA producer. The contract-visible path remains Weave-only and writes
distances and indices through the existing split merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_non128_frontier_7ee5_m64rag_v1 as m64rag
from . import knn_build_non128_frontier_8199_widecombine_v1 as widecombine
MODULE = 'loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_8227_wide_m64_v1'
D768_SHAPE = m64rag.D768_SHAPE
TARGET_SHAPES = widecombine.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"build_dim_sweep_b1_q1024_m1024_d96_k10": {"B": 1, "D": 96, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 1, "split_count": 8}, "build_dim_sweep_b1_q2048_m2048_d192_k10": {"B": 1, "D": 192, "K": 10, "M": 2048, "Q": 2048, "build": true, "feature_chunks": 2, "split_count": 8}, "build_highd_b1_q1024_m1024_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 3, "split_count": 8}, "rag_microbatch_highd_b1_q16_m50000_d768_k10": {"B": 1, "D": 768, "K": 10, "M": 50000, "Q": 16, "build": false, "feature_chunks": 6, "split_count": 72}, "search_rect_highd_b1_q512_m12000_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 12000, "Q": 512, "build": false, "feature_chunks": 3, "split_count": 32}}'))
SHAPE_SPECS[D768_SHAPE]['split_count'] = m64rag._split_count_for_label(D768_SHAPE)
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:stage1_d256_ir"}'))
stage1_d384_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:stage1_d384_ir"}'))
stage1_m64_d768_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:stage1_m64_d768_ir"}'))
pad_bf16_rows_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:pad_bf16_rows_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:merge_ir"}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_8227_WIDE_M64_VERIFY_KERNEL')
    if verify_kernel == 'stage1_d256':
        return stage1_d256_ir
    if verify_kernel == 'stage1_d384':
        return stage1_d384_ir
    if verify_kernel == 'stage1_m64_d768':
        return stage1_m64_d768_ir
    if verify_kernel == 'stage1_chunks1':
        return widecombine.splitretune.stage1_d128_ir
    if verify_kernel == 'pad_d96':
        return widecombine.splitretune.parent._pad_ir(128)
    if verify_kernel == 'pad_d192':
        return widecombine.splitretune.parent._pad_ir(256)
    if verify_kernel == 'pad_d320':
        return widecombine.splitretune.parent._pad_ir(384)
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_m64_d768_ir
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_8227_wide_m64_v1:ir"}'))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return widecombine._target_label_for_inputs(inputs)

def _uses_m64_d768(label: str | None) -> bool:
    return label == D768_SHAPE

def _split_count_for_label(label: str) -> int:
    if _uses_m64_d768(label):
        return m64rag._split_count_for_label(D768_SHAPE)
    return widecombine._split_count_for_label(label)

def _feature_dim_for_label(label: str) -> int:
    if _uses_m64_d768(label):
        return m64rag.M64_FEATURE_CHUNKS * m64rag.K_TILE
    return widecombine._feature_dim_for_label(label)

def _producer_for_label(label: str) -> str:
    if _uses_m64_d768(label):
        return 'm64_d768'
    return widecombine._producer_for_label(label)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return widecombine.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return f'{ROUTE_PREFIX}:{label}:d{int(spec['D'])}:s{_split_count_for_label(label)}:{_producer_for_label(label)}'

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        widecombine.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if _uses_m64_d768(label):
        m64rag.launch_from_contract_inputs(inputs, force_fallback=False)
        return
    widecombine.launch_from_contract_inputs(inputs, force_fallback=False)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return widecombine._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def benchmark_knn_build_non128_frontier_8227_wide_m64_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = None
    if use_cupti is not None:
        prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        if prior_use_cupti is not None:
            eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return widecombine._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback:
            rows.append({'shape_key': inputs['label'], 'selected_route': widecombine.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': widecombine.ROUTE_BASE_4247_ENTRYPOINT if hasattr(widecombine, 'ROUTE_BASE_4247_ENTRYPOINT') else widecombine.splitretune.ROUTE_FALLBACK, 'selected_seed': None, 'expected_seed': 'non128_frontier_8227_wide_m64_v1' if inputs['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'fallback', 'route_source': '8199-widecombine-parent-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'widecombine_parent_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        uses_m64 = _uses_m64_d768(label)
        feature_dim = _feature_dim_for_label(label)
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': f'{MODULE}:launch_from_contract_inputs', 'selected_seed': 'non128_frontier_8227_wide_m64_v1', 'expected_seed': 'non128_frontier_8227_wide_m64_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '8227_wide_m64_non128_exact_shape_guard', 'guard_condition': f'exact BF16 B={spec['B']} Q={spec['Q']} M={spec['M']} D={spec['D']} K={spec['K']} build={spec['build']}', 'feature_dim': feature_dim, 'split_count': _split_count_for_label(label), 'producer': _producer_for_label(label), 'preprocess_stage': f'd{int(spec['D'])}_weave_pad_to_d{feature_dim}' if int(spec['D']) != feature_dim and (not uses_m64) else None, 'source_route': m64rag.route_for_contract_inputs(inputs) if uses_m64 else widecombine.route_for_contract_inputs(inputs), 'classification': 'm64-d768' if uses_m64 else 'widecombine-parent'})
    return rows
