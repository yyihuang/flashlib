"""Paired exact RAG online/stream K10 split-64 route for kNN build/search.

Minimum target architecture: sm_100a. This additive shape-kernel candidate
routes exactly ``rag_online_b1_q1_m100000_d128_k10`` and
``rag_stream_b1_q128_m100000_d128_k10`` through the existing K10 tcgen05/TMA
stage-1 producer with a 64-way database split and a K10/S64 cached row-base
merge. Guard misses delegate to the c454 Weave dispatcher; no external runtime
fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from dataclasses import replace
from functools import lru_cache
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_combined_k20rag_weave_evolve_knn_build_c454_v1 as c454_dispatcher
from . import knn_build_rag_online_stream_801d_v45 as baseline_pair
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_v1 as parent_lowk
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_k32
from . import knn_build_evolve_7bfc_v1 as base_v1
ONLINE_SHAPE = baseline_pair.ONLINE_SHAPE
STREAM_SHAPE = baseline_pair.STREAM_SHAPE
TARGET_SHAPES = baseline_pair.TARGET_SHAPES
SPLIT_COUNT = 64
MERGE_THREADS = 32

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return replace(ir_obj, name=f'{ir_obj.name}_{suffix}', constants=constants)
merge_k10_s64_cache_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_rag_online_stream_split64_3d97_v1:merge_k10_s64_cache_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_SPLIT64_3D97_VERIFY_KERNEL')
    if verify_kernel == 'merge_k10_s64_cache':
        return merge_k10_s64_cache_ir
    return parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_rag_online_stream_split64_3d97_v1:ir", "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "cta_group": 1, "threads": 192}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, f'kernel_{ir_obj.name}')

def _compiled_merge_k10_s64_cache():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0078"}'))

def _eligible_rag_online_stream_split64(inputs: dict[str, Any]) -> bool:
    return baseline_pair._eligible_rag_online_stream(inputs)

def _launch_rag_online_stream_split64(inputs: dict[str, Any]) -> None:
    parent_lowk._launch_k10_cached_path(inputs, split_count=SPLIT_COUNT, merge_threads=MERGE_THREADS, merge_kernel=_compiled_merge_k10_s64_cache(), merge_ir=merge_k10_s64_cache_ir)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_rag_online_stream_split64(inputs):
        _launch_rag_online_stream_split64(inputs)
        return
    c454_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return c454_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool) -> dict[str, Any]:
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_online_stream_split64_3d97_v1:benchmark_knn_build_rag_online_stream_split64_3d97_v1', 'accelerated_shape_labels': list(TARGET_SHAPES), 'producer_split_count': SPLIT_COUNT, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': {label: report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'report': report}

def benchmark_knn_build_rag_online_stream_split64_3d97_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark for the exact online and stream RAG rows."""
    report = _run_with_timing_backend(use_cupti=use_cupti)
    return _benchmark_payload(report, use_cupti=use_cupti)
