"""kNN non-D128 frontier seed with exact D96 MMA.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
preserves the round-104 D320 exact-tail seed for D192, D320, and D768, and
routes only ``build_dim_sweep_b1_q1024_m1024_d96_k10`` through an exact-width
D96 tcgen05/TMA producer. The D96 stage uses 64B-swizzled BF16 SMEM rows and
issues a K=96 tcgen05 dot-product tile instead of padding D96 to D128.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_non128_frontier_8227_d320tail_v1 as d320tail
MODULE = 'loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_4be7_d96exact_v1'
BLOCK_Q = d320tail.BLOCK_Q
BLOCK_M = d320tail.BLOCK_M
K_TILE = d320tail.K_TILE
TOP_K_MAX = d320tail.TOP_K_MAX
THREADS = d320tail.THREADS
MERGE_THREADS = d320tail.MERGE_THREADS
GRID_DIM_DEFAULT = d320tail.GRID_DIM_DEFAULT
D96_FEAT_D = 96
D96_QUERY_BYTES = BLOCK_Q * D96_FEAT_D * 2
D96_DATABASE_BYTES = BLOCK_M * D96_FEAT_D * 2
DB_SQ_BYTES = BLOCK_M * 4
D96_SHAPE = 'build_dim_sweep_b1_q1024_m1024_d96_k10'
TARGET_SHAPES = d320tail.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"build_dim_sweep_b1_q1024_m1024_d96_k10": {"B": 1, "D": 96, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 1, "split_count": 8}, "build_dim_sweep_b1_q2048_m2048_d192_k10": {"B": 1, "D": 192, "K": 10, "M": 2048, "Q": 2048, "build": true, "feature_chunks": 2, "split_count": 8}, "build_highd_b1_q1024_m1024_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 3, "split_count": 8}, "rag_microbatch_highd_b1_q16_m50000_d768_k10": {"B": 1, "D": 768, "K": 10, "M": 50000, "Q": 16, "build": false, "feature_chunks": 6, "split_count": 72}, "search_rect_highd_b1_q512_m12000_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 12000, "Q": 512, "build": false, "feature_chunks": 3, "split_count": 32}}'))
stage1_d96exact_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:stage1_d96exact_ir"}'))
stage1_d320tail_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:stage1_d320tail_ir"}'))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:stage1_d256_ir"}'))
stage1_m64_d768_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:stage1_m64_d768_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:merge_ir"}'))
_TMAP64_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
knn_build_non128_frontier_4be7_d96exact_stage1 = _ir_proxy('loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:knn_build_non128_frontier_4be7_d96exact_stage1', 256)
stage1_d96exact_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:stage1_d96exact_ir"}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_4BE7_D96EXACT_VERIFY_KERNEL')
    if verify_kernel == 'stage1_d96exact':
        return stage1_d96exact_ir
    if verify_kernel == 'stage1_d320tail':
        return stage1_d320tail_ir
    if verify_kernel == 'stage1_d256':
        return stage1_d256_ir
    if verify_kernel == 'stage1_m64_d768':
        return stage1_m64_d768_ir
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_d96exact_ir
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_4be7_d96exact_v1:ir"}'))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return d320tail._target_label_for_inputs(inputs)

def _uses_d96exact(label: str | None) -> bool:
    return label == D96_SHAPE

def _split_count_for_label(label: str) -> int:
    if _uses_d96exact(label):
        env_key = f'LOOM_KNN_NON128_FRONTIER_4BE7_D96EXACT_SPLITS_{label.upper().replace('-', '_')}'
        override = os.environ.get(env_key)
        if override is not None:
            return int(override)
    return d320tail._split_count_for_label(label)

def _feature_dim_for_label(label: str) -> int:
    if _uses_d96exact(label):
        return D96_FEAT_D
    return d320tail._feature_dim_for_label(label)

def _producer_for_label(label: str) -> str:
    if _uses_d96exact(label):
        return 'd96_exact64b'
    return d320tail._producer_for_label(label)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return d320tail.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return f'{ROUTE_PREFIX}:{label}:d{int(spec['D'])}:s{_split_count_for_label(label)}:{_producer_for_label(label)}'

def _create_tensor_map_3d_oob_zero_swizzle64(data_ptr: int, global_height: int, shared_height: int, width: int, block_width: int):
    import torch
    from cuda.bindings import driver
    from .._dispatch_runtime import _tmap_to_device
    if width % 32 != 0 or block_width % 32 != 0:
        raise ValueError(f'64B BF16 tensor map requires width/block_width multiples of 32, got {width}/{block_width}')
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), int(width), int(block_width))
    cached = _TMAP64_CACHE.get(key)
    if cached is not None:
        return cached
    err, tmap = driver.cuTensorMapEncodeTiled(driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, data_ptr, [driver.cuuint64_t(32), driver.cuuint64_t(global_height), driver.cuuint64_t(width // 32)], [driver.cuuint64_t(width * 2), driver.cuuint64_t(64)], [driver.cuuint32_t(32), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // 32)], [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)], driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B, driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)
    if err != 0:
        raise RuntimeError(f'cuTensorMapEncodeTiled (3D, 64B OOB zero) failed: CUresult={err}')
    cached = _tmap_to_device(tmap).to(device=torch.device('cuda', device_index))
    _TMAP64_CACHE[key] = cached
    return cached

def _compiled_d96exact_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0021"}'))

def _launch_d96exact(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_4be7_d96exact_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    if dim != D96_FEAT_D:
        raise ValueError(f'D96 exact route expected D={D96_FEAT_D}, got {dim}')
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = _create_tensor_map_3d_oob_zero_swizzle64(query.data_ptr(), bsz * n_query, BLOCK_Q, D96_FEAT_D, D96_FEAT_D)
    tmap_database = _create_tensor_map_3d_oob_zero_swizzle64(database.data_ptr(), bsz * n_database, BLOCK_M, D96_FEAT_D, D96_FEAT_D)
    _compiled_d96exact_stage1().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=[inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, tmap_query, tmap_database, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=stage1_d96exact_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        d320tail.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if _uses_d96exact(label):
        _launch_d96exact(inputs, label)
        return
    d320tail.launch_from_contract_inputs(inputs, force_fallback=False)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return d320tail._select_contract_shapes(shape_labels)

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

def benchmark_knn_build_non128_frontier_4be7_d96exact_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
    return d320tail._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback:
            rows.append({'shape_key': inputs['label'], 'selected_route': d320tail.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': f'{d320tail.MODULE}:launch_from_contract_inputs', 'selected_seed': None, 'expected_seed': 'non128_frontier_4be7_d96exact_v1' if inputs['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'fallback', 'route_source': '8227-d320tail-parent-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'd320tail_parent_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        uses_d96exact = _uses_d96exact(label)
        feature_dim = _feature_dim_for_label(label)
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': f'{MODULE}:launch_from_contract_inputs', 'selected_seed': 'non128_frontier_4be7_d96exact_v1', 'expected_seed': 'non128_frontier_4be7_d96exact_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '4be7_d96exact_non128_exact_shape_guard', 'guard_condition': f'exact BF16 B={spec['B']} Q={spec['Q']} M={spec['M']} D={spec['D']} K={spec['K']} build={spec['build']}', 'feature_dim': feature_dim, 'split_count': _split_count_for_label(label), 'producer': _producer_for_label(label), 'preprocess_stage': None if uses_d96exact else d320tail.route_trace_for_contract_shapes((label,))[0].get('preprocess_stage'), 'source_route': 'd96_exact_64b_swizzle' if uses_d96exact else d320tail.route_for_contract_inputs(inputs), 'classification': 'd96-exact64b' if uses_d96exact else 'd320tail-parent'})
    return rows
