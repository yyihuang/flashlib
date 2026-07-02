"""kNN non-D128 frontier seed combining exact producers with cached merges.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
combines the source-policy-clean non-D128 repairs already present in this
worktree: D96 exact-width tcgen05/TMA, D320 exact-tail tcgen05/TMA, and the
D768 fused split merge. It changes the remaining split8 K10 build-row
consumers for D96, D192, and D320 build to a constexpr row-base cached merge.
D320 search and D768 delegate to the validated fused parent route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
import argparse
import json
import os
from functools import lru_cache
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_non128_frontier_4be7_d768fused_v1 as fused_parent
MODULE = 'loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_3d5a_cachedmerge_v1'
d96exact = fused_parent.d96exact
d320tail = d96exact.d320tail
widecombine = d320tail.wide_m64.widecombine
BLOCK_Q = d96exact.BLOCK_Q
BLOCK_M = d96exact.BLOCK_M
TOP_K_MAX = d96exact.TOP_K_MAX
THREADS = d96exact.THREADS
GRID_DIM_DEFAULT = d96exact.GRID_DIM_DEFAULT
FAST_MERGE_THREADS = 32
D96_BUILD_SHAPE = d96exact.D96_SHAPE
D192_BUILD_SHAPE = 'build_dim_sweep_b1_q2048_m2048_d192_k10'
D320_BUILD_SHAPE = d320tail.D320_BUILD_SHAPE
CACHED_SPLIT8_SHAPES = {D96_BUILD_SHAPE, D192_BUILD_SHAPE, D320_BUILD_SHAPE}
TARGET_SHAPES = fused_parent.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"build_dim_sweep_b1_q1024_m1024_d96_k10": {"B": 1, "D": 96, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 1, "split_count": 8}, "build_dim_sweep_b1_q2048_m2048_d192_k10": {"B": 1, "D": 192, "K": 10, "M": 2048, "Q": 2048, "build": true, "feature_chunks": 2, "split_count": 8}, "build_highd_b1_q1024_m1024_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 1024, "Q": 1024, "build": true, "feature_chunks": 3, "split_count": 8}, "rag_microbatch_highd_b1_q16_m50000_d768_k10": {"B": 1, "D": 768, "K": 10, "M": 50000, "Q": 16, "build": false, "feature_chunks": 6, "split_count": 72}, "search_rect_highd_b1_q512_m12000_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 12000, "Q": 512, "build": false, "feature_chunks": 3, "split_count": 32}}'))
stage1_d96exact_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:stage1_d96exact_ir"}'))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:stage1_d256_ir"}'))
stage1_d320tail_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:stage1_d320tail_ir"}'))
stage1_m64_d768_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:stage1_m64_d768_ir"}'))
fused_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:fused_merge_ir"}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:merge_generic_ir"}'))
knn_build_non128_frontier_3d5a_k10_merge_s8_rowbase_cache = _ir_proxy('loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:knn_build_non128_frontier_3d5a_k10_merge_s8_rowbase_cache', 256)
merge_k10_s8_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:merge_k10_s8_ir"}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_3D5A_CACHEDMERGE_VERIFY_KERNEL')
    if verify_kernel == 'merge_s8':
        return merge_k10_s8_ir
    if verify_kernel == 'stage1_d96exact':
        return stage1_d96exact_ir
    if verify_kernel == 'stage1_d256':
        return stage1_d256_ir
    if verify_kernel == 'stage1_d320tail':
        return stage1_d320tail_ir
    if verify_kernel == 'stage1_m64_d768':
        return stage1_m64_d768_ir
    if verify_kernel == 'fused_merge':
        return fused_parent._fused_merge_ir(fused_parent.D768_SPLIT_COUNT, fused_parent.D768_GROUP_COUNT)
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return merge_k10_s8_ir
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:ir"}'))

def _compiled_merge_k10_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0022"}'))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return fused_parent._target_label_for_inputs(inputs)

def _split_count_for_label(label: str) -> int:
    return fused_parent._split_count_for_label(label)

def _uses_cached_split8(label: str | None) -> bool:
    return label in CACHED_SPLIT8_SHAPES and _split_count_for_label(str(label)) == 8

def _feature_dim_for_label(label: str) -> int:
    return fused_parent._feature_dim_for_label(label)

def _producer_for_label(label: str) -> str:
    if not _uses_cached_split8(label):
        return fused_parent._producer_for_label(label)
    return f'{fused_parent._producer_for_label(label)}_cachedmerge_s8'

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return fused_parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if not _uses_cached_split8(label):
        return fused_parent.route_for_contract_inputs(inputs, force_fallback=False)
    spec = SHAPE_SPECS[label]
    return f'{ROUTE_PREFIX}:{label}:d{int(spec['D'])}:s8:{_producer_for_label(label)}'

def _launch_cached_merge(*, partial_dists, partial_indices, out_dists, out_indices, bsz: int, n_query: int) -> None:
    merge_grid = min((bsz * n_query + FAST_MERGE_THREADS - 1) // FAST_MERGE_THREADS, GRID_DIM_DEFAULT)
    _compiled_merge_k10_s8().launch(grid=(merge_grid, 1, 1), block=(FAST_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, out_dists, out_indices, bsz * n_query], shared_mem=merge_k10_s8_ir.computed_smem_bytes)

def _partial_buffers(*, split_count: int, bsz: int, n_query: int, top_k: int, device):
    return split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=device)

def _launch_d96_cached(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_3d5a_cachedmerge_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    if split_count != 8 or top_k != TOP_K_MAX or dim != d96exact.D96_FEAT_D:
        fused_parent.launch_from_contract_inputs(inputs, force_fallback=False)
        return
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = _partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = d96exact._create_tensor_map_3d_oob_zero_swizzle64(query.data_ptr(), bsz * n_query, BLOCK_Q, d96exact.D96_FEAT_D, d96exact.D96_FEAT_D)
    tmap_database = d96exact._create_tensor_map_3d_oob_zero_swizzle64(database.data_ptr(), bsz * n_database, BLOCK_M, d96exact.D96_FEAT_D, d96exact.D96_FEAT_D)
    d96exact._compiled_d96exact_stage1().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=[inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, tmap_query, tmap_database, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=stage1_d96exact_ir.computed_smem_bytes)
    _launch_cached_merge(partial_dists=partial_dists, partial_indices=partial_indices, out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], bsz=bsz, n_query=n_query)

def _launch_d192_cached(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_3d5a_cachedmerge_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    feature_dim = 256
    if split_count != 8 or top_k != TOP_K_MAX:
        fused_parent.launch_from_contract_inputs(inputs, force_fallback=False)
        return
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = _partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    database_tma = widecombine._pad_if_needed(database, rows=bsz * n_database, src_cols=dim, dst_cols=feature_dim)
    if query.data_ptr() == database.data_ptr():
        query_tma = database_tma
    else:
        query_tma = widecombine._pad_if_needed(query, rows=bsz * n_query, src_cols=dim, dst_cols=feature_dim)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query_tma.data_ptr(), bsz * n_query, BLOCK_Q, feature_dim, feature_dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database_tma.data_ptr(), bsz * n_database, BLOCK_M, feature_dim, feature_dim)
    widecombine.wide_d256._compiled_d256_stage1().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=[inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, tmap_query, tmap_database, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=stage1_d256_ir.computed_smem_bytes)
    _launch_cached_merge(partial_dists=partial_dists, partial_indices=partial_indices, out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], bsz=bsz, n_query=n_query)

def _launch_d320_build_cached(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_3d5a_cachedmerge_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    if split_count != 8 or top_k != TOP_K_MAX or dim != d320tail.D320_FEAT_D:
        fused_parent.launch_from_contract_inputs(inputs, force_fallback=False)
        return
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = _partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, d320tail.D320_FEAT_D, d320tail.D320_FEAT_D)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, d320tail.D320_FEAT_D, d320tail.D320_FEAT_D)
    d320tail._compiled_d320tail_stage1().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=[inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, tmap_query, tmap_database, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=stage1_d320tail_ir.computed_smem_bytes)
    _launch_cached_merge(partial_dists=partial_dists, partial_indices=partial_indices, out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], bsz=bsz, n_query=n_query)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None or (not _uses_cached_split8(label)):
        fused_parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if label == D96_BUILD_SHAPE:
        _launch_d96_cached(inputs, label)
        return
    if label == D192_BUILD_SHAPE:
        _launch_d192_cached(inputs, label)
        return
    if label == D320_BUILD_SHAPE:
        _launch_d320_build_cached(inputs, label)
        return
    fused_parent.launch_from_contract_inputs(inputs, force_fallback=False)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return fused_parent._select_contract_shapes(shape_labels)

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

def benchmark_knn_build_non128_frontier_3d5a_cachedmerge_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
    return fused_parent._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback or (not _uses_cached_split8(label)):
            row = fused_parent.route_trace_for_contract_shapes((inputs['label'],), force_fallback=force_fallback)[0]
            if label in TARGET_SHAPE_SET:
                row['expected_seed'] = 'non128_frontier_3d5a_cachedmerge_v1'
            rows.append(row)
            continue
        spec = SHAPE_SPECS[label]
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': f'{MODULE}:launch_from_contract_inputs', 'selected_seed': 'non128_frontier_3d5a_cachedmerge_v1', 'expected_seed': 'non128_frontier_3d5a_cachedmerge_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '3d5a_cachedmerge_non128_exact_shape_guard', 'guard_condition': f'exact BF16 B={spec['B']} Q={spec['Q']} M={spec['M']} D={spec['D']} K={spec['K']} build={spec['build']} split_count=8', 'feature_dim': _feature_dim_for_label(label), 'split_count': 8, 'producer': fused_parent._producer_for_label(label), 'merge_route': 'k10_s8_rowbase_cached', 'source_route': fused_parent.route_for_contract_inputs(inputs), 'classification': 'cached-merge-s8'})
    return rows
