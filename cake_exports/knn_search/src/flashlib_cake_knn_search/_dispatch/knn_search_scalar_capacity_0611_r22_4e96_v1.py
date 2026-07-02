"""Round-22 scalar capacity fallback for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80 for the scalar fallback path; inherited
tcgen05 MMA routes from the parent still require sm_100a/sm_103a. This
additive candidate keeps the round-21 composed dispatcher for the D=128,
K<=32 frontier and adds a source-clean CUDA-core split-M route for K64 and
non-128D contract backlog shapes. The Weave kernels still use BF16 vec8 global
loads, so host-side launch pads non-vec8 feature dimensions with zeros before
dispatch. Padding preserves exact squared-L2 results because the extra
coordinates are zero in both query and database tensors.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch_compose_0611_r21_4e96_v1 as parent
THREADS = 256
NUM_WARPS = THREADS // 32
K_CAP_MAX = 64
BLOCK_M = 512
BF16_VEC_ELEMS = 8
DIRECT_DIST_BYTES = NUM_WARPS * K_CAP_MAX * 4
DIRECT_IDX_BYTES = NUM_WARPS * K_CAP_MAX * 4
DIRECT_SMEM_BYTES = DIRECT_DIST_BYTES + DIRECT_IDX_BYTES
MERGE_DIST_BYTES = THREADS * K_CAP_MAX * 4
MERGE_IDX_BYTES = THREADS * K_CAP_MAX * 4
MERGE_SMEM_BYTES = MERGE_DIST_BYTES + MERGE_IDX_BYTES
_SCALAR_CAPACITY_KERNELS: dict[tuple[int, int], dict[str, Any]] = {}
_SCALAR_CAPACITY_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_scalar_capacity_direct_v1 = _ir_proxy('loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:knn_search_scalar_capacity_direct_v1', 256)
knn_search_scalar_capacity_partial_v1 = _ir_proxy('loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:knn_search_scalar_capacity_partial_v1', 256)
knn_search_scalar_capacity_merge_v1 = _ir_proxy('loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:knn_search_scalar_capacity_merge_v1', 256)
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:ir"}'))

def _k_bucket(k: int) -> int:
    if k <= 10:
        return 10
    if k <= 32:
        return 32
    return 64

def _supports_feature_dim(d: int) -> bool:
    return int(d) > 0

def _kernel_d(d: int) -> int:
    d = int(d)
    if not _supports_feature_dim(d):
        raise ValueError(f'feature dimension must be positive, got D={d}')
    return max(BF16_VEC_ELEMS, (d + BF16_VEC_ELEMS - 1) // BF16_VEC_ELEMS * BF16_VEC_ELEMS)

def _vec8_aligned_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    d = int(inputs['D'])
    padded_d = _kernel_d(d)
    if padded_d == d:
        return inputs
    import torch
    queries = inputs['queries']
    database = inputs['database']
    padded_queries = torch.zeros((*queries.shape[:-1], padded_d), device=queries.device, dtype=queries.dtype)
    padded_database = torch.zeros((*database.shape[:-1], padded_d), device=database.device, dtype=database.dtype)
    padded_queries[..., :d] = queries
    padded_database[..., :d] = database
    aligned = dict(inputs)
    aligned['queries'] = padded_queries.contiguous()
    aligned['database'] = padded_database.contiguous()
    aligned['D'] = padded_d
    aligned['original_D'] = d
    return aligned

def _use_scalar_capacity(inputs: dict[str, Any]) -> bool:
    d = int(inputs['D'])
    k = int(inputs['K'])
    if not _supports_feature_dim(d) or k > K_CAP_MAX:
        return False
    return d != parent.D_STATIC or k > 32

def _unsupported_scalar_capacity_reason(inputs: dict[str, Any]) -> str | None:
    d = int(inputs['D'])
    k = int(inputs['K'])
    if not _supports_feature_dim(d):
        return f'unsupported D={d}; feature dimension must be positive'
    if k > K_CAP_MAX:
        return f'unsupported K={k}; scalar capacity route supports K <= {K_CAP_MAX}'
    return None

def _compile_kernels(d: int, k_cap: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_scalar_capacity_partial_v1, validate=False, smem_bytes=0, D_=int(d), K_CAP_=int(k_cap))
    merge_source = generate_kernel(knn_search_scalar_capacity_merge_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES, K_CAP_=int(k_cap))
    direct_source = generate_kernel(knn_search_scalar_capacity_direct_v1, validate=False, smem_bytes=DIRECT_SMEM_BYTES, D_=int(d), K_CAP_=int(k_cap))
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    direct_cubin = compile_cuda(direct_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, f'kernel_{knn_search_scalar_capacity_partial_v1.name}'), 'merge': CUDAKernel(merge_cubin, f'kernel_{knn_search_scalar_capacity_merge_v1.name}'), 'direct': CUDAKernel(direct_cubin, f'kernel_{knn_search_scalar_capacity_direct_v1.name}')}

def _scratch(inputs: dict[str, Any], num_m_tiles: int, k_cap: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(num_m_tiles), int(k_cap), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCALAR_CAPACITY_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), NUM_WARPS, int(k_cap))
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCALAR_CAPACITY_SCRATCH[key] = cached
    return cached

def _launch_scalar_capacity(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    original_d = int(inputs['D'])
    k = int(inputs['K'])
    if not _supports_feature_dim(original_d):
        raise ValueError(f'knn_search_scalar_capacity_0611_r22_4e96_v1 unsupported D={original_d}')
    if k > K_CAP_MAX:
        raise ValueError(f'knn_search_scalar_capacity_0611_r22_4e96_v1 supports K <= {K_CAP_MAX}, got K={k}')
    kernel_inputs = _vec8_aligned_inputs(inputs)
    d = int(kernel_inputs['D'])
    k_cap = _k_bucket(k)
    key = (d, k_cap)
    kernels = _SCALAR_CAPACITY_KERNELS.get(key)
    if kernels is None:
        kernels = _compile_kernels(d, k_cap)
        _SCALAR_CAPACITY_KERNELS[key] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_m_tiles = math.ceil(m_rows / BLOCK_M)
    if num_m_tiles == 1:
        kernels['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[kernel_inputs['queries'], kernel_inputs['database'], kernel_inputs['out_distances'], kernel_inputs['out_indices'], bsz, q_rows, m_rows, k], shared_mem=DIRECT_SMEM_BYTES)
        torch.cuda.synchronize()
        return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
    partial_dist, partial_idx = _scratch(kernel_inputs, num_m_tiles, k_cap)
    kernels['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[kernel_inputs['queries'], kernel_inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, num_m_tiles], shared_mem=0)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[partial_dist, partial_idx, kernel_inputs['out_distances'], kernel_inputs['out_indices'], bsz, q_rows, k, num_m_tiles], shared_mem=MERGE_SMEM_BYTES)
    torch.cuda.synchronize()
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_scalar_capacity(inputs):
        return _launch_scalar_capacity(inputs)
    unsupported_reason = _unsupported_scalar_capacity_reason(inputs)
    if unsupported_reason is not None:
        raise ValueError(f'knn_search_scalar_capacity_0611_r22_4e96_v1 unsupported shape: {unsupported_reason}')
    return parent.launch_for_eval(inputs)

def launch_scalar_capacity_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Force the scalar-capacity Weave route for guarded coverage repairs."""
    return _launch_scalar_capacity(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_scalar_capacity(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
