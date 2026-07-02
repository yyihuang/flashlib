"""Flash-KMeans high-D paired packed-partial R2 Split-K candidate.

Minimum architecture: sm_100a. This additive bucket-kernel candidate keeps the
Blackwell tcgen05/TMEM G2 distance-product producer on the contract-visible
path for paired B=1,N=2048,K=4096 high-D rows, but replaces separate
partial-score and partial-index buffers with one ordered u64 key per row and
K-slice. A two-lane reducer scans packed keys and writes cluster IDs.
Huge-K focus rows delegate to the validated 4f2c stream-ordered child. It is
not intended for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_streamdep_4f2c_v1 as _streamdep
BLOCK_N = _g2r4.BLOCK_N
MMA_BLOCK_K = _g2r4.MMA_BLOCK_K
FEAT_CHUNK = _g2r4.FEAT_CHUNK
ROW16_LOAD_COUNT = _g2r4.ROW16_LOAD_COUNT
ROW16_LOGICAL_CHUNK_K = _g2r4.ROW16_LOGICAL_CHUNK_K
CSQ_STAGE_VEC = _g2r4.CSQ_STAGE_VEC
X_TILE_BYTES = _g2r4.X_TILE_BYTES
C_TILE_BYTES = _g2r4.C_TILE_BYTES
CSQ_TILE_BYTES = _g2r4.CSQ_TILE_BYTES
DEFAULT_SPLITK_GROUP_K_TILES = _g2r4.SPLITK_GROUP_K_TILES
SPLITK_GROUP_K_TILES_ENV = 'LOOM_FLASH_KMEANS_HIGHD_PAIRED_PACKEDPARTIAL_R2_7B3C_GROUP_K_TILES'

def _configured_splitk_group_k_tiles() -> int:
    raw = os.environ.get(SPLITK_GROUP_K_TILES_ENV)
    if raw is None or raw == '':
        return DEFAULT_SPLITK_GROUP_K_TILES
    value = int(raw)
    if value not in (1, DEFAULT_SPLITK_GROUP_K_TILES):
        raise ValueError(f'{SPLITK_GROUP_K_TILES_ENV} must be 1 or {DEFAULT_SPLITK_GROUP_K_TILES}, got {value}')
    return value
SPLITK_GROUP_K_TILES = _decode_capture(_json_loads('2'))
SPLITK_TILE_K = MMA_BLOCK_K * SPLITK_GROUP_K_TILES
SPLITK_GRID_CAP = _g2r4.SPLITK_GRID_CAP
SUPPORTED_DIMS = set(_base.SUPPORTED_DIMS)
BF16_DTYPE_NAMES = set(_base.BF16_DTYPE_NAMES)
PAIRED_PACKEDPARTIAL_DIMS = {448, 512}
NUM_COMPUTE_WARPS = _g2r4.NUM_COMPUTE_WARPS
REDUCE_LANES_PER_ROW = 2
REDUCE_THREADS = BLOCK_N * REDUCE_LANES_PER_ROW
U32_MASK = 4294967295
if SPLITK_GROUP_K_TILES == DEFAULT_SPLITK_GROUP_K_TILES:
    ROUTE_ID = 'highd_paired_packedpartial_r2_7b3c_v1'
    SEED_ID = 'highd-paired-packedpartial-r2-7b3c-v1'
else:
    ROUTE_ID = 'highd_paired_packedpartial_r2_g1_tileprobe_7b3c_v1'
    SEED_ID = 'highd-paired-packedpartial-r2-g1-tileprobe-7b3c-v1'
flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1 = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "cta_group": 1, "threads": 192}'))
flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1 = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:reduce_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_FLASH_KMEANS_HIGHD_PAIRED_PACKEDPARTIAL_R2_7B3C_VERIFY_KERNEL')
    if verify_kernel == 'reduce':
        return reduce_ir
    if verify_kernel == 'streamdep_partial':
        return _streamdep.partial_ir
    if verify_kernel == 'streamdep_reduce':
        return _streamdep.reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=1)
def _loaded_partial_key_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _base._compile_ir(partial_ir)
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

@lru_cache(maxsize=1)
def _loaded_reduce_key_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _base._compile_ir(reduce_ir)
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _use_paired_packedpartial(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        num_n_tiles = n_points // BLOCK_N
        k_tiles = n_clusters // MMA_BLOCK_K
        k_slices = k_tiles // SPLITK_GROUP_K_TILES
        total_work = bsz * num_n_tiles * k_slices
        _launch_paired_packedpartial(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices, total_work=total_work)
        trace = _route_trace_paired(inputs, total_work=total_work, grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), k_slices=k_slices)
        return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _streamdep.launch_for_eval(inputs)
    trace = _wrap_child_trace(inputs, child_outputs=outputs)
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _use_paired_packedpartial(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim in PAIRED_PACKEDPARTIAL_DIMS)

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    del bsz
    dtype_name = _base._dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(f'{ROUTE_ID} requires bfloat16 input, got {dtype_name}')
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f'{ROUTE_ID} supports D={sorted(SUPPORTED_DIMS)}, got {dim}')
    if n_points % BLOCK_N != 0:
        raise ValueError(f'N must be divisible by BLOCK_N={BLOCK_N}, got {n_points}')
    if n_clusters % MMA_BLOCK_K != 0:
        raise ValueError(f'K must be divisible by MMA_BLOCK_K={MMA_BLOCK_K}, got {n_clusters}')

def _launch_paired_packedpartial(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> None:
    import torch
    partial_keys = _partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _loaded_partial_key_kernel()
    partial_args = [inputs['c_sq'], partial_keys, tmap_x, tmap_c, bsz, n_points, dim, n_clusters, num_n_tiles, k_tiles, k_slices]
    partial_kernel.launch(grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _loaded_reduce_key_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reduce_kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)

def _partial_key_buffer(inputs: dict[str, Any], total_work: int) -> Any:
    import torch
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']), int(total_work), int(BLOCK_N), int(SPLITK_GROUP_K_TILES))
    cached = inputs.get('_flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1_keys')
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    keys = torch.empty((total_work, BLOCK_N), dtype=torch.int64, device=inputs['x'].device)
    inputs['_flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1_keys'] = (cache_key, keys)
    return keys

def _route_trace_paired(inputs: dict[str, Any], *, total_work: int, grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    return {'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_packedpartial_r2_7b3c_v1', 'guard_condition': f'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448, 512]; grouped_mma_k_tiles={SPLITK_GROUP_K_TILES}', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': grid, 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'producer_work_items': total_work, 'producer_to_consumer': 'u64_partial_key_buffer_plus_r2_reduce', 'partial_key': 'ordered_f32_score_high32_inverse_cluster_index_low32', 'reducer_lanes_per_row': REDUCE_LANES_PER_ROW, 'producer_to_reducer_sync': 'same_stream_ordering', 'tile_probe_env': SPLITK_GROUP_K_TILES_ENV if SPLITK_GROUP_K_TILES != DEFAULT_SPLITK_GROUP_K_TILES else None}, 'reason': f'paired high-D row keeps the G2 tcgen05 Split-K producer but packs score/index partials into one u64 buffer and scans keys with a two-lane row reducer; grouped_mma_k_tiles={SPLITK_GROUP_K_TILES}'}

def _wrap_child_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any]) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    child_route = child_outputs.get('selected_route') or child_trace.get('selected_route')
    trace = dict(child_trace)
    trace.update({'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_packedpartial_r2_7b3c_v1', 'guard_condition': 'dtype == bfloat16 and D in [384,448,512] and N % 64 == 0 and K % 256 == 0; paired B=1,N=2048,K=4096,D in [448,512] uses packed partial keys, all other focus rows use highd_splitk_blockn64_g2r4_streamdep_4f2c_v1', 'classification': 'route-ok', 'child_route': child_route, 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'reason': 'non-paired high-D focus row keeps the stream-ordered G2/R4 Split-K child'})
    return trace

def compile_and_launch_flash_kmeans_assign_highd_paired_packedpartial(B: int=1, N: int=2048, K: int=4096, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(51202)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': f'manual_b{B}_n{N}_k{K}_d{D}', 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref))}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': f'manual_b{B}_n{N}_k{K}_d{D}', 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 51202}}])
    return result
