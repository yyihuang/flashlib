"""Exact-D768 no-padding Split-K Flash-KMeans assignment seed.

Minimum architecture: sm_100a.  This exact priority-route reuses the proven
tcgen05/TMEM BLOCK_N=64 G1/R4 producer and four-lane Weave Split-K reducer,
but extends its runtime feature-tile loop to D=768 (twelve 64-wide chunks).
The producer writes per-slice argmax state to global partial buffers; the
reducer consumes those buffers to produce contract-visible ``cluster_ids``.
No D padding or non-Weave fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1 as _g1
TARGET_D = 768
ROUTE_ID = 'd768_no_padding_splitk_priority_d768_exact_seed_v1'
SEED_ID = 'd768-no-padding-splitk-priority-d768-exact-seed-v1'
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1:ir"}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    if dim != TARGET_D:
        raise ValueError(f'{ROUTE_ID} requires exact D={TARGET_D}, got D={dim}')
    dtype_name = _g1._base._dtype_name(inputs)
    if dtype_name not in _g1.BF16_DTYPE_NAMES:
        raise ValueError(f'{ROUTE_ID} requires bfloat16 input, got {dtype_name}')
    if n_points % _g1.BLOCK_N != 0 or n_clusters % _g1.MMA_BLOCK_K != 0:
        raise ValueError(f'{ROUTE_ID} requires N divisible by {_g1.BLOCK_N} and K divisible by {_g1.MMA_BLOCK_K}; got N={n_points}, K={n_clusters}')
    num_n_tiles = n_points // _g1.BLOCK_N
    k_tiles = n_clusters // _g1.MMA_BLOCK_K
    if num_n_tiles > _g1.MAX_POINT_TILES or k_tiles < _g1.SPLITK_MIN_K_TILES:
        raise ValueError(f'{ROUTE_ID} requires N/64 <= {_g1.MAX_POINT_TILES} and K/256 >= {_g1.SPLITK_MIN_K_TILES}; got N={n_points}, K={n_clusters}')
    k_slices = k_tiles // _g1.SPLITK_GROUP_K_TILES
    _g1._launch_blockn64_splitk(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices)
    total_work = bsz * num_n_tiles * k_slices
    trace = _g1._route_trace(inputs, total_work=total_work, grid=(min(total_work, _g1.SPLITK_GRID_CAP), 1, 1), k_slices=k_slices)
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1:launch_for_eval', 'selected_seed': SEED_ID, 'guard_id': 'guard_exact_d768_no_padding_splitk_priority_d768_exact_seed_v1', 'guard_condition': 'dtype == bfloat16 and D == 768 and N % 64 == 0 and K % 256 == 0', 'feature_tiles': TARGET_D // _g1.FEAT_CHUNK, 'reason': 'exact-D768 G1/R4 tcgen05 Split-K seed; twelve feature chunks with no D padding'})
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
