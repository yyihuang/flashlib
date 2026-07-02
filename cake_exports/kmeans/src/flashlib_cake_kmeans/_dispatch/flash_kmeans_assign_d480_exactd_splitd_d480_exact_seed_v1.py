"""Flash-KMeans exact-D480 staged seed.

Minimum architecture: sm_100a.  The contract-visible score producer is the
32-wide tcgen05/TMEM split-D core and accumulates all fifteen D=32 chunks
before the TMEM-to-register argmax writes ``cluster_ids``.  This is B200/B300
only; sm_120a/sm_121a reject tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2 as _core
BLOCK_N = _core.BLOCK_N
BLOCK_K = _core.BLOCK_K
CHUNK_D = _core.CHUNK_D
FEAT_D = 480
ROUTE_ID = 'd480_exactd_splitd_d480_exact_seed_v1'
SEED_ID = 'd480-exactd-splitd-d480-exact-seed-v1'
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.flash_kmeans_assign_d480_exactd_splitd_d480_exact_seed_v1:ir"}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    dim = int(inputs['D'])
    if dim != FEAT_D:
        raise ValueError(f'{ROUTE_ID} requires exact D={FEAT_D}, got {dim}')
    if int(inputs['N']) % BLOCK_N:
        raise ValueError(f'N must be divisible by {BLOCK_N}')
    if int(inputs['K']) % BLOCK_K:
        raise ValueError(f'K must be divisible by {BLOCK_K}')
    outputs = _core.launch_for_eval(inputs)
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': {'selected_route': ROUTE_ID, 'selected_seed': SEED_ID, 'route_kind': 'shape_specific_seed', 'guard_condition': 'dtype == bfloat16 and D == 480 and N % 128 == 0 and K % 256 == 0', 'producer': 'tcgen05_mma D=32 split-D -> TMEM -> register argmax -> cluster_ids'}}
