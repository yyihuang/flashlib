"""c492 extended dynamic-D generated-variant dispatcher.

Minimum target architecture: sm_100a for the generated tcgen05/TMEM route.
This wrapper consumes the existing ccef high-D direct-stride seed schedule for
the extended dynamic-D K10 bucket where the same schedule was validated with a
different ``D_ORIG_`` constexpr. It does not broaden unsupported tiny-D,
D>512, K64, or D128 K-capacity rows; guard misses delegate to c492.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_seed_portfolio_c492_v1 as base
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
SPLIT_M = base.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:current_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:base_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "cta_group": 1, "threads": 512}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q128_22d9_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "cta_group": 1, "threads": 640}'))
blockm640_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm640_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm640_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm640_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "cta_group": 1, "threads": 256}'))
b2_q128_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:b2_q128_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "cta_group": 1, "threads": 640}'))
b2_q128_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:b2_q128_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
blockm896_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm896_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
blockm896_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm896_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
blockm896_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:blockm896_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "cta_group": 1, "threads": 256}'))
q2_blockm640_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q2_blockm640_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
q2_blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q2_blockm640_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
q2_blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q2_blockm640_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "cta_group": 1, "threads": 256}'))
cc76_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:cc76_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
cc76_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:cc76_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "cta_group": 1, "threads": 128}'))
q3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q3_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
q3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q3_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "cta_group": 1, "threads": 128}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:scalar_capacity_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 256}'))
self_q2048_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:self_q2048_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "cta_group": 1, "threads": 640}'))
self_q2048_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:self_q2048_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
k1_merge8_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k1_merge8_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "cta_group": 1, "threads": 640}'))
k1_merge8_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k1_merge8_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 128}'))
q1_m262144_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q1_m262144_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "cta_group": 1, "threads": 256}'))
q1_m262144_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:q1_m262144_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "cta_group": 1, "threads": 256}'))
b2_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:b2_k64_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "cta_group": 1, "threads": 512}'))
b2_k64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:b2_k64_group_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
b2_k64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:b2_k64_final_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
d384_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:d384_q256_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "cta_group": 1, "threads": 640}'))
d384_q256_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:d384_q256_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
k64_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_q256_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "cta_group": 1, "threads": 512}'))
k64_q256_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_q256_group_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
k64_q256_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_q256_final_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
lowd_d256_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_d256_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "cta_group": 1, "threads": 256}'))
lowd_d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_d256_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "cta_group": 1, "threads": 256}'))
lowd_d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_d256_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
lowd_d256_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_d256_k64_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "cta_group": 1, "threads": 256}'))
lowd_d256_k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_d256_k64_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
lowd_dbscan_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:lowd_dbscan_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 12416, "cta_group": 1, "threads": 128}'))
tinyd_d3_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:tinyd_d3_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "cta_group": 1, "threads": 128}'))
tinyd_d3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:tinyd_d3_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "cta_group": 1, "threads": 128}'))
tinyd_d3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:tinyd_d3_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "cta_group": 1, "threads": 128}'))
tinyd_449d_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:tinyd_449d_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "cta_group": 1, "threads": 640}'))
tinyd_449d_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:tinyd_449d_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "cta_group": 1, "threads": 640}'))
highd_06f4_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:highd_06f4_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "cta_group": 1, "threads": 640}'))
highd_06f4_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:highd_06f4_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "cta_group": 1, "threads": 640}'))
k64_f0a3_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_f0a3_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "cta_group": 1, "threads": 256}'))
k64_f0a3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_f0a3_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "cta_group": 1, "threads": 256}'))
k64_f0a3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:k64_f0a3_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
remaining_a2ab_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:remaining_a2ab_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "cta_group": 1, "threads": 256}'))
d384_1e12_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:d384_1e12_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "cta_group": 1, "threads": 640}'))
d384_1e12_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:d384_1e12_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "cta_group": 1, "threads": 640}'))
d384_1e12_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:d384_1e12_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
ext_highd_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:ext_highd_partial_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "cta_group": 1, "threads": 640}'))
ext_highd_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:ext_highd_merge_ir", "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "cta_group": 1, "threads": 32}'))
PROFILE_BASE_C492 = base.PROFILE_ALL
PROFILE_ALL = 'c492_ext_dynamic_highd_9286'
ROUTE_BASE_C492 = base.PROFILE_ALL
ROUTE_EXT_DYNAMIC_HIGHD_9286 = '9286_ext_dynamic_d_highd_generated_variants'
CONSUMED_EXT_HIGHD_SEED = highd.CONSUMED_SEED
CONSUMED_SEEDS = (*base.CONSUMED_SEEDS, CONSUMED_EXT_HIGHD_SEED)
EXT_DYNAMIC_HIGHD_Q128_DIMS: frozenset[int] = frozenset({15, 31, 65, 127, 130, 255, 258})
EXT_DYNAMIC_HIGHD_LABELS: tuple[str, ...] = ('blind_ext_dyn_d15_q128_m65536_k10', 'blind_ext_dyn_d31_q128_m65536_k10', 'blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_ext_dyn_d130_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10', 'blind_ext_dyn_d258_q128_m65536_k10')
EXT_DYNAMIC_HIGHD_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d15_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 15], ["K", 10], ["dtype", "bfloat16"], ["seed", 610915], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 130], ["K", 10], ["dtype", "bfloat16"], ["seed", 610919], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d258_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 258], ["K", 10], ["dtype", "bfloat16"], ["seed", 610921], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_EXT_DYNAMIC_HIGHD_ENTRY: dict[str, Any] = {'overlay': 'ext_dynamic_highd_9286', 'shape_key': '9286_ext_dynamic_d_q128_d15_d31_d65_d127_d130_d255_d258_k10', 'labels': EXT_DYNAMIC_HIGHD_LABELS, 'guard': 'B == 1 and M == 65536 and K == 10 and not self_search and not forced_fallback and Q == 128 and D in {15,31,65,127,130,255,258} and arch in {sm_100a,sm_103a}', 'route': ROUTE_EXT_DYNAMIC_HIGHD_9286, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_EXT_HIGHD_SEED, 'source_task': 'weave-evolve ccef dynamic-D high bucket direct-stride repair', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'generated_variant_ext_dynamic_d_highd_q128_q64_k10', 'coverage_only': False}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_EXT_DYNAMIC_HIGHD_ENTRY, *base.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_ext_dynamic_highd(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    bsz, q_rows, m_rows, dim, k, self_search = _shape_key(inputs)
    if bsz != 1 or m_rows != 65536 or k != 10 or self_search:
        return False
    if q_rows == 128 and dim in EXT_DYNAMIC_HIGHD_Q128_DIMS:
        return highd.mma._tcgen05_capable_arch()
    return False

def _guard_order() -> list[str]:
    return [str(_EXT_DYNAMIC_HIGHD_ENTRY['shape_key']), *base._guard_order(base.PROFILE_ALL)]

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    if profile not in {PROFILE_ALL, PROFILE_BASE_C492}:
        raise ValueError(f'unknown 9286 extended dynamic-D profile: {profile}')
    if profile == PROFILE_ALL and _use_ext_dynamic_highd(inputs):
        return ROUTE_EXT_DYNAMIC_HIGHD_9286
    return base.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base.route_info(inputs))
    route = str(info.get('route') or info.get('selected_route') or base.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order() if profile == PROFILE_ALL else base._guard_order(base.PROFILE_ALL)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    info['forced_fallback'] = _forced_fallback(inputs)
    return info

def _ext_dynamic_highd_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base.selected_route(inputs))
    return {'profile': profile, 'route': _EXT_DYNAMIC_HIGHD_ENTRY['route'], 'selected_route': _EXT_DYNAMIC_HIGHD_ENTRY['route'], 'selected_entrypoint': _EXT_DYNAMIC_HIGHD_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'generated-variant', 'coverage_class': _EXT_DYNAMIC_HIGHD_ENTRY['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _EXT_DYNAMIC_HIGHD_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _EXT_DYNAMIC_HIGHD_ENTRY['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': _EXT_DYNAMIC_HIGHD_ENTRY['source_task'], 'source_round_doc': _EXT_DYNAMIC_HIGHD_ENTRY['source_round_doc'], 'selected_seed': _EXT_DYNAMIC_HIGHD_ENTRY['selected_seed'], 'selected_seed_task': _EXT_DYNAMIC_HIGHD_ENTRY['source_task'], 'replaced_seed': parent_info.get('selected_seed')}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile == PROFILE_ALL and _use_ext_dynamic_highd(inputs):
        return _ext_dynamic_highd_info(inputs, profile)
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {PROFILE_ALL, PROFILE_BASE_C492}:
        raise ValueError(f'unknown 9286 extended dynamic-D profile: {profile}')
    if profile == PROFILE_ALL and _use_ext_dynamic_highd(inputs):
        return highd._launch_high_dynamic_d_tcgen05(inputs)
    return base.launch_for_eval(inputs)

def launch_base_c492_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_C492)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return EXT_DYNAMIC_HIGHD_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0618_c492_ext_dynamic_highd_9286(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=EXT_DYNAMIC_HIGHD_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=EXT_DYNAMIC_HIGHD_LABELS) -> dict[str, Any]:
    return knn_search_compile_and_launch_dispatch0618_c492_ext_dynamic_highd_9286(benchmark=benchmark, shape_labels=shape_labels)
