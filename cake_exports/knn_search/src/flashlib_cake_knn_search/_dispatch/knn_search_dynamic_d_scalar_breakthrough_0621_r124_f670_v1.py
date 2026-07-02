"""Dynamic-D scalar-capacity breakthrough seed aggregator, expanded targets.

Minimum target architecture: sm_80 for the CUDA-core D3 routes and sm_100a for
the inherited tcgen05/TMEM routes. This additive bucket-kernel module does not
edit the production dispatcher. It exposes Weave-only exact-shape guards for
the 0621 dynamic-D scalar-capacity breakthrough lane, including additional
D65/D127/D129/D255/D257 target rows backed by source-clean tcgen05 seeds.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1 as ext_highd
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd_direct
from . import knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1 as d512_q64
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as remaining
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd_no_pack
from . import knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1 as d3_q128
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = ext_highd.THREADS
MERGE_THREADS = ext_highd.MERGE_THREADS
BLOCK_Q = ext_highd.BLOCK_Q
BLOCK_M = ext_highd.BLOCK_M
K_MAX = ext_highd.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:ir"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:partial_ir"}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:merge_ir"}'))
d3_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:d3_partial_ir"}'))
d3_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:d3_merge_ir"}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:tinyd_partial_ir"}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:tinyd_merge_ir"}'))
d512_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:d512_partial_ir"}'))
d512_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:d512_merge_ir"}'))
highd_direct_partial_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:highd_direct_partial_ir"}'))
highd_direct_merge_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:highd_direct_merge_ir"}'))
self_d3_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:self_d3_ir"}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:scalar_capacity_ir"}'))
ROUTE_DYNAMIC_D_BREAKTHROUGH_D3_Q128 = d3_q128.ROUTE_TINYD_D3_TILE_REDUCE
ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128 = 'f670_dynamic_d63_q128_no_pack_tcgen05'
ROUTE_DYNAMIC_D_BREAKTHROUGH_EXT_HIGHD_Q128 = ext_highd.ROUTE_EXT_DYNAMIC_HIGHD_9286
ROUTE_DYNAMIC_D_BREAKTHROUGH_HIGH_DIRECT_Q128 = highd_direct.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_DYNAMIC_D_BREAKTHROUGH_D31_Q128 = ROUTE_DYNAMIC_D_BREAKTHROUGH_EXT_HIGHD_Q128
ROUTE_DYNAMIC_D_BREAKTHROUGH_SELF_D3 = remaining.parent.ROUTE_SELF_Q2048_D3
ROUTE_DYNAMIC_D_BREAKTHROUGH_D512_Q64 = d512_q64.ROUTE_D512_Q64_TCGEN05
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_D3_Q128_SEED = 'weave-evolve-knn-search-b5b2'
CONSUMED_D63_Q128_SEED = 'weave-evolve-knn-search-449d'
CONSUMED_EXT_HIGHD_Q128_SEED = ext_highd.CONSUMED_EXT_HIGHD_SEED
CONSUMED_HIGH_DIRECT_Q128_SEED = highd_direct.CONSUMED_SEED
CONSUMED_D31_Q128_SEED = CONSUMED_EXT_HIGHD_Q128_SEED
CONSUMED_SELF_D3_SEED = 'weave-evolve-knn-search-a2ab'
CONSUMED_D512_Q64_SEED = d512_q64.CONSUMED_SEED
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_D3_Q128_SEED, CONSUMED_D63_Q128_SEED, CONSUMED_EXT_HIGHD_Q128_SEED, CONSUMED_HIGH_DIRECT_Q128_SEED, CONSUMED_SELF_D3_SEED, CONSUMED_D512_Q64_SEED)
EXT_HIGHD_TARGET_DIMS: frozenset[int] = frozenset({31, 65, 127, 255})
HIGH_DIRECT_TARGET_DIMS: frozenset[int] = frozenset({129, 257})
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10', 'blind_dyn_d3_q128_m65536_k10', 'blind_ext_dyn_d31_q128_m65536_k10', 'blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_ext_dyn_d512_q64_m65536_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"label": "blind_dyn_self_q2048_m2048_d3_k10", "params": {"B": 1, "D": 3, "K": 10, "M": 2048, "Q": 2048, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610810, "self_search": true}}, {"label": "blind_dyn_d3_q128_m65536_k10", "params": {"B": 1, "D": 3, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610801, "self_search": false}}, {"label": "blind_ext_dyn_d31_q128_m65536_k10", "params": {"B": 1, "D": 31, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610916, "self_search": false}}, {"label": "blind_ext_dyn_d65_q128_m65536_k10", "params": {"B": 1, "D": 65, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610917, "self_search": false}}, {"label": "blind_ext_dyn_d127_q128_m65536_k10", "params": {"B": 1, "D": 127, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610918, "self_search": false}}, {"label": "blind_dyn_d129_q128_m65536_k10", "params": {"B": 1, "D": 129, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610804, "self_search": false}}, {"label": "blind_dyn_d63_q128_m65536_k10", "params": {"B": 1, "D": 63, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610803, "self_search": false}}, {"label": "blind_ext_dyn_d255_q128_m65536_k10", "params": {"B": 1, "D": 255, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610920, "self_search": false}}, {"label": "blind_dyn_d257_q128_m65536_k10", "params": {"B": 1, "D": 257, "K": 10, "M": 65536, "Q": 128, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610805, "self_search": false}}, {"label": "blind_ext_dyn_d512_q64_m65536_k10", "params": {"B": 1, "D": 512, "K": 10, "M": 65536, "Q": 64, "dtype": "bfloat16", "min_recall": 0.999, "seed": 610922, "self_search": false}}]'))
REQUESTED_BUT_MISSING_CONTRACT_LABELS: tuple[str, ...] = ('blind_ext_dyn_d63_q128_m65536_k10', 'blind_ext_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10')
REQUESTED_LABEL_ALIASES: dict[str, str] = {'blind_dyn_d63_q128_m65536_k10': 'blind_ext_dyn_d63_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10': 'blind_ext_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10': 'blind_ext_dyn_d257_q128_m65536_k10'}
_D3_Q128_ENTRY: dict[str, Any] = {'shape_key': 'f670_dynamic_d3_q128_m65536_k10', 'label': 'blind_dyn_d3_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_D3_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1:launch_for_eval', 'selected_seed': CONSUMED_D3_Q128_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md', 'coverage_class': 'bucket_seed_dynamic_d3_q128_m65536_k10', 'arch_requirement': 'sm_80'}
_D63_Q128_ENTRY: dict[str, Any] = {'shape_key': 'f670_dynamic_d63_q128_m65536_k10', 'label': 'blind_dyn_d63_q128_m65536_k10', 'requested_label_alias': 'blind_ext_dyn_d63_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_D63_Q128_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md', 'coverage_class': 'bucket_seed_dynamic_d63_q128_m65536_k10', 'arch_requirement': 'sm_100a'}

def _q128_entry(*, dim: int, label: str, shape_prefix: str, route: str, source_entrypoint: str, selected_seed: str, source_round_doc: str, coverage_prefix: str) -> dict[str, Any]:
    entry: dict[str, Any] = {'shape_key': f'f670_{shape_prefix}_d{dim}_q128_m65536_k10', 'label': label, 'guard': f'B == 1 and Q == 128 and M == 65536 and D == {dim} and K == 10 and not self_search and not forced_fallback and arch in {{sm_100a,sm_103a}}', 'route': route, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'source_entrypoint': source_entrypoint, 'selected_seed': selected_seed, 'source_round_doc': source_round_doc, 'coverage_class': f'{coverage_prefix}_d{dim}_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
    alias = REQUESTED_LABEL_ALIASES.get(label)
    if alias is not None:
        entry['requested_label_alias'] = alias
    return entry
_EXT_HIGHD_ENTRIES = _decode_capture(_json_loads('{"127": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d127_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 127 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d127_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d127_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, "255": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d255_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 255 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d255_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d255_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, "31": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d31_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 31 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d31_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d31_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, "65": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d65_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 65 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d65_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d65_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}}'))
_HIGH_DIRECT_ENTRIES = _decode_capture(_json_loads('{"129": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_dynamic_d129_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_dyn_d129_q128_m65536_k10", "requested_label_alias": "blind_ext_dyn_d129_q128_m65536_k10", "route": "ccef_dynamic_d_high_q128_directstride_tcgen05", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_dynamic_d129_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md"}, "257": {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_dynamic_d257_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 257 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_dyn_d257_q128_m65536_k10", "requested_label_alias": "blind_ext_dyn_d257_q128_m65536_k10", "route": "ccef_dynamic_d_high_q128_directstride_tcgen05", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_dynamic_d257_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md"}}'))
_D31_Q128_ENTRY: dict[str, Any] = _EXT_HIGHD_ENTRIES[31]
_SELF_D3_ENTRY: dict[str, Any] = {'shape_key': 'f670_dynamic_self_q2048_m2048_d3_k10', 'label': 'blind_dyn_self_q2048_m2048_d3_k10', 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_SELF_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_SELF_D3_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md', 'coverage_class': 'bucket_seed_dynamic_d_self_q2048_m2048_d3_k10', 'arch_requirement': 'sm_80'}
_D512_Q64_ENTRY: dict[str, Any] = {'shape_key': 'f670_ext_dynamic_d512_q64_m65536_k10', 'label': 'blind_ext_dyn_d512_q64_m65536_k10', 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_D512_Q64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D512_Q64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_112_9286.md', 'coverage_class': 'bucket_seed_ext_dynamic_d512_q64_m65536_k10', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"arch_requirement": "sm_80", "coverage_class": "bucket_seed_dynamic_d_self_q2048_m2048_d3_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback", "label": "blind_dyn_self_q2048_m2048_d3_k10", "route": "ccef_dynamic_d_self_q2048_d3_tile_reduce", "selected_seed": "weave-evolve-knn-search-a2ab", "shape_key": "f670_dynamic_self_q2048_m2048_d3_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"}, {"arch_requirement": "sm_80", "coverage_class": "bucket_seed_dynamic_d3_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search and not forced_fallback", "label": "blind_dyn_d3_q128_m65536_k10", "route": "b5b2_dynamic_d3_tinyd_tile_reduce", "selected_seed": "weave-evolve-knn-search-b5b2", "shape_key": "f670_dynamic_d3_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d31_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 31 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d31_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d31_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d65_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 65 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d65_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d65_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d127_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 127 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d127_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d127_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d255_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 255 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d255_q128_m65536_k10", "route": "9286_ext_dynamic_d_highd_generated_variants", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_ext_dynamic_d255_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_dynamic_d129_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_dyn_d129_q128_m65536_k10", "requested_label_alias": "blind_ext_dyn_d129_q128_m65536_k10", "route": "ccef_dynamic_d_high_q128_directstride_tcgen05", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_dynamic_d129_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_dynamic_d257_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 257 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_dyn_d257_q128_m65536_k10", "requested_label_alias": "blind_ext_dyn_d257_q128_m65536_k10", "route": "ccef_dynamic_d_high_q128_directstride_tcgen05", "selected_seed": "weave-evolve-knn-search-ccef-highd-directstride", "shape_key": "f670_dynamic_d257_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_dynamic_d63_q128_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_dyn_d63_q128_m65536_k10", "requested_label_alias": "blind_ext_dyn_d63_q128_m65536_k10", "route": "f670_dynamic_d63_q128_no_pack_tcgen05", "selected_seed": "weave-evolve-knn-search-449d", "shape_key": "f670_dynamic_d63_q128_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval", "source_round_doc": "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"}, {"arch_requirement": "sm_100a", "coverage_class": "bucket_seed_ext_dynamic_d512_q64_m65536_k10", "entrypoint": "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval", "guard": "B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}", "label": "blind_ext_dyn_d512_q64_m65536_k10", "route": "9286_d512_q64_row16_directstride_tcgen05", "selected_seed": "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "shape_key": "f670_ext_dynamic_d512_q64_m65536_k10", "source_entrypoint": "loom.examples.weave.knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1:launch_for_eval", "source_round_doc": "design_doc/active/weave_evolve_knn_search_round_112_9286.md"}]}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_d3_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 3, 10, False) and (not _forced_fallback(inputs))

def _use_d63_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 63, 10, False) and (not _forced_fallback(inputs)) and tinyd_no_pack.mma._tcgen05_capable_arch()

def _use_ext_highd_q128(inputs: dict[str, Any]) -> bool:
    return ext_highd._use_ext_dynamic_highd(inputs) and int(inputs['D']) in EXT_HIGHD_TARGET_DIMS

def _use_high_direct_q128(inputs: dict[str, Any]) -> bool:
    return highd_direct._use_high_dynamic_d_tcgen05(inputs) and int(inputs['D']) in HIGH_DIRECT_TARGET_DIMS

def _use_self_d3(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 2048, 2048, 3, 10, True) and (not _forced_fallback(inputs))

def _use_d512_q64(inputs: dict[str, Any]) -> bool:
    return d512_q64._use_d512_q64_tcgen05(inputs)

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _use_self_d3(inputs):
        return _SELF_D3_ENTRY
    if _use_d3_q128(inputs):
        return _D3_Q128_ENTRY
    if _use_ext_highd_q128(inputs):
        return _EXT_HIGHD_ENTRIES[int(inputs['D'])]
    if _use_high_direct_q128(inputs):
        return _HIGH_DIRECT_ENTRIES[int(inputs['D'])]
    if _use_d63_q128(inputs):
        return _D63_Q128_ENTRY
    if _use_d512_q64(inputs):
        return _D512_Q64_ENTRY
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is None:
        return ROUTE_SCALAR_CAPACITY
    return str(entry['route'])

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _fallback_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'classification': 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return _fallback_info(inputs)
    info = {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}
    if 'requested_label_alias' in entry:
        info['requested_label_alias'] = entry['requested_label_alias']
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is _SELF_D3_ENTRY:
        return remaining.launch_for_eval(inputs)
    if entry is _D3_Q128_ENTRY:
        return d3_q128.launch_for_eval(inputs)
    if entry in _EXT_HIGHD_ENTRIES.values():
        return ext_highd.launch_for_eval(inputs)
    if entry in _HIGH_DIRECT_ENTRIES.values():
        return highd_direct.launch_for_eval(inputs)
    if entry is _D63_Q128_ENTRY:
        return tinyd_no_pack.launch_for_eval(inputs)
    if entry is _D512_Q64_ENTRY:
        return d512_q64.launch_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_scalar_breakthrough_0621_r124_f670(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
