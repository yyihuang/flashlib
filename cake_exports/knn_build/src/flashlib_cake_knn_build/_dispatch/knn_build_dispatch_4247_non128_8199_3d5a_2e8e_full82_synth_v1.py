"""Full82 q16split148/cachedmerge/K96 synthesis dispatcher over 8199 widecombine.

Minimum target architecture: sm_100a. This opt-in dispatcher candidate starts
from the 7e5d full82 matrix harness and replays sidecar seeds without changing
their schedules: 3f2d q16-only K32 split148, af67 non-D128 cached-merge, and
b6c4 K96 exact-all. Guard misses stay on the 8199-widecombine full82 Weave
dispatcher; no external implementation is on the production route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
import argparse
import json
from importlib import import_module
from pathlib import Path
from statistics import median
from typing import Any, Callable
from . import knn_build_dispatch_4247_non128_8199_c2eb_f533_8fcb_8227_full82_matrix_v1 as matrix_7e5d
from . import knn_build_dispatch_4247_non128_8199_widecombine_full82_v1 as baseline_wide
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_4247
from . import knn_build_non128_frontier_3d5a_cachedmerge_v1 as seed_cachedmerge
from . import knn_build_over64_k96_exactall_229a_v1 as seed_k96_exactall
from . import knn_build_rag_microbucket_k32_2e8e_q16split148_v1 as seed_q16split148
from . import knn_build_rag_microbucket_k32_8fcb_split148_v1 as seed_split148
from . import knn_build_ragonline_mbucket_4fc7_q1m262_v1 as seed_q1m262
eval_mod = matrix_7e5d.eval_mod
MODULE = 'loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1'
BASELINE_ID = 'baseline_8199_widecombine_full82_v1'
BASELINE_ENTRYPOINT = matrix_7e5d.BASELINE_ENTRYPOINT
SEED_SPLIT148_7E5D_ID = seed_split148.SEED_K32_8FCB_SPLIT148_ID
SEED_Q16SPLIT148_3F2D_ID = seed_q16split148.SEED_K32_2E8E_Q16_SPLIT148_ID
SEED_CACHEDMERGE_AF67_ID = 'non128_frontier_3d5a_cachedmerge_v1'
SEED_K96_EXACTALL_B6C4_ID = 'over64_k96_exactall_229a_v1_b6c4'
SEED_MIDK_E080_ID = 'knn_build_midk_k11k13_e080_v1'
SEED_Q1M262_980C_ID = 'ragonline_mbucket_4fc7_q1m262_v1_980c'
ROUTE_SPLIT148_7E5D_ENTRYPOINT = f'{seed_split148.MODULE}:launch_from_contract_inputs'
ROUTE_Q16SPLIT148_3F2D_ENTRYPOINT = f'{seed_q16split148.MODULE}:launch_from_contract_inputs'
ROUTE_CACHEDMERGE_AF67_ENTRYPOINT = f'{seed_cachedmerge.MODULE}:launch_from_contract_inputs'
ROUTE_K96_EXACTALL_B6C4_ENTRYPOINT = f'{seed_k96_exactall.__name__}:launch_from_contract_inputs'
ROUTE_MIDK_E080_ENTRYPOINT = 'loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs'
ROUTE_Q1M262_980C_ENTRYPOINT = f'{seed_q1m262.MODULE}:launch_from_contract_inputs'
ROUTE_BASELINE_ENTRYPOINT = f'{baseline_wide.MODULE}:launch_from_contract_inputs'
CACHEDMERGE_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10"]}'))
K32_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}'))
K96_EXACTALL_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96"]}'))
MIDK_E080_TARGET_SHAPES = ('build_k_sweep_qm2048_k11', 'build_k_sweep_qm2048_k12', 'build_k_sweep_qm2048_k13', 'build_k_sweep_qm4096_k13')
Q1M262_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10"]}'))
Q16_CACHEDMERGE_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}'))
ABEE_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm4096_k13", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10"]}'))
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm4096_k13", "rag_online_irregular_b1_q1_m262143_d128_k10"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm4096_k13", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm4096_k13", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
SOURCE_TASKS = {SEED_SPLIT148_7E5D_ID: 'generalize-auto-tuning-knn-build-7e5d / design_doc/active/generalize_auto_tuning_knn_build_round_106_7e5d.md', SEED_Q16SPLIT148_3F2D_ID: 'weave-evolve-knn-build-3f2d / design_doc/active/weave_evolve_knn_build_round_106_2e8e_q16split148.md', SEED_CACHEDMERGE_AF67_ID: 'weave-evolve-knn-build-af67 / design_doc/active/weave_evolve_knn_build_round_103_3d5a_cachedmerge.md', SEED_K96_EXACTALL_B6C4_ID: 'weave-evolve-knn-build-229a / design_doc/active/weave_evolve_knn_build_round_45_229a_k96exactall.md; same-session K96 audit in design_doc/active/generalize_auto_tuning_knn_build_round_107_adbd.md', SEED_MIDK_E080_ID: 'weave-evolve-knn-build-e080 / design_doc/active/weave_evolve_knn_build_round_108_e080_midk_k11k13.md', SEED_Q1M262_980C_ID: 'weave-evolve-knn-build-980c / design_doc/active/weave_evolve_knn_build_round_108_4fc7_q1m262.md'}
_CONTRACT_PARAMS_BY_LABEL = _decode_capture(_json_loads('{"build_batch_b2_q1024_m1024_d128_k10": {"B": 2, "D": 128, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "batch_generalization", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606211, "time_flashlib": true}, "build_common_d1024_b1_q512_m512_k10": {"B": 1, "D": 1024, "K": 10, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "common_embedding_dim_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 613024, "time_flashlib": true}, "build_common_d256_b1_q1024_m1024_k10": {"B": 1, "D": 256, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "common_embedding_dim_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612256, "time_flashlib": true}, "build_common_d4096_b1_q512_m512_k10": {"B": 1, "D": 4096, "K": 10, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "common_embedding_dim_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 614096, "time_flashlib": true}, "build_common_d768_b1_q1024_m1024_k10": {"B": 1, "D": 768, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "common_embedding_dim_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612768, "time_flashlib": true}, "build_dim_sweep_b1_q1024_m1024_d64_k10": {"B": 1, "D": 64, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "d64_dispatch_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608164, "time_flashlib": true}, "build_dim_sweep_b1_q1024_m1024_d96_k10": {"B": 1, "D": 96, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "d_generalization_non128", "dtype": "bfloat16", "recall_min": 0.999, "seed": 610096, "time_flashlib": true}, "build_dim_sweep_b1_q2048_m2048_d192_k10": {"B": 1, "D": 192, "K": 10, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "d_generalization_non128", "dtype": "bfloat16", "recall_min": 0.999, "seed": 610192, "time_flashlib": true}, "build_dim_sweep_b1_q2048_m2048_d256_k10": {"B": 1, "D": 256, "K": 10, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "d_generalization", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606256, "time_flashlib": true}, "build_dim_sweep_b1_q2048_m2048_d64_k10": {"B": 1, "D": 64, "K": 10, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "d_generalization", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606264, "time_flashlib": true}, "build_dim_sweep_b1_q4096_m4096_d64_k10": {"B": 1, "D": 64, "K": 10, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "d64_dispatch_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608464, "time_flashlib": true}, "build_dtype_fp16_b1_q2048_m2048_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "dtype_generalization", "dtype": "float16", "recall_min": 0.999, "seed": 606216, "time_flashlib": true}, "build_highd_b1_q1024_m1024_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "high_d_split_generalization", "dtype": "bfloat16", "recall_min": 0.999, "seed": 610320, "time_flashlib": true}, "build_k_sweep_qm1024_k12": {"B": 1, "D": 128, "K": 12, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606112, "time_flashlib": true}, "build_k_sweep_qm1024_k16": {"B": 1, "D": 128, "K": 16, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "mid_k_topk_bucket", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606116, "time_flashlib": true}, "build_k_sweep_qm1024_k20": {"B": 1, "D": 128, "K": 20, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606120, "time_flashlib": true}, "build_k_sweep_qm2048_k11": {"B": 1, "D": 128, "K": 11, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "mid_k_boundary_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 609211, "time_flashlib": true}, "build_k_sweep_qm2048_k12": {"B": 1, "D": 128, "K": 12, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "mid_k_boundary_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606212, "time_flashlib": true}, "build_k_sweep_qm2048_k13": {"B": 1, "D": 128, "K": 13, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "mid_k_boundary_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 609213, "time_flashlib": true}, "build_k_sweep_qm2048_k20": {"B": 1, "D": 128, "K": 20, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606220, "time_flashlib": true}, "build_k_sweep_qm2048_k24": {"B": 1, "D": 128, "K": 24, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "large_k_topk_ramp", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606224, "time_flashlib": true}, "build_k_sweep_qm2048_k28": {"B": 1, "D": 128, "K": 28, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "large_k_topk_ramp", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606228, "time_flashlib": true}, "build_k_sweep_qm4096_k12": {"B": 1, "D": 128, "K": 12, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "mid_k_boundary_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606412, "time_flashlib": true}, "build_k_sweep_qm4096_k13": {"B": 1, "D": 128, "K": 13, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "mid_k_boundary_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 609413, "time_flashlib": true}, "build_k_sweep_qm4096_k20": {"B": 1, "D": 128, "K": 20, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606420, "time_flashlib": true}, "build_k_sweep_qm4096_k24": {"B": 1, "D": 128, "K": 24, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "large_k_topk_ramp", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606424, "time_flashlib": true}, "build_k_sweep_qm4096_k28": {"B": 1, "D": 128, "K": 28, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "large_k_topk_ramp", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606428, "time_flashlib": true}, "build_k_sweep_qm4096_k30": {"B": 1, "D": 128, "K": 30, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "large_k_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606430, "time_flashlib": true}, "build_k_sweep_qm512_k1": {"B": 1, "D": 128, "K": 1, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606049, "time_flashlib": true}, "build_k_sweep_qm512_k10": {"B": 1, "D": 128, "K": 10, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606058, "time_flashlib": true}, "build_k_sweep_qm512_k2": {"B": 1, "D": 128, "K": 2, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606050, "time_flashlib": true}, "build_k_sweep_qm512_k4": {"B": 1, "D": 128, "K": 4, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "low_k_q512_k5_neighborhood", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606052, "time_flashlib": true}, "build_k_sweep_qm512_k5": {"B": 1, "D": 128, "K": 5, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606053, "time_flashlib": true}, "build_k_sweep_qm512_k6": {"B": 1, "D": 128, "K": 6, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "low_k_q512_k5_neighborhood", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606054, "time_flashlib": true}, "build_k_sweep_qm512_k8": {"B": 1, "D": 128, "K": 8, "M": 512, "Q": 512, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606056, "time_flashlib": true}, "build_large_b1_q6144_m6144_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 6144, "Q": 6144, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "large_build_k10_interpolation", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611614, "time_flashlib": true}, "build_large_b1_q8192_m8192_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 8192, "Q": 8192, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 128, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606819, "time_flashlib": true}, "build_large_b1_q8192_m8192_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 8192, "Q": 8192, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "large_build_mid_k", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606820, "time_flashlib": true}, "build_large_b1_q8192_m8192_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 8192, "Q": 8192, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "large_build_large_k", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606832, "time_flashlib": true}, "build_large_tail_b1_q6144_m6144_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 6144, "Q": 6144, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "large_build_mid_k_interpolation", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607144, "time_flashlib": true}, "build_largek_stress_qm4096_k32": {"B": 1, "D": 128, "K": 32, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "large_k_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606432, "time_flashlib": true}, "build_medium_b1_q4096_m4096_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606002, "time_flashlib": true}, "build_over32_stress_qm2048_k48": {"B": 1, "D": 128, "K": 48, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "over32_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606248, "time_flashlib": true}, "build_over32_stress_qm2048_k64": {"B": 1, "D": 128, "K": 64, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "over32_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606249, "time_flashlib": true}, "build_over32_stress_qm4096_k48": {"B": 1, "D": 128, "K": 48, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "over32_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606448, "time_flashlib": true}, "build_over32_stress_qm4096_k64": {"B": 1, "D": 128, "K": 64, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "over32_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607464, "time_flashlib": true}, "build_over64_stress_qm1024_k96": {"B": 1, "D": 128, "K": 96, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "over64_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 609196, "time_flashlib": true}, "build_over64_stress_qm2048_k96": {"B": 1, "D": 128, "K": 96, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "over64_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607296, "time_flashlib": true}, "build_over64_stress_qm4096_k96": {"B": 1, "D": 128, "K": 96, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "over64_topk_bottleneck", "dtype": "bfloat16", "recall_min": 0.999, "seed": 609496, "time_flashlib": true}, "build_qm1024_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606104, "time_flashlib": true}, "build_qm1024_d128_k8": {"B": 1, "D": 128, "K": 8, "M": 1024, "Q": 1024, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "low_k_build_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611108, "time_flashlib": true}, "build_qm2048_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606210, "time_flashlib": true}, "build_qm2048_d128_k8": {"B": 1, "D": 128, "K": 8, "M": 2048, "Q": 2048, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606208, "time_flashlib": true}, "build_qm4096_d128_k8": {"B": 1, "D": 128, "K": 8, "M": 4096, "Q": 4096, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "low_k_build_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611408, "time_flashlib": true}, "build_tail_b1_q1536_m1536_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 1536, "Q": 1536, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "non_power_of_two_build", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606153, "time_flashlib": true}, "build_tail_b1_q3072_m3072_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 3072, "Q": 3072, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "non_power_of_two_mid_k_build", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606327, "time_flashlib": true}, "build_verylarge_b1_q12288_m12288_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 12288, "Q": 12288, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "very_large_build", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606123, "time_flashlib": true}, "flashml_correctness_b1_q256_m256_d128_k5": {"B": 1, "D": 128, "K": 5, "M": 256, "Q": 256, "benchmark": true, "build": true, "check_correctness": true, "correctness_query_sample": 256, "dtype": "bfloat16", "recall_min": 0.99, "seed": 606001, "time_flashlib": true}, "rag_batch_b2_q256_m50000_d128_k10": {"B": 2, "D": 128, "K": 10, "M": 50000, "Q": 256, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "rag_batch_nonbuild", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607256, "time_flashlib": true}, "rag_irregular_b1_q512_m131071_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 131071, "Q": 512, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "rag_irregular_m_tail", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607512, "time_flashlib": true}, "rag_microbatch_b1_q16_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "rag_microbatch_guard_gap", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607016, "time_flashlib": true}, "rag_microbatch_b1_q32_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 32, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 32, "diagnostic_class": "rag_microbatch_guard_gap", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607032, "time_flashlib": true}, "rag_microbatch_b1_q4_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 4, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 4, "diagnostic_class": "rag_microbatch_q_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608004, "time_flashlib": true}, "rag_microbatch_b1_q64_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 64, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 64, "diagnostic_class": "rag_microbatch_q_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608064, "time_flashlib": true}, "rag_microbatch_b1_q8_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 8, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 8, "diagnostic_class": "rag_microbatch_guard_gap", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607008, "time_flashlib": true}, "rag_microbatch_common_d1024_b1_q4_m100000_k10": {"B": 1, "D": 1024, "K": 10, "M": 100000, "Q": 4, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 4, "diagnostic_class": "v12_common_d_rag_microbatch_tail", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616024, "time_flashlib": true}, "rag_microbatch_common_d1024_b1_q8_m50000_k10": {"B": 1, "D": 1024, "K": 10, "M": 50000, "Q": 8, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 8, "diagnostic_class": "common_embedding_dim_rag_microbatch", "dtype": "bfloat16", "recall_min": 0.999, "seed": 613034, "time_flashlib": true}, "rag_microbatch_common_d256_b1_q16_m50000_k10": {"B": 1, "D": 256, "K": 10, "M": 50000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "common_embedding_dim_rag_microbatch", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612266, "time_flashlib": true}, "rag_microbatch_common_d256_b1_q4_m100000_k10": {"B": 1, "D": 256, "K": 10, "M": 100000, "Q": 4, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 4, "diagnostic_class": "v12_common_d_rag_microbatch_tail", "dtype": "bfloat16", "recall_min": 0.999, "seed": 615256, "time_flashlib": true}, "rag_microbatch_common_d4096_b1_q4_m32768_k10": {"B": 1, "D": 4096, "K": 10, "M": 32768, "Q": 4, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 4, "diagnostic_class": "common_embedding_dim_rag_microbatch", "dtype": "bfloat16", "recall_min": 0.999, "seed": 614106, "time_flashlib": true}, "rag_microbatch_common_d64_b1_q16_m50000_k10": {"B": 1, "D": 64, "K": 10, "M": 50000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "common_embedding_dim_rag_microbatch", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612064, "time_flashlib": true}, "rag_microbatch_common_d64_b1_q4_m100000_k10": {"B": 1, "D": 64, "K": 10, "M": 100000, "Q": 4, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 4, "diagnostic_class": "v12_common_d_rag_microbatch_tail", "dtype": "bfloat16", "recall_min": 0.999, "seed": 615164, "time_flashlib": true}, "rag_microbatch_common_d768_b1_q8_m100000_k10": {"B": 1, "D": 768, "K": 10, "M": 100000, "Q": 8, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 8, "diagnostic_class": "v12_common_d_rag_microbatch_tail", "dtype": "bfloat16", "recall_min": 0.999, "seed": 615768, "time_flashlib": true}, "rag_microbatch_highd_b1_q16_m50000_d768_k10": {"B": 1, "D": 768, "K": 10, "M": 50000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "rag_embedding_dim_generalization", "dtype": "bfloat16", "recall_min": 0.999, "seed": 610768, "time_flashlib": true}, "rag_microbatch_largek_b1_q16_m100000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 100000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "rag_microbatch_large_k_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608316, "time_flashlib": true}, "rag_microbatch_largek_b1_q16_m131071_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 131071, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "rag_microbatch_large_k_irregular_m_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608317, "time_flashlib": true}, "rag_microbatch_largek_b1_q16_m250000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 250000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "rag_microbatch_large_k_large_m_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611325, "time_flashlib": true}, "rag_microbatch_largek_b1_q24_m100000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 100000, "Q": 24, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 24, "diagnostic_class": "rag_microbatch_large_k_q_interpolation", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611324, "time_flashlib": true}, "rag_microbatch_largek_b1_q32_m100000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 100000, "Q": 32, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 32, "diagnostic_class": "rag_microbatch_large_k_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608332, "time_flashlib": true}, "rag_microbatch_largek_b1_q8_m100000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 100000, "Q": 8, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 8, "diagnostic_class": "rag_microbatch_large_k_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608308, "time_flashlib": true}, "rag_microbatch_largek_common_d256_b1_q8_m100000_k32": {"B": 1, "D": 256, "K": 32, "M": 100000, "Q": 8, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 8, "diagnostic_class": "v12_common_d_large_k_rag", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616332, "time_flashlib": true}, "rag_microbatch_over32_d128_b1_q16_m100000_k48": {"B": 1, "D": 128, "K": 48, "M": 100000, "Q": 16, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 16, "diagnostic_class": "v12_rag_over32_topk", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616548, "time_flashlib": true}, "rag_offline_b1_q10000_m50000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 50000, "Q": 10000, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606904, "time_flashlib": true}, "rag_offline_b1_q4096_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 4096, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "dtype": "bfloat16", "recall_min": 0.999, "seed": 606902, "time_flashlib": true}, "rag_offline_batch_b1_q10000_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 10000, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "dtype": "bfloat16", "public_baseline_qps": 310000, "recall_min": 0.999, "seed": 606003, "time_flashlib": true}, "rag_offline_large_m_b1_q8192_m250000_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 250000, "Q": 8192, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 64, "diagnostic_class": "rag_large_m_mid_k", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606925, "time_flashlib": true}, "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64": {"B": 1, "D": 128, "K": 64, "M": 250000, "Q": 2048, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 64, "diagnostic_class": "rag_large_m_over32", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606964, "time_flashlib": true}, "rag_offline_largek_b1_q4096_m100000_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 100000, "Q": 4096, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 64, "diagnostic_class": "rag_large_k_topk", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606920, "time_flashlib": true}, "rag_online_b1_q1_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single", "dtype": "bfloat16", "public_baseline_qps": 2600, "recall_min": 0.999, "seed": 606901, "time_flashlib": true}, "rag_online_b1_q1_m65536_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 65536, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single_low_m_frontier", "dtype": "bfloat16", "public_baseline_qps": 2600, "recall_min": 0.999, "seed": 611901, "time_flashlib": true}, "rag_online_common_d4096_b1_q1_m65536_k10": {"B": 1, "D": 4096, "K": 10, "M": 65536, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "v12_common_d_rag_online_highd", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616096, "time_flashlib": true}, "rag_online_common_d64_b1_q1_m262143_k10": {"B": 1, "D": 64, "K": 10, "M": 262143, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "v12_common_d_rag_online_irregular", "dtype": "bfloat16", "recall_min": 0.999, "seed": 615064, "time_flashlib": true}, "rag_online_irregular_b1_q1_m131071_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 131071, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single_m_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608191, "time_flashlib": true}, "rag_online_irregular_b1_q1_m262143_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 262143, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single_m_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608195, "time_flashlib": true}, "rag_online_irregular_b1_q1_m524287_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 524287, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single_large_m_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611905, "time_flashlib": true}, "rag_online_large_m_b1_q1_m250000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 250000, "Q": 1, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 1, "diagnostic_class": "rag_online_single_m_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608194, "time_flashlib": true}, "rag_stream_b1_q128_m100000_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 100000, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "rag_streaming", "dtype": "bfloat16", "public_baseline_qps": 5000, "recall_min": 0.999, "seed": 606928, "time_flashlib": true}, "rag_stream_common_d256_b1_q128_m100000_k10": {"B": 1, "D": 256, "K": 10, "M": 100000, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "v12_common_d_rag_streaming", "dtype": "bfloat16", "recall_min": 0.999, "seed": 615356, "time_flashlib": true}, "rag_stream_largek_b1_q128_m100000_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 100000, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "rag_streaming_large_k", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607128, "time_flashlib": true}, "rag_stream_largek_b1_q128_m131071_d128_k32": {"B": 1, "D": 128, "K": 32, "M": 131071, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "rag_streaming_large_k_irregular_m_frontier", "dtype": "bfloat16", "recall_min": 0.999, "seed": 611328, "time_flashlib": true}, "rag_stream_largek_common_d256_b1_q128_m100000_k32": {"B": 1, "D": 256, "K": 32, "M": 100000, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "v12_common_d_large_k_rag", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616432, "time_flashlib": true}, "search_rect_b1_q1024_m32768_d64_k10": {"B": 1, "D": 64, "K": 10, "M": 32768, "Q": 1024, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "rectangular_search_d64_guard_blindspot", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608864, "time_flashlib": true}, "search_rect_b1_q1024_m8192_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 8192, "Q": 1024, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "rectangular_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606812, "time_flashlib": true}, "search_rect_b1_q1536_m65536_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 65536, "Q": 1536, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "rectangular_search_tail_guard", "dtype": "bfloat16", "recall_min": 0.999, "seed": 608655, "time_flashlib": true}, "search_rect_b1_q2048_m32768_d128_k10": {"B": 1, "D": 128, "K": 10, "M": 32768, "Q": 2048, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 512, "diagnostic_class": "rectangular_search_intermediate", "dtype": "bfloat16", "recall_min": 0.999, "seed": 607048, "time_flashlib": true}, "search_rect_b1_q4096_m65536_d128_k20": {"B": 1, "D": 128, "K": 20, "M": 65536, "Q": 4096, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "rectangular_search_mid_k", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606655, "time_flashlib": true}, "search_rect_common_d1024_b1_q256_m8192_k10": {"B": 1, "D": 1024, "K": 10, "M": 8192, "Q": 256, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "v12_common_d_rectangular_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616124, "time_flashlib": true}, "search_rect_common_d256_b1_q1024_m32768_k10": {"B": 1, "D": 256, "K": 10, "M": 32768, "Q": 1024, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "common_embedding_dim_rectangular_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612856, "time_flashlib": true}, "search_rect_common_d4096_b1_q128_m4096_k10": {"B": 1, "D": 4096, "K": 10, "M": 4096, "Q": 128, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 128, "diagnostic_class": "v12_common_d_rectangular_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 616496, "time_flashlib": true}, "search_rect_common_d768_b1_q512_m8192_k10": {"B": 1, "D": 768, "K": 10, "M": 8192, "Q": 512, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "common_embedding_dim_rectangular_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 612876, "time_flashlib": true}, "search_rect_highd_b1_q512_m12000_d320_k10": {"B": 1, "D": 320, "K": 10, "M": 12000, "Q": 512, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 256, "diagnostic_class": "flashlib_high_d_search", "dtype": "bfloat16", "recall_min": 0.999, "seed": 610321, "time_flashlib": true}, "search_rect_over32_b1_q2048_m65536_d128_k64": {"B": 1, "D": 128, "K": 64, "M": 65536, "Q": 2048, "benchmark": true, "build": false, "check_correctness": true, "correctness_query_sample": 64, "diagnostic_class": "rectangular_search_over32", "dtype": "bfloat16", "recall_min": 0.999, "seed": 606664, "time_flashlib": true}}'))
_K32_SHAPE_SPECS = {label: _CONTRACT_PARAMS_BY_LABEL[label] for label in K32_TARGET_SHAPES}
_K96_EXACTALL_SHAPE_SPECS = {label: _CONTRACT_PARAMS_BY_LABEL[label] for label in K96_EXACTALL_TARGET_SHAPES}
_MIDK_E080_SHAPE_SPECS = {label: _CONTRACT_PARAMS_BY_LABEL[label] for label in MIDK_E080_TARGET_SHAPES}
_Q1M262_SHAPE_SPECS = {label: _CONTRACT_PARAMS_BY_LABEL[label] for label in Q1M262_TARGET_SHAPES}
TARGETED_SEED_ROWS_BY_SEED = _decode_capture(_json_loads('{"knn_build_midk_k11k13_e080_v1": {"build_k_sweep_qm2048_k11": {"flashlib_ms": 0.074752, "kernel_ms": 0.052992, "ratio_vs_flashlib": 1.4106280193236715, "tflops": 20.260823520531403}, "build_k_sweep_qm2048_k12": {"flashlib_ms": 0.07648, "kernel_ms": 0.054848, "ratio_vs_flashlib": 1.3943990665110853, "tflops": 19.574983591014274}, "build_k_sweep_qm2048_k13": {"flashlib_ms": 0.081088, "kernel_ms": 0.059103, "ratio_vs_flashlib": 1.371977733786779, "tflops": 18.165433779570755}, "build_k_sweep_qm4096_k13": {"flashlib_ms": 0.1635205, "kernel_ms": 0.143616, "ratio_vs_flashlib": 1.1385952818627452, "tflops": 29.911229767713895}}, "non128_frontier_3d5a_cachedmerge_v1": {"build_dim_sweep_b1_q1024_m1024_d96_k10": {"flashlib_ms": 0.06852849999999999, "kernel_ms": 0.032609, "ratio_vs_flashlib": 2.1015210524701766, "tflops": 6.173957864393266}, "build_dim_sweep_b1_q2048_m2048_d192_k10": {"flashlib_ms": 0.11584, "kernel_ms": 0.07648, "ratio_vs_flashlib": 1.514644351464435, "tflops": 21.059266945606694}, "build_highd_b1_q1024_m1024_d320_k10": {"flashlib_ms": 0.074528, "kernel_ms": 0.033536, "ratio_vs_flashlib": 2.222328244274809, "tflops": 20.01099236641221}, "rag_microbatch_highd_b1_q16_m50000_d768_k10": {"flashlib_ms": 0.166849, "kernel_ms": 0.094592, "ratio_vs_flashlib": 1.7638806664411368, "tflops": 12.990527740189446}, "search_rect_highd_b1_q512_m12000_d320_k10": {"flashlib_ms": 0.150881, "kernel_ms": 0.115488, "ratio_vs_flashlib": 1.3064647409254642, "tflops": 34.04821280133001}}, "over64_k96_exactall_229a_v1_b6c4": {"build_over64_stress_qm1024_k96": {"flashlib_ms": 0.27504, "kernel_ms": 0.2179045, "ratio_vs_flashlib": 1.2622043142752903, "tflops": 1.2318949631604672}, "build_over64_stress_qm2048_k96": {"flashlib_ms": 0.679856, "kernel_ms": 0.351521, "ratio_vs_flashlib": 1.9339072203367653, "tflops": 3.0545595398283463}, "build_over64_stress_qm4096_k96": {"flashlib_ms": 1.218385, "kernel_ms": 0.46208, "ratio_vs_flashlib": 2.6367403912742384, "tflops": 9.294856509695292}}, "rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148": {"rag_microbatch_largek_b1_q16_m100000_d128_k32": {"flashlib_ms": 0.133824, "kernel_ms": 0.13584, "ratio_vs_flashlib": 0.9851590106007068, "tflops": 3.0153121319199063}, "rag_microbatch_largek_b1_q16_m131071_d128_k32": {"flashlib_ms": 0.157568, "kernel_ms": 0.161441, "ratio_vs_flashlib": 0.9760098116339716, "tflops": 3.3254676073612033}, "rag_microbatch_largek_b1_q32_m100000_d128_k32": {"flashlib_ms": 0.158624, "kernel_ms": 0.143809, "ratio_vs_flashlib": 1.1030185871537943, "tflops": 5.696444589698837}, "rag_microbatch_largek_b1_q8_m100000_d128_k32": {"flashlib_ms": 0.106624, "kernel_ms": 0.074336, "ratio_vs_flashlib": 1.4343521308652605, "tflops": 2.755058114507103}}, "rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148": {"rag_microbatch_largek_b1_q16_m100000_d128_k32": {"flashlib_ms": 0.134017, "kernel_ms": 0.135552, "ratio_vs_flashlib": 0.988675932483475, "tflops": 3.021718602455146}, "rag_microbatch_largek_b1_q16_m131071_d128_k32": {"flashlib_ms": 0.157824, "kernel_ms": 0.163168, "ratio_vs_flashlib": 0.967248480094136, "tflops": 3.2902702490684446}, "rag_microbatch_largek_b1_q32_m100000_d128_k32": {"flashlib_ms": 0.159264, "kernel_ms": 0.141312, "ratio_vs_flashlib": 1.1270380434782608, "tflops": 5.797101449275363}, "rag_microbatch_largek_b1_q8_m100000_d128_k32": {"flashlib_ms": 0.107232, "kernel_ms": 0.077121, "ratio_vs_flashlib": 1.3904384019916753, "tflops": 2.655567225528715}}, "ragonline_mbucket_4fc7_q1m262_v1_980c": {"rag_online_b1_q1_m100000_d128_k10": {"flashlib_ms": 0.065024, "kernel_ms": 0.057119, "ratio_vs_flashlib": 1.1383952800294121, "tflops": 0.4481871181218159}, "rag_online_irregular_b1_q1_m131071_d128_k10": {"flashlib_ms": 0.086496, "kernel_ms": 0.068832, "ratio_vs_flashlib": 1.2566248256624826, "tflops": 0.48747931194793115}, "rag_online_irregular_b1_q1_m262143_d128_k10": {"flashlib_ms": 0.091584, "kernel_ms": 0.110944, "ratio_vs_flashlib": 0.8254975483126622, "tflops": 0.6048872223824632}, "rag_online_large_m_b1_q1_m250000_d128_k10": {"flashlib_ms": 0.116736, "kernel_ms": 0.105824, "ratio_vs_flashlib": 1.103114605382522, "tflops": 0.6047777441790142}}}'))
PRODUCTION_ROUTE_MODULES = {**baseline_wide.PRODUCTION_ROUTE_MODULES, SEED_SPLIT148_7E5D_ID: ROUTE_SPLIT148_7E5D_ENTRYPOINT, SEED_Q16SPLIT148_3F2D_ID: ROUTE_Q16SPLIT148_3F2D_ENTRYPOINT, SEED_CACHEDMERGE_AF67_ID: ROUTE_CACHEDMERGE_AF67_ENTRYPOINT, SEED_K96_EXACTALL_B6C4_ID: ROUTE_K96_EXACTALL_B6C4_ENTRYPOINT, SEED_MIDK_E080_ID: ROUTE_MIDK_E080_ENTRYPOINT, SEED_Q1M262_980C_ID: ROUTE_Q1M262_980C_ENTRYPOINT, BASELINE_ID: ROUTE_BASELINE_ENTRYPOINT}
CANDIDATE_DISPATCHERS = ({'id': BASELINE_ID, 'entrypoint': BASELINE_ENTRYPOINT, 'consumed_seeds': (), 'guard_plan': ('8199 widecombine exact non-D128 guards', 'then exported 4247 guard stack'), 'expected_shape_wins': baseline_wide.TARGET_SHAPES, 'fallback': baseline_wide.ROUTE_BASE_4247_ENTRYPOINT, 'rejected_reason': 'same-session full82 baseline'}, {'id': 'candidate_split148_k32_overlay_7e5d_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_split148_k32_overlay_7e5d', 'consumed_seeds': (SEED_SPLIT148_7E5D_ID,), 'guard_plan': ('7e5d all-K32 split148 exact guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': K32_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'comparison candidate for q16split148 replay'}, {'id': 'candidate_q16split148_k32_overlay_3f2d_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_k32_overlay_3f2d', 'consumed_seeds': (SEED_Q16SPLIT148_3F2D_ID,), 'guard_plan': ('3f2d K32 q16-only split148 exact guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': K32_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on repeated full82 A/B'}, {'id': 'candidate_cachedmerge_non128_overlay_af67_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_cachedmerge_non128_overlay_af67', 'consumed_seeds': (SEED_CACHEDMERGE_AF67_ID,), 'guard_plan': ('af67 exact non-D128 cachedmerge guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': CACHEDMERGE_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on repeated full82 A/B'}, {'id': 'candidate_q16split148_plus_cachedmerge_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_plus_cachedmerge', 'consumed_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID), 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': Q16_CACHEDMERGE_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'selected synthesis candidate only if repeated full82 A/B wins'}, {'id': 'candidate_q16split148_cachedmerge_k96exactall_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_cachedmerge_k96exactall', 'consumed_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID, SEED_K96_EXACTALL_B6C4_ID), 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', 'b6c4 exact BF16 build B=1 Q=M in {1024,2048,4096} D=128 K=96 guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': ABEE_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'selected dispatcher-consumption candidate for K96 exact-all b6c4'}, {'id': 'candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_cachedmerge_k96exactall_e080_q1m262', 'consumed_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID, SEED_K96_EXACTALL_B6C4_ID, SEED_MIDK_E080_ID, SEED_Q1M262_980C_ID), 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', 'b6c4 exact BF16 build B=1 Q=M in {1024,2048,4096} D=128 K=96 guards', 'e080 exact BF16 build B=1 Q=M in {2048,4096} D=128 K in {11,12,13} guards', '980c exact BF16 online Q=1 D=128 K=10 M in {100000,131071,250000,262143} guards', 'then 8199-widecombine fallback'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'selected dispatcher-consumption candidate for e080 plus 980c over abee'})
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:ir"}'))

def _seed_midk_e080():
    return import_module('loom.examples.weave.knn_build_midk_k11k13_e080_v1')

def _dtype_name(inputs: dict[str, Any]) -> str:
    return matrix_7e5d._dtype_name(inputs)

def _matches_contract_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return matrix_7e5d._matches_contract_spec(inputs, spec)

def _cachedmerge_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = seed_cachedmerge._target_label_for_inputs(inputs)
    if label in CACHEDMERGE_TARGET_SHAPES:
        return label
    return None

def _k32_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        spec = _K32_SHAPE_SPECS.get(str(label))
        if spec is not None and _matches_contract_spec(inputs, spec):
            return str(label)
    for candidate_label, spec in _K32_SHAPE_SPECS.items():
        if _matches_contract_spec(inputs, spec):
            return candidate_label
    return None

def _k96_exactall_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        spec = _K96_EXACTALL_SHAPE_SPECS.get(str(label))
        if spec is not None and _matches_contract_spec(inputs, spec):
            return str(label)
    for candidate_label, spec in _K96_EXACTALL_SHAPE_SPECS.items():
        if _matches_contract_spec(inputs, spec):
            return candidate_label
    return None

def _midk_e080_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        spec = _MIDK_E080_SHAPE_SPECS.get(str(label))
        if spec is not None and _matches_contract_spec(inputs, spec):
            return str(label)
    for candidate_label, spec in _MIDK_E080_SHAPE_SPECS.items():
        if _matches_contract_spec(inputs, spec):
            return candidate_label
    return None

def _q1m262_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        spec = _Q1M262_SHAPE_SPECS.get(str(label))
        if spec is not None and _matches_contract_spec(inputs, spec):
            return str(label)
    for candidate_label, spec in _Q1M262_SHAPE_SPECS.items():
        if _matches_contract_spec(inputs, spec):
            return candidate_label
    return None

def _selected_seed_for_inputs(inputs: dict[str, Any], *, enable_cachedmerge: bool=True, enable_q16split148: bool=True, enable_split148_7e5d: bool=False, enable_k96_exactall: bool=True, enable_midk_e080: bool=True, enable_q1m262: bool=True) -> tuple[str | None, str | None]:
    if enable_cachedmerge:
        label = _cachedmerge_label_for_inputs(inputs)
        if label is not None:
            return (SEED_CACHEDMERGE_AF67_ID, label)
    if enable_q16split148:
        label = _k32_label_for_inputs(inputs)
        if label is not None:
            return (SEED_Q16SPLIT148_3F2D_ID, label)
    if enable_split148_7e5d:
        label = _k32_label_for_inputs(inputs)
        if label is not None:
            return (SEED_SPLIT148_7E5D_ID, label)
    if enable_k96_exactall:
        label = _k96_exactall_label_for_inputs(inputs)
        if label is not None:
            return (SEED_K96_EXACTALL_B6C4_ID, label)
    if enable_midk_e080:
        label = _midk_e080_label_for_inputs(inputs)
        if label is not None:
            return (SEED_MIDK_E080_ID, label)
    if enable_q1m262:
        label = _q1m262_label_for_inputs(inputs)
        if label is not None:
            return (SEED_Q1M262_980C_ID, label)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_cachedmerge: bool=True, enable_q16split148: bool=True, enable_split148_7e5d: bool=False, enable_k96_exactall: bool=True, enable_midk_e080: bool=True, enable_q1m262: bool=True) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_cachedmerge=enable_cachedmerge, enable_q16split148=enable_q16split148, enable_split148_7e5d=enable_split148_7e5d, enable_k96_exactall=enable_k96_exactall, enable_midk_e080=enable_midk_e080, enable_q1m262=enable_q1m262)
        if selected_seed == SEED_CACHEDMERGE_AF67_ID:
            return seed_cachedmerge.route_for_contract_inputs(inputs)
        if selected_seed == SEED_Q16SPLIT148_3F2D_ID:
            return seed_q16split148.route_for_contract_inputs(inputs)
        if selected_seed == SEED_SPLIT148_7E5D_ID:
            return seed_split148.route_for_contract_inputs(inputs)
        if selected_seed == SEED_K96_EXACTALL_B6C4_ID:
            return seed_k96_exactall.route_for_contract_inputs(inputs)
        if selected_seed == SEED_MIDK_E080_ID:
            return _seed_midk_e080().route_for_contract_inputs(inputs)
        if selected_seed == SEED_Q1M262_980C_ID:
            return seed_q1m262.route_for_contract_inputs(inputs)
    return baseline_wide.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_cachedmerge: bool=True, enable_q16split148: bool=True, enable_split148_7e5d: bool=False, enable_k96_exactall: bool=True, enable_midk_e080: bool=True, enable_q1m262: bool=True) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_cachedmerge=enable_cachedmerge, enable_q16split148=enable_q16split148, enable_split148_7e5d=enable_split148_7e5d, enable_k96_exactall=enable_k96_exactall, enable_midk_e080=enable_midk_e080, enable_q1m262=enable_q1m262)
        if selected_seed == SEED_CACHEDMERGE_AF67_ID:
            seed_cachedmerge.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_Q16SPLIT148_3F2D_ID:
            seed_q16split148.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_SPLIT148_7E5D_ID:
            seed_split148.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_K96_EXACTALL_B6C4_ID:
            seed_k96_exactall.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_MIDK_E080_ID:
            _seed_midk_e080().launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_Q1M262_980C_ID:
            seed_q1m262.launch_from_contract_inputs(inputs)
            return
    baseline_wide.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_split148_k32_overlay_7e5d(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=False, enable_q16split148=False, enable_split148_7e5d=True, enable_k96_exactall=False, enable_midk_e080=False, enable_q1m262=False)

def candidate_q16split148_k32_overlay_3f2d(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=False, enable_q16split148=True, enable_k96_exactall=False, enable_midk_e080=False, enable_q1m262=False)

def candidate_cachedmerge_non128_overlay_af67(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=True, enable_q16split148=False, enable_k96_exactall=False, enable_midk_e080=False, enable_q1m262=False)

def candidate_q16split148_plus_cachedmerge(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=True, enable_q16split148=True, enable_k96_exactall=False, enable_midk_e080=False, enable_q1m262=False)

def candidate_q16split148_cachedmerge_k96exactall(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=True, enable_q16split148=True, enable_k96_exactall=True, enable_midk_e080=False, enable_q1m262=False)

def candidate_q16split148_cachedmerge_k96exactall_e080_q1m262(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_cachedmerge=True, enable_q16split148=True, enable_k96_exactall=True, enable_midk_e080=True, enable_q1m262=True)

def candidate_baseline_wide(inputs: dict[str, Any]) -> None:
    baseline_wide.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)
CANDIDATE_KEYS = ('split148_k32_overlay_7e5d', 'q16split148_k32_overlay_3f2d', 'cachedmerge_non128_overlay_af67', 'q16split148_plus_cachedmerge', 'q16split148_cachedmerge_k96exactall', 'q16split148_cachedmerge_k96exactall_e080_q1m262')
DEFAULT_CANDIDATE_KEY = 'q16split148_cachedmerge_k96exactall_e080_q1m262'
CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {'split148_k32_overlay_7e5d': {'candidate_id': 'candidate_split148_k32_overlay_7e5d_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_split148_k32_overlay_7e5d', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_split148_k32_overlay_7e5d', 'kernel_fn': candidate_split148_k32_overlay_7e5d, 'enabled': {'enable_cachedmerge': False, 'enable_q16split148': False, 'enable_split148_7e5d': True, 'enable_k96_exactall': False, 'enable_midk_e080': False, 'enable_q1m262': False}, 'selected_seeds': (SEED_SPLIT148_7E5D_ID,), 'target_shapes': K32_TARGET_SHAPES, 'guard_plan': ('7e5d all-K32 split148 exact guards', '8199-widecombine fallback')}, 'q16split148_k32_overlay_3f2d': {'candidate_id': 'candidate_q16split148_k32_overlay_3f2d_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_k32_overlay_3f2d', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_q16split148_k32_overlay_3f2d', 'kernel_fn': candidate_q16split148_k32_overlay_3f2d, 'enabled': {'enable_cachedmerge': False, 'enable_q16split148': True, 'enable_split148_7e5d': False, 'enable_k96_exactall': False, 'enable_midk_e080': False, 'enable_q1m262': False}, 'selected_seeds': (SEED_Q16SPLIT148_3F2D_ID,), 'target_shapes': K32_TARGET_SHAPES, 'guard_plan': ('3f2d K32 q16-only split148 exact guards', '8199-widecombine fallback')}, 'cachedmerge_non128_overlay_af67': {'candidate_id': 'candidate_cachedmerge_non128_overlay_af67_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_cachedmerge_non128_overlay_af67', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_cachedmerge_non128_overlay_af67', 'kernel_fn': candidate_cachedmerge_non128_overlay_af67, 'enabled': {'enable_cachedmerge': True, 'enable_q16split148': False, 'enable_split148_7e5d': False, 'enable_k96_exactall': False, 'enable_midk_e080': False, 'enable_q1m262': False}, 'selected_seeds': (SEED_CACHEDMERGE_AF67_ID,), 'target_shapes': CACHEDMERGE_TARGET_SHAPES, 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '8199-widecombine fallback')}, 'q16split148_plus_cachedmerge': {'candidate_id': 'candidate_q16split148_plus_cachedmerge_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_plus_cachedmerge', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_q16split148_plus_cachedmerge', 'kernel_fn': candidate_q16split148_plus_cachedmerge, 'enabled': {'enable_cachedmerge': True, 'enable_q16split148': True, 'enable_split148_7e5d': False, 'enable_k96_exactall': False, 'enable_midk_e080': False, 'enable_q1m262': False}, 'selected_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID), 'target_shapes': Q16_CACHEDMERGE_TARGET_SHAPES, 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', '8199-widecombine fallback')}, 'q16split148_cachedmerge_k96exactall': {'candidate_id': 'candidate_q16split148_cachedmerge_k96exactall_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_cachedmerge_k96exactall', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_q16split148_cachedmerge_k96exactall', 'kernel_fn': candidate_q16split148_cachedmerge_k96exactall, 'enabled': {'enable_cachedmerge': True, 'enable_q16split148': True, 'enable_split148_7e5d': False, 'enable_k96_exactall': True, 'enable_midk_e080': False, 'enable_q1m262': False}, 'selected_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID, SEED_K96_EXACTALL_B6C4_ID), 'target_shapes': ABEE_TARGET_SHAPES, 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', 'b6c4 exact BF16 build B=1 Q=M in {1024,2048,4096} D=128 K=96 guards', '8199-widecombine fallback')}, 'q16split148_cachedmerge_k96exactall_e080_q1m262': {'candidate_id': 'candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1', 'entrypoint': f'{MODULE}:candidate_q16split148_cachedmerge_k96exactall_e080_q1m262', 'benchmark_entrypoint': f'{MODULE}:benchmark_candidate_q16split148_cachedmerge_k96exactall_e080_q1m262', 'kernel_fn': candidate_q16split148_cachedmerge_k96exactall_e080_q1m262, 'enabled': {'enable_cachedmerge': True, 'enable_q16split148': True, 'enable_split148_7e5d': False, 'enable_k96_exactall': True, 'enable_midk_e080': True, 'enable_q1m262': True}, 'selected_seeds': (SEED_CACHEDMERGE_AF67_ID, SEED_Q16SPLIT148_3F2D_ID, SEED_K96_EXACTALL_B6C4_ID, SEED_MIDK_E080_ID, SEED_Q1M262_980C_ID), 'target_shapes': TARGET_SHAPES, 'guard_plan': ('af67 exact non-D128 cachedmerge guards', '3f2d K32 q16-only split148 exact guards', 'b6c4 exact BF16 build B=1 Q=M in {1024,2048,4096} D=128 K=96 guards', 'e080 exact BF16 build B=1 Q=M in {2048,4096} D=128 K in {11,12,13} guards', '980c exact BF16 online Q=1 D=128 K=10 M in {100000,131071,250000,262143} guards', '8199-widecombine fallback')}}

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(f'unknown candidate key {candidate_key!r}; expected one of {CANDIDATE_KEYS}') from exc

def _candidate_enabled_kwargs(candidate_key: str) -> dict[str, bool]:
    return dict(_candidate_config(candidate_key)['enabled'])

def _candidate_target_shapes(candidate_key: str) -> tuple[str, ...]:
    return tuple(_candidate_config(candidate_key)['target_shapes'])

def _candidate_selected_seeds(candidate_key: str) -> tuple[str, ...]:
    return tuple(_candidate_config(candidate_key)['selected_seeds'])

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], Any]:
    return _candidate_config(candidate_key)['kernel_fn']

def _candidate_id(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_benchmark_entrypoint(candidate_key: str | None) -> str:
    if candidate_key is None:
        return BASELINE_ENTRYPOINT
    return str(_candidate_config(candidate_key)['benchmark_entrypoint'])

def _candidate_route_for_inputs(inputs: dict[str, Any], candidate_key: str | None) -> str:
    if candidate_key is None:
        return baseline_wide.route_for_contract_inputs(inputs)
    return route_for_contract_inputs(inputs, **_candidate_enabled_kwargs(candidate_key))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return baseline_wide._select_contract_shapes(shape_labels)

def _benchmark_shapes(shape_labels, *, time_flashlib: bool):
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = _benchmark_shapes(shape_labels, time_flashlib=time_flashlib)
        return evaluate_contract(shapes=shapes, correctness=correctness, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return baseline_wide._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return baseline_wide._inputs_for_label(label)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return baseline_wide._normalize_route_row(row)

def _baseline_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, comparison_candidate_key: str | None=None) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(baseline_wide.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = _candidate_route_for_inputs(inputs, comparison_candidate_key)
    row['base_4247_route'] = base_4247.route_for_contract_inputs(inputs)
    return _normalize_route_row(row)

def _shape_spec_for_seed(seed_id: str, label: str) -> dict[str, Any]:
    if seed_id == SEED_CACHEDMERGE_AF67_ID:
        return seed_cachedmerge.SHAPE_SPECS[label]
    if seed_id in {SEED_Q16SPLIT148_3F2D_ID, SEED_SPLIT148_7E5D_ID}:
        return _K32_SHAPE_SPECS[label]
    if seed_id == SEED_K96_EXACTALL_B6C4_ID:
        return _K96_EXACTALL_SHAPE_SPECS[label]
    if seed_id == SEED_MIDK_E080_ID:
        return _MIDK_E080_SHAPE_SPECS[label]
    if seed_id == SEED_Q1M262_980C_ID:
        return _Q1M262_SHAPE_SPECS[label]
    raise KeyError(seed_id)

def _seed_entrypoint(seed_id: str) -> str:
    return {SEED_CACHEDMERGE_AF67_ID: ROUTE_CACHEDMERGE_AF67_ENTRYPOINT, SEED_Q16SPLIT148_3F2D_ID: ROUTE_Q16SPLIT148_3F2D_ENTRYPOINT, SEED_SPLIT148_7E5D_ID: ROUTE_SPLIT148_7E5D_ENTRYPOINT, SEED_K96_EXACTALL_B6C4_ID: ROUTE_K96_EXACTALL_B6C4_ENTRYPOINT, SEED_MIDK_E080_ID: ROUTE_MIDK_E080_ENTRYPOINT, SEED_Q1M262_980C_ID: ROUTE_Q1M262_980C_ENTRYPOINT}[seed_id]

def _guard_id(seed_id: str) -> str:
    return {SEED_CACHEDMERGE_AF67_ID: 'af67_cachedmerge_non128_exact_shape_guard', SEED_Q16SPLIT148_3F2D_ID: '3f2d_k32_q16split148_exact_shape_guard', SEED_SPLIT148_7E5D_ID: '7e5d_k32_split148_exact_shape_guard', SEED_K96_EXACTALL_B6C4_ID: 'b6c4_k96_exactall_exact_qm_guard', SEED_MIDK_E080_ID: 'e080_midk_k11k13_exact_guard', SEED_Q1M262_980C_ID: '980c_q1_m262_exact_mbucket_guard'}[seed_id]

def _producer_for_seed(seed_id: str, label: str) -> str:
    if seed_id == SEED_CACHEDMERGE_AF67_ID:
        return seed_cachedmerge._producer_for_label(label)
    if seed_id == SEED_Q16SPLIT148_3F2D_ID:
        inputs = _inputs_for_label(label)
        if seed_q16split148._eligible_q16_split148(inputs):
            return 'b3e0_rowld1_q16_split148'
        return 'b3e0_parent_default_split144'
    if seed_id == SEED_SPLIT148_7E5D_ID:
        return matrix_7e5d._producer_for_seed(matrix_7e5d.SEED_K32_ID, label)
    if seed_id == SEED_K96_EXACTALL_B6C4_ID:
        return 'e5db_exact_no_tail_k96_stage1_prefill'
    if seed_id == SEED_MIDK_E080_ID:
        return 'e080_exact_midk_tcgen05_tma_stage1'
    if seed_id == SEED_Q1M262_980C_ID:
        return 'aa88_q1m_v3_split72_split74_tcgen05_tma_stage1'
    raise KeyError(seed_id)

def _split_count_for_seed(seed_id: str, label: str) -> int | None:
    if seed_id == SEED_CACHEDMERGE_AF67_ID:
        return seed_cachedmerge._split_count_for_label(label)
    if seed_id == SEED_Q16SPLIT148_3F2D_ID:
        inputs = _inputs_for_label(label)
        if seed_q16split148._eligible_q16_split148(inputs):
            return seed_q16split148.K32_Q16_SPLIT_COUNT
        return seed_q16split148.K32_DEFAULT_SPLIT_COUNT
    if seed_id == SEED_SPLIT148_7E5D_ID:
        return seed_split148.K32_SPLIT_COUNT
    if seed_id == SEED_K96_EXACTALL_B6C4_ID:
        return seed_k96_exactall._select_split_count(_K96_EXACTALL_SHAPE_SPECS[label]['Q'])
    if seed_id == SEED_MIDK_E080_ID:
        spec = _MIDK_E080_SHAPE_SPECS[label]
        return _seed_midk_e080()._split_count_for_shape(top_k=int(spec['K']), n_query=int(spec['Q']))
    if seed_id == SEED_Q1M262_980C_ID:
        spec = _Q1M262_SHAPE_SPECS[label]
        return seed_q1m262.SPLIT_BY_M[int(spec['M'])]
    return None

def _targeted_seed_row(seed_id: str, label: str) -> dict[str, Any]:
    return TARGETED_SEED_ROWS_BY_SEED[seed_id][label]

def _specialized_trace_record(inputs: dict[str, Any], seed_id: str, label: str, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> dict[str, Any]:
    spec = _shape_spec_for_seed(seed_id, label)
    targeted = dict(_targeted_seed_row(seed_id, label))
    baseline_route = _candidate_route_for_inputs(inputs, comparison_candidate_key)
    classification = 'kernel-slow' if targeted.get('ratio_vs_flashlib', 1.0) < 1.0 else 'seed-consumed'
    return {'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, **_candidate_enabled_kwargs(candidate_key)), 'selected_entrypoint': _seed_entrypoint(seed_id), 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': _guard_id(seed_id), 'guard_condition': f'exact BF16 B={spec['B']} Q={spec['Q']} M={spec['M']} D={spec['D']} K={spec['K']} build={spec.get('build', False)}', 'coverage': 'synthesized full82 seed route selected before the 8199-widecombine baseline', 'consumed_seed': seed_id, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'base_4247_route': base_4247.route_for_contract_inputs(inputs), 'producer': _producer_for_seed(seed_id, label), 'split_count': _split_count_for_seed(seed_id, label), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'row_selection': targeted, 'classification': classification, 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> dict[str, Any]:
    selected_seed, label = _selected_seed_for_inputs(inputs, **_candidate_enabled_kwargs(candidate_key))
    if force_fallback and selected_seed is not None and (label is not None):
        row = _baseline_route_trace_record(inputs, comparison_candidate_key=comparison_candidate_key)
        row['expected_seed'] = selected_seed
        row['guard_id'] = 'forced_fallback_synthesized_portfolio_disabled'
        row['guard_condition'] = f'forced fallback to {BASELINE_ID}; synthesized seed overlays disabled'
        row['forced_disabled_seeds'] = _candidate_selected_seeds(candidate_key)
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    if not force_fallback and selected_seed is not None and (label is not None):
        return _normalize_route_row(_specialized_trace_record(inputs, selected_seed, label, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key))
    return _baseline_route_trace_record(inputs, comparison_candidate_key=comparison_candidate_key)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return baseline_wide._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return baseline_wide._rows_for_labels(report, labels)

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> list[dict[str, Any]]:
    matrix = []
    enabled = _candidate_enabled_kwargs(candidate_key)
    for label in _candidate_target_shapes(candidate_key):
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        selected_seed, _selected_label = _selected_seed_for_inputs(inputs, **enabled)
        targeted = _targeted_seed_row(str(selected_seed), label) if selected_seed else {}
        matrix.append({'shape_key': label, 'baseline_route': _candidate_route_for_inputs(inputs, comparison_candidate_key), 'base_4247_route': base_4247.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, **enabled), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_dispatcher': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_kernel_ms': targeted.get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': targeted.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': item['candidate_id'], 'selected_seed': item['selected_seed'], 'metric_delta': item['metric_delta_ms'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None) -> dict[str, Any]:
    return {item['shape_key']: {'candidate_ms': item['candidate_ms'], 'baseline_dispatcher_ms': item['baseline_ms'], 'flashlib_ms': item['flashlib_ms'], 'speedup_vs_baseline_dispatcher': item['speedup_vs_baseline_dispatcher'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'candidate_route': item['candidate_route'], 'baseline_dispatcher_route': item['baseline_route'], 'base_4247_route': item['base_4247_route'], 'selected_seed': item['selected_seed'], 'targeted_seed_kernel_ms': item['targeted_seed_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': item['targeted_seed_ratio_vs_flashlib'], 'candidate_passed': candidate_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed'), 'baseline_passed': baseline_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed')} for item in _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key)}

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    annotated = []
    target_shape_set = set(_candidate_target_shapes(candidate_key))
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        out['route_changed_vs_baseline_dispatcher'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        ratio = candidate_row.get('ratio_vs_flashlib')
        if ratio is None:
            ratio = out.get('targeted_seed_ratio_vs_flashlib')
        if label in target_shape_set:
            if not out.get('selected_seed'):
                out['classification'] = 'guard-miss'
            elif isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow'
            elif out['relative_speedup_vs_baseline'] is not None and out['relative_speedup_vs_baseline'] < 1.0:
                out['classification'] = 'kernel-slow'
            elif not out.get('route_changed_vs_baseline_dispatcher'):
                out['classification'] = 'route-ok'
            else:
                out['classification'] = 'seed-consumed'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        elif not out.get('route_changed_vs_baseline_dispatcher'):
            out['classification'] = 'route-ok'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes(candidate_key=candidate_key)}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        trace_row = trace_by_label.get(label, {})
        ratio = row.get('ratio_vs_flashlib')
        if ratio is None:
            ratio = trace_row.get('targeted_seed_ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event'

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    return f'shape{len(tuple(shape_labels))}'

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, candidate_key: str=DEFAULT_CANDIDATE_KEY, comparison_candidate_key: str | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    cfg = _candidate_config(candidate_key)
    consumed_labels = _candidate_target_shapes(candidate_key)
    selected_route_labels = tuple(dict.fromkeys((*baseline_wide.SELECTED_TARGET_SHAPES, *consumed_labels)))
    timing_backend = _timing_backend_name(use_cupti)
    denominator = _denominator_name(shape_labels)
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_flashlib = _below_flashlib_rows(candidate_report, candidate_key=candidate_key)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'candidate_key': candidate_key, 'candidate_id': cfg['candidate_id'], 'comparison_baseline_candidate_key': comparison_candidate_key or BASELINE_ID, 'baseline_candidate_id': BASELINE_ID if comparison_candidate_key is None else _candidate_id(comparison_candidate_key), 'selected_seeds': _candidate_selected_seeds(candidate_key), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': cfg['benchmark_entrypoint'], 'baseline_entrypoint': _candidate_benchmark_entrypoint(comparison_candidate_key), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': selected_route_labels, 'consumed_seed_labels': consumed_labels, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, selected_route_labels), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, selected_route_labels), 'consumed_seed_rows': _rows_for_labels(candidate_report, consumed_labels), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, consumed_labels), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key), 'targeted_seed_rows_by_seed': TARGETED_SEED_ROWS_BY_SEED, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': cfg['candidate_id'], 'guard_plan': cfg['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True, candidate_key=candidate_key, comparison_candidate_key=comparison_candidate_key), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(*, candidate_key: str=DEFAULT_CANDIDATE_KEY, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, baseline_candidate_key: str | None=None, benchmark_correctness: bool=False, time_flashlib: bool=False) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_wide if baseline_candidate_key is None else _candidate_kernel_fn(baseline_candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, candidate_key=candidate_key, comparison_candidate_key=baseline_candidate_key, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_split148_k32_overlay_7e5d(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='split148_k32_overlay_7e5d', **kwargs)

def benchmark_candidate_q16split148_k32_overlay_3f2d(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='q16split148_k32_overlay_3f2d', **kwargs)

def benchmark_candidate_cachedmerge_non128_overlay_af67(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='cachedmerge_non128_overlay_af67', **kwargs)

def benchmark_candidate_q16split148_plus_cachedmerge(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='q16split148_plus_cachedmerge', **kwargs)

def benchmark_candidate_q16split148_cachedmerge_k96exactall(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='q16split148_cachedmerge_k96exactall', **kwargs)

def benchmark_candidate_q16split148_cachedmerge_k96exactall_e080_q1m262(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='q16split148_cachedmerge_k96exactall_e080_q1m262', **kwargs)

def benchmark_knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, baseline_candidate_key: str | None=None, benchmark_correctness: bool=False, time_flashlib: bool=False) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key=DEFAULT_CANDIDATE_KEY, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, baseline_candidate_key=baseline_candidate_key, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_subset_matrix(*, use_cupti: bool=True, shape_labels=None, candidate_keys: tuple[str, ...]=CANDIDATE_KEYS, baseline_candidate_key: str | None=None, benchmark_correctness: bool=False, time_flashlib: bool=False) -> dict[str, Any]:
    timing_backend = _timing_backend_name(use_cupti)
    denominator = _denominator_name(shape_labels)
    baseline_candidate_id = BASELINE_ID if baseline_candidate_key is None else _candidate_id(baseline_candidate_key)
    baseline_entrypoint = _candidate_benchmark_entrypoint(baseline_candidate_key)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_wide if baseline_candidate_key is None else _candidate_kernel_fn(baseline_candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payloads = {key: benchmark_candidate_portfolio(candidate_key=key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, baseline_candidate_key=baseline_candidate_key, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib) for key in candidate_keys}
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'matrix_id': 'q16split148_cachedmerge_matrix_over_8199_full82_v1', 'baseline_candidate_key': baseline_candidate_key or BASELINE_ID, 'baseline_candidate_id': baseline_candidate_id, 'baseline_entrypoint': baseline_entrypoint, 'baseline_tflops': baseline_metric, 'baseline_all_correct': baseline_report['summary']['all_correct'], 'baseline_report': baseline_report, 'candidate_keys': candidate_keys, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_summaries': {key: {'candidate_id': payload['candidate_id'], 'measured_entrypoint': payload['measured_entrypoint'], 'selected_seeds': payload['selected_seeds'], 'tflops': payload['tflops'], 'metric_delta': payload['metric_delta'], 'all_correct': payload['all_correct'], 'performance_comparable': payload['performance_comparable'], 'performance_coverage': payload['performance_coverage'], 'hot_bucket_blocker_count': len(payload['hot_bucket_blockers'])} for key, payload in payloads.items()}, 'payloads': payloads, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'timing_backends': _timing_backends_for_reports(baseline_report, *(payload['report'] for payload in payloads.values())), 'route_trace_included': True, 'rank_objective': {key: {'metric': 'tflops', 'direction': 'maximize', 'value': payload['tflops'], 'baseline_value': baseline_metric, 'delta': payload['metric_delta'], 'denominator': denominator} for key, payload in payloads.items()}}

def benchmark_repeated_pair_matrix(*, use_cupti: bool=True, shape_labels=None, candidate_keys: tuple[str, ...]=CANDIDATE_KEYS, baseline_candidate_key: str | None=None, repeated_pair_count: int=3, benchmark_correctness: bool=False, time_flashlib: bool=False) -> dict[str, Any]:
    timing_backend = _timing_backend_name(use_cupti)
    denominator = _denominator_name(shape_labels)
    pairs = []
    for pair_index in range(repeated_pair_count):
        for key in candidate_keys:
            candidate_first = pair_index % 2 == 0
            baseline_report = None
            if not candidate_first:
                baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_wide if baseline_candidate_key is None else _candidate_kernel_fn(baseline_candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
            payload = benchmark_candidate_portfolio(candidate_key=key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, baseline_candidate_key=baseline_candidate_key, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
            pairs.append({'pair_index': pair_index, 'order': (payload['candidate_id'], payload['baseline_candidate_id']) if candidate_first else (payload['baseline_candidate_id'], payload['candidate_id']), 'candidate_key': key, 'baseline_tflops': payload['baseline_tflops'], 'candidate_tflops': payload['tflops'], 'delta': payload['metric_delta'], 'all_correct': payload['all_correct'], 'baseline_all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['performance_comparable'], 'route_trace_included': payload['route_trace_included'], 'candidate_id': payload['candidate_id'], 'measured_entrypoint': payload['measured_entrypoint'], 'baseline_entrypoint': payload['baseline_entrypoint']})
    by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in candidate_keys}
    for row in pairs:
        by_key[str(row['candidate_key'])].append(row)
    aggregate = {}
    for key, rows in by_key.items():
        deltas = [float(row['delta']) for row in rows if row.get('delta') is not None]
        candidate_values = [float(row['candidate_tflops']) for row in rows if row.get('candidate_tflops') is not None]
        baseline_values = [float(row['baseline_tflops']) for row in rows if row.get('baseline_tflops') is not None]
        aggregate[key] = {'candidate_id': _candidate_id(key), 'pair_count': len(rows), 'median_delta': median(deltas) if deltas else None, 'min_delta': min(deltas) if deltas else None, 'max_delta': max(deltas) if deltas else None, 'median_candidate_tflops': median(candidate_values) if candidate_values else None, 'median_baseline_tflops': median(baseline_values) if baseline_values else None, 'all_correct': all((bool(row.get('all_correct')) for row in rows)), 'baseline_all_correct': all((bool(row.get('baseline_all_correct')) for row in rows))}
    default_rows = aggregate.get(DEFAULT_CANDIDATE_KEY, {})
    return {'matrix_id': 'q16split148_cachedmerge_repeated_pair_full82_v1', 'baseline_candidate_id': BASELINE_ID if baseline_candidate_key is None else _candidate_id(baseline_candidate_key), 'baseline_candidate_key': baseline_candidate_key or BASELINE_ID, 'baseline_entrypoint': _candidate_benchmark_entrypoint(baseline_candidate_key), 'candidate_keys': candidate_keys, 'repeated_pair_count': repeated_pair_count, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'randomized_or_interleaved_order': repeated_pair_count > 1, 'order_policy': 'interleaved alternating order; even zero-based pairs candidate before baseline, odd pairs baseline before candidate', 'same_gpu_class': True, 'same_backend': True, 'same_entrypoint_per_candidate': True, 'paired_rows': pairs, 'aggregate': aggregate, 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': default_rows.get('median_candidate_tflops'), 'baseline_value': default_rows.get('median_baseline_tflops'), 'delta': default_rows.get('median_delta'), 'denominator': denominator}}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return f'full{len(eval_mod.CANONICAL_SHAPES)}'
    return f'shape{len(tuple(shape_labels))}'

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, candidate_key: str | None=None, baseline_candidate_key: str | None=None, repeated_pair_count: int=0, benchmark_correctness: bool=False, time_flashlib: bool=False) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    candidate_keys = CANDIDATE_KEYS if candidate_key is None else (candidate_key,)
    matrix = benchmark_subset_matrix(use_cupti=use_cupti, shape_labels=shape_labels, candidate_keys=candidate_keys, baseline_candidate_key=baseline_candidate_key, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_slug = '8199_widecombine' if baseline_candidate_key is None else str(_candidate_id(baseline_candidate_key)).removeprefix('candidate_')
    baseline_path = out_dir / f'{denom}_same_session_baseline_{baseline_slug}_v1.json'
    summary_path = out_dir / f'{denom}_q16_cachedmerge_matrix_summary_v1.json'
    paths: dict[str, str] = {'same_session_baseline_payload': str(baseline_path), 'matrix_summary': str(summary_path)}
    baseline_path.write_text(json.dumps({'candidate_key': baseline_candidate_key or BASELINE_ID, 'candidate_id': matrix['baseline_candidate_id'], 'measured_entrypoint': matrix['baseline_entrypoint'], 'measured_shape_labels': matrix['measured_shape_labels'], 'timing_backend': timing_backend, 'denominator': denominator, 'timing_backend_requested': matrix['timing_backend_requested'], 'timing_backends': matrix['timing_backends'], 'benchmark_correctness_checked': matrix['benchmark_correctness_checked'], 'benchmark_time_flashlib': matrix['benchmark_time_flashlib'], 'tflops': matrix['baseline_tflops'], 'all_correct': matrix['baseline_all_correct'], 'performance_comparable': matrix['baseline_report']['summary']['performance_comparable'], 'contract_summary': matrix['baseline_report']['summary'], 'contract_performance': matrix['baseline_report']['performance'], 'route_trace': baseline_wide.route_trace_for_contract_shapes(shape_labels) if baseline_candidate_key is None else route_trace_for_contract_shapes(shape_labels, candidate_key=baseline_candidate_key), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': matrix['baseline_tflops'], 'denominator': denominator}, 'report': matrix['baseline_report']}, indent=2, sort_keys=True) + '\n')
    for key, payload in matrix['payloads'].items():
        candidate_id = str(payload['candidate_id']).removeprefix('candidate_')
        candidate_path = out_dir / f'{denom}_dispatch_{candidate_id}.json'
        route_trace_path = out_dir / f'{denom}_route_trace_{candidate_id}.json'
        forced_trace_path = out_dir / f'{denom}_forced_fallback_trace_{candidate_id}.json'
        seed_matrix_path = out_dir / f'{denom}_seed_delta_matrix_{candidate_id}.json'
        candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
        forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
        seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
        paths[f'{key}_candidate_payload'] = str(candidate_path)
        paths[f'{key}_route_trace'] = str(route_trace_path)
        paths[f'{key}_forced_fallback_trace'] = str(forced_trace_path)
        paths[f'{key}_seed_delta_matrix'] = str(seed_matrix_path)
    summary_payload = {key: value for key, value in matrix.items() if key not in {'payloads', 'baseline_report'}}
    summary_payload['candidate_payload_paths'] = {key: paths[f'{key}_candidate_payload'] for key in matrix['payloads']}
    summary_payload['same_session_baseline_payload'] = str(baseline_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + '\n')
    if repeated_pair_count:
        repeated = benchmark_repeated_pair_matrix(use_cupti=use_cupti, shape_labels=shape_labels, candidate_keys=candidate_keys, baseline_candidate_key=baseline_candidate_key, repeated_pair_count=repeated_pair_count, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        repeated_path = out_dir / f'{denom}_repeated_pair_q16_cachedmerge_matrix_v1.json'
        repeated_path.write_text(json.dumps(repeated, indent=2, sort_keys=True) + '\n')
        paths['repeated_pair_payload'] = str(repeated_path)
    return paths
