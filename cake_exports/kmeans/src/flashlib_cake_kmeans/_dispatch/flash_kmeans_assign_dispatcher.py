"""Flash-KMeans Euclidean assignment guarded seed dispatcher.

Minimum architecture: sm_100a. Every production route is Weave-only and uses
existing tcgen05/TMEM seed kernels; no external runtime fallback is present.
The tcgen05/TMEM routes are not intended for sm_120a/sm_121a where ptxas
rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
from dataclasses import dataclass
from typing import Any, Callable
from . import flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_b23d_v2 as _non128d_d160
from . import flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1 as _non128d_v1
from . import flash_kmeans_assign_cleanroom_tcgen05_v10 as _single
from . import flash_kmeans_assign_cleanroom_tcgen05_v15 as _paired
from . import flash_kmeans_assign_d128_splitk_priority_575c_v1 as _d128_priority
from . import flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1 as _tail_pad192_f9b2
from . import flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1 as _d288_hybrid
from . import flash_kmeans_assign_d352_exactd_splitk_c95c_v2 as _d352_exactd
from . import flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 as _d64_direct
from . import flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1 as _d768_priority
from . import flash_kmeans_assign_d480_splitk_k1024_eac2_v1 as _d480_eac2
from . import flash_kmeans_assign_gap_pad_v1 as _gap_pad
from . import flash_kmeans_assign_highd_paired_packedpartial_gridcap160_0194_v1 as _highd_paired_gridcap160
from . import flash_kmeans_assign_highd_splitd_6fcf_v1 as _highd
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _highd_splitk
from . import flash_kmeans_assign_lowdim_e50c_v1 as _lowdim
from . import flash_kmeans_assign_microdim_hybrid_9c0d_v1 as _microdim
from . import flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1 as _highd_no_padding_r63
BLOCK_N = _single.BLOCK_N
BLOCK_K = _single.BLOCK_K
FEAT_D = _single.FEAT_D
D144_D176_F9B2_DIMS = _decode_capture(_json_loads('{"__tuple__": [144, 176]}'))
SUPPORTED_DIMS = _decode_capture(_json_loads('{"__tuple__": [16, 32, 64, 48, 112, 224, 288, 352, 416, 480, 80, 96, 128, 144, 176, 160, 192, 256, 320, 384, 448, 512, 768]}'))
D64_SEED_ID = 'd64-direct-single64-1p2gap-9f2a-v1'
MICRODIM_HYBRID_SEED_ID = _microdim.SEED_ID
LOWDIM_E50C_SEED_ID = 'lowdim-e50c-v1'
SMALL_SEED_ID = 'small-grid-single-tile-v10'
PAIRED_SEED_ID = 'paired-large-v15'
D144_D160_D176_PAD192_F9B2_SEED_ID = 'd144-d160-d176-pad192-tail-repair-f9b2-v1'
D160_PADDED_SEED_ID = 'd160-padded_single-repeated-mma-b23d-v2'
D192_SINGLE_SEED_ID = 'd192-single-repeated-mma-v1'
D192_PAIRED_SEED_ID = 'd192-paired-repeated-mma-v1'
D256_SINGLE_SEED_ID = 'd256-single-repeated-mma-v1'
D288_PARENT_SPLITK_HYBRID_SEED_ID = _d288_hybrid.SEED_ID
HIGHD_SPLITD_SEED_ID = _highd.SEED_ID
HIGHD_SPLITK_SEED_ID = _highd_splitk.SEED_ID
HIGHD_PAIRED_PACKEDPARTIAL_SEED_ID = _highd_paired_gridcap160.SEED_ID
HIGHD_NO_PADDING_PORTFOLIO_R63_SEED_ID = _highd_no_padding_r63.SEED_ID
D768_PRIORITY_SEED_ID = _d768_priority.SEED_ID
D480_EAC2_SEED_ID = _d480_eac2.SEED_ID
GAP_PAD_SEED_ID = _gap_pad.SEED_ID
D352_EXACTD_SEED_ID = _d352_exactd.SEED_ID
D64_ROUTE_ID = 'd64_direct_single64_1p2gap_9f2a_v1'
MICRODIM_HYBRID_ROUTE_ID = _microdim.ROUTE_ID
LOWDIM_E50C_ROUTE_ID = 'lowdim_e50c_v1'
GAP_PAD_ROUTE_ID = _gap_pad.ROUTE_ID
D352_EXACTD_ROUTE_ID = _d352_exactd.ROUTE_ID
SMALL_ROUTE_ID = 'small_grid_single_tile_v10'
PAIRED_ROUTE_ID = 'paired_large_v15'
D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID = 'd128_even_near_floor_v10_repair'
D128_PRIORITY_SPLITK_ROUTE_ID = _d128_priority.ROUTE_ID
D144_D160_D176_PAD192_F9B2_ROUTE_ID = 'd144_d160_d176_pad192_tail_repair_f9b2_v1'
D160_PADDED_ROUTE_ID = 'd160_padded_single_repeated_mma_v2'
D192_SINGLE_ROUTE_ID = 'd192_single_repeated_mma_v1'
D192_PAIRED_ROUTE_ID = 'd192_paired_repeated_mma_v1'
D256_SINGLE_ROUTE_ID = 'd256_single_repeated_mma_v1'
D288_PARENT_SPLITK_HYBRID_ROUTE_ID = _d288_hybrid.ROUTE_ID
HIGHD_SPLITD_ROUTE_ID = _highd.ROUTE_ID
HIGHD_SPLITK_ROUTE_ID = _highd_splitk.ROUTE_ID
HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID = _highd_paired_gridcap160.ROUTE_ID
HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID = _highd_no_padding_r63.ROUTE_ID
D768_PRIORITY_ROUTE_ID = _d768_priority.ROUTE_ID
D480_EAC2_ROUTE_ID = _d480_eac2.ROUTE_ID
GENERIC_FALLBACK_ID = 'aligned_weave_v10_fallback'
UNSUPPORTED_ROUTE_ID = 'unsupported_shape'
SMALL_GRID_N_TILE_CAP = 8
SMALL_GRID_K_TILE_CAP = 2
SLOW_ROUTE_SPEEDUP_THRESHOLD = 0.98
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
D768_PRIORITY_SHAPES = frozenset({(1, 512, 4096), (1, 1024, 4096), (1, 512, 8192), (1, 1024, 8192), (1, 2048, 4096)})
PRIORITY_EXPECTED_SEEDS = {128: _d128_priority.SEED_ID}

@dataclass(frozen=True)
class RouteDecision:
    route_id: str
    entrypoint: str
    selected_seed: str
    route_kind: str
    route_source: str
    guard_id: str
    guard_condition: str
    reason: str

    def trace_row(self, *, shape_key: str | None=None, expected_seed: str | None=None, dispatcher_kernel_ms: float | None=None, shape_specific_kernel_ms: float | None=None, relative_speedup_vs_baseline: float | None=None) -> dict[str, Any]:
        return {'shape_key': shape_key, 'selected_route': self.route_id, 'selected_entrypoint': self.entrypoint, 'selected_seed': self.selected_seed, 'expected_seed': expected_seed, 'route_kind': self.route_kind, 'route_source': self.route_source, 'guard_id': self.guard_id, 'guard_condition': self.guard_condition, 'classification': _classify_route(self, expected_seed, relative_speedup_vs_baseline), 'dispatcher_kernel_ms': dispatcher_kernel_ms, 'shape_specific_kernel_ms': shape_specific_kernel_ms, 'relative_speedup_vs_baseline': relative_speedup_vs_baseline, 'reason': self.reason}
ROUTE_SMALL_V10 = RouteDecision(route_id=SMALL_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_small_grid_single_tile_v10', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 8 and K_tiles <= 2', reason='small-grid anchor uses the v10 single point-tile seed')
ROUTE_PAIRED_V15 = RouteDecision(route_id=PAIRED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v15:launch_for_eval', selected_seed=PAIRED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_paired_large_v15', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 0 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='even point-tile grids use the v15 paired point-tile seed')
ROUTE_ALIGNED_V10_FALLBACK = RouteDecision(route_id=GENERIC_FALLBACK_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='fallback', route_source='generic-weave-fallback', guard_id='guard_aligned_v10_weave_fallback', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 1 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='v15 paired kernel requires an even number of point tiles; v10 is the aligned Weave fallback')
ROUTE_D128_EVEN_NEAR_FLOOR_V10_REPAIR = RouteDecision(route_id=D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='specialized', route_source='generated-variant', guard_id='guard_d128_even_b8_n8192_k256_v10_repair', guard_condition='dtype == bfloat16 and B == 8 and D == 128 and N == 8192 and K == 256', reason='same-session post-D895 replay shows v10 beats paired v15 on the B8/N8192/K256 near-floor row')
ROUTE_D128_PRIORITY_SPLITK = RouteDecision(route_id=D128_PRIORITY_SPLITK_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d128_splitk_priority_575c_v1:launch_for_eval', selected_seed=_d128_priority.SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d128_no_padding_splitk_priority_575c_v1', guard_condition='B == 1 and D == 128 and N in [512,1024,2048] and K in [4096,8192]', reason='exact five-row D128 priority bucket consumes the 64x256 G1/R4 Split-K seed')
ROUTE_D64_DIRECT_9F2A = RouteDecision(route_id=D64_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1:launch_for_eval', selected_seed=D64_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d64_direct_single64_1p2gap_9f2a_v1', guard_condition='dtype == bfloat16 and D == 64 and N % 128 == 0 and K % 256 == 0', reason='D64 lowdim-tail bucket uses the 1d9f direct one-MMA tcgen05 score producer')
ROUTE_MICRODIM_HYBRID = RouteDecision(route_id=MICRODIM_HYBRID_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_microdim_hybrid_9c0d_v1:launch_for_eval', selected_seed=MICRODIM_HYBRID_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_microdim_hybrid_9c0d_v1', guard_condition='dtype == bfloat16 and D in [16, 32] and N % 128 == 0 and K % 256 == 0', reason="D16/D32 consume c92d's hybrid seed: short K=512 rows use direct staging and large/high-K rows keep 6cd2 pack+TMA")
ROUTE_LOWDIM_E50C = RouteDecision(route_id=LOWDIM_E50C_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_lowdim_e50c_v1:launch_for_eval', selected_seed=LOWDIM_E50C_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_lowdim_e50c_v1_d80_d96', guard_condition='dtype == bfloat16 and D in [80, 96] and N % 128 == 0 and K % 256 == 0', reason='D80/D96 lowdim-tail bucket uses the e50c fused pad-to-128 tcgen05 route after fdac missed the same-session floor')
ROUTE_GAP_PAD = RouteDecision(route_id=GAP_PAD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_gap_pad_v1:launch_for_eval', selected_seed=GAP_PAD_SEED_ID, route_kind='specialized', route_source='generated-variant', guard_id='guard_gap_pad_to_supported_seed_v1', guard_condition='dtype == bfloat16 and D in [48, 112, 224, 352, 416, 480] and N % 128 == 0 and K % 256 == 0', reason='between-bucket D rows zero-pad to the next supported Weave seed bucket without changing the delegated seed schedule')
ROUTE_D352_EXACTD = RouteDecision(route_id=D352_EXACTD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d352_exactd_splitk_c95c_v2:launch_for_eval', selected_seed=D352_EXACTD_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d352_exactd_splitk_c95c_v2', guard_condition='dtype == bfloat16 and D == 352 and N/64 <= 32 and K/256 >= 4', reason='exact-D352 D32-chunk Split-K seed removes the D384 pack on its validated envelope')
ROUTE_D480_EAC2 = RouteDecision(route_id=D480_EAC2_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d480_splitk_k1024_eac2_v1:launch_for_eval', selected_seed=D480_EAC2_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d480_splitk_k1024_eac2_v1', guard_condition='dtype == bfloat16 and D == 480 and N % 64 == 0 and K % 256 == 0 and N/64 <= 32 and K/256 >= 4', reason='D480 bounded point-tile K1024+ rows consume the validated tcgen05 producer/reducer seed; low-K and large-N rows retain the Weave gap-pad route')
ROUTE_D288_PARENT_SPLITK_HYBRID = RouteDecision(route_id=D288_PARENT_SPLITK_HYBRID_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1:launch_for_eval', selected_seed=D288_PARENT_SPLITK_HYBRID_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d288_parent_splitk_hybrid_20260629_v1', guard_condition='dtype == bfloat16 and D == 288 and N % 128 == 0 and K % 256 == 0 and (K <= 2048 or K >= 4096)', reason='D288 consumes the measured exact-D parent/Split-K hybrid; the high-K branch includes its CTA Split-K reduction in the dispatcher-timed path')
ROUTE_D144_D176_PAD192_F9B2 = RouteDecision(route_id=D144_D160_D176_PAD192_F9B2_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1:launch_for_eval', selected_seed=D144_D160_D176_PAD192_F9B2_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d144_d176_pad192_tail_repair_f9b2_v1', guard_condition='dtype == bfloat16 and D in [144, 176] and N % 128 == 0 and K % 256 == 0', reason='D144/D176 consume the f9b2 tail-safe seed: pack D to D=192 scratch, then reuse the D192 tcgen05 path; D160 remains on the faster promoted b23d route')
ROUTE_D160_PADDED = RouteDecision(route_id=D160_PADDED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_b23d_v2:launch_for_eval', selected_seed=D160_PADDED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d160_padded_single_repeated_mma_b23d_v2', guard_condition='dtype == bfloat16 and D == 160 and N % 128 == 0 and K % 256 == 0', reason='D160 consumes the 81d5 padded-tail wrapper: pack D=160 to D=192 scratch, then reuse the D192 tcgen05 path')
ROUTE_D192_SINGLE = RouteDecision(route_id=D192_SINGLE_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D192_SINGLE_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d192_single_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 192 and N % 128 == 0 and K % 256 == 0 and (num_n_tiles % 2 != 0 or (num_n_tiles <= 8 and K_tiles <= 2))', reason='D192 small or odd point-tile grids use the b23d single repeated-MMA path')
ROUTE_D192_PAIRED = RouteDecision(route_id=D192_PAIRED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D192_PAIRED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d192_paired_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 192 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 0 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='D192 even point-tile grids use the b23d paired repeated-MMA path')
ROUTE_D256_SINGLE = RouteDecision(route_id=D256_SINGLE_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D256_SINGLE_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d256_single_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 256 and N % 128 == 0 and K % 256 == 0', reason='D256 consumes the existing b23d single repeated-MMA route; paired D256 remains a resource-repair lane')
ROUTE_HIGHD_SPLITD = RouteDecision(route_id=HIGHD_SPLITD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_splitd_6fcf_v1:launch_for_eval', selected_seed=HIGHD_SPLITD_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_splitd_6fcf_v1', guard_condition='dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0 and not (num_n_tiles <= 16 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448)))', reason='high-D rows use the 6fcf split-D seed except where 8de8 split-K A/B showed wins')
ROUTE_HIGHD_SPLITK = RouteDecision(route_id=HIGHD_SPLITK_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_splitk_8de8_v1:launch_for_eval', selected_seed=HIGHD_SPLITK_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_splitk_8de8_v1', guard_condition='dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 16 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448))', reason='8de8 split-K is selected only for measured low point-tile high-K high-D wins')
ROUTE_HIGHD_PAIRED_PACKEDPARTIAL = RouteDecision(route_id=HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_gridcap160_0194_v1:launch_for_eval', selected_seed=HIGHD_PAIRED_PACKEDPARTIAL_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_paired_packedpartial_gridcap160_0194_v1', guard_condition='dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448, 512]', reason='round-35 same-session audit showed the paired packed-partial seed beats the exported highd_splitk route for the no-padding D448/D512 paired rows')
ROUTE_HIGHD_NO_PADDING_PORTFOLIO_R63 = RouteDecision(route_id=HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1:launch_for_eval', selected_seed=HIGHD_NO_PADDING_PORTFOLIO_R63_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_no_padding_highd_portfolio_r63_g1_d512n512_v1', guard_condition='dtype == bfloat16 and (B, N, K, D) is one of the exact no-padding high-D portfolio rows measured in R63', reason='R63 portfolio is the validated no-padding high-D path; it keeps the R52 portfolio for established rows and uses the G1/R4 streamdep child for D512/N512/K4096')
ROUTE_D768_PRIORITY = RouteDecision(route_id=D768_PRIORITY_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1:launch_for_eval', selected_seed=D768_PRIORITY_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d768_no_padding_priority_exact5_incumbent', guard_condition='dtype == bfloat16 and D == 768 and (B, N, K) in [(1,512,4096), (1,1024,4096), (1,512,8192), (1,1024,8192), (1,2048,4096)]', reason='exact-D768 priority comparator route reuses the validated no-padding G1/R4 Split-K Weave seed; it closes the contract comparator gap without padding')
_ROUTE_FNS: dict[str, Callable[[dict[str, Any]], Any]] = {D64_ROUTE_ID: _d64_direct.launch_for_eval, MICRODIM_HYBRID_ROUTE_ID: _microdim.launch_for_eval, LOWDIM_E50C_ROUTE_ID: _lowdim.launch_for_eval, D480_EAC2_ROUTE_ID: _d480_eac2.launch_for_eval, GAP_PAD_ROUTE_ID: _gap_pad.launch_for_eval, D352_EXACTD_ROUTE_ID: _d352_exactd.launch_for_eval, D288_PARENT_SPLITK_HYBRID_ROUTE_ID: _d288_hybrid.launch_for_eval, SMALL_ROUTE_ID: _single.launch_for_eval, PAIRED_ROUTE_ID: _paired.launch_for_eval, D128_PRIORITY_SPLITK_ROUTE_ID: _d128_priority.launch_for_eval, D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID: _single.launch_for_eval, GENERIC_FALLBACK_ID: _single.launch_for_eval, D144_D160_D176_PAD192_F9B2_ROUTE_ID: _tail_pad192_f9b2.launch_for_eval, D160_PADDED_ROUTE_ID: _non128d_d160.launch_for_eval, D192_SINGLE_ROUTE_ID: _non128d_v1.launch_for_eval, D192_PAIRED_ROUTE_ID: _non128d_v1.launch_for_eval, D256_SINGLE_ROUTE_ID: _non128d_v1.launch_for_eval, HIGHD_SPLITD_ROUTE_ID: _highd.launch_for_eval, HIGHD_SPLITK_ROUTE_ID: _highd_splitk.launch_for_eval, HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID: _highd_paired_gridcap160.launch_for_eval, HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID: _highd_no_padding_r63.launch_for_eval, D768_PRIORITY_ROUTE_ID: _d768_priority.launch_for_eval}
_LAST_ROUTE_TRACE: list[dict[str, Any]] = []

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Contract-harness entrypoint for the explicit guarded portfolio."""
    decision = select_route(inputs)
    return _launch_route(inputs, decision)

def launch_for_eval_forced_fallback(inputs: dict[str, Any]) -> dict[str, Any]:
    """Coverage probe entrypoint that forces the aligned Weave fallback route."""
    decision = _select_fallback_route(inputs)
    return _launch_route(inputs, decision)

def launch_for_eval_forced_d352_exactd(inputs: dict[str, Any]) -> dict[str, Any]:
    """ABI preflight entrypoint that forces the validated D352 exact-D route."""
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if not _use_d352_exactd(D=dim, num_n_tiles=n_points // BLOCK_N, k_tiles=n_clusters // BLOCK_K):
        raise ValueError('forced D352 exact-D route requires D=352, N/64 <= 32, and K/256 >= 4')
    return _launch_route(inputs, ROUTE_D352_EXACTD)

def last_route_trace() -> list[dict[str, Any]]:
    return [dict(row) for row in _LAST_ROUTE_TRACE]

def select_route(inputs: dict[str, Any]) -> RouteDecision:
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    return select_route_from_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)

def select_route_from_shape(*, B: int, N: int, D: int, K: int, dtype: Any='bfloat16') -> RouteDecision:
    _validate_supported_shape(B=B, N=N, D=D, K=K, dtype=dtype)
    num_n_tiles = N // BLOCK_N
    k_tiles = K // BLOCK_K
    if D == _d64_direct.FEAT_D_PAD:
        return ROUTE_D64_DIRECT_9F2A
    if D in _microdim.SUPPORTED_D:
        return ROUTE_MICRODIM_HYBRID
    if D == 288 and _use_d288_parent_splitk_hybrid(K=K):
        return ROUTE_D288_PARENT_SPLITK_HYBRID
    if _use_d352_exactd(D=D, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        return ROUTE_D352_EXACTD
    if _use_d480_eac2(D=D, N=N, k_tiles=k_tiles):
        return ROUTE_D480_EAC2
    if D in _gap_pad.SUPPORTED_D:
        return ROUTE_GAP_PAD
    if D in {80, 96}:
        return ROUTE_LOWDIM_E50C
    if D in D144_D176_F9B2_DIMS:
        return ROUTE_D144_D176_PAD192_F9B2
    if D == 160:
        return ROUTE_D160_PADDED
    if D == 192:
        if num_n_tiles % 2 != 0 or (num_n_tiles <= SMALL_GRID_N_TILE_CAP and k_tiles <= SMALL_GRID_K_TILE_CAP):
            return ROUTE_D192_SINGLE
        return ROUTE_D192_PAIRED
    if D == 256:
        return ROUTE_D256_SINGLE
    if D == _d768_priority.TARGET_D:
        return ROUTE_D768_PRIORITY
    if _use_d128_priority_splitk(B=B, N=N, D=D, K=K):
        return ROUTE_D128_PRIORITY_SPLITK
    if _use_highd_no_padding_portfolio_r63(B=B, N=N, D=D, K=K):
        return ROUTE_HIGHD_NO_PADDING_PORTFOLIO_R63
    if _use_highd_paired_packedpartial(B=B, N=N, D=D, K=K):
        return ROUTE_HIGHD_PAIRED_PACKEDPARTIAL
    if D in _highd.SUPPORTED_DIMS:
        if _use_highd_splitk(D=D, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
            return ROUTE_HIGHD_SPLITK
        return ROUTE_HIGHD_SPLITD
    if B == 8 and N == 8192 and (K == 256):
        return ROUTE_D128_EVEN_NEAR_FLOOR_V10_REPAIR
    if num_n_tiles <= SMALL_GRID_N_TILE_CAP and k_tiles <= SMALL_GRID_K_TILE_CAP:
        return ROUTE_SMALL_V10
    if num_n_tiles % 2 == 0:
        return ROUTE_PAIRED_V15
    return ROUTE_ALIGNED_V10_FALLBACK

def route_trace_for_shape(label: str, params: dict[str, Any], *, expected_seed: str | None=None, dispatcher_kernel_ms: float | None=None, shape_specific_kernel_ms: float | None=None, relative_speedup_vs_baseline: float | None=None) -> dict[str, Any]:
    try:
        decision = select_route(params)
    except ValueError as exc:
        return _unsupported_trace_row(shape_key=label, expected_seed=expected_seed, dispatcher_kernel_ms=dispatcher_kernel_ms, shape_specific_kernel_ms=shape_specific_kernel_ms, relative_speedup_vs_baseline=relative_speedup_vs_baseline, reason=str(exc))
    return decision.trace_row(shape_key=label, expected_seed=expected_seed, dispatcher_kernel_ms=dispatcher_kernel_ms, shape_specific_kernel_ms=shape_specific_kernel_ms, relative_speedup_vs_baseline=relative_speedup_vs_baseline)

def route_trace_for_shapes(shapes: list[dict[str, Any]], *, expected_seeds: dict[str, str] | None=None, per_shape_metrics: dict[str, dict[str, Any]] | None=None, baseline_metrics: dict[str, dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        label = str(shape['label'])
        params = dict(shape.get('params', {}))
        metrics = (per_shape_metrics or {}).get(label, {})
        baseline = (baseline_metrics or {}).get(label, {})
        dispatcher_ms = _optional_float(metrics.get('kernel_ms'))
        baseline_ms = _optional_float(baseline.get('kernel_ms'))
        speedup = None
        if dispatcher_ms is not None and baseline_ms is not None and (dispatcher_ms > 0.0):
            speedup = baseline_ms / dispatcher_ms
        rows.append(route_trace_for_shape(label, params, expected_seed=(expected_seeds or {}).get(label), dispatcher_kernel_ms=dispatcher_ms, shape_specific_kernel_ms=baseline_ms, relative_speedup_vs_baseline=speedup))
    return rows

def compile_and_launch_flash_kmeans_assign_dispatcher(B: int=2, N: int=1024, K: int=512, D: int=128, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(5101)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': f'manual_b{B}_n{N}_k{K}_d{D}', 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref)), 'route_trace': last_route_trace()}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': f'manual_b{B}_n{N}_k{K}_d{D}', 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 5101}}])
    return result

def _launch_route(inputs: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    trace = decision.trace_row(shape_key=_shape_key(inputs))
    inputs['_flash_kmeans_assign_dispatch_route'] = trace
    _LAST_ROUTE_TRACE[:] = [trace]
    outputs = _ROUTE_FNS[decision.route_id](inputs)
    normalized = _normalize_outputs(outputs, inputs)
    _LAST_ROUTE_TRACE[:] = [trace]
    normalized['selected_route'] = decision.route_id
    normalized['route_trace'] = trace
    return normalized

def _select_fallback_route(inputs: dict[str, Any]) -> RouteDecision:
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if dim != FEAT_D:
        raise ValueError(f'forced fallback is only defined for D={FEAT_D}, got {dim}')
    return ROUTE_ALIGNED_V10_FALLBACK

def _normalize_outputs(outputs: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    if outputs is None:
        return {'cluster_ids': inputs['out']}
    if hasattr(outputs, 'shape'):
        return {'cluster_ids': outputs}
    if isinstance(outputs, dict):
        normalized = dict(outputs)
        if 'cluster_ids' not in normalized and 'out' in normalized:
            normalized['cluster_ids'] = normalized['out']
        if 'cluster_ids' in normalized:
            return normalized
    raise TypeError("flash_kmeans_assign dispatcher route must return cluster_ids or write inputs['out']")

def _shape_fields(inputs: dict[str, Any]) -> tuple[int, int, int, int, str]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = _dtype_name(inputs)
    return (bsz, n_points, dim, n_clusters, dtype)

def _validate_supported_shape(*, B: int, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(f'flash_kmeans_assign_dispatcher requires bfloat16 input, got {dtype}')
    if D not in SUPPORTED_DIMS:
        raise ValueError(f'flash_kmeans_assign_dispatcher has no Weave route for D={D}; supported D values are {SUPPORTED_DIMS}')
    if D == _d768_priority.TARGET_D and (not _is_d768_priority_shape(B=B, N=N, K=K)):
        raise ValueError(f'flash_kmeans_assign_dispatcher has a D=768 Weave route only for the validated priority shapes {sorted(D768_PRIORITY_SHAPES)}; got N={N}, K={K}')
    if N % BLOCK_N != 0:
        raise ValueError(f'N must be divisible by BLOCK_N={BLOCK_N}, got {N}')
    if K % BLOCK_K != 0:
        raise ValueError(f'K must be divisible by BLOCK_K={BLOCK_K}, got {K}')

def _use_highd_splitk(*, D: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= 32 or (k_tiles >= _highd_splitk.SPLITK_MIN_K_TILES and D >= 448)

def _use_d128_priority_splitk(*, B: int, N: int, D: int, K: int) -> bool:
    return B == 1 and D == _d128_priority.TARGET_D and (N in (512, 1024, 2048)) and (K in (4096, 8192))

def _is_d768_priority_shape(*, B: int, N: int, K: int) -> bool:
    return (B, N, K) in D768_PRIORITY_SHAPES

def _use_d288_parent_splitk_hybrid(*, K: int) -> bool:
    """Keep unsupported middle-K values on the existing gap-pad route."""
    return K <= 2048 or K >= 4096

def _use_d352_exactd(*, D: int, num_n_tiles: int, k_tiles: int) -> bool:
    return D == _d352_exactd.FEAT_D and num_n_tiles <= _d352_exactd.MAX_POINT_TILES and (k_tiles >= 4)

def _use_d480_eac2(*, D: int, N: int, k_tiles: int) -> bool:
    """Apply the seed guard in eac2's 64-row point-tile units."""
    return _d480_eac2._use_k1024_splitk(dim=D, num_n_tiles=N // _d480_eac2.BLOCK_N, k_tiles=k_tiles)

def _use_highd_paired_packedpartial(*, B: int, N: int, D: int, K: int) -> bool:
    return B == 1 and N == 2048 and (K == 4096) and (D in _highd_paired_gridcap160.PAIRED_PACKEDPARTIAL_DIMS)

def _use_highd_no_padding_portfolio_r63(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, K, D) in _highd_no_padding_r63.NO_PADDING_HIGHD_SHAPES

def _classify_route(route: RouteDecision, expected_seed: str | None, relative_speedup_vs_baseline: float | None) -> str:
    if expected_seed is not None:
        if route.selected_seed != expected_seed:
            return 'guard-miss'
        if relative_speedup_vs_baseline is not None and relative_speedup_vs_baseline < SLOW_ROUTE_SPEEDUP_THRESHOLD:
            return 'fallback-slow' if route.route_kind == 'fallback' else 'kernel-slow'
        return 'seed-consumed'
    if route.route_kind == 'fallback':
        return 'coverage-only'
    return 'route-ok'

def _unsupported_trace_row(*, shape_key: str, expected_seed: str | None, dispatcher_kernel_ms: float | None, shape_specific_kernel_ms: float | None, relative_speedup_vs_baseline: float | None, reason: str) -> dict[str, Any]:
    return {'shape_key': shape_key, 'selected_route': UNSUPPORTED_ROUTE_ID, 'selected_entrypoint': 'none', 'selected_seed': None, 'expected_seed': expected_seed, 'route_kind': 'none', 'route_source': 'unknown', 'guard_id': 'guard_miss_no_supported_weave_route', 'guard_condition': 'no production Weave guard matched this shape', 'classification': 'guard-miss' if expected_seed is not None else 'unmeasured', 'dispatcher_kernel_ms': dispatcher_kernel_ms, 'shape_specific_kernel_ms': shape_specific_kernel_ms, 'relative_speedup_vs_baseline': relative_speedup_vs_baseline, 'reason': reason}

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return f'b{int(inputs['B'])}_n{int(inputs['N'])}_k{int(inputs['K'])}_d{int(inputs['D'])}'

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = inputs.get('dtype')
    if dtype is not None:
        return str(dtype).replace('torch.', '')
    x = inputs.get('x')
    if x is not None and hasattr(x, 'dtype'):
        return str(x.dtype).replace('torch.', '')
    return 'bfloat16'

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
