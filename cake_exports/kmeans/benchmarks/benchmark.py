from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from itertools import permutations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flash_kmeans_shapes import (  # noqa: E402
    FLASH_KMEANS_EVOLUTION_ARTIFACT,
    FLASH_KMEANS_EVOLUTION_SUMMARY,
)
from flash_kmeans_triton_h200 import euclid_assign_triton_h200  # noqa: E402
from flash_kmeans_triton_h200_raw_adapter import (  # noqa: E402
    BASELINE_COMMIT,
    PREPROCESS_IMPL,
    TritonH20007cfRawAdapter,
)
from flash_kmeans_triton_h200_raw_adapter import (  # noqa: E402
    BASELINE_NAME as PUBLIC_RAW_BASELINE_NAME,
)
from flashlib_cake_kmeans._benchmark import (  # noqa: E402
    bench_gpu_time,
    measure_host_call,
    require_cupti,
)

SHAPE_RECORDS = json.loads((Path(__file__).with_name("shape_records.json")).read_text(encoding="utf-8"))
ROUTE_MANIFEST = json.loads((Path(__file__).with_name("expected_routes.json")).read_text(encoding="utf-8"))
ALL_ROUTE_MANIFEST = json.loads((Path(__file__).with_name("all_expected_routes.json")).read_text(encoding="utf-8"))
EXPECTED_ROUTES = {row["shape"]: row["selected_route"] for row in ALL_ROUTE_MANIFEST}


def _shape_from_record(record: dict[str, Any]) -> dict[str, Any]:
    params = dict(record["params"])
    recorded = dict(record.get("recorded", {}))
    row = {"label": record["label"], **params}
    row["runtime_coverage"] = bool(record.get("runtime_coverage", False))
    row["route"] = recorded.get("evolution_route", params.get("route"))
    for key in (
        "evolution_kernel_ms",
        "evolution_flashlib_ms",
        "evolution_tflops",
        "evolution_flashlib_equiv_tflops",
        "evolution_speedup",
    ):
        row[key] = recorded.get(key, params.get(key))
    return row


FLASH_KMEANS_SHAPES = [_shape_from_record(record) for record in SHAPE_RECORDS]
_SHAPES_BY_LABEL = {row["label"]: row for row in FLASH_KMEANS_SHAPES}
FLASH_KMEANS_REGISTRY_SHAPES = [_SHAPES_BY_LABEL[route["shape"]] for route in ROUTE_MANIFEST]
SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"
PRECOMPUTED_BASELINE_NAME = "triton_h200_07cf_precomputed"
BASELINE_REGISTRY_KEY = "triton_h200_07cf_dual_lane_v1"
# Updated together with benchmark_data.json by the registry sync gate.
BASELINE_REGISTRY_SHA256 = "5e04170da6729f8155b6be98e8c6f209abe9c2959e8c4141ed2e4e4a3b3351ce"
REGISTRY_CANDIDATE_ENTRYPOINT = SEMANTIC_ENTRYPOINT
MEASURED_CANDIDATE_ENTRYPOINT = "flashlib_cake_kmeans.interface:FlashKMeansAssignRuntime.compute"
BASELINE_ENTRYPOINT = "benchmarks.flash_kmeans_triton_h200_raw_adapter:TritonH20007cfRawAdapter.compute"
PARITY_CANDIDATE_ENTRYPOINT = "flashlib_cake_kmeans.interface:flash_kmeans_assign_prepared"
PARITY_BASELINE_ENTRYPOINT = "benchmarks.flash_kmeans_triton_h200:euclid_assign_triton_h200"
CANDIDATE_TIMING_BOUNDARY = "raw_inputs_default_output_internal_route_required_fused_norms_synchronized_e2e"
BASELINE_TIMING_BOUNDARY = "raw_inputs_default_output_shared_fused_pair_norm_plus_pinned_07cf_assign_synchronized_e2e"
PARITY_CANDIDATE_TIMING_BOUNDARY = "precomputed_norms_preallocated_output_prepared_assignment_gpu_span"
PARITY_BASELINE_TIMING_BOUNDARY = "precomputed_norms_preallocated_output_pinned_07cf_assignment_gpu_span"
BASELINE_REGISTRY_PROFILE = {
    "registry_candidate_entrypoint": REGISTRY_CANDIDATE_ENTRYPOINT,
    "shared_preprocess": {
        "implementation_entrypoint": PREPROCESS_IMPL,
        "source_sha256": "de4410a1b997eaa64847f05e4af5e37f44b001409717c9923b1610d21f2c4104",
        "result_source_sha256_field": "preprocess_source_sha256",
    },
    "official": {
        "role": "publication",
        "candidate_entrypoint": MEASURED_CANDIDATE_ENTRYPOINT,
        "baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": BASELINE_ENTRYPOINT,
        "candidate_timing_boundary": CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": BASELINE_TIMING_BOUNDARY,
        "candidate_timing_backend_field": "candidate_public_raw_timing_backend",
        "baseline_timing_backend_field": "baseline_07cf_adapter_timing_backend",
        "speedup_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "speedup_numerator_metric": "baseline_07cf_adapter_synchronized_e2e_ms",
        "speedup_denominator_metric": "candidate_public_raw_synchronized_e2e_ms",
        "timing_backend": "cupti",
    },
    "parity": {
        "role": "diagnostic_only",
        "candidate_entrypoint": PARITY_CANDIDATE_ENTRYPOINT,
        "baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": PARITY_BASELINE_ENTRYPOINT,
        "candidate_timing_boundary": PARITY_CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": PARITY_BASELINE_TIMING_BOUNDARY,
        "candidate_timing_backend_field": "candidate_precomputed_timing_backend",
        "baseline_timing_backend_field": "baseline_07cf_precomputed_timing_backend",
        "speedup_metric": "precomputed_gpu_speedup_vs_07cf",
        "speedup_numerator_metric": "baseline_07cf_precomputed_gpu_span_ms",
        "speedup_denominator_metric": "candidate_precomputed_gpu_span_ms",
        "timing_backend": "cupti",
    },
}
WARMUP_MS = 20.0
BENCH_MS = 100.0
MEASUREMENT_ORDER_SEED = "flashlib-kmeans-export-dual-lane-v1"


def _preprocess_source_digest() -> str:
    """Hash the exact Python/CUDA pair-row-norm implementation in this export."""

    source_dir = Path(__file__).resolve().parent
    candidates = (
        (
            ROOT / "src/flashlib_cake_kmeans/_row_norm.py",
            ROOT / "src/flashlib_cake_kmeans/_row_norm.cu",
        ),
        (source_dir / "row_norm.py", source_dir / "row_norm.cu"),
    )
    for paths in candidates:
        if all(path.is_file() for path in paths):
            digest = hashlib.sha256()
            for canonical_name, path in zip(("_row_norm.py", "_row_norm.cu"), paths, strict=True):
                digest.update(canonical_name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            return digest.hexdigest()
    searched = ", ".join(str(path) for pair in candidates for path in pair)
    raise RuntimeError(f"cannot locate exported pair-row-norm sources; searched {searched}")


def _timing_diagnostics(timing: Any, *, primary_metric: str) -> dict[str, Any]:
    return {
        "lane_primary_metric": primary_metric,
        # KMeans uses an adaptive 100 ms window that can contain thousands of
        # iterations. The generated BenchResult retains every exact sample;
        # the publication JSON records medians and sample count to avoid
        # multiplying artifact size across the 228-shape portfolio.
        "sample_count": len(timing.times_ms),
        "gpu_span_ms": {"median": timing.median_gpu_span_ms},
        "kernel_sum_ms": {"median": timing.median_kernel_sum_ms},
        "active_union_ms": {"median": timing.median_active_union_ms},
        "inter_kernel_gap_ms": {"median": timing.median_inter_kernel_gap_ms},
        "activity_count": {"median": timing.median_activity_count},
        "correlated_launch_activity_count": {"median": timing.median_launch_activity_count},
        "correlated_kernel_activity_count": {"median": timing.median_kernel_activity_count},
        "host_enqueue_ms": {"median": timing.median_host_enqueue_ms},
        "synchronized_e2e_ms": {"median": timing.median_synchronized_e2e_ms},
        "cold_first_call": {
            "host_enqueue_ms": timing.cold_first_call_host_enqueue_ms,
            "synchronized_e2e_ms": timing.cold_first_call_synchronized_e2e_ms,
        },
    }


def _bench_original_07cf_window(fn, *, cold_first_call=None):
    """Match the 07cf adaptive 20 ms warmup / 100 ms CUPTI window."""
    probe = bench_gpu_time(fn, warmup_iters=5, bench_iters=20, cold_l2=True)
    estimate = probe.median_ms
    return bench_gpu_time(
        fn,
        warmup_iters=max(1, math.ceil(WARMUP_MS / estimate)),
        bench_iters=max(1, math.ceil(BENCH_MS / estimate)),
        cold_l2=True,
        cold_first_call=cold_first_call,
    )


def _label_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _measurement_order(label: str) -> tuple[str, str, str, str]:
    """Choose one stable order for separate CUPTI timing blocks."""

    orders = tuple(
        permutations(
            (
                "candidate_public_raw",
                "baseline_public_raw",
                "candidate_precomputed",
                "baseline_precomputed",
            )
        )
    )
    digest = hashlib.sha256(f"{MEASUREMENT_ORDER_SEED}:{label}".encode()).digest()
    return orders[int.from_bytes(digest[:2], "little") % len(orders)]


def _alternating_call(first, second):
    """Return a callable that alternates two equivalent fresh-pointer cases."""

    calls = (first, second)
    next_index = 0

    def invoke():
        nonlocal next_index
        current = next_index
        next_index ^= 1
        return calls[current]()

    return invoke


def _required_median(value: float | None, *, metric: str) -> float:
    if value is None:
        raise RuntimeError(f"CUPTI benchmark did not report required {metric}")
    return float(value)


def _dual_lane_speedups(
    *,
    candidate_public_raw_timing: Any,
    baseline_public_raw_timing: Any,
    candidate_precomputed_timing: Any,
    baseline_precomputed_timing: Any,
) -> dict[str, float]:
    """Return the one official E2E ratio and the precomputed GPU parity ratio."""

    candidate_public_raw_e2e = _required_median(
        candidate_public_raw_timing.median_synchronized_e2e_ms,
        metric="candidate public-raw synchronized_e2e_ms",
    )
    baseline_public_raw_e2e = _required_median(
        baseline_public_raw_timing.median_synchronized_e2e_ms,
        metric="07cf raw-adapter synchronized_e2e_ms",
    )
    return {
        "public_raw_e2e_speedup_vs_07cf_adapter": baseline_public_raw_e2e / candidate_public_raw_e2e,
        "precomputed_gpu_speedup_vs_07cf": (
            float(baseline_precomputed_timing.median_gpu_span_ms)
            / float(candidate_precomputed_timing.median_gpu_span_ms)
        ),
    }


def _cold_call_fields(timing: Any | None) -> dict[str, float | None]:
    return {
        "host_enqueue_ms": timing.host_enqueue_ms if timing is not None else None,
        "synchronized_e2e_ms": timing.synchronized_e2e_ms if timing is not None else None,
    }


def _shape_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["B"]),
        int(row["N"]),
        int(row["D"]),
        int(row["K"]),
        str(row.get("dtype", "bfloat16")),
    )


def _selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(FLASH_KMEANS_SHAPES if args.include_runtime_coverage else FLASH_KMEANS_REGISTRY_SHAPES)
    if args.source:
        wanted_sources = set(args.source)
        rows = [row for row in rows if row["source"] in wanted_sources]
    if args.shape:
        by_label = {row["label"]: row for row in rows}
        missing = sorted(set(args.shape) - set(by_label))
        if missing:
            available = ", ".join(sorted(by_label))
            raise SystemExit(f"unknown shape label(s) {missing}. Available: {available}")
        rows = [by_label[label] for label in args.shape]
    if args.unique:
        seen: set[tuple[int, int, int, int, str]] = set()
        unique_rows: list[dict[str, Any]] = []
        for row in rows:
            key = _shape_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        rows = unique_rows
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.shard_count]
    return rows


def _write_json_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_inputs(row: dict[str, Any], *, variant: int = 0):
    import torch

    generator = torch.Generator(device="cuda")
    seed = int(row.get("seed", _label_seed(str(row["label"]))))
    if variant != 0:
        seed = _label_seed(f"{row['label']}:fresh-pointer:{variant}:{seed}")
    generator.manual_seed(seed)
    x = torch.randn(
        (int(row["B"]), int(row["N"]), int(row["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    centroids = torch.randn(
        (int(row["B"]), int(row["K"]), int(row["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    return x, centroids


def _measure_shared_preprocess_cold_compile(row: dict[str, Any], *, arch: str):
    """Compile the shared row-norm support once, outside either comparison lane."""

    import torch
    from flashlib_cake_kmeans._row_norm import prepare_bf16_pair_row_norm

    x, centroids = _make_inputs(row, variant=0xC01D)
    x_sq = torch.empty((int(row["B"]), int(row["N"])), dtype=torch.float32, device=x.device)
    c_sq = torch.empty((int(row["B"]), int(row["K"])), dtype=torch.float32, device=x.device)
    # Fixture generation is asynchronous. Complete it before the cold bracket
    # so neither lane inherits unrelated random-generation/allocation work.
    torch.cuda.synchronize()

    def prepare_shared_support():
        prepared = prepare_bf16_pair_row_norm(
            x,
            centroids,
            x_sq,
            c_sq,
            compute_x=True,
            compute_c=True,
            arch=arch,
        )
        prepared.release_bound_callers(x_sq)

    _, cold_compile = measure_host_call(prepare_shared_support)
    return cold_compile


def _reference_assign(x, centroids, *, chunk_rows: int):
    import torch

    bsz = int(x.shape[0])
    n_points = int(x.shape[1])
    x_f32 = x.float()
    c_f32 = centroids.float()
    c_sq = (c_f32 * c_f32).sum(-1)
    ref = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
    with torch.no_grad():
        for b in range(bsz):
            c_t = c_f32[b].transpose(0, 1).contiguous()
            c_bias = 0.5 * c_sq[b]
            for start in range(0, n_points, chunk_rows):
                q = x_f32[b, start : start + chunk_rows]
                scores = torch.matmul(q, c_t) - c_bias.unsqueeze(0)
                ref[b, start : start + q.shape[0]] = scores.argmax(dim=-1).to(torch.int32)
    return ref


def _assignment_correctness(cluster_ids, ref, x, centroids) -> dict[str, Any]:
    """Validate assignment indices while accepting exact-distance ties."""

    import torch

    matches = cluster_ids == ref
    mismatch_count = int((~matches).sum().item())
    diagnostics: dict[str, Any] = {
        "match_rate": float(matches.float().mean().item()),
        "mismatch_count": mismatch_count,
    }
    if mismatch_count == 0:
        diagnostics.update({"correct": True, "tie_inclusive": False})
        return diagnostics

    dim = int(x.shape[-1])
    pred_idx = cluster_ids.to(torch.int64).unsqueeze(-1).expand(-1, -1, dim)
    ref_idx = ref.to(torch.int64).unsqueeze(-1).expand(-1, -1, dim)
    pred_centroids = torch.gather(centroids, 1, pred_idx).float()
    ref_centroids = torch.gather(centroids, 1, ref_idx).float()
    points = x.float()
    pred_dist = ((points - pred_centroids) ** 2).sum(-1)
    ref_dist = ((points - ref_centroids) ** 2).sum(-1)
    distance_delta = (pred_dist - ref_dist).abs()
    tie_ok = matches | (distance_delta <= 1.0e-3)
    diagnostics.update(
        {
            "correct": bool(tie_ok.all().item()),
            "tie_inclusive": True,
            "tie_inclusive_match_rate": float(tie_ok.float().mean().item()),
            "max_selected_distance_delta": float(distance_delta.max().item()),
        }
    )
    return diagnostics


def _run_shape(
    row: dict[str, Any],
    *,
    runtime: Any,
    baseline_adapter: Any,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    reference_chunk_rows: int,
    measurement_session_id: str,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_kmeans import (
        flash_kmeans_assign_prepared,
        prepare_flash_kmeans_assign,
    )

    x, centroids = _make_inputs(row, variant=0)
    fresh_x, fresh_centroids = _make_inputs(row, variant=1)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    fresh_x_sq = (fresh_x.float() ** 2).sum(-1).contiguous()
    fresh_c_sq = (fresh_centroids.float() ** 2).sum(-1).contiguous()
    out_shape = (int(row["B"]), int(row["N"]))
    candidate_precomputed_out = torch.empty(out_shape, dtype=torch.int32, device=x.device)
    fresh_candidate_precomputed_out = torch.empty_like(candidate_precomputed_out)
    baseline_precomputed_out = torch.empty_like(candidate_precomputed_out)
    fresh_baseline_precomputed_out = torch.empty_like(candidate_precomputed_out)

    pointer_pairs = (
        (x, fresh_x),
        (centroids, fresh_centroids),
        (x_sq, fresh_x_sq),
        (c_sq, fresh_c_sq),
    )
    if any(first.data_ptr() == second.data_ptr() for first, second in pointer_pairs):
        raise RuntimeError("fresh-pointer benchmark inputs unexpectedly alias the first input set")
    # Keep cold diagnostics honest: random input generation, explicit parity
    # norms, and preallocated parity outputs all precede every measured call.
    torch.cuda.synchronize()

    def baseline_public_raw_first_call():
        return baseline_adapter.compute(x, centroids, return_info=True)

    def baseline_public_raw_fresh_pointer_call():
        return baseline_adapter.compute(fresh_x, fresh_centroids, return_info=True)

    def baseline_precomputed_first_call():
        return euclid_assign_triton_h200(
            x,
            centroids,
            x_sq,
            c_sq,
            out=baseline_precomputed_out,
        )

    def baseline_precomputed_fresh_pointer_call():
        return euclid_assign_triton_h200(
            fresh_x,
            fresh_centroids,
            fresh_x_sq,
            fresh_c_sq,
            out=fresh_baseline_precomputed_out,
        )

    def candidate_public_raw_shape_first_call():
        return runtime.compute(
            x,
            centroids,
            return_info=True,
        )

    def candidate_public_raw_fresh_pointer_call():
        return runtime.compute(
            fresh_x,
            fresh_centroids,
            return_info=True,
        )

    baseline_public_raw_shape_cold_call = None
    baseline_public_raw_fresh_cold_call = None
    baseline_precomputed_first_cold_call = None
    baseline_precomputed_fresh_cold_call = None
    candidate_public_raw_shape_cold_call = None
    candidate_public_raw_fresh_cold_call = None
    candidate_precomputed_prepare_cold_call = None
    fresh_candidate_precomputed_prepare_cold_call = None
    candidate_precomputed_first_cold_call = None
    fresh_candidate_precomputed_first_cold_call = None
    if benchmark:
        (
            (
                baseline_public_raw_out,
                baseline_public_raw_shape_info,
            ),
            baseline_public_raw_shape_cold_call,
        ) = measure_host_call(baseline_public_raw_first_call)
        (
            (
                fresh_baseline_public_raw_out,
                baseline_public_raw_fresh_info,
            ),
            baseline_public_raw_fresh_cold_call,
        ) = measure_host_call(baseline_public_raw_fresh_pointer_call)
        (
            (
                candidate_public_raw_out,
                candidate_public_raw_shape_info,
            ),
            candidate_public_raw_shape_cold_call,
        ) = measure_host_call(candidate_public_raw_shape_first_call)
        (
            (
                fresh_candidate_public_raw_out,
                candidate_public_raw_info,
            ),
            candidate_public_raw_fresh_cold_call,
        ) = measure_host_call(candidate_public_raw_fresh_pointer_call)
        candidate_prepared, candidate_precomputed_prepare_cold_call = measure_host_call(
            lambda: prepare_flash_kmeans_assign(
                x,
                centroids,
                out=candidate_precomputed_out,
                x_sq=x_sq,
                c_sq=c_sq,
                arch=arch,
            )
        )
        fresh_candidate_prepared, fresh_candidate_precomputed_prepare_cold_call = measure_host_call(
            lambda: prepare_flash_kmeans_assign(
                fresh_x,
                fresh_centroids,
                out=fresh_candidate_precomputed_out,
                x_sq=fresh_x_sq,
                c_sq=fresh_c_sq,
                arch=arch,
            )
        )
        (
            (
                candidate_precomputed_result,
                candidate_precomputed_info,
            ),
            candidate_precomputed_first_cold_call,
        ) = measure_host_call(lambda: flash_kmeans_assign_prepared(candidate_prepared, return_info=True))
        (
            (
                fresh_candidate_precomputed_result,
                fresh_candidate_precomputed_info,
            ),
            fresh_candidate_precomputed_first_cold_call,
        ) = measure_host_call(lambda: flash_kmeans_assign_prepared(fresh_candidate_prepared, return_info=True))
        (
            (
                baseline_precomputed_result,
                baseline_precomputed_config,
            ),
            baseline_precomputed_first_cold_call,
        ) = measure_host_call(baseline_precomputed_first_call)
        (
            (
                fresh_baseline_precomputed_result,
                fresh_baseline_precomputed_config,
            ),
            baseline_precomputed_fresh_cold_call,
        ) = measure_host_call(baseline_precomputed_fresh_pointer_call)
    else:
        (
            baseline_public_raw_out,
            baseline_public_raw_shape_info,
        ) = baseline_public_raw_first_call()
        (
            fresh_baseline_public_raw_out,
            baseline_public_raw_fresh_info,
        ) = baseline_public_raw_fresh_pointer_call()
        candidate_public_raw_out, candidate_public_raw_shape_info = candidate_public_raw_shape_first_call()
        fresh_candidate_public_raw_out, candidate_public_raw_info = candidate_public_raw_fresh_pointer_call()
        candidate_prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=candidate_precomputed_out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
        )
        fresh_candidate_prepared = prepare_flash_kmeans_assign(
            fresh_x,
            fresh_centroids,
            out=fresh_candidate_precomputed_out,
            x_sq=fresh_x_sq,
            c_sq=fresh_c_sq,
            arch=arch,
        )
        candidate_precomputed_result, candidate_precomputed_info = flash_kmeans_assign_prepared(
            candidate_prepared,
            return_info=True,
        )
        fresh_candidate_precomputed_result, fresh_candidate_precomputed_info = flash_kmeans_assign_prepared(
            fresh_candidate_prepared,
            return_info=True,
        )
        baseline_precomputed_result, baseline_precomputed_config = baseline_precomputed_first_call()
        fresh_baseline_precomputed_result, fresh_baseline_precomputed_config = baseline_precomputed_fresh_pointer_call()
        torch.cuda.synchronize()

    baseline_configs = (
        baseline_public_raw_shape_info["triton_h200_07cf_config"],
        baseline_public_raw_fresh_info["triton_h200_07cf_config"],
        baseline_precomputed_config,
        fresh_baseline_precomputed_config,
    )
    if any(config != baseline_configs[0] for config in baseline_configs[1:]):
        raise RuntimeError("07cf Triton selected different configs across equivalent dual-lane calls")
    if candidate_public_raw_info["selected_route"] != candidate_public_raw_shape_info["selected_route"]:
        raise RuntimeError(
            "runtime shape-first and fresh-pointer calls selected different routes: "
            f"{candidate_public_raw_shape_info['selected_route']!r} != "
            f"{candidate_public_raw_info['selected_route']!r}"
        )
    if not candidate_public_raw_info["runtime_cache_hit"]:
        raise RuntimeError("fresh-pointer runtime.compute call did not reuse the shape/stream launch plan")
    if not baseline_public_raw_fresh_info["runtime_cache_hit"]:
        raise RuntimeError("fresh-pointer 07cf raw-adapter call did not reuse the shape/stream norm plan")
    candidate_shape_norm_fields = tuple(candidate_public_raw_shape_info.get("norm_compute_fields", ()))
    candidate_fresh_norm_fields = tuple(candidate_public_raw_info.get("norm_compute_fields", ()))
    if candidate_shape_norm_fields != candidate_fresh_norm_fields:
        raise RuntimeError(
            "runtime shape-first and fresh-pointer calls computed different norm fields: "
            f"{candidate_shape_norm_fields!r} != {candidate_fresh_norm_fields!r}"
        )
    for prepared_info in (candidate_precomputed_info, fresh_candidate_precomputed_info):
        if prepared_info["selected_route"] != candidate_public_raw_info["selected_route"]:
            raise RuntimeError(
                "public-raw and precomputed Flash-KMeans calls selected different routes: "
                f"{candidate_public_raw_info['selected_route']!r} != {prepared_info['selected_route']!r}"
            )

    candidate_shape_first_was_cache_hit = bool(candidate_public_raw_shape_info["runtime_cache_hit"])
    baseline_shape_first_was_cache_hit = bool(baseline_public_raw_shape_info["runtime_cache_hit"])

    result: dict[str, Any] = {
        "shape": row["label"],
        "label": row["label"],
        "source": row["source"],
        "seed": row.get("seed"),
        "runtime_coverage": bool(row.get("runtime_coverage", False)),
        "B": int(row["B"]),
        "N": int(row["N"]),
        "D": int(row["D"]),
        "K": int(row["K"]),
        "dtype": row.get("dtype", "bfloat16"),
        "semantic_entrypoint": candidate_public_raw_info["semantic_entrypoint"],
        "expected_route": EXPECTED_ROUTES[row["label"]],
        "selected_route": candidate_public_raw_info["selected_route"],
        "child_route": candidate_public_raw_info.get("child_route"),
        "route_matches_expected": bool(candidate_public_raw_info["selected_route"] == EXPECTED_ROUTES[row["label"]]),
        "evolution_kernel_ms": row["evolution_kernel_ms"],
        "evolution_flashlib_ms": row["evolution_flashlib_ms"],
        "evolution_tflops": row["evolution_tflops"],
        "evolution_speedup": row["evolution_speedup"],
        "public_raw_baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "precomputed_baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "triton_h200_07cf_config": baseline_configs[0],
        "measurement_session_id": measurement_session_id,
        "candidate_public_raw_assignment_launch_count": candidate_public_raw_info["assignment_launch_count"],
        "candidate_public_raw_norm_launch_count": candidate_public_raw_info["norm_launch_count"],
        "candidate_public_raw_norm_compute_fields": list(candidate_fresh_norm_fields),
        "candidate_public_raw_total_launch_count": candidate_public_raw_info["runtime_launch_count"],
        "baseline_public_raw_assignment_launch_count": baseline_public_raw_fresh_info["assignment_launch_count"],
        "baseline_public_raw_norm_launch_count": baseline_public_raw_fresh_info["norm_launch_count"],
        "baseline_public_raw_norm_compute_fields": list(baseline_public_raw_fresh_info["norm_compute_fields"]),
        "baseline_public_raw_total_launch_count": baseline_public_raw_fresh_info["runtime_launch_count"],
        "candidate_precomputed_launch_count": candidate_precomputed_info["prepared_launch_count"],
        "candidate_public_raw_norm_policy": "route_required_internal_fused_bf16_pair_row_norm",
        "baseline_public_raw_norm_policy": "shared_kernel_all_fields_required_by_frozen_07cf",
        "candidate_precomputed_norm_policy": "explicit_precomputed_outside_timing",
        "baseline_precomputed_norm_policy": "explicit_precomputed_outside_timing",
        "candidate_public_raw_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
        "baseline_public_raw_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
        "candidate_precomputed_output_policy": "preallocated_outside_timing",
        "baseline_precomputed_output_policy": "preallocated_outside_timing",
        "candidate_public_raw_shape_first_was_cache_hit": candidate_shape_first_was_cache_hit,
        "baseline_public_raw_shape_first_was_cache_hit": baseline_shape_first_was_cache_hit,
        "fresh_pointer_rebind_verified": True,
        "cold_candidate_public_raw_shape_miss": (
            None if candidate_shape_first_was_cache_hit else _cold_call_fields(candidate_public_raw_shape_cold_call)
        ),
        "cold_candidate_public_raw_existing_shape_hit": (
            _cold_call_fields(candidate_public_raw_shape_cold_call) if candidate_shape_first_was_cache_hit else None
        ),
        "cold_candidate_public_raw_fresh_pointer_hit": _cold_call_fields(candidate_public_raw_fresh_cold_call),
        "cold_baseline_public_raw_shape_miss": (
            None if baseline_shape_first_was_cache_hit else _cold_call_fields(baseline_public_raw_shape_cold_call)
        ),
        "cold_baseline_public_raw_existing_shape_hit": (
            _cold_call_fields(baseline_public_raw_shape_cold_call) if baseline_shape_first_was_cache_hit else None
        ),
        "cold_baseline_public_raw_fresh_pointer_hit": _cold_call_fields(baseline_public_raw_fresh_cold_call),
        "cold_candidate_precomputed_prepare_first_pointer": _cold_call_fields(candidate_precomputed_prepare_cold_call),
        "cold_candidate_precomputed_prepare_fresh_pointer": _cold_call_fields(
            fresh_candidate_precomputed_prepare_cold_call
        ),
        "cold_candidate_precomputed_first_pointer": _cold_call_fields(candidate_precomputed_first_cold_call),
        "cold_candidate_precomputed_fresh_pointer": _cold_call_fields(fresh_candidate_precomputed_first_cold_call),
        "cold_baseline_precomputed_first_pointer": _cold_call_fields(baseline_precomputed_first_cold_call),
        "cold_baseline_precomputed_fresh_pointer": _cold_call_fields(baseline_precomputed_fresh_cold_call),
    }

    if correctness:
        ref = _reference_assign(x, centroids, chunk_rows=reference_chunk_rows)
        fresh_ref = _reference_assign(fresh_x, fresh_centroids, chunk_rows=reference_chunk_rows)
        torch.cuda.synchronize()
        baseline_public_raw_correctness = _assignment_correctness(
            baseline_public_raw_out,
            ref,
            x,
            centroids,
        )
        fresh_baseline_public_raw_correctness = _assignment_correctness(
            fresh_baseline_public_raw_out,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        candidate_public_raw_correctness = _assignment_correctness(
            candidate_public_raw_out,
            ref,
            x,
            centroids,
        )
        fresh_candidate_public_raw_correctness = _assignment_correctness(
            fresh_candidate_public_raw_out,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        candidate_precomputed_correctness = _assignment_correctness(
            candidate_precomputed_result,
            ref,
            x,
            centroids,
        )
        fresh_candidate_precomputed_correctness = _assignment_correctness(
            fresh_candidate_precomputed_result,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        baseline_precomputed_correctness = _assignment_correctness(
            baseline_precomputed_result,
            ref,
            x,
            centroids,
        )
        fresh_baseline_precomputed_correctness = _assignment_correctness(
            fresh_baseline_precomputed_result,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        result["candidate_public_raw_correctness"] = candidate_public_raw_correctness
        result["candidate_public_raw_fresh_pointer_correctness"] = fresh_candidate_public_raw_correctness
        result["baseline_public_raw_correctness"] = baseline_public_raw_correctness
        result["baseline_public_raw_fresh_pointer_correctness"] = fresh_baseline_public_raw_correctness
        result["candidate_precomputed_correctness"] = candidate_precomputed_correctness
        result["candidate_precomputed_fresh_pointer_correctness"] = fresh_candidate_precomputed_correctness
        result["baseline_precomputed_correctness"] = baseline_precomputed_correctness
        result["baseline_precomputed_fresh_pointer_correctness"] = fresh_baseline_precomputed_correctness
        correctness_rows = (
            candidate_public_raw_correctness,
            fresh_candidate_public_raw_correctness,
            baseline_public_raw_correctness,
            fresh_baseline_public_raw_correctness,
            candidate_precomputed_correctness,
            fresh_candidate_precomputed_correctness,
            baseline_precomputed_correctness,
            fresh_baseline_precomputed_correctness,
        )
        result["correct"] = all(bool(item["correct"]) for item in correctness_rows)

    if benchmark:
        baseline_public_raw_alternating = _alternating_call(
            lambda: baseline_adapter.compute(x, centroids),
            lambda: baseline_adapter.compute(fresh_x, fresh_centroids),
        )
        candidate_public_raw_alternating = _alternating_call(
            lambda: runtime.compute(
                x,
                centroids,
            ),
            lambda: runtime.compute(
                fresh_x,
                fresh_centroids,
            ),
        )
        candidate_precomputed_alternating = _alternating_call(
            lambda: flash_kmeans_assign_prepared(candidate_prepared),
            lambda: flash_kmeans_assign_prepared(fresh_candidate_prepared),
        )
        baseline_precomputed_alternating = _alternating_call(
            baseline_precomputed_first_call,
            baseline_precomputed_fresh_pointer_call,
        )

        def time_baseline_public_raw():
            return _bench_original_07cf_window(
                baseline_public_raw_alternating,
                cold_first_call=baseline_public_raw_fresh_cold_call,
            )

        def time_candidate_public_raw():
            return _bench_original_07cf_window(
                candidate_public_raw_alternating,
                cold_first_call=candidate_public_raw_fresh_cold_call,
            )

        def time_candidate_precomputed():
            return _bench_original_07cf_window(
                candidate_precomputed_alternating,
                cold_first_call=fresh_candidate_precomputed_first_cold_call,
            )

        def time_baseline_precomputed():
            return _bench_original_07cf_window(
                baseline_precomputed_alternating,
                cold_first_call=baseline_precomputed_fresh_cold_call,
            )

        measurement_order = _measurement_order(str(row["label"]))
        timers = {
            "candidate_public_raw": time_candidate_public_raw,
            "baseline_public_raw": time_baseline_public_raw,
            "candidate_precomputed": time_candidate_precomputed,
            "baseline_precomputed": time_baseline_precomputed,
        }
        timings = {name: timers[name]() for name in measurement_order}
        candidate_public_raw_timing = timings["candidate_public_raw"]
        baseline_public_raw_timing = timings["baseline_public_raw"]
        candidate_precomputed_timing = timings["candidate_precomputed"]
        baseline_precomputed_timing = timings["baseline_precomputed"]
        timing_backends = {
            candidate_public_raw_timing.backend,
            baseline_public_raw_timing.backend,
            candidate_precomputed_timing.backend,
            baseline_precomputed_timing.backend,
        }
        if timing_backends != {"cupti"}:
            raise RuntimeError(f"dual-lane benchmark requires CUPTI for every timing block, got {timing_backends}")
        result["measurement_order"] = list(measurement_order)
        result["measurement_order_policy"] = "deterministic_sha256_per_shape_permutation"
        result["timing_backend"] = "cupti"
        result["candidate_public_raw_gpu_span_ms"] = candidate_public_raw_timing.median_gpu_span_ms
        result["candidate_public_raw_kernel_sum_ms"] = candidate_public_raw_timing.median_kernel_sum_ms
        result["candidate_public_raw_inter_kernel_gap_ms"] = candidate_public_raw_timing.median_inter_kernel_gap_ms
        result["candidate_public_raw_host_enqueue_ms"] = _required_median(
            candidate_public_raw_timing.median_host_enqueue_ms,
            metric="candidate public-raw host_enqueue_ms",
        )
        result["candidate_public_raw_synchronized_e2e_ms"] = _required_median(
            candidate_public_raw_timing.median_synchronized_e2e_ms,
            metric="candidate public-raw synchronized_e2e_ms",
        )
        result["candidate_public_raw_timing_backend"] = candidate_public_raw_timing.backend
        result["candidate_public_raw_bench_iters"] = len(candidate_public_raw_timing.times_ms)
        result["candidate_public_raw_timing_diagnostics"] = _timing_diagnostics(
            candidate_public_raw_timing,
            primary_metric="synchronized_e2e_ms",
        )

        result["baseline_07cf_adapter_gpu_span_ms"] = baseline_public_raw_timing.median_gpu_span_ms
        result["baseline_07cf_adapter_kernel_sum_ms"] = baseline_public_raw_timing.median_kernel_sum_ms
        result["baseline_07cf_adapter_inter_kernel_gap_ms"] = baseline_public_raw_timing.median_inter_kernel_gap_ms
        result["baseline_07cf_adapter_host_enqueue_ms"] = _required_median(
            baseline_public_raw_timing.median_host_enqueue_ms,
            metric="07cf raw-adapter host_enqueue_ms",
        )
        result["baseline_07cf_adapter_synchronized_e2e_ms"] = _required_median(
            baseline_public_raw_timing.median_synchronized_e2e_ms,
            metric="07cf raw-adapter synchronized_e2e_ms",
        )
        result["baseline_07cf_adapter_timing_backend"] = baseline_public_raw_timing.backend
        result["baseline_07cf_adapter_bench_iters"] = len(baseline_public_raw_timing.times_ms)
        result["baseline_07cf_adapter_timing_diagnostics"] = _timing_diagnostics(
            baseline_public_raw_timing,
            primary_metric="synchronized_e2e_ms",
        )

        result["candidate_precomputed_gpu_span_ms"] = candidate_precomputed_timing.median_gpu_span_ms
        result["candidate_precomputed_kernel_sum_ms"] = candidate_precomputed_timing.median_kernel_sum_ms
        result["candidate_precomputed_inter_kernel_gap_ms"] = candidate_precomputed_timing.median_inter_kernel_gap_ms
        result["candidate_precomputed_timing_backend"] = candidate_precomputed_timing.backend
        result["candidate_precomputed_bench_iters"] = len(candidate_precomputed_timing.times_ms)
        result["candidate_precomputed_timing_diagnostics"] = _timing_diagnostics(
            candidate_precomputed_timing,
            primary_metric="gpu_span_ms",
        )

        result["baseline_07cf_precomputed_gpu_span_ms"] = baseline_precomputed_timing.median_gpu_span_ms
        result["baseline_07cf_precomputed_kernel_sum_ms"] = baseline_precomputed_timing.median_kernel_sum_ms
        result["baseline_07cf_precomputed_inter_kernel_gap_ms"] = baseline_precomputed_timing.median_inter_kernel_gap_ms
        result["baseline_07cf_precomputed_timing_backend"] = baseline_precomputed_timing.backend
        result["baseline_07cf_precomputed_bench_iters"] = len(baseline_precomputed_timing.times_ms)
        result["baseline_07cf_precomputed_timing_diagnostics"] = _timing_diagnostics(
            baseline_precomputed_timing,
            primary_metric="gpu_span_ms",
        )

        flops = 2.0 * int(row["B"]) * int(row["N"]) * int(row["K"]) * int(row["D"])
        result["candidate_public_raw_tflops_from_gpu_span"] = (
            flops / candidate_public_raw_timing.median_gpu_span_ms / 1e9
        )
        result.update(
            _dual_lane_speedups(
                candidate_public_raw_timing=candidate_public_raw_timing,
                baseline_public_raw_timing=baseline_public_raw_timing,
                candidate_precomputed_timing=candidate_precomputed_timing,
                baseline_precomputed_timing=baseline_precomputed_timing,
            )
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for flashlib_cake_kmeans.flash_kmeans_assign"
    )
    parser.add_argument("--shape", action="append", help="Shape label to run. Repeatable.")
    parser.add_argument(
        "--source",
        action="append",
        help="Source set to run, for example full95. Repeatable.",
    )
    parser.add_argument("--unique", action="store_true", help="Run one row per unique (B,N,D,K,dtype).")
    parser.add_argument(
        "--include-runtime-coverage",
        action="store_true",
        help="Include the 104 runtime-only rows; publication defaults to the registry 124.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit selected rows after filtering.")
    parser.add_argument("--arch", default=None, help="NVRTC architecture, e.g. sm_100a.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Emit available benchmark metadata without CUDA.",
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip PyTorch reference correctness checks.",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip CUPTI timing.")
    parser.add_argument(
        "--reference-chunk-rows",
        type=int,
        default=128,
        help="Rows per chunk for PyTorch reference.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based validation shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of validation shards.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the full JSON payload.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.reference_chunk_rows <= 0:
        parser.error("--reference-chunk-rows must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < count and count must be positive")
    if args.json is not None:
        args.json.unlink(missing_ok=True)

    rows = _selected_rows(args)
    measurement_session_id = uuid.uuid4().hex
    preprocess_source_digest = _preprocess_source_digest()
    if preprocess_source_digest != BASELINE_REGISTRY_PROFILE["shared_preprocess"]["source_sha256"]:
        raise RuntimeError("exported preprocessing sources do not match the pinned registry profile")
    profile_digest = hashlib.sha256(
        json.dumps(
            BASELINE_REGISTRY_PROFILE,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if profile_digest != BASELINE_REGISTRY_SHA256:
        raise RuntimeError("embedded registry baseline profile digest is stale")
    payload: dict[str, Any] = {
        "api": "flashlib_cake_kmeans.init(...).compute",
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "publication_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "parity_metric": "precomputed_gpu_speedup_vs_07cf",
        "baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "public_raw_baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "precomputed_baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": BASELINE_ENTRYPOINT,
        "benchmark_registry_baseline_key": BASELINE_REGISTRY_KEY,
        "benchmark_registry_baseline_sha256": BASELINE_REGISTRY_SHA256,
        "benchmark_registry_baseline_profile": BASELINE_REGISTRY_PROFILE,
        "registry_candidate_entrypoint": REGISTRY_CANDIDATE_ENTRYPOINT,
        "measured_candidate_entrypoint": MEASURED_CANDIDATE_ENTRYPOINT,
        "candidate_timing_boundary": CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": BASELINE_TIMING_BOUNDARY,
        "publication_speedup_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "publication_timing_backend": "cupti",
        "publication_speedup_convention": (
            "public_raw_e2e_speedup_vs_07cf_adapter = "
            "baseline_07cf_adapter_synchronized_e2e_ms / candidate_public_raw_synchronized_e2e_ms"
        ),
        "precomputed_parity_speedup_convention": (
            "precomputed_gpu_speedup_vs_07cf = "
            "baseline_07cf_precomputed_gpu_span_ms / candidate_precomputed_gpu_span_ms"
        ),
        "preprocess_impl": PREPROCESS_IMPL,
        "preprocess_source_sha256": preprocess_source_digest,
        "timing_window_ms": {"warmup_ms": WARMUP_MS, "bench_ms": BENCH_MS},
        "artifact": FLASH_KMEANS_EVOLUTION_ARTIFACT,
        "evolution_summary": FLASH_KMEANS_EVOLUTION_SUMMARY,
        "selected_row_count": len(rows),
        "selected_unique_shape_count": len({_shape_key(row) for row in rows}),
        "metadata_only": bool(args.metadata_only),
        "validation_shard": {"index": args.shard_index, "count": args.shard_count},
        "shapes": rows,
        "measurement_session": {
            "id": measurement_session_id,
            "scope": "dual_lane_separate_cupti_blocks_with_pointer_alternation",
            "same_process": True,
            "same_cupti_session": False,
            "interleaved": False,
            "timing_blocks": "separate_deterministically_ordered",
            "alternate_two_pointer_sets": True,
            "public_raw_e2e": {
                "candidate_api": "flashlib_cake_kmeans.init(...).compute(raw_inputs)",
                "baseline_api": "triton_h200_07cf_raw_adapter_v1.compute(raw_inputs)",
                "comparison_scope": "complete_raw_input_operators_not_assignment_only",
                "candidate_norm_policy": "route_required_internal_fused_bf16_pair_row_norm",
                "baseline_norm_policy": "shared_kernel_all_fields_required_by_frozen_07cf",
                "candidate_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
                "baseline_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
                "candidate_scratch_policy": "per_shape_per_stream_cached",
                "baseline_scratch_policy": "per_shape_per_stream_cached",
                "candidate_runtime_initialized_once": True,
                "baseline_runtime_initialized_once": True,
                "preprocess_impl": PREPROCESS_IMPL,
                "preprocess_source_sha256": preprocess_source_digest,
                "shared_preprocess_cold_compile_attributed_to_lane": None,
                "fixture_synchronized_before_cold_calls": True,
                "assignment_baseline": "frozen_07cf",
            },
            "precomputed_kernel_parity": {
                "candidate_api": "flash_kmeans_assign_prepared",
                "baseline_api": "euclid_assign_triton_h200_07cf",
                "candidate_norm_policy": "explicit_precomputed_outside_timing",
                "baseline_norm_policy": "explicit_precomputed_outside_timing",
                "candidate_output_policy": "preallocated_outside_timing",
                "baseline_output_policy": "preallocated_outside_timing",
                "both_pointer_sets_prepared_before_timing": True,
            },
            "runtime_instances_reused_across_shapes": True,
            "resident_multi_shape_cache_benchmarked": False,
            "cache_policy": "synchronize_and_clear_after_each_completed_shape",
            "baseline_commit": BASELINE_COMMIT,
            "order_policy": "deterministic_sha256_per_shape_permutation",
            "order_seed": MEASUREMENT_ORDER_SEED,
        },
    }
    if args.metadata_only:
        payload["results"] = []
    else:
        import torch
        from flashlib_cake_kmeans import init

        payload["hardware"] = {
            "device": torch.cuda.get_device_name(),
            "arch": f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}a",
        }
        if not args.no_benchmark:
            require_cupti()
            runtime, runtime_init_call = measure_host_call(
                lambda: init(
                    device=torch.cuda.current_device(),
                    arch=args.arch,
                    compile="lazy",
                )
            )
            baseline_adapter, baseline_adapter_init_call = measure_host_call(
                lambda: TritonH20007cfRawAdapter(
                    device_index=runtime.device_index,
                    arch=runtime.arch,
                )
            )
        else:
            runtime = init(
                device=torch.cuda.current_device(),
                arch=args.arch,
                compile="lazy",
            )
            runtime_init_call = None
            baseline_adapter = TritonH20007cfRawAdapter(
                device_index=runtime.device_index,
                arch=runtime.arch,
            )
            baseline_adapter_init_call = None
        shared_preprocess_cold_compile = (
            _measure_shared_preprocess_cold_compile(rows[0], arch=runtime.arch)
            if rows and not args.no_benchmark
            else None
        )
        payload["cold_candidate_runtime_init"] = _cold_call_fields(runtime_init_call)
        payload["cold_baseline_07cf_adapter_init"] = _cold_call_fields(baseline_adapter_init_call)
        payload["cold_shared_preprocess_compile"] = _cold_call_fields(shared_preprocess_cold_compile)
        results = []
        for row in rows:
            try:
                results.append(
                    _run_shape(
                        row,
                        runtime=runtime,
                        baseline_adapter=baseline_adapter,
                        arch=args.arch,
                        correctness=not args.no_correctness,
                        benchmark=not args.no_benchmark,
                        reference_chunk_rows=args.reference_chunk_rows,
                        measurement_session_id=measurement_session_id,
                    )
                )
            finally:
                runtime.clear()
                baseline_adapter.clear()
        payload["results"] = results
        payload["runtime_workspace_lifecycle"] = (
            "both_init_once_runtimes_synchronize_and_clear_after_each_completed_shape"
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.json, text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
