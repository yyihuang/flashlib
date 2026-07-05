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
    FLASH_KMEANS_SHAPES,
)
from flash_kmeans_triton_h200 import euclid_assign_triton_h200  # noqa: E402
from flashlib_cake_kmeans._benchmark import (  # noqa: E402
    bench_gpu_time,
    measure_host_call,
    require_cupti,
)

ROUTE_MANIFEST = json.loads((Path(__file__).with_name("expected_routes.json")).read_text(encoding="utf-8"))
EXPECTED_ROUTES = {row["shape"]: row["selected_route"] for row in ROUTE_MANIFEST}
SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"
BASELINE_NAME = "triton_h200_07cf"
BASELINE_COMMIT = "07cf2a27928aacf6790c950a265d8b8dc83c87cf"
WARMUP_MS = 20.0
BENCH_MS = 100.0
MEASUREMENT_ORDER_SEED = "flashlib-kmeans-export-paired-v1"


def _timing_diagnostics(timing: Any) -> dict[str, Any]:
    return {
        "official_gpu_metric": "gpu_span_ms",
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


def _measurement_order(label: str) -> tuple[str, str, str]:
    """Choose one stable per-shape order for baseline/public/prepared timing."""

    orders = tuple(permutations(("baseline", "public", "prepared")))
    digest = hashlib.sha256(f"{MEASUREMENT_ORDER_SEED}:{label}".encode()).digest()
    return orders[int.from_bytes(digest[:2], "little") % len(orders)]


def _shape_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["B"]),
        int(row["N"]),
        int(row["D"]),
        int(row["K"]),
        str(row.get("dtype", "bfloat16")),
    )


def _selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(FLASH_KMEANS_SHAPES)
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


def _make_inputs(row: dict[str, Any]):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(_label_seed(str(row["label"])))
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
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    reference_chunk_rows: int,
    measurement_session_id: str,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_kmeans import (
        flash_kmeans_assign,
        flash_kmeans_assign_prepared,
        prepare_flash_kmeans_assign,
    )

    x, centroids = _make_inputs(row)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((int(row["B"]), int(row["N"])), dtype=torch.int32, device=x.device)
    baseline_out = torch.empty_like(out)

    def baseline_first_call():
        return euclid_assign_triton_h200(x, centroids, x_sq, c_sq, out=baseline_out)

    def public_first_call():
        return flash_kmeans_assign(x, centroids, out=out, x_sq=x_sq, c_sq=c_sq, arch=arch, return_info=True)

    baseline_cold_first_call = None
    public_cold_first_call = None
    prepare_cold_call = None
    prepared_cold_first_call = None
    if benchmark:
        (baseline_out, baseline_config), baseline_cold_first_call = measure_host_call(baseline_first_call)
        (_, public_route_info), public_cold_first_call = measure_host_call(public_first_call)
        prepared, prepare_cold_call = measure_host_call(
            lambda: prepare_flash_kmeans_assign(
                x,
                centroids,
                out=out,
                x_sq=x_sq,
                c_sq=c_sq,
                arch=arch,
            )
        )
        (cluster_ids, route_info), prepared_cold_first_call = measure_host_call(
            lambda: flash_kmeans_assign_prepared(prepared, return_info=True)
        )
        if public_route_info["selected_route"] != route_info["selected_route"]:
            raise RuntimeError(
                "cold public and prepared Flash-KMeans calls selected different routes: "
                f"{public_route_info['selected_route']!r} != {route_info['selected_route']!r}"
            )
    else:
        baseline_out, baseline_config = baseline_first_call()
        prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
        )
        cluster_ids, route_info = flash_kmeans_assign_prepared(prepared, return_info=True)
        torch.cuda.synchronize()

    result: dict[str, Any] = {
        "shape": row["label"],
        "label": row["label"],
        "source": row["source"],
        "B": int(row["B"]),
        "N": int(row["N"]),
        "D": int(row["D"]),
        "K": int(row["K"]),
        "dtype": row.get("dtype", "bfloat16"),
        "semantic_entrypoint": route_info["semantic_entrypoint"],
        "expected_route": EXPECTED_ROUTES[row["label"]],
        "selected_route": route_info["selected_route"],
        "child_route": route_info.get("child_route"),
        "route_matches_expected": bool(route_info["selected_route"] == EXPECTED_ROUTES[row["label"]]),
        "evolution_kernel_ms": row["evolution_kernel_ms"],
        "evolution_flashlib_ms": row["evolution_flashlib_ms"],
        "evolution_tflops": row["evolution_tflops"],
        "evolution_speedup": row["evolution_speedup"],
        "baseline_name": BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "triton_h200_config": baseline_config,
        "measurement_session_id": measurement_session_id,
        "triton_h200_07cf_measurement_session_id": measurement_session_id,
        "triton_h200_07cf_same_session": True,
        "prepared_launch_count": route_info["prepared_launch_count"],
        "cold_public_call": {
            "host_enqueue_ms": (public_cold_first_call.host_enqueue_ms if public_cold_first_call is not None else None),
            "synchronized_e2e_ms": (
                public_cold_first_call.synchronized_e2e_ms if public_cold_first_call is not None else None
            ),
        },
        "prepared_setup_after_cold_public": {
            "host_enqueue_ms": prepare_cold_call.host_enqueue_ms if prepare_cold_call is not None else None,
            "synchronized_e2e_ms": (prepare_cold_call.synchronized_e2e_ms if prepare_cold_call is not None else None),
        },
    }

    if correctness:
        ref = _reference_assign(x, centroids, chunk_rows=reference_chunk_rows)
        torch.cuda.synchronize()
        baseline_correctness = _assignment_correctness(baseline_out, ref, x, centroids)
        result.update({f"triton_h200_{key}": value for key, value in baseline_correctness.items()})
        result.update(_assignment_correctness(cluster_ids, ref, x, centroids))

    if benchmark:

        def time_baseline():
            return _bench_original_07cf_window(
                lambda: euclid_assign_triton_h200(
                    x,
                    centroids,
                    x_sq,
                    c_sq,
                    out=baseline_out,
                ),
                cold_first_call=baseline_cold_first_call,
            )

        def time_public():
            # Public timing is diagnostic: setup happens before the first GPU
            # activity and is reported by the host brackets below. Keep this
            # sample count bounded so tiny kernels do not trigger thousands of
            # repeated prepare/capture operations merely to fill a GPU window.
            return bench_gpu_time(
                lambda: flash_kmeans_assign(
                    x,
                    centroids,
                    out=out,
                    x_sq=x_sq,
                    c_sq=c_sq,
                    arch=arch,
                ),
                warmup_iters=5,
                bench_iters=20,
                cold_l2=True,
                cold_first_call=public_cold_first_call,
            )

        def time_prepared():
            return _bench_original_07cf_window(
                lambda: flash_kmeans_assign_prepared(prepared),
                cold_first_call=prepared_cold_first_call,
            )

        measurement_order = _measurement_order(str(row["label"]))
        timers = {
            "baseline": time_baseline,
            "public": time_public,
            "prepared": time_prepared,
        }
        timings = {name: timers[name]() for name in measurement_order}
        baseline_timing = timings["baseline"]
        public_timing = timings["public"]
        prepared_timing = timings["prepared"]
        result["measurement_order"] = list(measurement_order)
        result["measurement_order_policy"] = "deterministic_sha256_per_shape_permutation"
        result["triton_h200_07cf_ms"] = baseline_timing.median_ms
        result["triton_h200_07cf_timing_backend"] = baseline_timing.backend
        result["triton_h200_07cf_bench_iters"] = len(baseline_timing.times_ms)
        result["triton_h200_07cf_kernel_sum_ms"] = baseline_timing.median_kernel_sum_ms
        result["triton_h200_07cf_inter_kernel_gap_ms"] = baseline_timing.median_inter_kernel_gap_ms
        result["triton_h200_07cf_timing_diagnostics"] = _timing_diagnostics(baseline_timing)
        result["kernel_ms"] = public_timing.median_ms
        result["public_gpu_span_ms"] = public_timing.median_gpu_span_ms
        result["public_kernel_sum_ms"] = public_timing.median_kernel_sum_ms
        result["public_inter_kernel_gap_ms"] = public_timing.median_inter_kernel_gap_ms
        result["prepared_gpu_span_ms"] = prepared_timing.median_gpu_span_ms
        result["prepared_kernel_sum_ms"] = prepared_timing.median_kernel_sum_ms
        result["prepared_inter_kernel_gap_ms"] = prepared_timing.median_inter_kernel_gap_ms
        result["public_timing_diagnostics"] = _timing_diagnostics(public_timing)
        result["prepared_timing_diagnostics"] = _timing_diagnostics(prepared_timing)
        result["public_over_prepared"] = public_timing.median_ms / prepared_timing.median_ms
        result["timing_backend"] = public_timing.backend
        result["bench_iters"] = len(public_timing.times_ms)
        result["kernel_sum_ms"] = public_timing.median_kernel_sum_ms
        result["inter_kernel_gap_ms"] = public_timing.median_inter_kernel_gap_ms
        result["timing_diagnostics"] = _timing_diagnostics(public_timing)
        flops = 2.0 * int(row["B"]) * int(row["N"]) * int(row["K"]) * int(row["D"])
        result["tflops"] = flops / public_timing.median_ms / 1e9
        result["baseline_ms"] = baseline_timing.median_ms
        result["speedup_vs_baseline"] = baseline_timing.median_ms / public_timing.median_ms
        result["prepared_speedup_vs_baseline"] = baseline_timing.median_ms / prepared_timing.median_ms

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
    payload: dict[str, Any] = {
        "api": "flashlib_cake_kmeans.flash_kmeans_assign",
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "baseline_name": BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "speedup_convention": "triton_h200_07cf_ms / exported_cake_ms",
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
            "scope": "per_shape_interleaved_candidate_baseline",
            "candidate_and_07cf_same_process": True,
            "baseline_commit": BASELINE_COMMIT,
            "order_policy": "deterministic_sha256_per_shape_permutation",
            "order_seed": MEASUREMENT_ORDER_SEED,
        },
    }
    if args.metadata_only:
        payload["results"] = []
    else:
        import torch

        payload["hardware"] = {
            "device": torch.cuda.get_device_name(),
            "arch": f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}a",
        }
        if not args.no_benchmark:
            require_cupti()
        payload["results"] = [
            _run_shape(
                row,
                arch=args.arch,
                correctness=not args.no_correctness,
                benchmark=not args.no_benchmark,
                reference_chunk_rows=args.reference_chunk_rows,
                measurement_session_id=measurement_session_id,
            )
            for row in rows
        ]

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.json, text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
