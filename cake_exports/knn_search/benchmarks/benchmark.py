from __future__ import annotations

import argparse
import hashlib
import importlib
import json
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

from flashlib_cake_knn_search._benchmark import (  # noqa: E402
    bench_gpu_time,
    measure_host_call,
    require_cupti,
)

SHAPE_RECORDS = json.loads((Path(__file__).with_name("shape_records.json")).read_text(encoding="utf-8"))
ROUTE_MANIFEST = json.loads((Path(__file__).with_name("expected_routes.json")).read_text(encoding="utf-8"))
EXPECTED_ROUTES = {row["shape"]: row["selected_route"] for row in ROUTE_MANIFEST}
SEMANTIC_ENTRYPOINT = "loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:launch_for_eval"
SHAPES: dict[str, dict[str, Any]] = {
    row["label"]: {**row["params"], "recorded": row["recorded"]} for row in SHAPE_RECORDS
}
BASELINE_NAME = "flashlib.flash_knn"
MEASUREMENT_ORDER_SEED = "flashlib-knn-search-export-paired-v1"


def _measurement_order(label: str) -> tuple[str, str, str]:
    """Choose one stable per-shape order for baseline/public/prepared timing."""

    orders = tuple(permutations(("baseline", "public", "prepared")))
    digest = hashlib.sha256(f"{MEASUREMENT_ORDER_SEED}:{label}".encode()).digest()
    return orders[int.from_bytes(digest[:2], "little") % len(orders)]


def _measurement_session_fields(measurement_session_id: str) -> dict[str, Any]:
    if not isinstance(measurement_session_id, str) or not measurement_session_id.strip():
        raise ValueError("measurement_session_id must be a non-empty string")
    return {
        "measurement_session_id": measurement_session_id,
        "baseline_measurement_session_id": measurement_session_id,
        "public_measurement_session_id": measurement_session_id,
        "prepared_measurement_session_id": measurement_session_id,
        "baseline_public_prepared_same_session": True,
    }


def _recorded_diagnostics(recorded: dict[str, Any]) -> dict[str, Any]:
    return {f"recorded_{key}": value for key, value in recorded.items()}


def _shape_metadata(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in shape.items() if key != "recorded"},
        **_recorded_diagnostics(shape["recorded"]),
    }


def _load_flashlib_baseline():
    flash_knn = getattr(importlib.import_module("flashlib"), "flash_knn", None)
    if not callable(flash_knn):
        raise RuntimeError("flashlib.flash_knn is required for same-process KNN-search benchmarking")
    return flash_knn


def _require_correct_baseline(name: str, diagnostics: dict[str, Any]) -> None:
    if bool(diagnostics.get("correct")):
        return
    raise RuntimeError(
        f"{BASELINE_NAME} correctness failed for {name}: "
        f"recall={diagnostics.get('recall')!r}, "
        f"max_abs_dist_error={diagnostics.get('max_abs_dist_error')!r}, "
        f"required_recall={diagnostics.get('required_recall')!r}"
    )


def _host_call_diagnostics(timing: Any) -> dict[str, float] | None:
    if timing is None:
        return None
    return {
        "host_enqueue_ms": timing.host_enqueue_ms,
        "synchronized_e2e_ms": timing.synchronized_e2e_ms,
    }


def _timing_diagnostics(timing: Any) -> dict[str, Any]:
    return {
        "official_gpu_metric": "gpu_span_ms",
        "gpu_span_ms": {"median": timing.median_gpu_span_ms, "iterations": timing.times_ms},
        "kernel_sum_ms": {
            "median": timing.median_kernel_sum_ms,
            "iterations": timing.kernel_sum_times_ms,
        },
        "active_union_ms": {
            "median": timing.median_active_union_ms,
            "iterations": timing.active_union_times_ms,
        },
        "inter_kernel_gap_ms": {
            "median": timing.median_inter_kernel_gap_ms,
            "iterations": timing.inter_kernel_gap_times_ms,
        },
        "activity_count": {
            "median": timing.median_activity_count,
            "iterations": timing.activity_counts,
        },
        "correlated_launch_activity_count": {
            "median": timing.median_launch_activity_count,
            "iterations": timing.launch_activity_counts,
        },
        "correlated_kernel_activity_count": {
            "median": timing.median_kernel_activity_count,
            "iterations": timing.kernel_activity_counts,
        },
        "host_enqueue_ms": {
            "median": timing.median_host_enqueue_ms,
            "iterations": timing.host_enqueue_times_ms,
        },
        "synchronized_e2e_ms": {
            "median": timing.median_synchronized_e2e_ms,
            "iterations": timing.synchronized_e2e_times_ms,
        },
        "cold_first_call": {
            "host_enqueue_ms": timing.cold_first_call_host_enqueue_ms,
            "synchronized_e2e_ms": timing.cold_first_call_synchronized_e2e_ms,
        },
    }


def _write_json_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_inputs(shape: dict[str, Any]):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(shape["seed"]))
    database = torch.randn(
        (int(shape["B"]), int(shape["M"]), int(shape["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    if bool(shape.get("self_search", False)):
        if int(shape["Q"]) != int(shape["M"]):
            raise ValueError("self_search shapes require Q == M")
        query = database
    else:
        query = torch.randn(
            (int(shape["B"]), int(shape["Q"]), int(shape["D"])),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).contiguous()
    return query, database


def _reference_topk(query, database, k: int):
    import torch

    values: list[Any] = []
    indices: list[Any] = []
    q_f32 = query.float()
    db_f32 = database.float()
    db_sq = (db_f32 * db_f32).sum(-1)
    block = 128
    for start in range(0, int(query.shape[1]), block):
        q = q_f32[:, start : start + block, :]
        q_sq = (q * q).sum(-1)
        dots = torch.matmul(q, db_f32.transpose(-1, -2))
        dists = q_sq.unsqueeze(-1) + db_sq.unsqueeze(1) - 2.0 * dots
        vals, idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
        values.append(vals)
        indices.append(idx.to(torch.int32))
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)


def _recall(got_indices, expected_indices) -> float:
    matches = (got_indices.unsqueeze(-1) == expected_indices.unsqueeze(-2)).any(dim=-1)
    return float(matches.to(got_indices.device, dtype=got_indices.float().dtype).mean().item())


def _distances_for_indices(query, database, indices):
    import torch

    q_f32 = query.float()
    db_f32 = database.float()
    bsz, q_rows, dim = q_f32.shape
    m_rows = int(db_f32.shape[1])
    safe_indices = indices.to(torch.int64).clamp(0, m_rows - 1)
    gather_src = db_f32.unsqueeze(1).expand(bsz, q_rows, m_rows, dim)
    gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    neighbors = torch.gather(gather_src, 2, gather_idx)
    return ((q_f32.unsqueeze(2) - neighbors) ** 2).sum(-1)


def _knn_correctness_diagnostics(
    query,
    database,
    output,
    expected_indices,
    *,
    required_recall: float,
) -> dict[str, Any]:
    exact_distances = _distances_for_indices(query, database, output[1])
    recall = _recall(output[1], expected_indices)
    max_abs_dist_error = float((output[0] - exact_distances).abs().max().item())
    return {
        "recall": recall,
        "max_abs_dist_error": max_abs_dist_error,
        "required_recall": required_recall,
        "correct": bool(recall >= required_recall and max_abs_dist_error <= 1.0e-2),
    }


def _run_shape(
    name: str,
    shape: dict[str, Any],
    *,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    measurement_session_id: str | None = None,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_knn_search import knn_search, knn_search_prepared, prepare_knn_search

    session_id = measurement_session_id or uuid.uuid4().hex
    session_fields = _measurement_session_fields(session_id)
    query, database = _make_inputs(shape)
    k = int(shape["K"])

    def public_first_call():
        return knn_search(query, database, k, arch=arch, return_info=True)

    baseline_out = None
    baseline_cold_first_call = None
    public_cold_first_call = None
    prepare_cold_call = None
    prepared_cold_first_call = None
    if benchmark:
        flash_knn = _load_flashlib_baseline()
        baseline_out, baseline_cold_first_call = measure_host_call(
            lambda: flash_knn(query, database, k=k)
        )
        (_, public_route_info), public_cold_first_call = measure_host_call(public_first_call)
        prepared, prepare_cold_call = measure_host_call(
            lambda: prepare_knn_search(query, database, k, arch=arch)
        )
        (out, route_info), prepared_cold_first_call = measure_host_call(
            lambda: knn_search_prepared(prepared, return_info=True)
        )
        if public_route_info["selected_route"] != route_info["selected_route"]:
            raise RuntimeError(
                "cold public and prepared KNN-search calls selected different routes: "
                f"{public_route_info['selected_route']!r} != {route_info['selected_route']!r}"
            )
    else:
        out, route_info = public_first_call()
        torch.cuda.synchronize()

    result: dict[str, Any] = {
        "shape": name,
        "B": int(shape["B"]),
        "Q": int(shape["Q"]),
        "M": int(shape["M"]),
        "D": int(shape["D"]),
        "K": k,
        "semantic_entrypoint": route_info["semantic_entrypoint"],
        "selected_route": route_info["selected_route"],
        "launch_entrypoint": route_info["launch_entrypoint"],
        "exact_launch_plan": route_info["exact_launch_plan"],
        "expected_route": EXPECTED_ROUTES[name],
        "route_matches_expected": route_info["selected_route"] == EXPECTED_ROUTES[name],
        "baseline_name": BASELINE_NAME,
        "baseline_entrypoint": BASELINE_NAME,
        **session_fields,
        **_recorded_diagnostics(shape["recorded"]),
    }
    if benchmark:
        result.update(
            {
                "prepared_launch_count": route_info["prepared_launch_count"],
                "cold_baseline_call": _host_call_diagnostics(baseline_cold_first_call),
                "cold_public_call": _host_call_diagnostics(public_cold_first_call),
                "prepared_setup_after_cold_public": _host_call_diagnostics(prepare_cold_call),
                "cold_prepared_call": _host_call_diagnostics(prepared_cold_first_call),
            }
        )

    reference_indices = None
    if correctness or benchmark:
        _, reference_indices = _reference_topk(query, database, k)
        torch.cuda.synchronize()

    required_recall = float(shape.get("min_recall", 0.999))
    if correctness:
        result.update(
            _knn_correctness_diagnostics(
                query,
                database,
                out,
                reference_indices,
                required_recall=required_recall,
            )
        )

    if benchmark:
        baseline_correctness = _knn_correctness_diagnostics(
            query,
            database,
            baseline_out,
            reference_indices,
            required_recall=required_recall,
        )
        result.update({f"baseline_{key}": value for key, value in baseline_correctness.items()})
        _require_correct_baseline(name, baseline_correctness)

        baseline_output_holder = [baseline_out]
        public_output_holder = [out]

        def run_baseline():
            baseline_output_holder[0] = flash_knn(query, database, k=k)
            return baseline_output_holder[0]

        def run_public():
            public_output_holder[0] = knn_search(query, database, k, arch=arch)
            return public_output_holder[0]

        def time_baseline():
            return bench_gpu_time(
                run_baseline,
                cold_l2=True,
                cold_first_call=baseline_cold_first_call,
            )

        def time_public():
            return bench_gpu_time(
                run_public,
                cold_l2=True,
                cold_first_call=public_cold_first_call,
            )

        def time_prepared():
            return bench_gpu_time(
                lambda: knn_search_prepared(prepared),
                cold_l2=True,
                cold_first_call=prepared_cold_first_call,
            )

        measurement_order = _measurement_order(name)
        timers = {
            "baseline": time_baseline,
            "public": time_public,
            "prepared": time_prepared,
        }
        timings = {timer_name: timers[timer_name]() for timer_name in measurement_order}
        baseline_timing = timings["baseline"]
        public_timing = timings["public"]
        prepared_timing = timings["prepared"]
        timing_backends = {
            baseline_timing.backend,
            public_timing.backend,
            prepared_timing.backend,
        }
        if timing_backends != {"cupti"}:
            raise RuntimeError(
                f"baseline/public/prepared must all use CUPTI, got {sorted(timing_backends)!r}"
            )

        result.update(
            {
                "measurement_order": list(measurement_order),
                "measurement_order_policy": "deterministic_sha256_per_shape_permutation",
                "baseline_ms": baseline_timing.median_ms,
                "baseline_gpu_span_ms": baseline_timing.median_gpu_span_ms,
                "baseline_kernel_sum_ms": baseline_timing.median_kernel_sum_ms,
                "baseline_active_union_ms": baseline_timing.median_active_union_ms,
                "baseline_inter_kernel_gap_ms": baseline_timing.median_inter_kernel_gap_ms,
                "baseline_timing_backend": baseline_timing.backend,
                "baseline_bench_iters": len(baseline_timing.times_ms),
                "baseline_timing_diagnostics": _timing_diagnostics(baseline_timing),
                "kernel_ms": public_timing.median_ms,
                "public_gpu_span_ms": public_timing.median_gpu_span_ms,
                "public_kernel_sum_ms": public_timing.median_kernel_sum_ms,
                "public_active_union_ms": public_timing.median_active_union_ms,
                "public_inter_kernel_gap_ms": public_timing.median_inter_kernel_gap_ms,
                "public_timing_diagnostics": _timing_diagnostics(public_timing),
                "prepared_gpu_span_ms": prepared_timing.median_gpu_span_ms,
                "prepared_kernel_sum_ms": prepared_timing.median_kernel_sum_ms,
                "prepared_active_union_ms": prepared_timing.median_active_union_ms,
                "prepared_inter_kernel_gap_ms": prepared_timing.median_inter_kernel_gap_ms,
                "prepared_timing_diagnostics": _timing_diagnostics(prepared_timing),
                "public_over_prepared": public_timing.median_ms / prepared_timing.median_ms,
                "timing_backend": public_timing.backend,
                "bench_iters": len(public_timing.times_ms),
                "speedup_vs_baseline": baseline_timing.median_ms / public_timing.median_ms,
                "prepared_speedup_vs_baseline": (
                    baseline_timing.median_ms / prepared_timing.median_ms
                ),
            }
        )
        flops = 2.0 * int(shape["B"]) * int(shape["Q"]) * int(shape["M"]) * int(shape["D"])
        result["tflops"] = flops / public_timing.median_ms / 1e9
        result["qps"] = int(shape["B"]) * int(shape["Q"]) / (public_timing.median_ms / 1000.0)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for flashlib_cake_knn_search.knn_search"
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=sorted(SHAPES),
        help="Shape label to run. Repeatable.",
    )
    parser.add_argument("--arch", default=None, help="NVRTC architecture, e.g. sm_100a.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Emit available benchmark metadata without CUDA.",
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip candidate reference checks. The measured FlashLib baseline remains fail-closed.",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip CUPTI timing.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based validation shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of validation shards.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the full JSON payload.")
    args = parser.parse_args()

    selected = args.shape or list(SHAPES)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < count and count must be positive")
    if args.json is not None:
        args.json.unlink(missing_ok=True)
    selected = selected[args.shard_index :: args.shard_count]
    measurement_session_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "api": "flashlib_cake_knn_search.knn_search",
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "baseline_name": BASELINE_NAME,
        "baseline_entrypoint": BASELINE_NAME,
        "speedup_convention": "same_session_flashlib_flash_knn_gpu_span_ms / exported_cake_gpu_span_ms",
        "shapes": {name: _shape_metadata(SHAPES[name]) for name in selected},
        "metadata_only": bool(args.metadata_only),
        "validation_shard": {"index": args.shard_index, "count": args.shard_count},
        "measurement_session": {
            "id": measurement_session_id,
            "scope": "per_shape_interleaved_baseline_public_prepared",
            "baseline_candidate_same_process": True,
            "baseline_public_prepared_same_session": True,
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
                name,
                SHAPES[name],
                arch=args.arch,
                correctness=not args.no_correctness,
                benchmark=not args.no_benchmark,
                measurement_session_id=measurement_session_id,
            )
            for name in selected
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
