from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from flashlib_cake_kmeans._benchmark import bench_gpu_time, require_cupti  # noqa: E402


def _label_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


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
            raise SystemExit(
                f"unknown shape label(s) {missing}. Available: {available}"
            )
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
    return rows


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
                ref[b, start : start + q.shape[0]] = scores.argmax(dim=-1).to(
                    torch.int32
                )
    return ref


def _run_shape(
    row: dict[str, Any],
    *,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    reference_chunk_rows: int,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_kmeans import flash_kmeans_assign

    x, centroids = _make_inputs(row)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty(
        (int(row["B"]), int(row["N"])), dtype=torch.int32, device=x.device
    )
    cluster_ids, route_info = flash_kmeans_assign(
        x,
        centroids,
        out=out,
        x_sq=x_sq,
        c_sq=c_sq,
        arch=arch,
        return_info=True,
    )
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "label": row["label"],
        "source": row["source"],
        "B": int(row["B"]),
        "N": int(row["N"]),
        "D": int(row["D"]),
        "K": int(row["K"]),
        "dtype": row.get("dtype", "bfloat16"),
        "expected_route": row["route"],
        "selected_route": route_info["selected_route"],
        "child_route": route_info.get("child_route"),
        "route_matches_artifact": bool(route_info["selected_route"] == row["route"]),
        "evolution_kernel_ms": row["evolution_kernel_ms"],
        "evolution_flashlib_ms": row["evolution_flashlib_ms"],
        "evolution_tflops": row["evolution_tflops"],
        "evolution_speedup": row["evolution_speedup"],
    }

    if correctness:
        ref = _reference_assign(x, centroids, chunk_rows=reference_chunk_rows)
        torch.cuda.synchronize()
        matches = cluster_ids == ref
        result["match_rate"] = float(matches.float().mean().item())
        result["mismatch_count"] = int((~matches).sum().item())
        result["correct"] = bool(result["mismatch_count"] == 0)

    if benchmark:
        timing = bench_gpu_time(
            lambda: flash_kmeans_assign(
                x,
                centroids,
                out=out,
                x_sq=x_sq,
                c_sq=c_sq,
                arch=arch,
            ),
            cold_l2=True,
        )
        result["kernel_ms"] = timing.median_ms
        result["timing_backend"] = timing.backend
        result["bench_iters"] = len(timing.times_ms)
        flops = 2.0 * int(row["B"]) * int(row["N"]) * int(row["K"]) * int(row["D"])
        result["tflops"] = flops / timing.median_ms / 1e9
        result["speedup_vs_evolution_flashlib_ms"] = (
            row["evolution_flashlib_ms"] / timing.median_ms
        )
        result["baseline_name"] = "Cake-recorded FlashLib baseline"
        result["baseline_ms"] = row["evolution_flashlib_ms"]
        result["speedup_vs_baseline"] = result["speedup_vs_evolution_flashlib_ms"]

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for flashlib_cake_kmeans.flash_kmeans_assign"
    )
    parser.add_argument(
        "--shape", action="append", help="Shape label to run. Repeatable."
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Source set to run, for example full95. Repeatable.",
    )
    parser.add_argument(
        "--unique", action="store_true", help="Run one row per unique (B,N,D,K,dtype)."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit selected rows after filtering."
    )
    parser.add_argument(
        "--arch", default=None, help="NVRTC architecture, e.g. sm_100a."
    )
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
    parser.add_argument(
        "--no-benchmark", action="store_true", help="Skip CUPTI timing."
    )
    parser.add_argument(
        "--reference-chunk-rows",
        type=int,
        default=128,
        help="Rows per chunk for PyTorch reference.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path for JSON output."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.reference_chunk_rows <= 0:
        parser.error("--reference-chunk-rows must be positive")

    rows = _selected_rows(args)
    payload: dict[str, Any] = {
        "api": "flashlib_cake_kmeans.flash_kmeans_assign",
        "baseline_name": "Cake-recorded FlashLib baseline",
        "artifact": FLASH_KMEANS_EVOLUTION_ARTIFACT,
        "evolution_summary": FLASH_KMEANS_EVOLUTION_SUMMARY,
        "selected_row_count": len(rows),
        "selected_unique_shape_count": len({_shape_key(row) for row in rows}),
        "metadata_only": bool(args.metadata_only),
        "shapes": rows,
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
            )
            for row in rows
        ]

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
