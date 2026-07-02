from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashlib_cake_knn_build._benchmark import bench_gpu_time, require_cupti  # noqa: E402


SHAPE_RECORDS = json.loads((Path(__file__).with_name("shape_records.json")).read_text(encoding="utf-8"))
SHAPES: dict[str, dict[str, Any]] = {
    row["label"]: {**row["params"], "recorded": row["recorded"]} for row in SHAPE_RECORDS
}


def _make_database(shape: dict[str, Any]):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(shape["seed"]))
    return torch.randn(
        (int(shape["B"]), int(shape["M"]), int(shape["D"])),
        dtype=torch.float16 if shape.get("dtype") == "float16" else torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()


def _reference_topk(database, k: int):
    import torch

    values: list[Any] = []
    indices: list[Any] = []
    db_f32 = database.float()
    db_sq = (db_f32 * db_f32).sum(-1)
    block = 256
    for start in range(0, int(database.shape[1]), block):
        query = db_f32[:, start : start + block, :]
        q_sq = (query * query).sum(-1)
        dots = torch.matmul(query, db_f32.transpose(-1, -2))
        dists = q_sq.unsqueeze(-1) + db_sq.unsqueeze(1) - 2.0 * dots
        vals, idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
        values.append(vals)
        indices.append(idx.to(torch.int32))
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)


def _recall(got_indices, expected_indices) -> float:
    matches = (got_indices.unsqueeze(-1) == expected_indices.unsqueeze(-2)).any(dim=-1)
    return float(
        matches.to(got_indices.device, dtype=got_indices.float().dtype).mean().item()
    )


def _distances_for_indices(database, indices):
    import torch

    db_f32 = database.float()
    bsz, n_rows, dim = db_f32.shape
    safe_indices = indices.to(torch.int64).clamp(0, n_rows - 1)
    gather_src = db_f32.unsqueeze(1).expand(bsz, n_rows, n_rows, dim)
    gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    neighbors = torch.gather(gather_src, 2, gather_idx)
    query = db_f32.unsqueeze(2)
    return ((query - neighbors) ** 2).sum(-1)


def _run_shape(
    name: str,
    shape: dict[str, Any],
    *,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_knn_build import knn_build

    database = _make_database(shape)
    k = int(shape["K"])
    out = knn_build(database, k, arch=arch)
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "shape": name,
        "B": int(shape["B"]),
        "Q": int(shape["Q"]),
        "M": int(shape["M"]),
        "D": int(shape["D"]),
        "K": k,
    }

    if correctness:
        _, ref_indices = _reference_topk(database, k)
        torch.cuda.synchronize()
        exact_dists = _distances_for_indices(database, out[1])
        result["recall"] = _recall(out[1], ref_indices)
        result["max_abs_dist_error"] = float((out[0] - exact_dists).abs().max().item())
        result["correct"] = bool(
            result["recall"] >= 0.999 and result["max_abs_dist_error"] <= 1.0e-2
        )

    if benchmark:
        timing = bench_gpu_time(lambda: knn_build(database, k, arch=arch), cold_l2=True)
        result["kernel_ms"] = timing.median_ms
        result["timing_backend"] = timing.backend
        result["bench_iters"] = len(timing.times_ms)
        flops = (
            2.0 * int(shape["B"]) * int(shape["Q"]) * int(shape["M"]) * int(shape["D"])
        )
        result["tflops"] = flops / timing.median_ms / 1e9
        result["qps"] = int(shape["B"]) * int(shape["Q"]) / (timing.median_ms / 1000.0)
        result["baseline_name"] = shape["recorded"]["baseline_name"]
        result["baseline_ms"] = float(shape["recorded"]["baseline_ms"])
        result["speedup_vs_baseline"] = result["baseline_ms"] / timing.median_ms

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for flashlib_cake_knn_build.knn_build"
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=sorted(SHAPES),
        help="Shape label to run. Repeatable.",
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
        help="Skip reference correctness checks.",
    )
    parser.add_argument(
        "--no-benchmark", action="store_true", help="Skip CUPTI timing."
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path for JSON output."
    )
    args = parser.parse_args()

    selected = args.shape or list(SHAPES)
    payload: dict[str, Any] = {
        "api": "flashlib_cake_knn_build.knn_build",
        "baseline_name": "Cake-recorded FlashLib baseline",
        "shapes": {name: SHAPES[name] for name in selected},
        "metadata_only": bool(args.metadata_only),
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
            )
            for name in selected
        ]

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
