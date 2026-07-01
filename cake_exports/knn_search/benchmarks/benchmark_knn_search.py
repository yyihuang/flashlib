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

from flashlib_cake_knn_search._benchmark import bench_gpu_time, require_cupti  # noqa: E402


SHAPES: dict[str, dict[str, Any]] = {
    "search_direct_q4_m256_k5": {
        "B": 1,
        "Q": 4,
        "M": 256,
        "D": 128,
        "K": 5,
        "seed": 43,
    },
    "search_split_q16_m4096_k10": {
        "B": 1,
        "Q": 16,
        "M": 4096,
        "D": 128,
        "K": 10,
        "seed": 47,
    },
    "search_q1_m131072_k10": {
        "B": 1,
        "Q": 1,
        "M": 131072,
        "D": 128,
        "K": 10,
        "seed": 53,
    },
}


def _make_inputs(shape: dict[str, Any]):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(shape["seed"]))
    query = torch.randn(
        (int(shape["B"]), int(shape["Q"]), int(shape["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    database = torch.randn(
        (int(shape["B"]), int(shape["M"]), int(shape["D"])),
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
    return float(
        matches.to(got_indices.device, dtype=got_indices.float().dtype).mean().item()
    )


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


def _run_shape(
    name: str,
    shape: dict[str, Any],
    *,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_knn_search import knn_search

    query, database = _make_inputs(shape)
    k = int(shape["K"])
    out = knn_search(query, database, k, arch=arch)
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
        _, ref_indices = _reference_topk(query, database, k)
        torch.cuda.synchronize()
        exact_dists = _distances_for_indices(query, database, out[1])
        result["recall"] = _recall(out[1], ref_indices)
        result["max_abs_dist_error"] = float((out[0] - exact_dists).abs().max().item())
        result["correct"] = bool(
            result["recall"] >= 0.999 and result["max_abs_dist_error"] <= 1.0e-2
        )

    if benchmark:
        timing = bench_gpu_time(
            lambda: knn_search(query, database, k, arch=arch), cold_l2=True
        )
        result["kernel_ms"] = timing.median_ms
        result["timing_backend"] = timing.backend
        result["bench_iters"] = len(timing.times_ms)
        flops = (
            2.0 * int(shape["B"]) * int(shape["Q"]) * int(shape["M"]) * int(shape["D"])
        )
        result["tflops"] = flops / timing.median_ms / 1e9
        result["qps"] = int(shape["B"]) * int(shape["Q"]) / (timing.median_ms / 1000.0)

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
        "api": "flashlib_cake_knn_search.knn_search",
        "shapes": {name: SHAPES[name] for name in selected},
        "metadata_only": bool(args.metadata_only),
    }
    if args.metadata_only:
        payload["results"] = []
    else:
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
