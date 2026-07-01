from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._runtime import CUDAKernel


@dataclass(frozen=True)
class KernelSpec:
    name: str
    symbol: str
    source: str
    threads: int
    shared_mem_bytes: int
    cluster_dims: tuple[int, int, int]
    launch_mode: str
    parameters: tuple[dict[str, str], ...]
    specializations: dict[str, int | str]

    @staticmethod
    def from_manifest(entry: dict[str, Any]) -> "KernelSpec":
        return KernelSpec(
            name=entry["name"],
            symbol=entry["symbol"],
            source=entry["source"],
            threads=int(entry["threads"]),
            shared_mem_bytes=int(entry["shared_mem_bytes"]),
            cluster_dims=tuple(int(v) for v in entry["cluster_dims"]),
            launch_mode=entry["launch_mode"],
            parameters=tuple(entry["parameters"]),
            specializations=dict(entry.get("specializations", {})),
        )


class ExportedKernel:
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self._compiled: dict[tuple[str | None, tuple[str, ...]], "CUDAKernel"] = {}

    @property
    def parameters(self) -> tuple[dict[str, str], ...]:
        return self.spec.parameters

    def source_text(self) -> str:
        package = __package__ or __name__.rpartition(".")[0]
        return resources.files(package).joinpath(self.spec.source).read_text(encoding="utf-8")

    def compile(self, *, arch: str | None = None, options: list[str] | None = None) -> "CUDAKernel":
        key = (arch, tuple(options or ()))
        kernel = self._compiled.get(key)
        if kernel is None:
            from ._runtime import CUDAKernel, compile_cuda

            cubin = compile_cuda(self.source_text(), arch=arch, name=f"{self.spec.name}.cu", options=options)
            kernel = CUDAKernel(cubin, self.spec.symbol)
            self._compiled[key] = kernel
        return kernel

    def launch(
        self,
        *args,
        grid: tuple[int, int, int],
        block: tuple[int, int, int] | None = None,
        shared_mem: int | None = None,
        stream=None,
        timeout_ms: float | None = None,
        arch: str | None = None,
        options: list[str] | None = None,
    ) -> None:
        if len(args) != len(self.spec.parameters):
            expected = ", ".join(p["name"] for p in self.spec.parameters)
            raise TypeError(f"{self.spec.name} expects {len(self.spec.parameters)} args ({expected}), got {len(args)}")
        if block is None:
            block = (self.spec.threads, 1, 1)
        if shared_mem is None:
            shared_mem = self.spec.shared_mem_bytes
        arg_types = [p["ctype"] for p in self.spec.parameters]
        kernel = self.compile(arch=arch, options=options)
        if self.spec.launch_mode == "cluster":
            kernel.launch_cluster(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                cluster_dims=self.spec.cluster_dims,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
            )
        elif self.spec.launch_mode == "cooperative":
            kernel.launch_cooperative(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
            )
        else:
            kernel.launch(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
            )


def _load_manifest() -> dict[str, Any]:
    package = __package__ or __name__.rpartition(".")[0]
    return json.loads(resources.files(package).joinpath("manifest.json").read_text(encoding="utf-8"))


_MANIFEST = _load_manifest()
KERNELS = {entry["name"]: ExportedKernel(KernelSpec.from_manifest(entry)) for entry in _MANIFEST["kernels"]}


def get_kernel(name: str) -> ExportedKernel:
    try:
        return KERNELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise KeyError(f"Unknown exported kernel {name!r}. Available: {available}") from exc


search_knn_search_warp_direct_v1 = get_kernel('search_knn_search_warp_direct_v1')


def launch_search_knn_search_warp_direct_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_direct_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_knn_search_warp_split_partial_v1 = get_kernel('search_knn_search_warp_split_partial_v1')


def launch_search_knn_search_warp_split_partial_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_split_partial_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_knn_search_warp_split_merge_v1 = get_kernel('search_knn_search_warp_split_merge_v1')


def launch_search_knn_search_warp_split_merge_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_split_merge_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_q1_knn_search_q1_tile_reduce_partial_v1 = get_kernel('search_q1_knn_search_q1_tile_reduce_partial_v1')


def launch_search_q1_knn_search_q1_tile_reduce_partial_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_q1_knn_search_q1_tile_reduce_partial_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_q1_knn_search_q1_tile_reduce_merge_v1 = get_kernel('search_q1_knn_search_q1_tile_reduce_merge_v1')


def launch_search_q1_knn_search_q1_tile_reduce_merge_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_q1_knn_search_q1_tile_reduce_merge_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_base_ir = get_kernel('knn_search_base_ir')


def launch_knn_search_base_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_base_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_b2_q128_ir = get_kernel('knn_search_b2_q128_ir')


def launch_knn_search_b2_q128_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_b2_q128_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_blockm640_ir = get_kernel('knn_search_blockm640_ir')


def launch_knn_search_blockm640_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_blockm640_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_blockm640_merge_ir = get_kernel('knn_search_blockm640_merge_ir')


def launch_knn_search_blockm640_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_blockm640_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_b2_q128_merge_ir = get_kernel('knn_search_b2_q128_merge_ir')


def launch_knn_search_b2_q128_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_b2_q128_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_blockm896_ir = get_kernel('knn_search_blockm896_ir')


def launch_knn_search_blockm896_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_blockm896_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_blockm896_merge_ir = get_kernel('knn_search_blockm896_merge_ir')


def launch_knn_search_blockm896_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_blockm896_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_q3_partial_ir = get_kernel('knn_search_q3_partial_ir')


def launch_knn_search_q3_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_q3_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_q3_merge_ir = get_kernel('knn_search_q3_merge_ir')


def launch_knn_search_q3_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_q3_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_scalar_capacity_ir = get_kernel('knn_search_scalar_capacity_ir')


def launch_knn_search_scalar_capacity_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_scalar_capacity_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_k1_merge8_partial_ir = get_kernel('knn_search_k1_merge8_partial_ir')


def launch_knn_search_k1_merge8_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_k1_merge8_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_k1_merge8_merge_ir = get_kernel('knn_search_k1_merge8_merge_ir')


def launch_knn_search_k1_merge8_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_k1_merge8_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_q1_m262144_ir = get_kernel('knn_search_q1_m262144_ir')


def launch_knn_search_q1_m262144_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_q1_m262144_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_q1_m262144_merge_ir = get_kernel('knn_search_q1_m262144_merge_ir')


def launch_knn_search_q1_m262144_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_q1_m262144_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_b2_k64_partial_ir = get_kernel('knn_search_b2_k64_partial_ir')


def launch_knn_search_b2_k64_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_b2_k64_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_b2_k64_group_merge_ir = get_kernel('knn_search_b2_k64_group_merge_ir')


def launch_knn_search_b2_k64_group_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_b2_k64_group_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_b2_k64_final_merge_ir = get_kernel('knn_search_b2_k64_final_merge_ir')


def launch_knn_search_b2_k64_final_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_b2_k64_final_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_d384_q256_partial_ir = get_kernel('knn_search_d384_q256_partial_ir')


def launch_knn_search_d384_q256_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_d384_q256_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_d384_q256_merge_ir = get_kernel('knn_search_d384_q256_merge_ir')


def launch_knn_search_d384_q256_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_d384_q256_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_k64_q256_partial_ir = get_kernel('knn_search_k64_q256_partial_ir')


def launch_knn_search_k64_q256_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_k64_q256_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_lowd_d256_ir = get_kernel('knn_search_lowd_d256_ir')


def launch_knn_search_lowd_d256_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_lowd_d256_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_lowd_d256_merge_ir = get_kernel('knn_search_lowd_d256_merge_ir')


def launch_knn_search_lowd_d256_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_lowd_d256_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_lowd_d256_k64_partial_ir = get_kernel('knn_search_lowd_d256_k64_partial_ir')


def launch_knn_search_lowd_d256_k64_partial_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_lowd_d256_k64_partial_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_lowd_d256_k64_merge_ir = get_kernel('knn_search_lowd_d256_k64_merge_ir')


def launch_knn_search_lowd_d256_k64_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_lowd_d256_k64_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_search_lowd_dbscan_ir = get_kernel('knn_search_lowd_dbscan_ir')


def launch_knn_search_lowd_dbscan_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_search_lowd_dbscan_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )
