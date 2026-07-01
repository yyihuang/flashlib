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


stage1_k32_unordered_ir = get_kernel('stage1_k32_unordered_ir')


def launch_stage1_k32_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k32_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k12_ir = get_kernel('stage1_k12_ir')


def launch_stage1_k12_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k12_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k16_ir = get_kernel('stage1_k16_ir')


def launch_stage1_k16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k20_ir = get_kernel('stage1_k20_ir')


def launch_stage1_k20_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k20_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k25_ir = get_kernel('stage1_k25_ir')


def launch_stage1_k25_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k25_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k30_ir = get_kernel('stage1_k30_ir')


def launch_stage1_k30_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k30_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_ir = get_kernel('stage1_ir')


def launch_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k20_unordered_ir = get_kernel('stage1_k20_unordered_ir')


def launch_stage1_k20_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k20_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k30_unordered_ir = get_kernel('stage1_k30_unordered_ir')


def launch_stage1_k30_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k30_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache = get_kernel('knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache')


def launch_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k12_ir = get_kernel('merge_k12_ir')


def launch_merge_k12_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k12_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k16_ir = get_kernel('merge_k16_ir')


def launch_merge_k16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k20_ir = get_kernel('merge_k20_ir')


def launch_merge_k20_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k20_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k25_ir = get_kernel('merge_k25_ir')


def launch_merge_k25_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k25_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_ir = get_kernel('merge_k30_ir')


def launch_merge_k30_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_ir = get_kernel('merge_ir')


def launch_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k32_unordered_ir = get_kernel('merge_k32_unordered_ir')


def launch_merge_k32_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k32_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k20_unordered_ir = get_kernel('merge_k20_unordered_ir')


def launch_merge_k20_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k20_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_unordered_ir = get_kernel('merge_k30_unordered_ir')


def launch_merge_k30_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_s8_ir = get_kernel('merge_k30_s8_ir')


def launch_merge_k30_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k12_s8_ir = get_kernel('merge_k12_s8_ir')


def launch_merge_k12_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k12_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_stage1_ir = get_kernel('k5t64_stage1_ir')


def launch_k5t64_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_ir = get_kernel('k5t64_merge_ir')


def launch_k5t64_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_k10_s4_cache_ir = get_kernel('k5t64_merge_k10_s4_cache_ir')


def launch_k5t64_merge_k10_s4_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_k10_s4_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_k10_s7_cache_ir = get_kernel('k5t64_merge_k10_s7_cache_ir')


def launch_k5t64_merge_k10_s7_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_k10_s7_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_stage1_ir = get_kernel('k10t32_stage1_ir')


def launch_k10t32_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_ir = get_kernel('k10t32_merge_ir')


def launch_k10t32_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_k10_s4_cache_ir = get_kernel('k10t32_merge_k10_s4_cache_ir')


def launch_k10t32_merge_k10_s4_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_k10_s4_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_k10_s7_cache_ir = get_kernel('k10t32_merge_k10_s7_cache_ir')


def launch_k10t32_merge_k10_s7_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_k10_s7_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10root_stage1 = get_kernel('k10root_stage1')


def launch_k10root_stage1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10root_stage1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_d64_build_aa88_v2_stage1_d64_split_ir = get_kernel('knn_build_d64_build_aa88_v2_stage1_d64_split_ir')


def launch_knn_build_d64_build_aa88_v2_stage1_d64_split_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_d64_build_aa88_v2_stage1_d64_split_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_d64_build_aa88_v2_merge_generic_ir = get_kernel('knn_build_d64_build_aa88_v2_merge_generic_ir')


def launch_knn_build_d64_build_aa88_v2_merge_generic_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_d64_build_aa88_v2_merge_generic_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_d64_build_aa88_v2_merge_ir = get_kernel('knn_build_d64_build_aa88_v2_merge_ir')


def launch_knn_build_d64_build_aa88_v2_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_d64_build_aa88_v2_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_d64_build_aa88_v2_merge_k10_s4_ir = get_kernel('knn_build_d64_build_aa88_v2_merge_k10_s4_ir')


def launch_knn_build_d64_build_aa88_v2_merge_k10_s4_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_d64_build_aa88_v2_merge_k10_s4_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_73a9_v1_stage1_d64_split_ir = get_kernel('knn_build_dim_midk_73a9_v1_stage1_d64_split_ir')


def launch_knn_build_dim_midk_73a9_v1_stage1_d64_split_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_73a9_v1_stage1_d64_split_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_73a9_v1_merge_generic_ir = get_kernel('knn_build_dim_midk_73a9_v1_merge_generic_ir')


def launch_knn_build_dim_midk_73a9_v1_merge_generic_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_73a9_v1_merge_generic_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_stage1_k24_s8_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_stage1_k24_s8_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_stage1_k24_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_stage1_k24_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_s8_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_s8_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_unordered_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_unordered_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_merge_k24_s8_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_merge_k24_s8_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_merge_k24_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_merge_k24_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_merge_k28_s8_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_merge_k28_s8_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_merge_k28_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_merge_k28_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k24k28_v1_merge_k28_unordered_ir = get_kernel('knn_build_dim_midk_bad5_k24k28_v1_merge_k28_unordered_ir')


def launch_knn_build_dim_midk_bad5_k24k28_v1_merge_k28_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k24k28_v1_merge_k28_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k64split8_v1_stage1_k64_s8_tailinf_ir = get_kernel('knn_build_dim_midk_bad5_k64split8_v1_stage1_k64_s8_tailinf_ir')


def launch_knn_build_dim_midk_bad5_k64split8_v1_stage1_k64_s8_tailinf_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k64split8_v1_stage1_k64_s8_tailinf_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_k64split8_v1_merge_k64_s8_warp_select_ir = get_kernel('knn_build_dim_midk_bad5_k64split8_v1_merge_k64_s8_warp_select_ir')


def launch_knn_build_dim_midk_bad5_k64split8_v1_merge_k64_s8_warp_select_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_k64split8_v1_merge_k64_s8_warp_select_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k24_s8_ir = get_kernel('knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k24_s8_ir')


def launch_knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k24_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k24_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k28_s8_ir = get_kernel('knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k28_s8_ir')


def launch_knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k28_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k28_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_midkcleanup_v1_merge_k24_s8_ir = get_kernel('knn_build_dim_midk_bad5_midkcleanup_v1_merge_k24_s8_ir')


def launch_knn_build_dim_midk_bad5_midkcleanup_v1_merge_k24_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_midkcleanup_v1_merge_k24_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_bad5_midkcleanup_v1_merge_k28_s8_ir = get_kernel('knn_build_dim_midk_bad5_midkcleanup_v1_merge_k28_s8_ir')


def launch_knn_build_dim_midk_bad5_midkcleanup_v1_merge_k28_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_bad5_midkcleanup_v1_merge_k28_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_df2f_v1_stage1_d256_split_ir = get_kernel('knn_build_dim_midk_df2f_v1_stage1_d256_split_ir')


def launch_knn_build_dim_midk_df2f_v1_stage1_d256_split_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_df2f_v1_stage1_d256_split_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_df2f_v1_stage1_fp16_split_ir = get_kernel('knn_build_dim_midk_df2f_v1_stage1_fp16_split_ir')


def launch_knn_build_dim_midk_df2f_v1_stage1_fp16_split_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_df2f_v1_stage1_fp16_split_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_df2f_v1_merge_generic_ir = get_kernel('knn_build_dim_midk_df2f_v1_merge_generic_ir')


def launch_knn_build_dim_midk_df2f_v1_merge_generic_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_df2f_v1_merge_generic_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_f8c3_q4096k64split_v1_stage1_k64_tailinf_ir = get_kernel('knn_build_dim_midk_f8c3_q4096k64split_v1_stage1_k64_tailinf_ir')


def launch_knn_build_dim_midk_f8c3_q4096k64split_v1_stage1_k64_tailinf_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_f8c3_q4096k64split_v1_stage1_k64_tailinf_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s8_ir = get_kernel('knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s8_ir')


def launch_knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s12_ir = get_kernel('knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s12_ir')


def launch_knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s12_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s12_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s16_ir = get_kernel('knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s16_ir')


def launch_knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1 = get_kernel('knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1')


def launch_knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1 = get_kernel('knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1')


def launch_knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_7399_d15e_df2f_full55_v1 = get_kernel('knn_build_dispatch_7399_d15e_df2f_full55_v1')


def launch_knn_build_dispatch_7399_d15e_df2f_full55_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_7399_d15e_df2f_full55_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_7399_d15e_full55_v1 = get_kernel('knn_build_dispatch_7399_d15e_full55_v1')


def launch_knn_build_dispatch_7399_d15e_full55_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_7399_d15e_full55_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_b6d4_d15e_fd02_v1 = get_kernel('knn_build_dispatch_b6d4_d15e_fd02_v1')


def launch_knn_build_dispatch_b6d4_d15e_fd02_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_b6d4_d15e_fd02_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_d64_fdd7_e3de_v1 = get_kernel('knn_build_dispatch_d64_fdd7_e3de_v1')


def launch_knn_build_dispatch_d64_fdd7_e3de_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_d64_fdd7_e3de_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_e3de_9138_bcb3_4247_v1 = get_kernel('knn_build_dispatch_e3de_9138_bcb3_4247_v1')


def launch_knn_build_dispatch_e3de_9138_bcb3_4247_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_e3de_9138_bcb3_4247_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_rag_seed_portfolio_8700_v1 = get_kernel('knn_build_dispatch_rag_seed_portfolio_8700_v1')


def launch_knn_build_dispatch_rag_seed_portfolio_8700_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_rag_seed_portfolio_8700_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_397b_v1 = get_kernel('knn_build_dispatch_selected_portfolio_397b_v1')


def launch_knn_build_dispatch_selected_portfolio_397b_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_397b_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_4a72_v1 = get_kernel('knn_build_dispatch_selected_portfolio_4a72_v1')


def launch_knn_build_dispatch_selected_portfolio_4a72_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_4a72_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_e51c_v1 = get_kernel('knn_build_dispatch_selected_portfolio_e51c_v1')


def launch_knn_build_dispatch_selected_portfolio_e51c_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_e51c_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_f16b_v1 = get_kernel('knn_build_dispatch_selected_portfolio_f16b_v1')


def launch_knn_build_dispatch_selected_portfolio_f16b_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_f16b_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_f552_v1 = get_kernel('knn_build_dispatch_selected_portfolio_f552_v1')


def launch_knn_build_dispatch_selected_portfolio_f552_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_f552_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_f853_v1 = get_kernel('knn_build_dispatch_selected_portfolio_f853_v1')


def launch_knn_build_dispatch_selected_portfolio_f853_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_f853_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_selected_portfolio_f8c3_v1 = get_kernel('knn_build_dispatch_selected_portfolio_f8c3_v1')


def launch_knn_build_dispatch_selected_portfolio_f8c3_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_selected_portfolio_f8c3_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_dispatch_split72_4e09_de1a_3dc7_v48 = get_kernel('knn_build_dispatch_split72_4e09_de1a_3dc7_v48')


def launch_knn_build_dispatch_split72_4e09_de1a_3dc7_v48(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_dispatch_split72_4e09_de1a_3dc7_v48.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_large_square_k20k32_a989_v1 = get_kernel('knn_build_large_square_k20k32_a989_v1')


def launch_knn_build_large_square_k20k32_a989_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_large_square_k20k32_a989_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_large_square_k32_8a83_v1_merge_k32_s2_warp_select_ir = get_kernel('knn_build_large_square_k32_8a83_v1_merge_k32_s2_warp_select_ir')


def launch_knn_build_large_square_k32_8a83_v1_merge_k32_s2_warp_select_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_large_square_k32_8a83_v1_merge_k32_s2_warp_select_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_large_square_k32_8a83_v1_stage1_k32_split2_ir = get_kernel('knn_build_large_square_k32_8a83_v1_stage1_k32_split2_ir')


def launch_knn_build_large_square_k32_8a83_v1_stage1_k32_split2_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_large_square_k32_8a83_v1_stage1_k32_split2_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_large_tail_frontier_6a73_v1 = get_kernel('knn_build_large_tail_frontier_6a73_v1')


def launch_knn_build_large_tail_frontier_6a73_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_large_tail_frontier_6a73_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_stage1_q512_lowk_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_stage1_q512_lowk_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_stage1_q512_lowk_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_stage1_q512_lowk_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_merge_q512_generic_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_merge_q512_generic_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_merge_q512_generic_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_merge_q512_generic_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_stage1_q1024_k16_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_stage1_q1024_k16_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_stage1_q1024_k16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_stage1_q1024_k16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s4_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s4_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s4_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s4_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s8_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s8_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s16_ir = get_kernel('knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s16_ir')


def launch_knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_stage1_k96_over64_ir = get_kernel('knn_build_over64_k96_a2f8_v1_stage1_k96_over64_ir')


def launch_knn_build_over64_k96_a2f8_v1_stage1_k96_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_stage1_k96_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_merge_k96_over64_ir = get_kernel('knn_build_over64_k96_a2f8_v1_merge_k96_over64_ir')


def launch_knn_build_over64_k96_a2f8_v1_merge_k96_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_merge_k96_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_knn_build_k96_stage1_sort4_chunked = get_kernel('knn_build_over64_k96_a2f8_v1_knn_build_k96_stage1_sort4_chunked')


def launch_knn_build_over64_k96_a2f8_v1_knn_build_k96_stage1_sort4_chunked(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_knn_build_k96_stage1_sort4_chunked.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_stage1_k96_sort4_chunked_over64_ir = get_kernel('knn_build_over64_k96_a2f8_v1_stage1_k96_sort4_chunked_over64_ir')


def launch_knn_build_over64_k96_a2f8_v1_stage1_k96_sort4_chunked_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_stage1_k96_sort4_chunked_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_knn_build_k96_merge_s8_unordered_chunkprefill = get_kernel('knn_build_over64_k96_a2f8_v1_knn_build_k96_merge_s8_unordered_chunkprefill')


def launch_knn_build_over64_k96_a2f8_v1_knn_build_k96_merge_s8_unordered_chunkprefill(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_knn_build_k96_merge_s8_unordered_chunkprefill.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a2f8_v1_merge_k96_s8_chunkprefill_over64_ir = get_kernel('knn_build_over64_k96_a2f8_v1_merge_k96_s8_chunkprefill_over64_ir')


def launch_knn_build_over64_k96_a2f8_v1_merge_k96_s8_chunkprefill_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a2f8_v1_merge_k96_s8_chunkprefill_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a989_v1_stage1_k96_over64_ir = get_kernel('knn_build_over64_k96_a989_v1_stage1_k96_over64_ir')


def launch_knn_build_over64_k96_a989_v1_stage1_k96_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a989_v1_stage1_k96_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a989_v1_merge_k96_over64_ir = get_kernel('knn_build_over64_k96_a989_v1_merge_k96_over64_ir')


def launch_knn_build_over64_k96_a989_v1_merge_k96_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a989_v1_merge_k96_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a989_v1_knn_build_k96_merge_s8_unordered_chunkprefill = get_kernel('knn_build_over64_k96_a989_v1_knn_build_k96_merge_s8_unordered_chunkprefill')


def launch_knn_build_over64_k96_a989_v1_knn_build_k96_merge_s8_unordered_chunkprefill(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a989_v1_knn_build_k96_merge_s8_unordered_chunkprefill.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_over64_k96_a989_v1_merge_k96_s8_chunkprefill_over64_ir = get_kernel('knn_build_over64_k96_a989_v1_merge_k96_s8_chunkprefill_over64_ir')


def launch_knn_build_over64_k96_a989_v1_merge_k96_s8_chunkprefill_over64_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_over64_k96_a989_v1_merge_k96_s8_chunkprefill_over64_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_frontier_4b5c_v1 = get_kernel('knn_build_rag_frontier_4b5c_v1')


def launch_knn_build_rag_frontier_4b5c_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_frontier_4b5c_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_frontier_4fbf_v7_stage1_k32_tailinf_ir = get_kernel('knn_build_rag_frontier_4fbf_v7_stage1_k32_tailinf_ir')


def launch_knn_build_rag_frontier_4fbf_v7_stage1_k32_tailinf_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_frontier_4fbf_v7_stage1_k32_tailinf_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_frontier_4fbf_v7_fused_merge_generalized_ir = get_kernel('knn_build_rag_frontier_4fbf_v7_fused_merge_generalized_ir')


def launch_knn_build_rag_frontier_4fbf_v7_fused_merge_generalized_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_frontier_4fbf_v7_fused_merge_generalized_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_frontier_7399_v1 = get_kernel('knn_build_rag_frontier_7399_v1')


def launch_knn_build_rag_frontier_7399_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_frontier_7399_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_4a72_v1_fused_merge_ir = get_kernel('knn_build_rag_microbatch_4a72_v1_fused_merge_ir')


def launch_knn_build_rag_microbatch_4a72_v1_fused_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_4a72_v1_fused_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_4a72_v1_ir = get_kernel('knn_build_rag_microbatch_4a72_v1_ir')


def launch_knn_build_rag_microbatch_4a72_v1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_4a72_v1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_4a72_v2_stage1_cta1_ir = get_kernel('knn_build_rag_microbatch_4a72_v2_stage1_cta1_ir')


def launch_knn_build_rag_microbatch_4a72_v2_stage1_cta1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_4a72_v2_stage1_cta1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_4a72_v2_fused_merge_ir = get_kernel('knn_build_rag_microbatch_4a72_v2_fused_merge_ir')


def launch_knn_build_rag_microbatch_4a72_v2_fused_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_4a72_v2_fused_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_4a72_v2_ir = get_kernel('knn_build_rag_microbatch_4a72_v2_ir')


def launch_knn_build_rag_microbatch_4a72_v2_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_4a72_v2_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rag_microbatch_m64_d4f7_v1 = get_kernel('knn_build_rag_microbatch_m64_d4f7_v1')


def launch_knn_build_rag_microbatch_m64_d4f7_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rag_microbatch_m64_d4f7_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s72_k10_ir = get_kernel('knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s72_k10_ir')


def launch_knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s72_k10_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s72_k10_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s74_k10_ir = get_kernel('knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s74_k10_ir')


def launch_knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s74_k10_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s74_k10_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_ragonline_mbucket_aa88_q1m_v3_ir = get_kernel('knn_build_ragonline_mbucket_aa88_q1m_v3_ir')


def launch_knn_build_ragonline_mbucket_aa88_q1m_v3_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_ragonline_mbucket_aa88_q1m_v3_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_d64_cf49_v3_stage1_d64_split_ir = get_kernel('knn_build_rect_d64_cf49_v3_stage1_d64_split_ir')


def launch_knn_build_rect_d64_cf49_v3_stage1_d64_split_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_d64_cf49_v3_stage1_d64_split_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_d64_cf49_v3_merge_generic_ir = get_kernel('knn_build_rect_d64_cf49_v3_merge_generic_ir')


def launch_knn_build_rect_d64_cf49_v3_merge_generic_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_d64_cf49_v3_merge_generic_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_d64_cf49_v3_merge_s16_cached_ir = get_kernel('knn_build_rect_d64_cf49_v3_merge_s16_cached_ir')


def launch_knn_build_rect_d64_cf49_v3_merge_s16_cached_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_d64_cf49_v3_merge_s16_cached_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s8_cache_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s8_cache_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s8_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s8_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s12_cache_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s12_cache_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s12_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s12_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s16_cache_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s16_cache_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s16_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s16_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s24_cache_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s24_cache_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s24_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s24_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s32_cache_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s32_cache_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s32_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s32_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_intermediate_frontier_6a73_4452_v2_ir = get_kernel('knn_build_rect_intermediate_frontier_6a73_4452_v2_ir')


def launch_knn_build_rect_intermediate_frontier_6a73_4452_v2_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_intermediate_frontier_6a73_4452_v2_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s8_cache_ir = get_kernel('knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s8_cache_ir')


def launch_knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s8_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s8_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s16_cache_ir = get_kernel('knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s16_cache_ir')


def launch_knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s16_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s16_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s32_cache_ir = get_kernel('knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s32_cache_ir')


def launch_knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s32_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s32_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_rect_smallq_largem_ff59_d15e_v1_ir = get_kernel('knn_build_rect_smallq_largem_ff59_d15e_v1_ir')


def launch_knn_build_rect_smallq_largem_ff59_d15e_v1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_rect_smallq_largem_ff59_d15e_v1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )
