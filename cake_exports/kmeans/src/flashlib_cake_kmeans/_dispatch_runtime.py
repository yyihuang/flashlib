from __future__ import annotations

import json
from importlib import resources
from types import SimpleNamespace

from .kernels import get_kernel


def _decode_capture(value):
    if isinstance(value, dict) and set(value) == {"__ir__"}:
        return _ir_proxy(value["__ir__"])
    if isinstance(value, dict) and set(value) == {"__kernel__"}:
        return DispatchKernel(value["__kernel__"])
    if isinstance(value, dict) and set(value) == {"__kernel_source__"}:
        return value["__kernel_source__"]
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_capture(item) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {key: _decode_capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_capture(item) for item in value]
    return value


class _IRProxy:
    def __init__(self, name, threads=256):
        self.name = name.rpartition(":")[2]
        self.threads = int(threads)
        self.computed_smem_bytes = 0
        self.grid = SimpleNamespace(cluster_dims=(1, 1, 1), cta_group=1)


def _ir_proxy(name, threads=256):
    return _IRProxy(name, threads)


class DispatchKernel:
    def __init__(self, alias, symbol=None):
        self.exported = get_kernel(alias)
        self.symbol = symbol or self.exported.spec.symbol

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def launch(self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs):
        self.exported.launch(
            *args, grid=grid, block=block, shared_mem=shared_mem, stream=stream,
            timeout_ms=timeout_ms, options=["--use_fast_math"],
        )

    def launch_cluster(
        self, *, grid, block, args, cluster_dims, shared_mem=0, stream=None,
        timeout_ms=None, **kwargs
    ):
        kernel = self.exported.compile(options=["--use_fast_math"])
        kernel.launch_cluster(
            grid=grid, block=block, args=list(args),
            arg_types=[item["ctype"] for item in self.exported.spec.parameters],
            cluster_dims=cluster_dims, shared_mem=shared_mem, stream=stream,
            timeout_ms=timeout_ms,
        )

    def launch_cooperative(
        self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs
    ):
        kernel = self.exported.compile(options=["--use_fast_math"])
        kernel.launch_cooperative(
            grid=grid, block=block, args=list(args),
            arg_types=[item["ctype"] for item in self.exported.spec.parameters],
            shared_mem=shared_mem, stream=stream, timeout_ms=timeout_ms,
        )


CUDAKernel = DispatchKernel


def compile_cuda(source, **kwargs):
    return source


def detect_gpu_arch():
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def _cuda_include_dirs():
    return []


def generate_kernel(ir, **kwargs):
    raise RuntimeError(f"uncaptured dispatcher specialization for {ir.name}")


def generate_kernel_bundle(*args, **kwargs):
    raise RuntimeError("uncaptured dispatcher bundle specialization")


def _all_shapes():
    package = __package__ or __name__.rpartition(".")[0]
    text = resources.files(package).joinpath("_dispatch_shapes.json").read_text(encoding="utf-8")
    return json.loads(text)


def select_named_shapes(labels):
    labels = [labels] if isinstance(labels, str) else list(labels)
    by_label = {row["label"]: row for row in _all_shapes()}
    return [by_label[label] for label in labels]


def evaluate(*args, **kwargs):
    raise RuntimeError("Cake eval harness is not part of the standalone runtime")
