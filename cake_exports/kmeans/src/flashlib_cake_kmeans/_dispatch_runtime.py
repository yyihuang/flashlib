from __future__ import annotations
_KERNEL_ALIAS_BY_IR_NAME = {'flash_kmeans_assign_lowdim_pack_e50c_v1': 'dispatch_kernel_0000', 'flash_kmeans_assign_lowdim_e50c_v1': 'dispatch_kernel_0001', 'flash_kmeans_assign_cleanroom_tcgen05_v10': 'dispatch_kernel_0002', 'flash_kmeans_assign_d160_pad192_pack_f9b2_v1': 'dispatch_kernel_0004', 'flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1': 'dispatch_kernel_0005', 'flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1': 'dispatch_kernel_0008', 'flash_kmeans_assign_highd_splitd_6fcf_v1': 'dispatch_kernel_0009', 'flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1': 'dispatch_kernel_0011', 'flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1': 'dispatch_kernel_0012', 'flash_kmeans_assign_microdim_pack_6cd2_v1': 'dispatch_kernel_0013', 'flash_kmeans_assign_microdim_6cd2_v1': 'dispatch_kernel_0014', 'flash_kmeans_assign_gap_pad_pack_v1': 'dispatch_kernel_0015', 'flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1': 'dispatch_kernel_0016', 'flash_kmeans_assign_cleanroom_tcgen05_v15': 'dispatch_kernel_0048', 'flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1': 'dispatch_kernel_0059', 'flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1': 'dispatch_kernel_0060', 'flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1': 'dispatch_kernel_0101', 'flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1': 'dispatch_kernel_0109', 'flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1': 'dispatch_kernel_0118', 'flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1': 'dispatch_kernel_0119', 'flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1': 'dispatch_kernel_0121', 'flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1': 'dispatch_kernel_0122', 'flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4': 'dispatch_kernel_0237', 'flash_kmeans_assign_d288_exactd_a532_v1': 'dispatch_kernel_0244', 'flash_kmeans_assign_d288_splitk_cta_0438_v1_partial': 'dispatch_kernel_0245', 'flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce': 'dispatch_kernel_0246', 'flash_kmeans_assign_d480_splitk_partial_d32k256_v1': 'dispatch_kernel_0266', 'flash_kmeans_assign_d480_splitk_reduce_d32k256_v1': 'dispatch_kernel_0267', 'flash_kmeans_assign_highd_splitk_partial_8de8_v1': 'dispatch_kernel_0285', 'flash_kmeans_assign_highd_splitk_reduce_8de8_v1': 'dispatch_kernel_0286', 'flash_kmeans_assign_microdim_direct_9c0d_v1': 'dispatch_kernel_0338', 'flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2': 'dispatch_kernel_0353'}
_KERNEL_ALIAS_BY_REQUEST = {'{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_lowdim_pack_e50c_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0000', '{"computed_smem_bytes":100352,"constants":[],"ir_name":"flash_kmeans_assign_lowdim_e50c_v1","kwargs":{"smem_bytes":100352,"validate":false},"threads":192}': 'dispatch_kernel_0001', '{"computed_smem_bytes":100352,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_v10","kwargs":{"smem_bytes":100352,"validate":false},"threads":192}': 'dispatch_kernel_0002', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d160_pad192_pack_f9b2_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0004', '{"computed_smem_bytes":149504,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1","kwargs":{"smem_bytes":149504,"validate":false},"threads":192}': 'dispatch_kernel_0005', '{"computed_smem_bytes":198656,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1","kwargs":{"smem_bytes":198656,"validate":false},"threads":192}': 'dispatch_kernel_0008', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitd_6fcf_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0009', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0011', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0012', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_microdim_pack_6cd2_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0013', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_microdim_6cd2_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0014', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_gap_pad_pack_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0015', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0016', '{"computed_smem_bytes":133120,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_v15","kwargs":{"smem_bytes":133120,"validate":false},"threads":192}': 'dispatch_kernel_0048', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0059', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0060', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0101', '{"computed_smem_bytes":198656,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1","kwargs":{"smem_bytes":198656,"validate":false},"threads":192}': 'dispatch_kernel_0109', '{"computed_smem_bytes":75776,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1","kwargs":{"smem_bytes":75776,"validate":false},"threads":192}': 'dispatch_kernel_0118', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":64}': 'dispatch_kernel_0119', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0121', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0122', '{"computed_smem_bytes":22528,"constants":[],"ir_name":"flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4","kwargs":{"smem_bytes":22528,"validate":false},"threads":192}': 'dispatch_kernel_0237', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d288_exactd_a532_v1","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0244', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d288_splitk_cta_0438_v1_partial","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0245', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0246', '{"computed_smem_bytes":22528,"constants":[],"ir_name":"flash_kmeans_assign_d480_splitk_partial_d32k256_v1","kwargs":{"smem_bytes":22528,"validate":false},"threads":192}': 'dispatch_kernel_0266', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d480_splitk_reduce_d32k256_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0267', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_8de8_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0285', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_8de8_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0286', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_microdim_direct_9c0d_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0338', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0353'}

import json
import ctypes
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace as _dataclass_replace
from importlib import resources
from types import SimpleNamespace

from .kernels import get_kernel
from ._runtime import launch_stream_context, resolve_launch_defaults


_DISPATCH_LAUNCH_OPTIONS = ContextVar("dispatch_launch_options", default=(None, None))


@contextmanager
def dispatch_launch_options(*, stream=None, timeout_ms=None):
    token = _DISPATCH_LAUNCH_OPTIONS.set((stream, timeout_ms))
    try:
        yield
    finally:
        _DISPATCH_LAUNCH_OPTIONS.reset(token)


def _resolved_launch_options(stream, timeout_ms):
    default_stream, default_timeout_ms = _DISPATCH_LAUNCH_OPTIONS.get()
    return (
        default_stream if stream is None else stream,
        default_timeout_ms if timeout_ms is None else timeout_ms,
    )


_active_launch_capture = ContextVar("flashlib_active_launch_capture", default=None)


def _replace(value, /, **changes):
    replacer = getattr(value, "__replace__", None)
    if callable(replacer):
        return replacer(**changes)
    return _dataclass_replace(value, **changes)


dc = SimpleNamespace(replace=_replace)


def _import_dispatch_module(short_name):
    return importlib.import_module(f"{__package__}._dispatch.{short_name}")


def _decode_capture(value):
    if isinstance(value, dict) and "__ir__" in value:
        return _ir_proxy(
            value["__ir__"],
            value.get("threads", 256),
            value.get("computed_smem_bytes", 0),
            value.get("cluster_dims", (1, 1, 1)),
            value.get("cta_group", 1),
            value.get("constants", ()),
            value.get("arg_keys", ()),
        )
    if isinstance(value, dict) and set(value) == {"__kernel__"}:
        return DispatchKernel(value["__kernel__"])
    if isinstance(value, dict) and set(value) == {"__kernel_source__"}:
        return value["__kernel_source__"]
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_capture(item) for item in value["__tuple__"])
    if isinstance(value, dict) and set(value) == {"__dict_items__"}:
        return {
            _decode_capture(key): _decode_capture(item)
            for key, item in value["__dict_items__"]
        }
    if isinstance(value, dict):
        return {key: _decode_capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_capture(item) for item in value]
    return value


@dataclass(frozen=True)
class _IRProxy:
    symbol: str
    threads: int = 256
    computed_smem_bytes: int = 0
    constants: tuple = ()
    grid: object = None
    arg_keys: tuple = ()

    def __replace__(self, /, **changes):
        values = {
            "symbol": self.symbol,
            "threads": self.threads,
            "computed_smem_bytes": self.computed_smem_bytes,
            "constants": self.constants,
            "grid": self.grid,
            "arg_keys": self.arg_keys,
        }
        unknown = sorted(set(changes) - set(values))
        if unknown:
            raise TypeError(f"unknown frozen WeaveIR field(s): {unknown}")
        values.update(changes)
        return _IRProxy(**values)


def _ir_proxy(
    name, threads=256, computed_smem_bytes=0, cluster_dims=(1, 1, 1),
    cta_group=1, constants=(), arg_keys=(),
):
    return _IRProxy(
        name.rpartition(":")[2], int(threads), int(computed_smem_bytes),
        tuple(tuple(item) for item in constants),
        SimpleNamespace(cluster_dims=tuple(cluster_dims), cta_group=int(cta_group)),
        tuple(arg_keys),
    )


def pack_kernel_args(schedule, /, **bindings):
    expected = tuple(schedule.arg_keys)
    missing = sorted(set(expected) - set(bindings))
    unexpected = sorted(set(bindings) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"kernel argument bindings do not match frozen WeaveIR.args: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return [bindings[key] for key in expected]


class PreparedKernelSequence:
    def __init__(self, launches, result):
        if not launches:
            raise RuntimeError("prepared semantic route did not capture a CUDA launch")
        self._launches = tuple(launches)
        self._result = result

    @property
    def launch_count(self):
        return len(self._launches)

    def __call__(self, _inputs=None, *, stream=None, timeout_ms=None):
        last = len(self._launches) - 1
        for index, launch in enumerate(self._launches):
            launch.launch(stream=stream, timeout_ms=timeout_ms if index == last else None)
        return self._result


class KernelLaunchCapture:
    def __init__(self, *, stream=None, arch=None):
        self.stream = stream
        self.arch = arch
        self._launches = []

    def add(self, launch):
        self._launches.append(launch)

    def bind(self, result):
        return PreparedKernelSequence(self._launches, result)


@contextmanager
def capture_kernel_launches(*, stream=None, arch=None):
    import torch

    if _active_launch_capture.get() is not None:
        raise RuntimeError("nested kernel launch capture is not supported")
    capture = KernelLaunchCapture(stream=stream, arch=arch)
    token = _active_launch_capture.set(capture)
    try:
        if stream is None:
            yield capture
        else:
            with torch.cuda.stream(stream):
                yield capture
    finally:
        _active_launch_capture.reset(token)


class DispatchKernel:
    def __init__(self, alias, symbol=None):
        self.exported = get_kernel(alias)
        self.symbol = symbol or self.exported.spec.symbol

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def launch(self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add(self.exported.prepare_launch(
                *args, grid=grid, block=block, shared_mem=shared_mem,
                stream=capture.stream if capture.stream is not None else stream,
                arch=capture.arch, options=["--use_fast_math"],
            ))
            return
        self.exported.launch(
            *args, grid=grid, block=block, shared_mem=shared_mem, stream=stream,
            timeout_ms=timeout_ms, options=["--use_fast_math"],
        )

    def launch_cluster(
        self, *, grid, block, args, cluster_dims, shared_mem=0, stream=None,
        timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None, stream=stream, timeout_ms=timeout_ms
        )
        capture = _active_launch_capture.get()
        kernel = self.exported.compile(
            arch=capture.arch if capture is not None and capture.arch is not None else arch,
            options=["--use_fast_math"],
        )
        if capture is not None:
            capture.add(kernel.prepare_launch_cluster(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims, shared_mem=shared_mem,
                stream=capture.stream if capture.stream is not None else stream,
            ))
            return
        with launch_stream_context(stream):
            kernel.launch_cluster(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims, shared_mem=shared_mem, stream=stream,
                timeout_ms=timeout_ms,
            )

    def launch_cooperative(
        self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None, stream=stream, timeout_ms=timeout_ms
        )
        capture = _active_launch_capture.get()
        kernel = self.exported.compile(
            arch=capture.arch if capture is not None and capture.arch is not None else arch,
            options=["--use_fast_math"],
        )
        if capture is not None:
            capture.add(kernel.prepare_launch_cooperative(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem,
                stream=capture.stream if capture.stream is not None else stream,
            ))
            return
        with launch_stream_context(stream):
            kernel.launch_cooperative(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem, stream=stream, timeout_ms=timeout_ms,
            )

    def prepare_launch(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch, kwargs)

    def prepare_launch_cluster(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cluster, kwargs)

    def prepare_launch_cooperative(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cooperative, kwargs)


class _PreparedDispatchLaunch:
    def __init__(self, launch, kwargs):
        self._launch = launch
        self._kwargs = dict(kwargs)

    def launch(self, timeout_ms=None):
        kwargs = dict(self._kwargs)
        if timeout_ms is not None:
            kwargs["timeout_ms"] = timeout_ms
        return self._launch(**kwargs)


CUDAKernel = DispatchKernel


def compile_cuda(source, **kwargs):
    return source


def detect_gpu_arch():
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def _cuda_include_dirs():
    return []


def arch_flag_for_cc(major, minor):
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


def _tmap_to_device(tmap, metadata=None):
    import torch
    del metadata
    host_ptr = tmap.getPtr()
    raw = bytes((ctypes.c_ubyte * 128).from_address(host_ptr))
    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    device = torch.empty(128, dtype=torch.uint8, device="cuda")
    device.copy_(host)
    return device


class Swizzle:
    # Standalone spellings used only by stripped TMA metadata helpers.
    SZ_128B = "128B"
    SZ_64B = "64B"
    SZ_32B = "32B"
    NONE = "none"


class TensorMapMetadata:
    # Compatibility carrier; frozen launch packing needs no metadata.
    def __init__(self, **values):
        self.__dict__.update(values)


def attach_tma_metadata(tensor, metadata):
    del metadata
    return tensor


def _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, swizzle):
    from cuda.bindings import driver
    atom = 64 if swizzle == "128B" else 32
    swizzle_value = (
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle == "128B" else driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
    )
    err, tmap = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        data_ptr,
        [driver.cuuint64_t(atom), driver.cuuint64_t(global_height), driver.cuuint64_t(width // atom)],
        [driver.cuuint64_t(width * 2), driver.cuuint64_t(atom * 2)],
        [driver.cuuint32_t(atom), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // atom)],
        [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_value,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={err}")
    return _tmap_to_device(tmap)


def create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "128B")


def create_tensor_map_3d_64b(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "64B")


def generate_kernel(ir, **kwargs):
    request_key = json.dumps(
        {
            "ir_name": ir.symbol,
            "constants": [[str(name), value] for name, value in ir.constants],
            "threads": int(ir.threads),
            "computed_smem_bytes": int(ir.computed_smem_bytes),
            "kwargs": kwargs,
        },
        sort_keys=True, separators=(",", ":"), default=repr,
    )
    alias = _KERNEL_ALIAS_BY_REQUEST.get(request_key)
    if alias is None:
        alias = _KERNEL_ALIAS_BY_IR_NAME.get(ir.symbol)
    if alias is None:
        raise RuntimeError(f"uncaptured dispatcher specialization for {ir.symbol}")
    return alias


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
