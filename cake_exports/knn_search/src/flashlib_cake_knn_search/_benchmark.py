from __future__ import annotations

import bisect
import ctypes
import importlib
import importlib.metadata
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    times_ms: list[float]
    backend: str = "cupti"

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.times_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.times_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.times_ms))


class _L2Flusher:
    def __init__(self) -> None:
        import torch

        l2_size = int(torch.cuda.get_device_properties(0).L2_cache_size)
        if l2_size <= 0:
            raise RuntimeError("CUDA device did not report a positive L2 cache size")
        self._buffer = torch.empty(2 * l2_size, dtype=torch.int8, device="cuda")

    def flush(self) -> None:
        self._buffer.zero_()


_CUPTI: Any | None = None


def _extend_cuda_namespace_for_pathfinder() -> None:
    try:
        import cuda
    except ImportError:
        return
    cuda_paths = getattr(cuda, "__path__", None)
    if cuda_paths is None:
        return
    known = {str(path) for path in cuda_paths}
    for entry in sys.path:
        if not entry:
            continue
        cuda_root = Path(entry) / "cuda"
        if (cuda_root / "pathfinder").is_dir() and str(cuda_root) not in known:
            cuda_paths.append(str(cuda_root))
            known.add(str(cuda_root))


def _preload_cupti_library() -> None:
    try:
        distribution = importlib.metadata.distribution("nvidia-cuda-cupti")
        major = importlib.metadata.version("cupti-python").split(".", 1)[0]
        packaged = next(
            (
                distribution.locate_file(path)
                for path in distribution.files or ()
                if str(path).endswith(f"/libcupti.so.{major}")
            ),
            None,
        )
        if packaged is not None and Path(packaged).is_file():
            ctypes.CDLL(str(packaged), mode=ctypes.RTLD_GLOBAL)
            return
    except (importlib.metadata.PackageNotFoundError, OSError):
        pass
    _extend_cuda_namespace_for_pathfinder()
    try:
        pathfinder = importlib.import_module("cuda.pathfinder")
    except ImportError:
        return
    loader = getattr(pathfinder, "load_nvidia_dynamic_lib", None)
    if loader is None:
        return
    loaded = loader("cupti")
    loaded_path = str(getattr(loaded, "abs_path", ""))
    try:
        major = importlib.metadata.version("cupti-python").split(".", 1)[0]
    except importlib.metadata.PackageNotFoundError:
        return
    expected = f"libcupti.so.{major}" if major.isdigit() else None
    if expected and loaded_path:
        name = Path(loaded_path).name
        if name.startswith("libcupti.so.") and name != expected:
            raise ImportError(f"incompatible CUPTI library {loaded_path}; cupti-python expects {expected}")


def require_cupti() -> Any:
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        _preload_cupti_library()
        from cupti import cupti
    except ImportError as exc:
        raise RuntimeError(
            "CUPTI timing is required; install the exported repository with its benchmark extra: "
            "python -m pip install -e '.[benchmark]'"
        ) from exc
    _CUPTI = cupti
    return _CUPTI


def _correlate(
    cpu_brackets: list[tuple[int, int]],
    launches: list[tuple[int, int, int]],
    kernels: list[tuple[int, int, int]],
) -> list[float]:
    if not launches or not kernels:
        raise RuntimeError("CUPTI collected no launch/kernel activities")
    launches.sort(key=lambda item: item[0])
    launch_starts = [item[0] for item in launches]
    kernels_by_correlation: dict[int, list[tuple[int, int]]] = {}
    for start, end, correlation_id in kernels:
        kernels_by_correlation.setdefault(correlation_id, []).append((start, end))

    times_ms: list[float] = []
    for bracket_start, bracket_end in cpu_brackets:
        lo = bisect.bisect_left(launch_starts, bracket_start)
        hi = bisect.bisect_right(launch_starts, bracket_end)
        correlation_ids = {launches[index][2] for index in range(lo, hi)}
        spans = [span for cid in correlation_ids for span in kernels_by_correlation.get(cid, ())]
        if not spans:
            raise RuntimeError("CUPTI could not correlate a benchmark iteration with GPU kernel activity")
        times_ms.append((max(end for _, end in spans) - min(start for start, _ in spans)) / 1e6)
    return times_ms


def bench_gpu_time(
    fn: Callable[[], Any],
    *,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    cold_l2: bool = True,
) -> BenchResult:
    """Measure a zero-argument GPU workload with strict CUPTI activity tracing.

    L2 is flushed before every warmup and measured iteration. This function
    never falls back to wall-clock or CUDA-event timing.
    """
    if warmup_iters < 0 or bench_iters <= 0:
        raise ValueError("warmup_iters must be non-negative and bench_iters must be positive")

    import torch

    cupti = require_cupti()
    flusher = _L2Flusher() if cold_l2 else None
    for _ in range(warmup_iters):
        if flusher is not None:
            flusher.flush()
        fn()
    torch.cuda.synchronize()

    launch_kinds = {int(cupti.ActivityKind.RUNTIME), int(cupti.ActivityKind.DRIVER)}
    kernel_kinds = {int(cupti.ActivityKind.CONCURRENT_KERNEL)}
    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int]] = []

    def _buffer_requested():
        return 8 * 1024 * 1024, 0

    def _buffer_completed(activities: list[Any]) -> None:
        for activity in activities:
            record = (activity.start, activity.end, activity.correlation_id)
            kind = int(activity.kind)
            if kind in launch_kinds:
                launches.append(record)
            elif kind in kernel_kinds:
                kernels.append(record)

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    enabled: list[Any] = []
    cpu_brackets: list[tuple[int, int]] = []
    cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
    try:
        for kind in kinds:
            cupti.activity_enable(kind)
            enabled.append(kind)

        for _ in range(bench_iters):
            if flusher is not None:
                flusher.flush()
            start = cupti.get_timestamp()
            fn()
            end = cupti.get_timestamp()
            torch.cuda.synchronize()
            cpu_brackets.append((start, end))
        cupti.activity_flush_all(1)
    finally:
        for kind in reversed(enabled):
            cupti.activity_disable(kind)
        cupti.finalize()

    return BenchResult(_correlate(cpu_brackets, launches, kernels))
