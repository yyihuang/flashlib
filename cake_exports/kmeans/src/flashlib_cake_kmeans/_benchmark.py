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
    """Strict CUPTI timing plus explicitly non-official host diagnostics.

    Cold-L2 flushing completes before host timestamps begin, so synchronized
    E2E isolates semantic call start through candidate completion. ``times_ms``
    remains the official correlated GPU span.
    """

    times_ms: list[float]
    backend: str = "cupti"
    kernel_sum_times_ms: list[float] | None = None
    inter_kernel_gap_times_ms: list[float] | None = None
    active_union_times_ms: list[float] | None = None
    activity_counts: list[int] | None = None
    launch_activity_counts: list[int] | None = None
    kernel_activity_counts: list[int] | None = None
    submission_times_ms: list[float] | None = None
    synchronized_e2e_times_ms: list[float] | None = None
    cold_first_call_host_enqueue_ms: float | None = None
    cold_first_call_synchronized_e2e_ms: float | None = None

    def __post_init__(self) -> None:
        if self.backend != "cupti":
            raise ValueError(f"exported benchmark timing backend must be 'cupti', got {self.backend!r}")

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.times_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.times_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.times_ms))

    @property
    def median_gpu_span_ms(self) -> float:
        return self.median_ms

    @property
    def median_kernel_sum_ms(self) -> float | None:
        if self.kernel_sum_times_ms is None:
            return None
        return float(statistics.median(self.kernel_sum_times_ms))

    @property
    def median_inter_kernel_gap_ms(self) -> float | None:
        if self.inter_kernel_gap_times_ms is None:
            return None
        return float(statistics.median(self.inter_kernel_gap_times_ms))

    @property
    def median_active_union_ms(self) -> float | None:
        if self.active_union_times_ms is None:
            return None
        return float(statistics.median(self.active_union_times_ms))

    @property
    def median_activity_count(self) -> float | None:
        if self.activity_counts is None:
            return None
        return float(statistics.median(self.activity_counts))

    @property
    def median_launch_activity_count(self) -> float | None:
        if self.launch_activity_counts is None:
            return None
        return float(statistics.median(self.launch_activity_counts))

    @property
    def median_kernel_activity_count(self) -> float | None:
        if self.kernel_activity_counts is None:
            return None
        return float(statistics.median(self.kernel_activity_counts))

    @property
    def median_submission_ms(self) -> float | None:
        if self.submission_times_ms is None:
            return None
        return float(statistics.median(self.submission_times_ms))

    @property
    def median_host_enqueue_ms(self) -> float | None:
        return self.median_submission_ms

    @property
    def host_enqueue_times_ms(self) -> list[float] | None:
        return self.submission_times_ms

    @property
    def median_synchronized_e2e_ms(self) -> float | None:
        if self.synchronized_e2e_times_ms is None:
            return None
        return float(statistics.median(self.synchronized_e2e_times_ms))


@dataclass(frozen=True)
class HostCallTiming:
    """Diagnostic host brackets for one call; never an official GPU timing."""

    host_enqueue_ms: float
    synchronized_e2e_ms: float


@dataclass(frozen=True)
class _CuptiTiming:
    gpu_span_ms: list[float]
    kernel_sum_ms: list[float]
    inter_kernel_gap_ms: list[float]
    active_union_ms: list[float]
    activity_count: list[int]
    launch_activity_count: list[int]
    kernel_activity_count: list[int]


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
    # cupti-python imports cuda.pathfinder even when libcupti was found through
    # the nvidia-cuda-cupti distribution.  Extend the split CUDA namespace
    # before either loading path so the later extension import is reliable.
    _extend_cuda_namespace_for_pathfinder()
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


def measure_host_call(fn: Callable[[], Any]) -> tuple[Any, HostCallTiming]:
    """Run one call and return explicitly labeled host diagnostics.

    The enqueue bracket ends when ``fn`` returns.  The synchronized bracket
    ends after ``torch.cuda.synchronize()``.  A function that synchronizes
    internally will therefore have a blocking enqueue bracket; callers should
    not interpret either value as GPU-only execution time.
    """
    import torch

    cupti = require_cupti()
    start = cupti.get_timestamp()
    value = fn()
    submitted = cupti.get_timestamp()
    torch.cuda.synchronize()
    completed = cupti.get_timestamp()
    return value, HostCallTiming(
        host_enqueue_ms=(submitted - start) / 1e6,
        synchronized_e2e_ms=(completed - start) / 1e6,
    )


def _complete_l2_flush_before_bracket(flusher: Any, synchronize: Callable[[], None]) -> None:
    """Finish cold-L2 preconditioning before host latency timestamps begin."""
    if flusher is None:
        return
    flusher.flush()
    synchronize()


def _correlate(
    cpu_brackets: list[tuple[int, int, int]],
    launches: list[tuple[int, int, int]],
    kernels: list[tuple[int, int, int]],
) -> _CuptiTiming:
    if not launches or not kernels:
        raise RuntimeError("CUPTI collected no launch/kernel activities")
    launches.sort(key=lambda item: item[0])
    launch_starts = [item[0] for item in launches]
    kernels.sort(key=lambda item: item[0])
    kernel_indices_by_correlation: dict[int, list[int]] = {}
    for index, (_start, _end, correlation_id) in enumerate(kernels):
        kernel_indices_by_correlation.setdefault(correlation_id, []).append(index)

    gpu_span_ms: list[float] = []
    kernel_sum_ms: list[float] = []
    inter_kernel_gap_ms: list[float] = []
    active_union_ms: list[float] = []
    activity_count: list[int] = []
    launch_activity_count: list[int] = []
    kernel_activity_count: list[int] = []
    for bracket_start, bracket_end, _completed in cpu_brackets:
        lo = bisect.bisect_left(launch_starts, bracket_start)
        hi = bisect.bisect_right(launch_starts, bracket_end)
        correlation_ids = {launches[index][2] for index in range(lo, hi)}
        launch_activity_count.append(
            sum(
                1
                for index in range(lo, hi)
                if kernel_indices_by_correlation.get(launches[index][2])
            )
        )
        selected_indices = {
            index
            for correlation_id in correlation_ids
            for index in kernel_indices_by_correlation.get(correlation_id, ())
        }
        if not selected_indices:
            raise RuntimeError("CUPTI could not correlate a benchmark iteration with GPU kernel activity")
        spans = [(kernels[index][0], kernels[index][1]) for index in selected_indices]
        span_ns = max(end for _, end in spans) - min(start for start, _ in spans)
        sum_ns = sum(end - start for start, end in spans)
        covered_ns = 0
        current_start, current_end = sorted(spans)[0]
        for start, end in sorted(spans)[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_ns += current_end - current_start
                current_start, current_end = start, end
        covered_ns += current_end - current_start
        gpu_span_ms.append(span_ns / 1e6)
        kernel_sum_ms.append(sum_ns / 1e6)
        active_union_ms.append(covered_ns / 1e6)
        inter_kernel_gap_ms.append((span_ns - covered_ns) / 1e6)
        activity_count.append(len(selected_indices))
        kernel_activity_count.append(len(selected_indices))
    return _CuptiTiming(
        gpu_span_ms=gpu_span_ms,
        kernel_sum_ms=kernel_sum_ms,
        inter_kernel_gap_ms=inter_kernel_gap_ms,
        active_union_ms=active_union_ms,
        activity_count=activity_count,
        launch_activity_count=launch_activity_count,
        kernel_activity_count=kernel_activity_count,
    )


def bench_gpu_time(
    fn: Callable[[], Any],
    *,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    cold_l2: bool = True,
    cold_first_call: HostCallTiming | None = None,
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
    cpu_brackets: list[tuple[int, int, int]] = []
    cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
    try:
        for kind in kinds:
            cupti.activity_enable(kind)
            enabled.append(kind)

        for _ in range(bench_iters):
            _complete_l2_flush_before_bracket(flusher, torch.cuda.synchronize)
            start = cupti.get_timestamp()
            fn()
            submitted = cupti.get_timestamp()
            torch.cuda.synchronize()
            completed = cupti.get_timestamp()
            cpu_brackets.append((start, submitted, completed))
        cupti.activity_flush_all(1)
    finally:
        for kind in reversed(enabled):
            cupti.activity_disable(kind)
        cupti.finalize()

    timing = _correlate(cpu_brackets, launches, kernels)
    return BenchResult(
        times_ms=timing.gpu_span_ms,
        kernel_sum_times_ms=timing.kernel_sum_ms,
        inter_kernel_gap_times_ms=timing.inter_kernel_gap_ms,
        active_union_times_ms=timing.active_union_ms,
        activity_counts=timing.activity_count,
        launch_activity_counts=timing.launch_activity_count,
        kernel_activity_counts=timing.kernel_activity_count,
        submission_times_ms=[(submitted - start) / 1e6 for start, submitted, _ in cpu_brackets],
        synchronized_e2e_times_ms=[
            (completed - start) / 1e6 for start, _, completed in cpu_brackets
        ],
        cold_first_call_host_enqueue_ms=(
            cold_first_call.host_enqueue_ms if cold_first_call is not None else None
        ),
        cold_first_call_synchronized_e2e_ms=(
            cold_first_call.synchronized_e2e_ms if cold_first_call is not None else None
        ),
    )
