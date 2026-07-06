from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from threading import Condition, Event, RLock
from typing import Any

from ._dispatch_runtime import capture_kernel_launches, detect_gpu_arch
from ._launch_plan import RouteDecision, resolve_route
from ._runtime import launch_context

SEMANTIC_ENTRYPOINT = "loom.examples.weave.knn_search_registry_b653_compat0701_v1:launch_for_eval"
_PREPARE_LOCK = RLock()


@dataclass
class _KNNSearchRuntimeSlot:
    """One stream-bound, shape-specialized launch sequence and its scratch state."""

    inputs: dict[str, Any]
    launch_plan: RouteDecision
    direct_launcher: Any
    lock: Any


@dataclass
class _PendingPreparation:
    event: Event = field(default_factory=Event, repr=False)
    error: BaseException | None = field(default=None, repr=False)


def _guard_runtime_compute(method: Any) -> Any:
    """Keep clear() from crossing an in-progress lookup/rebind/enqueue call."""

    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            self._active_calls += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            with self._lifecycle:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._lifecycle.notify_all()

    return guarded


@dataclass(frozen=True)
class PreparedKNNSearch:
    """Reusable tensors and a fully marshalled, stream-bound launch sequence."""

    inputs: dict[str, Any]
    launch_plan: RouteDecision
    direct_launcher: Any
    arch: str
    device_index: int
    stream: Any
    stream_handle: int
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.launch_plan.route_id

    @property
    def launch_count(self) -> int:
        return int(self.direct_launcher.launch_count)


def prepare_knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedKNNSearch:
    """Validate tensors and freeze one allocation-free direct launch sequence."""
    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (query, database)):
        raise TypeError("query and database must be CUDA torch.Tensor objects")
    if query.dtype is not torch.bfloat16 or database.dtype is not torch.bfloat16:
        raise TypeError("query and database dtype must be bfloat16")
    if query.ndim != 3 or database.ndim != 3 or not query.is_contiguous() or not database.is_contiguous():
        raise ValueError("query and database must be contiguous [B, rows, D] tensors")
    bsz, q_rows, dim = map(int, query.shape)
    db_bsz, m_rows, db_dim = map(int, database.shape)
    k = int(k)
    if (bsz, dim) != (db_bsz, db_dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    if not 0 < k <= m_rows:
        raise ValueError(f"k must be in [1, {m_rows}], got {k}")
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != device_index:
            raise ValueError(
                f"KNN-search stream device {stream_device_index} does not match input device {device_index}"
            )
        stream_handle = int(resolved_stream.cuda_stream)
        with torch.cuda.stream(resolved_stream), _PREPARE_LOCK:
            detected_arch = detect_gpu_arch()
            resolved_arch = detected_arch if arch is None else str(arch)
            if resolved_arch != detected_arch:
                raise ValueError(
                    "KNN-search launch arch must match the active device: "
                    f"requested {resolved_arch}, detected {detected_arch}"
                )
            expected = (bsz, q_rows, k)
            if out is None:
                out = (
                    torch.empty(expected, dtype=torch.float32, device=query.device),
                    torch.empty(expected, dtype=torch.int32, device=query.device),
                )
            out_distances, out_indices = _validate_runtime_outputs(
                torch,
                out,
                expected,
                query.device,
            )
            inputs = {
                "B": bsz,
                "Q": q_rows,
                "M": m_rows,
                "D": dim,
                "K": k,
                "dtype": "bfloat16",
                "self_search": query.data_ptr() == database.data_ptr(),
                "queries": query,
                "database": database,
                "out_distances": out_distances,
                "out_indices": out_indices,
                "_knn_search_prepared_stream_key": (device_index, stream_handle),
            }
            launch_plan, direct_launcher = _capture_direct_launcher(
                inputs,
                stream=resolved_stream,
                arch=resolved_arch,
            )
    return PreparedKNNSearch(
        inputs=inputs,
        launch_plan=launch_plan,
        direct_launcher=direct_launcher,
        arch=resolved_arch,
        device_index=device_index,
        stream=resolved_stream,
        stream_handle=stream_handle,
        timeout_ms=timeout_ms,
    )


def _capture_direct_launcher(
    inputs: dict[str, Any],
    *,
    stream: Any,
    arch: str,
) -> tuple[RouteDecision, Any]:
    """Resolve once, then capture the exact leaf launches for pointer rebinding."""

    launch_plan = resolve_route(inputs)
    with capture_kernel_launches(stream=stream, arch=arch, inputs=inputs) as captured:
        with launch_context(arch=arch, stream=stream, timeout_ms=None):
            prepared_result = launch_plan.launch(
                inputs,
                stream=stream,
                timeout_ms=None,
            )
    _require_owned_outputs(prepared_result, inputs)
    return launch_plan, captured.bind(prepared_result)


def _require_owned_outputs(outputs: Any, inputs: dict[str, Any]) -> None:
    """Reject prepared routes whose hot path would require an uncaptured copy."""

    owned = (inputs["out_distances"], inputs["out_indices"])
    if outputs is None:
        normalized = owned
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        normalized = (outputs[0], outputs[1])
    elif isinstance(outputs, dict):
        distances = outputs.get("distances", outputs.get("dists", outputs.get("out_distances")))
        indices = outputs.get("indices", outputs.get("idxs", outputs.get("out_indices")))
        if distances is None or indices is None:
            raise TypeError("knn_search dispatcher output dict must contain distances and indices")
        normalized = (distances, indices)
    else:
        raise TypeError("knn_search dispatcher must return (distances, indices), a matching dict, or write outputs")
    if any(source is not destination for destination, source in zip(owned, normalized, strict=True)):
        raise RuntimeError("prepared KNN-search route must write caller-owned outputs through captured launches")


def knn_search_prepared(
    prepared: PreparedKNNSearch,
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Submit a frozen exact KNN search without routing, allocation, or packing."""
    if not isinstance(prepared, PreparedKNNSearch):
        raise TypeError("prepared must be returned by prepare_knn_search")
    import torch

    if arch is not None and str(arch) != prepared.arch:
        raise ValueError(
            f"prepared KNN-search route targets {prepared.arch}; requested incompatible arch {arch}"
        )
    with torch.cuda.device(prepared.device_index):
        requested_stream = prepared.stream if stream is None else stream
        requested_handle = int(requested_stream.cuda_stream)
        if requested_handle != prepared.stream_handle:
            raise RuntimeError(
                "prepared KNN-search route is stream-bound: "
                f"prepared on stream 0x{prepared.stream_handle:x}, requested 0x{requested_handle:x}; "
                "prepare a separate plan inside the target torch.cuda.stream(...) context"
            )
        effective_timeout_ms = prepared.timeout_ms if timeout_ms is None else timeout_ms
        try:
            prepared.direct_launcher(
                prepared.inputs,
                stream=None,
                timeout_ms=effective_timeout_ms,
            )
        finally:
            prepared.direct_launcher.record_stream(requested_stream)
            _record_stream(
                (
                    prepared.inputs["queries"],
                    prepared.inputs["database"],
                    prepared.inputs["out_distances"],
                    prepared.inputs["out_indices"],
                ),
                requested_stream,
            )
    out = (prepared.inputs["out_distances"], prepared.inputs["out_indices"])
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": prepared.launch_plan.launch_entrypoint,
        "exact_launch_plan": prepared.launch_plan.exact_contract,
        "prepared_launch_count": prepared.launch_count,
        "arch": prepared.arch,
        "device_index": prepared.device_index,
        "stream_handle": prepared.stream_handle,
    }
    return (out, info) if return_info else out


def knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Run exact squared-L2 kNN with the frozen production dispatcher."""
    prepared = prepare_knn_search(
        query,
        database,
        k,
        out=out,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return knn_search_prepared(prepared, arch=arch, return_info=return_info)


class KNNSearchRuntime:
    """Reusable KNN-search runtime with lazy per-shape, per-stream launch plans.

    A cache miss resolves the production dispatcher and captures its exact leaf
    launches.  Cache hits update public tensor fields, rebind the already
    marshalled launch arguments, and replay without traversing the dispatcher.
    Route-owned scratch remains in the slot's persistent ``inputs`` mapping.
    """

    def __init__(
        self,
        *,
        device: Any = None,
        arch: str | None = None,
        timeout_ms: float | None = None,
        max_cached_shapes: int | None = None,
        compile: str = "lazy",
    ) -> None:
        import torch

        if compile != "lazy":
            raise ValueError("compile must be 'lazy'; use warmup() to eagerly prepare known shapes")
        if max_cached_shapes is not None:
            if isinstance(max_cached_shapes, bool) or not isinstance(max_cached_shapes, int):
                raise TypeError("max_cached_shapes must be a positive integer or None")
            if max_cached_shapes <= 0:
                raise ValueError("max_cached_shapes must be positive")
        self.device_index = _normalize_device_index(torch, device)
        with torch.cuda.device(self.device_index):
            detected_arch = detect_gpu_arch()
        self.arch = detected_arch if arch is None else str(arch)
        if self.arch != detected_arch:
            raise ValueError(
                "KNN-search runtime arch must match its device: "
                f"requested {self.arch}, detected {detected_arch}"
            )
        self.timeout_ms = timeout_ms
        self.max_cached_shapes = max_cached_shapes
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_calls = 0
        self._clearing = False
        self._slots: OrderedDict[tuple[Any, ...], _KNNSearchRuntimeSlot] = OrderedDict()
        self._preparing: dict[tuple[Any, ...], _PendingPreparation] = {}
        self._hits = 0
        self._misses = 0

    @_guard_runtime_compute
    def compute(
        self,
        query: Any,
        database: Any,
        k: int,
        *,
        out: tuple[Any, Any] | None = None,
        stream: Any = None,
        timeout_ms: float | None = None,
        return_info: bool = False,
    ):
        """Compute exact squared-L2 KNN, reusing the slot for this shape/stream."""
        import torch

        shape = _validate_runtime_tensors(torch, query, database, k, self.device_index)
        bsz, q_rows, m_rows, dim, k = shape
        with torch.cuda.device(self.device_index):
            resolved_stream = torch.cuda.current_stream(self.device_index) if stream is None else stream
            stream_device = getattr(resolved_stream, "device", None)
            stream_device_index = getattr(stream_device, "index", stream_device)
            if stream_device_index is not None and int(stream_device_index) != self.device_index:
                raise ValueError(
                    f"KNN-search stream device {stream_device_index} does not match runtime device "
                    f"{self.device_index}"
                )
            stream_handle = int(resolved_stream.cuda_stream)
            expected = (bsz, q_rows, k)
            if out is None:
                out = (
                    torch.empty(expected, dtype=torch.float32, device=query.device),
                    torch.empty(expected, dtype=torch.int32, device=query.device),
                )
            out_distances, out_indices = _validate_runtime_outputs(torch, out, expected, query.device)
            self_search = query.data_ptr() == database.data_ptr()
            public_inputs = {
                "B": bsz,
                "Q": q_rows,
                "M": m_rows,
                "D": dim,
                "K": k,
                "dtype": "bfloat16",
                "self_search": self_search,
                "queries": query,
                "database": database,
                "out_distances": out_distances,
                "out_indices": out_indices,
                "_knn_search_prepared_stream_key": (self.device_index, stream_handle),
            }
            cache_key = (
                self.device_index,
                self.arch,
                bsz,
                q_rows,
                m_rows,
                dim,
                k,
                query.dtype,
                self_search,
                stream_handle,
            )
            with torch.cuda.stream(resolved_stream):
                slot, cache_hit, owns_slot_lock = self._get_or_create_slot(
                    cache_key,
                    public_inputs,
                    stream=resolved_stream,
                )
                if not owns_slot_lock:
                    slot.lock.acquire()
                try:
                    if cache_hit:
                        slot.inputs.update(public_inputs)
                        slot.direct_launcher.rebind_inputs(slot.inputs)
                    effective_timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
                    try:
                        slot.direct_launcher(
                            slot.inputs,
                            stream=None,
                            timeout_ms=effective_timeout_ms,
                        )
                    finally:
                        slot.direct_launcher.record_stream(resolved_stream)
                        _record_stream(
                            (query, database, out_distances, out_indices),
                            resolved_stream,
                        )
                    slot.direct_launcher.release_bound_inputs()
                    for name in ("queries", "database", "out_distances", "out_indices"):
                        slot.inputs[name] = None
                finally:
                    slot.lock.release()
        outputs = (out_distances, out_indices)
        info = {
            "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
            "selected_route": slot.launch_plan.route_id,
            "launch_entrypoint": slot.launch_plan.launch_entrypoint,
            "exact_launch_plan": slot.launch_plan.exact_contract,
            "prepared_launch_count": int(slot.direct_launcher.launch_count),
            "arch": self.arch,
            "device_index": self.device_index,
            "stream_handle": stream_handle,
            "runtime_cache_hit": cache_hit,
        }
        return (outputs, info) if return_info else outputs

    def _get_or_create_slot(
        self,
        cache_key: tuple[Any, ...],
        inputs: dict[str, Any],
        *,
        stream: Any,
    ) -> tuple[_KNNSearchRuntimeSlot, bool, bool]:
        """Return a slot; a newly published slot is returned with its lock held."""

        while True:
            with self._cache_lock:
                slot = self._slots.get(cache_key)
                if slot is not None:
                    self._slots.move_to_end(cache_key)
                    self._hits += 1
                    return slot, True, False
                pending = self._preparing.get(cache_key)
                if pending is None:
                    if (
                        self.max_cached_shapes is not None
                        and len(self._slots) + len(self._preparing) >= self.max_cached_shapes
                    ):
                        raise RuntimeError(
                            "KNNSearchRuntime cache is full; call clear() only after in-flight work completes"
                        )
                    pending = _PendingPreparation()
                    self._preparing[cache_key] = pending
                    break
            pending.event.wait()
            if pending.error is not None:
                raise RuntimeError("KNN-search slot preparation failed in another thread") from pending.error

        slot: _KNNSearchRuntimeSlot | None = None
        try:
            with _PREPARE_LOCK:
                launch_plan, direct_launcher = _capture_direct_launcher(
                    inputs,
                    stream=stream,
                    arch=self.arch,
                )
            slot = _KNNSearchRuntimeSlot(
                inputs=inputs,
                launch_plan=launch_plan,
                direct_launcher=direct_launcher,
                lock=RLock(),
            )
            slot.lock.acquire()
        except BaseException as error:
            if slot is not None:
                try:
                    slot.lock.release()
                except RuntimeError:
                    pass
            with self._cache_lock:
                self._preparing.pop(cache_key, None)
                pending.error = error
                pending.event.set()
            raise
        publication_committed = False
        try:
            with self._cache_lock:
                old_misses = self._misses
                try:
                    self._slots[cache_key] = slot
                    self._misses = old_misses + 1
                    self._preparing.pop(cache_key, None)
                    pending.event.set()
                    publication_committed = True
                except BaseException as error:
                    if self._slots.get(cache_key) is slot:
                        self._slots.pop(cache_key, None)
                    self._misses = old_misses
                    if self._preparing.get(cache_key) is pending:
                        self._preparing.pop(cache_key, None)
                    pending.error = error
                    pending.event.set()
                    raise
        except BaseException as error:
            if not publication_committed:
                with self._cache_lock:
                    if self._slots.get(cache_key) is slot:
                        self._slots.pop(cache_key, None)
                        self._misses -= 1
                    if self._preparing.get(cache_key) is pending:
                        self._preparing.pop(cache_key, None)
                    if pending.error is None:
                        pending.error = error
                    pending.event.set()
            slot.lock.release()
            raise
        return slot, False, True

    def cache_info(self) -> dict[str, int | None]:
        """Return stable cache counters without exposing mutable slot state."""

        with self._cache_lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._slots),
                "max_cached_shapes": self.max_cached_shapes,
                "stream_count": len({key[-1] for key in self._slots}),
            }

    def clear(self, *, synchronize: bool = True) -> None:
        """Drop cached plans after host calls finish.

        The default waits for device completion. With ``synchronize=False``,
        recorded-stream ownership keeps tensor storage allocator-safe, but the
        caller remains responsible for observing asynchronous completion.
        """
        import torch

        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            try:
                self._clearing = True
                while self._active_calls:
                    self._lifecycle.wait()
                if synchronize:
                    with torch.cuda.device(self.device_index):
                        torch.cuda.synchronize()
                with self._cache_lock:
                    self._slots.clear()
                    self._hits = 0
                    self._misses = 0
            finally:
                self._clearing = False
                self._lifecycle.notify_all()

    def warmup(self, *args: Any, synchronize: bool = True, **kwargs: Any):
        result = self.compute(*args, **kwargs)
        if synchronize:
            import torch

            with torch.cuda.device(self.device_index):
                torch.cuda.synchronize()
        return result


def _normalize_device_index(torch: Any, device: Any) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    if isinstance(device, int):
        return int(device)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type is None or device_index is None:
        device_ctor = getattr(torch, "device", None)
        if device_ctor is None:
            raise TypeError("device must identify a CUDA device")
        normalized = device_ctor(device)
        device_type = getattr(normalized, "type", None)
        device_index = getattr(normalized, "index", None)
    if device_type != "cuda":
        raise ValueError(f"KNN-search runtime requires a CUDA device, got {device_type!r}")
    return int(torch.cuda.current_device() if device_index is None else device_index)


def _validate_runtime_tensors(
    torch: Any,
    query: Any,
    database: Any,
    k: int,
    device_index: int,
) -> tuple[int, int, int, int, int]:
    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (query, database)):
        raise TypeError("query and database must be CUDA torch.Tensor objects")
    if query.dtype is not torch.bfloat16 or database.dtype is not torch.bfloat16:
        raise TypeError("query and database dtype must be bfloat16")
    if query.ndim != 3 or database.ndim != 3 or not query.is_contiguous() or not database.is_contiguous():
        raise ValueError("query and database must be contiguous [B, rows, D] tensors")
    bsz, q_rows, dim = map(int, query.shape)
    db_bsz, m_rows, db_dim = map(int, database.shape)
    k = int(k)
    if (bsz, dim) != (db_bsz, db_dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    query_device_index = query.device.index
    if query_device_index is None:
        query_device_index = torch.cuda.current_device()
    if int(query_device_index) != device_index:
        raise ValueError(
            f"KNN-search input device {query_device_index} does not match runtime device {device_index}"
        )
    if not 0 < k <= m_rows:
        raise ValueError(f"k must be in [1, {m_rows}], got {k}")
    return bsz, q_rows, m_rows, dim, k


def _validate_runtime_outputs(
    torch: Any,
    out: tuple[Any, Any],
    expected: tuple[int, int, int],
    device: Any,
) -> tuple[Any, Any]:
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError("out must be a (distances, indices) pair")
    out_distances, out_indices = out
    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in out):
        raise TypeError("out tensors must be CUDA torch.Tensor objects")
    if out_distances.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out distances/indices dtype must be float32/int32")
    if out_distances.device != device or out_indices.device != device:
        raise ValueError("out tensors must be on the same device as query and database")
    if not out_distances.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    if tuple(out_distances.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    return out_distances, out_indices


def _record_stream(tensors: tuple[Any, ...], stream: Any) -> None:
    for tensor in tensors:
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


def init(
    device: Any = None,
    arch: str | None = None,
    timeout_ms: float | None = None,
    max_cached_shapes: int | None = None,
    compile: str = "lazy",
) -> KNNSearchRuntime:
    """Initialize one reusable KNN-search runtime for a CUDA device."""

    return KNNSearchRuntime(
        device=device,
        arch=arch,
        timeout_ms=timeout_ms,
        max_cached_shapes=max_cached_shapes,
        compile=compile,
    )
