from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from threading import Condition, Event, RLock
from typing import Any

from . import _direct_plan as _direct_plan_runtime
from ._direct_plan import PreparedDirectRoute, prepare_route

SEMANTIC_ENTRYPOINT = (
    "loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:launch_from_contract_inputs"
)


@dataclass(frozen=True)
class PreparedKNNBuild:
    """Fixed inputs and a direct route resolved once outside the hot path."""

    inputs: dict[str, Any]
    launch_plan: PreparedDirectRoute
    shape_label: str | None
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.launch_plan.route_id


@dataclass
class _RuntimeSlot:
    inputs: dict[str, Any]
    launch_plan: PreparedDirectRoute
    query_norm_plan: Any = None
    database_norm_plan: Any = None
    internal_query_norm: bool = False
    internal_database_norm: bool = False
    norm_compute_fields: tuple[str, ...] = ()
    lock: Any = field(default_factory=RLock, repr=False)


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


def _prepare_inputs(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool,
    shape_label: str | None,
    out: tuple[Any, Any] | None,
    query_sq: Any = None,
    database_sq: Any = None,
    defer_missing_norms: bool = False,
) -> dict[str, Any]:
    """Validate the public ABI and allocate one fixed set of intermediates."""
    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    if not isinstance(database, torch.Tensor) or not database.is_cuda:
        raise TypeError("database must be a CUDA torch.Tensor")
    if query.dtype not in (torch.bfloat16, torch.float16) or database.dtype != query.dtype:
        raise TypeError("query and database must have the same bfloat16 or float16 dtype")
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("query must be contiguous with shape [B, Q, D]")
    if database.ndim != 3 or not database.is_contiguous():
        raise ValueError("database must be contiguous with shape [B, N, D]")
    bsz, n_query, dim = map(int, query.shape)
    db_bsz, n_database, db_dim = map(int, database.shape)
    if (db_bsz, db_dim) != (bsz, dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    if build and (n_query != n_database or query.data_ptr() != database.data_ptr()):
        raise ValueError("build=True requires query to alias database and Q == M")
    k = int(k)
    if not 0 < k <= n_database:
        raise ValueError(f"k must be in [1, {n_database}], got {k}")
    expected = (bsz, n_query, k)
    if out is None:
        out = (
            torch.empty(expected, dtype=torch.float32, device=database.device),
            torch.empty(expected, dtype=torch.int32, device=database.device),
        )
    out_dists, out_indices = out
    if tuple(out_dists.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    if out_dists.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out must be (float32 distances, int32 indices)")
    if out_dists.device != database.device or out_indices.device != database.device:
        raise ValueError("out tensors must be on the query/database device")
    if not out_dists.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    if build:
        if query_sq is None and database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), database.device)
            query_sq = database_sq
        elif query_sq is None and not defer_missing_norms:
            query_sq = (query.float() ** 2).sum(-1).contiguous()
        elif query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), database.device)
        if database_sq is not None and database_sq is not query_sq:
            raise ValueError("build=True requires query_sq and database_sq to alias when both are provided")
        database_sq = query_sq
    else:
        if query_sq is None and not defer_missing_norms:
            query_sq = (query.float() ** 2).sum(-1).contiguous()
        elif query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), database.device)
        if database_sq is None and not defer_missing_norms:
            database_sq = (database.float() ** 2).sum(-1).contiguous()
        elif database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), database.device)
    return {
        "label": shape_label,
        "B": bsz,
        "Q": n_query,
        "M": n_database,
        "D": dim,
        "K": k,
        "dtype": str(database.dtype).removeprefix("torch."),
        "build": bool(build),
        "query": query,
        "database": database,
        "query_sq": query_sq,
        "database_sq": database_sq,
        "out_dists": out_dists,
        "out_indices": out_indices,
    }


def _validate_norm(name: str, value: Any, expected: tuple[int, int], device: Any) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise TypeError(f"{name} must be a CUDA torch.Tensor")
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have float32 dtype")
    if value.device != device:
        raise ValueError(f"{name} must be on the query/database device")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _tensor_device_index(tensor: Any) -> int:
    import torch

    index = tensor.device.index
    return int(torch.cuda.current_device() if index is None else index)


def _runtime_device_index(device: Any) -> int:
    import torch

    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int) and not isinstance(device, bool):
        return int(device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"KNN-build runtime requires a CUDA device, got {resolved}")
    return int(torch.cuda.current_device() if resolved.index is None else resolved.index)


def _validate_timeout(timeout_ms: float | None) -> float | None:
    if timeout_ms is None:
        return None
    value = float(timeout_ms)
    if value <= 0:
        raise ValueError("timeout_ms must be positive")
    return value


def _record_stream(inputs: dict[str, Any], stream: Any) -> None:
    seen: set[int] = set()
    for key in (
        "query",
        "database",
        "query_sq",
        "database_sq",
        "out_dists",
        "out_indices",
    ):
        tensor = inputs[key]
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


class KNNBuildRuntime:
    """One device runtime with reusable launch plans keyed by shape and stream."""

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
        self.device_index = _runtime_device_index(device)
        if max_cached_shapes is not None:
            if isinstance(max_cached_shapes, bool) or not isinstance(max_cached_shapes, int):
                raise TypeError("max_cached_shapes must be a positive integer or None")
            if int(max_cached_shapes) <= 0:
                raise ValueError("max_cached_shapes must be positive")
            max_cached_shapes = int(max_cached_shapes)
        self.timeout_ms = _validate_timeout(timeout_ms)
        self.max_cached_shapes = max_cached_shapes
        with torch.cuda.device(self.device_index):
            detected_arch = str(_direct_plan_runtime.detect_gpu_arch())
        self.arch = detected_arch if arch is None else str(arch)
        if self.arch != detected_arch:
            raise ValueError(
                f"KNN-build runtime arch must match its device: requested {self.arch}, detected {detected_arch}"
            )
        self._cache: OrderedDict[tuple[Any, ...], _RuntimeSlot] = OrderedDict()
        self._preparing: dict[tuple[Any, ...], _PendingPreparation] = {}
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_calls = 0
        self._clearing = False
        self._hits = 0
        self._misses = 0

    def _cache_key(self, inputs: dict[str, Any], stream_handle: int) -> tuple[Any, ...]:
        query_sq = inputs["query_sq"]
        database_sq = inputs["database_sq"]
        if query_sq is None or database_sq is None:
            norm_alias = bool(inputs["build"] and query_sq is None and database_sq is None)
        else:
            norm_alias = int(query_sq.data_ptr()) == int(database_sq.data_ptr())
        return (
            self.device_index,
            self.arch,
            int(inputs["B"]),
            int(inputs["Q"]),
            int(inputs["M"]),
            int(inputs["D"]),
            int(inputs["K"]),
            str(inputs["dtype"]),
            bool(inputs["build"]),
            int(inputs["query"].data_ptr()) == int(inputs["database"].data_ptr()),
            query_sq is None,
            database_sq is None,
            norm_alias,
            int(stream_handle),
        )

    @staticmethod
    def _replace_public_inputs(slot: _RuntimeSlot, source: dict[str, Any]) -> None:
        target = slot.inputs
        for key in (
            "query",
            "database",
            "out_dists",
            "out_indices",
        ):
            target[key] = source[key]
        if not slot.internal_query_norm:
            target["query_sq"] = source["query_sq"]
        if not slot.internal_database_norm:
            target["database_sq"] = source["database_sq"]

    @_guard_runtime_compute
    def compute(
        self,
        query: Any,
        database: Any,
        k: int,
        *,
        build: bool = False,
        shape_label: str | None = None,
        out: tuple[Any, Any] | None = None,
        query_sq: Any = None,
        database_sq: Any = None,
        stream: Any = None,
        timeout_ms: float | None = None,
        return_info: bool = False,
    ):
        """Run one input while reusing the cached route, scratch, and arg pack."""

        import torch

        if not isinstance(query, torch.Tensor) or not query.is_cuda:
            raise TypeError("query must be a CUDA torch.Tensor")
        input_device_index = _tensor_device_index(query)
        if input_device_index != self.device_index:
            raise ValueError(
                f"KNN-build runtime is bound to cuda:{self.device_index}, but query is on cuda:{input_device_index}"
            )
        effective_timeout_ms = self.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
        with torch.cuda.device(self.device_index):
            resolved_stream = torch.cuda.current_stream(self.device_index) if stream is None else stream
            stream_device = getattr(resolved_stream, "device", None)
            stream_device_index = getattr(stream_device, "index", stream_device)
            if stream_device_index is not None and int(stream_device_index) != self.device_index:
                raise ValueError(
                    f"KNN-build stream device {stream_device_index} does not match runtime device {self.device_index}"
                )
            stream_handle = int(resolved_stream.cuda_stream)
            with torch.cuda.stream(resolved_stream):
                call_inputs = _prepare_inputs(
                    query,
                    database,
                    k,
                    build=build,
                    shape_label=shape_label,
                    out=out,
                    query_sq=query_sq,
                    database_sq=database_sq,
                    defer_missing_norms=True,
                )
                key = self._cache_key(call_inputs, stream_handle)
                slot, cache_hit, owns_slot_lock = self._get_or_create_slot(
                    key,
                    call_inputs,
                    stream=resolved_stream,
                )
                if not owns_slot_lock:
                    slot.lock.acquire()
                try:
                    if cache_hit:
                        self._replace_public_inputs(slot, call_inputs)
                        slot.launch_plan.rebind_inputs(slot.inputs, stream=resolved_stream)
                    # Record allocator ownership before the first asynchronous
                    # driver launch so timeout/error paths cannot drop storage
                    # that is still in use on this stream.
                    _record_stream(slot.inputs, resolved_stream)
                    if slot.query_norm_plan is not None:
                        try:
                            slot.query_norm_plan.rebind(
                                slot.inputs["query"],
                                slot.inputs["query_sq"],
                                stream=resolved_stream,
                            )
                            slot.query_norm_plan.launch(
                                stream=resolved_stream,
                                timeout_ms=effective_timeout_ms,
                            )
                        finally:
                            slot.query_norm_plan.release_bound_input(
                                slot.inputs["query_sq"],
                                stream=resolved_stream,
                            )
                    if slot.database_norm_plan is not None:
                        try:
                            slot.database_norm_plan.rebind(
                                slot.inputs["database"],
                                slot.inputs["database_sq"],
                                stream=resolved_stream,
                            )
                            slot.database_norm_plan.launch(
                                stream=resolved_stream,
                                timeout_ms=effective_timeout_ms,
                            )
                        finally:
                            slot.database_norm_plan.release_bound_input(
                                slot.inputs["database_sq"],
                                stream=resolved_stream,
                            )
                    slot.launch_plan.launch(
                        slot.inputs,
                        stream=resolved_stream,
                        timeout_ms=effective_timeout_ms,
                    )
                    result = (slot.inputs["out_dists"], slot.inputs["out_indices"])
                    info = {
                        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
                        "selected_route": slot.launch_plan.route_id,
                        "launch_entrypoint": slot.launch_plan.launch_entrypoint,
                        "exact_launch_plan": slot.launch_plan.exact_contract,
                        "shape_label": slot.launch_plan.shape_label,
                        "prepared_launch_count": slot.launch_plan.launch_count,
                        "runtime_launch_count": (
                            slot.launch_plan.launch_count
                            + int(slot.query_norm_plan is not None)
                            + int(slot.database_norm_plan is not None)
                        ),
                        "norm_launch_count": (
                            int(slot.query_norm_plan is not None) + int(slot.database_norm_plan is not None)
                        ),
                        "norm_compute_fields": list(slot.norm_compute_fields),
                        "norm_mode": (
                            "internal_fused_row_norm:" + ",".join(slot.norm_compute_fields)
                            if slot.norm_compute_fields
                            else "route_elided_internal_norms"
                            if slot.internal_query_norm or slot.internal_database_norm
                            else "explicit_precomputed"
                        ),
                        "arch": slot.launch_plan.arch,
                        "device_index": slot.launch_plan.device_index,
                        "stream_handle": slot.launch_plan.stream_handle,
                        "runtime_cache_hit": cache_hit,
                    }
                finally:
                    try:
                        slot.launch_plan.direct_launcher.release_bound_inputs()
                    finally:
                        for public_name in ("query", "database", "out_dists", "out_indices"):
                            slot.inputs[public_name] = None
                        if not slot.internal_query_norm:
                            slot.inputs["query_sq"] = None
                        if not slot.internal_database_norm:
                            slot.inputs["database_sq"] = None
                        slot.lock.release()
        return (result, info) if return_info else result

    def _get_or_create_slot(
        self,
        key: tuple[Any, ...],
        inputs: dict[str, Any],
        *,
        stream: Any,
    ) -> tuple[_RuntimeSlot, bool, bool]:
        """Prepare a cold key without blocking unrelated hot cache hits."""

        while True:
            with self._cache_lock:
                slot = self._cache.get(key)
                if slot is not None:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return slot, True, False
                pending = self._preparing.get(key)
                if pending is None:
                    if (
                        self.max_cached_shapes is not None
                        and len(self._cache) + len(self._preparing) >= self.max_cached_shapes
                    ):
                        raise RuntimeError(
                            "KNNBuildRuntime cache is full; call clear() only after in-flight work completes"
                        )
                    pending = _PendingPreparation()
                    self._preparing[key] = pending
                    break
            pending.event.wait()
            if pending.error is not None:
                raise RuntimeError("KNN-build slot preparation failed in another thread") from pending.error

        slot: _RuntimeSlot | None = None
        try:
            import torch

            from ._row_norm import prepare_row_squared_norm

            internal_query_norm = inputs["query_sq"] is None
            internal_database_norm = inputs["database_sq"] is None
            if internal_query_norm:
                inputs["query_sq"] = torch.empty(
                    (int(inputs["B"]), int(inputs["Q"])),
                    dtype=torch.float32,
                    device=inputs["query"].device,
                )
            if bool(inputs["build"]) and internal_database_norm:
                inputs["database_sq"] = inputs["query_sq"]
            elif internal_database_norm:
                inputs["database_sq"] = torch.empty(
                    (int(inputs["B"]), int(inputs["M"])),
                    dtype=torch.float32,
                    device=inputs["database"].device,
                )
            launch_plan = prepare_route(inputs, arch=self.arch, stream=stream)
            bound_input_keys = set(launch_plan.direct_launcher.bound_input_keys)
            compute_query_norm = internal_query_norm and (
                "query_sq" in bound_input_keys or bool(inputs["build"]) and "database_sq" in bound_input_keys
            )
            compute_database_norm = (
                internal_database_norm and not bool(inputs["build"]) and "database_sq" in bound_input_keys
            )
            query_norm_plan = (
                prepare_row_squared_norm(
                    inputs["query"],
                    inputs["query_sq"],
                    arch=self.arch,
                    stream=stream,
                )
                if compute_query_norm
                else None
            )
            database_norm_plan = (
                prepare_row_squared_norm(
                    inputs["database"],
                    inputs["database_sq"],
                    arch=self.arch,
                    stream=stream,
                )
                if compute_database_norm
                else None
            )
            norm_compute_fields = tuple(
                name
                for name, enabled in (
                    ("query_sq", compute_query_norm),
                    ("database_sq", compute_database_norm),
                )
                if enabled
            )
            slot = _RuntimeSlot(
                inputs=inputs,
                launch_plan=launch_plan,
                query_norm_plan=query_norm_plan,
                database_norm_plan=database_norm_plan,
                internal_query_norm=internal_query_norm,
                internal_database_norm=internal_database_norm,
                norm_compute_fields=norm_compute_fields,
            )
            slot.lock.acquire()
        except BaseException as error:
            if slot is not None:
                try:
                    slot.lock.release()
                except RuntimeError:
                    pass
            with self._cache_lock:
                self._preparing.pop(key, None)
                pending.error = error
                pending.event.set()
            raise
        publication_committed = False
        try:
            with self._cache_lock:
                old_misses = self._misses
                try:
                    self._cache[key] = slot
                    self._misses = old_misses + 1
                    self._preparing.pop(key, None)
                    pending.event.set()
                    publication_committed = True
                except BaseException as error:
                    if self._cache.get(key) is slot:
                        self._cache.pop(key, None)
                    self._misses = old_misses
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    pending.error = error
                    pending.event.set()
                    raise
        except BaseException as error:
            if not publication_committed:
                with self._cache_lock:
                    if self._cache.get(key) is slot:
                        self._cache.pop(key, None)
                        self._misses -= 1
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    if pending.error is None:
                        pending.error = error
                    pending.event.set()
            slot.lock.release()
            raise
        return slot, False, True

    def cache_info(self) -> dict[str, int | None]:
        with self._cache_lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_cached_shapes": self.max_cached_shapes,
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
                    self._cache.clear()
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


def init(
    device: Any = None,
    arch: str | None = None,
    timeout_ms: float | None = None,
    max_cached_shapes: int | None = None,
    compile: str = "lazy",
) -> KNNBuildRuntime:
    """Initialize one reusable KNN-build runtime without binding input tensors."""

    return KNNBuildRuntime(
        device=device,
        arch=arch,
        timeout_ms=timeout_ms,
        max_cached_shapes=max_cached_shapes,
        compile=compile,
    )


def prepare_knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedKNNBuild:
    """Prepare norms, outputs, scratch, and a fully marshalled direct leaf."""

    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != int(device_index):
            raise ValueError(
                f"KNN-build stream device {stream_device_index} does not match input device {device_index}"
            )
        with torch.cuda.stream(resolved_stream):
            inputs = _prepare_inputs(query, database, k, build=build, shape_label=shape_label, out=out)
            launch_plan = prepare_route(inputs, arch=arch, stream=resolved_stream)
    return PreparedKNNBuild(
        inputs=inputs,
        launch_plan=launch_plan,
        shape_label=launch_plan.shape_label,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def knn_build_prepared(
    prepared: PreparedKNNBuild,
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Launch a fixed route without re-entering the parent dispatcher or importer."""
    if not isinstance(prepared, PreparedKNNBuild):
        raise TypeError("prepared must be returned by prepare_knn_build")
    plan = prepared.launch_plan
    if arch is not None and str(arch) != plan.arch:
        raise ValueError(f"prepared KNN-build route targets {plan.arch}, requested incompatible arch {arch}")
    plan.launch(
        prepared.inputs,
        stream=stream,
        timeout_ms=prepared.timeout_ms if timeout_ms is None else timeout_ms,
    )
    out = (prepared.inputs["out_dists"], prepared.inputs["out_indices"])
    if not return_info:
        return out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": plan.launch_entrypoint,
        "exact_launch_plan": plan.exact_contract,
        "shape_label": prepared.shape_label,
        "prepared_launch_count": plan.launch_count,
        "arch": plan.arch,
        "device_index": plan.device_index,
        "stream_handle": plan.stream_handle,
    }
    return out, info


def knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Run one direct frozen KNN build/search route."""
    prepared = prepare_knn_build(
        query,
        database,
        k,
        build=build,
        shape_label=shape_label,
        out=out,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return knn_build_prepared(
        prepared,
        arch=arch,
        stream=prepared.stream,
        timeout_ms=timeout_ms,
        return_info=return_info,
    )
