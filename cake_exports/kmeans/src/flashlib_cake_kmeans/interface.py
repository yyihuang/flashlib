from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, Event, RLock
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _dispatcher
from ._runtime import detect_gpu_arch, launch_context

SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"


@dataclass(frozen=True)
class PreparedFlashKMeansAssign:
    """Exact tensors, workspace, output, and direct launch plan for KMeans."""

    inputs: dict[str, Any]
    launch_plan: Any
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def out(self) -> Any:
        return self.inputs["out"]

    @property
    def selected_route(self) -> str:
        return self.launch_plan.selected_route


@dataclass
class _RuntimeSlot:
    inputs: dict[str, Any]
    launch_plan: Any
    norm_plan: Any = None
    internal_x_sq: bool = False
    internal_c_sq: bool = False
    norm_compute_fields: tuple[str, ...] = ()
    lock: RLock = field(default_factory=RLock, repr=False)


@dataclass
class _PendingPreparation:
    event: Event = field(default_factory=Event, repr=False)
    error: BaseException | None = field(default=None, repr=False)


def _normalize_device_index(torch: Any, device: Any) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be non-negative")
        return int(device)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type is None:
        device_ctor = getattr(torch, "device", None)
        if device_ctor is None:
            raise TypeError("device must identify a CUDA device")
        normalized = device_ctor(device)
        device_type = getattr(normalized, "type", None)
        device_index = getattr(normalized, "index", None)
    if device_type != "cuda":
        raise ValueError(f"Flash-KMeans runtime requires a CUDA device, got {device_type!r}")
    resolved_index = int(torch.cuda.current_device() if device_index is None else device_index)
    if resolved_index < 0:
        raise ValueError("CUDA device index must be non-negative")
    return resolved_index


def _validate_timeout(timeout_ms: float | None) -> float | None:
    if timeout_ms is None:
        return None
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int | float):
        raise TypeError("timeout_ms must be a positive finite number or None")
    value = float(timeout_ms)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_ms must be a positive finite number")
    return value


def _pointer_alias_topology(*tensors: Any) -> tuple[int, ...]:
    """Canonical pointer-equivalence classes independent of concrete addresses."""

    classes: dict[int, int] = {}
    topology: list[int] = []
    for tensor in tensors:
        if tensor is None:
            topology.append(-1)
            continue
        pointer = int(tensor.data_ptr())
        if pointer not in classes:
            classes[pointer] = len(classes)
        topology.append(classes[pointer])
    return tuple(topology)


class FlashKMeansAssignRuntime:
    """Long-lived KMeans assignment runtime with per-shape, per-stream plans.

    ``compute`` accepts new tensors and new shapes.  The first call for one
    shape/stream resolves and captures the exact route; later calls only
    refresh data-derived norms, rebind caller tensor pointers, and enqueue the
    already captured launch sequence.  Workspace is never shared across CUDA
    streams.
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
            detected_arch = str(detect_gpu_arch())
        self.arch = detected_arch if arch is None else str(arch)
        if self.arch != detected_arch:
            raise ValueError(
                f"Flash-KMeans runtime arch must match its device: requested {self.arch}, detected {detected_arch}"
            )
        self.timeout_ms = _validate_timeout(timeout_ms)
        self.max_cached_shapes = max_cached_shapes
        self._slots: OrderedDict[tuple[Any, ...], _RuntimeSlot] = OrderedDict()
        self._preparing: dict[tuple[Any, ...], _PendingPreparation] = {}
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_computes = 0
        self._clearing = False
        self._hits = 0
        self._misses = 0

    @contextmanager
    def _compute_lifecycle(self):
        """Admit concurrent compute calls while making clear() an exclusive barrier."""

        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            self._active_computes += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._active_computes -= 1
                if self._active_computes == 0:
                    self._lifecycle.notify_all()

    def cache_info(self) -> dict[str, int | None]:
        with self._cache_lock:
            return {
                "size": len(self._slots),
                "hits": self._hits,
                "misses": self._misses,
                "max_cached_shapes": self.max_cached_shapes,
            }

    def _cache_key(self, inputs: dict[str, Any], stream_handle: int) -> tuple[Any, ...]:
        return (
            self.device_index,
            self.arch,
            int(inputs["B"]),
            int(inputs["N"]),
            int(inputs["D"]),
            int(inputs["K"]),
            str(inputs["dtype"]),
            _pointer_alias_topology(
                inputs["x"],
                inputs["centroids"],
                inputs["x_sq"],
                inputs["c_sq"],
            ),
            int(stream_handle),
        )

    def clear(self, *, synchronize: bool = True) -> None:
        """Exclusively release slots after admitted host calls finish.

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
                while self._active_computes:
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

    def compute(
        self,
        x: Any,
        centroids: Any,
        *,
        out: Any | None = None,
        x_sq: Any | None = None,
        c_sq: Any | None = None,
        stream: Any = None,
        timeout_ms: float | None = None,
        return_info: bool = False,
    ):
        with self._compute_lifecycle():
            return self._compute(
                x,
                centroids,
                out=out,
                x_sq=x_sq,
                c_sq=c_sq,
                stream=stream,
                timeout_ms=timeout_ms,
                return_info=return_info,
            )

    def _compute(
        self,
        x: Any,
        centroids: Any,
        *,
        out: Any | None,
        x_sq: Any | None,
        c_sq: Any | None,
        stream: Any,
        timeout_ms: float | None,
        return_info: bool,
    ):
        import torch

        effective_timeout_ms = self.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
        inputs, resolved_stream, resolved_arch = _prepare_inputs(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            device_index=self.device_index,
            arch=self.arch,
            stream=stream,
            arch_is_validated=True,
            timeout_ms=effective_timeout_ms,
            defer_missing_norms=True,
        )
        stream_handle = int(resolved_stream.cuda_stream)
        key = self._cache_key(inputs, stream_handle)
        slot, cache_hit, owns_slot_lock = self._get_or_create_slot(
            key,
            inputs,
            resolved_stream=resolved_stream,
            resolved_arch=resolved_arch,
        )
        if not owns_slot_lock:
            slot.lock.acquire()
        norm_mode = (
            "fused_bf16_pair_row_norm:" + ",".join(slot.norm_compute_fields)
            if slot.norm_compute_fields
            else "route_elided_internal_norms"
            if slot.internal_x_sq or slot.internal_c_sq
            else "explicit_precomputed"
        )
        try:
            with torch.cuda.device(self.device_index), torch.cuda.stream(resolved_stream):
                if cache_hit:
                    for name in ("x", "centroids", "out"):
                        slot.inputs[name] = inputs[name]
                    if not slot.internal_x_sq:
                        slot.inputs["x_sq"] = inputs["x_sq"]
                    if not slot.internal_c_sq:
                        slot.inputs["c_sq"] = inputs["c_sq"]
                    slot.inputs["_norm_mode"] = norm_mode
                    slot.launch_plan.direct_launcher.rebind_inputs(slot.inputs)
                # Register allocator ownership before any custom driver
                # launch, including timeout/error paths.
                _record_stream(
                    resolved_stream, *(slot.inputs[name] for name in ("x", "centroids", "x_sq", "c_sq", "out"))
                )
                if slot.norm_plan is not None:
                    norm_keepalive = slot.inputs["x_sq"] if slot.internal_x_sq else slot.inputs["c_sq"]
                    try:
                        slot.norm_plan.rebind(
                            slot.inputs["x"],
                            slot.inputs["centroids"],
                            slot.inputs["x_sq"],
                            slot.inputs["c_sq"],
                            stream=resolved_stream,
                        )
                        slot.norm_plan.launch(
                            stream=resolved_stream,
                            timeout_ms=effective_timeout_ms,
                        )
                    finally:
                        slot.norm_plan.release_bound_callers(
                            norm_keepalive,
                            stream=resolved_stream,
                        )
                slot.launch_plan.launch(stream=resolved_stream, timeout_ms=effective_timeout_ms)
                output = slot.inputs["out"]
        finally:
            try:
                slot.launch_plan.direct_launcher.release_bound_inputs()
            finally:
                for name in ("x", "centroids", "out"):
                    slot.inputs[name] = None
                if not slot.internal_x_sq:
                    slot.inputs["x_sq"] = None
                if not slot.internal_c_sq:
                    slot.inputs["c_sq"] = None
                slot.lock.release()

        if not return_info:
            return output
        return output, {
            "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
            "selected_route": slot.launch_plan.selected_route,
            "launch_entrypoint": slot.launch_plan.launch_entrypoint,
            "exact_launch_plan": True,
            "runtime_cache_hit": cache_hit,
            "assignment_launch_count": slot.launch_plan.launch_count,
            "norm_launch_count": int(slot.norm_plan is not None),
            "norm_compute_fields": list(slot.norm_compute_fields),
            "runtime_launch_count": slot.launch_plan.launch_count + int(slot.norm_plan is not None),
            "norm_mode": norm_mode,
            "arch": slot.launch_plan.arch,
            "device_index": slot.launch_plan.device_index,
            "stream_handle": slot.launch_plan.stream_handle,
        }

    def _get_or_create_slot(
        self,
        key: tuple[Any, ...],
        inputs: dict[str, Any],
        *,
        resolved_stream: Any,
        resolved_arch: str,
    ) -> tuple[_RuntimeSlot, bool, bool]:
        """Return a slot; a newly published slot is returned with its lock held."""

        import torch

        while True:
            with self._cache_lock:
                slot = self._slots.get(key)
                if slot is not None:
                    self._slots.move_to_end(key)
                    self._hits += 1
                    return slot, True, False
                pending = self._preparing.get(key)
                if pending is None:
                    if (
                        self.max_cached_shapes is not None
                        and len(self._slots) + len(self._preparing) >= self.max_cached_shapes
                    ):
                        raise RuntimeError(
                            "FlashKMeansAssignRuntime cache is full; call clear() only after in-flight work completes"
                        )
                    pending = _PendingPreparation()
                    self._preparing[key] = pending
                    break
            pending.event.wait()
            if pending.error is not None:
                raise RuntimeError("KMeans slot preparation failed in another thread") from pending.error

        slot: _RuntimeSlot | None = None
        try:
            with torch.cuda.device(self.device_index), torch.cuda.stream(resolved_stream):
                internal_x_sq = inputs["x_sq"] is None
                internal_c_sq = inputs["c_sq"] is None
                if internal_x_sq:
                    inputs["x_sq"] = torch.empty(
                        (int(inputs["B"]), int(inputs["N"])),
                        dtype=torch.float32,
                        device=inputs["x"].device,
                    )
                if internal_c_sq:
                    inputs["c_sq"] = torch.empty(
                        (int(inputs["B"]), int(inputs["K"])),
                        dtype=torch.float32,
                        device=inputs["x"].device,
                    )
                launch_plan = _dispatcher.prepare_launch_plan(
                    inputs,
                    arch=resolved_arch,
                    stream=resolved_stream,
                    timeout_ms=None,
                )
                bound_input_keys = set(launch_plan.direct_launcher.bound_input_keys)
                compute_x_sq = internal_x_sq and "x_sq" in bound_input_keys
                compute_c_sq = internal_c_sq and "c_sq" in bound_input_keys
                norm_plan = None
                if compute_x_sq or compute_c_sq:
                    from ._row_norm import prepare_bf16_pair_row_norm

                    norm_plan = prepare_bf16_pair_row_norm(
                        inputs["x"],
                        inputs["centroids"],
                        inputs["x_sq"],
                        inputs["c_sq"],
                        compute_x=compute_x_sq,
                        compute_c=compute_c_sq,
                        arch=resolved_arch,
                        stream=resolved_stream,
                    )
                norm_compute_fields = tuple(
                    name for name, enabled in (("x_sq", compute_x_sq), ("c_sq", compute_c_sq)) if enabled
                )
            slot = _RuntimeSlot(
                inputs=inputs,
                launch_plan=launch_plan,
                norm_plan=norm_plan,
                internal_x_sq=internal_x_sq,
                internal_c_sq=internal_c_sq,
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
                    self._slots[key] = slot
                    self._misses = old_misses + 1
                    self._preparing.pop(key, None)
                    pending.event.set()
                    publication_committed = True
                except BaseException as error:
                    if self._slots.get(key) is slot:
                        self._slots.pop(key, None)
                    self._misses = old_misses
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    pending.error = error
                    pending.event.set()
                    raise
        except BaseException as error:
            if not publication_committed:
                with self._cache_lock:
                    if self._slots.get(key) is slot:
                        self._slots.pop(key, None)
                        self._misses -= 1
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    if pending.error is None:
                        pending.error = error
                    pending.event.set()
            slot.lock.release()
            raise
        return slot, False, True


def init(
    *,
    device: Any = None,
    arch: str | None = None,
    timeout_ms: float | None = None,
    max_cached_shapes: int | None = None,
    compile: str = "lazy",
) -> FlashKMeansAssignRuntime:
    """Create one reusable runtime; shape-specific plans remain lazy."""

    return FlashKMeansAssignRuntime(
        device=device,
        arch=arch,
        timeout_ms=timeout_ms,
        max_cached_shapes=max_cached_shapes,
        compile=compile,
    )


def _require_aux_tensor(
    tensor: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype is not dtype:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def _record_stream(stream: Any, *tensors: Any) -> None:
    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


def _prepare_inputs(
    x: Any,
    centroids: Any,
    *,
    out: Any | None,
    x_sq: Any | None,
    c_sq: Any | None,
    device_index: int | None,
    arch: str | None,
    stream: Any,
    arch_is_validated: bool = False,
    timeout_ms: float | None = None,
    defer_missing_norms: bool = False,
) -> tuple[dict[str, Any], Any, str]:
    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
        raise TypeError("x and centroids must be CUDA torch.Tensor objects")
    if x.dtype is not torch.bfloat16 or centroids.dtype is not torch.bfloat16:
        raise TypeError("x and centroids dtype must be bfloat16")
    if x.ndim != 3 or centroids.ndim != 3 or not x.is_contiguous() or not centroids.is_contiguous():
        raise ValueError("x and centroids must be contiguous [B, rows, D] tensors")
    bsz, n_points, dim = map(int, x.shape)
    c_bsz, n_clusters, c_dim = map(int, centroids.shape)
    if (bsz, dim) != (c_bsz, c_dim) or x.device != centroids.device:
        raise ValueError("x and centroids batch/feature dimensions and device must match")
    input_device_index = x.device.index
    if input_device_index is None:
        input_device_index = torch.cuda.current_device()
    input_device_index = int(input_device_index)
    if device_index is not None and input_device_index != int(device_index):
        raise ValueError(
            f"Flash-KMeans runtime targets CUDA device {device_index}, got input device {input_device_index}"
        )
    with torch.cuda.device(input_device_index):
        resolved_stream = torch.cuda.current_stream(input_device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != input_device_index:
            raise ValueError(
                f"Flash-KMeans stream device {stream_device_index} does not match input device {input_device_index}"
            )
        with torch.cuda.stream(resolved_stream):
            if arch_is_validated:
                if arch is None:
                    raise RuntimeError("validated Flash-KMeans runtime must provide its active arch")
                resolved_arch = str(arch)
            else:
                detected_arch = str(detect_gpu_arch())
                resolved_arch = detected_arch if arch is None else str(arch)
                if resolved_arch != detected_arch:
                    raise ValueError(
                        "Flash-KMeans launch arch must match the active device: "
                        f"requested {resolved_arch}, detected {detected_arch}"
                    )
            if out is None:
                out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
            _require_aux_tensor(
                out,
                name="out",
                shape=(bsz, n_points),
                dtype=torch.int32,
                device=x.device,
            )
            compute_x_sq = x_sq is None
            compute_c_sq = c_sq is None
            if not defer_missing_norms:
                if compute_x_sq:
                    x_sq = torch.empty((bsz, n_points), dtype=torch.float32, device=x.device)
                if compute_c_sq:
                    c_sq = torch.empty((bsz, n_clusters), dtype=torch.float32, device=x.device)
            if x_sq is not None:
                _require_aux_tensor(
                    x_sq,
                    name="x_sq",
                    shape=(bsz, n_points),
                    dtype=torch.float32,
                    device=x.device,
                )
            if c_sq is not None:
                _require_aux_tensor(
                    c_sq,
                    name="c_sq",
                    shape=(bsz, n_clusters),
                    dtype=torch.float32,
                    device=x.device,
                )
            if (compute_x_sq or compute_c_sq) and not defer_missing_norms:
                # This single custom launch converts BF16, squares, and reduces
                # both row sets without PyTorch conversion/reduction temporaries.
                from ._row_norm import launch_bf16_pair_row_norm

                _record_stream(resolved_stream, x, centroids, x_sq, c_sq, out)
                launch_bf16_pair_row_norm(
                    x,
                    centroids,
                    x_sq,
                    c_sq,
                    compute_x=compute_x_sq,
                    compute_c=compute_c_sq,
                    stream=resolved_stream,
                    arch=resolved_arch,
                    timeout_ms=timeout_ms,
                )
            norm_mode = (
                "fused_bf16_pair_row_norm"
                if compute_x_sq and compute_c_sq
                else "mixed_fused_bf16_pair_row_norm"
                if compute_x_sq or compute_c_sq
                else "explicit_precomputed"
            )
            inputs = {
                "B": bsz,
                "N": n_points,
                "D": dim,
                "K": n_clusters,
                "dtype": "bfloat16",
                "x": x,
                "centroids": centroids,
                "x_sq": x_sq,
                "c_sq": c_sq,
                "out": out,
                "_norm_mode": norm_mode,
            }
    return inputs, resolved_stream, resolved_arch


def prepare_flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedFlashKMeansAssign:
    """Validate and prepare an allocation-free direct launch for exact tensors."""

    timeout_ms = _validate_timeout(timeout_ms)

    inputs, resolved_stream, resolved_arch = _prepare_inputs(
        x,
        centroids,
        out=out,
        x_sq=x_sq,
        c_sq=c_sq,
        device_index=None,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    import torch

    input_device_index = inputs["x"].device.index
    device_index = int(torch.cuda.current_device() if input_device_index is None else input_device_index)
    with torch.cuda.device(device_index), torch.cuda.stream(resolved_stream):
        launch_plan = _dispatcher.prepare_launch_plan(
            inputs,
            arch=resolved_arch,
            stream=resolved_stream,
            timeout_ms=timeout_ms,
        )
    return PreparedFlashKMeansAssign(
        inputs=inputs,
        launch_plan=launch_plan,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def flash_kmeans_assign_prepared(
    prepared: PreparedFlashKMeansAssign,
    *,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Submit a prepared plan without route selection, allocation, or packing."""

    if not isinstance(prepared, PreparedFlashKMeansAssign):
        raise TypeError("prepared must be returned by prepare_flash_kmeans_assign")
    effective_timeout_ms = prepared.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
    resolved_stream = prepared.stream if stream is None else stream
    _record_stream(
        resolved_stream,
        *(prepared.inputs[name] for name in ("x", "centroids", "x_sq", "c_sq", "out")),
    )
    result = _dispatcher.launch_prepared(
        prepared.launch_plan,
        stream=resolved_stream,
        timeout_ms=effective_timeout_ms,
    )
    produced = result.get("cluster_ids", prepared.out) if isinstance(result, dict) else prepared.out
    if produced is not prepared.out:
        prepared.out.copy_(produced)
    if not return_info:
        return prepared.out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": prepared.launch_plan.launch_entrypoint,
        "exact_launch_plan": True,
        "prepared_launch_count": prepared.launch_plan.launch_count,
        "arch": prepared.launch_plan.arch,
        "device_index": prepared.launch_plan.device_index,
        "stream_handle": prepared.launch_plan.stream_handle,
    }
    return prepared.out, info


def flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Assign points to centroids through a cold prepared/direct launch."""
    timeout_ms = _validate_timeout(timeout_ms)
    with launch_context(arch=arch, stream=stream, timeout_ms=timeout_ms):
        prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return flash_kmeans_assign_prepared(
            prepared,
            stream=prepared.stream,
            return_info=return_info,
        )
