from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _root
from ._dispatch_runtime import _import_dispatch_module, dispatch_launch_options

_WEAVE_PREFIX = 'loom.examples.weave.'
_ROOT_MODULE = 'flash_kmeans_assign_dispatcher'
_ROOT_CALLABLE = 'launch_for_eval'
_EXACT_LAUNCH_SPECS = {}


@dataclass(frozen=True)
class RouteDecision:
    """Resolved semantic route with a launcher that can be reused directly."""

    route_id: str
    launch_entrypoint: str
    launcher: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False)
    exact_contract: bool = False

    def launch(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
        timeout_ms: float | None = None,
    ) -> Any:
        with dispatch_launch_options(stream=stream, timeout_ms=timeout_ms):
            return self.launcher(inputs)


def _route_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, str, bool, bool]:
    dtype = str(inputs.get("dtype", "bfloat16"))
    if dtype.startswith("torch."):
        dtype = dtype[6:]
    return (
        *(int(inputs[name]) for name in ("B", "Q", "M", "D", "K")),
        dtype,
        bool(inputs.get("self_search", False)),
        bool(inputs.get("force_fallback", False)),
    )


@lru_cache(maxsize=None)
def _load_launcher(module_name: str, callable_name: str) -> Callable[[dict[str, Any]], Any]:
    module = _import_dispatch_module(module_name)
    launcher = getattr(module, callable_name, None)
    if not callable(launcher):
        raise RuntimeError(f"resolved dispatcher launcher is not callable: {module_name}:{callable_name}")
    return launcher


@lru_cache(maxsize=None)
def _make_decision(
    route_id: str,
    module_name: str,
    callable_name: str,
    exact_contract: bool,
) -> RouteDecision:
    return RouteDecision(
        route_id=route_id,
        launch_entrypoint=f"{_WEAVE_PREFIX}{module_name}:{callable_name}",
        launcher=_load_launcher(module_name, callable_name),
        exact_contract=exact_contract,
    )


def _entrypoint_spec(entrypoint: object) -> tuple[str, str] | None:
    if not isinstance(entrypoint, str):
        return None
    module_name, separator, callable_name = entrypoint.partition(":")
    if not separator or not module_name.startswith(_WEAVE_PREFIX) or not callable_name.isidentifier():
        return None
    return module_name.removeprefix(_WEAVE_PREFIX), callable_name


def _generic_decision(inputs: dict[str, Any]) -> RouteDecision:
    info_fn = getattr(_root, "route_info", None)
    info = dict(info_fn(inputs)) if callable(info_fn) else {}
    route_id = info.get("selected_route", info.get("route"))
    if route_id is None:
        select_route = getattr(_root, "selected_route", None)
        route_id = select_route(inputs) if callable(select_route) else _ROOT_CALLABLE
    entrypoint = info.get("resolved_launch_entrypoint") or info.get("selected_entrypoint")
    spec = _entrypoint_spec(entrypoint)
    if spec is None:
        spec = (_ROOT_MODULE, _ROOT_CALLABLE)
    return _make_decision(str(route_id), *spec, False)


def resolve_route(inputs: dict[str, Any]) -> RouteDecision:
    """Resolve once; exact exported shapes never re-enter the root dispatcher."""

    spec = _EXACT_LAUNCH_SPECS.get(_route_key(inputs))
    if spec is None:
        return _generic_decision(inputs)
    return _make_decision(*spec, True)
