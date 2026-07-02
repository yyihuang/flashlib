"""Exact D320 producer-grid recurrence candidate for kNN search.

Minimum target architecture: sm_100a.  This exact-shape variant preserves the
parent's TMA-fed tcgen05 (K128 + K128 + K64) producer, four-database-tile
split-local K10 recurrence, and 48-way Weave merge.  It changes only producer
ownership: the launcher exposes all 192 work items as CTAs instead of capping
the persistent producer grid at 148.  Both contract outputs remain parent-owned
Weave buffers.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _decode_capture, _ir_proxy
import argparse
import json
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from .. import _dispatch_runtime as eval_mod
from . import knn_build_d320_ownership_topology_9150_v1 as parent
MODULE = 'loom.examples.weave.knn_build_d320_producer_recurrence_search_f556_v1'
ROUTE_PREFIX = 'knn_build_d320_producer_recurrence_search_f556_v1'
TARGET_SHAPE = 'search_rect_highd_b1_q512_m12000_d320_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
SPLIT_COUNT = parent.SPLIT_COUNT
DB_TILES_PER_SPLIT = parent.DB_TILES_PER_SPLIT
TOTAL_WORK = parent.TOTAL_WORK
PRODUCER_GRID = TOTAL_WORK
ir = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_build_d320_producer_recurrence_search_f556_v1:ir"}'))

def _is_target(inputs: dict[str, Any]) -> bool:
    return parent._is_target(inputs)

@contextmanager
def _full_producer_grid() -> Iterator[None]:
    """Temporarily make the parent launch one CTA for each producer work item."""
    original = parent.exact_d320.GRID_DIM_DEFAULT
    parent.exact_d320.GRID_DIM_DEFAULT = PRODUCER_GRID
    try:
        yield
    finally:
        parent.exact_d320.GRID_DIM_DEFAULT = original

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback or not _is_target(inputs):
        return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return f'{ROUTE_PREFIX}:{TARGET_SHAPE}:exact_d320_k128_k128_k64:splits{SPLIT_COUNT}:dbtiles{DB_TILES_PER_SPLIT}:work{TOTAL_WORK}:grid{PRODUCER_GRID}'

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if force_fallback:
        parent.launch_from_contract_inputs(inputs)
        return
    if not _is_target(inputs):
        parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    with _full_producer_grid():
        parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    wanted = set(shape_labels)
    return [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
