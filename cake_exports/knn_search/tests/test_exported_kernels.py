from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE_NAME = 'flashlib_cake_knn_search'


def _manifest() -> dict:
    return json.loads((SRC / PACKAGE_NAME / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_matches_package_exports():
    pkg = importlib.import_module(PACKAGE_NAME)
    manifest = _manifest()
    names = [entry["name"] for entry in manifest["kernels"]]

    assert names
    assert set(pkg.KERNELS) == set(names)
    assert manifest["package"] == PACKAGE_NAME
    package_exports = manifest.get("export_plan", {}).get("package_exports", {})
    for public_name in package_exports:
        assert hasattr(pkg, public_name), public_name
        assert public_name in pkg.__all__, public_name

    if "export_plan" in manifest:
        entrypoints = manifest["export_plan"].get("entrypoints", {})
        assert set(entrypoints) == {
            "python_interface",
            "correctness_test",
            "performance_benchmark",
        }
        for path in entrypoints.values():
            assert (ROOT / path).is_file(), path


def test_manifest_sources_exist_and_contain_symbols():
    manifest = _manifest()
    package_dir = SRC / PACKAGE_NAME

    for entry in manifest["kernels"]:
        source_path = package_dir / entry["source"]
        assert source_path.is_file(), entry["source"]
        source = source_path.read_text(encoding="utf-8")
        assert entry["symbol"] in source
        assert entry["parameters"]


def test_package_import_and_source_text_do_not_require_cuda_runtime():
    pkg = importlib.import_module(PACKAGE_NAME)

    for name, kernel in pkg.KERNELS.items():
        assert kernel.source_text().startswith("typedef "), name
        assert kernel.parameters


def test_exported_repo_docs_and_benchmarks_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "RESULTS.md").is_file()
    assert (ROOT / "benchmarks" / "benchmark_exported_kernels.py").is_file()
    assert (ROOT / "benchmarks" / "benchmark_shapes.py").is_file()
    assert (ROOT / "benchmarks" / "workload.py").is_file()


def test_benchmark_runtime_requires_cupti_without_event_or_wall_clock_fallback():
    source = (SRC / PACKAGE_NAME / "_benchmark.py").read_text(encoding="utf-8")
    assert "activity_register_callbacks" in source
    assert "torch.cuda.Event" not in source
    assert "perf_counter" not in source


def test_launch_argument_count_is_checked_before_compilation():
    pkg = importlib.import_module(PACKAGE_NAME)
    kernel = next(iter(pkg.KERNELS.values()))
    bad_args = [object()] * (len(kernel.parameters) + 1)

    with pytest.raises(TypeError, match="expects"):
        kernel.launch(*bad_args, grid=(1, 1, 1))

