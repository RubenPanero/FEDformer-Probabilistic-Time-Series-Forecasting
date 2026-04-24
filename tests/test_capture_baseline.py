from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "docs" / "capture_baseline.py"


def _load_module():
    assert MODULE_PATH.exists(), f"Missing baseline capture module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("capture_baseline", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_baseline_report_includes_environment_hotspots_and_benchmarks() -> None:
    module = _load_module()
    report = module.build_baseline_report(label="unit-test")

    assert report["label"] == "unit-test"
    assert report["metadata"]["python_version"]
    assert report["metadata"]["python_executable"]
    assert report["metadata"]["platform"]
    assert report["metadata"]["torch_version"]
    assert isinstance(report["metadata"]["cuda_available"], bool)
    assert report["metadata"]["timestamp_utc"]
    assert {hotspot["id"] for hotspot in report["hotspots"]} == {
        "mc_dropout",
        "fourier_attention",
        "flow_log_prob",
        "dataset_fit_transform",
        "optuna_subprocess",
    }
    assert set(report["benchmarks"]) == {
        "mc_dropout",
        "fourier_modes",
        "flow_checkpointing",
    }


def test_write_baseline_report_persists_json_artifact(tmp_path: Path) -> None:
    module = _load_module()

    output_path = module.write_baseline_report(output_dir=tmp_path, label="persisted")

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["label"] == "persisted"
    assert payload["metadata"]["timestamp_utc"]
    assert output_path.suffix == ".json"
