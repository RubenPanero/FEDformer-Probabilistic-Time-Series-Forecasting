from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "optimization"
    / "critical_bottlenecks_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "critical_bottlenecks_benchmark", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.benchmark
def test_critical_bottlenecks_benchmarks_collect_metrics() -> None:
    metrics = MODULE.run_all_benchmarks(refresh=True)

    assert set(metrics) == {"mc_dropout", "fourier_modes", "flow_checkpointing"}
    for benchmark_name, result in metrics.items():
        assert result["baseline_time_s"] >= 0.0, benchmark_name
        assert result["optimized_time_s"] >= 0.0, benchmark_name
        assert "time_delta_pct" in result, benchmark_name
        assert result["baseline_peak_bytes"] >= 0.0, benchmark_name
        assert result["optimized_peak_bytes"] >= 0.0, benchmark_name
        assert "memory_delta_pct" in result, benchmark_name


@pytest.mark.benchmark
def test_critical_bottlenecks_benchmark_report_exposes_only_contractual_guardrails() -> (
    None
):
    report = MODULE.run_benchmark_report()

    assert set(report) == {"benchmarks", "guardrails"}
    assert set(report["benchmarks"]) == {
        "mc_dropout",
        "fourier_modes",
        "flow_checkpointing",
    }
    assert set(report["guardrails"]) == {"mc_dropout", "fourier_modes"}
    assert "flow_checkpointing" not in report["guardrails"]

    mc_dropout_guardrail = report["guardrails"]["mc_dropout"]
    assert mc_dropout_guardrail["metric"] == "time_delta_pct"
    assert mc_dropout_guardrail["max_allowed_regression_pct"] == 5.0
    assert isinstance(mc_dropout_guardrail["observed_delta_pct"], float)
    assert mc_dropout_guardrail["status"] in {"passed", "failed"}
    assert isinstance(mc_dropout_guardrail["passed"], bool)

    fourier_guardrail = report["guardrails"]["fourier_modes"]
    assert fourier_guardrail["metric"] == "time_delta_pct"
    assert fourier_guardrail["max_allowed_regression_pct"] == 15.0
    assert isinstance(fourier_guardrail["observed_delta_pct"], float)
    assert fourier_guardrail["status"] in {"passed", "failed"}
    assert isinstance(fourier_guardrail["passed"], bool)


def test_critical_bottlenecks_guardrails_fail_when_time_delta_exceeds_budget() -> None:
    metrics = {
        "mc_dropout": {
            "time_delta_pct": 8.5,
        },
        "fourier_modes": {
            "time_delta_pct": 16.0,
        },
    }

    guardrails = MODULE.evaluate_guardrails(metrics)

    assert guardrails["mc_dropout"]["passed"] is False
    assert guardrails["mc_dropout"]["status"] == "failed"
    assert "8.5" in guardrails["mc_dropout"]["failure_reason"]
    assert "5.0" in guardrails["mc_dropout"]["failure_reason"]
    assert guardrails["fourier_modes"]["passed"] is False
    assert guardrails["fourier_modes"]["status"] == "failed"
    assert "16.0" in guardrails["fourier_modes"]["failure_reason"]
    assert "15.0" in guardrails["fourier_modes"]["failure_reason"]


def test_critical_bottlenecks_guardrails_pass_with_empty_failure_reason_within_budget() -> (
    None
):
    metrics = {
        "mc_dropout": {
            "time_delta_pct": 4.9,
        },
        "fourier_modes": {
            "time_delta_pct": 14.9,
        },
    }

    guardrails = MODULE.evaluate_guardrails(metrics)

    assert guardrails["mc_dropout"]["passed"] is True
    assert guardrails["mc_dropout"]["status"] == "passed"
    assert guardrails["mc_dropout"]["failure_reason"] == ""
    assert guardrails["fourier_modes"]["passed"] is True
    assert guardrails["fourier_modes"]["status"] == "passed"
    assert guardrails["fourier_modes"]["failure_reason"] == ""


def test_critical_bottlenecks_guardrails_do_not_create_budgets_for_flow_checkpointing() -> (
    None
):
    metrics = {
        "mc_dropout": {
            "time_delta_pct": 0.5,
        },
        "fourier_modes": {
            "time_delta_pct": 1.5,
        },
        "flow_checkpointing": {
            "time_delta_pct": 999.0,
        },
    }

    guardrails = MODULE.evaluate_guardrails(metrics)

    assert set(guardrails) == {"mc_dropout", "fourier_modes"}
    assert "flow_checkpointing" not in guardrails
