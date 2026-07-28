"""Publication configuration invariants for the three-hour GPU experiment."""

from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    return OmegaConf.load(ROOT / relative)


def test_three_hour_solver_and_figure_windows_match() -> None:
    anuga = _load("config/anuga/baseline.yaml")
    jax = _load("config/jax_solver/shallow_water.yaml")
    viz = _load("config/viz/report.yaml")
    assert float(anuga.duration_s) == float(jax.duration_s) == 10_800.0
    assert list(viz.key_times_s) == [3600.0, 7200.0, 10_800.0]
    assert 10_800.0 % float(anuga.outputstep_s) == 0.0
    assert 10_800.0 % float(jax.output_interval_s) == 0.0


def test_publication_accelerated_phases_reject_cpu_fallback() -> None:
    jax = _load("config/jax_solver/shallow_water.yaml")
    model = _load("config/model/residual_cnn.yaml")
    hybrid = _load("config/hybrid/coupled.yaml")
    benchmark = _load("config/benchmark/frameworks.yaml")
    assert bool(jax.execution.require_gpu)
    assert bool(model.execution.require_gpu)
    assert bool(hybrid.execution.require_gpu)
    assert bool(benchmark.require_gpu)
    assert not bool(benchmark.allow_cpu)


def test_hybrid_feedback_is_selected_without_test_leakage() -> None:
    hybrid = _load("config/hybrid/coupled.yaml")
    selection = hybrid.rollout.relaxation_selection
    assert bool(selection.enabled)
    assert list(selection.candidates) == [0.0, 0.01, 0.025, 0.05, 0.1]
    assert str(selection.metric) == "validation_hybrid_depth_rmse"
