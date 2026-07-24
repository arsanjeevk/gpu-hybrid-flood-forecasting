"""Runtime selection tests that do not require a CUDA-capable test host."""

from __future__ import annotations

from hybrid_flood.jax_solver.runtime import configure_jax_runtime


def test_explicit_jax_platform_is_preserved(monkeypatch) -> None:
    """User-selected backends must never be overwritten by auto-detection."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    status = configure_jax_runtime()

    assert status["explicit_jax_platforms"] == "cpu"
    assert status["effective_jax_platforms"] == "cpu"
    assert status["automatic_cpu_fallback"] is False


def test_cuda_probe_has_a_diagnostic_reason() -> None:
    """The preflight result must always explain why CUDA is/is not usable."""
    status = configure_jax_runtime()

    assert isinstance(status["cuda_driver"]["available"], bool)
    assert status["cuda_driver"]["reason"]
