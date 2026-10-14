"""Explicit configuration for the illustrative local Selene noise benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
from numbers import Integral, Real
from typing import Any

from selene_sim.backends.bundled_error_models import (
    DepolarizingErrorModel,
)

MODEL_LABEL = "illustrative_selene_depolarizing_v1"
PARAMETER_SOURCE = (
    "Quantinuum Selene README, Usage examples / Using Selene with HUGR and Guppy"
)
PARAMETER_SOURCE_URL = "https://github.com/Quantinuum/selene#usage-examples"
ASSUMPTIONS = (
    "The four values are illustrative Selene per-operation probabilities, "
    "not calibrated Quantinuum H2 hardware parameters.",
    "The bundled model uses stochastic Pauli gate errors, symmetric "
    "measurement-result flips, and X errors after reset.",
    "This model does not represent T1/T2, idle-time errors, leakage, or "
    "measurement crosstalk.",
)


@dataclass(frozen=True)
class DepolarizingNoiseConfig:
    """Required probabilities for the illustrative Selene benchmark."""

    p_1q: float
    p_2q: float
    p_meas: float
    p_init: float

    def __post_init__(self) -> None:
        for name in ("p_1q", "p_2q", "p_meas", "p_init"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite real number in [0, 1]")
            object.__setattr__(self, name, value)


def _validate_seed(random_seed: int | None) -> int | None:
    if random_seed is None:
        return None
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer or None")
    return int(random_seed)


def _validate_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def make_selene_error_model(
    config: DepolarizingNoiseConfig, *, random_seed: int | None
) -> DepolarizingErrorModel:
    """Construct the installed Selene model; the seed must be supplied explicitly."""
    if not isinstance(config, DepolarizingNoiseConfig):
        raise TypeError("config must be a DepolarizingNoiseConfig")
    return DepolarizingErrorModel(
        random_seed=_validate_seed(random_seed),
        p_1q=config.p_1q,
        p_2q=config.p_2q,
        p_meas=config.p_meas,
        p_init=config.p_init,
    )


def attach_error_model(emulator_config: Any, error_model: DepolarizingErrorModel) -> Any:
    """Return the emulator configuration with this model attached."""
    with_error_model = getattr(emulator_config, "with_error_model", None)
    if not callable(with_error_model):
        raise TypeError("emulator_config must support with_error_model(model)")
    return with_error_model(error_model)


def make_run_metadata(
    config: DepolarizingNoiseConfig,
    *,
    random_seed: int | None,
    shots: int | None,
    repeats: int | None,
    n_qubits: int | None,
) -> dict[str, Any]:
    """Create JSON-compatible provenance and execution metadata."""
    if not isinstance(config, DepolarizingNoiseConfig):
        raise TypeError("config must be a DepolarizingNoiseConfig")
    return {
        "model_label": MODEL_LABEL,
        "parameters": {
            "p_1q": config.p_1q,
            "p_2q": config.p_2q,
            "p_meas": config.p_meas,
            "p_init": config.p_init,
        },
        "parameter_provenance": {
            "description": PARAMETER_SOURCE,
            "source_url": PARAMETER_SOURCE_URL,
        },
        "assumptions": list(ASSUMPTIONS),
        "limitations": [
            "Illustrative benchmark only; not calibrated to H2 hardware.",
            "No T1/T2, idle-time, leakage, or crosstalk channel.",
        ],
        "package_versions": {
            "guppylang": version("guppylang"),
            "selene-sim": version("selene-sim"),
            "selene-core": version("selene-core"),
        },
        "random_seed": _validate_seed(random_seed),
        "shots": _validate_count("shots", shots),
        "repeats": _validate_count("repeats", repeats),
        "execution_settings": {
            "backend": "local Guppy/Selene emulator",
            "simulator": "Quest",
            "n_qubits": _validate_count("n_qubits", n_qubits),
        },
    }