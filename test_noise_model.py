"""Focused tests for the illustrative Selene depolarizing configuration."""
import math

import pytest
from selene_sim.backends.bundled_error_models import DepolarizingErrorModel

from noise_model import (
    MODEL_LABEL,
    PARAMETER_SOURCE_URL,
    DepolarizingNoiseConfig,
    attach_error_model,
    make_run_metadata,
    make_selene_error_model,
)

VALUES = {
    "p_1q": 1e-5,
    "p_2q": 1e-6,
    "p_meas": 1e-2,
    "p_init": 1e-3,
}


def test_config_requires_all_probabilities_and_accepts_valid_values():
    config = DepolarizingNoiseConfig(**VALUES)
    assert config.p_1q == pytest.approx(1e-5)
    assert config.p_2q == pytest.approx(1e-6)
    assert config.p_meas == pytest.approx(1e-2)
    assert config.p_init == pytest.approx(1e-3)


def test_config_does_not_supply_probability_defaults():
    with pytest.raises(TypeError):
        DepolarizingNoiseConfig()


def test_config_rejects_missing_probability():
    incomplete = dict(VALUES)
    incomplete.pop("p_init")
    with pytest.raises(TypeError):
        DepolarizingNoiseConfig(**incomplete)


@pytest.mark.parametrize("field", ["p_1q", "p_2q", "p_meas", "p_init"])
@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_config_rejects_invalid_probabilities(field, value):
    params = dict(VALUES)
    params[field] = value
    with pytest.raises(ValueError, match=field):
        DepolarizingNoiseConfig(**params)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_config_accepts_probability_boundaries(value):
    config = DepolarizingNoiseConfig(**{name: value for name in VALUES})
    assert all(getattr(config, name) == value for name in VALUES)


def test_factory_constructs_installed_selene_model_and_forwards_seed():
    config = DepolarizingNoiseConfig(**VALUES)
    model = make_selene_error_model(config, random_seed=8123)

    assert isinstance(model, DepolarizingErrorModel)
    assert model.random_seed == 8123
    for name, value in VALUES.items():
        assert getattr(model, name) == pytest.approx(value)


def test_factory_requires_explicit_seed_argument():
    config = DepolarizingNoiseConfig(**VALUES)
    with pytest.raises(TypeError):
        make_selene_error_model(config)


def test_emulator_helper_attaches_model_and_preserves_other_settings():
    class FakeEmulatorConfig:
        def __init__(self, shots, seed, error_model=None):
            self.shots = shots
            self.seed = seed
            self.error_model = error_model

        def with_error_model(self, model):
            return FakeEmulatorConfig(self.shots, self.seed, model)

    config = DepolarizingNoiseConfig(**VALUES)
    model = make_selene_error_model(config, random_seed=99)
    original = FakeEmulatorConfig(shots=17, seed=23)

    configured = attach_error_model(original, model)

    assert configured is not original
    assert configured.shots == 17
    assert configured.seed == 23
    assert configured.error_model is model
    assert original.error_model is None


def test_metadata_records_provenance_versions_and_execution_settings():
    config = DepolarizingNoiseConfig(**VALUES)
    metadata = make_run_metadata(
        config,
        random_seed=8123,
        shots=5000,
        repeats=1,
        n_qubits=12,
    )

    assert metadata["model_label"] == MODEL_LABEL
    assert metadata["parameters"] == VALUES
    assert metadata["parameter_provenance"]["source_url"] == PARAMETER_SOURCE_URL
    assert metadata["assumptions"]
    assert metadata["package_versions"] == {
        "guppylang": "0.21.16",
        "selene-sim": "0.2.18",
        "selene-core": "0.2.11",
    }
    assert metadata["random_seed"] == 8123
    assert metadata["shots"] == 5000
    assert metadata["repeats"] == 1
    assert metadata["execution_settings"]["n_qubits"] == 12


def test_metadata_requires_run_settings_to_be_explicit():
    config = DepolarizingNoiseConfig(**VALUES)
    with pytest.raises(TypeError):
        make_run_metadata(config)
# Source-backed circuit for a small real Selene integration smoke test.
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit


@guppy
def tiny_selene_noise_smoke() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    m0 = measure(q0)
    m1 = measure(q1)
    result("m0", m0)
    result("m1", m1)


def test_model_executes_through_selene_without_state_dump():
    config = DepolarizingNoiseConfig(**VALUES)
    model = make_selene_error_model(config, random_seed=20261003)
    emulator = (
        tiny_selene_noise_smoke.emulator(n_qubits=2)
        .with_shots(4)
        .with_seed(20261003)
    )

    output = attach_error_model(emulator, model).run()

    assert len(output.results) == 4
    assert all({"m0", "m1"} <= {tag for tag, _ in shot.entries} for shot in output.results)
