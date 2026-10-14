"""Tests for the repetition-code component and its Selene circuit path."""
import ast

import numpy as np
import pytest

import repetition_code as rc
from qec_repetition_demo import (
    bitflip_repetition_q0_x_error_demo,
    run_selene_repetition_component,
    unprotected_q0_x_error_demo,
)


@pytest.mark.parametrize("variant", ["bit_flip", "phase_flip"])
def test_encoding_is_normalized_and_matches_requested_variant(variant):
    state = rc.encode_state(0.6, 0.8, variant)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    if variant == "bit_flip":
        expected = np.zeros(8, dtype=complex)
        expected[0] = 0.6
        expected[7] = 0.8
        assert state == pytest.approx(expected)
    else:
        assert np.count_nonzero(np.abs(state) > 1e-12) == 8


@pytest.mark.parametrize("variant", ["bit_flip", "phase_flip"])
def test_no_error_has_zero_syndrome_and_no_correction(variant):
    state = rc.encode_state(0.6, 0.8, variant)
    syndrome = rc.measure_syndrome(state, variant)
    assert syndrome == (0, 0)
    assert rc.decode(syndrome) is None


@pytest.mark.parametrize(
    ("variant", "expected_gate"),
    [("bit_flip", rc.X), ("phase_flip", rc.Z)],
)
@pytest.mark.parametrize("error_qubit", [0, 1, 2])
def test_single_supported_pauli_error_is_identified_and_corrected(
    variant, expected_gate, error_qubit
):
    state, corrected, syndrome, faulty_qubit = rc.run_full_correction_cycle(
        0.6, 0.8, error_qubit, variant
    )
    assert syndrome == {
        0: (1, 0),
        1: (1, 1),
        2: (0, 1),
    }[error_qubit]
    assert faulty_qubit == error_qubit
    assert abs(np.vdot(state, corrected)) ** 2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("variant", ["bit_flip", "phase_flip"])
@pytest.mark.parametrize("error_pair", [(0, 1), (0, 2), (1, 2)])
def test_two_data_errors_are_outside_the_single_error_guarantee(variant, error_pair):
    state = rc.encode_state(0.6, 0.8, variant)
    for qubit_index in error_pair:
        state = rc.apply_error(state, qubit_index, variant)
    syndrome = rc.measure_syndrome(state, variant)
    guessed_qubit = rc.decode(syndrome)
    corrected = rc.apply_correction(state, guessed_qubit, variant)
    fidelity = abs(np.vdot(rc.encode_state(0.6, 0.8, variant), corrected)) ** 2
    assert fidelity < 0.99


def test_decoder_rejects_unrecognized_syndrome():
    with pytest.raises(ValueError, match="unrecognized syndrome"):
        rc.decode((2, 0))


def test_encoder_and_source_builder_reject_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        rc.encode_state(0.6, 0.8, "other")
    with pytest.raises(ValueError, match="unknown variant"):
        rc.build_encode_and_correct_guppy_source("other")


@pytest.mark.parametrize("variant", ["bit_flip", "phase_flip"])
def test_existing_generated_guppy_source_is_valid_python(variant):
    source = rc.build_encode_and_correct_guppy_source(variant)
    ast.parse(source)
    assert "repetition_code_round.check()" in source


def test_ideal_selene_corrects_the_injected_x_error():
    report = run_selene_repetition_component(
        shots=8, seed=20261004, n_processes=1
    )
    assert report["compiled_modules"] == {"unprotected": 1, "encoded": 1}
    assert report["unprotected_counts"] == {"1": 8}
    assert report["encoded_syndrome_counts_s0_s1"] == {"10": 8}
    assert report["encoded_decoded_counts"] == {"0": 8}
    assert report["encoded_recovery_success_probability"] == 1.0
    assert report["full_logical_ipea"] is False
def test_phase_flip_generated_guppy_corrects_physical_z_error_in_selene(tmp_path):
    """Check the generated phase-flip path corrects Z in its H-rotated frame."""
    import importlib.util
    import sys

    source = rc.build_encode_and_correct_guppy_source("phase_flip")
    hadamard_wrap = "    h(q0)\n    h(q1)\n    h(q2)\n"
    assert hadamard_wrap in source
    source = source.replace(hadamard_wrap, "    z(q0)\n" + hadamard_wrap, 1)
    source = source.replace(
        "from guppylang.std.quantum import cx, h, measure, qubit, x, discard",
        "from guppylang.std.quantum import cx, h, measure, qubit, x, z, discard",
    )
    source += '''

from guppylang.std.debug import state_result

@guppy
def phaseflip_z_error_demo() -> None:
    q0 = qubit()
    h(q0)
    q0 = repetition_code_round(q0)
    state_result("recovered", q0)
    discard(q0)
'''
    path = tmp_path / "generated_phaseflip_probe.py"
    path.write_text(source, encoding="utf-8")
    module_name = "generated_phaseflip_probe_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        compiled = module.phaseflip_z_error_demo.compile()
        emulator_result = (
            module.phaseflip_z_error_demo.emulator(n_qubits=5)
            .with_seed(20261012)
            .run()
        )
    finally:
        sys.modules.pop(module_name, None)

    assert len(compiled.modules) == 1
    recovered = emulator_result.partial_state_dicts()[0][
        "recovered"
    ].state_distribution()[0].state
    expected = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    assert abs(np.vdot(expected, recovered)) ** 2 == pytest.approx(1.0, abs=1e-8)