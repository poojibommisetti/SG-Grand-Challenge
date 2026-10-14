"""Tests for the logical controlled-U component executed through Selene."""
import numpy as np
import pytest

import repetition_code as rc
from ipea import phase_correction_angle
from qec_logical_ops import run_logical_controlled_u_probe


@pytest.mark.parametrize("mode", ["zero", "one", "plus"])
def test_logical_basis_and_superposition_match_exact_controlled_matrix(mode):
    report = run_logical_controlled_u_probe(mode=mode)
    assert report["compiled_modules"] == 1
    assert report["selene_n_qubits"] == 7
    assert report["pre_syndrome_counts"] == {"00": 1}
    assert report["post_syndrome_counts"] == {"00": 1}
    assert report["logical_state_fidelity_to_exact_matrix"] == pytest.approx(
        1.0, abs=1e-8
    )
    assert report["validation_passed"]


@pytest.mark.parametrize("stage", ["pre_error", "post_error"])
@pytest.mark.parametrize(
    ("data_index", "expected_syndrome"), [(0, "10"), (1, "11"), (2, "01")]
)
def test_selene_recovers_single_x_at_each_data_wire_and_boundary(
    stage, data_index, expected_syndrome
):
    kwargs = {stage: data_index}
    report = run_logical_controlled_u_probe(mode="plus", **kwargs)
    syndrome_key = (
        "pre_syndrome_counts" if stage == "pre_error" else "post_syndrome_counts"
    )
    other_key = (
        "post_syndrome_counts" if stage == "pre_error" else "pre_syndrome_counts"
    )
    assert report[syndrome_key] == {expected_syndrome: 1}
    assert report[other_key] == {"00": 1}
    assert report["logical_state_fidelity_to_exact_matrix"] == pytest.approx(
        1.0, abs=1e-8
    )
    assert report["validation_passed"]


def test_existing_ipea_phase_correction_is_applied_as_rz_in_half_turns():
    lower_bits = [1, 0, 1, 0]
    expected_angle = phase_correction_angle(0, lower_bits, n_bits=5)
    report = run_logical_controlled_u_probe(
        mode="plus", phase_correction_bits=lower_bits, n_bits=5
    )
    assert report["phase_correction_angle_radians"] == pytest.approx(expected_angle)
    assert report["logical_state_fidelity_to_exact_matrix"] == pytest.approx(
        1.0, abs=1e-8
    )
    assert report["validation_passed"]


def test_bitflip_code_does_not_detect_a_single_z_error():
    logical_plus = rc.encode_state(1 / np.sqrt(2), 1 / np.sqrt(2), "bit_flip")
    z_on_d0 = np.kron(np.kron(rc.Z, rc.I2), rc.I2)
    errored = z_on_d0 @ logical_plus
    assert rc.measure_syndrome(errored, "bit_flip") == (0, 0)
    assert abs(np.vdot(logical_plus, errored)) ** 2 == pytest.approx(0.0, abs=1e-12)


def test_two_x_errors_can_turn_into_a_logical_bit_flip():
    logical_zero = rc.encode_state(1.0, 0.0, "bit_flip")
    errored = rc._single_qubit_gate_3(rc.X, 0) @ logical_zero
    errored = rc._single_qubit_gate_3(rc.X, 1) @ errored
    syndrome = rc.measure_syndrome(errored, "bit_flip")
    recovered = rc.apply_correction(errored, rc.decode(syndrome), "bit_flip")
    assert abs(np.vdot(logical_zero, recovered)) ** 2 == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("bad_mode", ["", "invalid", None])
def test_invalid_logical_mode_is_rejected(bad_mode):
    with pytest.raises(ValueError, match="mode must be"):
        run_logical_controlled_u_probe(mode=bad_mode)