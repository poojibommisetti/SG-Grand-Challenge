"""Focused Selene tests for targeted X-correction at the k=0 boundary."""
from __future__ import annotations

import pytest

from qec_logical_ipea import run_qec_logical_ipea_5bit
from targeted_fault_injection import run_unprotected_fault_ipea


@pytest.mark.parametrize(
    ("location", "fault", "syndrome"),
    [
        ("d0", "x_post_k0_d0", "10"),
        ("d1", "x_post_k0_d1", "11"),
        ("d2", "x_post_k0_d2", "01"),
    ],
)
def test_encoded_single_x_at_each_data_wire_is_recovered(location, fault, syndrome):
    shots = 128
    result = run_qec_logical_ipea_5bit(
        shots=shots, seed=20261031, n_processes=1, fault_injection=fault
    )
    assert result["settings"]["fault_injection"] == fault
    assert result["syndrome_counts_by_iteration_and_stage"]["k0_post"] == {syndrome: shots}
    assert sum(result["measured_counts_msb_first"].values()) == shots
    assert result["measured_counts_msb_first"].get("10100", 0) / shots > 0.8
    assert result["within_ideal_distribution_tolerance"]


def test_unprotected_x_at_same_feedforward_boundary_changes_phase_distribution():
    shots = 128
    clean = run_unprotected_fault_ipea(
        shots=shots, seed=20261031, n_processes=1, fault=None
    )
    faulted = run_unprotected_fault_ipea(
        shots=shots, seed=20261031, n_processes=1, fault="x_post_k0"
    )
    assert faulted["fault"] == "x_post_k0"
    assert faulted["fault_boundary"].startswith("after exact controlled-U^1 (k=0)")
    assert sum(faulted["measured_counts_msb_first"].values()) == shots
    assert clean["reference_probability"] > 0.8
    assert faulted["reference_probability"] < 0.75


def test_z_fault_is_undetected_and_double_x_is_a_limitation():
    shots = 128
    z_result = run_qec_logical_ipea_5bit(
        shots=shots, seed=20261031, n_processes=1, fault_injection="z_post_k0_d1"
    )
    assert z_result["syndrome_counts_by_iteration_and_stage"]["k0_post"] == {"00": shots}
    two_x_result = run_qec_logical_ipea_5bit(
        shots=shots, seed=20261031, n_processes=1, fault_injection="x2_post_k0_d0_d1"
    )
    assert two_x_result["syndrome_counts_by_iteration_and_stage"]["k0_post"] == {"01": shots}
    assert not two_x_result["within_ideal_distribution_tolerance"]
