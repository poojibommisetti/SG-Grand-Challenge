"""
test_state_prep.py
Member 1 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Unit tests for state_prep.py. These must pass BEFORE the Guppy circuit
(prepare_ground_state) is trusted enough to wire into ipea.py -- this is
the "test locally before simulator/emulator" step from the team's
validation chain.

Run:
    pytest test_state_prep.py -v
"""

import math

import numpy as np
import pytest

from state_prep import (
    compute_theta,
    ry_matrix,
    cnot_matrix,
    state_prep_unitary,
    load_ground_state,
    verify_against_e0,
)


def test_ry_matrix_is_unitary():
    """Basic sanity check: Ry(theta) must satisfy U^dagger U = I for any
    theta, or every downstream result is meaningless."""
    for theta in [0.0, 0.7, -2.916, math.pi, -math.pi / 3]:
        U = ry_matrix(theta)
        assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-12)


def test_ry_zero_angle_is_identity():
    """theta=0 should leave |0> untouched -- catches a sign/offset bug
    in the angle convention immediately."""
    U = ry_matrix(0.0)
    assert np.allclose(U, np.eye(2), atol=1e-12)


def test_cnot_matches_truth_table():
    """CNOT(q0=control, q1=target) must match the textbook truth table
    exactly, in the |q0 q1> basis ordering used throughout this project."""
    U = cnot_matrix()
    basis = {
        "00": np.array([1, 0, 0, 0], dtype=complex),
        "01": np.array([0, 1, 0, 0], dtype=complex),
        "10": np.array([0, 0, 1, 0], dtype=complex),
        "11": np.array([0, 0, 0, 1], dtype=complex),
    }
    expected_flip = {"00": "00", "01": "01", "10": "11", "11": "10"}
    for inp, out in expected_flip.items():
        result = U @ basis[inp]
        assert np.allclose(result, basis[out], atol=1e-12), (
            f"CNOT({inp}) should give {out}"
        )


def test_compute_theta_recovers_known_angle():
    """Round-trip check: build (a, b) from a chosen theta, recompute theta
    from (a, b), confirm we get the same angle back. This isolates the
    atan2 logic from e0.json entirely."""
    for true_theta in [0.3, -0.3, 2.0, -2.916, math.pi - 0.01]:
        a = math.cos(true_theta / 2.0)
        b = math.sin(true_theta / 2.0)
        recovered = compute_theta(a, b)
        assert math.isclose(recovered, true_theta, abs_tol=1e-9)


def test_compute_theta_preserves_sign_of_b():
    """This is the specific failure mode atan2 (vs. arccos) guards
    against: a negative b must not collapse to the same theta as a
    positive b with the same magnitude."""
    theta_pos = compute_theta(0.1115, 0.9938)
    theta_neg = compute_theta(0.1115, -0.9938)
    assert not math.isclose(theta_pos, theta_neg, abs_tol=1e-3)


def test_full_circuit_reproduces_e0_ground_state():
    """The end-to-end check: does the 2-gate circuit, built from a theta
    derived purely from e0.json, actually reproduce e0.json's own stored
    ground-state vector? This is the test that must pass before
    state_prep is considered done."""
    ok, theta, produced = verify_against_e0(tol=1e-6)
    assert ok, (
        f"State-prep circuit does not reproduce e0.json ground state. "
        f"theta={theta}, produced={produced}"
    )


def test_produced_state_is_normalized():
    """A basic physical sanity check independent of e0.json: whatever
    state comes out, it must be a valid (normalized) quantum state."""
    a, b, _ = load_ground_state()
    theta = compute_theta(a, b)
    U = state_prep_unitary(theta)
    produced = U @ np.array([1, 0, 0, 0], dtype=complex)
    norm = np.sum(np.abs(produced) ** 2)
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_produced_state_has_zero_amplitude_on_01_and_10():
    """Confirms the circuit doesn't leak any population into |01>/|10>,
    which would indicate a wiring error (e.g. CNOT direction reversed)."""
    a, b, _ = load_ground_state()
    theta = compute_theta(a, b)
    U = state_prep_unitary(theta)
    produced = U @ np.array([1, 0, 0, 0], dtype=complex)
    assert abs(produced[1]) < 1e-9
    assert abs(produced[2]) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
