"""
test_time_evolution.py
Member 1 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Unit tests for time_evolution.py. These must pass before U_power(...) is
trusted enough to wire into ipea.py's controlled-U step.

Run:
    pytest test_time_evolution.py -v
"""

import math

import numpy as np
import pytest

from time_evolution import (
    load_hamiltonian,
    diagonalize,
    U_of_t,
    U_power,
    controlled_U_matrix,
    choose_evolution_time,
    verify_unitary,
    term_matrix,
)


def test_hamiltonian_is_hermitian():
    H, _ = load_hamiltonian()
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_eigenvalues_match_known_e0():
    """Cross-check against Member 2's own e0.json ground-state energy --
    the lowest eigenvalue here must match E0 to high precision."""
    H, _ = load_hamiltonian()
    eigvals, _ = diagonalize(H)
    known_E0 = -1.1373060511988444  # from results/e0.json
    assert math.isclose(min(eigvals), known_E0, abs_tol=1e-8)


def test_U_of_t_is_unitary_for_various_t():
    H, _ = load_hamiltonian()
    eigvals, V = diagonalize(H)
    for t in [0.01, 0.5, 1.0, 3.7, -2.0]:
        U = U_of_t(eigvals, V, t)
        assert verify_unitary(U), f"U(t) not unitary at t={t}"


def test_U_of_zero_is_identity():
    """t=0 means no time evolution at all -- must be exactly identity.
    Catches a sign/formula bug immediately."""
    H, _ = load_hamiltonian()
    eigvals, V = diagonalize(H)
    U = U_of_t(eigvals, V, 0.0)
    assert np.allclose(U, np.eye(4), atol=1e-10)


def test_U_power_matches_repeated_multiplication():
    """The exact shortcut (t -> t*2^k) must agree with brute-force
    matrix exponentiation via repeated multiplication, for small k
    where repeated multiplication is still numerically trustworthy."""
    H, _ = load_hamiltonian()
    eigvals, V = diagonalize(H)
    t = 0.3
    for k in range(4):
        fast = U_power(eigvals, V, t, k)
        U1 = U_of_t(eigvals, V, t)
        brute = np.linalg.matrix_power(U1, 2 ** k)
        assert np.allclose(fast, brute, atol=1e-8), f"Mismatch at k={k}"


def test_controlled_U_is_unitary_and_block_structured():
    H, _ = load_hamiltonian()
    eigvals, V = diagonalize(H)
    U = U_of_t(eigvals, V, 0.7)
    CU = controlled_U_matrix(U)
    assert CU.shape == (8, 8)
    assert verify_unitary(CU)
    # ancilla=0 block must be identity (nothing happens if control is |0>)
    assert np.allclose(CU[:4, :4], np.eye(4), atol=1e-12)
    # off-diagonal blocks must be exactly zero (no leakage between
    # ancilla=0 and ancilla=1 subspaces)
    assert np.allclose(CU[:4, 4:], 0, atol=1e-12)
    assert np.allclose(CU[4:, :4], 0, atol=1e-12)
    # ancilla=1 block must be exactly U itself
    assert np.allclose(CU[4:, 4:], U, atol=1e-12)


def test_choose_evolution_time_avoids_aliasing():
    """The chosen t must map the full eigenvalue spread to strictly less
    than 2*pi of phase -- otherwise two different eigenvalues could
    alias onto the same measured phase."""
    H, _ = load_hamiltonian()
    eigvals, _ = diagonalize(H)
    t = choose_evolution_time(eigvals)
    spread = max(eigvals) - min(eigvals)
    assert spread * t < 2 * math.pi


def test_choose_evolution_time_rejects_degenerate_spectrum():
    """If all eigenvalues were identical (spread=0), there is no
    meaningful way to choose t from spread alone -- must raise, not
    silently return something nonsensical like infinity or zero."""
    degenerate = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        choose_evolution_time(degenerate)


def test_term_matrix_known_cases():
    """Sanity-check the Pauli term builder directly against hand-known
    matrices, independent of anything from hamiltonian.json."""
    assert np.allclose(term_matrix("I"), np.eye(4))
    Z0 = term_matrix("Z0")
    expected_Z0 = np.diag([1, 1, -1, -1]).astype(complex)
    assert np.allclose(Z0, expected_Z0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
