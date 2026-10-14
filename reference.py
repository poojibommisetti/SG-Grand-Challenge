"""
reference.py
Member 2 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Classical exact-diagonalization reference (E0) for the 2-qubit reduced
H2/STO-3G Hamiltonian:

    H = h1*Z1 + h2*Z2 + h3*Y1Y2 + h4*Z1Z2 + h5*I

Input:
    hamiltonian.json, written by hamiltonian.py (this run's own PySCF +
    OpenFermion derivation). If that file isn't present yet, falls back
    to the published coefficients from arXiv:2306.16608, Eq. (1), so this
    script is runnable standalone during development -- but for the real
    submission, hamiltonian.json (your own computed numbers) MUST be used.

Output:
    results/e0.json -- ground state energy E0 and the full spectrum,
    which is the ground truth E1 (ideal QPE), E2 (noisy), E3
    (QEC-protected) are all measured against.
"""

import json
import os
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Published fallback [VERIFIED-WEB, arXiv:2306.16608, Eq. (1) & Sec. IV]
FALLBACK_COEFFS = {
    "Z0": -0.3980,
    "Z1": -0.3980,
    "Y0 Y1": -0.1809,
    "Z0 Z1": 0.0112,
    "I": -0.3322,
}
FALLBACK_SOURCE = (
    "FALLBACK: published coefficients from arXiv:2306.16608 Eq.(1). "
    "Replace with your own hamiltonian.json before final submission."
)


def load_coefficients(path="hamiltonian.json"):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["pauli_coefficients_hartree"], f"own derivation ({path})"
    return FALLBACK_COEFFS, FALLBACK_SOURCE


def pauli_term_matrix(label: str) -> np.ndarray:
    """Build the 4x4 matrix for a term like 'Z0', 'Y0 Y1', 'Z0 Z1', or 'I'."""
    if label == "I":
        return np.kron(I2, I2)
    ops = {0: I2, 1: I2}
    for factor in label.split():
        pauli, qubit = factor[0], int(factor[1])
        ops[qubit] = {"X": X, "Y": Y, "Z": Z, "I": I2}[pauli]
    # qubit 0 is the "outer" tensor factor, qubit 1 the "inner" -- must match
    # the same convention hamiltonian.py's qubit indexing uses.
    return np.kron(ops[0], ops[1])


def build_hamiltonian_matrix(coeffs: dict) -> np.ndarray:
    H = np.zeros((4, 4), dtype=complex)
    for label, val in coeffs.items():
        H += val * pauli_term_matrix(label)
    return H


def main():
    coeffs, source = load_coefficients()
    print(f"Coefficient source: {source}")
    print("Coefficients (hartree):")
    for k, v in coeffs.items():
        print(f"  {v:+.4f}  {k}")

    H = build_hamiltonian_matrix(coeffs)
    assert np.allclose(H, H.conj().T), "Hamiltonian is not Hermitian -- check coefficients/signs"

    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = float(eigvals[0].real)
    ground_state = eigvecs[:, 0]

    print(f"\nFull spectrum (hartree): {[round(e.real, 6) for e in eigvals]}")
    print(f"E0 (ground state energy): {E0:.6f} hartree")

    os.makedirs("results", exist_ok=True)
    out = {
        "coefficient_source": source,
        "coefficients_hartree": coeffs,
        "full_spectrum_hartree": [float(e.real) for e in eigvals],
        "E0_hartree": E0,
        "ground_state_vector_real": [float(np.real(c)) for c in ground_state],
        "ground_state_vector_imag": [float(np.imag(c)) for c in ground_state],
        "method": "numpy.linalg.eigh exact diagonalization of 4x4 qubit Hamiltonian",
    }
    with open("results/e0.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/e0.json")


if __name__ == "__main__":
    main()