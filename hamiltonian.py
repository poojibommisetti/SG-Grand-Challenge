

"""
hamiltonian.py
Member 2 — Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Builds the 2-qubit reduced H2/STO-3G Hamiltonian:

    H = h1*Z1 + h2*Z2 + h3*Y1Y2 + h4*Z1Z2 + h5*I   (hartree)

Pipeline (all open-source, no licensed InQuanto dependency):
  1. PySCF: RHF/STO-3G for H2 -> one- and two-electron MO integrals
  2. OpenFermion: build fermionic Hamiltonian, Jordan-Wigner transform,
     then reduce 4 qubits -> 2 qubits via symmetry_conserving_bravyi_kitaev
     (this performs the parity-mapping + Z2-tapering step in one call,
     using particle-number and spin conservation).

Reference geometry / cross-check target:
  Yamamoto, Duffield, Kikuchi, Munoz Ramo,
  "Demonstrating Bayesian Quantum Phase Estimation with Quantum Error
  Detection", arXiv:2306.16608 (2023), Eq. (1) and Sec. IV:
      R_HH = 0.73486 Angstrom
      (h1, h2, h3, h4, h5) =
          (-0.3980, -0.3980, -0.1809, 0.0112, -0.3322)  hartree

Run:
    pip install pyscf openfermion openfermionpyscf --break-system-packages
    python hamiltonian.py
"""

import json
import numpy as np

from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    symmetry_conserving_bravyi_kitaev,
)
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator

# ---- literature cross-check target [VERIFIED-WEB, arXiv:2306.16608] ----
R_HH_ANGSTROM = 0.73486
PUBLISHED_COEFFS = {
    "Z0": -0.3980,
    "Z1": -0.3980,
    "Y0 Y1": -0.1809,
    "Z0 Z1": 0.0112,
    "I": -0.3322,
}
MATCH_TOL_HARTREE = 1e-3  # loose; published values are rounded to 4 dp


def build_h2_molecule(bond_length_angstrom: float) -> MolecularData:
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length_angstrom))]
    molecule = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,   # singlet
        charge=0,
        description=f"H2_{bond_length_angstrom}A",
    )
    # run_scf performs RHF; run_fci gives the exact-diagonalization
    # reference energy your reference.py should also reproduce.
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def build_two_qubit_hamiltonian(molecule: MolecularData) -> QubitOperator:
    fermion_h = get_fermion_operator(molecule.get_molecular_hamiltonian())
    # active_orbitals=None, active_fermions=None -> use all orbitals/electrons
    # (2 spatial orbitals, 4 spin-orbitals, 2 electrons for H2/STO-3G).
    # This performs Bravyi-Kitaev + Z2 tapering on particle-number and
    # spin-parity symmetries, giving the 2-qubit reduced Hamiltonian.
    qubit_h = symmetry_conserving_bravyi_kitaev(
        fermion_h,
        active_orbitals=2 * molecule.n_orbitals,
        active_fermions=molecule.n_electrons,
    )
    return qubit_h


def extract_pauli_coeffs(qubit_h: QubitOperator) -> dict:
    coeffs = {}
    for term, coeff in qubit_h.terms.items():
        if term == ():
            label = "I"
        else:
            label = " ".join(f"{op}{qubit}" for qubit, op in term)
        coeffs[label] = float(np.real(coeff))
    return coeffs


_I2 = np.eye(2, dtype=complex)
_PAULI = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _term_matrix(label: str) -> np.ndarray:
    """4x4 matrix for a term like 'Z0', 'Y0 Y1', 'Z0 Z1', or 'I'."""
    if label == "I":
        return np.kron(_I2, _I2)
    ops = {0: _I2, 1: _I2}
    for factor in label.split():
        pauli, qubit = factor[0], int(factor[1])
        ops[qubit] = _PAULI[pauli]
    return np.kron(ops[0], ops[1])


def hamiltonian_matrix(coeffs: dict) -> np.ndarray:
    H = np.zeros((4, 4), dtype=complex)
    for label, val in coeffs.items():
        H += val * _term_matrix(label)
    return H


def cross_check(computed: dict, published: dict, tol: float) -> bool:
    """Compare Hamiltonians by physical content (eigenvalue spectrum),
    not by which Pauli letter each library happened to assign to each
    qubit. Different fermion-to-qubit mapping implementations can pick
    different, equally valid Z2-tapered bases (e.g. X0 X1 vs Y0 Y1, with
    an accompanying sign flip on the single-qubit Z terms) and still
    describe the exact same physics. A naive term-by-term label diff
    would flag that as broken when it isn't -- the spectrum is the only
    thing that's actually physically meaningful to compare.
    """
    print("\nTerm-level comparison (informational only -- labels can\n"
          "legitimately differ by Z2-tapering/basis convention):")
    for label, pub_val in published.items():
        comp_val = computed.get(label)
        if comp_val is None:
            print(f"  {label:8s}  computed=  n/a    published={pub_val:+.4f}")
        else:
            print(f"  {label:8s}  computed={comp_val:+.4f}  published={pub_val:+.4f}")

    computed_spectrum = np.sort(np.linalg.eigvalsh(hamiltonian_matrix(computed)).real)
    published_spectrum = np.sort(np.linalg.eigvalsh(hamiltonian_matrix(published)).real)
    diff = np.max(np.abs(computed_spectrum - published_spectrum))

    print("\nSpectrum-level cross-check vs arXiv:2306.16608 (the physically"
          " meaningful check):")
    print(f"  computed  spectrum : {[f'{e:+.6f}' for e in computed_spectrum]}")
    print(f"  published spectrum : {[f'{e:+.6f}' for e in published_spectrum]}")
    print(f"  max eigenvalue diff: {diff:.2e}  (tol={tol:.0e})")

    return bool(diff < tol)


def main():
    molecule = build_h2_molecule(R_HH_ANGSTROM)
    qubit_h = build_two_qubit_hamiltonian(molecule)
    coeffs = extract_pauli_coeffs(qubit_h)

    print(f"H2/STO-3G at R_HH = {R_HH_ANGSTROM} Angstrom")
    print(f"HF energy  : {molecule.hf_energy:.6f} hartree")
    print(f"FCI energy : {molecule.fci_energy:.6f} hartree  (this is your E0 reference)")
    print("\n2-qubit Hamiltonian terms:")
    for label, val in coeffs.items():
        print(f"  {val:+.6f}  {label}")

    all_match = cross_check(coeffs, PUBLISHED_COEFFS, MATCH_TOL_HARTREE)
    if all_match:
        print("\nOverall: MATCHES LITERATURE (verified via eigenvalue spectrum;"
              " term labels may differ by Z2-tapering/Pauli-frame convention"
              " -- see decision_log.md)")
    else:
        print("\nOverall: DOES NOT MATCH — spectrum itself differs, this is a"
              " real discrepancy (check mapping/geometry/basis), not just a"
              " labeling convention")

    out = {
        "bond_length_angstrom": R_HH_ANGSTROM,
        "basis": "sto-3g",
        "hf_energy_hartree": molecule.hf_energy,
        "fci_energy_hartree": molecule.fci_energy,
        "pauli_coefficients_hartree": coeffs,
        "matches_arxiv_2306_16608": all_match,
    }
    with open("hamiltonian.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote hamiltonian.json")


if __name__ == "__main__":
    main()