"""
time_evolution.py
Member 1 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Builds the EXACT time-evolution operator U(t) = exp(-i * H * t) for the
2-qubit reduced H2 Hamiltonian, and the controlled powers U(t)^(2^k) that
IPEA's iteration loop needs -- with NO Trotter approximation, per the
team's locked scope decision (see decision_log.md).

--------------------------------------------------------------------------
WHY THIS AVOIDS TROTTER, AND WHAT "EXACT" MEANS HERE
--------------------------------------------------------------------------
The Hamiltonian is a sum of Pauli terms:

    H = h1*I + h2*Z0 + h3*Z1 + h4*Z0Z1 + h5*X0X1

Z0, Z1, and Z0Z1 all commute with each other (any two diagonal-in-the-
computational-basis operators commute), but X0X1 does NOT commute with
the Z terms. Because of that, exp(-i*H*t) is NOT simply the product of
exp(-i*h_j*P_j*t) for each term separately -- computing it that way would
be a Trotter APPROXIMATION, and it gets worse (more "Trotter error") the
deeper you iterate, which is exactly the extra error source the team
explicitly chose to avoid for Submission 1.

Instead, this module uses the EXACT approach: diagonalize the full 4x4
matrix H once (H = V @ diag(eigenvalues) @ V^dagger, via the spectral
theorem -- valid for any Hermitian matrix, which reference.py already
confirmed H is), then

    exp(-i*H*t) = V @ diag(exp(-i*eigenvalues*t)) @ V^dagger

This is mathematically exact for ANY t, with zero approximation error --
the only numerical error is ordinary floating-point rounding. It is also
DELIBERATELY built from the same diagonalization as Member 2's E0/reference
result, so that E0 (used as the answer key) and U(t) (used to generate the
QPE circuit) come from one shared, self-consistent calculation rather than
two independently-implemented ones that might quietly disagree.

--------------------------------------------------------------------------
POWERS OF U, WITHOUT MATRIX-MULTIPLYING U BY ITSELF REPEATEDLY
--------------------------------------------------------------------------
IPEA's iteration k needs U(t)^(2^k), not just U(t). Because
    [exp(-i*H*t)]^n = exp(-i*H*(n*t))
exactly (for any integer n -- this is a basic property of the matrix
exponential of a fixed Hermitian operator), U(t)^(2^k) is computed by
calling the SAME exact formula above with t replaced by t * 2^k. This is
both simpler and more numerically stable than repeated matrix
multiplication (which would accumulate rounding error with each
multiplication).

--------------------------------------------------------------------------
CHOOSING THE EVOLUTION TIME t (SCALE FACTOR)
--------------------------------------------------------------------------
QPE reads out a PHASE in [0, 2*pi), not an energy directly. The
eigenvalues of H must be mapped onto that phase circle via
    phase = -(E * t) mod 2*pi
If t is too large, different eigenvalues can map to phases that "wrap
around" the circle and become indistinguishable from each other -- this
is a real failure mode (aliasing), not a numerical nicety. `choose_evolution_time`
below picks t so the full spread of eigenvalues fits inside less than one
full turn, with a safety margin, which is the standard way to avoid this.

--------------------------------------------------------------------------
GUPPY/SELENE EXECUTION PATH
--------------------------------------------------------------------------
The exact 8x8 controlled matrix is represented as a pytket Unitary3qBox,
decomposed, and rebased to CX and Rx/Ry/Rz gates before Guppy imports it.
The ancilla is qubit 0 (the outer tensor factor) and controls the two
system qubits. This is exact gate synthesis, not Trotterization.

"""

import json
import math
import os

import numpy as np

from guppylang import guppy
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard, h, qubit
from pytket.circuit import Circuit, OpType, Unitary3qBox
from pytket.passes import AutoRebase, DecomposeBoxes

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def term_matrix(label: str) -> np.ndarray:
    """4x4 matrix for a term like 'Z0', 'X0 X1', 'Z0 Z1', or 'I'.
    Same convention as hamiltonian.py/reference.py: qubit 0 is the outer
    tensor factor, qubit 1 the inner one.
    """
    if label == "I":
        return np.kron(I2, I2)
    ops = {0: I2, 1: I2}
    for factor in label.split():
        pauli, qubit = factor[0], int(factor[1])
        ops[qubit] = PAULI[pauli]
    return np.kron(ops[0], ops[1])


def load_hamiltonian(path="results/hamiltonian.json"):
    """Read Member 2's Pauli coefficients and rebuild the 4x4 matrix H."""
    with open(path) as f:
        data = json.load(f)
    coeffs = data["pauli_coefficients_hartree"]
    H = np.zeros((4, 4), dtype=complex)
    for label, val in coeffs.items():
        H += val * term_matrix(label)
    return H, coeffs


def diagonalize(H: np.ndarray):
    """H = V @ diag(eigvals) @ V^dagger. Requires H Hermitian (checked)."""
    assert np.allclose(H, H.conj().T), (
        "Hamiltonian is not Hermitian -- check coefficients/signs before "
        "trusting anything downstream of this."
    )
    eigvals, V = np.linalg.eigh(H)
    return eigvals.real, V


def U_of_t(eigvals: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
    """Exact exp(-i*H*t) via spectral decomposition. No Trotter."""
    phases = np.exp(-1j * eigvals * t)
    return V @ np.diag(phases) @ V.conj().T


def U_power(eigvals: np.ndarray, V: np.ndarray, t: float, k: int) -> np.ndarray:
    """Exact U(t)^(2^k), computed as U(t * 2^k) -- see module docstring
    for why this is exact and preferred over repeated matrix multiplication.
    """
    return U_of_t(eigvals, V, t * (2 ** k))


def controlled_U_matrix(U: np.ndarray) -> np.ndarray:
    """8x8 matrix for controlled-U with the ancilla as control (ancilla
    is the FIRST/outer tensor factor here, system qubits second) --
    matches the corrected circuit diagram: control dot on the ancilla
    line, U box spanning the system qubits.
    This exact block matrix is also the input to the pytket/Guppy synthesis
    path below, so the ancilla remains the first/outer tensor factor.
    """
    dim = U.shape[0]
    top_left = np.eye(dim, dtype=complex)      # ancilla = |0>: do nothing
    bottom_right = U                            # ancilla = |1>: apply U
    zeros = np.zeros((dim, dim), dtype=complex)
    return np.block([[top_left, zeros], [zeros, bottom_right]])


def choose_evolution_time(eigvals: np.ndarray, margin: float = 0.9) -> float:
    """Pick t so the full eigenvalue spread maps to less than one full
    2*pi turn, leaving `margin` (default 90%) of the circle used -- a
    safety buffer against aliasing/wraparound between eigenvalues.
    """
    spread = float(np.max(eigvals) - np.min(eigvals))
    if spread == 0:
        raise ValueError("Eigenvalues are degenerate -- cannot choose a "
                          "meaningful evolution time from spread alone.")
    return (2 * math.pi * margin) / spread


def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol))


def main():
    H, coeffs = load_hamiltonian()
    eigvals, V = diagonalize(H)

    print("Hamiltonian coefficients (hartree):")
    for label, val in coeffs.items():
        print(f"  {val:+.6f}  {label}")

    print(f"\nEigenvalues (hartree): {[round(e, 6) for e in eigvals]}")

    t = choose_evolution_time(eigvals)
    print(f"\nChosen evolution time t = {t:.6f} (1/hartree)")
    print(f"  eigenvalue spread = {np.max(eigvals) - np.min(eigvals):.6f} hartree")
    print(f"  spread * t = {(np.max(eigvals) - np.min(eigvals)) * t:.6f} "
          f"rad (should be < 2*pi = {2*math.pi:.6f})")

    U0 = U_of_t(eigvals, V, t)
    print(f"\nU(t) is unitary: {verify_unitary(U0)}")

    n_bits_to_check = 5  # matches the recommended starting precision-bit count
    print(f"\nChecking U(t)^(2^k) for k = 0..{n_bits_to_check - 1}:")
    for k in range(n_bits_to_check):
        Uk = U_power(eigvals, V, t, k)
        ok = verify_unitary(Uk)
        CUk = controlled_U_matrix(Uk)
        ok_c = verify_unitary(CUk)
        print(f"  k={k}: U(t)^(2^{k}) unitary={ok}, "
              f"controlled version (8x8) unitary={ok_c}")

    os.makedirs("results", exist_ok=True)
    out = {
        "eigenvalues_hartree": [float(e) for e in eigvals],
        "evolution_time": t,
        "margin_used": 0.9,
        "n_bits_checked": n_bits_to_check,
        "all_powers_unitary": all(
            verify_unitary(U_power(eigvals, V, t, k)) for k in range(n_bits_to_check)
        ),
        "note": "U(t)^(2^k) verified as an exact unitary via spectral "
                "decomposition; Guppy/Selene synthesis is exposed separately "
                "by selene_smoke_test().",
    }
    with open("results/time_evolution_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/time_evolution_check.json")



# ---------------------------------------------------------------------------
# Guppy/Selene exact controlled-U path
# ---------------------------------------------------------------------------

def controlled_u_pytket_circuit(
    eigvals: np.ndarray, V: np.ndarray, t: float, k: int
) -> tuple[Circuit, np.ndarray]:
    """Synthesize exact ancilla-controlled U(t)^(2^k) to CX/Rx/Ry/Rz.

    Qubit 0 is the ancilla control; qubits 1 and 2 are the system register.
    The input unitary is precisely controlled_U_matrix(U_power(...)); no
    product-formula or Trotter approximation is used.
    """
    U = U_power(eigvals, V, t, k)
    controlled = controlled_U_matrix(U)
    circuit = Circuit(3)
    circuit.add_unitary3qbox(Unitary3qBox(controlled), 0, 1, 2)
    DecomposeBoxes().apply(circuit)
    AutoRebase({OpType.CX, OpType.Rx, OpType.Ry, OpType.Rz}).apply(circuit)

    allowed = {OpType.CX, OpType.Rx, OpType.Ry, OpType.Rz}
    unsupported = {command.op.type for command in circuit.get_commands()} - allowed
    if unsupported:
        raise RuntimeError(f"Synthesis left unsupported gates: {unsupported}")
    if not np.allclose(circuit.get_unitary(), controlled, atol=2e-10):
        raise AssertionError("pytket synthesis changed the exact controlled-U matrix")
    return circuit, controlled


# The imported circuit is supplied by selene_smoke_test before this Guppy
# entrypoint is compiled. The control is explicitly the ancilla on qubit 0.
_controlled_u_for_emulator = None


@guppy
def emulator_demo() -> None:
    """Prepare |+> on the ancilla, apply controlled-U to |00>, and record state."""
    ancilla = qubit()
    q0 = qubit()
    q1 = qubit()
    h(ancilla)
    _controlled_u_for_emulator(ancilla, q0, q1)
    state_result("controlled_u_state", ancilla, q0, q1)
    discard(ancilla)
    discard(q0)
    discard(q1)


def selene_smoke_test(k: int = 0, tol: float = 2e-9) -> dict:
    """Compile and run one exact controlled-U power in Selene's emulator.

    The ancilla starts in |+> and the system in |00>, exercising both control
    branches: |0>|00> is unchanged and |1>|00> receives U(t)^(2^k).
    """
    global _controlled_u_for_emulator
    H, _ = load_hamiltonian()
    eigvals, V = diagonalize(H)
    t = choose_evolution_time(eigvals)
    circuit, _ = controlled_u_pytket_circuit(eigvals, V, t, k)
    _controlled_u_for_emulator = guppy.load_pytket(
        f"controlled_u_k{k}", circuit, use_arrays=False
    )

    package = emulator_demo.compile()
    result = emulator_demo.emulator(n_qubits=3).run()
    state_distribution = result.partial_state_dicts()[0][
        "controlled_u_state"
    ].state_distribution()
    if len(state_distribution) != 1:
        raise AssertionError("Expected a pure state from ideal statevector simulation")
    state = state_distribution[0].state
    U = U_power(eigvals, V, t, k)
    expected = np.concatenate(
        (np.array([1, 0, 0, 0], dtype=complex), U[:, 0])
    ) / math.sqrt(2)
    if not np.allclose(state, expected, atol=tol):
        raise AssertionError(
            f"Selene state differs from exact controlled-U result: {state!r}"
        )
    return {
        "k": k,
        "evolution_time": t,
        "compiled_modules": len(package.modules),
        "pytket_gate_count": circuit.n_gates,
        "pytket_gate_types": sorted(
            {str(command.op.type) for command in circuit.get_commands()}
        ),
        "state_probability": state_distribution[0].probability,
        "state": state,
        "expected_state": expected,
        "max_amplitude_error": float(np.max(np.abs(state - expected))),
    }

if __name__ == "__main__":
    main()
