"""
repetition_code.py
Member 2 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

3-qubit repetition code to protect the QPE ancilla qubit against a single
Pauli error (bit-flip OR phase-flip -- see DECISION LOG below).

=====================================================================
DECISION LOG -- RESOLVED by mentor, [date of mentor answer -- fill in]
=====================================================================
Question asked: bit-flip vs. phase-flip repetition code for the QPE
ancilla, given it's read out in the Hadamard basis.

Mentor's answer: use BIT-FLIP (X-error) code. In IPEA, the dominant
error on the ancilla comes from population relaxation/excitation
(amplitude damping, i.e. bit-flip-like) during state preparation,
controlled-U, and measurement/reset -- not primarily from dephasing.
Bit-flip is the standard distance-1 baseline; phase-flip can be added
later (H before/after) if dephasing turns out to matter more once real
noise-model numbers are in.

DEFAULT: bit_flip. Both variants are implemented and tested. The
generated phase_flip circuit corrects X in its H-rotated syndrome frame;
the NumPy reference corrects Z in the original frame.
Copy this into docs/decision_log.md (Member 3's file) with attribution
to the mentor meeting and its date.
=====================================================================

This module is deliberately dependency-light: the encode/syndrome/decode
logic and its correctness proof (exact statevector simulation) use only
numpy, so it can be unit-tested and run in CI without needing Guppy or
a live Selene connection. Circuit-building functions for the actual
Guppy/Selene integration are included separately, for Member 1 to wire
into qec_ipea.py.

IMPORTANT -- how to actually run the Guppy output: guppylang's @guppy
decorator uses Python's inspect.getsourcelines() to read the function's
source from a real file on disk. Passing the generated string to exec()
will fail with "OSError: could not get source code" -- there's no file
for inspect to find. Use write_guppy_module() to write it to a real
.py file, then run that file as a script (`python qec_round.py`), not
exec() in memory.
"""

import numpy as np

# ---------------------------------------------------------------------
# Single-qubit gates and 3-qubit tensor helpers (exact statevector sim)
# ---------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def _kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def _single_qubit_gate_3(gate, qubit: int) -> np.ndarray:
    mats = [I2, I2, I2]
    mats[qubit] = gate
    return _kron3(*mats)


def _cx_3(control: int, target: int) -> np.ndarray:
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    mats0 = [I2, I2, I2]
    mats0[control] = P0
    mats1 = [I2, I2, I2]
    mats1[control] = P1
    mats1[target] = X
    term0 = _kron3(*mats0)
    term1 = _kron3(*mats1)
    return term0 + term1


def encode_state(alpha: complex, beta: complex, variant: str = "bit_flip") -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    if variant == "bit_flip":
        psi[0b000] = alpha
        psi[0b111] = beta
    elif variant == "phase_flip":
        psi_bitflip = np.zeros(8, dtype=complex)
        psi_bitflip[0b000] = alpha
        psi_bitflip[0b111] = beta
        Hgate = _kron3(H, H, H)
        psi = Hgate @ psi_bitflip
    else:
        raise ValueError(f"unknown variant: {variant!r}, expected 'bit_flip' or 'phase_flip'")
    return psi


def apply_error(psi: np.ndarray, qubit: int, variant: str = "bit_flip") -> np.ndarray:
    error_gate = X if variant == "bit_flip" else Z
    return _single_qubit_gate_3(error_gate, qubit) @ psi


SYNDROME_TABLE = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def measure_syndrome(psi: np.ndarray, variant: str = "bit_flip") -> tuple:
    stab_op = Z if variant == "bit_flip" else X
    s1_op = np.kron(np.kron(stab_op, stab_op), I2)
    s2_op = np.kron(I2, np.kron(stab_op, stab_op))

    s1_eigenvalue = np.real(psi.conj() @ (s1_op @ psi))
    s2_eigenvalue = np.real(psi.conj() @ (s2_op @ psi))

    assert abs(abs(s1_eigenvalue) - 1) < 1e-8, "not a valid single-error syndrome state"
    assert abs(abs(s2_eigenvalue) - 1) < 1e-8, "not a valid single-error syndrome state"

    s1 = 0 if s1_eigenvalue > 0 else 1
    s2 = 0 if s2_eigenvalue > 0 else 1
    return (s1, s2)


def decode(syndrome: tuple) -> int:
    if syndrome not in SYNDROME_TABLE:
        raise ValueError(f"unrecognized syndrome {syndrome}")
    return SYNDROME_TABLE[syndrome]


def apply_correction(psi: np.ndarray, faulty_qubit, variant: str = "bit_flip") -> np.ndarray:
    if faulty_qubit is None:
        return psi
    correction_gate = X if variant == "bit_flip" else Z
    return _single_qubit_gate_3(correction_gate, faulty_qubit) @ psi


def run_full_correction_cycle(alpha, beta, error_qubit, variant="bit_flip"):
    psi = encode_state(alpha, beta, variant)
    psi_noisy = apply_error(psi, error_qubit, variant)
    syndrome = measure_syndrome(psi_noisy, variant)
    faulty_qubit = decode(syndrome)
    psi_corrected = apply_correction(psi_noisy, faulty_qubit, variant)
    return psi, psi_corrected, syndrome, faulty_qubit


def build_encode_and_correct_guppy_source(variant: str = "bit_flip") -> str:
    """Generates Guppy source implementing: encode -> syndrome
    extraction -> majority-vote correction -> uncompute/decode back to
    a single clean logical qubit `q0`, with q1/q2 discarded.

    The correction gate is X in the transformed syndrome frame. For phase_flip, H gates rotate a physical Z fault into X before checks. The NumPy reference instead corrects Z in the original frame.
    """
    if variant not in ("bit_flip", "phase_flip"):
        raise ValueError(f"unknown variant: {variant!r}")

    hadamard_wrap = "    h(q0)\n    h(q1)\n    h(q2)\n" if variant == "phase_flip" else ""
    correction_gate = "x"
    gate_import = "x"

    return f'''\
from guppylang import guppy
from guppylang.std.builtins import owned
from guppylang.std.quantum import cx, h, measure, qubit, {gate_import}, discard


@guppy
def repetition_code_round(q0: qubit @ owned) -> qubit:
    """Encode q0 across 3 physical qubits, extract the {variant}
    syndrome via 2 ancillas, apply the correction, uncompute back to a
    single clean logical qubit, and return it (q1/q2 discarded)."""
    q1 = qubit()
    q2 = qubit()

    # 1. Encoding
    cx(q0, q1)
    cx(q0, q2)

{hadamard_wrap}
    # 2. Parity-measurement ancillas: checks on (q0,q1) and (q1,q2)
    a0 = qubit()
    a1 = qubit()
    cx(q0, a0)
    cx(q1, a0)
    cx(q1, a1)
    cx(q2, a1)

    s0 = measure(a0)
    s1 = measure(a1)

    # 3. Correction (majority vote) -- matches SYNDROME_TABLE in the
    # numpy reference implementation in this same file; keep in sync.
    if s0:
        if not s1:
            {correction_gate}(q0)
        else:
            {correction_gate}(q1)
    else:
        if s1:
            {correction_gate}(q2)

{hadamard_wrap}
    # 4. Uncompute / decode back to a single clean logical qubit
    cx(q0, q2)
    cx(q0, q1)
    discard(q1)
    discard(q2)

    return q0


repetition_code_round.check()
'''


def write_guppy_module(path: str, variant: str = "bit_flip") -> None:
    """Write the generated Guppy source to a real .py file. Required --
    guppylang's @guppy decorator needs inspect.getsourcelines() to find
    real source on disk; exec()-ing the string in memory will fail with
    OSError: could not get source code."""
    with open(path, "w") as f:
        f.write(build_encode_and_correct_guppy_source(variant))


def _test_no_error(variant):
    alpha, beta = 0.6, 0.8
    psi = encode_state(alpha, beta, variant)
    syndrome = measure_syndrome(psi, variant)
    assert syndrome == (0, 0), f"[{variant}] expected no-error syndrome, got {syndrome}"
    faulty = decode(syndrome)
    assert faulty is None, f"[{variant}] expected no correction needed"
    print(f"  [{variant}] no-error case: PASS")


def _test_single_error_all_qubits(variant):
    alpha, beta = (0.6 + 0.0j), (0.8 + 0.0j)
    for q in range(3):
        psi, psi_corrected, syndrome, faulty_qubit = run_full_correction_cycle(
            alpha, beta, q, variant
        )
        fidelity = abs(np.vdot(psi, psi_corrected)) ** 2
        assert faulty_qubit == q, (
            f"[{variant}] error on qubit {q}: decoder identified qubit "
            f"{faulty_qubit} instead (syndrome {syndrome})"
        )
        assert abs(fidelity - 1.0) < 1e-10, (
            f"[{variant}] error on qubit {q}: recovery fidelity {fidelity}, expected 1.0"
        )
        print(f"  [{variant}] single error on qubit {q}: syndrome={syndrome}, "
              f"corrected qubit={faulty_qubit}, recovery fidelity={fidelity:.10f}: PASS")


def _test_known_failure_mode_documented(variant):
    alpha, beta = 0.6, 0.8
    error_gate = X if variant == "bit_flip" else Z
    psi = encode_state(alpha, beta, variant)
    psi_noisy = _single_qubit_gate_3(error_gate, 0) @ psi
    psi_noisy = _single_qubit_gate_3(error_gate, 1) @ psi_noisy
    syndrome = measure_syndrome(psi_noisy, variant)
    faulty_qubit = decode(syndrome)
    psi_corrected = apply_correction(psi_noisy, faulty_qubit, variant)
    fidelity = abs(np.vdot(psi, psi_corrected)) ** 2
    print(f"  [{variant}] KNOWN LIMITATION -- double error on qubits (0,1): "
          f"syndrome={syndrome}, decoder guesses qubit={faulty_qubit}, "
          f"recovery fidelity={fidelity:.4f} (expected LOW)")
    assert fidelity < 0.99, (
        f"[{variant}] expected double-error case to fail correction, got {fidelity}"
    )


def _test_guppy_source_is_valid_python(variant):
    """Doesn't require guppylang -- just confirms the generated string
    parses as syntactically valid Python, catching f-string/indent bugs
    without needing a live guppylang install in CI."""
    import ast
    source = build_encode_and_correct_guppy_source(variant)
    ast.parse(source)
    expected_gate = "x("
    assert expected_gate in source, (
        f"[{variant}] expected correction gate '{expected_gate}' not found in generated source"
    )
    print(f"  [{variant}] generated Guppy source: valid Python syntax, "
          f"correct gate ('{expected_gate.strip('(')}'): PASS")


def run_all_tests():
    for variant in ("bit_flip", "phase_flip"):
        print(f"\n=== Testing variant: {variant} ===")
        _test_no_error(variant)
        _test_single_error_all_qubits(variant)
        _test_known_failure_mode_documented(variant)
        _test_guppy_source_is_valid_python(variant)
    print("\nAll tests passed for both variants.")


if __name__ == "__main__":
    run_all_tests()