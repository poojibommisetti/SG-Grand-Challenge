"""
state_prep.py
Member 1 -- Quantinuum SG Grand Challenge, Error-Corrected Quantum Chemistry

Prepares the H2 ground state |psi> = a|00> + b|11> on the two system
qubits (q0, q1), where (a, b) come from Member 2's exact-diagonalization
reference (results/e0.json -> ground_state_vector_real/_imag).

Because ground_state_vector_imag is all zeros for this Hamiltonian, the
state is a real-amplitude superposition of only |00> and |11> -- exactly
the shape a single Ry rotation + one CNOT produces. See decision_log.md
for the reasoning (Ry chosen over Rx/Rz because the target amplitudes
are purely real).

Circuit:
    q0: --[Ry(theta)]--o--
                        |
    q1: ----------------X--

theta is derived from (a, b) via theta = 2*atan2(b, a), which correctly
handles the sign of b (atan2, not arccos, so the sign of the second
amplitude is not lost).

This module provides:
  1. compute_theta(a, b)          -- the angle, from first principles
  2. state_prep_unitary(theta)    -- the 4x4 unitary of the full circuit,
                                      built directly from Ry (x) I and CNOT,
                                      used ONLY to unit-test correctness
                                      (see test_state_prep.py). This is not
                                      how the circuit runs on hardware/Selene;
                                      it exists purely to check the gate
                                      angle math independently, the same way
                                      reference.py checks the Hamiltonian by
                                      diagonalizing it directly in numpy.
  3. prepare_ground_state(...)    -- the actual Guppy circuit definition
                                      that runs on Selene. Not exercised by
                                      the unit test below (guppylang is not
                                      available in this environment) -- must
                                      be smoke-tested directly on Selene
                                      before it's trusted.

Run:
    python state_prep.py            # prints theta and the resulting state
    pytest test_state_prep.py -v    # verifies the numpy circuit reproduces
                                     # e0.json's ground state to high precision
"""

import json
import math
import os

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def load_ground_state(path="results/e0.json"):
    """Read Member 2's ground-state vector (a, b) for a|00> + b|11>."""
    with open(path) as f:
        data = json.load(f)

    real = data["ground_state_vector_real"]
    imag = data["ground_state_vector_imag"]

    if any(abs(v) > 1e-12 for v in imag):
        raise ValueError(
            "ground_state_vector_imag is non-zero -- the Ry-only state-prep "
            "in this module assumes a purely real ground state and is NOT "
            "valid here. An Rx gate or an added Rz phase correction is "
            "needed instead; see decision_log.md before using this module."
        )

    if abs(real[1]) > 1e-9 or abs(real[2]) > 1e-9:
        raise ValueError(
            f"Expected only |00> and |11> populated (a|00> + b|11>), but "
            f"|01>/|10> components are ({real[1]}, {real[2]}), not ~0. "
            f"This circuit shape (Ry + single CNOT) does not apply -- a "
            f"more general 2-qubit state-prep circuit would be needed."
        )

    a, b = real[0], real[3]
    return a, b, data["E0_hartree"]


def compute_theta(a: float, b: float) -> float:
    """theta such that Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    matches a|0> + b|1> exactly, including sign. atan2 (not arccos)
    is what correctly preserves the sign of b.
    """
    return 2.0 * math.atan2(b, a)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def cnot_matrix() -> np.ndarray:
    """CNOT with q0 as control, q1 as target, basis order |q0 q1>:
    |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def state_prep_unitary(theta: float) -> np.ndarray:
    """Full 4x4 unitary: (Ry(theta) tensor I) followed by CNOT(q0->q1).
    Only used to independently verify the gate angle math (see module
    docstring) -- not the execution path used on Selene/hardware.
    """
    ry_on_q0 = np.kron(ry_matrix(theta), I2)
    cnot = cnot_matrix()
    return cnot @ ry_on_q0


def verify_against_e0(path="results/e0.json", tol=1e-6):
    """Rebuild the circuit from theta and confirm it reproduces e0.json's
    ground state to within `tol`. Returns (ok, theta, produced_state).
    """
    a, b, _ = load_ground_state(path)
    theta = compute_theta(a, b)
    U = state_prep_unitary(theta)
    zero_state = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    produced = U @ zero_state
    expected = np.array([a, 0, 0, b], dtype=complex)
    ok = bool(np.allclose(produced, expected, atol=tol))
    return ok, theta, produced


# ---------------------------------------------------------------------------
# Guppy circuit (Selene execution path) -- NOT exercised by the numpy check
# above. guppylang is not available in this development environment, and I
# (Claude) could NOT independently verify the exact `angle` construction
# syntax below against live Guppy documentation -- my search for it returned
# irrelevant results. Treat the `angle(...)` call specifically as UNVERIFIED
# pseudocode, not confirmed working code. Before trusting this:
#   1. Check docs.quantinuum.com/guppy for the current guppylang.std.angles
#      API (how to build an `angle` value from a Python float in radians).
#   2. Confirm `ry` takes an `angle`, not a raw float (confirmed true as of
#      this writing -- only the *construction* of the angle is unverified).
#   3. Smoke-test this exact function against Selene before wiring it into
#      ipea.py.
# ---------------------------------------------------------------------------
import math
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, ry, discard
from guppylang.std.lang import py
from guppylang.std.angles import angle  # exact constructor: VERIFY in docs
from guppylang.std.debug import state_result
from guppylang.std.quantum import qubit

@guppy
def prepare_ground_state(q0: qubit, q1: qubit, theta_rad: float) -> None:
    """Prepares a|00> + b|11> on (q0, q1), theta = 2*atan2(b, a).
    theta is computed classically ahead of time from results/e0.json
    (see compute_theta above), converted to Guppy's `angle` type, and
    passed in as a fixed parameter -- it is NOT re-derived on-circuit.
    """
    ry(q0, angle(theta_rad / py(math.pi)))
    cx(q0, q1)

# Example of building the angle value from compute_theta()'s output --
# UNVERIFIED syntax, check the real API before using:
#   from guppylang.std.angles import angle
#   theta_rad = compute_theta(a, b)          # plain Python float, radians
#   theta_guppy = angle(theta_rad)           # <-- confirm this constructor

@guppy
def emulator_demo() -> None:
    q0 = qubit()
    q1 = qubit()
    prepare_ground_state(q0, q1, -2.9181122519)
    state_result("prepared", q0, q1)
    discard(q0)
    discard(q1)

def main():
    a, b, E0 = load_ground_state()
    theta = compute_theta(a, b)
    ok, theta_check, produced = verify_against_e0()

    print(f"Ground state amplitudes: a={a:.10f}, b={b:.10f}")
    print(f"E0 (hartree): {E0:.10f}")
    print(f"theta = 2*atan2(b, a) = {theta:.10f} rad ({math.degrees(theta):.4f} deg)")
    print(f"Numpy circuit reproduces e0.json ground state: {ok}")
    print(f"Produced state: {np.round(produced.real, 8)} (imag ~0: "
          f"{np.allclose(produced.imag, 0, atol=1e-9)})")


if __name__ == "__main__":
    main()
