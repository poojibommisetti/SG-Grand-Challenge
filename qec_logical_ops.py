"""Logical controlled-U component using a three-qubit bit-flip code.

This is a QEC component demonstration, not a fault-tolerant or full logical
IPEA implementation. It encodes the IPEA ancilla as |000> / |111>, uses
d0 as the physical control wire for the exact imported controlled-U, and
extracts fresh ZZ syndromes before and after that gate. Only injected
single-X faults at those two boundaries are corrected here; faults during
the imported gate are not injected or claimed to be corrected.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Sequence
import math

import numpy as np

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import result
from guppylang.std.debug import state_result
from guppylang.std.lang import py
from guppylang.std.quantum import cx, discard, h, measure, qubit, rz, x

from state_prep import compute_theta, load_ground_state, prepare_ground_state

_controlled_u_for_logical_probe = None


@guppy
def _logical_controlled_u_body(
    logical_mode: int,
    pre_error: int,
    post_error: int,
    theta_rad: float,
    correction_angle_rad: float,
) -> None:
    """Run one encoded-ancilla controlled-U probe.

    Error locations are 0, 1, or 2 for d0/d1/d2, and 3 for no injected
    error. logical_mode is 0, 1, or 2 for |0_L>, |1_L>, or |+_L>.
    """
    d0 = qubit()
    d1 = qubit()
    d2 = qubit()
    system0 = qubit()
    system1 = qubit()

    # Prepare a logical basis state and encode |0> -> |000>, |1> -> |111>.
    if logical_mode == 1:
        x(d0)
    if logical_mode == 2:
        h(d0)
    cx(d0, d1)
    cx(d0, d2)

    # Use the existing H2 ground-state preparation and exact controlled-U.
    prepare_ground_state(system0, system1, theta_rad)

    # Boundary fault probe before the controlled operation.
    if pre_error == 0:
        x(d0)
    if pre_error == 1:
        x(d1)
    if pre_error == 2:
        x(d2)

    pre_syndrome0 = qubit()
    pre_syndrome1 = qubit()
    cx(d0, pre_syndrome0)
    cx(d1, pre_syndrome0)
    cx(d1, pre_syndrome1)
    cx(d2, pre_syndrome1)
    pre_s0 = measure(pre_syndrome0)
    pre_s1 = measure(pre_syndrome1)
    result("pre_s0", pre_s0)
    result("pre_s1", pre_s1)

    # 10 -> X(d0), 11 -> X(d1), 01 -> X(d2), 00 -> no correction.
    if pre_s0:
        if pre_s1:
            x(d1)
        else:
            x(d0)
    if not pre_s0:
        if pre_s1:
            x(d2)

    # The pytket circuit's first wire is explicitly the ancilla control.
    _controlled_u_for_logical_probe(d0, system0, system1)

    # Boundary fault probe after the controlled operation. These syndromes
    # use new ancillas because measurement destructively consumes each one.
    if post_error == 0:
        x(d0)
    if post_error == 1:
        x(d1)
    if post_error == 2:
        x(d2)

    post_syndrome0 = qubit()
    post_syndrome1 = qubit()
    cx(d0, post_syndrome0)
    cx(d1, post_syndrome0)
    cx(d1, post_syndrome1)
    cx(d2, post_syndrome1)
    post_s0 = measure(post_syndrome0)
    post_s1 = measure(post_syndrome1)
    result("post_s0", post_s0)
    result("post_s1", post_s1)

    if post_s0:
        if post_s1:
            x(d1)
        else:
            x(d0)
    if not post_s0:
        if post_s1:
            x(d2)

    # Existing IPEA alpha is in radians. Guppy angle stores half-turns, so
    # divide by pi inside Guppy. Rz(d0) acts as the logical phase rotation
    # on the |000>, |111> code space because Z_L is represented by Z(d0).
    rz(d0, angle(correction_angle_rad / py(math.pi)))

    # Snapshot the encoded ancilla and H2 register before final readout.
    state_result(
        "logical_controlled_state", d0, d1, d2, system0, system1
    )

    # Decode only after both recovery rounds, then perform IPEA's final H
    # and destructive ancilla measurement. System qubits are never measured.
    cx(d0, d2)
    cx(d0, d1)
    h(d0)
    measured_ancilla = measure(d0)
    result("logical_ancilla", measured_ancilla)
    discard(d1)
    discard(d2)
    discard(system0)
    discard(system1)


_probe_mode = 0
_probe_pre_error = 3
_probe_post_error = 3
_probe_theta_rad = 0.0
_probe_correction_angle_rad = 0.0


@guppy
def logical_controlled_u_emulator_demo() -> None:
    """No-input Guppy entrypoint; host parameters are embedded at compile time."""
    _logical_controlled_u_body(
        py(_probe_mode),
        py(_probe_pre_error),
        py(_probe_post_error),
        py(_probe_theta_rad),
        py(_probe_correction_angle_rad),
    )


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _error_index(name: str, value: int | None) -> int:
    if value is None:
        return 3
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1, 2):
        raise ValueError(f"{name} must be None or a data index 0, 1, or 2")
    return int(value)


def _logical_reference_state(
    mode: int,
    ground_state: np.ndarray,
    controlled_matrix: np.ndarray,
    correction_angle_rad: float,
) -> np.ndarray:
    """Exact five-qubit state in d0,d1,d2,system0,system1 order."""
    logical_ancilla = np.zeros(2, dtype=complex)
    if mode == 0:
        logical_ancilla[0] = 1.0
    elif mode == 1:
        logical_ancilla[1] = 1.0
    else:
        logical_ancilla[:] = 1.0 / math.sqrt(2.0)

    # controlled_matrix is in [control, system0, system1] basis order.
    unencoded = controlled_matrix @ np.kron(logical_ancilla, ground_state)
    expected = np.zeros(32, dtype=complex)
    for logical_bit in (0, 1):
        phase = np.exp(
            (-1j if logical_bit == 0 else 1j) * correction_angle_rad / 2.0
        )
        codeword = 0 if logical_bit == 0 else 0b111
        for system_index in range(4):
            amplitude = unencoded[logical_bit * 4 + system_index]
            expected[(codeword << 2) | system_index] = phase * amplitude
    return expected


def run_logical_controlled_u_probe(
    *,
    mode: str,
    k: int = 0,
    pre_error: int | None = None,
    post_error: int | None = None,
    phase_correction_bits: Sequence[int] | None = None,
    n_bits: int = 5,
    shots: int = 1,
    seed: int = 20261012,
    n_qubits: int = 7,
) -> dict:
    """Compile and execute one encoded controlled-U probe on Guppy/Selene.

    pre_error and post_error inject a deterministic X on one data wire
    immediately before or after the imported controlled-U. Supplying phase
    correction bits applies the existing IPEA convention for this k.
    """
    modes = {"zero": 0, "one": 1, "plus": 2}
    if mode not in modes:
        raise ValueError("mode must be 'zero', 'one', or 'plus'")
    mode_index = modes[mode]
    pre_index = _error_index("pre_error", pre_error)
    post_index = _error_index("post_error", post_error)
    shots = _positive_integer("shots", shots)
    seed = _nonnegative_integer("seed", seed)
    n_qubits = _positive_integer("n_qubits", n_qubits)
    if n_qubits < 7:
        raise ValueError("n_qubits must be at least 7 for the two fresh syndrome pairs")

    correction_angle_rad = 0.0
    if phase_correction_bits is not None:
        from ipea import phase_correction_angle

        correction_angle_rad = phase_correction_angle(
            k, list(phase_correction_bits), n_bits
        )

    a, b, _ = load_ground_state()
    theta_rad = compute_theta(a, b)
    ground_state = np.array([a, 0.0, 0.0, b], dtype=complex)

    from time_evolution import (
        choose_evolution_time,
        controlled_u_pytket_circuit,
        diagonalize,
        load_hamiltonian,
    )

    global _controlled_u_for_logical_probe
    global _probe_mode, _probe_pre_error, _probe_post_error
    global _probe_theta_rad, _probe_correction_angle_rad

    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)
    pytket_circuit, controlled_matrix = controlled_u_pytket_circuit(
        eigenvalues, eigenvectors, evolution_time, k
    )
    _controlled_u_for_logical_probe = guppy.load_pytket(
        f"logical_controlled_u_k{k}", pytket_circuit, use_arrays=False
    )

    _probe_mode = mode_index
    _probe_pre_error = pre_index
    _probe_post_error = post_index
    _probe_theta_rad = theta_rad
    _probe_correction_angle_rad = correction_angle_rad

    compiled = logical_controlled_u_emulator_demo.compile()
    emulator_result = (
        logical_controlled_u_emulator_demo.emulator(n_qubits=n_qubits)
        .with_shots(shots)
        .with_seed(seed)
        .with_n_processes(1)
        .run()
    )
    if len(emulator_result.results) != shots:
        raise AssertionError(
            f"expected {shots} shots, received {len(emulator_result.results)}"
        )

    partial_states = emulator_result.partial_state_dicts()
    if not partial_states:
        raise AssertionError("Selene did not return a state_result snapshot")
    traced_states = partial_states[0]["logical_controlled_state"].state_distribution()
    if not traced_states:
        raise AssertionError("Selene returned no logical statevector")
    observed_state = np.asarray(traced_states[0].state, dtype=complex)

    expected_state = _logical_reference_state(
        mode_index, ground_state, controlled_matrix, correction_angle_rad
    )
    fidelity = float(abs(np.vdot(expected_state, observed_state)) ** 2)

    pre_syndromes = Counter()
    post_syndromes = Counter()
    ancilla_counts = Counter()
    for shot in emulator_result.results:
        entries = dict(shot.entries)
        pre_syndromes[f"{int(entries['pre_s0'])}{int(entries['pre_s1'])}"] += 1
        post_syndromes[f"{int(entries['post_s0'])}{int(entries['post_s1'])}"] += 1
        ancilla_counts[str(int(entries["logical_ancilla"]))] += 1

    expected_syndrome = {None: "00", 0: "10", 1: "11", 2: "01"}
    validated = (
        fidelity >= 1.0 - 1e-8
        and pre_syndromes == Counter({expected_syndrome[pre_error]: shots})
        and post_syndromes == Counter({expected_syndrome[post_error]: shots})
    )
    return {
        "experiment": "encoded-ancilla exact controlled-U component",
        "full_logical_ipea": False,
        "code": "three-qubit bit-flip repetition code",
        "mode": mode,
        "k": int(k),
        "control_wire": "d0 (physical member of logical ancilla codeword)",
        "controlled_u_exact_matrix_verified_by_pytket": True,
        "phase_correction_angle_radians": float(correction_angle_rad),
        "phase_correction_bits_acquisition_order": (
            None if phase_correction_bits is None
            else [int(bit) for bit in phase_correction_bits]
        ),
        "shots": shots,
        "seed": seed,
        "selene_n_qubits": n_qubits,
        "peak_live_qubits": 7,
        "compiled_modules": len(compiled.modules),
        "pytket_gate_count": int(pytket_circuit.n_gates),
        "pre_syndrome_counts": dict(sorted(pre_syndromes.items())),
        "post_syndrome_counts": dict(sorted(post_syndromes.items())),
        "logical_ancilla_counts_after_final_h": dict(sorted(ancilla_counts.items())),
        "state_probability": float(traced_states[0].probability),
        "logical_state_fidelity_to_exact_matrix": fidelity,
        "logical_state": observed_state,
        "reference_state": expected_state,
        "validation_passed": bool(validated),
        "scope_limitations": [
            "This is a QEC component demonstration, not a complete logical IPEA run.",
            "Only one injected X at a pre-CU or post-CU boundary is corrected.",
            "Errors during the imported controlled-U are not injected or claimed corrected.",
            "The bit-flip code does not detect a single Z phase error.",
            "Two data errors are outside the code's single-error correction guarantee.",
            "The controlled-U and phase correction are not demonstrated fault-tolerant logical gates.",
        ],
    }