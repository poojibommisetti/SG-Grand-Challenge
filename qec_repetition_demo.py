"""Standalone Selene demo of single-X correction with a 3-qubit repetition code.

This is a QEC component test only; it is not a logical IPEA circuit.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, discard, measure, qubit, x

from noise_model import (
    DepolarizingNoiseConfig,
    attach_error_model,
    make_run_metadata,
    make_selene_error_model,
)


@guppy
def unprotected_q0_x_error_demo() -> None:
    """Prepare |0>, inject X, and read out the unprotected qubit."""
    q0 = qubit()
    x(q0)
    output_bit = measure(q0)
    result("unprotected_bit", output_bit)


@guppy
def bitflip_repetition_q0_x_error_demo() -> None:
    """Encode |0>, inject X on data q0, correct its syndrome, and decode."""
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()

    # Encode alpha|0> + beta|1> as alpha|000> + beta|111>.
    cx(q0, q1)
    cx(q0, q2)

    # Deterministic fault used to verify the circuit-level recovery path.
    x(q0)

    # Measure Z0*Z1 and Z1*Z2 using two fresh ancillas.
    syndrome0 = qubit()
    syndrome1 = qubit()
    cx(q0, syndrome0)
    cx(q1, syndrome0)
    cx(q1, syndrome1)
    cx(q2, syndrome1)
    s0 = measure(syndrome0)
    s1 = measure(syndrome1)
    result("syndrome_s0", s0)
    result("syndrome_s1", s1)

    # 10 -> X(q0), 11 -> X(q1), 01 -> X(q2), 00 -> no correction.
    if s0:
        if s1:
            x(q1)
        else:
            x(q0)
    else:
        if s1:
            x(q2)

    # Decode and check the recovered logical data bit.
    cx(q0, q2)
    cx(q0, q1)
    decoded_bit = measure(q0)
    result("decoded_bit", decoded_bit)
    discard(q1)
    discard(q2)


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def run_selene_repetition_component(
    *,
    shots: int,
    seed: int,
    n_processes: int,
    noise_config: DepolarizingNoiseConfig | None = None,
) -> dict:
    """Run unprotected and encoded known-X probes through local Selene."""
    shots = _validate_positive_integer("shots", shots)
    seed = _validate_positive_integer("seed", seed)
    n_processes = _validate_positive_integer("n_processes", n_processes)

    unprotected_compiled = unprotected_q0_x_error_demo.compile()
    encoded_compiled = bitflip_repetition_q0_x_error_demo.compile()

    unprotected_emulator = (
        unprotected_q0_x_error_demo.emulator(n_qubits=1)
        .with_shots(shots)
        .with_seed(seed)
        .with_n_processes(n_processes)
    )
    encoded_emulator = (
        bitflip_repetition_q0_x_error_demo.emulator(n_qubits=5)
        .with_shots(shots)
        .with_seed(seed)
        .with_n_processes(n_processes)
    )

    noise_metadata = None
    if noise_config is not None:
        unprotected_emulator = attach_error_model(
            unprotected_emulator,
            make_selene_error_model(noise_config, random_seed=seed),
        )
        encoded_emulator = attach_error_model(
            encoded_emulator,
            make_selene_error_model(noise_config, random_seed=seed),
        )
        noise_metadata = make_run_metadata(
            noise_config,
            random_seed=seed,
            shots=shots,
            repeats=1,
            n_qubits=5,
        )

    unprotected_result = unprotected_emulator.run()
    encoded_result = encoded_emulator.run()

    unprotected_counts: Counter[str] = Counter()
    syndrome_counts: Counter[str] = Counter()
    decoded_counts: Counter[str] = Counter()
    joint_counts: Counter[str] = Counter()
    for shot in unprotected_result.results:
        values = dict(shot.entries)
        unprotected_counts[str(int(values["unprotected_bit"]))] += 1
    for shot in encoded_result.results:
        values = dict(shot.entries)
        s0 = int(values["syndrome_s0"])
        s1 = int(values["syndrome_s1"])
        decoded = int(values["decoded_bit"])
        syndrome_counts[f"{s0}{s1}"] += 1
        decoded_counts[str(decoded)] += 1
        joint_counts[f"{s0}{s1}:{decoded}"] += 1

    if sum(unprotected_counts.values()) != shots or sum(decoded_counts.values()) != shots:
        raise AssertionError("Selene returned an unexpected number of probe results")

    return {
        "experiment": "single injected X on q0; bit-flip repetition component only",
        "full_logical_ipea": False,
        "shots_per_circuit": shots,
        "repeats": 1,
        "seed": seed,
        "n_processes": n_processes,
        "physical_qubits": {"unprotected_probe": 1, "encoded_peak": 5},
        "compiled_modules": {
            "unprotected": len(unprotected_compiled.modules),
            "encoded": len(encoded_compiled.modules),
        },
        "noise_model_metadata": noise_metadata,
        "expected_ideal": {
            "unprotected_bit": 1,
            "encoded_syndrome_s0_s1": "10",
            "encoded_decoded_bit": 0,
        },
        "unprotected_counts": dict(sorted(unprotected_counts.items())),
        "encoded_syndrome_counts_s0_s1": dict(sorted(syndrome_counts.items())),
        "encoded_decoded_counts": dict(sorted(decoded_counts.items())),
        "encoded_joint_counts_syndrome_then_decoded": dict(sorted(joint_counts.items())),
        "encoded_recovery_success_probability": decoded_counts.get("0", 0) / shots,
    }