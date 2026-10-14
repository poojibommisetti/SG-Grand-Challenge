"""Five-bit logical IPEA experiment using the three-qubit bit-flip code.

The circuit is an E3 component experiment, not a fault-tolerant implementation.
It preserves the project H2 Hamiltonian, exact pytket controlled-U powers,
phase convention, and feed-forward order from ipea.py. Each measured logical
ancilla and syndrome pair is destructively consumed; the next round allocates
fresh qubits. The H2 system register remains live through all five rounds.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import owned, result
from guppylang.std.debug import state_result
from guppylang.std.lang import py
from guppylang.std.quantum import cx, discard, h, measure, qubit, rz, x, z

from ipea import (
    bits_to_phase,
    energy_from_phase,
    exact_phase_from_energy,
    expected_adaptive_distribution,
    measured_bits_to_msb_first,
    phase_correction_angle,
    phase_to_bits,
)
from noise_model import (
    DepolarizingNoiseConfig,
    attach_error_model,
    make_run_metadata,
    make_selene_error_model,
)
from state_prep import compute_theta, load_ground_state, prepare_ground_state

_controlled_u_k4 = None
_controlled_u_k3 = None
_controlled_u_k2 = None
_controlled_u_k1 = None
_controlled_u_k0 = None
_theta_rad_for_qec = 0.0
_capture_system_states = False
_fault_code = 0  # deterministic test-only fault selector


@guppy
def _encode_logical_plus() -> tuple[qubit, qubit, qubit]:
    """Prepare a fresh logical |+> as (|000> + |111>)/sqrt(2)."""
    d0 = qubit()
    d1 = qubit()
    d2 = qubit()
    h(d0)
    cx(d0, d1)
    cx(d0, d2)
    return d0, d1, d2


@guppy
def _recover_bitflip(
    d0: qubit @ owned, d1: qubit @ owned, d2: qubit @ owned
) -> tuple[qubit, qubit, qubit, bool, bool]:
    """Measure fresh ZZ stabilizer ancillas, correct one X, return data+syndrome."""
    s0 = qubit()
    s1 = qubit()
    cx(d0, s0)
    cx(d1, s0)
    cx(d1, s1)
    cx(d2, s1)
    bit0 = measure(s0)
    bit1 = measure(s1)

    # 10 -> X(d0), 11 -> X(d1), 01 -> X(d2), 00 -> no correction.
    if bit0:
        if bit1:
            x(d1)
        else:
            x(d0)
    if not bit0:
        if bit1:
            x(d2)
    return d0, d1, d2, bit0, bit1


@guppy
def _decode_and_measure_logical(
    d0: qubit @ owned, d1: qubit @ owned, d2: qubit @ owned
) -> bool:
    """Decode after recovery, apply IPEA's final H, and destructively measure."""
    cx(d0, d2)
    cx(d0, d1)
    h(d0)
    bit = measure(d0)
    discard(d1)
    discard(d2)
    return bit


@guppy
def qec_logical_ipea_5bit_emulator_demo() -> None:
    """Sequential adaptive IPEA with a fresh encoded ancilla per bit."""
    system0 = qubit()
    system1 = qubit()
    prepare_ground_state(system0, system1, py(_theta_rad_for_qec))

    # k=4: acquire b5. Optional deterministic boundary probes are test-only.
    d0, d1, d2 = _encode_logical_plus()
    if py(_fault_code == 1):
        x(d1)
    if py(_fault_code == 2):
        z(d1)
    d0, d1, d2, pre0, pre1 = _recover_bitflip(d0, d1, d2)
    result("k4_pre_s0", pre0)
    result("k4_pre_s1", pre1)
    _controlled_u_k4(d0, system0, system1)
    # Post-controlled-U fault boundary: inject before the second syndrome.
    if py(_fault_code == 3):
        x(d0)
    if py(_fault_code == 4):
        x(d1)
    if py(_fault_code == 5):
        x(d2)
    if py(_fault_code == 6):
        z(d1)
    if py(_fault_code == 7):
        x(d0)
        x(d1)
    d0, d1, d2, post0, post1 = _recover_bitflip(d0, d1, d2)
    result("k4_post_s0", post0)
    result("k4_post_s1", post1)
    if py(_capture_system_states):
        state_result("system_after_k4", system0, system1)
    b5 = _decode_and_measure_logical(d0, d1, d2)
    result("b5", b5)

    # k=3: acquire b4, feed forward b5 using the existing IPEA formula.
    d0, d1, d2 = _encode_logical_plus()
    d0, d1, d2, pre0, pre1 = _recover_bitflip(d0, d1, d2)
    result("k3_pre_s0", pre0)
    result("k3_pre_s1", pre1)
    _controlled_u_k3(d0, system0, system1)
    d0, d1, d2, post0, post1 = _recover_bitflip(d0, d1, d2)
    result("k3_post_s0", post0)
    result("k3_post_s1", post1)
    if b5:
        rz(d0, angle(py(phase_correction_angle(3, [1], 5) / math.pi)))
    if py(_capture_system_states):
        state_result("system_after_k3", system0, system1)
    b4 = _decode_and_measure_logical(d0, d1, d2)
    result("b4", b4)

    # k=2: acquire b3, conditionally add the two existing correction terms.
    d0, d1, d2 = _encode_logical_plus()
    d0, d1, d2, pre0, pre1 = _recover_bitflip(d0, d1, d2)
    result("k2_pre_s0", pre0)
    result("k2_pre_s1", pre1)
    _controlled_u_k2(d0, system0, system1)
    d0, d1, d2, post0, post1 = _recover_bitflip(d0, d1, d2)
    result("k2_post_s0", post0)
    result("k2_post_s1", post1)
    if b5:
        rz(d0, angle(py(phase_correction_angle(2, [1, 0], 5) / math.pi)))
    if b4:
        rz(d0, angle(py(phase_correction_angle(2, [0, 1], 5) / math.pi)))
    if py(_capture_system_states):
        state_result("system_after_k2", system0, system1)
    b3 = _decode_and_measure_logical(d0, d1, d2)
    result("b3", b3)

    # k=1: acquire b2, conditionally add three correction terms.
    d0, d1, d2 = _encode_logical_plus()
    d0, d1, d2, pre0, pre1 = _recover_bitflip(d0, d1, d2)
    result("k1_pre_s0", pre0)
    result("k1_pre_s1", pre1)
    _controlled_u_k1(d0, system0, system1)
    d0, d1, d2, post0, post1 = _recover_bitflip(d0, d1, d2)
    result("k1_post_s0", post0)
    result("k1_post_s1", post1)
    if b5:
        rz(d0, angle(py(phase_correction_angle(1, [1, 0, 0], 5) / math.pi)))
    if b4:
        rz(d0, angle(py(phase_correction_angle(1, [0, 1, 0], 5) / math.pi)))
    if b3:
        rz(d0, angle(py(phase_correction_angle(1, [0, 0, 1], 5) / math.pi)))
    if py(_capture_system_states):
        state_result("system_after_k1", system0, system1)
    b2 = _decode_and_measure_logical(d0, d1, d2)
    result("b2", b2)

    # k=0: acquire b1, conditionally add all four correction terms.
    d0, d1, d2 = _encode_logical_plus()
    d0, d1, d2, pre0, pre1 = _recover_bitflip(d0, d1, d2)
    result("k0_pre_s0", pre0)
    result("k0_pre_s1", pre1)
    _controlled_u_k0(d0, system0, system1)
    # Targeted fault boundary: after exact controlled-U and before the
    # syndrome/recovery and nonzero adaptive phase correction.
    if py(_fault_code == 8):
        x(d0)
    if py(_fault_code == 9):
        x(d1)
    if py(_fault_code == 10):
        x(d2)
    if py(_fault_code == 11):
        z(d1)
    if py(_fault_code == 12):
        x(d0)
        x(d1)
    d0, d1, d2, post0, post1 = _recover_bitflip(d0, d1, d2)
    result("k0_post_s0", post0)
    result("k0_post_s1", post1)
    if b5:
        rz(d0, angle(py(phase_correction_angle(0, [1, 0, 0, 0], 5) / math.pi)))
    if b4:
        rz(d0, angle(py(phase_correction_angle(0, [0, 1, 0, 0], 5) / math.pi)))
    if b3:
        rz(d0, angle(py(phase_correction_angle(0, [0, 0, 1, 0], 5) / math.pi)))
    if b2:
        rz(d0, angle(py(phase_correction_angle(0, [0, 0, 0, 1], 5) / math.pi)))
    if py(_capture_system_states):
        state_result("system_after_k0", system0, system1)
    b1 = _decode_and_measure_logical(d0, d1, d2)
    result("b1", b1)

    # The H2 system is kept live through all powers and is never measured.
    discard(system0)
    discard(system1)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_fault(fault_injection: str | None) -> int:
    allowed = {
        None: 0,
        "x_pre_k4_d1": 1,
        "z_pre_k4_d1": 2,
        "x_post_k4_d0": 3,
        "x_post_k4_d1": 4,
        "x_post_k4_d2": 5,
        "z_post_k4_d1": 6,
        "x2_post_k4_d0_d1": 7,
        "x_post_k0_d0": 8,
        "x_post_k0_d1": 9,
        "x_post_k0_d2": 10,
        "z_post_k0_d1": 11,
        "x2_post_k0_d0_d1": 12,
    }
    if fault_injection not in allowed:
        raise ValueError(
            "unsupported fault_injection; see _validate_fault for supported probes"
        )
    return allowed[fault_injection]


def _phase_statistics(
    emulator_result, shots: int, evolution_time: float, exact_phase: float,
    energy_interval: tuple[float, float], expected_distribution: dict[str, float],
) -> dict:
    measured_order_counts: Counter[str] = Counter()
    msb_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    energy_counts: Counter[str] = Counter()
    unresolved = 0
    syndrome_counts: dict[str, Counter[str]] = {
        f"k{k}_{stage}": Counter()
        for k in range(4, -1, -1)
        for stage in ("pre", "post")
    }
    for shot in emulator_result.results:
        values = dict(shot.entries)
        measured_bits = [int(values[tag]) for tag in ("b5", "b4", "b3", "b2", "b1")]
        raw_key = "".join(str(bit) for bit in measured_bits)
        measured_order_counts[raw_key] += 1
        msb_bits = measured_bits_to_msb_first(measured_bits)
        msb_key = "".join(str(bit) for bit in msb_bits)
        msb_counts[msb_key] += 1
        phase = bits_to_phase(msb_bits)
        phase_counts[f"{phase:.5f}"] += 1
        try:
            energy = energy_from_phase(phase, evolution_time, energy_interval)
        except ValueError:
            unresolved += 1
        else:
            energy_counts[f"{energy:.12f}"] += 1

        for k in range(4, -1, -1):
            for stage in ("pre", "post"):
                s0 = int(values[f"k{k}_{stage}_s0"])
                s1 = int(values[f"k{k}_{stage}_s1"])
                syndrome_counts[f"k{k}_{stage}"][f"{s0}{s1}"] += 1

    if sum(measured_order_counts.values()) != shots:
        raise AssertionError("Selene returned a different number of five-bit results")

    within_tolerance = True
    for outcome, probability in expected_distribution.items():
        observed = measured_order_counts.get(outcome, 0)
        expected_count = shots * probability
        sigma = math.sqrt(shots * probability * (1.0 - probability))
        if abs(observed - expected_count) > 5.0 * sigma + 1.0:
            within_tolerance = False

    modal_raw = max(measured_order_counts, key=measured_order_counts.get)
    modal_msb_bits = measured_bits_to_msb_first([int(bit) for bit in modal_raw])
    modal_msb = "".join(str(bit) for bit in modal_msb_bits)
    modal_phase = bits_to_phase(modal_msb_bits)
    try:
        modal_energy = energy_from_phase(
            modal_phase, evolution_time, energy_interval
        )
    except ValueError:
        modal_energy = None
    return {
        "measured_counts_descending_k": {
            outcome: int(measured_order_counts.get(outcome, 0))
            for outcome in sorted(expected_distribution)
        },
        "measured_counts_msb_first": {
            outcome: int(msb_counts.get(outcome, 0))
            for outcome in sorted(msb_counts)
        },
        "reconstructed_phase_counts": dict(sorted(phase_counts.items())),
        "energy_distribution_hartree": dict(sorted(energy_counts.items())),
        "unresolved_shots_outside_energy_interval": int(unresolved),
        "modal_measurement_descending_k": modal_raw,
        "modal_bits_msb_first": modal_msb,
        "modal_frequency": int(measured_order_counts[modal_raw]),
        "modal_probability": measured_order_counts[modal_raw] / shots,
        "modal_phase": modal_phase,
        "modal_energy_hartree": modal_energy,
        "modal_absolute_energy_error_hartree": (
            None if modal_energy is None
            else abs(modal_energy - energy_from_phase(
                exact_phase, evolution_time, energy_interval
            ))
        ),
        "within_ideal_distribution_tolerance": bool(within_tolerance),
        "phase_resolution_check": bool(abs(modal_phase - exact_phase) < 1 / 32),
        "syndrome_counts_by_iteration_and_stage": {
            key: dict(sorted(counts.items()))
            for key, counts in syndrome_counts.items()
        },
    }


def run_qec_logical_ipea_5bit(
    *,
    shots: int,
    seed: int,
    n_processes: int,
    noise_config: DepolarizingNoiseConfig | None = None,
    fault_injection: str | None = None,
    capture_system_states: bool = False,
    energy_interval: tuple[float, float] = (-1.2, -1.0),
) -> dict:
    """Compile and run sequential five-bit logical IPEA using local Selene."""
    shots = _positive_integer("shots", shots)
    seed = _nonnegative_integer("seed", seed)
    n_processes = _positive_integer("n_processes", n_processes)
    if not isinstance(capture_system_states, bool):
        raise TypeError("capture_system_states must be bool")
    fault_code = _validate_fault(fault_injection)

    from ipea import phase_correction_angle
    from time_evolution import (
        choose_evolution_time,
        controlled_u_pytket_circuit,
        diagonalize,
        load_hamiltonian,
    )

    global _controlled_u_k4, _controlled_u_k3, _controlled_u_k2
    global _controlled_u_k1, _controlled_u_k0, _theta_rad_for_qec
    global _capture_system_states, _fault_code

    a, b, e0_from_state = load_ground_state()
    _theta_rad_for_qec = compute_theta(a, b)
    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)
    circuits = {}
    gates = {}
    for k in range(4, -1, -1):
        circuit, _ = controlled_u_pytket_circuit(
            eigenvalues, eigenvectors, evolution_time, k=k
        )
        circuits[k] = circuit
        gates[k] = guppy.load_pytket(
            f"qec_logical_ipea_exact_controlled_u_k{k}",
            circuit,
            use_arrays=False,
        )
    _controlled_u_k4 = gates[4]
    _controlled_u_k3 = gates[3]
    _controlled_u_k2 = gates[2]
    _controlled_u_k1 = gates[1]
    _controlled_u_k0 = gates[0]
    _capture_system_states = capture_system_states
    _fault_code = fault_code

    compiled = qec_logical_ipea_5bit_emulator_demo.compile()
    emulator = (
        qec_logical_ipea_5bit_emulator_demo.emulator(n_qubits=7)
        .with_shots(shots)
        .with_seed(seed)
        .with_n_processes(n_processes)
    )
    noise_metadata = None
    if noise_config is not None:
        error_model = make_selene_error_model(noise_config, random_seed=seed)
        emulator = attach_error_model(emulator, error_model)
        noise_metadata = make_run_metadata(
            noise_config,
            random_seed=seed,
            shots=shots,
            repeats=1,
            n_qubits=7,
        )
    emulator_result = emulator.run()

    exact_energy = float(eigenvalues[0])
    exact_phase = exact_phase_from_energy(exact_energy, evolution_time)
    reference_bits = phase_to_bits(exact_phase, 5)
    reference_phase = bits_to_phase(reference_bits)
    expected_distribution = expected_adaptive_distribution(exact_phase, 5)
    statistics = _phase_statistics(
        emulator_result, shots, evolution_time, exact_phase,
        energy_interval, expected_distribution,
    )
    statistics["modal_absolute_energy_error_vs_e0_hartree"] = (
        None if statistics["modal_energy_hartree"] is None
        else abs(statistics["modal_energy_hartree"] - float(e0_from_state))
    )
    ideal_checks_passed = (
        statistics["within_ideal_distribution_tolerance"]
        and statistics["phase_resolution_check"]
        and statistics["modal_energy_hartree"] is not None
    )

    system_fidelities = None
    if capture_system_states:
        partial_states = emulator_result.partial_state_dicts()
        if not partial_states:
            raise AssertionError("Selene returned no requested system snapshots")
        expected_system = np.array([a, 0.0, 0.0, b], dtype=complex)
        system_fidelities = {}
        for k in (4, 3, 2, 1, 0):
            tag = f"system_after_k{k}"
            traces = partial_states[0][tag].state_distribution()
            if len(traces) != 1:
                raise AssertionError(f"expected a pure system state after k={k}")
            state = np.asarray(traces[0].state, dtype=complex)
            system_fidelities[tag] = float(abs(np.vdot(expected_system, state)) ** 2)

    from time_evolution import controlled_u_pytket_circuit as _unused  # API provenance

    return {
        "label": (
            "E3 logical IPEA with illustrative Selene depolarizing model"
            if noise_config is not None
            else "E3 logical IPEA ideal"
        ),
        "full_five_bit_logical_ipea": True,
        "execution_completed": True,
        "validation_passed": (
            None if noise_config is not None else bool(ideal_checks_passed)
        ),
        "ideal_distribution_checks_passed": bool(ideal_checks_passed),
        "noise_model_metadata": noise_metadata,
        "environment": {
            "python": "3.11.9",
            "guppylang": version("guppylang"),
            "selene-sim": version("selene-sim"),
            "selene-core": version("selene-core"),
            "pytket": version("pytket"),
        },
        "settings": {
            "shots": shots,
            "repeats": 1,
            "seed": seed,
            "n_processes": n_processes,
            "energy_interval_hartree": [float(x) for x in energy_interval],
            "n_qubits_available": 7,
            "peak_live_qubits": 7,
            "simulator": "local Guppy/Selene Quest statevector emulator",
            "fault_injection": fault_injection,
            "capture_system_states": capture_system_states,
        },
        "compiled_modules": len(compiled.modules),
        "iteration_k_order": [4, 3, 2, 1, 0],
        "measurement_order": ["b5", "b4", "b3", "b2", "b1"],
        "bitstring_convention": (
            "acquisition is b5..b1; measured_bits_to_msb_first reverses it "
            "before phase reconstruction, matching ipea.py"
        ),
        "exact_ground_energy_hartree": exact_energy,
        "e0_state_file_energy_hartree": float(e0_from_state),
        "evolution_time": float(evolution_time),
        "exact_ground_phase": exact_phase,
        "reference_bits_msb_first": reference_bits,
        "reference_phase": reference_phase,
        "expected_measurement_probabilities_descending_k": expected_distribution,
        "controlled_u_gate_counts_k4_to_k0": [
            int(circuits[k].n_gates) for k in range(4, -1, -1)
        ],
        "logical_recovery_syndromes": statistics[
            "syndrome_counts_by_iteration_and_stage"
        ],
        "system_state_fidelities_after_each_iteration": system_fidelities,
        **statistics,
    }


def write_e0_e1_e2_e3_comparison(
    e3_ideal: dict,
    e3_noisy: dict,
    *,
    source_path: str | Path = "results/ipea_e1_e2_comparison.json",
    e0_path: str | Path = "results/e0.json",
    output_path: str | Path = "results/e0_e1_e2_e3_comparison.json",
) -> dict:
    """Write a new comparison record without modifying E0-E2 source records."""
    with Path(source_path).open(encoding="utf-8") as handle:
        prior = json.load(handle)
    with Path(e0_path).open(encoding="utf-8") as handle:
        e0 = json.load(handle)
    artifact = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "H2/STO-3G five-bit ideal, noisy, and logical IPEA comparison",
        "e0_reference": prior["e0_reference"],
        "e0_source_record": {
            "path": str(e0_path),
            "energy_hartree": float(e0["E0_hartree"]),
            "coefficient_source": e0["coefficient_source"],
            "method": e0["method"],
        },
        "comparison_settings": {
            "phase_convention": "U(t)=exp(-iHt), phi=frac(-E*t/(2*pi))",
            "energy_interval_hartree": [-1.2, -1.0],
            "energy_conversion": "ipea.energy_from_phase with explicit interval",
            "repeats_per_condition": 1,
            "uncertainty_note": (
                "One run per condition; shot-count variation is sampling "
                "variation, not repeat-to-repeat uncertainty."
            ),
        },
        "E1_ideal": prior["E1_ideal"],
        "E2_noisy": prior["E2_noisy"],
        "E3_logical_ideal": e3_ideal,
        "E3_logical_noisy": e3_noisy,
        "comparison_warnings": [
            "E1/E2 use n_qubits=12; this E3 circuit uses seven simultaneously live qubits.",
            "All conditions use 5000 shots, seed 20261003, and the same H2 Hamiltonian, phase convention, and energy interval.",
            "E2's recorded ideal-distribution tolerance flag is false.",
            "E3 uses an illustrative Selene noise model, not calibrated H2 hardware noise.",
            "This bit-flip code does not correct phase errors, multiple data errors, errors during controlled-U, or errors on the unencoded H2 system.",
            "No condition has repeated runs, so repeat-to-repeat uncertainty is unavailable.",
        ],
    }
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    return artifact


def _write_json_record(path: str | Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

def main() -> None:
    shots, seed, n_processes = 5000, 20261003, 8
    ideal = run_qec_logical_ipea_5bit(
        shots=shots, seed=seed, n_processes=n_processes
    )
    print("E3 ideal validation:", ideal["validation_passed"])
    if not ideal["validation_passed"]:
        raise RuntimeError(
            "Ideal E3 failed the expected-distribution or phase-resolution check; "
            "no noisy E3 run or comparison artifact was created."
        )

    _write_json_record("results/ipea_e3_logical_ideal.json", ideal)

    config = DepolarizingNoiseConfig(
        p_1q=1e-5,
        p_2q=1e-6,
        p_meas=1e-2,
        p_init=1e-3,
    )
    noisy = run_qec_logical_ipea_5bit(
        shots=shots,
        seed=seed,
        n_processes=n_processes,
        noise_config=config,
    )
    _write_json_record("results/ipea_e3_logical_noisy.json", noisy)
    artifact = write_e0_e1_e2_e3_comparison(ideal, noisy)
    print("Wrote results/e0_e1_e2_e3_comparison.json")
    print(json.dumps({
        "E3_ideal": {
            "modal_bits_msb_first": ideal["modal_bits_msb_first"],
            "modal_frequency": ideal["modal_frequency"],
            "modal_phase": ideal["modal_phase"],
            "modal_energy_hartree": ideal["modal_energy_hartree"],
            "absolute_error_vs_e0_hartree": ideal["modal_absolute_energy_error_vs_e0_hartree"],
            "validation_passed": ideal["validation_passed"],
        },
        "E3_noisy": {
            "modal_bits_msb_first": noisy["modal_bits_msb_first"],
            "modal_frequency": noisy["modal_frequency"],
            "modal_phase": noisy["modal_phase"],
            "modal_energy_hartree": noisy["modal_energy_hartree"],
            "absolute_error_vs_e0_hartree": noisy["modal_absolute_energy_error_vs_e0_hartree"],
            "noise_model_metadata": noisy["noise_model_metadata"],
        },
        "comparison_conditions": artifact["comparison_settings"],
    }, indent=2))


if __name__ == "__main__":
    main()
