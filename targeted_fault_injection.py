"""Targeted Pauli faults at the final feed-forward boundary of five-bit IPEA.

This is a controlled fault-injection study, not a depolarizing-noise or general
fault-tolerance benchmark. The physical X fault is placed after exact
controlled-U (k=0), before the repetition-code syndrome/recovery and phase
correction. At this boundary the IPEA phase correction makes an ancilla X
observable in the final phase bit.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.lang import py
from guppylang.std.quantum import discard, h, measure, qubit, rz, x, z

from ipea import (
    bits_to_phase,
    energy_from_phase,
    expected_adaptive_distribution,
    phase_correction_angle,
    phase_to_bits,
)
from state_prep import prepare_ground_state

_theta_rad = 0.0
_cu_k4 = None
_cu_k3 = None
_cu_k2 = None
_cu_k1 = None
_cu_k0 = None


@guppy
def _measure_b5(system0: qubit, system1: qubit) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k4(ancilla, system0, system1)
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b4(system0: qubit, system1: qubit, b5: bool) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k3(ancilla, system0, system1)
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(3, [1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b3(system0: qubit, system1: qubit, b5: bool, b4: bool) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k2(ancilla, system0, system1)
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(2, [1, 0], 5) / math.pi)))
    if b4:
        rz(ancilla, angle(py(phase_correction_angle(2, [0, 1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b2(
    system0: qubit, system1: qubit, b5: bool, b4: bool, b3: bool
) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k1(ancilla, system0, system1)
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(1, [1, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla, angle(py(phase_correction_angle(1, [0, 1, 0], 5) / math.pi)))
    if b3:
        rz(ancilla, angle(py(phase_correction_angle(1, [0, 0, 1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b1_ideal(
    system0: qubit, system1: qubit, b5: bool, b4: bool, b3: bool, b2: bool
) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k0(ancilla, system0, system1)
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(0, [1, 0, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 1, 0, 0], 5) / math.pi)))
    if b3:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 1, 0], 5) / math.pi)))
    if b2:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 0, 1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b1_x_fault(
    system0: qubit, system1: qubit, b5: bool, b4: bool, b3: bool, b2: bool
) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k0(ancilla, system0, system1)
    x(ancilla)  # fault: after CU, before adaptive phase correction
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(0, [1, 0, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 1, 0, 0], 5) / math.pi)))
    if b3:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 1, 0], 5) / math.pi)))
    if b2:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 0, 1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _measure_b1_z_fault(
    system0: qubit, system1: qubit, b5: bool, b4: bool, b3: bool, b2: bool
) -> bool:
    ancilla = qubit()
    h(ancilla)
    _cu_k0(ancilla, system0, system1)
    z(ancilla)  # limitation probe: phase-flip fault at the same boundary
    if b5:
        rz(ancilla, angle(py(phase_correction_angle(0, [1, 0, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 1, 0, 0], 5) / math.pi)))
    if b3:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 1, 0], 5) / math.pi)))
    if b2:
        rz(ancilla, angle(py(phase_correction_angle(0, [0, 0, 0, 1], 5) / math.pi)))
    h(ancilla)
    return measure(ancilla).read()


@guppy
def _ipea_ideal_demo() -> None:
    system0 = qubit()
    system1 = qubit()
    prepare_ground_state(system0, system1, py(_theta_rad))
    b5 = _measure_b5(system0, system1)
    b4 = _measure_b4(system0, system1, b5)
    b3 = _measure_b3(system0, system1, b5, b4)
    b2 = _measure_b2(system0, system1, b5, b4, b3)
    b1 = _measure_b1_ideal(system0, system1, b5, b4, b3, b2)
    result("b5", b5)
    result("b4", b4)
    result("b3", b3)
    result("b2", b2)
    result("b1", b1)
    discard(system0)
    discard(system1)


@guppy
def _ipea_x_fault_demo() -> None:
    system0 = qubit()
    system1 = qubit()
    prepare_ground_state(system0, system1, py(_theta_rad))
    b5 = _measure_b5(system0, system1)
    b4 = _measure_b4(system0, system1, b5)
    b3 = _measure_b3(system0, system1, b5, b4)
    b2 = _measure_b2(system0, system1, b5, b4, b3)
    b1 = _measure_b1_x_fault(system0, system1, b5, b4, b3, b2)
    result("b5", b5)
    result("b4", b4)
    result("b3", b3)
    result("b2", b2)
    result("b1", b1)
    discard(system0)
    discard(system1)


@guppy
def _ipea_z_fault_demo() -> None:
    system0 = qubit()
    system1 = qubit()
    prepare_ground_state(system0, system1, py(_theta_rad))
    b5 = _measure_b5(system0, system1)
    b4 = _measure_b4(system0, system1, b5)
    b3 = _measure_b3(system0, system1, b5, b4)
    b2 = _measure_b2(system0, system1, b5, b4, b3)
    b1 = _measure_b1_z_fault(system0, system1, b5, b4, b3, b2)
    result("b5", b5)
    result("b4", b4)
    result("b3", b3)
    result("b2", b2)
    result("b1", b1)
    discard(system0)
    discard(system1)


def _full_counts(emulator_result, shots: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for shot in emulator_result.results:
        values = dict(shot.entries)
        descending_k = "".join(str(int(values[tag])) for tag in ("b5", "b4", "b3", "b2", "b1"))
        counts[descending_k[::-1]] += 1
    normalized = {format(i, "05b"): int(counts.get(format(i, "05b"), 0)) for i in range(32)}
    if sum(normalized.values()) != shots:
        raise AssertionError("targeted unprotected result count does not equal shots")
    return normalized


def _statistics(counts: dict[str, int], *, shots: int, seed: int, n_processes: int,
                fault: str | None, compiled_modules: int, exact_energy: float,
                exact_phase: float, evolution_time: float, e0_energy: float,
                energy_interval: tuple[float, float], gate_counts: list[int]) -> dict:
    reference = "".join(map(str, phase_to_bits(exact_phase, 5)))
    expected_desc = expected_adaptive_distribution(exact_phase, 5)
    expected_msb = {bits[::-1]: probability for bits, probability in expected_desc.items()}
    tv = 0.5 * math.fsum(abs(counts[bits] / shots - expected_msb[bits]) for bits in counts)
    within = all(
        abs(counts[bits] - shots * expected_msb[bits])
        <= 5.0 * math.sqrt(shots * expected_msb[bits] * (1.0 - expected_msb[bits])) + 1.0
        for bits in counts
    )
    phases = {bits: bits_to_phase([int(ch) for ch in bits]) for bits in counts}
    energies: dict[str, float] = {}
    unresolved: dict[str, int] = {}
    for bits, count in counts.items():
        try:
            energies[bits] = energy_from_phase(phases[bits], evolution_time, energy_interval)
        except ValueError as exc:
            if "no phase alias" not in str(exc):
                raise
            if count:
                unresolved[bits] = count
    modal = max(counts, key=counts.get)
    modal_energy = energies.get(modal)
    return {
        "label": "targeted unprotected five-bit IPEA; deterministic boundary fault probe",
        "mode": "unprotected",
        "fault": fault,
        "fault_boundary": None if fault is None else "after exact controlled-U^1 (k=0), before phase correction and final H/readout",
        "shots": shots,
        "seed": seed,
        "n_processes": n_processes,
        "n_qubits": 3,
        "simulator": "local Guppy/Selene Quest emulator",
        "compiled_modules": compiled_modules,
        "controlled_u_k_order": [4, 3, 2, 1, 0],
        "measurement_order": ["b5", "b4", "b3", "b2", "b1"],
        "energy_interval_hartree": list(energy_interval),
        "evolution_time": evolution_time,
        "exact_ground_energy_hartree": exact_energy,
        "e0_energy_hartree": e0_energy,
        "exact_ground_phase": exact_phase,
        "reference_bits_msb_first": reference,
        "expected_distribution_msb_first": expected_msb,
        "measured_counts_msb_first": counts,
        "modal_bits_msb_first": modal,
        "modal_probability": counts[modal] / shots,
        "reference_probability": counts.get(reference, 0) / shots,
        "phase_outcome_total_variation_from_ideal": tv,
        "within_ideal_distribution_tolerance": within,
        "modal_phase": phases[modal],
        "modal_energy_hartree": modal_energy,
        "modal_energy_absolute_error_vs_e0_hartree": None if modal_energy is None else abs(modal_energy - e0_energy),
        "resolved_shots": shots - sum(unresolved.values()),
        "unresolved_shots": sum(unresolved.values()),
        "unresolved_phase_bit_counts": unresolved,
        "pytket_gate_counts_k4_to_k0": gate_counts,
    }


def run_unprotected_fault_ipea(
    *, shots: int, seed: int, n_processes: int = 1, fault: str | None = None,
    energy_interval: tuple[float, float] = (-1.2, -1.0),
) -> dict:
    """Run the unprotected ancilla with no fault, X, or Z after the k=0 CU."""
    if fault not in (None, "x_post_k0", "z_post_k0"):
        raise ValueError("fault must be None, 'x_post_k0', or 'z_post_k0'")
    if shots < 1 or seed < 0 or n_processes < 1:
        raise ValueError("shots/process count must be positive and seed non-negative")

    from guppylang import guppy as guppy_module
    from time_evolution import choose_evolution_time, controlled_u_pytket_circuit, diagonalize, load_hamiltonian
    from state_prep import compute_theta, load_ground_state

    global _theta_rad, _cu_k4, _cu_k3, _cu_k2, _cu_k1, _cu_k0
    a, b, e0_energy = load_ground_state()
    _theta_rad = compute_theta(a, b)
    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)
    circuits = {}
    gates = {}
    for k in range(4, -1, -1):
        circuit, _ = controlled_u_pytket_circuit(eigenvalues, eigenvectors, evolution_time, k=k)
        circuits[k] = circuit
        gates[k] = guppy_module.load_pytket(f"targeted_unprotected_exact_cu_k{k}", circuit, use_arrays=False)
    _cu_k4, _cu_k3, _cu_k2, _cu_k1, _cu_k0 = (gates[k] for k in (4, 3, 2, 1, 0))
    entrypoint = {None: _ipea_ideal_demo, "x_post_k0": _ipea_x_fault_demo, "z_post_k0": _ipea_z_fault_demo}[fault]
    compiled = entrypoint.compile()
    emulator_result = (
        entrypoint.emulator(n_qubits=3)
        .with_shots(int(shots))
        .with_seed(int(seed))
        .with_n_processes(int(n_processes))
        .run()
    )
    exact_energy = float(eigenvalues[0])
    exact_phase = (-exact_energy * evolution_time / (2.0 * math.pi)) % 1.0
    counts = _full_counts(emulator_result, shots)
    return _statistics(
        counts, shots=shots, seed=seed, n_processes=n_processes, fault=fault,
        compiled_modules=len(compiled.modules), exact_energy=exact_energy,
        exact_phase=exact_phase, evolution_time=float(evolution_time),
        e0_energy=float(e0_energy), energy_interval=energy_interval,
        gate_counts=[int(circuits[k].n_gates) for k in (4, 3, 2, 1, 0)],
    )


def _summary(record: dict, *, encoded: bool) -> dict:
    shots = record["settings"]["shots"] if encoded else record["shots"]
    counts = record["measured_counts_msb_first"]
    ref_prob = counts.get("10100", 0) / shots
    if encoded:
        expected_desc = record["expected_measurement_probabilities_descending_k"]
        expected_msb = {bits[::-1]: probability for bits, probability in expected_desc.items()}
    else:
        expected_msb = record["expected_distribution_msb_first"]
    tv = 0.5 * math.fsum(
        abs(counts.get(bits, 0) / shots - expected_msb.get(bits, 0.0))
        for bits in (format(i, "05b") for i in range(32))
    )
    return {
        "reference_probability": ref_prob,
        "modal_bits_msb_first": record["modal_bits_msb_first"],
        "modal_probability": record["modal_probability"],
        "modal_energy_absolute_error_vs_e0_hartree": record.get("modal_absolute_energy_error_vs_e0_hartree", record.get("modal_energy_absolute_error_vs_e0_hartree")),
        "unresolved_probability": (record["unresolved_shots_outside_energy_interval"] if encoded else record["unresolved_shots"]) / shots,
        "within_ideal_distribution_tolerance": record.get("within_ideal_distribution_tolerance"),
        "phase_outcome_total_variation_from_ideal": tv,
    }


def run_targeted_fault_suite(
    *, shots: int = 1000, seed: int = 20261021, n_processes: int = 8,
    energy_interval: tuple[float, float] = (-1.2, -1.0),
    output_path: str | Path = "results/e3_targeted_x_fault_experiment.json",
) -> dict:
    """Run no-fault, all single-X data locations, plus Z and double-X limits."""
    from qec_logical_ipea import run_qec_logical_ipea_5bit

    cases = [
        ("no_fault", None, None, "00"),
        ("single_x_d0", "x_post_k0", "x_post_k0_d0", "10"),
        ("single_x_d1", "x_post_k0", "x_post_k0_d1", "11"),
        ("single_x_d2", "x_post_k0", "x_post_k0_d2", "01"),
        ("single_z_d1_limitation", "z_post_k0", "z_post_k0_d1", "00"),
        ("double_x_d0_d1_limitation", None, "x2_post_k0_d0_d1", "01"),
    ]
    artifact = {
        "label": "targeted Pauli fault-injection study; not a depolarizing-noise or general QEC benchmark",
        "status": "running",
        "settings": {
            "shots_per_condition": shots,
            "base_seed": seed,
            "seed_per_case": "base_seed + case_index; paired ideal/unprotected and encoded settings share a seed but use different circuit draws",
            "n_processes": n_processes,
            "energy_interval_hartree": list(energy_interval),
            "simulator": "local Guppy/Selene Quest emulator",
            "phase_reference_bits_msb_first": "10100",
            "fault_boundary": "after exact controlled-U^1 (k=0), before phase correction, final H, and readout; encoded faults precede k0 post-syndrome extraction",
            "noise_model": None,
        },
        "conditions": [],
        "limitations": [
            "Deterministic Pauli injections only; not a general noise model.",
            "The three-qubit bit-flip repetition code detects and corrects a single X on the encoded ancilla at this tested syndrome boundary.",
            "A single Z is not detected by these ZZ syndromes; two X faults can be miscorrected into a logical error.",
            "This does not protect the H2 system register or faults inside the controlled-U decomposition.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for index, (name, unprotected_fault, encoded_fault, expected_syndrome) in enumerate(cases):
            case_seed = seed + index
            print(f"Fault case {name}: seed={case_seed}, shots={shots}", flush=True)
            unprotected = None
            if name != "double_x_d0_d1_limitation":
                unprotected = run_unprotected_fault_ipea(
                    shots=shots, seed=case_seed, n_processes=n_processes,
                    fault=unprotected_fault, energy_interval=energy_interval,
                )
            encoded = run_qec_logical_ipea_5bit(
                shots=shots, seed=case_seed, n_processes=n_processes,
                fault_injection=encoded_fault, energy_interval=energy_interval,
            )
            syndrome_counts = encoded["syndrome_counts_by_iteration_and_stage"]["k0_post"]
            syndrome_ok = syndrome_counts == {expected_syndrome: shots}
            up_summary = None if unprotected is None else _summary(unprotected, encoded=False)
            enc_summary = _summary(encoded, encoded=True)
            ref_delta = None if up_summary is None else enc_summary["reference_probability"] - up_summary["reference_probability"]
            if name.startswith("single_x_"):
                evidence = bool(
                    syndrome_ok
                    and encoded["within_ideal_distribution_tolerance"]
                    and enc_summary["phase_outcome_total_variation_from_ideal"] < 0.10
                    and up_summary["phase_outcome_total_variation_from_ideal"] > enc_summary["phase_outcome_total_variation_from_ideal"] + 0.15
                    and ref_delta > 0.30
                )
            else:
                evidence = None
            artifact["conditions"].append({
                "case": name,
                "seed": case_seed,
                "expected_encoded_k0_post_syndrome": expected_syndrome,
                "observed_encoded_k0_post_syndrome_counts": syndrome_counts,
                "encoded_syndrome_check_passed": syndrome_ok,
                "unprotected_summary": up_summary,
                "encoded_summary": enc_summary,
                "encoded_minus_unprotected_reference_probability": ref_delta,
                "targeted_x_correction_evidence_supported": evidence,
                "unprotected_record": unprotected,
                "encoded_record": encoded,
            })
            save()
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed_or_interrupted"
        artifact["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    save()
    return artifact


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--n-processes", type=int, default=8)
    parser.add_argument("--output", default="results/e3_targeted_x_fault_experiment.json")
    args = parser.parse_args()
    result = run_targeted_fault_suite(shots=args.shots, seed=args.seed, n_processes=args.n_processes, output_path=args.output)
    print(json.dumps({"status": result["status"], "cases": len(result["conditions"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
