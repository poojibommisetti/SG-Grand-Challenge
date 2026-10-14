"""Classical reference helpers for iterative phase estimation (IPEA).

The project uses U(t) = exp(-i H t), so an energy E corresponds to the
normalized phase frac(-E*t/(2*pi)). This module contains no quantum circuit
or measurement implementation.
"""
import math
from numbers import Integral, Real
from typing import Sequence

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import result
from guppylang.std.lang import py
from guppylang.std.quantum import discard, h, measure, qubit, rz

from state_prep import compute_theta, load_ground_state, prepare_ground_state
from noise_model import DepolarizingNoiseConfig


# Filled by ipea_1bit_validation before the Guppy entrypoint is compiled.
# The theta is embedded as a Guppy compile-time Python value; the controlled
# unitary is imported from the verified pytket construction in time_evolution.
_theta_rad_for_ipea = 0.0
_alpha_2bit_if_b2_one = 0.0
_controlled_u_k1_for_ipea = None
_controlled_u_k0_for_ipea = None
_controlled_u_k4_for_ipea_5bit = None
_controlled_u_k3_for_ipea_5bit = None
_controlled_u_k2_for_ipea_5bit = None
_controlled_u_k1_for_ipea_5bit = None
_controlled_u_k0_for_ipea_5bit = None


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_bits(bits, name="bits"):
    try:
        result = list(bits)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of 0/1 bits") from exc
    if any(not isinstance(bit, Integral) or isinstance(bit, bool) for bit in result):
        raise TypeError(f"{name} must contain only integer 0/1 values")
    if any(int(bit) not in (0, 1) for bit in result):
        raise ValueError(f"{name} must contain only 0/1 values")
    return [int(bit) for bit in result]


def exact_phase_from_energy(E: float, t: float) -> float:
    """Return frac(-E*t/(2*pi)) in the normalized interval [0, 1)."""
    energy = _finite_real(E, "E")
    time = _finite_real(t, "t")
    raw_phase = -energy * time / (2.0 * math.pi)
    return raw_phase % 1.0


def phase_to_bits(phi: float, n_bits: int) -> list[int]:
    """Extract binary fractional digits MSB-first without rounding.

    Repeatedly multiply the remaining fraction by two and take its integer
    part. The tail is truncated after n_bits digits.
    """
    phase = _finite_real(phi, "phi")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phi must be normalized to the interval [0, 1)")
    if isinstance(n_bits, bool) or not isinstance(n_bits, Integral):
        raise TypeError("n_bits must be an integer")
    if n_bits < 0:
        raise ValueError("n_bits must be nonnegative")
    bits = []
    remainder = phase
    for _ in range(int(n_bits)):
        remainder *= 2.0
        bit = int(remainder)
        bits.append(bit)
        remainder -= bit
    return bits


def bits_to_phase(bits: Sequence[int]) -> float:
    """Convert MSB-first fractional bits to sum(b_i / 2**i)."""
    validated = _validated_bits(bits)
    return math.fsum(bit / (2 ** index)
                     for index, bit in enumerate(validated, start=1))


def measured_bits_to_msb_first(measured_bits: Sequence[int]) -> list[int]:
    """Reverse measurements for k=n-1,...,0 into [b1,...,bn].

    Descending controlled-power order yields [bn,...,b1].
    """
    return list(reversed(_validated_bits(measured_bits, "measured_bits")))


def phase_correction_angle(
    k: int, measured_lower_bits: Sequence[int], n_bits: int
) -> float:
    """Return alpha_k in radians using the inspected IPEA convention.

    measured_lower_bits must be in acquisition order [b_m,...,b_(k+2)].
    """
    if isinstance(n_bits, bool) or not isinstance(n_bits, Integral):
        raise TypeError("n_bits must be an integer")
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    if not 0 <= k < n_bits:
        raise ValueError("k must satisfy 0 <= k < n_bits")
    known = _validated_bits(measured_lower_bits, "measured_lower_bits")
    expected_count = int(n_bits) - int(k) - 1
    if len(known) != expected_count:
        raise ValueError(
            f"for k={k} and n_bits={n_bits}, expected {expected_count} "
            "previous bits in descending-k acquisition order"
        )
    terms = (
        bit / (2 ** (j - int(k)))
        for j, bit in enumerate(reversed(known), start=int(k) + 2)
    )
    return -2.0 * math.pi * math.fsum(terms)


def energy_from_phase(
    phi: float, t: float, energy_interval: tuple[float, float]
) -> float:
    """Resolve phase aliases using a required, explicit closed energy interval.

    Candidates are E_n = -(2*pi/t)*(phi+n). Exactly one must lie in the
    interval; no or multiple matches raise ValueError.
    """
    phase = _finite_real(phi, "phi")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phi must be normalized to the interval [0, 1)")
    time = _finite_real(t, "t")
    if time <= 0.0:
        raise ValueError("t must be positive")
    try:
        if len(energy_interval) != 2:
            raise ValueError("energy_interval must contain (minimum, maximum)")
        lower = _finite_real(energy_interval[0], "energy_interval minimum")
        upper = _finite_real(energy_interval[1], "energy_interval maximum")
    except TypeError as exc:
        raise TypeError("energy_interval must be a pair of finite real numbers") from exc
    if lower > upper:
        raise ValueError("energy_interval minimum must not exceed its maximum")

    period = 2.0 * math.pi / time
    n_min_real = -upper / period - phase
    n_max_real = -lower / period - phase
    eps = 1e-12
    n_min = math.ceil(n_min_real - eps)
    n_max = math.floor(n_max_real + eps)
    candidates = []
    for alias in range(n_min, n_max + 1):
        energy = -period * (phase + alias)
        tolerance = 1e-12 * max(1.0, abs(lower), abs(upper), abs(energy))
        if lower - tolerance <= energy <= upper + tolerance:
            candidates.append((alias, energy))
    if not candidates:
        raise ValueError(
            f"no phase alias lies inside energy_interval=({lower}, {upper})"
        )
    if len(candidates) > 1:
        aliases = [alias for alias, _ in candidates]
        raise ValueError(
            f"energy_interval contains multiple phase aliases {aliases}; "
            "provide an interval that identifies one energy"
        )
    return float(candidates[0][1])


def h2_reference_report(
    energy_interval: tuple[float, float], n_bits: int = 5
) -> dict:
    """Return H2 reference values using time_evolution.py's existing APIs.

    energy_interval is required because phase alone cannot resolve energy
    aliases. No answer-key energy file is consulted.
    """
    from time_evolution import choose_evolution_time, diagonalize, load_hamiltonian

    hamiltonian, _ = load_hamiltonian()
    eigenvalues, _ = diagonalize(hamiltonian)
    time = choose_evolution_time(eigenvalues)
    ground_phase = exact_phase_from_energy(float(eigenvalues[0]), time)
    bits = phase_to_bits(ground_phase, n_bits)
    reconstructed_phase = bits_to_phase(bits)
    reconstructed_energy = energy_from_phase(
        reconstructed_phase, time, energy_interval
    )
    return {
        "eigenvalues": [float(value) for value in eigenvalues],
        "evolution_time": float(time),
        "energy_interval": tuple(float(value) for value in energy_interval),
        "ground_energy": float(eigenvalues[0]),
        "exact_ground_phase": ground_phase,
        "phase_bits_msb_first": bits,
        "reconstructed_phase": reconstructed_phase,
        "reconstructed_energy": reconstructed_energy,
    }


def expected_1bit_distribution(phi: float) -> dict:
    """Return the ideal ancilla distribution for one-bit iterative PE.

    After controlled phase 2*pi*phi and the final Hadamard,
    P(0)=cos(pi*phi)^2 and P(1)=sin(pi*phi)^2. The decoded one-bit
    estimate is the most-significant fractional bit, floor(2*phi).
    """
    phase = _finite_real(phi, "phi")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phi must be normalized to the interval [0, 1)")
    p0 = math.cos(math.pi * phase) ** 2
    p1 = math.sin(math.pi * phase) ** 2
    return {
        "expected_bit": int(phase >= 0.5),
        "probability_0": p0,
        "probability_1": p1,
    }


@guppy
def ipea_1bit_emulator_demo() -> None:
    """One-bit IPEA: H ancilla, exact controlled-U(t), H, measure ancilla.

    The only correction is alpha=0, so no rotation is inserted between the
    controlled unitary and the final Hadamard. The system register is
    discarded after the ancilla measurement and is never measured.
    """
    ancilla = qubit()
    q0 = qubit()
    q1 = qubit()
    h(ancilla)
    prepare_ground_state(q0, q1, py(_theta_rad_for_ipea))
    _controlled_u_k0_for_ipea(ancilla, q0, q1)
    h(ancilla)
    measured_bit = measure(ancilla).read()
    result("ipea_ancilla", measured_bit)
    discard(q0)
    discard(q1)


def ipea_1bit_validation(shots: int = 1000, seed: int = 20261001) -> dict:
    """Compile and run one-bit H2 ground-state IPEA on Guppy/Selene.

    Reuses state_prep.py to load/prepare the reference ground state and
    time_evolution.py to synthesize the exact ancilla-controlled U(t).
    The function measures only the ancilla and compares observed shots with
    the exact ideal one-bit distribution using a five-standard-error bound.
    """
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")

    from time_evolution import (
        choose_evolution_time,
        controlled_u_pytket_circuit,
        diagonalize,
        load_hamiltonian,
    )

    global _theta_rad_for_ipea, _controlled_u_k0_for_ipea

    a, b, _ = load_ground_state()
    _theta_rad_for_ipea = compute_theta(a, b)

    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)
    circuit, _ = controlled_u_pytket_circuit(
        eigenvalues, eigenvectors, evolution_time, k=0
    )
    _controlled_u_k0_for_ipea = guppy.load_pytket(
        "ipea_exact_controlled_u_k0", circuit, use_arrays=False
    )

    compiled = ipea_1bit_emulator_demo.compile()
    emulator_result = (
        ipea_1bit_emulator_demo.emulator(n_qubits=3)
        .with_shots(int(shots))
        .with_seed(int(seed))
        .run()
    )
    register_counts = emulator_result.register_counts().get("ipea_ancilla", {})
    count_0 = int(register_counts.get("0", 0))
    count_1 = int(register_counts.get("1", 0))
    total = count_0 + count_1
    if total != shots:
        raise AssertionError(
            f"expected {shots} ancilla results, received {total}"
        )

    ground_energy = float(eigenvalues[0])
    exact_phase = exact_phase_from_energy(ground_energy, evolution_time)
    expected = expected_1bit_distribution(exact_phase)
    observed_p0 = count_0 / total
    observed_p1 = count_1 / total
    expected_p1 = expected["probability_1"]
    standard_error = math.sqrt(expected_p1 * (1.0 - expected_p1) / shots)
    within_statistical_tolerance = (
        abs(observed_p1 - expected_p1) <= 5.0 * standard_error + 1.0 / shots
    )
    majority_bit = int(count_1 > count_0)

    return {
        "compiled_modules": len(compiled.modules),
        "shots": int(shots),
        "seed": int(seed),
        "k": 0,
        "correction_angle_radians": 0.0,
        "ground_energy": ground_energy,
        "evolution_time": float(evolution_time),
        "exact_ground_phase": exact_phase,
        "expected_bit": expected["expected_bit"],
        "expected_probabilities": {
            "0": expected["probability_0"],
            "1": expected["probability_1"],
        },
        "ancilla_counts": {"0": count_0, "1": count_1},
        "observed_probabilities": {"0": observed_p0, "1": observed_p1},
        "majority_bit": majority_bit,
        "within_statistical_tolerance": within_statistical_tolerance,
        "pytket_gate_count": circuit.n_gates,
        "pytket_gate_types": sorted(
            {str(command.op.type) for command in circuit.get_commands()}
        ),
        "validation_passed": (
            majority_bit == expected["expected_bit"]
            and within_statistical_tolerance
        ),
    }


def expected_2bit_distribution(phi: float) -> dict[str, float]:
    """Ideal joint probabilities in measurement order [b2, b1].

    The first iteration measures k=1. The second measures k=0 after
    conditionally applying the correction returned by
    phase_correction_angle(0, [b2], 2).
    """
    phase = _finite_real(phi, "phi")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phi must be normalized to the interval [0, 1)")

    first_phase = (2.0 * phase) % 1.0
    first_p1 = math.sin(math.pi * first_phase) ** 2
    first_probs = {0: 1.0 - first_p1, 1: first_p1}
    joint = {}
    for b2 in (0, 1):
        correction = phase_correction_angle(0, [b2], n_bits=2)
        corrected_phase = (phase + correction / (2.0 * math.pi)) % 1.0
        second_p1 = math.sin(math.pi * corrected_phase) ** 2
        second_probs = {0: 1.0 - second_p1, 1: second_p1}
        for b1 in (0, 1):
            joint[f"{b2}{b1}"] = first_probs[b2] * second_probs[b1]
    return joint


@guppy
def ipea_2bit_emulator_demo() -> None:
    """Adaptive two-bit IPEA, measuring first k=1 (b2), then k=0 (b1)."""
    q0 = qubit()
    q1 = qubit()
    prepare_ground_state(q0, q1, py(_theta_rad_for_ipea))

    # k=1 produces the lower-significance bit b2 first.
    ancilla_b2 = qubit()
    h(ancilla_b2)
    _controlled_u_k1_for_ipea(ancilla_b2, q0, q1)
    h(ancilla_b2)
    b2 = measure(ancilla_b2).read()
    result("ipea_b2", b2)

    # Measurement consumes its ancilla, so the next iteration allocates one.
    # The helper computes alpha_0 for b2=1; b2=0 means the correction is zero.
    ancilla_b1 = qubit()
    h(ancilla_b1)
    _controlled_u_k0_for_ipea(ancilla_b1, q0, q1)
    if b2:
        rz(ancilla_b1, angle(py(_alpha_2bit_if_b2_one / math.pi)))
    h(ancilla_b1)
    b1 = measure(ancilla_b1).read()
    result("ipea_b1", b1)

    # The system register is not measured.
    discard(q0)
    discard(q1)


def ipea_2bit_validation(shots: int = 2000, seed: int = 20261002) -> dict:
    """Run adaptive two-bit H2 IPEA in Selene and validate its distribution."""
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")

    from collections import Counter

    from time_evolution import (
        choose_evolution_time,
        controlled_u_pytket_circuit,
        diagonalize,
        load_hamiltonian,
    )

    global _theta_rad_for_ipea
    global _alpha_2bit_if_b2_one
    global _controlled_u_k1_for_ipea, _controlled_u_k0_for_ipea

    a, b, _ = load_ground_state()
    _theta_rad_for_ipea = compute_theta(a, b)
    _alpha_2bit_if_b2_one = phase_correction_angle(0, [1], n_bits=2)

    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)
    circuit_k1, _ = controlled_u_pytket_circuit(
        eigenvalues, eigenvectors, evolution_time, k=1
    )
    circuit_k0, _ = controlled_u_pytket_circuit(
        eigenvalues, eigenvectors, evolution_time, k=0
    )
    _controlled_u_k1_for_ipea = guppy.load_pytket(
        "ipea_exact_controlled_u_k1", circuit_k1, use_arrays=False
    )
    _controlled_u_k0_for_ipea = guppy.load_pytket(
        "ipea_exact_controlled_u_k0_adaptive", circuit_k0, use_arrays=False
    )

    compiled = ipea_2bit_emulator_demo.compile()
    emulator_result = (
        ipea_2bit_emulator_demo.emulator(n_qubits=5)
        .with_shots(int(shots))
        .with_seed(int(seed))
        .run()
    )

    measurement_order_counts = Counter()
    msb_bit_counts = Counter()
    phase_counts = Counter()
    for shot in emulator_result.results:
        values = dict(shot.entries)
        b2 = int(values["ipea_b2"])
        b1 = int(values["ipea_b1"])
        measurement_order_counts[f"{b2}{b1}"] += 1
        msb_bits = measured_bits_to_msb_first([b2, b1])
        msb_key = "".join(str(bit) for bit in msb_bits)
        msb_bit_counts[msb_key] += 1
        phase_counts[bits_to_phase(msb_bits)] += 1

    if sum(measurement_order_counts.values()) != shots:
        raise AssertionError(
            f"expected {shots} two-bit results, received "
            f"{sum(measurement_order_counts.values())}"
        )

    exact_phase = exact_phase_from_energy(
        float(eigenvalues[0]), evolution_time
    )
    exact_bit_prefix = phase_to_bits(exact_phase, 2)
    truncated_reference_phase = bits_to_phase(exact_bit_prefix)
    expected_distribution = expected_2bit_distribution(exact_phase)
    within_statistical_tolerance = True
    for outcome, probability in expected_distribution.items():
        observed = measurement_order_counts.get(outcome, 0)
        expected_count = shots * probability
        sigma = math.sqrt(shots * probability * (1.0 - probability))
        if abs(observed - expected_count) > 5.0 * sigma + 1.0:
            within_statistical_tolerance = False

    modal_measurement = max(
        measurement_order_counts,
        key=lambda outcome: measurement_order_counts[outcome],
    )
    modal_msb_bits = measured_bits_to_msb_first(
        [int(bit) for bit in modal_measurement]
    )
    modal_phase = bits_to_phase(modal_msb_bits)
    phase_resolution = 1.0 / (2 ** 2)
    modal_phase_within_resolution = (
        abs(modal_phase - exact_phase) < phase_resolution
    )

    return {
        "compiled_modules": len(compiled.modules),
        "shots": int(shots),
        "seed": int(seed),
        "iteration_k_order": [1, 0],
        "measurement_order": ["b2", "b1"],
        "correction_angle_if_b2_is_1_radians": _alpha_2bit_if_b2_one,
        "exact_ground_phase": exact_phase,
        "exact_two_bit_prefix_msb_first": exact_bit_prefix,
        "truncated_reference_phase": truncated_reference_phase,
        "expected_measurement_probabilities_b2b1": expected_distribution,
        "measured_counts_b2b1": {
            outcome: int(measurement_order_counts.get(outcome, 0))
            for outcome in ("00", "01", "10", "11")
        },
        "measured_counts_msb_first": {
            outcome: int(msb_bit_counts.get(outcome, 0))
            for outcome in ("00", "01", "10", "11")
        },
        "reconstructed_phase_counts": {
            f"{phase:.2f}": int(count) for phase, count in sorted(phase_counts.items())
        },
        "modal_measurement_b2b1": modal_measurement,
        "modal_phase_bits_msb_first": modal_msb_bits,
        "modal_reconstructed_phase": modal_phase,
        "modal_phase_absolute_error": abs(modal_phase - exact_phase),
        "phase_resolution": phase_resolution,
        "within_statistical_tolerance": within_statistical_tolerance,
        "modal_phase_within_resolution": modal_phase_within_resolution,
        "pytket_gate_counts_k1_k0": [circuit_k1.n_gates, circuit_k0.n_gates],
        "validation_passed": (
            within_statistical_tolerance and modal_phase_within_resolution
        ),
    }


def expected_adaptive_distribution(phi: float, n_bits: int) -> dict[str, float]:
    """Return ideal IPEA probabilities keyed by descending-k outcomes.

    Keys follow measurement order [b_n,...,b_1]. Each round uses the
    existing phase_correction_angle helper to compute its conditional phase.
    """
    phase = _finite_real(phi, "phi")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phi must be normalized to the interval [0, 1)")
    if isinstance(n_bits, bool) or not isinstance(n_bits, Integral) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")

    paths = {"": 1.0}
    for k in range(int(n_bits) - 1, -1, -1):
        next_paths = {}
        for measured_prefix, prefix_probability in paths.items():
            previous_bits = [int(bit) for bit in measured_prefix]
            correction = phase_correction_angle(k, previous_bits, int(n_bits))
            effective_phase = (
                (2 ** k) * phase + correction / (2.0 * math.pi)
            ) % 1.0
            probability_1 = math.sin(math.pi * effective_phase) ** 2
            next_paths[measured_prefix + "0"] = (
                prefix_probability * (1.0 - probability_1)
            )
            next_paths[measured_prefix + "1"] = (
                prefix_probability * probability_1
            )
        paths = next_paths
    return paths


@guppy
def ipea_5bit_emulator_demo() -> None:
    """Five-bit adaptive IPEA, processing controlled powers k=4 down to 0."""
    q0 = qubit()
    q1 = qubit()
    prepare_ground_state(q0, q1, py(_theta_rad_for_ipea))

    ancilla4 = qubit()
    h(ancilla4)
    _controlled_u_k4_for_ipea_5bit(ancilla4, q0, q1)
    h(ancilla4)
    b5 = measure(ancilla4).read()
    result("b5", b5)

    ancilla3 = qubit()
    h(ancilla3)
    _controlled_u_k3_for_ipea_5bit(ancilla3, q0, q1)
    if b5:
        rz(ancilla3, angle(py(phase_correction_angle(3, [1], 5) / math.pi)))
    h(ancilla3)
    b4 = measure(ancilla3).read()
    result("b4", b4)

    ancilla2 = qubit()
    h(ancilla2)
    _controlled_u_k2_for_ipea_5bit(ancilla2, q0, q1)
    if b5:
        rz(ancilla2, angle(py(phase_correction_angle(2, [1, 0], 5) / math.pi)))
    if b4:
        rz(ancilla2, angle(py(phase_correction_angle(2, [0, 1], 5) / math.pi)))
    h(ancilla2)
    b3 = measure(ancilla2).read()
    result("b3", b3)

    ancilla1 = qubit()
    h(ancilla1)
    _controlled_u_k1_for_ipea_5bit(ancilla1, q0, q1)
    if b5:
        rz(ancilla1, angle(py(phase_correction_angle(1, [1, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla1, angle(py(phase_correction_angle(1, [0, 1, 0], 5) / math.pi)))
    if b3:
        rz(ancilla1, angle(py(phase_correction_angle(1, [0, 0, 1], 5) / math.pi)))
    h(ancilla1)
    b2 = measure(ancilla1).read()
    result("b2", b2)

    ancilla0 = qubit()
    h(ancilla0)
    _controlled_u_k0_for_ipea_5bit(ancilla0, q0, q1)
    if b5:
        rz(ancilla0, angle(py(phase_correction_angle(0, [1, 0, 0, 0], 5) / math.pi)))
    if b4:
        rz(ancilla0, angle(py(phase_correction_angle(0, [0, 1, 0, 0], 5) / math.pi)))
    if b3:
        rz(ancilla0, angle(py(phase_correction_angle(0, [0, 0, 1, 0], 5) / math.pi)))
    if b2:
        rz(ancilla0, angle(py(phase_correction_angle(0, [0, 0, 0, 1], 5) / math.pi)))
    h(ancilla0)
    b1 = measure(ancilla0).read()
    result("b1", b1)

    # The system register is never measured.
    discard(q0)
    discard(q1)


def ipea_5bit_validation(
    shots: int = 5000,
    seed: int = 20261003,
    noise_config: DepolarizingNoiseConfig | None = None,
    n_processes: int = 1,
) -> dict:
    """Run five-bit H2 IPEA; optionally attach an explicit Selene noise config."""
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if (
        isinstance(n_processes, bool)
        or not isinstance(n_processes, Integral)
        or n_processes < 1
    ):
        raise ValueError("n_processes must be a positive integer")

    from collections import Counter

    from time_evolution import (
        choose_evolution_time,
        controlled_u_pytket_circuit,
        diagonalize,
        load_hamiltonian,
    )

    global _theta_rad_for_ipea
    global _controlled_u_k4_for_ipea_5bit, _controlled_u_k3_for_ipea_5bit
    global _controlled_u_k2_for_ipea_5bit, _controlled_u_k1_for_ipea_5bit
    global _controlled_u_k0_for_ipea_5bit

    a, b, _ = load_ground_state()
    _theta_rad_for_ipea = compute_theta(a, b)
    hamiltonian, _ = load_hamiltonian()
    eigenvalues, eigenvectors = diagonalize(hamiltonian)
    evolution_time = choose_evolution_time(eigenvalues)

    circuits = {}
    loaded_gates = {}
    for k in range(4, -1, -1):
        circuit, _ = controlled_u_pytket_circuit(
            eigenvalues, eigenvectors, evolution_time, k=k
        )
        circuits[k] = circuit
        loaded_gates[k] = guppy.load_pytket(
            f"ipea_exact_controlled_u_k{k}_five_bit", circuit, use_arrays=False
        )
    _controlled_u_k4_for_ipea_5bit = loaded_gates[4]
    _controlled_u_k3_for_ipea_5bit = loaded_gates[3]
    _controlled_u_k2_for_ipea_5bit = loaded_gates[2]
    _controlled_u_k1_for_ipea_5bit = loaded_gates[1]
    _controlled_u_k0_for_ipea_5bit = loaded_gates[0]

    compiled = ipea_5bit_emulator_demo.compile()
    emulator_instance = (
        ipea_5bit_emulator_demo.emulator(n_qubits=12)
        .with_shots(int(shots))
        .with_seed(int(seed))
        .with_n_processes(int(n_processes))
    )
    noise_model_metadata = None
    if noise_config is not None:
        from noise_model import (
            attach_error_model,
            make_run_metadata,
            make_selene_error_model,
        )

        selene_error_model = make_selene_error_model(
            noise_config, random_seed=int(seed)
        )
        emulator_instance = attach_error_model(
            emulator_instance, selene_error_model
        )
        noise_model_metadata = make_run_metadata(
            noise_config,
            random_seed=int(seed),
            shots=int(shots),
            repeats=1,
            n_qubits=12,
        )
    emulator_result = emulator_instance.run()

    measured_order_counts = Counter()
    msb_counts = Counter()
    phase_counts = Counter()
    for shot in emulator_result.results:
        values = dict(shot.entries)
        measured_bits = [int(values[tag]) for tag in ("b5", "b4", "b3", "b2", "b1")]
        raw_key = "".join(str(bit) for bit in measured_bits)
        measured_order_counts[raw_key] += 1
        msb_bits = measured_bits_to_msb_first(measured_bits)
        msb_key = "".join(str(bit) for bit in msb_bits)
        msb_counts[msb_key] += 1
        phase_counts[bits_to_phase(msb_bits)] += 1

    if sum(measured_order_counts.values()) != shots:
        raise AssertionError(
            f"expected {shots} five-bit results, received "
            f"{sum(measured_order_counts.values())}"
        )

    exact_phase = exact_phase_from_energy(float(eigenvalues[0]), evolution_time)
    reference_bits = phase_to_bits(exact_phase, 5)
    reference_phase = bits_to_phase(reference_bits)
    expected_distribution = expected_adaptive_distribution(exact_phase, 5)
    within_statistical_tolerance = True
    for outcome, probability in expected_distribution.items():
        observed = measured_order_counts.get(outcome, 0)
        expected_count = shots * probability
        sigma = math.sqrt(shots * probability * (1.0 - probability))
        if abs(observed - expected_count) > 5.0 * sigma + 1.0:
            within_statistical_tolerance = False

    modal_measured_bits = max(
        measured_order_counts,
        key=lambda outcome: measured_order_counts[outcome],
    )
    modal_msb_bits = measured_bits_to_msb_first(
        [int(bit) for bit in modal_measured_bits]
    )
    modal_phase = bits_to_phase(modal_msb_bits)
    phase_resolution = 1.0 / 32.0
    modal_phase_within_resolution = abs(modal_phase - exact_phase) < phase_resolution

    return {
        "compiled_modules": len(compiled.modules),
        "shots": int(shots),
        "seed": int(seed),
        "n_processes": int(n_processes),
        "noise_model_metadata": noise_model_metadata,
        "iteration_k_order": [4, 3, 2, 1, 0],
        "measurement_order": ["b5", "b4", "b3", "b2", "b1"],
        "exact_ground_energy": float(eigenvalues[0]),
        "evolution_time": float(evolution_time),
        "exact_ground_phase": exact_phase,
        "reference_bits_msb_first": reference_bits,
        "reference_phase": reference_phase,
        "expected_measurement_probabilities": expected_distribution,
        "measured_counts_descending_k": {
            outcome: int(measured_order_counts.get(outcome, 0))
            for outcome in sorted(expected_distribution)
        },
        "measured_counts_msb_first": {
            outcome: int(msb_counts.get(outcome, 0))
            for outcome in sorted(expected_distribution)
        },
        "reconstructed_phase_counts": {
            f"{phase:.5f}": int(count) for phase, count in sorted(phase_counts.items())
        },
        "modal_measurement_descending_k": modal_measured_bits,
        "modal_bits_msb_first": modal_msb_bits,
        "modal_reconstructed_phase": modal_phase,
        "modal_phase_absolute_error": abs(modal_phase - exact_phase),
        "phase_resolution": phase_resolution,
        "within_statistical_tolerance": within_statistical_tolerance,
        "modal_phase_within_resolution": modal_phase_within_resolution,
        "controlled_u_gate_counts_k4_to_k0": [circuits[k].n_gates for k in range(4, -1, -1)],
        "validation_passed": (
            within_statistical_tolerance and modal_phase_within_resolution
        ),
    }


def ipea_5bit_energy_validation(
    energy_interval: tuple[float, float],
    shots: int = 1000,
    seed: int = 20261005,
) -> dict:
    """Run 5-bit IPEA and reconstruct shot energies using an explicit interval.

    Every shot's measured MSB-first phase is passed to energy_from_phase.
    Outcomes with no alias in the supplied interval are counted as unresolved;
    they are not forced to the ground-state branch.
    """
    from collections import Counter

    from time_evolution import (
        diagonalize,
        load_hamiltonian,
    )

    result = ipea_5bit_validation(shots=shots, seed=seed)
    evolution_time = result["evolution_time"]
    exact_ground_energy = result["exact_ground_energy"]
    exact_phase = result["exact_ground_phase"]
    lower, upper = energy_interval

    hamiltonian, _ = load_hamiltonian()
    eigenvalues, _ = diagonalize(hamiltonian)
    eigenvalues_inside_interval = [
        float(value) for value in eigenvalues if lower <= value <= upper
    ]
    if len(eigenvalues_inside_interval) != 1 or not math.isclose(
        eigenvalues_inside_interval[0], exact_ground_energy, abs_tol=1e-10
    ):
        raise ValueError(
            "energy_interval must contain exactly the H2 ground-state eigenvalue"
        )

    exact_phase_energy = energy_from_phase(
        exact_phase, evolution_time, energy_interval
    )
    exact_alias_index = round(
        -exact_ground_energy * evolution_time / (2.0 * math.pi) - exact_phase
    )
    exact_alias_correct = math.isclose(
        exact_phase_energy, exact_ground_energy, rel_tol=0.0, abs_tol=1e-10
    )

    energy_counts = Counter()
    unresolved_phase_counts = Counter()
    selected_alias_counts = Counter()
    phase_outcomes = result["measured_counts_msb_first"]
    for bitstring, count in phase_outcomes.items():
        phase = bits_to_phase([int(bit) for bit in bitstring])
        for _ in range(count):
            try:
                energy = energy_from_phase(phase, evolution_time, energy_interval)
            except ValueError as exc:
                if "no phase alias" not in str(exc):
                    raise
                unresolved_phase_counts[f"{phase:.5f}"] += 1
                continue
            energy_counts[f"{energy:.8f}"] += 1
            alias = round(
                -energy * evolution_time / (2.0 * math.pi) - phase
            )
            selected_alias_counts[str(alias)] += 1

    modal_bits = result["modal_bits_msb_first"]
    modal_phase = result["modal_reconstructed_phase"]
    modal_energy = energy_from_phase(
        modal_phase, evolution_time, energy_interval
    )
    modal_alias_index = round(
        -modal_energy * evolution_time / (2.0 * math.pi) - modal_phase
    )
    absolute_energy_error = abs(modal_energy - exact_ground_energy)
    energy_resolution = 2.0 * math.pi / (evolution_time * 32.0)
    alias_branch_correct = (
        exact_alias_correct and modal_alias_index == exact_alias_index
    )
    energy_error_within_resolution = absolute_energy_error < energy_resolution

    result.update(
        {
            "energy_interval": tuple(float(value) for value in energy_interval),
            "exact_phase_resolved_energy": exact_phase_energy,
            "exact_phase_alias_index": exact_alias_index,
            "alias_branch_correct": alias_branch_correct,
            "modal_bitstring_msb_first": modal_bits,
            "modal_phase": modal_phase,
            "modal_reconstructed_energy": modal_energy,
            "modal_energy_alias_index": modal_alias_index,
            "absolute_energy_error": absolute_energy_error,
            "energy_resolution_hartree": energy_resolution,
            "energy_distribution_hartree": dict(sorted(energy_counts.items())),
            "unresolved_phase_counts_outside_interval": dict(
                sorted(unresolved_phase_counts.items())
            ),
            "resolved_alias_index_counts": dict(sorted(selected_alias_counts.items())),
            "resolved_shots": sum(energy_counts.values()),
            "unresolved_shots": sum(unresolved_phase_counts.values()),
            "energy_error_within_resolution": energy_error_within_resolution,
            "energy_validation_passed": (
                result["validation_passed"]
                and alias_branch_correct
                and energy_error_within_resolution
            ),
        }
    )
    return result
