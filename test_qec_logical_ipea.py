"""Integration tests for sequential five-bit logical IPEA."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ipea import expected_adaptive_distribution
from qec_logical_ipea import run_qec_logical_ipea_5bit


PROJECT = Path(__file__).resolve().parent
E1_RECORD = PROJECT / "results" / "ipea_e1_e2_comparison.json"


def _expected_counts_match(counts: dict[str, int], probabilities: dict[str, float], shots: int) -> bool:
    for outcome, probability in probabilities.items():
        expected = shots * probability
        sigma = np.sqrt(shots * probability * (1.0 - probability))
        if abs(counts.get(outcome, 0) - expected) > 5.0 * sigma + 1.0:
            return False
    return True


def test_sequential_logical_ipea_matches_saved_e1_phase_and_distribution():
    shots = 512
    report = run_qec_logical_ipea_5bit(
        shots=shots, seed=20261003, n_processes=1
    )
    with E1_RECORD.open(encoding="utf-8") as handle:
        e1 = json.load(handle)["E1_ideal"]

    assert report["compiled_modules"] == 1
    assert report["full_five_bit_logical_ipea"]
    assert report["settings"]["peak_live_qubits"] == 7
    assert report["iteration_k_order"] == [4, 3, 2, 1, 0]
    assert report["measurement_order"] == ["b5", "b4", "b3", "b2", "b1"]
    e1_modal_bits = "".join(str(bit) for bit in e1["modal_bits_msb_first"])
    assert report["modal_bits_msb_first"] == e1_modal_bits
    assert report["modal_phase"] == e1["modal_phase"]
    assert report["reference_bits_msb_first"] == [1, 0, 1, 0, 0]
    assert report["validation_passed"]
    assert _expected_counts_match(
        report["measured_counts_descending_k"],
        report["expected_measurement_probabilities_descending_k"],
        shots,
    )


def test_system_ground_state_is_preserved_across_all_five_iterations():
    report = run_qec_logical_ipea_5bit(
        shots=1,
        seed=20261003,
        n_processes=1,
        capture_system_states=True,
    )
    fidelities = report["system_state_fidelities_after_each_iteration"]
    assert list(fidelities) == [
        "system_after_k4",
        "system_after_k3",
        "system_after_k2",
        "system_after_k1",
        "system_after_k0",
    ]
    assert all(value == pytest.approx(1.0, abs=1e-8) for value in fidelities.values())


def test_correctable_boundary_x_fault_is_removed_before_first_controlled_u():
    report = run_qec_logical_ipea_5bit(
        shots=256,
        seed=20261003,
        n_processes=1,
        fault_injection="x_pre_k4_d1",
    )
    syndromes = report["syndrome_counts_by_iteration_and_stage"]
    assert syndromes["k4_pre"] == {"11": 256}
    assert syndromes["k4_post"] == {"00": 256}
    assert report["within_ideal_distribution_tolerance"]
    assert _expected_counts_match(
        report["measured_counts_descending_k"],
        report["expected_measurement_probabilities_descending_k"],
        256,
    )


def test_uncorrected_z_fault_has_zero_syndrome_and_flips_first_measurement():
    report = run_qec_logical_ipea_5bit(
        shots=128,
        seed=20261003,
        n_processes=1,
        fault_injection="z_pre_k4_d1",
    )
    syndromes = report["syndrome_counts_by_iteration_and_stage"]
    assert syndromes["k4_pre"] == {"00": 128}
    first_bit_ones = sum(
        count
        for outcome, count in report["measured_counts_descending_k"].items()
        if outcome.startswith("1")
    )
    assert first_bit_ones > 120
    assert not report["validation_passed"]


def test_phase_correction_formula_and_bit_conversion_match_ipea_helpers():
    report = run_qec_logical_ipea_5bit(
        shots=256, seed=20261003, n_processes=1
    )
    raw = report["modal_measurement_descending_k"]
    expected_msb = "".join(reversed(raw))
    assert report["modal_bits_msb_first"] == expected_msb
    expected = expected_adaptive_distribution(report["exact_ground_phase"], 5)
    assert report["expected_measurement_probabilities_descending_k"] == expected
    assert report["validation_passed"]