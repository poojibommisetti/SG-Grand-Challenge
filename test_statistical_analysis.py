"""Tests for statistical post-processing of saved IPEA experiments."""
from __future__ import annotations

import pytest

from analyze_e0_e1_e2_e3 import analyze_saved_records, load_and_validate_records, wilson_interval


def test_saved_records_have_exact_e1_e3_ideal_distribution_agreement():
    e12, e3, _ = load_and_validate_records()
    a = e12["E1_ideal"]["measured_counts_msb_first"]
    b = e3["ideal"]["measured_counts_msb_first"]
    assert all(a.get(format(i, "05b"), 0) == b.get(format(i, "05b"), 0) for i in range(32))


def test_analysis_reports_all_bins_and_keeps_unresolved_outcomes_explicit():
    report = analyze_saved_records(bootstrap_replicates=100)
    assert report["E1_vs_E3_ideal_distribution_agreement"]["total_variation_distance"] == 0.0
    for name, condition in report["conditions"].items():
        assert len(condition["phase_outcome_distribution_msb_first"]) == 32
        shots = condition["shots"]
        phase_total = sum(row["shots"] for row in condition["phase_outcome_distribution_msb_first"])
        resolved = condition["energy_statistics_conditional_on_resolved_outcomes"]["resolved_shots"]
        unresolved = condition["unresolved_outcomes"]["shots"]
        assert phase_total == shots
        assert resolved + unresolved == shots
        assert condition["run_to_run_variability"] is None
        assert condition["unresolved_outcomes"]["handling"].startswith("retained in the phase distribution")
    assert report["conditions"]["E1_ideal"]["reference_probability"] == pytest.approx(0.9886)
    assert report["conditions"]["E3_logical_noisy"]["unresolved_outcomes"]["shots"] == 420


def test_wilson_intervals_cover_probability_boundaries_and_center():
    lo0, hi0 = wilson_interval(0, 100)
    lo1, hi1 = wilson_interval(100, 100)
    lo50, hi50 = wilson_interval(50, 100)
    assert lo0 == 0.0 and hi0 > 0.0
    assert hi1 == 1.0 and lo1 < 1.0
    assert lo50 < 0.5 < hi50


def test_wilson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson_interval(4, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_repeated_seed_analysis_keeps_each_distribution_and_energy_summary():
    report = analyze_saved_records(
        repeats_path="results/e2_e3_spread_seed_repeats.json",
        bootstrap_replicates=0,
    )
    repeated = report["repeated_run_summary"]
    assert repeated["settings"]["seed_list"] == [131, 104729, 1000003, 8675309, 987654321]
    for condition in repeated["conditions"].values():
        assert condition["independent_runs"] == 5
        assert len(condition["phase_outcome_probability_per_run"]) == 5
        assert all(len(run["probabilities_msb_first"]) == 32 for run in condition["phase_outcome_probability_per_run"])
        assert len(condition["energy_statistics_conditional_on_resolved_outcomes_per_run"]) == 5
        assert all("unresolved_phase_bit_counts" in run for run in condition["energy_statistics_conditional_on_resolved_outcomes_per_run"])
    assert not repeated["conditions"]["E3_logical_noisy"]["identical_full_phase_distributions_across_runs"]
