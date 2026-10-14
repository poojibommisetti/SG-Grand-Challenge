"""Tests for the classical IPEA reference layer."""
import math

import pytest

import ipea


def test_phase_sign_uses_project_convention():
    energy, time = 0.75, 2.0
    expected = (-energy * time / (2 * math.pi)) % 1.0
    assert ipea.exact_phase_from_energy(energy, time) == pytest.approx(expected)


@pytest.mark.parametrize("energy", [5.0, -5.0])
def test_phase_wraps_negative_and_above_one_raw_values(energy):
    time = 2.0
    raw = -energy * time / (2 * math.pi)
    assert raw < 0.0 or raw > 1.0
    phase = ipea.exact_phase_from_energy(energy, time)
    assert 0.0 <= phase < 1.0
    assert phase == pytest.approx(raw % 1.0)


@pytest.mark.parametrize(
    ("phase", "expected"),
    [(20 / 32, [1, 0, 1, 0, 0]), (13 / 32, [0, 1, 1, 0, 1])],
)
def test_phase_to_bits_extracts_msb_first(phase, expected):
    assert ipea.phase_to_bits(phase, 5) == expected


def test_bits_to_phase_reconstructs_fraction():
    assert ipea.bits_to_phase([1, 0, 1, 0, 0]) == pytest.approx(20 / 32)
    assert ipea.bits_to_phase([0, 1, 1, 0, 1]) == pytest.approx(13 / 32)


@pytest.mark.parametrize("bits", [[1, 0, 1, 0, 0], [0, 1, 1, 0, 1], [1, 1, 1]])
def test_exact_n_bit_phases_round_trip(bits):
    phase = ipea.bits_to_phase(bits)
    recovered = ipea.bits_to_phase(ipea.phase_to_bits(phase, len(bits)))
    assert recovered == pytest.approx(phase, abs=1e-15)


def test_measurement_order_reverses_to_msb_first():
    assert ipea.measured_bits_to_msb_first([0, 0, 1, 0, 1]) == [1, 0, 1, 0, 0]


def test_phase_correction_matches_formula_with_known_lower_bits():
    # m=5, k=2, acquired [b5,b4]=[1,1]: -2*pi*(1/4 + 1/8).
    assert ipea.phase_correction_angle(2, [1, 1], 5) == pytest.approx(
        -3 * math.pi / 4
    )
    assert ipea.phase_correction_angle(4, [], 5) == pytest.approx(0.0)


def test_energy_alias_selects_unique_value_inside_interval():
    # phi=1/4, t=2; alias n=-1 gives 3*pi/4 in (2, 2.5).
    assert ipea.energy_from_phase(0.25, 2.0, (2.0, 2.5)) == pytest.approx(
        3 * math.pi / 4
    )


def test_energy_alias_raises_when_interval_has_no_candidate():
    with pytest.raises(ValueError, match="no phase alias"):
        ipea.energy_from_phase(0.2, 1.0, (1.0, 2.0))


def test_h2_reference_uses_project_hamiltonian_and_time():
    # Temporary validation interval for this unit test only. The project
    # has not defined a production energy interval.
    temporary_test_interval = (-1.2, -1.0)
    report = ipea.h2_reference_report(temporary_test_interval)
    assert report["exact_ground_phase"] == pytest.approx(0.626922, abs=2e-6)
    assert report["phase_bits_msb_first"] == [1, 0, 1, 0, 0]
    assert report["reconstructed_phase"] == pytest.approx(0.625)
    assert report["reconstructed_energy"] == pytest.approx(
        report["ground_energy"], abs=0.02
    )
    assert report["energy_interval"] == temporary_test_interval


def test_expected_one_bit_ipea_distribution():
    expected = ipea.expected_1bit_distribution(0.625)
    assert expected["expected_bit"] == 1
    assert expected["probability_0"] == pytest.approx(0.1464466094067262)
    assert expected["probability_1"] == pytest.approx(0.8535533905932737)


def test_expected_two_bit_distribution_uses_feedback_convention():
    expected = ipea.expected_2bit_distribution(0.625)
    assert sum(expected.values()) == pytest.approx(1.0)
    assert expected["00"] == pytest.approx(0.0732233047033631)
    assert expected["01"] == pytest.approx(0.4267766952966369)
    assert expected["10"] == pytest.approx(0.0732233047033631)
    assert expected["11"] == pytest.approx(0.4267766952966369)


def test_two_bit_measurement_order_reconstructs_phase_prefix():
    measured_b2_b1 = [0, 1]
    msb_bits = ipea.measured_bits_to_msb_first(measured_b2_b1)
    assert msb_bits == [1, 0]
    assert ipea.bits_to_phase(msb_bits) == pytest.approx(0.5)


def test_adaptive_distribution_matches_existing_two_bit_reference():
    generic = ipea.expected_adaptive_distribution(0.625, 2)
    existing = ipea.expected_2bit_distribution(0.625)
    assert generic == pytest.approx(existing)


def test_five_bit_dyadic_adaptive_distribution_is_deterministic():
    distribution = ipea.expected_adaptive_distribution(0.625, 5)
    assert max(distribution, key=distribution.get) == "00101"
    assert distribution["00101"] == pytest.approx(1.0)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_five_bit_modal_phase_resolves_energy_on_explicit_ground_branch():
    exact_phase = 0.6269220080527413
    modal_phase = ipea.bits_to_phase([1, 0, 1, 0, 0])
    evolution_time = 3.4635067188751005
    interval = (-1.2, -1.0)
    exact_ground_energy = -1.1373060511988444
    reconstructed_energy = ipea.energy_from_phase(
        modal_phase, evolution_time, interval
    )
    exact_resolved_energy = ipea.energy_from_phase(
        exact_phase, evolution_time, interval
    )
    assert exact_resolved_energy == pytest.approx(exact_ground_energy, abs=1e-12)
    assert reconstructed_energy == pytest.approx(-1.1338193154314637, abs=1e-12)
