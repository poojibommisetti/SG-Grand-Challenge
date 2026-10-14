"""Statistical analysis for saved H2 IPEA records; never edits source records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from ipea import bits_to_phase, energy_from_phase

BITSTRINGS = tuple(format(i, "05b") for i in range(32))
REFERENCE_BITS = "10100"
WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> list[float]:
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("successes and trials must be integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    p = successes / trials
    z2 = z * z
    if successes == 0:
        return [0.0, z2 / (trials + z2)]
    if successes == trials:
        return [trials / (trials + z2), 1.0]
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _five_bit_counts(record: dict, shots: int) -> dict[str, int]:
    raw = record["measured_counts_msb_first"]
    counts = {bits: int(raw.get(bits, 0)) for bits in BITSTRINGS}
    if any(count < 0 for count in counts.values()):
        raise ValueError("phase counts cannot be negative")
    if sum(counts.values()) != shots:
        raise ValueError(f"phase counts sum to {sum(counts.values())}, expected {shots}")
    return counts


def _weighted_quantile(values: list[float], weights: list[int], q: float) -> float:
    total = sum(weights)
    if total < 1:
        raise ValueError("cannot take a quantile of an empty resolved sample")
    ordered = sorted(zip(values, weights))
    target = q * (total - 1)
    lower_index = math.floor(target)
    upper_index = math.ceil(target)
    cumulative = 0
    lower_value = upper_value = None
    for value, weight in ordered:
        previous = cumulative
        cumulative += weight
        if lower_value is None and lower_index < cumulative:
            lower_value = value
        if upper_index < cumulative:
            upper_value = value
            break
    assert lower_value is not None and upper_value is not None
    if lower_index == upper_index:
        return float(lower_value)
    fraction = target - lower_index
    return float(lower_value + fraction * (upper_value - lower_value))


def analyze_condition(
    name: str,
    record: dict,
    *,
    e0_hartree: float,
    evolution_time: float,
    energy_interval: tuple[float, float],
    bootstrap_replicates: int = 4000,
    bootstrap_seed: int = 88421,
) -> dict:
    settings = record.get("settings") or {}
    shots = int(record.get("shots", settings.get("shots", 0)))
    if shots < 1:
        raise ValueError(f"{name}: missing positive shot count")
    counts = _five_bit_counts(record, shots)
    phases = {bits: bits_to_phase([int(bit) for bit in bits]) for bits in BITSTRINGS}
    energy_by_bits: dict[str, float | None] = {}
    unresolved_counts: dict[str, int] = {}
    resolved_counts: dict[str, int] = {}
    for bits in BITSTRINGS:
        try:
            energy_by_bits[bits] = energy_from_phase(
                phases[bits], evolution_time, energy_interval
            )
        except ValueError as exc:
            if "no phase alias" not in str(exc):
                raise
            energy_by_bits[bits] = None
        if energy_by_bits[bits] is None:
            unresolved_counts[bits] = counts[bits]
        elif counts[bits]:
            resolved_counts[bits] = counts[bits]

    unresolved_n = sum(unresolved_counts.values())
    resolved_n = shots - unresolved_n
    if resolved_n < 1:
        raise ValueError(f"{name}: no phase outcomes resolved by the requested energy interval")

    phase_distribution = []
    for bits in BITSTRINGS:
        n = counts[bits]
        phase_distribution.append({
            "bits_msb_first": bits,
            "shots": n,
            "probability": n / shots,
            "probability_ci95_wilson": wilson_interval(n, shots),
            "phase": phases[bits],
            "energy_hartree": energy_by_bits[bits],
            "energy_mapping": "resolved" if energy_by_bits[bits] is not None else "unresolved_no_alias_in_interval",
        })

    modal_bits = max(BITSTRINGS, key=lambda bits: counts[bits])
    modal_n = counts[modal_bits]
    ref_n = counts[REFERENCE_BITS]
    resolved_energy_values = [float(energy_by_bits[b]) for b in resolved_counts]
    resolved_energy_weights = [resolved_counts[b] for b in resolved_counts]
    mean_energy = sum(e * w for e, w in zip(resolved_energy_values, resolved_energy_weights)) / resolved_n
    variance = sum(w * (e - mean_energy) ** 2 for e, w in zip(resolved_energy_values, resolved_energy_weights)) / max(1, resolved_n - 1)
    energy_errors = [e - e0_hartree for e in resolved_energy_values]
    mean_error = sum(err * w for err, w in zip(energy_errors, resolved_energy_weights)) / resolved_n
    mae = sum(abs(err) * w for err, w in zip(energy_errors, resolved_energy_weights)) / resolved_n
    rmse = math.sqrt(sum(err * err * w for err, w in zip(energy_errors, resolved_energy_weights)) / resolved_n)

    bootstrap_ci = None
    if bootstrap_replicates > 0 and resolved_n > 1:
        rng = np.random.default_rng(bootstrap_seed)
        probs = np.asarray(resolved_energy_weights, dtype=float) / resolved_n
        energy_array = np.asarray(resolved_energy_values, dtype=float)
        draws = rng.multinomial(resolved_n, probs, size=bootstrap_replicates)
        means = draws @ energy_array / resolved_n
        rmse_draws = np.sqrt(draws @ ((energy_array - e0_hartree) ** 2) / resolved_n)
        bootstrap_ci = {
            "method": "conditional multinomial bootstrap over resolved shots within this saved run",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "mean_energy_hartree_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "resolved_energy_rmse_hartree_ci95": [float(x) for x in np.quantile(rmse_draws, [0.025, 0.975])],
        }

    noise = record.get("noise_model_metadata")
    if noise is None:
        noise = record.get("noise_model_metadata")
    return {
        "condition": name,
        "source_label": record.get("label"),
        "shots": shots,
        "seed": record.get("seed", settings.get("seed")),
        "repeats_recorded": record.get("repeats", settings.get("repeats", 1)),
        "energy_interval_hartree": list(energy_interval),
        "energy_mapping": "ipea.energy_from_phase on bits_to_phase(bits) with the explicit saved interval",
        "noise_model": noise,
        "modal_phase_bits_msb_first": modal_bits,
        "modal_count": modal_n,
        "modal_probability": modal_n / shots,
        "modal_probability_ci95_wilson": wilson_interval(modal_n, shots),
        "reference_bits_msb_first": REFERENCE_BITS,
        "reference_count": ref_n,
        "reference_probability": ref_n / shots,
        "reference_probability_ci95_wilson": wilson_interval(ref_n, shots),
        "exact_phase": record.get("exact_ground_phase"),
        "modal_phase": phases[modal_bits],
        "modal_energy_hartree": energy_by_bits[modal_bits],
        "modal_energy_absolute_error_vs_e0_hartree": (
            None if energy_by_bits[modal_bits] is None else abs(float(energy_by_bits[modal_bits]) - e0_hartree)
        ),
        "phase_outcome_distribution_msb_first": phase_distribution,
        "energy_statistics_conditional_on_resolved_outcomes": {
            "resolved_shots": resolved_n,
            "mean_energy_hartree": mean_energy,
            "median_energy_hartree": _weighted_quantile(resolved_energy_values, resolved_energy_weights, 0.5),
            "sample_std_energy_hartree": math.sqrt(variance),
            "mean_error_vs_e0_hartree": mean_error,
            "mean_absolute_error_vs_e0_hartree": mae,
            "rmse_vs_e0_hartree": rmse,
            "energy_p025_hartree": _weighted_quantile(resolved_energy_values, resolved_energy_weights, 0.025),
            "energy_p975_hartree": _weighted_quantile(resolved_energy_values, resolved_energy_weights, 0.975),
            "within_run_bootstrap_ci95": bootstrap_ci,
        },
        "unresolved_outcomes": {
            "shots": unresolved_n,
            "probability": unresolved_n / shots,
            "probability_ci95_wilson": wilson_interval(unresolved_n, shots),
            "phase_bit_counts": unresolved_counts,
            "handling": "retained in the phase distribution and counted as unresolved; excluded only from explicitly conditional resolved-energy summaries",
        },
        "run_to_run_variability": None,
        "run_variability_note": "Only one original run is available for this condition; shot-level intervals above do not estimate emulator run-to-run variability.",
    }


def load_and_validate_records(
    e12_path: str | Path = "results/ipea_e1_e2_comparison.json",
    e3_ideal_path: str | Path = "results/ipea_e3_logical_ideal.json",
    e3_noisy_path: str | Path = "results/ipea_e3_logical_noisy.json",
    comparison_path: str | Path = "results/e0_e1_e2_e3_comparison.json",
    e0_path: str | Path = "results/e0.json",
) -> tuple[dict, dict, dict]:
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    e12, e3i, e3n, comparison, e0 = map(read, (e12_path, e3_ideal_path, e3_noisy_path, comparison_path, e0_path))
    for key in ("E1_ideal", "E2_noisy"):
        if e12[key]["measured_counts_msb_first"] != comparison[key]["measured_counts_msb_first"]:
            raise ValueError(f"{key} counts differ between direct and combined records")
    for name, record in (("E3_logical_ideal", e3i), ("E3_logical_noisy", e3n)):
        if record["measured_counts_msb_first"] != comparison[name]["measured_counts_msb_first"]:
            raise ValueError(f"{name} counts differ between direct and combined records")
    e0_hartree = float(e0["E0_hartree"])
    if not math.isclose(e0_hartree, float(comparison["e0_reference"]["energy_hartree"]), abs_tol=1e-12):
        raise ValueError("E0 direct record and comparison record disagree")
    if not math.isclose(e0_hartree, float(comparison["e0_source_record"]["energy_hartree"]), abs_tol=1e-12):
        raise ValueError("E0 direct record and E0 source record disagree")
    return e12, {"ideal": e3i, "noisy": e3n}, {"comparison": comparison, "e0": e0}


def analyze_saved_records(
    *,
    e12_path: str | Path = "results/ipea_e1_e2_comparison.json",
    e3_ideal_path: str | Path = "results/ipea_e3_logical_ideal.json",
    e3_noisy_path: str | Path = "results/ipea_e3_logical_noisy.json",
    comparison_path: str | Path = "results/e0_e1_e2_e3_comparison.json",
    e0_path: str | Path = "results/e0.json",
    repeats_path: str | Path | None = None,
    bootstrap_replicates: int = 4000,
    bootstrap_seed: int = 88421,
) -> dict:
    e12, e3, refs = load_and_validate_records(e12_path, e3_ideal_path, e3_noisy_path, comparison_path, e0_path)
    comparison = refs["comparison"]
    e0 = refs["e0"]
    e0_hartree = float(e0["E0_hartree"])
    records = {
        "E1_ideal": e12["E1_ideal"],
        "E2_noisy": e12["E2_noisy"],
        "E3_logical_ideal": e3["ideal"],
        "E3_logical_noisy": e3["noisy"],
    }
    reference_record = records["E1_ideal"]
    t = float(reference_record["evolution_time"])
    interval = tuple(float(x) for x in reference_record["energy_interval_hartree"])
    for name, record in records.items():
        if not math.isclose(float(record["evolution_time"]), t, abs_tol=1e-12):
            raise ValueError(f"{name} uses a different evolution time")
        if list(record.get("energy_interval_hartree", record.get("settings", {}).get("energy_interval_hartree", []))) != list(interval):
            raise ValueError(f"{name} uses a different energy interval")
        if record["measurement_order"] != reference_record["measurement_order"] or record["iteration_k_order"] != reference_record["iteration_k_order"]:
            raise ValueError(f"{name} uses a different bit-order/power convention")
    conditions = {
        name: analyze_condition(name, record, e0_hartree=e0_hartree, evolution_time=t, energy_interval=interval, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
        for name, record in records.items()
    }
    counts1 = {row["bits_msb_first"]: row["shots"] for row in conditions["E1_ideal"]["phase_outcome_distribution_msb_first"]}
    counts3 = {row["bits_msb_first"]: row["shots"] for row in conditions["E3_logical_ideal"]["phase_outcome_distribution_msb_first"]}
    l1 = sum(abs(counts1[b] - counts3[b]) for b in BITSTRINGS)
    shot_count = conditions["E1_ideal"]["shots"]
    repeat_summary = None
    if repeats_path is not None:
        repeated = json.loads(Path(repeats_path).read_text(encoding="utf-8"))
        repeat_summary = _summarize_repeats(
            repeated, e0_hartree=e0_hartree, evolution_time=t,
            energy_interval=interval,
        )
    return {
        "analysis_label": "single-run sampling analysis plus optional independent-run summaries",
        "recorded_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "e0_reference": {
            "energy_hartree": e0_hartree,
            "method": e0["method"],
            "coefficient_source": e0["coefficient_source"],
            "provenance_note": "Saved project reference; saved FCI cross-check was not regenerated by this analysis.",
        },
        "conditions": conditions,
        "E1_vs_E3_ideal_distribution_agreement": {
            "exact_count_map_equality": l1 == 0,
            "L1_count_distance": l1,
            "total_variation_distance": 0.5 * l1 / shot_count,
            "shot_count_per_run": shot_count,
            "note": "Counts are normalized over all 32 MSB-first strings, filling omitted sparse E3 bins with zero.",
        },
        "sampling_design": {
            "original_repeats_per_condition": 1,
            "probability_intervals": "95% Wilson score intervals over within-run shot outcomes",
            "energy_intervals": "95% multinomial bootstrap intervals conditional on the resolved shots in one saved run",
            "run_level_variability": "not available from the original records; shots are not treated as independent runs",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "repeated_run_summary": repeat_summary,
        "inputs": {
            "E1_E2": str(e12_path),
            "E3_ideal": str(e3_ideal_path),
            "E3_noisy": str(e3_noisy_path),
            "combined_comparison": str(comparison_path),
            "E0": str(e0_path),
            "repeats": None if repeats_path is None else str(repeats_path),
        },
    }


def _summarize_repeats(
    record: dict, *, e0_hartree: float, evolution_time: float,
    energy_interval: tuple[float, float],
) -> dict:
    if record.get("status") != "completed":
        raise ValueError("repeat artifact is not marked completed")
    recorded_interval = tuple(float(x) for x in record["settings"]["energy_interval_hartree"])
    if recorded_interval != tuple(energy_interval):
        raise ValueError("repeat artifact uses a different energy interval")
    expected_seeds = list(record["settings"]["seed_list"])
    summaries = {}
    for condition, runs in record["conditions"].items():
        if not runs:
            raise ValueError(f"{condition}: repeat artifact contains no completed runs")
        if [run["seed"] for run in runs] != expected_seeds:
            raise ValueError(f"{condition}: repeated run seeds do not match the declared seed list")
        reference_probs = [run["summary"]["reference_probability"] for run in runs]
        modal_errors = [run["summary"]["modal_energy_absolute_error_vs_e0_hartree"] for run in runs if run["summary"]["modal_energy_absolute_error_vs_e0_hartree"] is not None]
        unresolved_rates = [run["summary"]["unresolved_probability"] for run in runs]
        distributions = []
        energy_stats = []
        unresolved_phase_counts = []
        for run in runs:
            raw = run["record"]
            shots = int(raw.get("shots", raw.get("settings", {}).get("shots", 0)))
            if shots != int(record["settings"]["shots_per_run"]):
                raise ValueError(f"{condition}: a repeat has an inconsistent shot count")
            if not math.isclose(float(raw["evolution_time"]), evolution_time, abs_tol=1e-12):
                raise ValueError(f"{condition}: a repeat uses a different evolution time")
            counts = _five_bit_counts(raw, shots)
            distributions.append({bits: counts[bits] / shots for bits in BITSTRINGS})
            resolved_energy: dict[str, float] = {}
            resolved_count: dict[str, int] = {}
            unresolved_count: dict[str, int] = {}
            for bits in BITSTRINGS:
                phase = bits_to_phase([int(ch) for ch in bits])
                try:
                    energy = energy_from_phase(phase, evolution_time, energy_interval)
                except ValueError as exc:
                    if "no phase alias" not in str(exc):
                        raise
                    if counts[bits]:
                        unresolved_count[bits] = counts[bits]
                else:
                    if counts[bits]:
                        resolved_energy[bits] = energy
                        resolved_count[bits] = counts[bits]
            resolved_n = sum(resolved_count.values())
            if resolved_n:
                values = list(resolved_energy.values())
                weights = list(resolved_count.values())
                mean = sum(e*w for e, w in zip(values, weights)) / resolved_n
                sample_var = sum(w*(e-mean)**2 for e, w in zip(values, weights)) / max(1, resolved_n-1)
                energy_errors = [e-e0_hartree for e in values]
                energy_stats.append({
                    "seed": run["seed"],
                    "resolved_shots": resolved_n,
                    "unresolved_shots": shots-resolved_n,
                    "mean_energy_hartree": mean,
                    "median_energy_hartree": _weighted_quantile(values, weights, 0.5),
                    "sample_std_energy_hartree": math.sqrt(sample_var),
                    "mean_absolute_error_vs_e0_hartree": sum(abs(e)*w for e,w in zip(energy_errors,weights))/resolved_n,
                    "rmse_vs_e0_hartree": math.sqrt(sum(e*e*w for e,w in zip(energy_errors,weights))/resolved_n),
                    "unresolved_phase_bit_counts": unresolved_count,
                })
            unresolved_phase_counts.append({"seed": run["seed"], "counts": unresolved_count})
        phase_means = {
            bits: float(np.mean([distribution[bits] for distribution in distributions]))
            for bits in BITSTRINGS
        }
        phase_sds = {
            bits: float(np.std([distribution[bits] for distribution in distributions], ddof=1)) if len(distributions) > 1 else None
            for bits in BITSTRINGS
        }
        summaries[condition] = {
            "independent_runs": len(runs),
            "seeds": [run["seed"] for run in runs],
            "shots_per_run": record["settings"]["shots_per_run"],
            "reference_probability_per_run": reference_probs,
            "reference_probability_mean": float(np.mean(reference_probs)),
            "reference_probability_sample_std": float(np.std(reference_probs, ddof=1)) if len(reference_probs) > 1 else None,
            "reference_probability_range": [min(reference_probs), max(reference_probs)],
            "modal_energy_absolute_error_vs_e0_per_run_hartree": modal_errors,
            "modal_energy_absolute_error_mean_hartree": float(np.mean(modal_errors)) if modal_errors else None,
            "modal_energy_absolute_error_sample_std_hartree": float(np.std(modal_errors, ddof=1)) if len(modal_errors) > 1 else None,
            "unresolved_probability_per_run": unresolved_rates,
            "unresolved_probability_mean": float(np.mean(unresolved_rates)),
            "unresolved_probability_sample_std": float(np.std(unresolved_rates, ddof=1)) if len(unresolved_rates) > 1 else None,
            "phase_outcome_probability_per_run": [
                {"seed": run["seed"], "probabilities_msb_first": distribution}
                for run, distribution in zip(runs, distributions)
            ],
            "phase_outcome_probability_mean_across_runs": phase_means,
            "phase_outcome_probability_sample_std_across_runs": phase_sds,
            "energy_statistics_conditional_on_resolved_outcomes_per_run": energy_stats,
            "unresolved_phase_counts_per_run": unresolved_phase_counts,
            "identical_full_phase_distributions_across_runs": all(
                distribution == distributions[0] for distribution in distributions[1:]
            ),
        }
    return {
        "settings": record["settings"],
        "conditions": summaries,
        "uncertainty_note": "Sample standard deviations and ranges are across distinct emulator seed runs; they are distinct from per-run shot-level Wilson/bootstrap intervals. Identical distributions across seeds are reported literally and should not be interpreted as evidence that run variability is absent.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e12", default="results/ipea_e1_e2_comparison.json")
    parser.add_argument("--e3-ideal", default="results/ipea_e3_logical_ideal.json")
    parser.add_argument("--e3-noisy", default="results/ipea_e3_logical_noisy.json")
    parser.add_argument("--comparison", default="results/e0_e1_e2_e3_comparison.json")
    parser.add_argument("--e0", default="results/e0.json")
    parser.add_argument("--repeats")
    parser.add_argument("--output", default="results/e0_e1_e2_e3_statistical_analysis.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=88421)
    args = parser.parse_args()
    artifact = analyze_saved_records(
        e12_path=args.e12,
        e3_ideal_path=args.e3_ideal,
        e3_noisy_path=args.e3_noisy,
        comparison_path=args.comparison,
        e0_path=args.e0,
        repeats_path=args.repeats,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "distribution_agreement": artifact["E1_vs_E3_ideal_distribution_agreement"],
        "conditions": {
            key: {
                "modal_bits": row["modal_phase_bits_msb_first"],
                "modal_probability": row["modal_probability"],
                "reference_probability": row["reference_probability"],
                "resolved": row["energy_statistics_conditional_on_resolved_outcomes"]["resolved_shots"],
                "unresolved": row["unresolved_outcomes"]["shots"],
            }
            for key, row in artifact["conditions"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
