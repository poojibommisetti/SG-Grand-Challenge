"""Run matched, independent-seed E2/E3 illustrative noise repeats."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ipea import bits_to_phase, energy_from_phase
from noise_model import DepolarizingNoiseConfig

DEFAULT_NOISE = {"p_1q": 1e-5, "p_2q": 1e-6, "p_meas": 1e-2, "p_init": 1e-3}
DEFAULT_SEEDS = [20261014, 20261015, 20261016, 20261017, 20261018]


def _counts(record: dict) -> dict[str, int]:
    raw = record["measured_counts_msb_first"]
    return {format(i, "05b"): int(raw.get(format(i, "05b"), 0)) for i in range(32)}


def summarize_run(record: dict, *, shots: int, e0: float, interval: tuple[float, float]) -> dict:
    counts = _counts(record)
    if sum(counts.values()) != shots:
        raise ValueError("phase count total does not match shots")
    t = float(record["evolution_time"])
    energies: dict[str, float] = {}
    unresolved_by_phase: dict[str, int] = {}
    for bits, count in counts.items():
        phase = bits_to_phase([int(ch) for ch in bits])
        try:
            energy = energy_from_phase(phase, t, interval)
        except ValueError as exc:
            if "no phase alias" not in str(exc):
                raise
            unresolved_by_phase[bits] = count
        else:
            energies[bits] = energy
    modal = max(counts, key=counts.get)
    try:
        modal_energy = energy_from_phase(bits_to_phase([int(ch) for ch in modal]), t, interval)
    except ValueError:
        modal_energy = None
    unresolved = sum(unresolved_by_phase.values())
    return {
        "modal_phase_bits_msb_first": modal,
        "modal_probability": counts[modal] / shots,
        "reference_probability": counts.get("10100", 0) / shots,
        "reference_count": counts.get("10100", 0),
        "modal_phase": bits_to_phase([int(ch) for ch in modal]),
        "modal_energy_hartree": modal_energy,
        "modal_energy_absolute_error_vs_e0_hartree": None if modal_energy is None else abs(modal_energy-e0),
        "unresolved_shots": unresolved,
        "unresolved_probability": unresolved / shots,
        "unresolved_phase_bit_counts": unresolved_by_phase,
        "resolved_shots": shots-unresolved,
        "phase_counts_msb_first": counts,
    }


def run_repeated_noise_comparison(
    *,
    shots_per_run: int,
    seeds: list[int],
    n_processes: int = 8,
    energy_interval: tuple[float, float] = (-1.2, -1.0),
    output_path: str | Path = "results/e2_e3_repeated_noise_runs.json",
) -> dict:
    if shots_per_run < 1 or n_processes < 1 or not seeds:
        raise ValueError("shots, process count, and repeat count must be positive")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("each repeat must use a distinct seed")

    from ipea import ipea_5bit_validation
    from qec_logical_ipea import run_qec_logical_ipea_5bit

    noise = DepolarizingNoiseConfig(**DEFAULT_NOISE)
    e0_data = json.loads(Path("results/e0.json").read_text(encoding="utf-8"))
    e0 = float(e0_data["E0_hartree"])
    artifact = {
        "label": "Repeated matched E2/E3 depolarizing comparison; illustrative emulator benchmark",
        "status": "running",
        "settings": {
            "shots_per_run": shots_per_run,
            "repeats_per_condition": len(seeds),
            "seed_list": seeds,
            "n_processes": n_processes,
            "energy_interval_hartree": list(energy_interval),
            "noise_parameters": DEFAULT_NOISE,
            "noise_parameter_label": "illustrative, not calibrated to H2 hardware",
            "E2_n_qubits": 12,
            "E3_n_qubits": 7,
            "simulator": "local Guppy/Selene Quest emulator",
            "E0_energy_hartree": e0,
            "run_pairing_note": "E2 and E3 use the same seed list and shots per run; distinct circuits mean equal seeds do not pair individual shot streams.",
        },
        "conditions": {"E2_noisy": [], "E3_logical_noisy": []},
        "completed_seed_pairs": 0,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for seed in seeds:
            print(f"Running matched E2/E3 repeat, seed={seed}, shots={shots_per_run}", flush=True)
            e2 = ipea_5bit_validation(
                shots=shots_per_run,
                seed=seed,
                noise_config=noise,
                n_processes=n_processes,
            )
            e2_summary = summarize_run(e2, shots=shots_per_run, e0=e0, interval=energy_interval)
            e3 = run_qec_logical_ipea_5bit(
                shots=shots_per_run,
                seed=seed,
                n_processes=n_processes,
                noise_config=noise,
                energy_interval=energy_interval,
            )
            e3_summary = summarize_run(e3, shots=shots_per_run, e0=e0, interval=energy_interval)
            artifact["conditions"]["E2_noisy"].append({"seed": seed, "summary": e2_summary, "record": e2})
            artifact["conditions"]["E3_logical_noisy"].append({"seed": seed, "summary": e3_summary, "record": e3})
            artifact["completed_seed_pairs"] += 1
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-per-run", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", help="comma-separated distinct seeds; defaults to five recorded explicit seeds")
    parser.add_argument("--n-processes", type=int, default=8)
    parser.add_argument("--output", default="results/e2_e3_repeated_noise_runs.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    seeds = DEFAULT_SEEDS if args.seeds is None else [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) != args.repeats:
        parser.error(f"received {len(seeds)} seeds but --repeats={args.repeats}")
    result = run_repeated_noise_comparison(
        shots_per_run=args.shots_per_run,
        seeds=seeds,
        n_processes=args.n_processes,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "completed_seed_pairs": result["completed_seed_pairs"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()