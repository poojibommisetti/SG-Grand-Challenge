"""Create separate figures for statistical IPEA and targeted-fault artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_modal_energy_figure(analysis: dict, output: Path) -> list[Path]:
    names = list(analysis["conditions"])
    values = [1000.0 * analysis["conditions"][name]["modal_energy_absolute_error_vs_e0_hartree"] for name in names]
    labels = [name.replace("_", "\n") for name in names]
    colors = ["#2878B5", "#D9534F", "#59A14F", "#B279A2"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(np.arange(len(names)), values, color=colors[:len(names)])
    ax.set_xticks(np.arange(len(names)), labels)
    ax.set_ylabel("Absolute modal-energy error vs E0 (millihartree)")
    ax.set_title("H$_2$/STO-3G simulated IPEA: modal energy errors")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values, default=0.0) * 1.3 + 0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.04, f"{value:.3f} mHa", ha="center", fontsize=8)
    fig.text(0.5, 0.015, "Modal error is one accuracy summary; distributions and repeated-run results are reported separately. Noise is illustrative.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, output, "e0_e1_e2_e3_statistical_modal_energy_errors")


def make_phase_distribution_figure(analysis: dict, output: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    xs = np.arange(32)
    for name, condition in analysis["conditions"].items():
        dist = condition["phase_outcome_distribution_msb_first"]
        probs = [row["probability"] for row in dist]
        ax.plot(xs, probs, marker="o", markersize=3, linewidth=1, label=name)
    ax.axvline(int("10100", 2), color="black", linestyle=":", label="reference 10100")
    ax.set_xticks(xs, [format(i, "05b") for i in xs], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Five-bit phase outcome (MSB first)")
    ax.set_ylabel("Observed probability")
    ax.set_title("Full phase-outcome distributions (all 32 bins)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.text(0.5, 0.012, "Points are within-run observed proportions; original records contain one run per condition.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, output, "e0_e1_e2_e3_phase_outcome_distributions")


def make_fault_figure(faults: dict, output: Path) -> list[Path]:
    cases = [row for row in faults["conditions"] if row["case"] in ("no_fault", "single_x_d0", "single_x_d1", "single_x_d2", "single_z_d1_limitation", "double_x_d0_d1_limitation")]
    labels, unprotected_ref, encoded_ref = [], [], []
    unprotected_tv, encoded_tv = [], []
    for row in cases:
        labels.append(row["case"].replace("_", "\n"))
        unprotected_ref.append(None if row["unprotected_summary"] is None else row["unprotected_summary"]["reference_probability"])
        encoded_ref.append(row["encoded_summary"]["reference_probability"])
        unprotected_tv.append(None if row["unprotected_summary"] is None else row["unprotected_summary"]["phase_outcome_total_variation_from_ideal"])
        encoded_tv.append(row["encoded_summary"]["phase_outcome_total_variation_from_ideal"])
    xs = np.arange(len(labels)); width = 0.36
    fig, (rax, tax) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, p in enumerate(unprotected_ref):
        if p is not None:
            rax.bar(xs[i]-width/2, p, width, color="#9E9E9E", label="unprotected" if i == 0 else None)
    rax.bar(xs+width/2, encoded_ref, width, color="#59A14F", label="encoded + syndrome recovery")
    rax.set_ylim(0, 1.08)
    rax.set_ylabel("Probability of reference phase 10100")
    rax.set_title("Targeted k=0 Pauli fault injections (simulated)")
    rax.grid(axis="y", alpha=0.25)
    rax.legend(fontsize=8)
    for i, value in enumerate(unprotected_tv):
        if value is not None:
            tax.bar(xs[i]-width/2, value, width, color="#9E9E9E")
    tax.bar(xs+width/2, encoded_tv, width, color="#59A14F")
    tax.set_ylim(0, 1.08)
    tax.set_ylabel("Total variation from ideal phase distribution")
    tax.set_xticks(xs, labels, fontsize=8)
    tax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.012, "Deterministic faults after controlled-U$^1$ and before feed-forward; not a depolarizing-noise QEC benchmark. Double-X has no unprotected paired condition.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, output, "e3_targeted_x_fault_correction_performance")


def make_repeat_figure(repeats: dict, output: Path) -> list[Path]:
    names = list(repeats["conditions"])
    all_runs = [repeats["conditions"][name] for name in names]
    if not all_runs or min(len(runs) for runs in all_runs) < 2:
        return []
    fig, (pax, eax) = plt.subplots(1, 2, figsize=(11, 5.5))
    colors = ["#D9534F", "#B279A2"]
    x = np.arange(len(names))
    for i, (name, runs) in enumerate(zip(names, all_runs)):
        ps = np.array([run["summary"]["reference_probability"] for run in runs])
        es = np.array([1000*run["summary"]["modal_energy_absolute_error_vs_e0_hartree"] for run in runs])
        pax.scatter(np.full(len(ps), i), ps, color=colors[i], alpha=.75)
        pax.errorbar(i, ps.mean(), yerr=ps.std(ddof=1), fmt="s", color="black", capsize=4)
        eax.scatter(np.full(len(es), i), es, color=colors[i], alpha=.75)
        eax.errorbar(i, es.mean(), yerr=es.std(ddof=1), fmt="s", color="black", capsize=4)
    for ax in (pax, eax):
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.grid(axis="y", alpha=.25)
    pax.set_ylabel("Reference probability")
    pax.set_title("Across-seed reference probability")
    eax.set_ylabel("Modal energy error vs E0 (mHa)")
    eax.set_title("Across-seed modal-energy error")
    n = min(len(runs) for runs in all_runs)
    fig.suptitle(f"Independent noisy E2/E3 runs; n={n} seeds, {repeats['settings']['shots_per_run']} shots/run")
    fig.text(0.5, 0.012, "Black squares show mean and sample standard deviation across distinct seed runs; noise parameters are illustrative.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _save(fig, output, "e2_e3_repeated_run_variability")


def _save(fig, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output/f"{stem}.png", output/f"{stem}.svg"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="results/e0_e1_e2_e3_statistical_analysis.json")
    parser.add_argument("--faults", default="results/e3_targeted_x_fault_experiment.json")
    parser.add_argument("--repeats", default="results/e2_e3_repeated_noise_runs.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    output = Path(args.output_dir)
    outputs = []
    if Path(args.analysis).exists():
        analysis = _read(args.analysis)
        outputs += make_modal_energy_figure(analysis, output)
        outputs += make_phase_distribution_figure(analysis, output)
    if Path(args.faults).exists():
        outputs += make_fault_figure(_read(args.faults), output)
    if Path(args.repeats).exists():
        outputs += make_repeat_figure(_read(args.repeats), output)
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
