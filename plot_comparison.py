"""Plot saved E0/E1/E2 results and the separate E3 component benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def make_energy_and_phase_figure(data: dict, output_dir: Path) -> list[Path]:
    e0 = float(data["e0_reference"]["energy_hartree"])
    ideal = data["E1_ideal"]
    noisy = data["E2_noisy"]
    energies = [
        e0,
        float(ideal["modal_energy_hartree"]),
        float(noisy["modal_energy_hartree"]),
    ]
    labels = ["E0 exact diagonalization", "E1 ideal IPEA", "E2 noisy IPEA"]
    colors = ["black", "#2878B5", "#D9534F"]

    fig, (energy_ax, phase_ax) = plt.subplots(
        1,
        2,
        figsize=(16, 6.2),
        gridspec_kw={"width_ratios": [1.0, 1.8]},
    )
    x = np.arange(3)
    energy_ax.axhline(e0, color="black", linestyle="--", linewidth=1.2, label="E0 reference")
    energy_ax.scatter(x, energies, s=72, c=colors, zorder=3)
    span = max(energies) - min(energies)
    margin = max(0.002, span * 0.25)
    energy_ax.set_ylim(min(energies) - margin, max(energies) + margin)
    energy_ax.set_xticks(x, labels, rotation=15, ha="right")
    energy_ax.set_ylabel("Energy (Hartree)")
    energy_ax.set_title("Modal energy estimates")
    energy_ax.grid(axis="y", alpha=0.25)
    for xpos, energy, label in zip(x, energies, labels):
        if xpos == 0:
            annotation = f"{energy:.9f} Ha\nreference"
        else:
            error_mha = abs(energy - e0) * 1000.0
            annotation = f"{energy:.9f} Ha\n|error|={error_mha:.3f} mHa"
        energy_ax.annotate(
            annotation,
            (xpos, energy),
            xytext=(0, 12 if xpos != 2 else -38),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    energy_ax.legend(loc="lower right", fontsize=8)

    bitstrings = [format(value, "05b") for value in range(32)]
    ideal_counts = ideal["measured_counts_msb_first"]
    noisy_counts = noisy["measured_counts_msb_first"]
    ideal_y = [int(ideal_counts.get(bits, 0)) for bits in bitstrings]
    noisy_y = [int(noisy_counts.get(bits, 0)) for bits in bitstrings]
    bins = np.arange(32)
    width = 0.42
    phase_ax.bar(bins - width / 2, ideal_y, width, label="E1 ideal", color="#2878B5")
    phase_ax.bar(bins + width / 2, noisy_y, width, label="E2 noisy", color="#D9534F", alpha=0.82)
    reference_bits = "".join(str(bit) for bit in ideal["reference_bits_msb_first"])
    reference_index = int(reference_bits, 2)
    phase_ax.axvline(reference_index, color="#333333", linestyle=":", linewidth=1.2)
    phase_ax.set_xticks(bins, bitstrings, rotation=60, ha="right", fontsize=7)
    phase_ax.set_xlabel("Measured phase outcome (MSB-first 5-bit string)")
    phase_ax.set_ylabel("Shots")
    phase_ax.set_title(
        f"Phase distribution; exact phase={float(ideal['exact_ground_phase']):.6f}, "
        f"reference bits={reference_bits}"
    )
    phase_ax.grid(axis="y", alpha=0.25)
    phase_ax.legend()

    fig.suptitle("H₂/STO-3G: ideal and illustrative noisy five-bit IPEA", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "One run per condition; no repeat-run error bars. E3 has no logical-IPEA energy estimate.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "e0_e1_e2_comparison.png",
        output_dir / "e0_e1_e2_comparison.svg",
    ]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_qec_component_figure(data: dict, output_dir: Path) -> list[Path]:
    ideal = data["ideal"]
    noisy = data["noisy"]
    shots = int(data["settings"]["shots_per_circuit"])

    categories = [
        "Unprotected\nideal",
        "Repetition code\nideal",
        "Unprotected\nnoisy",
        "Repetition code\nnoisy",
    ]
    success_counts = [
        int(ideal["unprotected_counts"].get("0", 0)),
        int(ideal["encoded_decoded_counts"].get("0", 0)),
        int(noisy["unprotected_counts"].get("0", 0)),
        int(noisy["encoded_decoded_counts"].get("0", 0)),
    ]
    probabilities = [count / shots for count in success_counts]
    colors = ["#9E9E9E", "#59A14F", "#9E9E9E", "#59A14F"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(categories))
    bars = ax.bar(x, probabilities, color=colors, width=0.68)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction read as decoded |0⟩")
    ax.set_title("Three-qubit bit-flip component: known injected X(q0)")
    ax.grid(axis="y", alpha=0.25)
    for bar, count, probability in zip(bars, success_counts, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(probability + 0.025, 1.07),
            f"{count}/{shots}\n{probability:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.015,
        "Single known X error only; this is a component test, not QEC-protected IPEA or an energy result.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "e3_repetition_component.png",
        output_dir / "e3_repetition_component.svg",
    ]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipea", default="results/ipea_e1_e2_comparison.json")
    parser.add_argument("--qec", default="results/e3_repetition_component.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    outputs = make_energy_and_phase_figure(_read_json(args.ipea), output_dir)
    outputs.extend(make_qec_component_figure(_read_json(args.qec), output_dir))
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()