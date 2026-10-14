"""Plot saved E0/E1/E2/E3 logical-IPEA comparison results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _bitstring(value) -> str:
    if isinstance(value, str):
        return value
    return "".join(str(int(bit)) for bit in value)


def make_e0_e1_e2_e3_figure(data: dict, output_dir: str | Path) -> list[Path]:
    e0 = float(data["e0_reference"]["energy_hartree"])
    estimates = [
        ("E1 ideal", data["E1_ideal"], "#2878B5"),
        ("E2 noisy", data["E2_noisy"], "#D9534F"),
        ("E3 logical ideal", data["E3_logical_ideal"], "#59A14F"),
        ("E3 logical noisy", data["E3_logical_noisy"], "#B279A2"),
    ]
    fig, (energy_ax, phase_ax) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.0, 2.1]}
    )
    x = np.arange(len(estimates) + 1)
    energies = [e0] + [float(row["modal_energy_hartree"]) for _, row, _ in estimates]
    errors_mha = [0.0] + [abs(energy - e0) * 1000.0 for energy in energies[1:]]
    labels = ["E0 exact"] + [name for name, _, _ in estimates]
    colors = ["black"] + [color for _, _, color in estimates]
    bars = energy_ax.bar(x, errors_mha, color=colors, width=0.64, zorder=3)
    energy_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    energy_ax.set_ylim(0.0, max(errors_mha) * 1.35 + 0.25)
    energy_ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    energy_ax.set_ylabel("Absolute error from E0 (millihartree)")
    energy_ax.set_title("Absolute modal energy error vs E0")
    energy_ax.grid(axis="y", alpha=0.25)
    energy_ax.scatter([0], [0], marker="*", s=110, color="black", zorder=4)
    for xpos, error in enumerate(errors_mha):
        y = error + 0.08 if error else 0.08
        text = "0.000 mHa" if xpos == 0 else f"{error:.3f} mHa"
        energy_ax.text(xpos, y, text, ha="center", va="bottom", fontsize=8)
    bitstrings = [format(value, "05b") for value in range(32)]
    bins = np.arange(32)
    width = 0.19
    reference_bits = _bitstring(data["E1_ideal"]["reference_bits_msb_first"])
    for offset, (label, row, color) in enumerate(estimates):
        counts = row["measured_counts_msb_first"]
        y = [int(counts.get(bits, 0)) for bits in bitstrings]
        phase_ax.bar(
            bins + (offset - 1.5) * width,
            y,
            width,
            label=label,
            color=color,
            alpha=0.88,
        )
    phase_ax.axvline(int(reference_bits, 2), color="#333333", linestyle=":", linewidth=1.2,
                     label=f"E1 reference {reference_bits}")
    phase_ax.set_xticks(bins, bitstrings, rotation=60, ha="right", fontsize=7)
    phase_ax.set_xlabel("Measured phase outcome (MSB-first five-bit string)")
    phase_ax.set_ylabel("Shots")
    phase_ax.set_title(f"Phase distributions; exact phase={data['E1_ideal']['exact_ground_phase']:.6f}")
    phase_ax.grid(axis="y", alpha=0.25)
    phase_ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"H₂/STO-3G: E0={e0:.9f} Ha reference; E1/E2/E3 IPEA", fontsize=14)
    fig.text(
        0.5,
        0.012,
        "One run per condition; no repeat-run uncertainty bars. E2/E3 noise is illustrative. "
        "The bit-flip code corrects only demonstrated isolated X faults at syndrome boundaries.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "e0_e1_e2_e3_logical_ipea.png", output / "e0_e1_e2_e3_logical_ipea.svg"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/e0_e1_e2_e3_comparison.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    for path in make_e0_e1_e2_e3_figure(_read(args.input), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()