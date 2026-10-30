"""Plot the primary rates and confidence intervals reported in Table 2.

Run ``python src/plot_primary_results.py`` from the repository root.
Add ``--compact`` for a figure sized for one ACL column.
Reads the saved analysis results without rerunning or changing the analysis.
Writes PDF, SVG, and 400-dpi PNG versions to results/.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PERSONAS = [
    ("neutral", "Neutral"),
    ("childrens_tutor", "Children's tutor"),
    ("medical_professional", "Medical"),
    ("cybersecurity_professional", "Cybersecurity"),
    ("creative_fiction_writer", "Creative writer"),
    ("customer_support_agent", "Customer support"),
]


def save_figure(fig, basename: str) -> None:
    output = ROOT / "results" / basename
    for extension in ("pdf", "svg", "png"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=400)
        print(path)
    plt.close(fig)


def plot_compact(rows: dict) -> None:
    """Compare both outcomes on one scale, with fonts sized for one column."""
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.19, top=0.84)
    ax.axhspan(-0.43, 0.43, color="#f1f1f1", zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    for key, label, color, marker, offset in [
        ("over_refusal", "Over-refusal", "#0072B2", "o", -0.16),
        ("under_refusal", "Under-refusal", "#D55E00", "D", 0.16),
    ]:
        estimates, lows, highs = [
            [float(rows[persona][key + suffix]) * 100 for persona, _ in PERSONAS]
            for suffix in ("", "_lo", "_hi")
        ]
        if not all(0 <= low <= estimate <= high <= 100
                   for estimate, low, high in zip(estimates, lows, highs)):
            raise ValueError(f"Invalid confidence intervals for {key}")
        ax.errorbar(
            estimates,
            [y + offset for y in range(len(PERSONAS))],
            xerr=[
                [estimate - low for estimate, low in zip(estimates, lows)],
                [high - estimate for estimate, high in zip(estimates, highs)],
            ],
            fmt=marker,
            label=label,
            color=color,
            markersize=3.2,
            elinewidth=1,
            capsize=2,
            capthick=0.8,
            zorder=3,
        )
    ax.set_yticks(range(len(PERSONAS)), [
        "Neutral", "Tutor", "Medical", "Cybersecurity", "Creative", "Support"
    ])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(len(PERSONAS) - 0.48, -0.52)
    ax.set_xlim(-0.6, 16)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Rate (%)", fontsize=9, labelpad=4)
    ax.tick_params(axis="y", length=0, pad=5, labelsize=8.5)
    ax.tick_params(axis="x", length=3, color="#777777", labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.6)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    save_figure(fig, "fig_primary_results_single_column")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="Fit one paper column")
    args = parser.parse_args()
    with (ROOT / "results/rates_by_scenario.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        primary = [row for row in csv.DictReader(handle) if row["scenario"] == "primary"]
    rows = {row["persona"]: row for row in primary}
    if len(primary) != len(PERSONAS) or set(rows) != {p for p, _ in PERSONAS}:
        raise ValueError("Expected exactly one primary-results row for each persona.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    if args.compact:
        plot_compact(rows)
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.20, top=0.79, wspace=0.17)
    panels = [
        ("over_refusal", "(a) Over-refusal", "Safe prompts (n = 250)", "#0072B2", (-0.25, 6), range(7)),
        ("under_refusal", "(b) Under-refusal", "Unsafe prompts (n = 200)", "#D55E00", (-0.65, 16), [0, 4, 8, 12, 16]),
    ]
    for ax, (key, title, subtitle, color, limits, ticks) in zip(axes, panels):
        ax.axhspan(-0.47, 0.43, color="#f1f1f1", zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dddddd", linewidth=0.55)

        for y, (persona, _) in enumerate(PERSONAS):
            row = rows[persona]
            estimate, low, high = [
                float(row[column]) * 100
                for column in (key, key + "_lo", key + "_hi")
            ]
            if not 0 <= low <= estimate <= high <= 100:
                raise ValueError(f"Invalid confidence interval for {persona}: {key}")
            ax.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=color,
                markersize=4.5,
                elinewidth=1.25,
                capsize=3,
                capthick=1,
                zorder=3,
            )
            ax.annotate(
                f"{estimate:.1f}",
                (estimate, y),
                xytext=(0, 6.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#222222",
            )

        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_ylim(len(PERSONAS) - 0.55, -0.7)
        ax.set_xlabel("Rate (%)", labelpad=6)
        ax.set_title(title, loc="left", pad=31, fontweight="bold")
        ax.text(0, 1.07, subtitle, transform=ax.transAxes, fontsize=9, color="#444444")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=3, color="#777777")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["bottom"].set_linewidth(0.6)

    axes[0].set_yticks(range(len(PERSONAS)), [label for _, label in PERSONAS])
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.text(
        0.205,
        0.035,
        "Points: estimates; bars: 95% bootstrap CIs. Panels use different horizontal scales.",
        fontsize=8,
        color="#444444",
    )

    save_figure(fig, "fig_primary_results")


if __name__ == "__main__":
    main()
