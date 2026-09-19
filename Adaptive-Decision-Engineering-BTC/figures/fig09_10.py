"""
Paper Figures 9 and 10 (thesis originals 31 and 32).

Figure 9: win rate vs final equity (accuracy-profitability paradox).
Figure 10: risk-return scatter; baseline is the five-model $4,134.67 / 0.0307,
not the seven-model $4,124.

HOW TO RUN
    python fig09_10.py
Writes PNG (300 dpi) and PDF into paper_figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(r"e:\2A_backtest_4k\2A_backtest_4k\paper_figures")

NAVY = "#12263A"
LINE = "#41546B"
MUTED = "#55657A"
BLUE = "#3B6EA5"
GOLD = "#C8901B"
GREEN = "#3F7D52"
RED = "#B54A3C"
PAPER = "#FAFBFC"
GRID = "#E4E8EE"
V9C = "#3B6EA5"
V16C = "#C07B14"
FAIL = "#B54A3C"
OTHER = "#8BA7C4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": LINE,
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "text.color": NAVY,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": PAPER,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = ["Base", "V4", "V5", "V6", "V7", "V8",
          "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16"]
EQUITY = np.array([4134.67, 4487.10, 1996.99, 4679.16, 1098.02, 4448.94,
                   4822.42, 3665.43, 6282.17, 4724.30, 5346.97, 1833.73,
                   1106.26, 4734.36])
SHARPE = np.array([0.0307, 0.0310, 0.0343, 0.0339, 0.0043, 0.0355,
                   0.0328, 0.0339, 0.0297, 0.0321, 0.0310, 0.0217,
                   0.0102, 0.0359])
WIN = np.array([51.63, 51.89, 51.89, 51.89, 47.36, 52.04,
                52.04, 52.04, 52.04, 52.43, 52.04, 56.33,
                56.31, 52.04])
PART = np.array([100.0, 100.0, 100.0, 100.0, 7.1, 69.4,
                 69.4, 69.4, 69.4, 42.6, 69.4, 11.4,
                 3.1, 69.4])
FAILED = {"V5", "V7", "V14", "V15"}


def _color(name: str) -> str:
    if name == "V9":
        return V9C
    if name == "V16":
        return V16C
    if name in FAILED:
        return FAIL
    return OTHER


def _save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {stem}.png and .pdf")


def make_fig09() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    x = np.arange(len(LABELS))
    colors = [_color(n) for n in LABELS]
    ax.bar(x, EQUITY, color=colors, edgecolor=NAVY, linewidth=0.55,
           width=0.72, zorder=3, label="Final equity")

    ax2 = ax.twinx()
    ax2.plot(x, WIN, color=V16C, marker="D",
             markersize=6.5, lw=2.0, zorder=4, label="Win rate")
    ax2.axhline(50, color=MUTED, ls=":", lw=1.0, zorder=1)
    ax2.text(-0.35, 50.15, "50%", fontsize=8, color=MUTED, va="bottom")

    # callouts: V9 (index 6) and V14 (index 11)
    ax.annotate("V9: 52.04% win rate\nbut USD 4,822 equity",
                xy=(6, 4822), xytext=(3.1, 6100),
                fontsize=9.0, color=V9C, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color=V9C, lw=1.15),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=V9C, linewidth=0.8))
    ax.annotate("V14: 56.33% win rate\nbut only USD 1,834",
                xy=(11, 1834), xytext=(12.35, 2550),
                fontsize=9.0, color=RED, fontweight="bold",
                ha="left",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.15),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=RED, linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5)
    ax.set_ylabel("Final equity (USD)", fontsize=10.5)
    ax2.set_ylabel("Win rate (%)", fontsize=10.5)
    ax.set_ylim(0, 7200)
    ax.set_xlim(-0.7, 14.55)
    ax2.set_ylim(44, 61)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    extra = [
        Patch(facecolor=V9C, edgecolor=NAVY, label="Version 9"),
        Patch(facecolor=V16C, edgecolor=NAVY, label="Version 16"),
        Patch(facecolor=FAIL, edgecolor=NAVY, label="Failed filters"),
    ]
    ax.legend(h1 + h2 + extra, l1 + l2 + [e.get_label() for e in extra],
              frameon=False, fontsize=8.4, loc="upper left", ncol=2)

    ax.set_title("Win rate versus final equity  (gross / zero-fee)",
                 fontsize=13.0, fontweight="bold", color=NAVY, pad=12)
    fig.text(0.5, -0.02,
             "The two highest win rates (V14 56.33%, V15 56.31%) finish among the three lowest equities. "
             "Win rate alone does not determine profitability.",
             ha="center", fontsize=9.0, color=MUTED)
    _save(fig, "fig09_accuracy_vs_profit")
    plt.close(fig)


def make_fig10() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 7.8))
    sizes = np.clip(PART, 3, 100) * 9.0
    for i, name in enumerate(LABELS):
        ax.scatter(SHARPE[i], EQUITY[i], s=sizes[i],
                   color=_color(name), edgecolors=NAVY, linewidths=0.8,
                   zorder=3, alpha=0.95)

    # Leader lines sit off the bubbles, matching the original scatter layout.
    # (text, x, y, text_x, text_y, ha, rad, bold)
    callouts = [
        (r"V11  (\$6,282)", 0.0297, 6282.17, 0.0172, 6580, "left", 0.16, False),
        (r"V13  (\$5,347)", 0.0310, 5346.97, 0.0195, 5920, "left", 0.22, False),
        (r"V9  (\$4,822)",  0.0328, 4822.42, 0.0172, 5180, "left", -0.14, True),
        (r"V6  (\$4,679)",  0.0339, 4679.16, 0.0386, 5580, "left", 0.32, False),
        (r"V16  (\$4,734)", 0.0359, 4734.36, 0.0386, 5050, "left", 0.18, True),
        (r"V12  (\$4,724)", 0.0321, 4724.30, 0.0172, 4700, "left", -0.08, False),
        (r"V8  (\$4,449)",  0.0355, 4448.94, 0.0386, 4280, "left", -0.22, False),
        (r"V4  (\$4,487)",  0.0310, 4487.10, 0.0172, 4360, "left", 0.08, False),
        (r"Base  (\$4,135)", 0.0307, 4134.67, 0.0158, 3920, "left", -0.12, True),
        (r"V10  (\$3,665)", 0.0339, 3665.43, 0.0386, 3480, "left", -0.16, False),
        (r"V14  (\$1,834)", 0.0217, 1833.73, 0.0248, 2480, "left", 0.20, False),
        (r"V5  (\$1,997)",  0.0343, 1996.99, 0.0386, 1680, "left", -0.18, False),
        (r"V7  (\$1,098)",  0.0043, 1098.02, 0.0082, 1520, "left", 0.22, False),
        (r"V15  (\$1,106)", 0.0102, 1106.26, 0.0140, 680,  "left", -0.18, False),
    ]
    box = dict(boxstyle="round,pad=0.30", facecolor="white",
               edgecolor="#D5DCE4", linewidth=0.4, alpha=0.96)
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    for text, x, y, tx, ty, ha, rad, bold in callouts:
        ax.annotate(
            text, xy=(x, y), xytext=(tx, ty),
            fontsize=8.8, fontweight="bold" if bold else "normal",
            color=NAVY, ha=ha, va="center", zorder=5,
            bbox=box,
            path_effects=halo,
            arrowprops=dict(
                arrowstyle="-|>", color="#4A5B70", lw=1.15,
                mutation_scale=10,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=1, shrinkB=6,
            ),
        )

    ax.axhline(4134.67, color=NAVY, ls=":", lw=1.0, zorder=1)
    ax.axvline(0.0307, color=NAVY, ls=":", lw=0.8, zorder=1)

    ax.set_xlabel("Per-candle Sharpe ratio   (higher = better risk-adjusted return)",
                  fontsize=10.5)
    ax.set_ylabel("Final equity (USD)", fontsize=10.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlim(0.0012, 0.0448)
    ax.set_ylim(420, 6950)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_sizes = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E849C",
               markeredgecolor=NAVY, markersize=14, label="100% of candles traded"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E849C",
               markeredgecolor=NAVY, markersize=10, label="~50% of candles traded"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E849C",
               markeredgecolor=NAVY, markersize=5.5, label="~10% of candles traded"),
        Patch(facecolor=V9C, edgecolor=NAVY, label="Version 9"),
        Patch(facecolor=V16C, edgecolor=NAVY, label="Version 16"),
        Patch(facecolor=FAIL, edgecolor=NAVY, label="Failed filters"),
    ]
    leg = ax.legend(handles=legend_sizes, frameon=True, fontsize=8.8,
                    loc="center left", bbox_to_anchor=(0.012, 0.40),
                    facecolor="white", edgecolor=NAVY, framealpha=0.97,
                    labelcolor=NAVY, borderpad=0.7)
    for t in leg.get_texts():
        t.set_color(NAVY)
        t.set_fontweight("bold")
    ax.set_title("Risk-return scatter of all fourteen configurations  (gross / zero-fee)",
                 fontsize=13.0, fontweight="bold", color=NAVY, pad=12)
    fig.text(0.5, -0.01,
             "Bubble size is the fraction of test candles traded. Baseline is the five-model "
             "standardised ensemble (USD 4,135, Sharpe 0.0307).",
             ha="center", fontsize=9.0, color=MUTED)
    _save(fig, "fig10_risk_return_scatter")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_fig09()
    make_fig10()
    print("done")
