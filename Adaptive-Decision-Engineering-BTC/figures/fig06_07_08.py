"""
Paper Figures 6, 7 and 8 (thesis originals 28, 29, 30).

Corrects the seven-model baseline bar ($4,124 / 0.0305 / 11.64%)
to the five-model standardised baseline ($4,134.67 / 0.0307 / 11.57%).
All other values are Table 5. Visual language matches fig12_13_14.py.

HOW TO RUN
    python fig06_07_08.py
Writes PNG (300 dpi) and PDF into paper_figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

OUT = Path(r"e:\2A_backtest_4k\2A_backtest_4k\paper_figures")

NAVY = "#12263A"
LINE = "#41546B"
MUTED = "#55657A"
BLUE = "#3B6EA5"
GOLD = "#C8901B"
GREEN = "#3F7D52"
RED = "#B54A3C"
ORANGE = "#C07B14"
PAPER = "#FAFBFC"
GRID = "#E4E8EE"
V9C = "#3B6EA5"
V16C = "#C07B14"
FAIL = "#B54A3C"
OTHER = "#8BA7C4"
ZONE_SAFE = "#E7F1EA"
ZONE_MOD = "#FBF4E6"
ZONE_HIGH = "#F8E8E6"

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

# Table 5, experimental order Base, V4..V16
LABELS = ["Base", "V4", "V5", "V6", "V7", "V8",
          "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16"]
EQUITY = np.array([4134.67, 4487.10, 1996.99, 4679.16, 1098.02, 4448.94,
                   4822.42, 3665.43, 6282.17, 4724.30, 5346.97, 1833.73,
                   1106.26, 4734.36])
SHARPE = np.array([0.0307, 0.0310, 0.0343, 0.0339, 0.0043, 0.0355,
                   0.0328, 0.0339, 0.0297, 0.0321, 0.0310, 0.0217,
                   0.0102, 0.0359])
DD = np.array([11.57, 10.75, 3.77, 8.56, 8.27, 9.05,
               8.47, 7.34, 13.80, 10.04, 11.81, 7.45,
               3.08, 7.63])

CHAMPIONS = {"V9", "V16"}
FAILED = {"V5", "V7", "V14", "V15"}


def _bar_color(name: str) -> str:
    if name == "V9":
        return V9C
    if name == "V16":
        return V16C
    if name in FAILED:
        return FAIL
    return OTHER


def _save(fig, stem: str) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved {png.name} and {pdf.name}")


def make_fig06() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    x = np.arange(len(LABELS))
    colors = [_bar_color(n) for n in LABELS]
    bars = ax.bar(x, EQUITY, color=colors, edgecolor=NAVY, linewidth=0.6,
                  width=0.72, zorder=3)

    ax.axhline(1000, color=MUTED, ls="--", lw=1.1, zorder=2)
    ax.axhline(4134.67, color=NAVY, ls=":", lw=1.15, zorder=2)

    for bar, val, name in zip(bars, EQUITY, LABELS):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 90,
                f"${val:,.0f}", ha="center", va="bottom",
                fontsize=7.0, fontweight="bold" if name in CHAMPIONS else "normal",
                color=NAVY, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Final equity (USD)", fontsize=9.5)
    ax.set_ylim(0, 7200)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=V9C, edgecolor=NAVY, label="Version 9 (champion, equity)"),
        Patch(facecolor=V16C, edgecolor=NAVY, label="Version 16 (champion, Sharpe / DD)"),
        Patch(facecolor=FAIL, edgecolor=NAVY, label="Failed filters (V5, V7, V14, V15)"),
        Patch(facecolor=OTHER, edgecolor=NAVY, label="Other configurations"),
        Line2D([0], [0], color=NAVY, ls=":", lw=1.2, label="Baseline  USD 4,135"),
        Line2D([0], [0], color=MUTED, ls="--", lw=1.1, label="Starting capital  USD 1,000"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.4, ncol=2,
              loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.set_title("Final equity across all fourteen configurations  (gross / zero-fee)",
                 fontsize=12.0, fontweight="bold", color=NAVY, pad=18)
    fig.text(0.5, -0.02,
             "Baseline bar is the five-model standardised ensemble (USD 4,134.67), "
             "not the earlier seven-model USD 4,124 run.",
             ha="center", fontsize=7.4, color=MUTED, style="italic")
    _save(fig, "fig06_equity_comparison")
    plt.close(fig)


def make_fig07() -> None:
    order = np.argsort(SHARPE)  # low to high so highest sits at top
    names = [LABELS[i] for i in order]
    vals = SHARPE[order]
    colors = [_bar_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor=NAVY, linewidth=0.6,
            height=0.72, zorder=3)

    for yi, val, name in zip(y, vals, names):
        tag = f"{val:.4f}" + ("   best" if name == "V16" else "")
        ax.text(val + 0.00035, yi, tag, va="center", ha="left",
                fontsize=8.0, fontweight="bold" if name == "V16" else "normal",
                color=V16C if name == "V16" else NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Per-candle Sharpe ratio   (higher = better risk-adjusted return)",
                  fontsize=9.5)
    ax.set_xlim(0, 0.0435)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=V16C, edgecolor=NAVY, label="Version 16 (0.0359)"),
        Patch(facecolor=V9C, edgecolor=NAVY, label="Version 9 (0.0328)"),
        Patch(facecolor=FAIL, edgecolor=NAVY, label="Failed filters"),
        Patch(facecolor=OTHER, edgecolor=NAVY, label="Other configurations"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.0, loc="lower right")
    ax.set_title("Risk-adjusted ranking by per-candle Sharpe ratio  (gross / zero-fee)",
                 fontsize=12.0, fontweight="bold", color=NAVY, pad=10)
    fig.text(0.5, -0.01,
             "Sharpe is per candle with a zero risk-free rate; only the ordering is interpreted. "
             "Baseline is 0.0307 (five-model), not 0.0305.",
             ha="center", fontsize=7.4, color=MUTED, style="italic")
    _save(fig, "fig07_sharpe_ranking")
    plt.close(fig)


def _zone_color(dd: float) -> str:
    if dd <= 8.0:
        return GREEN
    if dd <= 10.05:
        return ORANGE
    return RED


def make_fig08() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    ax.axhspan(0, 8.0, facecolor=ZONE_SAFE, edgecolor="none", zorder=0)
    ax.axhspan(8.0, 10.0, facecolor=ZONE_MOD, edgecolor="none", zorder=0)
    ax.axhspan(10.0, 16.5, facecolor=ZONE_HIGH, edgecolor="none", zorder=0)
    ax.axhline(8.0, color=GREEN, ls="--", lw=0.8, zorder=1)
    ax.axhline(10.0, color=RED, ls="--", lw=0.8, zorder=1)

    x = np.arange(len(LABELS))
    colors = [_zone_color(v) for v in DD]
    bars = ax.bar(x, DD, color=colors, edgecolor=NAVY, linewidth=0.6,
                  width=0.72, zorder=3)

    for bar, val in zip(bars, DD):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.18,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Maximum drawdown  (%)", fontsize=9.5)
    ax.set_ylim(0, 16.4)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(13.55, 14.6, "HIGH RISK\n> 10%", fontsize=7.2, color=RED,
            ha="left", va="center", fontweight="bold")
    ax.text(13.55, 9.0, "MODERATE", fontsize=7.2, color=ORANGE,
            ha="left", va="center", fontweight="bold")
    ax.text(13.55, 3.6, "SAFE\n" + r"$\leq$ 8%", fontsize=7.2, color=GREEN,
            ha="left", va="center", fontweight="bold")

    handles = [
        Patch(facecolor=GREEN, edgecolor=NAVY, label=r"Safe  ($\leq$ 8%)"),
        Patch(facecolor=ORANGE, edgecolor=NAVY, label="Moderate  (8–10%)"),
        Patch(facecolor=RED, edgecolor=NAVY, label="High risk  (> 10%)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.0, loc="upper left")
    ax.set_title("Maximum drawdown by configuration  (gross / zero-fee)",
                 fontsize=12.0, fontweight="bold", color=NAVY, pad=10)
    fig.text(0.5, -0.02,
             "Lower is safer. Version 15 (3.08%) is the lowest drawdown in the study; "
             "Version 16 (7.63%) is the lowest among high-participation configurations.",
             ha="center", fontsize=7.4, color=MUTED, style="italic")
    _save(fig, "fig08_drawdown")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_fig06()
    make_fig07()
    make_fig08()
    print("done")
