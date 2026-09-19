"""
Figures 12, 13 and 14 for the research paper.

Shared visual language with fig01_workflow_flowchart.py:
navy ink, muted fills, no 3-D, no rainbow, every number printed on the chart.

HOW TO RUN
    python fig12_13_14.py
Writes PNG (300 dpi) and PDF next to this file. Copy the PNGs into
    e:\\2A_backtest_4k\\2A_backtest_4k\\paper_figures\\

Only matplotlib and numpy are required.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

# ------------------------------------------------------------------ palette
NAVY   = "#12263A"
LINE   = "#41546B"
MUTED  = "#55657A"
BLUE   = "#3B6EA5"
GOLD   = "#C8901B"
GREEN  = "#3F7D52"
RED    = "#B54A3C"
CREAM  = "#FDF3DE"
PAPER  = "#FAFBFC"
GRID   = "#E4E8EE"
V9C    = "#3B6EA5"
V16C   = "#C07B14"

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


def _save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {stem}.png and .pdf")


def _panel_label(ax, letter, title):
    ax.set_title(f"  {letter}   {title}", loc="left", fontsize=10.5,
                 fontweight="bold", color=NAVY, pad=8)


# ================================================================== FIGURE 12
def make_fig12():
    rates = ["0%\n(gross)", "0.02%\n(maker)", "0.05%\n(taker)"]
    v9_eq  = [4822.42, -61.58, -3577.36]
    v16_eq = [4734.36,  71.56, -3234.01]
    v9_pnl  = [3822.42, 1346.51, 1362.50]
    v16_pnl = [3734.36, 1375.39, 1343.22]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6),
                             gridspec_kw={"wspace": 0.28})

    # ---- Panel A: final equity
    ax = axes[0]
    x = np.arange(len(rates))
    w = 0.34
    b1 = ax.bar(x - w / 2, v9_eq,  w, color=V9C,  edgecolor=NAVY, linewidth=0.7,
                zorder=3, label="Version 9")
    b2 = ax.bar(x + w / 2, v16_eq, w, color=V16C, edgecolor=NAVY, linewidth=0.7,
                zorder=3, label="Version 16")
    ax.axhline(1000, color=GREEN, ls="--", lw=1.2, zorder=2)
    ax.axhline(0,    color=RED,   ls=":",  lw=1.1, zorder=2)
    ax.set_xlim(-0.55, 2.55)
    ax.text(2.48, 1120, "starting capital  $1,000", fontsize=7.2,
            color=GREEN, ha="right", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.94))
    ax.annotate(
        r"insolvency line  \$0",
        xy=(1.55, 0), xytext=(1.05, -720),
        fontsize=7.6, color=RED, ha="center", va="top", zorder=6,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="#E8C4C0", linewidth=0.55, alpha=0.98),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9),
    )

    def _ann(bars, vals, below_ok=True):
        for bar, v in zip(bars, vals):
            y = bar.get_height()
            if v >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 80,
                        f"${v:,.0f}", ha="center", va="bottom",
                        fontsize=7.4, fontweight="bold", color=NAVY)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, y - 80,
                        f"${v:,.0f}", ha="center", va="top",
                        fontsize=7.4, fontweight="bold", color=RED)

    _ann(b1, v9_eq)
    _ann(b2, v16_eq)

    ax.set_xticks(x)
    ax.set_xticklabels(rates, fontsize=9)
    ax.set_ylabel("Final equity  (USD)", fontsize=9.5)
    ax.set_ylim(-4300, 5800)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    # callout on the 0.05% pair
    ax.annotate("simulator continued\nafter equity < $0\n(not a live P&L)",
                xy=(2.17, -3234), xytext=(1.15, -2500),
                fontsize=7, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9),
                style="italic")
    _panel_label(ax, "A", "Outcome: neither cost rate is survivable")

    # ---- Panel B: implied gross P&L (the mechanism)
    ax = axes[1]
    b1 = ax.bar(x - w / 2, v9_pnl,  w, color=V9C,  edgecolor=NAVY, linewidth=0.7,
                zorder=3, label="Version 9")
    b2 = ax.bar(x + w / 2, v16_pnl, w, color=V16C, edgecolor=NAVY, linewidth=0.7,
                zorder=3, label="Version 16")
    offsets = [(-0.02, 70), (0.02, 70)]  # slight x-stagger so adjacent labels do not collide
    for bars, vals, (dx, dy) in ((b1, v9_pnl, offsets[0]), (b2, v16_pnl, offsets[1])):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2 + dx, bar.get_height() + dy,
                    f"${v:,.0f}", ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold", color=NAVY)

    ax.axhspan(1325, 1390, color="#E9F5EC", zorder=1)
    ax.text(2.42, 2100, "degenerate-regime band\nUSD 1,343 to 1,375",
            fontsize=7.2, color=GREEN, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(rates, fontsize=9)
    ax.set_ylabel("Implied gross P&L  (USD)", fontsize=9.5)
    ax.set_ylim(0, 4600)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.annotate("friction does not subtract from\na preserved gross profit;\nit prevents compounding",
                xy=(1.0, 1375), xytext=(1.55, 2800),
                fontsize=7.4, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9),
                style="italic")
    _panel_label(ax, "B", "Mechanism: implied gross P&L collapses, then stalls")

    fig.suptitle("Transaction-cost sensitivity of the two strongest decision layers",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)
    fig.text(0.5, -0.02,
             "Implied gross P&L = final equity minus USD 1,000 plus total costs charged.  "
             "Peak equity never exceeded USD 999.81 in any fee-adjusted run, so the milestone ratchet never engaged.",
             ha="center", fontsize=7.6, color=MUTED, style="italic")
    _save(fig, "fig12_gross_vs_fee")
    plt.close(fig)


# ================================================================== FIGURE 13
def make_fig13():
    names = ["V8", "V9", "V10", "V11", "V13", "V16"]
    equity = [4448.94, 4822.42, 3665.43, 6282.17, 5346.97, 4734.36]
    dd     = [9.05,    8.47,    7.34,    13.80,   11.81,   7.63]
    sharpe = [0.0355,  0.0328,  0.0339,  0.0297,  0.0310,  0.0359]
    colors = [BLUE, V9C, "#6A8BB0", "#8B3A32", "#6A5AA8", V16C]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7),
                             gridspec_kw={"wspace": 0.55})

    # ---- Panel A: identical selection
    # Win-rate axis only. Trade count is constant, so a second y-axis is not
    # needed and was colliding with Panel B's "Final equity" label.
    ax = axes[0]
    x = np.arange(6)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(51.55, 52.55)
    ax.axhline(52.04, color=GOLD, lw=2.0, zorder=2)
    ax.scatter(x, [52.04] * 6, s=110, c=colors, edgecolors=NAVY,
               linewidths=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Directional win rate  (%)", fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.03, 0.93, "win rate  52.04%  on every row",
            transform=ax.transAxes, fontsize=8.2, color=GOLD,
            fontweight="bold", va="top")
    ax.text(0.03, 0.08, "trade count  24,900  (69.4%)  on every row",
            transform=ax.transAxes, fontsize=8.2, color=GREEN,
            fontweight="bold", va="bottom")
    _panel_label(ax, "A", "Selection is identical by construction")

    # ---- Panel B: outcomes diverge
    ax = axes[1]
    x = np.arange(len(names))
    bars = ax.bar(x, equity, color=colors, edgecolor=NAVY, linewidth=0.7, zorder=3)
    for bar, v in zip(bars, equity):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 70,
                f"${v:,.0f}", ha="center", va="bottom",
                fontsize=7.3, fontweight="bold", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Final equity  (USD)", fontsize=9.5, labelpad=8)
    ax.set_ylim(0, 7600)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    axr = ax.twinx()
    axr.plot(x, dd, color=RED, marker="D", ms=7, lw=1.8, zorder=4)
    halo = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.92)
    dd_off = [0.55, 0.55, -1.05, 0.60, 0.55, 0.55]
    dd_va  = ["bottom", "bottom", "top", "bottom", "bottom", "bottom"]
    for i, (v, off, va) in enumerate(zip(dd, dd_off, dd_va)):
        # V11's marker sits on the tallest bar: put its label to the left.
        dx = -0.22 if i == 3 else 0.22
        ha = "right" if i == 3 else "left"
        axr.text(i + dx, v + off, f"{v:.2f}%", ha=ha, va=va,
                 fontsize=8.0, fontweight="bold", color=RED, bbox=halo, zorder=6)
    axr.set_ylabel("Maximum drawdown  (%)", fontsize=9.5, color=RED, labelpad=8)
    axr.tick_params(axis="y", colors=RED)
    axr.set_ylim(0, 18)
    axr.spines["top"].set_visible(False)

    ax.annotate("same 24,900 trades\n1.71x equity spread",
                xy=(3, 6282), xytext=(1.35, 6800),
                fontsize=7.4, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9),
                style="italic")
    _panel_label(ax, "B", "Sizing alone produces a 1.71x equity spread")

    fig.suptitle("Identical trade selection, divergent financial outcomes",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)
    fig.text(0.5, -0.02,
             "Versions 8, 9, 10, 11, 13 and 16 share the 80% consensus gate and the five frozen checkpoints, "
             "so they select the same candles and score the same directional outcomes. "
             "Everything that separates them is the decision layer.",
             ha="center", fontsize=7.6, color=MUTED, style="italic")
    _save(fig, "fig13_identical_selection")
    plt.close(fig)


# ================================================================== FIGURE 14
def make_fig14():
    configs = ["Base", "V4", "V5", "V6", "V7", "V8",
               "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16"]
    # columns: equity, win rate, 100-DD (so higher is better), Sharpe, PF, Calmar
    labels = ["Final\nequity ($)", "Win rate\n(%)", "Max DD\n(%) ↓",
              "Sharpe", "Profit\nfactor", "Calmar"]
    # raw values; DD is stored as the actual drawdown (lower better)
    raw = np.array([
        # eq      wr     dd     sharpe   pf     calmar
        [4134.67, 51.63, 11.57, 0.0307, 1.111, 26.44],
        [4487.10, 51.89, 10.75, 0.0310, 1.124, 31.65],
        [1996.99, 51.89,  3.77, 0.0343, 1.118, 25.84],
        [4679.16, 51.89,  8.56, 0.0339, 1.125, 41.97],
        [1098.02, 47.36,  8.27, 0.0043, 1.055,  1.16],
        [4448.94, 52.04,  9.05, 0.0355, 1.161, 37.19],
        [4822.42, 52.04,  8.47, 0.0328, 1.144, 44.05],
        [3665.43, 52.04,  7.34, 0.0339, 1.136, 35.43],
        [6282.17, 52.04, 13.80, 0.0297, 1.154, 37.37],
        [4724.30, 52.43, 10.04, 0.0321, 1.183, 36.22],
        [5346.97, 52.04, 11.81, 0.0310, 1.149, 35.93],
        [1833.73, 56.33,  7.45, 0.0217, 1.234, 14.57],
        [1106.26, 56.31,  3.08, 0.0102, 1.187,  3.37],
        [4734.36, 52.04,  7.63, 0.0359, 1.154, 47.75],
    ], dtype=float)

    # display strings
    disp = [
        ["4,135", "51.63", "11.57", "0.0307", "1.111", "26.44"],
        ["4,487", "51.89", "10.75", "0.0310", "1.124", "31.65"],
        ["1,997", "51.89",  "3.77", "0.0343", "1.118", "25.84"],
        ["4,679", "51.89",  "8.56", "0.0339", "1.125", "41.97"],
        ["1,098", "47.36",  "8.27", "0.0043", "1.055",  "1.16"],
        ["4,449", "52.04",  "9.05", "0.0355", "1.161", "37.19"],
        ["4,822", "52.04",  "8.47", "0.0328", "1.144", "44.05"],
        ["3,665", "52.04",  "7.34", "0.0339", "1.136", "35.43"],
        ["6,282", "52.04", "13.80", "0.0297", "1.154", "37.37"],
        ["4,724", "52.43", "10.04", "0.0321", "1.183", "36.22"],
        ["5,347", "52.04", "11.81", "0.0310", "1.149", "35.93"],
        ["1,834", "56.33",  "7.45", "0.0217", "1.234", "14.57"],
        ["1,106", "56.31",  "3.08", "0.0102", "1.187",  "3.37"],
        ["4,734", "52.04",  "7.63", "0.0359", "1.154", "47.75"],
    ]

    # min-max normalise; invert drawdown so brighter = better
    norm = np.zeros_like(raw)
    invert = {2}  # column index of drawdown
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        if hi == lo:
            n = np.ones_like(col) * 0.5
        else:
            n = (col - lo) / (hi - lo)
        if j in invert:
            n = 1.0 - n
        norm[:, j] = n

    cmap = LinearSegmentedColormap.from_list(
        "paper", ["#F4F1EA", "#D9E2EC", "#8BA7C4", "#3B6EA5", "#12263A"]
    )

    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(len(configs)):
        for j in range(len(labels)):
            # white text on dark cells
            tc = "white" if norm[i, j] > 0.62 else NAVY
            weight = "bold" if abs(norm[i, j] - 1.0) < 1e-9 else "normal"
            ax.text(j, i, disp[i][j], ha="center", va="center",
                    fontsize=9.0, color=tc, fontweight=weight)

    # outline the column-wise winners
    for j in range(len(labels)):
        i_win = int(np.argmax(norm[:, j]))
        ax.add_patch(plt.Rectangle((j - 0.5, i_win - 0.5), 1, 1,
                                   fill=False, edgecolor=GOLD, lw=2.0, zorder=5))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.6)
    ax.set_yticks(np.arange(len(configs)))
    ax.set_yticklabels(configs, fontsize=10.0, fontweight="bold")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(configs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Column-wise min–max  (brighter = better)", fontsize=8.2)
    cbar.outline.set_visible(False)

    ax.set_title("Normalised performance across all fourteen configurations  (gross / zero-fee)",
                 fontsize=13.0, fontweight="bold", color=NAVY, pad=14)
    fig.text(0.5, 0.005,
             "Gold outline marks the winner of each column. The six column winners fall on four rows "
             "(V11, V14, V15, V16): no configuration dominates.",
             ha="center", fontsize=9.2, color=MUTED)
    out = Path(r"e:\2A_backtest_4k\2A_backtest_4k\paper_figures") / "fig14_metric_heatmap"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {out}.png and .pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_fig12()
    make_fig13()
    make_fig14()
