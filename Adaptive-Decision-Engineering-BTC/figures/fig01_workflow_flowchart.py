"""
Figure 1 - Methodology flowchart with standard flowchart symbols.

Addresses the supervisor's request: keep it concise, keep every step, and use
proper flowchart graphics instead of identical rectangles. Each shape carries a
conventional meaning and a legend panel explains them.

  stadium / capsule   start and end terminators
  parallelogram       data input or output
  rectangle           process step
  hexagon             data preparation (windowing)
  cylinder            stored data (the frozen model weights)
  diamond             decision point
  document            generated report
  small circle        off-page connector, joins column 1 to column 2

Layout is two columns so the figure stays page-width rather than forcing the
reader to scroll a very tall single column. Column 1 ends in connector A and
column 2 begins at connector A, which is standard flowchart practice.

HOW TO RUN
  Paste into one Google Colab cell and run, or run locally:
      python fig01_workflow_flowchart.py
  Writes fig01_workflow_flowchart.png (300 dpi) and .pdf
  Put the PNG in:  e:\\2A_backtest_4k\\2A_backtest_4k\\paper_figures\\

Only matplotlib is required.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Polygon,
                                Circle, Ellipse, Rectangle)

# ----------------------------------------------------------------- palette
NAVY = "#12263A"
LINE = "#41546B"
PAL = {
    "term":  ("#D9E2EC", "#334E68"),   # terminators
    "io":    ("#E8F1FA", "#3B6EA5"),   # data input / output
    "proc":  ("#EDE9F6", "#6A5AA8"),   # generic process
    "prep":  ("#E4EEF7", "#4A7FA8"),   # preparation
    "store": ("#E6E0F3", "#5B4B9C"),   # frozen weights
    "dec":   ("#FDF3DE", "#C8901B"),   # decision
    "act":   ("#FCEFD8", "#C07B14"),   # decision-layer action
    "skip":  ("#FFF7CC", "#B58900"),   # skip / dampen branch
    "out":   ("#E9F5EC", "#3F7D52"),   # report
}


def _label(ax, cx, cy, text, fs, weight="normal", color=NAVY):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=color, zorder=5, linespacing=1.35)


def rect(ax, cx, cy, w, h, text, key="proc", fs=7.2, weight="normal", ls="-"):
    f, e = PAL[key]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                lw=1.4, facecolor=f, edgecolor=e, linestyle=ls,
                                zorder=3))
    _label(ax, cx, cy, text, fs, weight)
    return dict(cx=cx, cy=cy, w=w, h=h)


def stadium(ax, cx, cy, w, h, text, key="term", fs=7.6):
    f, e = PAL[key]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                lw=1.5, facecolor=f, edgecolor=e, zorder=3))
    _label(ax, cx, cy, text, fs, "bold")
    return dict(cx=cx, cy=cy, w=w, h=h)


def parallelogram(ax, cx, cy, w, h, text, key="io", fs=7.2, skew=4.5):
    f, e = PAL[key]
    pts = [(cx - w / 2 + skew, cy + h / 2), (cx + w / 2, cy + h / 2),
           (cx + w / 2 - skew, cy - h / 2), (cx - w / 2, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, lw=1.4, facecolor=f, edgecolor=e,
                         zorder=3))
    _label(ax, cx, cy, text, fs)
    return dict(cx=cx, cy=cy, w=w, h=h)


def hexagon(ax, cx, cy, w, h, text, key="prep", fs=7.2, notch=5.5):
    f, e = PAL[key]
    pts = [(cx - w / 2 + notch, cy + h / 2), (cx + w / 2 - notch, cy + h / 2),
           (cx + w / 2, cy), (cx + w / 2 - notch, cy - h / 2),
           (cx - w / 2 + notch, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, lw=1.4, facecolor=f, edgecolor=e,
                         zorder=3))
    _label(ax, cx, cy, text, fs)
    return dict(cx=cx, cy=cy, w=w, h=h)


def diamond(ax, cx, cy, w, h, text, key="dec", fs=7.0):
    f, e = PAL[key]
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, lw=1.5, facecolor=f, edgecolor=e,
                         zorder=3))
    _label(ax, cx, cy, text, fs, "bold")
    return dict(cx=cx, cy=cy, w=w, h=h)


def cylinder(ax, cx, cy, w, h, text, key="store", fs=7.2, ls="--"):
    """Stored-data symbol: body plus elliptical top and bottom."""
    f, e = PAL[key]
    eh = h * 0.26
    ax.add_patch(Ellipse((cx, cy - h / 2 + eh / 2), w, eh, lw=1.5,
                         facecolor=f, edgecolor=e, linestyle=ls, zorder=3))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + eh / 2), w, h - eh,
                           lw=0, facecolor=f, zorder=3))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2],
            color=e, lw=1.5, ls=ls, zorder=4)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2],
            color=e, lw=1.5, ls=ls, zorder=4)
    ax.add_patch(Ellipse((cx, cy + h / 2 - eh / 2), w, eh, lw=1.5,
                         facecolor=f, edgecolor=e, linestyle=ls, zorder=4))
    _label(ax, cx, cy - eh * 0.15, text, fs, "bold")
    return dict(cx=cx, cy=cy, w=w, h=h)


def document(ax, cx, cy, w, h, text, key="out", fs=7.2):
    """Report symbol: rectangle with a wavy bottom edge."""
    f, e = PAL[key]
    xs = np.linspace(-w / 2, w / 2, 60)
    wave = -h / 2 + 1.05 * np.sin(np.linspace(0, 2 * np.pi, 60))
    pts = [(cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2)]
    pts += [(cx + x, cy + y) for x, y in zip(xs[::-1], wave[::-1])]
    ax.add_patch(Polygon(pts, closed=True, lw=1.4, facecolor=f, edgecolor=e,
                         zorder=3))
    _label(ax, cx, cy + 0.6, text, fs)
    return dict(cx=cx, cy=cy, w=w, h=h)


def connector(ax, cx, cy, r, text, key="term", fs=8.5):
    f, e = PAL[key]
    ax.add_patch(Circle((cx, cy), r, lw=1.6, facecolor=f, edgecolor=e, zorder=3))
    _label(ax, cx, cy, text, fs, "bold")
    return dict(cx=cx, cy=cy, w=2 * r, h=2 * r)


# ------------------------------------------------------------ anchors / arrows
def bot(n):
    return (n["cx"], n["cy"] - n["h"] / 2)


def top(n):
    return (n["cx"], n["cy"] + n["h"] / 2)


def rgt(n):
    return (n["cx"] + n["w"] / 2, n["cy"])


def lft(n):
    return (n["cx"] - n["w"] / 2, n["cy"])


def arrow(ax, p1, p2, color=LINE, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=2))


def vlink(ax, a, b, **kw):
    arrow(ax, bot(a), top(b), **kw)


def elbow(ax, p1, p2, via_x=None, via_y=None, color=LINE, lw=1.6, ls="-"):
    """L-shaped connector. Give via_y for down-then-across, via_x for across-then-down."""
    if via_y is not None:
        ax.plot([p1[0], p1[0]], [p1[1], via_y], color=color, lw=lw, ls=ls,
                zorder=2, solid_capstyle="round")
        ax.plot([p1[0], p2[0]], [via_y, via_y], color=color, lw=lw, ls=ls,
                zorder=2, solid_capstyle="round")
        arrow(ax, (p2[0], via_y), p2, color, lw, ls)
    else:
        ax.plot([p1[0], via_x], [p1[1], p1[1]], color=color, lw=lw, ls=ls,
                zorder=2, solid_capstyle="round")
        ax.plot([via_x, via_x], [p1[1], p2[1]], color=color, lw=lw, ls=ls,
                zorder=2, solid_capstyle="round")
        arrow(ax, (via_x, p2[1]), p2, color, lw, ls)


def yes_no(ax, p, text, dx=0, dy=0, color="#8A6D1A"):
    ax.text(p[0] + dx, p[1] + dy, text, fontsize=6.8, fontweight="bold",
            color=color, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))


# -------------------------------------------------------------------- canvas
fig, ax = plt.subplots(figsize=(14.5, 9.6))
ax.set_xlim(0, 218)
ax.set_ylim(0, 144)
ax.set_aspect("equal")
ax.axis("off")

CL, CR, WCOL = 40, 126, 58          # column centres and node width
BR = 180                            # branch-box centre (right of column 2)

# =========================== COLUMN 1 : data and frozen predictive layer
n01 = stadium(ax, CL, 137, 30, 7.5, "START")
n02 = parallelogram(ax, CL, 125, WCOL, 11,
                    "Fetch Binance BTC/USDT 15-minute OHLCV\n"
                    "~276,000 candles (Aug 2017 - Jul 2025)")
n03 = rect(ax, CL, 111, WCOL, 10,
           "Chronological split\n240,000 train   /   35,903 test", "proc")
n04 = rect(ax, CL, 97, WCOL, 11,
           "Engineer eight price-action features\n"
           "(returns, wicks, rejection, support,\nvolatility, normalised volume)", "proc")
n05 = rect(ax, CL, 83, WCOL, 10,
           "Fit MaxAbsScaler on the training\npartition only, then apply to test", "proc")
n06 = hexagon(ax, CL, 69.5, WCOL, 10,
              "Build sliding windows\n96 timesteps x 8 features")

# three architectures side by side
mw = (WCOL - 2 * 2.5) / 3
mods = []
for i, t in enumerate(["Conv1D x 2\n32+64 filters\nk=3, GELU",
                       "LSTM x 1\nhidden 64",
                       "Hybrid x 2\nConv64 k=3\n-> LSTM 32"]):
    mods.append(rect(ax, CL - WCOL / 2 + mw / 2 + i * (mw + 2.5), 54.5, mw, 12,
                     t, "proc", fs=6.4))
n08 = cylinder(ax, CL, 35, WCOL, 17.5,
               "Five FROZEN checkpoints\nepochs 25, 80, 122, 78, 60\n"
               "saved when test-block equity > \\$3,000\nNEVER RETRAINED", fs=6.3)
nA1 = connector(ax, CL, 20, 4.2, "A")

for a, b in [(n01, n02), (n02, n03), (n03, n04), (n04, n05), (n05, n06)]:
    vlink(ax, a, b)
arrow(ax, bot(n06), top(mods[1]))
for m in mods:
    arrow(ax, bot(m), (m["cx"], 43.9))
vlink(ax, n08, nA1)

ax.text(CL, 144, "PHASE 1   Data preparation and frozen predictive layer",
        ha="center", va="top", fontsize=8.6, fontweight="bold", color="#4A7FA8")

# =========================== COLUMN 2 : decision layer and evaluation
nA2 = connector(ax, CR, 137, 4.2, "A")
n09 = rect(ax, CR, 126, WCOL, 10,
           "Weighted ensemble vote\ngraduated weights 1..5, normalised", "act")
d10 = diamond(ax, CR, 110, WCOL + 6, 17,
              "Consensus\n>= 80% ?", "dec", fs=7.2)
n11 = rect(ax, CR, 93, WCOL, 11,
           "Dynamic velocity scoring\n"
           "v = clip(1 + (WR$_{50}$ - 0.50) x 4,  0.30,  2.00)", "act", fs=6.8)
d12 = diamond(ax, CR, 75, WCOL + 6, 17,
              "Drawdown >= 10%\nor 2 losses ?", "dec", fs=7.0)
n13 = rect(ax, CR, 58, WCOL, 10,
           "Milestone tier sizing\n\\$750 -> \\$1,000 -> \\$1,300 -> \\$1,600", "act")
n14 = rect(ax, CR, 44, WCOL, 10,
           "Execute Long or Short\nsame candle: open -> close", "act")
n15 = rect(ax, CR, 30, WCOL, 10,
           "Update equity, rolling win rate\nand velocity window", "act")
n16 = document(ax, CR, 15, WCOL, 12,
               "Ten standardised metrics; gross comparison of\n"
               "baseline + V4-V16; cost sensitivity at 0.05% / 0.02%")
n17 = stadium(ax, BR, 15, 30, 7.5, "END")

vlink(ax, nA2, n09)
vlink(ax, n09, d10)
arrow(ax, bot(d10), top(n11)); yes_no(ax, (CR, 101), "YES", dx=6)
vlink(ax, n11, d12)
arrow(ax, bot(d12), top(n13)); yes_no(ax, (CR, 66), "NO", dx=6)
vlink(ax, n13, n14)
vlink(ax, n14, n15)
vlink(ax, n15, n16)
arrow(ax, rgt(n16), lft(n17))

# branch: consensus not met -> skip
b_skip = rect(ax, BR, 110, 34, 11, "Skip candle\nno trade\n11,003 of 35,903 (30.6%)",
              "skip", fs=6.6)
arrow(ax, rgt(d10), lft(b_skip), color="#B58900")
yes_no(ax, ((rgt(d10)[0] + lft(b_skip)[0]) / 2, 113.5), "NO")

# branch: risk trigger -> halve
b_halve = rect(ax, BR, 75, 34, 11, "Halve position size\n(soft circuit breaker /\nstreak guard)",
               "skip", fs=6.6)
arrow(ax, rgt(d12), lft(b_halve), color="#B58900")
yes_no(ax, ((rgt(d12)[0] + lft(b_halve)[0]) / 2, 78.5), "YES")
elbow(ax, bot(b_halve), top(n13), via_y=64.5, color="#B58900")

# feedback loop: updated state returns to the sizing rule
FB_X = CR - WCOL / 2 - 11
elbow(ax, lft(n15), lft(n11), via_x=FB_X, color="#2E7D46", ls="--")
ax.text(FB_X - 3.4, 61.5, "feedback: realised outcomes resize the next trade",
        fontsize=6.3, style="italic", color="#2E7D46", ha="center", va="center",
        rotation=90)

ax.text(CR, 144, "PHASE 2   Decision layer (varied across Versions 4-16) and evaluation",
        ha="center", va="top", fontsize=8.6, fontweight="bold", color="#C8901B")

# ==================================================================== legend
ax.add_patch(FancyBboxPatch((194, 22), 22, 112,
                            boxstyle="round,pad=0,rounding_size=2",
                            lw=1.2, facecolor="#FAFBFC", edgecolor="#B6C2CF",
                            zorder=1))
ax.text(205, 130.5, "SYMBOL KEY", fontsize=7.4, fontweight="bold", color=NAVY,
        ha="center", va="center", zorder=6)

leg = [(stadium,       "Start / end"),
       (parallelogram, "Data input"),
       (rect,          "Process"),
       (hexagon,       "Prepare data"),
       (cylinder,      "Frozen weights"),
       (diamond,       "Decision"),
       (document,      "Report"),
       (connector,     "Connector")]

y = 121
for fn, desc in leg:
    if fn is connector:
        fn(ax, 205, y, 2.7, "A", fs=6.2)
    elif fn is diamond:
        fn(ax, 205, y, 14, 7.2, "", fs=6)
    elif fn is cylinder:
        fn(ax, 205, y, 13, 6.8, "", fs=6)
    elif fn is document:
        fn(ax, 205, y, 14, 6.2, "", fs=6)
    else:
        fn(ax, 205, y, 14, 5.4, "", fs=6)
    ax.text(205, y - 5.6, desc, fontsize=5.8, color="#55657A",
            ha="center", va="center", zorder=6)
    y -= 12.9

# ------------------------------------------------------------------ captions
ax.text(2, 6.5,
        "The predictive layer (Phase 1) is frozen: five pretrained checkpoints are loaded and never retrained, refitted or reselected. "
        "Only the decision layer (Phase 2) varies across\nVersions 4-16, so every reported difference between versions is attributable to that layer. "
        "Execution is a same-candle open-to-close round-trip.",
        fontsize=7, style="italic", color="#55657A", va="top", ha="left")

plt.savefig("fig01_workflow_flowchart.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("fig01_workflow_flowchart.pdf", bbox_inches="tight", facecolor="white")
plt.show()
print("Saved fig01_workflow_flowchart.png and .pdf")
