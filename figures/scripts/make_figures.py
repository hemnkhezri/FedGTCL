"""Figures 2-13 of the paper, drawn from the values reported in the manuscript (Tables 6-13) and the logs of
src/13_cost_curve_data.py, src/14_scalability_analysis.py and src/15_multiclass_per_attack_accuracy.py.
No data are needed. Each figure is drawn at its printed width (6.5 in column), smallest text 8 pt, and saved as
figures/final/figNN.png (600 dpi) and figures/final/figNN.pdf (vector). Fig. 1 is produced by fig01_iiot_architecture.py.
Run from the repository root:  python figures/scripts/make_figures.py
"""
import os, sys, textwrap, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch, Polygon
from matplotlib.lines import Line2D

OUT = "figures/final"; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#3f3e3b", "#7a7974", "#e1e0dc", "#f2f1ee"
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b", "#8a8984"
BLUE_L = "#9cc3ef"
# diagram fills
F_CLIENT, E_CLIENT = "#eef5fd", "#2a78d6"
F_SERVER, E_SERVER = "#fdf1ea", "#c4561f"
F_BOX = "#ffffff"
W = 6.5
FS, TICK, TITLE, SMALL = 9.5, 9, 10, 8.5
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": FS, "axes.titlesize": TITLE,
    "axes.labelsize": FS, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.9, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
})


def save(fig, name):
    kw = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{OUT}/{name}.png", dpi=600, **kw)
    fig.savefig(f"{OUT}/{name}.pdf", **kw)
    print("saved", f"{OUT}/{name}.png and .pdf")
    plt.close(fig)


def wrap(s, n):
    return "\n".join(textwrap.fill(p, n, break_on_hyphens=False, break_long_words=False) for p in s.split("\n"))


def box(ax, x, y, w, h, title, sub=None, fc=F_BOX, ec=MUTED, tc=INK, lw=1.0, tsize=FS, ssize=SMALL, wrap_t=None,
        wrap_s=None, bold=True, ls="-", r=0.06):
    """Rounded box with a bold title and an optional smaller subtitle, laid out as one centered block."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    t = wrap(title, wrap_t) if wrap_t else title
    s = (wrap(sub, wrap_s) if wrap_s else sub) if sub else None
    lt, lsub, gap = tsize * 1.38 / 72, ssize * 1.42 / 72, 0.06
    nt = t.count("\n") + 1; ns = s.count("\n") + 1 if s else 0
    block = nt * lt + (gap + ns * lsub if s else 0)
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, t, ha="center", va="top", fontsize=tsize, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.18)
    if s:
        ax.text(x + w / 2, top - nt * lt - gap, s, ha="center", va="top", fontsize=ssize, color=INK2, linespacing=1.22)


def arrow(ax, p, q, color=INK2, lw=1.2, ls="-", ms=11, cs="arc3"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    mutation_scale=ms, shrinkA=0, shrinkB=0, connectionstyle=cs))


def canvas(h):
    fig = plt.figure(figsize=(W, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(0, h); return fig, ax


# ======================================================================================= Fig. 2
def fig2():
    H = 6.7
    fig, ax = canvas(H)
    # ---- client panel
    cx0, cw, top = 0.05, 3.72, 6.62
    ax.add_patch(FancyBboxPatch((cx0, 1.05), cw, top - 1.05, boxstyle="round,pad=0,rounding_size=0.1", fc=F_CLIENT, ec=E_CLIENT, lw=1.4))
    ax.add_patch(FancyBboxPatch((cx0, top - 0.38), cw, 0.38, boxstyle="round,pad=0,rounding_size=0.1", fc=E_CLIENT, ec=E_CLIENT, lw=1.4))
    ax.text(cx0 + cw / 2, top - 0.19, "Edge client k  (each of K sites, in parallel)", ha="center", va="center",
            fontsize=TITLE, color="white", fontweight="bold")
    bx, bw = cx0 + 0.15, cw - 0.3; mid = bx + bw / 2
    box(ax, bx, 5.4, bw, 0.68, "1  Local IIoT traffic", "sensors, PLCs and actuators; flow records never leave the site", wrap_s=52)
    box(ax, bx, 4.4, bw, 0.75, "2  Device-interaction graphs  (Section 3.2)",
        "60-s windows; hosts as nodes, aggregated flows as edges; min-max node features (Eq. (1))", wrap_s=52)
    box(ax, bx, 3.3, bw, 0.88, "3  Graph-temporal encoder  (Section 3.3, Fig. 5)",
        "sparsified low-rank graph attention (Eqs. (2)-(3)), attention readout, depthwise-separable temporal convolution (Eq. (4)); output: window embedding z",
        wrap_s=52)
    lw2 = (bw - 0.14) / 2; rx = bx + lw2 + 0.14
    box(ax, bx, 1.79, lw2, 1.25, "4a  Local training", "contrastive loss on all windows (Eqs. (5), (5a)) plus supervised loss on labeled windows (Eq. (6)); $E$ Adam epochs",
        wrap_s=27)
    box(ax, rx, 1.79, lw2, 1.25, "4b  Detection", "classification head of the global model: benign or attack for each window", wrap_s=26)
    box(ax, bx, 1.25, lw2, 0.5, "5  Sparsify the update", r"top-$s$ of $\Delta_k$ (Eq. (7))", fc="#dcebfb", ec=E_CLIENT, tsize=SMALL + 0.5, ssize=8)
    for y0, y1 in ((5.4, 5.15), (4.4, 4.15)):
        arrow(ax, (mid, y0), (mid, y1))
    arrow(ax, (bx + lw2 / 2, 3.3), (bx + lw2 / 2, 3.04)); arrow(ax, (rx + lw2 / 2, 3.3), (rx + lw2 / 2, 3.04))
    arrow(ax, (bx + lw2 / 2, 1.79), (bx + lw2 / 2, 1.75))
    # ---- server panel
    sx0, sw = 4.52, 1.93
    ax.add_patch(FancyBboxPatch((sx0, 1.05), sw, top - 1.05, boxstyle="round,pad=0,rounding_size=0.1", fc=F_SERVER, ec=E_SERVER, lw=1.4))
    ax.add_patch(FancyBboxPatch((sx0, top - 0.38), sw, 0.38, boxstyle="round,pad=0,rounding_size=0.1", fc=E_SERVER, ec=E_SERVER, lw=1.4))
    ax.text(sx0 + sw / 2, top - 0.19, "Server: FedAdaptOpt", ha="center", va="center", fontsize=TITLE, color="white", fontweight="bold")
    ax.text(sx0 + sw / 2, top - 0.58, "(Section 3.5)", ha="center", va="center", fontsize=SMALL, color=INK2)
    sbx, sbw = sx0 + 0.13, sw - 0.26; smid = sbx + sbw / 2
    box(ax, sbx, 1.1, sbw, 0.8, "Weighted aggregation", r"$\Delta^{(t)} = \sum_k (n_k/n)\,\Delta_k$" + "\n(Algorithm 1, line 19)")
    box(ax, sbx, 2.45, sbw, 0.75, "Moment estimates", r"$m^{(t)}$, $v^{(t)}$  (Eq. (8))")
    box(ax, sbx, 3.75, sbw, 1.05, "Adaptive server step", "new global model\n" + r"$\theta^{(t+1)}$ (Eq. (9))")
    arrow(ax, (smid, 1.9), (smid, 2.45)); arrow(ax, (smid, 3.2), (smid, 3.75))
    # ---- exchange
    arrow(ax, (bx + lw2, 1.5), (sbx, 1.5), color=E_CLIENT, lw=2.0, ms=14)
    ax.text(4.14, 1.42, "upload\nsparse " + r"$\Delta_k$", ha="center", va="top", fontsize=SMALL, color=E_CLIENT, fontweight="bold", linespacing=1.1)
    gx = 4.14; yb, ye = 4.27, 3.74
    ax.plot([sbx, gx, gx], [yb, yb, ye], color="#0f7a55", lw=2.0, ls=(0, (4, 2)), solid_capstyle="butt")
    arrow(ax, (gx, ye), (bx + bw, ye), color="#0f7a55", lw=2.0, ms=14, ls=(0, (4, 2)))
    ax.text(4.14, 4.36, "broadcast\n" + r"$\theta^{(t+1)}$", ha="center", va="bottom", fontsize=SMALL, color="#0f7a55", fontweight="bold",
            linespacing=1.1)
    ax.text(4.14, 2.55, "only model\nupdates\ncross the\nnetwork", ha="center", va="center", fontsize=8, color=INK2, style="italic",
            linespacing=1.1)
    # ---- envisioned deployment layer
    dy = 0.08
    ax.add_patch(FancyBboxPatch((0.05, dy), 6.4, 0.72, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=GRAY, lw=1.0,
                                ls=(0, (3, 2))))
    ax.text(3.25, dy + 0.55, "Envisioned deployment layer (not evaluated in this study)", ha="center", va="center", fontsize=SMALL,
            color=INK2, fontweight="bold")
    xs = [1.9, 3.25, 4.6]
    for x, t in zip(xs, ["alert generation", "prioritization", "mitigation"]):
        ax.text(x, dy + 0.22, t, ha="center", va="center", fontsize=SMALL, color=INK2)
    for a, b in ((2.55, 2.75), (3.8, 4.1)):
        arrow(ax, (a, dy + 0.22), (b, dy + 0.22), color=GRAY, lw=1.0, ms=9)
    arrow(ax, (rx + lw2 / 2, 1.79), (rx + lw2 / 2, dy + 0.72), color=GRAY, lw=1.1, ls=(0, (3, 2)), ms=10)
    # legend
    h = [Line2D([], [], color=INK2, lw=1.2), Line2D([], [], color=E_CLIENT, lw=1.8), Line2D([], [], color="#0f7a55", lw=1.8, ls=(0, (4, 2))),
         Line2D([], [], color=GRAY, lw=1.1, ls=(0, (3, 2)))]
    fig.legend(h, ["processing step", "client-to-server upload", "server-to-client broadcast", "envisioned, not evaluated"],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.0)
    save(fig, "fig02")


# ======================================================================================= Fig. 3
def fig3():
    H = 3.0
    fig, ax = canvas(H)
    cw, ch, y = 1.65, 0.72, 2.15
    xs = [0.25, 2.425, 4.6]
    for x, t in zip(xs, ["Edge client 1", "Edge client 2", "Edge client K"]):
        box(ax, x, y, cw, ch, t, "local data and training", fc=F_CLIENT, ec=E_CLIENT, tc="#174f93")
    ax.text(4.32, y + ch / 2, "…", ha="center", va="center", fontsize=14, color=INK2)
    sx, sw, sy, sh = 1.75, 3.0, 0.25, 0.8
    box(ax, sx, sy, sw, sh, "FedAdaptOpt server", "momentum-corrected, sparsification-aware aggregation", fc=F_SERVER, ec=E_SERVER,
        tc="#8a3a12", wrap_s=48)
    for x in xs:
        c = x + cw / 2
        arrow(ax, (c - 0.18, y), (sx + sw / 2 + (c - W / 2) * 0.55 - 0.12, sy + sh), color=E_CLIENT, lw=1.6, ms=12)
        arrow(ax, (sx + sw / 2 + (c - W / 2) * 0.55 + 0.12, sy + sh), (c + 0.18, y), color=AQUA, lw=1.6, ms=12, ls=(0, (4, 2)))
    h = [Line2D([], [], color=E_CLIENT, lw=1.6), Line2D([], [], color=AQUA, lw=1.6, ls=(0, (4, 2)))]
    fig.legend(h, [r"sparsified update $\Delta_k$ (Eq. (7))", r"global model $\theta^{(t+1)}$, every round $t$"], loc="lower center",
               bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=SMALL, handlelength=2.4)
    save(fig, "fig03")


# ======================================================================================= Fig. 4
def fig4():
    H = 2.25
    fig, ax = canvas(H)
    y = 0.75; h = 0.8
    box(ax, 0.05, y, 1.2, h, "Raw IIoT\ntraffic window", None, fc=SURF)
    box(ax, 1.55, 1.27, 1.6, 0.78, "Host extraction", "distinct source and destination IPs → nodes V", wrap_s=26, fc=F_CLIENT, ec=E_CLIENT)
    box(ax, 1.55, 0.25, 1.6, 0.78, "Flow aggregation", "communicating pairs → weighted edges E", wrap_s=26, fc=F_CLIENT, ec=E_CLIENT)
    box(ax, 3.45, y, 1.5, h, "Node-feature normalization", "bounded min-max (Eq. (1))", wrap_t=16, fc="#e8f6ef", ec=AQUA)
    box(ax, 5.2, y, 1.25, h, "Device-interaction graph", r"$G_k^t$", wrap_t=13, fc="#efe9fb", ec="#6b55c7")
    arrow(ax, (1.25, y + h / 2 + 0.1), (1.55, 1.66)); arrow(ax, (1.25, y + h / 2 - 0.1), (1.55, 0.64))
    arrow(ax, (3.15, 1.66), (3.45, y + h / 2 + 0.1)); arrow(ax, (3.15, 0.64), (3.45, y + h / 2 - 0.1))
    arrow(ax, (4.95, y + h / 2), (5.25, y + h / 2))
    save(fig, "fig04")


# ======================================================================================= Fig. 5
def fig5():
    H = 2.6
    fig, ax = canvas(H)
    y, h = 1.25, 1.0
    items = [("Graph window", "sequence of T\ngraphs", SURF, MUTED),
             ("Sparsified graph attention", "top-m neighbors,\nEqs. (2)-(3)", F_CLIENT, E_CLIENT),
             ("Graph-level readout", "attention-weighted\npooling", F_CLIENT, E_CLIENT),
             ("Depthwise-separable temporal convolution", "over T embeddings,\nEq. (4)", "#e8f6ef", AQUA),
             ("Embedding z", None, SURF, MUTED)]
    ws = [1.0, 1.2, 1.05, 1.45, 0.85]; gap = (W - 0.1 - sum(ws)) / 4
    x = 0.05; centers = []
    for (t, s, fc, ec), w in zip(items, ws):
        box(ax, x, y, w, h, t, s, fc=fc, ec=ec, wrap_t=14 if w < 1.2 else 18, tsize=SMALL + 0.5, ssize=8)
        centers.append((x, x + w)); x += w + gap
    for (a, b), (c, d) in zip(centers[:-1], centers[1:]):
        arrow(ax, (b, y + h / 2), (c, y + h / 2))
    zx = (centers[-1][0] + centers[-1][1]) / 2
    box(ax, 2.55, 0.08, 1.85, 0.72, "Contrastive projection head", "pretraining only (Eq. (5))", fc="#fdeee6", ec=ORANGE,
        tsize=SMALL + 0.5, ssize=8, wrap_t=30)
    box(ax, 4.6, 0.08, 1.85, 0.72, "Classification head", "benign or attack per window", fc="#efe9fb", ec="#6b55c7",
        tsize=SMALL + 0.5, ssize=8)
    arrow(ax, (zx - 0.2, y), (3.6, 0.8)); arrow(ax, (zx + 0.05, y), (5.5, 0.8))
    save(fig, "fig05")


# ======================================================================================= Fig. 6
def fig6():
    H = 5.92
    fig, ax = canvas(H)
    # lanes
    lx, lw_, rx, rw = 0.05, 3.0, 3.5, 2.95
    for x, w, fc, ec, t in ((lx, lw_, F_CLIENT, E_CLIENT, "Client k = 1, …, K  (in parallel)"),
                            (rx, rw, F_SERVER, E_SERVER, "Server (FedAdaptOpt)")):
        ax.add_patch(FancyBboxPatch((x, 0.05), w, 5.8, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=1.3))
        ax.add_patch(FancyBboxPatch((x, 5.5), w, 0.35, boxstyle="round,pad=0,rounding_size=0.1", fc=ec, ec=ec, lw=1.3))
        ax.text(x + w / 2, 5.675, t, ha="center", va="center", fontsize=TITLE, color="white", fontweight="bold")
    bx, bw = lx + 0.2, lw_ - 0.4; c = bx + bw / 2
    ax.add_patch(FancyBboxPatch((c - 0.55, 5.03), 1.1, 0.3, boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec=INK2, lw=1))
    ax.text(c, 5.18, "start, t = 0", ha="center", va="center", fontsize=SMALL, color=INK)
    steps = [(r"Download global model $\theta^{(t)}$", None, 4.3, 0.5),
             ("Build local graphs", "Section 3.2, Fig. 4", 3.4, 0.62),
             ("Local training, E epochs", "contrastive + supervised loss (Eqs. (5)-(6)); Adam step per epoch", 2.2, 0.95),
             ("Sparsify update", "top-s magnitude, Eq. (7)", 1.3, 0.62),
             (r"Send $\Delta_k$ to the server", None, 0.45, 0.5)]
    for t, s, y, h in steps:
        box(ax, bx, y, bw, h, t, s, fc="white", ec=E_CLIENT, tsize=SMALL + 0.5, ssize=8, wrap_s=34)
    tops = [(5.03, 4.8), (4.3, 4.02), (3.4, 3.15), (2.2, 1.92), (1.3, 0.95)]
    for a, b in tops:
        arrow(ax, (c, a), (c, b))
    sbx, sbw = rx + 0.25, rw - 0.5; sc = sbx + sbw / 2
    box(ax, sbx, 0.4, sbw, 0.75, "Aggregate updates", "sample-weighted sum (Algorithm 1, line 19)", fc="white", ec=E_SERVER,
        tsize=SMALL + 0.5, ssize=8, wrap_s=32)
    box(ax, sbx, 1.45, sbw, 0.75, "Momentum-corrected step", "Adam-style server update, Eqs. (8)-(9)", fc="white", ec=E_SERVER,
        tsize=SMALL + 0.5, ssize=8, wrap_s=32)
    arrow(ax, (bx + bw, 0.7), (sbx, 0.76), color=E_CLIENT, lw=1.6, ms=12)
    ax.text((bx + bw + sbx) / 2, 0.6, "upload", ha="center", va="top", fontsize=8, color=E_CLIENT, fontweight="bold")
    arrow(ax, (sc, 1.15), (sc, 1.45))
    # decision diamond
    dy, dh, dw = 2.95, 0.42, 0.95
    ax.add_patch(Polygon([(sc, dy + dh), (sc + dw, dy), (sc, dy - dh), (sc - dw, dy)], closed=True, fc="#fff4d6", ec=YELLOW, lw=1.3))
    ax.text(sc, dy, "t + 1 = R ?", ha="center", va="center", fontsize=SMALL + 0.5, color=INK, fontweight="bold")
    arrow(ax, (sc, 2.2), (sc, dy - dh))
    arrow(ax, (sc, dy + dh), (sc, 4.1))
    ax.text(sc + 0.08, dy + dh + 0.12, "yes", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.add_patch(FancyBboxPatch((sc - 0.7, 4.1), 1.4, 0.42, boxstyle="round,pad=0,rounding_size=0.2", fc=INK2, ec=INK2))
    ax.text(sc, 4.31, r"return $\theta^{*}$", ha="center", va="center", fontsize=SMALL + 0.5, color="white", fontweight="bold")
    # no -> broadcast loop
    gx = 3.27
    ax.plot([sc - dw, gx, gx], [dy, dy, 4.55], color="#0f7a55", lw=1.6, ls=(0, (4, 2)))
    arrow(ax, (gx, 4.55), (bx + bw, 4.55), color="#0f7a55", lw=1.6, ms=12, ls=(0, (4, 2)))
    ax.text(sc - dw - 0.08, dy + 0.1, "no", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.text(3.58, 3.95, "broadcast\n" + r"$\theta^{(t+1)}$;" + "\nnext round", ha="left", va="center", fontsize=8, color="#0f7a55",
            fontweight="bold", linespacing=1.15)
    save(fig, "fig06")


# ======================================================================================= charts (Figs. 7-13)
def label_bars(ax, bars, fmt="{:.1f}", dy=1.2, size=SMALL):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=size, color=INK)


def fig8():
    r = list(range(1, 16))
    g = [3.9273, 3.911, 3.9062, 3.5169, 3.3396, 3.5489, 3.1968, 3.4313, 3.2113, 2.9641, 4.5122, 4.2573, 5.5632, 4.1451, 4.1528]
    f = [0.6895, 0.6939, 0.6897, 0.6897, 0.6933, 0.6867, 0.6898, 0.6907, 0.6866, 0.6933, 0.6853, 0.687, 0.6942, 0.6897, 0.686]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"wspace": 0.42})
    a = axes[0]; a.plot(r, g, "-o", color=BLUE, lw=2, ms=5)
    a.set_ylim(2.5, 6); a.set_ylabel("Mean training loss\n" + r"($L_{local}$, Eq. (6))"); a.set_title("(a) FedGTCL", loc="left", color=INK)
    a = axes[1]; a.plot(r, f, "-s", color=ORANGE, lw=2, ms=5)
    a.axhline(0.6931, color=GRAY, ls=(0, (4, 2)), lw=1.1, label="ln(2) ≈ 0.693, random guess")
    a.set_ylim(0.6, 1.0); a.set_ylabel("Mean training loss\n(class-weighted CE)"); a.set_title("(b) FedAvg-GCN-GRU", loc="left", color=INK)
    a.legend(frameon=False, loc="upper right", fontsize=SMALL)
    for a in axes:
        a.set_xlabel("Communication round"); a.set_xticks([1, 3, 5, 7, 9, 11, 13, 15]); a.yaxis.grid(True, color=GRID, lw=0.7); a.set_axisbelow(True)
    save(fig, "fig07")


def fig10():
    ds = ["WUSTL-\nIIoT-2021\nIID", "WUSTL-\nIIoT-2021\nnon-IID", "TON_IoT-\nNetwork\nIID", "TON_IoT-\nNetwork\nnon-IID",
          "Edge-\nIIoTset\nIID", "Edge-\nIIoTset\nnon-IID"]
    fed = [73.47, 0.00, 93.40, 58.65, 94.32, 88.17]
    base = [0.00, 0.00, 94.36, 87.20, 95.03, 72.51]
    deg = [True, True, False, True, False, True]
    fig, ax = plt.subplots(figsize=(W, 3.5)); x = np.arange(6); w = 0.37
    b1 = ax.bar(x - w / 2, fed, w, color=BLUE)
    b2 = ax.bar(x + w / 2, base, w, color=[RED if d else AQUA for d in deg])
    from decimal import Decimal, ROUND_HALF_UP
    for bars in (b1, b2):
        for bb in bars:
            h = bb.get_height(); s = str(Decimal(str(round(h, 2))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
            ax.text(bb.get_x() + bb.get_width() / 2, h + 1.2, s, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=SMALL, linespacing=1.05); ax.set_ylabel("F1-score (%)"); ax.set_ylim(0, 112)
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=BLUE, label="FedGTCL (proposed)"), Patch(fc=AQUA, label="Best baseline, non-degenerate"),
                       Patch(fc=RED, label="Best baseline, degenerate (always attack)")], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=SMALL, columnspacing=1.2, handlelength=1.2)
    save(fig, "fig08")


def fig11():
    v = ["Full\nFedGTCL", "Without\ncontrastive loss", "Without\nFedAdaptOpt"]
    iid, non = [72.34, 46.32, 0.00], [7.14, 0.00, 0.00]
    fig, ax = plt.subplots(figsize=(4.9, 3.2)); x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, iid, w, color=BLUE, label="IID"); b2 = ax.bar(x + w / 2, non, w, color=BLUE_L, label="Non-IID")
    label_bars(ax, list(b1) + list(b2))
    ax.set_xticks(x); ax.set_xticklabels(v); ax.set_ylabel("F1-score (%)"); ax.set_ylim(0, 85)
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper right")
    save(fig, "fig09")


SEEDS = [[72.34, 78.95, 51.61], [70.83, 38.46, 0.00], [0.00, 0.00, 22.86], [0.00, 0.00, 0.00]]
GROUPS = ["FedGTCL\nIID", "FedAvg-\nGCN-GRU\nIID", "FedGTCL\nnon-IID", "FedAvg-\nGCN-GRU\nnon-IID"]
GCOL = [BLUE, ORANGE, BLUE, ORANGE]


def fig12():
    m = [np.mean(p) for p in SEEDS]; sd = [np.std(p, ddof=1) for p in SEEDS]
    fig, ax = plt.subplots(figsize=(4.9, 3.3)); x = np.arange(4)
    ax.bar(x, m, 0.6, color=GCOL, alpha=0.8, yerr=[[min(a, b) for a, b in zip(m, sd)], sd], capsize=6,
           error_kw={"lw": 1.4, "ecolor": INK})
    for i, p in enumerate(SEEDS):
        ax.scatter([i] * 3, p, s=28, color=INK, zorder=5)
    ax.text(0.99, 0.98, "SD bars truncated at 0", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(GROUPS, linespacing=1.05); ax.set_ylabel("F1-score (%)"); ax.set_ylim(-5, 100)
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    save(fig, "fig10")


def fig14():
    r = [5, 10, 15, 20, 25, 30]
    iid = [43.14, 72.34, 81.82, 27.08, 76.92, 51.61]; non = [0.00, 71.64, 57.14, 71.70, 27.78, 26.32]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot(r, iid, "-o", color=BLUE, lw=2, ms=6, label="IID")
    ax.plot(r, non, "-s", color=ORANGE, lw=2, ms=6, label="Non-IID (α = 0.3)")
    ax.axhline(94.44, color=GRAY, ls=(0, (4, 2)), lw=1.1, label="Centralized GCN-GRU upper bound")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Validation F1-score (%)"); ax.set_ylim(-5, 100); ax.set_xticks(r)
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=SMALL)
    save(fig, "fig11")


def fig15():
    data = {
        "WUSTL-IIoT-2021": {"benign": 99.88, "DoS": 99.95, "Recon.": 99.88, "Cmd. inj.": 92.31, "Backdoor": 93.02},
        "TON_IoT-Network": {"normal": 71.90, "scanning": 76.67, "DDoS": 89.55, "injection": 79.16, "password": 5.81, "DoS": 5.05,
                            "backdoor": 99.36, "XSS": 70.24, "ransomware": 80.59, "MITM": 90.43},
        "Edge-IIoTset (10%)": {"Normal": 100.00, "DDoS_UDP": 100.00, "DDoS_ICMP": 98.33, "SQL_injection": 34.77, "Password": 72.18,
                               "Vuln.scan": 86.14, "DDoS_TCP": 57.19, "DDoS_HTTP": 62.06, "Uploading": 37.18, "Backdoor": 93.78,
                               "Port_Scan": 50.00, "XSS": 76.18, "Ransomware": 13.70, "Fingerprint": 65.00, "MITM": 100.00},
    }
    fig, axes = plt.subplots(3, 1, figsize=(W, 7.6), gridspec_kw={"hspace": 0.75, "height_ratios": [1, 1, 1]})
    for i, (ax, (nm, cl)) in enumerate(zip(axes, data.items())):
        n = list(cl); v = list(cl.values())
        col = [GRAY if k.lower() in ("benign", "normal") else (RED if val < 50 else BLUE) for k, val in zip(n, v)]
        b = ax.bar(n, v, color=col, width=0.7)
        for bb, val in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=7.8 if len(n) > 10 else SMALL)
        ax.set_ylim(0, 112); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"({'abc'[i]}) {nm}", loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=8.5 if len(n) <= 10 else 8)
        plt.setp(ax.get_xticklabels(), rotation=35 if len(n) <= 10 else 45, ha="right", rotation_mode="anchor")
        ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    axes[0].legend(handles=[Patch(fc=GRAY, label="benign class"), Patch(fc=BLUE, label="attack class"),
                            Patch(fc=RED, label="attack class below 50%")], frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=SMALL)
    save(fig, "fig12")


def fig16():
    n = [20000, 40000, 60000, 80000, 100000, 150000]
    a = [0.004, 0.009, 0.017, 0.019, 0.028, 0.038]; b = [0.002, 0.004, 0.007, 0.01, 0.012, 0.016]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot(n, a, "-o", color=BLUE, lw=2, ms=6, label="FedGTCL encoder")
    ax.plot(n, b, "-s", color=ORANGE, lw=2, ms=6, label="Dense GCN-GRU baseline")
    ax.set_xticks(n); ax.set_xticklabels(["20,000", "40,000", "60,000", "80,000", "100,000", "150,000"], rotation=0, fontsize=8)
    ax.set_xlabel("Dataset size (raw flow records, chronological)"); ax.set_ylabel("Inference time, all sequences (s)")
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left")
    save(fig, "fig13")


ALL = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig8", "fig10", "fig11", "fig12", "fig14", "fig15", "fig16"]
# function names follow the drafting order; the output files carry the paper's figure numbers (fig02-fig13)
if __name__ == "__main__":
    for f in sys.argv[1:] or ALL:
        globals()[f]()
