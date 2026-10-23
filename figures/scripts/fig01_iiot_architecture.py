"""Fig. 1: layered IIoT architecture (perceptual -> network -> application), drawn as vector art.
Run from the repository root; writes figures/final/fig01.png (600 dpi) and fig01.pdf."""
import sys
sys.path.insert(0, "figures/scripts")
import make_figures as M
from make_figures import plt, FancyBboxPatch, box, arrow, INK, INK2, MUTED, BLUE, AQUA, ORANGE
W, H = 5.0, 5.75
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
# top: cloud and gateway
box(ax, 0.45, 5.0, 2.0, 0.55, "Cloud", "central infrastructure", fc="#eef5fd", ec=BLUE, tsize=9, ssize=8.5)
box(ax, 3.05, 5.0, 1.45, 0.55, "IoT gateway", fc="#ffffff", ec=INK2, tsize=9)
arrow(ax, (3.05, 5.275), (2.45, 5.275), color=INK2, lw=1.2)
ax.plot([3.775, 3.775], [5.0, 4.72], ls=(0, (3, 2)), color=BLUE, lw=1.1)
ax.add_patch(FancyBboxPatch((0.05, 0.05), W - 0.1, 4.67, boxstyle="round,pad=0,rounding_size=0.08", fc="none",
                            ec=MUTED, lw=0.9, ls=(0, (4, 3))))
LAYERS = [  # (name, colour, fill, rows of items)
    ("Perceptual layer", BLUE, "#eef5fd", [["RFID", "Sensors", "Card reader", "Robot"],
                                            ["Programmable logic controllers (PLC)", "Meter"],
                                            ["Camera", "Tracking devices", "Facial recognition devices"]]),
    ("Network layer", AQUA, "#ebf8f2", [["MQTT", "LAN", "WLAN", "Modbus"], ["LoRaWAN", "ZigBee", "Internet"]]),
    ("Application layer", ORANGE, "#fdf1ea", [["Smart houses", "Smart logistics", "Smart grids"],
                                              ["Smart manufacturing", "HMI (human-machine interface)"]]),
]
x0, x1 = 0.22, W - 0.22
ytop = 4.52
GAP = 0.42
heights = [1.42, 1.05, 1.05]
centers = []
for (name, col, fill, rows), h in zip(LAYERS, heights):
    y = ytop - h
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=col, lw=1.1))
    tw = 0.13 + 0.075 * len(name)
    ax.add_patch(FancyBboxPatch((x0 + 0.12, ytop - 0.13), tw, 0.26, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=col, ec=col, lw=0))
    ax.text(x0 + 0.12 + tw / 2, ytop, name, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    rh, rg = 0.27, 0.08
    yr = ytop - 0.30 - rh
    for row in rows:
        widths = [0.16 + 0.062 * len(t) for t in row]
        gap = 0.14
        xx = (x0 + x1) / 2 - (sum(widths) + gap * (len(row) - 1)) / 2
        for t, w_ in zip(row, widths):
            ax.add_patch(FancyBboxPatch((xx, yr), w_, rh, boxstyle="round,pad=0,rounding_size=0.04", fc="white",
                                        ec=col, lw=0.9))
            ax.text(xx + w_ / 2, yr + rh / 2, t, ha="center", va="center", fontsize=8.5, color=INK)
            xx += w_ + gap
        yr -= rh + rg
    centers.append((y, ytop))
    ytop = y - GAP
# arrows between layers: upward data flow (left), downward control flow (right)
for (ylo_up, _), (_, yhi_dn) in zip(centers[:-1], centers[1:]):
    arrow(ax, (2.15, yhi_dn + 0.04), (2.15, ylo_up - 0.04), color=INK2, lw=2.2, ms=14)
    arrow(ax, (2.85, ylo_up - 0.04), (2.85, yhi_dn + 0.04), color=BLUE, lw=2.2, ms=14)
M.save(fig, "fig01")
