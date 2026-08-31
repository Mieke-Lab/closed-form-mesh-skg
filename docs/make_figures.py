"""
make_figures.py
---------------
Regenerates the two README figures:

    docs/figures/fig1_system_architecture.png
    docs/figures/fig2_channel_probing.png

Everything is drawn from the constants below, so the figures stay in sync
with the pipeline by editing this file — no dataset is read.

Usage:
    python3 docs/make_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b1f24"
MUTED = "#5b6672"
ALICE = "#2f6f9f"
BOB = "#2e8b6b"
CHARLIE = "#b8791f"
EVE = "#a63a3a"
SHARED = "#6a5acd"


def box(ax, x, y, w, h, label, *, face, edge=None, fontsize=8.5, weight="normal",
        text_color=None, radius=0.055):
    """Rounded box with a centred label."""
    edge = edge or face
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.1, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color or INK, weight=weight,
            zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, *, color=MUTED, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def tint(hex_color, amount=0.86):
    """Blend a colour toward white to get a readable fill."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: (c + (255 - c) * amount) / 255
    return (mix(r), mix(g), mix(b))


# --------------------------------------------------------------------------
# Figure 1 — system architecture
# --------------------------------------------------------------------------
def figure_architecture():
    fig, ax = plt.subplots(figsize=(12.6, 8.4), dpi=200)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    ax.text(6.3, 8.24,
            "Closed-Form Group Key Generation for 802.11s Mesh "
            "under Pilot Contamination",
            ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(6.3, 7.94,
            "System architecture — three-node group SKG pipeline "
            "(Alice coordinator, Bob and Charlie responders)",
            ha="center", fontsize=9.5, color=MUTED)

    # Column geometry: three lanes with a gap wide enough for the exchange labels.
    lanes = [(0.30, ALICE, "ALICE  (coordinator)"),
             (4.50, BOB, "BOB  (responder)"),
             (8.70, CHARLIE, "CHARLIE  (responder)")]
    LW = 3.60
    ROWS = {1: 5.90, 2: 5.10, 3: 4.30, 4: 3.50, 5: 2.70}
    BH = 0.62

    # ---- Layer 0: acquisition -------------------------------------------
    box(ax, 0.30, 7.02, 12.0, 0.70,
        "LAYER 0  ·  CHANNEL PROBING\n"
        "802.11s mesh (batman-adv)  ·  bidirectional UDP probes  ·  "
        "per-packet RSSI from radiotap (tshark on mon0)",
        face=tint(SHARED, 0.9), edge=SHARED, fontsize=9, weight="bold")

    for x, color, name in lanes:
        box(ax, x, 6.44, LW, 0.42, name, face=tint(color, 0.78), edge=color,
            fontsize=9.5, weight="bold")
        arrow(ax, (x + LW / 2, 7.02), (x + LW / 2, 6.86), color=color)

    stages = {
        ALICE: {
            1: "00  Session seed\nSHA-256 per-session entropy",
            2: "01  Distribution adjustment\nMTD-DOSS filter + BKRR Kalman AR(1) fill",
            3: "02  Channel obfuscation\ncirculant projection  $b_A = P_A\\,\\tilde{x}_A$",
            4: "03  Key encoding\ngroup key $K$ + BCH(63,30,t=6) encapsulation",
            5: "06  Reconciliation join\ncollect peer error indices, correct $b_A$",
        },
        BOB: {
            1: "04  Preparation\nlocal DOSS + reciprocity guard",
            2: "05  TRSD deobfuscation\nclosed-form Tikhonov spectral inverse (FFT)",
            4: "06  Reconciliation\nBCH decode + trial-flip (5-bit) recovery",
            5: "07  BMR evaluation\npre / post error rate vs. Alice",
        },
        CHARLIE: {
            1: "04  Preparation\nlocal DOSS + reciprocity guard",
            2: "05  TRSD deobfuscation\nclosed-form Tikhonov spectral inverse (FFT)",
            4: "06  Reconciliation\nBCH decode + trial-flip (5-bit) recovery",
            5: "07  BMR evaluation\npre / post error rate vs. Alice",
        },
    }

    for x, color, _ in lanes:
        rows = sorted(stages[color])
        for r in rows:
            box(ax, x, ROWS[r], LW, BH, stages[color][r],
                face=tint(color), edge=color, fontsize=8)
        arrow(ax, (x + LW / 2, 6.44), (x + LW / 2, ROWS[rows[0]] + BH), color=color)
        for a, b in zip(rows, rows[1:]):
            arrow(ax, (x + LW / 2, ROWS[a]), (x + LW / 2, ROWS[b] + BH), color=color)

    # ---- Cross-node exchanges, drawn in the lane gaps --------------------
    gap_ab, gap_bc = 4.05, 8.25          # centres of the two gaps
    y4 = ROWS[4] + BH / 2
    y5 = ROWS[5] + BH / 2

    arrow(ax, (3.90, y4), (4.50, y4), color=ALICE, lw=1.4)
    ax.text(gap_ab, y4 + 0.30, "$b_A$ +\nBCH synd.", ha="center", va="center",
            fontsize=6.8, color=ALICE, linespacing=1.3)

    arrow(ax, (8.10, y4), (8.70, y4), color=SHARED, style="<|-|>", lw=1.4)
    ax.text(gap_bc, y4 + 0.30, "CARR\nrelay $H_{AC}$", ha="center", va="center",
            fontsize=6.8, color=SHARED, linespacing=1.3)

    arrow(ax, (4.50, y5), (3.90, y5), color=BOB, lw=1.4, ls="--")
    ax.text(gap_ab, y5 + 0.30, "error indices\n(Bob, Charlie)", ha="center",
            va="center", fontsize=6.8, color=BOB, linespacing=1.3)
    arrow(ax, (8.70, y5), (8.10, y5), color=CHARLIE, lw=1.4, ls="--")
    ax.text(gap_bc, y5 + 0.30, "error\nindices", ha="center", va="center",
            fontsize=6.8, color=CHARLIE, linespacing=1.3)

    # ---- Layer: privacy amplification -----------------------------------
    box(ax, 0.30, 1.72, 12.0, 0.62,
        "08  PRIVACY AMPLIFICATION   —   session-seeded Toeplitz hash   →   "
        "identical 256-bit group key at Alice, Bob and Charlie",
        face=tint(SHARED, 0.88), edge=SHARED, fontsize=9, weight="bold")
    for x, _, _ in lanes:
        arrow(ax, (x + LW / 2, ROWS[5]), (x + LW / 2, 2.34), color=MUTED)

    # ---- Validation row --------------------------------------------------
    validation = [
        (0.30, 2.85, "10  NIST SP 800-22\n8 randomness tests", INK),
        (3.35, 2.85, "11  Eve analysis\npassive eavesdrop, key mismatch", EVE),
        (6.40, 2.85, "12  Paper metrics\nKDR · KGR · BMR · runtime", INK),
        (9.45, 2.85, "13–18  Verifiers\ntables, ablation, TRSD-vs-ADMM,\nmin-entropy (LHL)", INK),
    ]
    for x, w, label, color in validation:
        box(ax, x, 0.62, w, 0.76, label, face=tint(color, 0.93), edge=color,
            fontsize=7.6)
    arrow(ax, (6.3, 1.72), (6.3, 1.38), color=MUTED)

    ax.text(6.3, 0.20,
            "The key is never extracted from the channel: Alice pre-generates it "
            "and hides it behind a circulant projection, so the channel acts as "
            "bearer rather than entropy source.",
            ha="center", fontsize=8.2, color=MUTED, style="italic")

    path = OUT / "fig1_system_architecture.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2 — channel probing mechanism
# --------------------------------------------------------------------------
def figure_probing():
    fig = plt.figure(figsize=(12.4, 6.6), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.06,
                          left=0.03, right=0.97, top=0.86, bottom=0.08)

    fig.text(0.5, 0.955, "Channel probing mechanism — simultaneous "
                         "three-link mesh acquisition",
             ha="center", fontsize=13, weight="bold", color=INK)
    fig.text(0.5, 0.915,
             "Every node pair measures the same link in both directions; "
             "reciprocity of the paired RSSI is what key generation rests on.",
             ha="center", fontsize=9, color=MUTED)

    # ---- Left panel: topology -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    nodes = {
        "Alice": (5.0, 8.1, ALICE, "10.0.0.1"),
        "Bob": (1.9, 3.6, BOB, "10.0.0.2"),
        "Charlie": (8.1, 3.6, CHARLIE, "10.0.0.3"),
    }
    links = [
        ("Alice", "Bob", "AB  ·  UDP 9001", "Alice = coordinator"),
        ("Alice", "Charlie", "AC  ·  UDP 9002", "Alice = coordinator"),
        ("Bob", "Charlie", "BC  ·  UDP 9003", "Bob = coordinator"),
    ]
    for a, b, port, role in links:
        (xa, ya, *_), (xb, yb, *_) = nodes[a], nodes[b]
        # Pull the endpoints clear of the node circles so both arrowheads show.
        dx, dy = xb - xa, yb - ya
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ux, uy = dx / dist, dy / dist
        arrow(ax, (xa + ux * 0.95, ya + uy * 0.95),
              (xb - ux * 0.95, yb - uy * 0.95),
              color=SHARED, style="<|-|>", lw=1.6)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        off = 0.0 if a == "Bob" else (-1.45 if b == "Bob" else 1.45)
        ax.text(mx + off, my + (0.42 if a == "Bob" else 0.15),
                f"{port}\n{role}", ha="center", va="center",
                fontsize=7.6, color=SHARED, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="none"))

    for name, (x, y, color, ip) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.86, facecolor=tint(color, 0.75),
                                edgecolor=color, linewidth=1.6, zorder=4))
        ax.text(x, y + 0.14, name, ha="center", va="center", fontsize=10,
                weight="bold", color=INK, zorder=5)
        ax.text(x, y - 0.36, ip, ha="center", va="center", fontsize=7.4,
                color=MUTED, zorder=5)

    box(ax, 3.15, 0.35, 3.7, 0.95,
        "EVE  ·  wlan0mon, ch. 6\npassive capture, never transmits",
        face=tint(EVE, 0.9), edge=EVE, fontsize=8)
    for src in ("Bob", "Charlie", "Alice"):
        x, y, *_ = nodes[src]
        arrow(ax, (x, y - 0.9), (5.0, 1.35), color=EVE, ls=":", lw=1.0, rad=0.1)

    ax.text(5.0, 9.65, "802.11s mesh (batman-adv), all links active at once",
            ha="center", fontsize=8.6, color=MUTED, style="italic")

    # ---- Right panel: packet exchange + timeline ------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    ax2.text(0.15, 9.6, "Probe exchange per sample", fontsize=9.6,
             weight="bold", color=INK)
    box(ax2, 0.15, 8.35, 2.5, 0.62, "Coordinator", face=tint(ALICE, 0.8),
        edge=ALICE, fontsize=8.5, weight="bold")
    box(ax2, 6.6, 8.35, 2.5, 0.62, "Reflector", face=tint(BOB, 0.8),
        edge=BOB, fontsize=8.5, weight="bold")
    arrow(ax2, (2.75, 8.52), (6.5, 8.52), color=INK, lw=1.3)
    ax2.text(4.62, 8.72, "probe   !Id?  (seq, t, is_reply)", ha="center",
             fontsize=7.6, color=INK)
    arrow(ax2, (6.5, 8.05), (2.75, 8.05), color=INK, lw=1.3)
    ax2.text(4.62, 7.62, "reply   !IddB  (seq, $t_0$, rssi_local, flag)",
             ha="center", fontsize=7.6, color=INK)
    ax2.text(0.15, 7.0,
             "coordinator CSV:  seq, timestamp, rssi_local, rssi_remote, rtt_ms\n"
             "2000 samples per link  ·  110 ms interval  ·  ~220 s per run",
             fontsize=7.8, color=MUTED, linespacing=1.6)

    ax2.text(0.15, 6.05, "Execution timeline (threads run concurrently)",
             fontsize=9.6, weight="bold", color=INK)
    tracks = [
        (5.05, "Thread 1   Alice -> Bob       (9001)", ALICE),
        (4.25, "Thread 2   Alice -> Charlie   (9002)", ALICE),
        (3.45, "Thread 3   Bob   -> Charlie   (9003)", BOB),
        (2.55, "Eve        tshark wlan0mon (passive)", EVE),
    ]
    x0, x1 = 3.95, 9.1
    for y, label, color in tracks:
        ax2.text(0.15, y + 0.22, label, fontsize=7.5, color=INK, va="center",
                 family="monospace")
        ax2.add_patch(FancyBboxPatch(
            (x0, y), x1 - x0, 0.44,
            boxstyle="round,pad=0.008,rounding_size=0.05",
            facecolor=tint(color, 0.72), edgecolor=color, linewidth=1.0))

    for x, tick, note in ((x0, "t = 2 s", "probing starts"),
                          (x1, "t ~ 220 s", "2000 x 0.11 s")):
        ax2.plot([x, x], [2.15, 5.72], color=MUTED, linewidth=0.8,
                 linestyle=":", zorder=0)
        ax2.text(x, 5.85, tick, ha="center", fontsize=7.6, color=MUTED)
        ax2.text(x, 1.85, note, ha="center", fontsize=7, color=MUTED,
                 style="italic")

    box(ax2, 0.15, 0.35, 8.95, 1.15,
        "t = 0 s     start tshark on mon0 at every node, launch reflectors\n"
        "t ~ 225 s   stop capture  ->  per-packet radiotap RSSI  ->  DOSS series\n"
        "mon0 is added on top of wlan1 without tearing down the mesh interface",
        face="#f4f6f8", edge="#c9d1d9", fontsize=7.8)

    path = OUT / "fig2_channel_probing.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (figure_architecture(), figure_probing()):
        print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
