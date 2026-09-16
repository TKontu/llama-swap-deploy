"""Chart the DeepSeek-V4-Flash tuning trials measured on the inference box."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "docs/deepseek-v4-flash-trials.png"  # run from the repo root

# (label, tok/s or None for OOM, note)
TRIALS = [
    ("43 in RAM · drafter · ts 1,1", 11.35, "first deployed"),
    ("43 · drafter · no mmap", 10.94, "3-min load"),
    ("41 · drafter · ts 1,1", 10.57, "acceptance 43%"),
    ("39 · no drafter · ts 1.3,1", 12.34, "11.4 / 22.9 GiB"),
    ("39 · no drafter · ts 13,1", 11.76, "23.0 / 10.7 GiB"),
    ("37 · no drafter · ts 8,1", None, "out of memory"),
    ("36 · no drafter · ts 6,1", 12.50, "21.9 / 22.2 GiB"),
]

INK = "#10242a"
INK2 = "#4a666e"
MUTED = "#8aa6ac"
ACCENT = "#0d6a72"
ACCENT_SOFT = "#7fadb1"
CRITICAL = "#a33a26"
GRID = "#d8e3e5"

fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ys = list(range(len(TRIALS)))[::-1]
for y, (label, val, note) in zip(ys, TRIALS):
    shipped = label.startswith("36")
    if val is None:
        ax.add_patch(Rectangle((0, y - 0.3), 8.6, 0.6, fill=False, ls=(0, (3, 3)),
                               ec=CRITICAL, lw=1.1, alpha=.8))
        ax.text(0.25, y, "out of memory — compute buffers, card 1", va="center",
                ha="left", fontsize=9, color=CRITICAL, family="monospace")
        continue
    ax.barh(y, val, height=0.6, color=ACCENT if shipped else ACCENT_SOFT, zorder=3)
    ax.text(val + 0.18, y, f"{val:.2f}", va="center", ha="left", fontsize=10.5,
            color=INK, family="monospace",
            fontweight="bold" if shipped else "normal", zorder=4)
    tag = "  ← shipped" if shipped else ""
    if tag:
        ax.text(val + 1.35, y, tag.strip(), va="center", ha="left", fontsize=9.5,
                color=ACCENT, family="monospace", fontweight="bold")
    ax.text(val - 0.25, y, note, va="center", ha="right", fontsize=8.5,
            color="white" if shipped else "#f4fbfb", family="monospace", zorder=4)

ax.set_yticks(ys)
ax.set_yticklabels([t[0] for t in TRIALS], fontsize=9.5, family="monospace", color=INK2)
ax.set_xlim(0, 15)
ax.set_ylim(-0.7, len(TRIALS) - 0.3)
ax.set_xticks([0, 4, 8, 12])
ax.tick_params(axis="x", colors=MUTED, labelsize=9)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("decode speed — tokens / second (greedy, 300-token completion)",
              fontsize=9.5, color=INK2, labelpad=9)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(GRID)

fig.text(0.035, 0.955, "DeepSeek-V4-Flash (0731)  ·  UD-Q4_K_XL  ·  2 × RTX 3090 + 216 GiB DDR4",
         fontsize=12.5, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.035, 0.902,
         "284B total / 13B active  ·  144.4 GiB of weights  ·  1,048,576-token context  ·  llama.cpp v0.4.0",
         fontsize=8.8, color=MUTED, ha="left", va="top")
fig.text(0.035, 0.862,
         "Left column: layers keeping their experts in system RAM, of 43  ·  ts = layer split across the two cards",
         fontsize=8.8, color=MUTED, ha="left", va="top")

fig.tight_layout(rect=(0, 0, 1, 0.845))
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
