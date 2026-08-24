"""Generate README chart PNGs (Lilly palette) -> docs/assets/."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, ACCENT, GREEN, RED, AMBER, MUTED, HAIR = (
    "#1C2430", "#3A5BC7", "#1F9D66", "#C93B3B", "#C77E14", "#67707D", "#E4E8EE")

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "docs", "assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": HAIR, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---- Chart 1: version progress (grouped bars) -------------------------------
versions = ["v1.0\nLR + TF-IDF", "v1.1\nDistilBERT", "v1.2\nLabel science",
            "v2.0\nτ-gate (committed)"]
accuracy = [0.85, 0.8547, 0.8545, 0.9118]
macro_f1 = [0.62, 0.6952, 0.6819, 0.7148]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(versions))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], accuracy, w, label="Accuracy", color=ACCENT)
b2 = ax.bar([i + w / 2 for i in x], macro_f1, w, label="Macro-F1", color=AMBER)
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.3f}".lstrip("0"),
                    (b.get_x() + b.get_width() / 2, b.get_height() + 0.012),
                    ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(versions, fontsize=10)
ax.set_ylim(0, 1.05); ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_yticklabels(["0", ".25", ".50", ".75", "1.00"])
ax.set_ylabel("Full test set (n = 4,529)")
ax.set_title("Lilly · version progress — v1.0 → v2.0", fontsize=14, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
ax.text(0.99, -0.16, "v1.0 classical numbers ≈ (pre-artifact era) · v2.0 = committed bucket @ 86.2% coverage",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "version_progress.png"), dpi=150)
plt.close(fig)

# ---- Chart 2: risk-coverage curve -------------------------------------------
cov = [100, 86.2, 77.8, 72.4]
acc = [85.5, 91.2, 95.1, 96.7]
labels = ["no gate", "τ=.70 ★ deployed", "τ=.80", "τ=.90"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cov, acc, "-o", color=ACCENT, linewidth=2.5, markersize=9, zorder=3)
offsets = [(-8, -16), (10, 6), (10, 6), (-30, 8)]
for (cx, cy), lab, (dx, dy) in zip(zip(cov, acc), labels, offsets):
    star = lab.startswith("τ=.70")
    ax.annotate(lab,
                (cx, cy), textcoords="offset points", xytext=(dx, dy),
                fontsize=10, fontweight="bold" if star else "normal",
                color=GREEN if star else MUTED)
ax.scatter([86.2], [91.2], s=340, facecolors="none", edgecolors=GREEN, linewidths=2.5, zorder=4)
ax.set_xlabel("Coverage  (% of reviews Lilly rates)"); ax.set_xlim(68, 104)
ax.set_ylabel("Committed accuracy (%)"); ax.set_ylim(83, 99)
ax.set_title("Risk–coverage: the τ-gate trade-off", fontsize=14, fontweight="bold")
ax.text(0.99, -0.16, "every point right of τ=.70 buys coverage by spending accuracy — the gate makes the trade explicit",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "risk_coverage.png"), dpi=150)
plt.close(fig)

# ---- Chart 3: human-gold subset (label science payoff) ----------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
models = ["v1.1 baseline", "v1.2 relabeled"]
gold = [0.3665, 0.3941]
bars = ax.bar(models, gold, width=0.45, color=[MUTED, GREEN])
for b in bars:
    ax.annotate(f"{b.get_height():.4f}".lstrip("0"),
                (b.get_x() + b.get_width() / 2, b.get_height() + 0.006),
                ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.5)
ax.set_ylabel("Macro-F1 on human-labeled\nrating-3 subset (n = 199)")
ax.set_title("What relabeling actually bought (human gold)", fontsize=13, fontweight="bold")
ax.text(0.99, -0.2, "modest but real — and honest: the κ ≈ 0.15 annotation ceiling bounds this metric",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gold_subset.png"), dpi=150)
plt.close(fig)

print("charts written to", OUT)
