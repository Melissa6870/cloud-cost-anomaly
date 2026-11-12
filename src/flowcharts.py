"""Draws the two flowcharts used in the conference paper: figures/fig_method_flow.png and figures/fig_decision_flow.png."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

os.makedirs("figures", exist_ok=True)
arrow = dict(arrowstyle="-|>", color="#333", lw=1.2)

# ---------- Experimental methodology flowchart ----------
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
steps = [
 ("1. Generate data", "Seed 42: 30 resources,\n90 days, 180 events", "#dbe9f6"),
 ("2. Build features", "19 features vs the\nprevious 7 days", "#dbe9f6"),
 ("3. Split by time", "Train days 1-63\nTest days 64-90", "#dbe9f6"),
 ("4. Train models", "4 detectors,\n5 seeds each", "#fde9c9"),
 ("5. Score", "Precision, recall,\nF1, FPR, AUC", "#fde9c9"),
 ("6. Significance", "Wilcoxon on daily\nF1, Holm-corrected", "#fde9c9"),
 ("7. Drift test", "Train days 1-20,\nno retraining", "#e9e1f3"),
 ("8. Real-data check", "17 NAB AWS\nCloudWatch streams", "#e9e1f3"),
 ("9. Recommend", "Type each incident,\nmap to an action", "#dbefd9"),
 ("10. Savings", "Rand saving for 200,\n1,000, 5,000 staff", "#dbefd9"),
]
W, H = 16, 15
xs = [1 + i * 20 for i in range(5)]
for i, (t, d, c) in enumerate(steps):
    row = 0 if i < 5 else 1
    x = xs[i] if row == 0 else xs[9 - i]
    y = 26 if row == 0 else 3
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.4", fc=c, ec="#333", lw=1))
    ax.text(x + W / 2, y + H - 3.5, t, ha="center", va="center", fontsize=8.3, weight="bold")
    ax.text(x + W / 2, y + 5.5, d, ha="center", va="center", fontsize=7, linespacing=1.3)
for i in range(4):
    ax.annotate("", xy=(xs[i + 1] - 0.6, 33.5), xytext=(xs[i] + W + 0.6, 33.5), arrowprops=arrow)
    ax.annotate("", xy=(xs[i] + W + 0.6, 10.5), xytext=(xs[i + 1] - 0.6, 10.5), arrowprops=arrow)
ax.annotate("", xy=(xs[4] + W / 2, 18.6), xytext=(xs[4] + W / 2, 25.4), arrowprops=arrow)
fig.tight_layout(); fig.savefig("figures/fig_method_flow.png", dpi=300); plt.close(fig)

# ---------- Recommendation decision flowchart ----------
fig, ax = plt.subplots(figsize=(7.2, 4.1)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-9, 54)
def box(x, y, w, h, text, c, bold=False, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35", fc=c, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.3)
def diamond(cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc="#fff4cc", ec="#333", lw=1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7, linespacing=1.25)
box(1, 41, 17, 10, "Hours flagged\nby the detector", "#dbe9f6", True)
box(23, 41, 18, 10, "Group into an\nincident (gaps ≤ 2 h)", "#dbe9f6")
diamond(55, 46, 22, 12, "Median CPU\nratio < 0.25?")
diamond(55, 25, 22, 13, "≤ 8 h long and\ncost ratio ≥ 1.8?")
box(72, 41, 25, 10, "IDLE WASTE\nStop or rightsize; move\ndata to cheaper storage", "#dbefd9")
box(72, 20, 25, 10, "SPIKE\nFind runaway process; set\nauto-scaling limit + alert", "#fde2e2")
box(72, 1, 25, 10, "DRIFT\nStop forgotten job; add\nlifecycle or shutdown policy", "#e9e1f3")
box(1, -8, 96, 5.5, "Every incident then gets an estimated waste (cost above normal, or the full cost when idle) and a priority: High, Medium or Low", "#eeeeee", fs=7.2)
ax.annotate("", xy=(22.6, 46), xytext=(18.4, 46), arrowprops=arrow)
ax.annotate("", xy=(43.8, 46), xytext=(41.4, 46), arrowprops=arrow)
ax.annotate("", xy=(71.6, 46), xytext=(66.2, 46), arrowprops=arrow); ax.text(68, 47.3, "Yes", fontsize=7)
ax.annotate("", xy=(55, 31.7), xytext=(55, 39.8), arrowprops=arrow); ax.text(56, 35.5, "No", fontsize=7)
ax.annotate("", xy=(71.6, 25), xytext=(66.2, 25), arrowprops=arrow); ax.text(68, 26.3, "Yes", fontsize=7)
ax.plot([55, 55], [18.4, 6], color="#333", lw=1.2); ax.text(56, 12, "No", fontsize=7)
ax.annotate("", xy=(71.6, 6), xytext=(55, 6), arrowprops=arrow)
ax.annotate("", xy=(84.5, -2.2), xytext=(84.5, 0.6), arrowprops=arrow)
fig.tight_layout(); fig.savefig("figures/fig_decision_flow.png", dpi=300); plt.close(fig)
