"""Column-width flowcharts for conference paper v4. Run from the project folder: python src/flowcharts_column.py figures"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
out = sys.argv[1]
arrow = dict(arrowstyle="-|>", color="#333", lw=1.1)

# ---------- methodology: two columns, 1-5 down the left, 6-10 up... down the right ----------
fig, ax = plt.subplots(figsize=(3.5, 4.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
steps = [
 ("1. Generate data", "Seed 42: 30 resources,\n90 days, 180 events", "#dbe9f6"),
 ("2. Build features", "19 features compared\nwith previous 7 days", "#dbe9f6"),
 ("3. Split by time", "Train days 1-63,\ntest days 64-90", "#dbe9f6"),
 ("4. Train models", "4 detectors,\n5 seeds each", "#fde9c9"),
 ("5. Score", "Precision, recall,\nF1, FPR, AUC", "#fde9c9"),
 ("6. Significance", "Wilcoxon on daily F1,\nHolm correction", "#fde9c9"),
 ("7. Drift test", "Train days 1-20,\nno retraining", "#e9e1f3"),
 ("8. Real-data check", "NAB AWS\nCloudWatch streams", "#e9e1f3"),
 ("9. Recommend", "Type each incident,\nmap to an action", "#dbefd9"),
 ("10. Savings", "Rand saving for 200,\n1,000, 5,000 staff", "#dbefd9"),
]
W, H, gap = 42, 15.5, 4.3
xs = [3, 55]; ys = [100 - 2 - H - i * (H + gap) for i in range(5)]
pos = []
for i, (t, d, c) in enumerate(steps):
    col = 0 if i < 5 else 1
    x, y = xs[col], ys[i % 5]
    pos.append((x, y))
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.6", fc=c, ec="#333", lw=0.9))
    ax.text(x + W / 2, y + H - 4, t, ha="center", va="center", fontsize=7.6, weight="bold")
    ax.text(x + W / 2, y + 5.2, d, ha="center", va="center", fontsize=6.6, linespacing=1.25)
for i in list(range(4)) + list(range(5, 9)):
    x, y = pos[i]
    ax.annotate("", xy=(x + W / 2, y - gap + 0.8), xytext=(x + W / 2, y - 0.8), arrowprops=arrow)
# 5 -> 6: from bottom-left box across and up to top of right column
x5, y5 = pos[4]; x6, y6 = pos[5]
ax.plot([x5 + W / 2, x5 + W / 2, 50, 50], [y5 - 0.8, y5 - 2.2, y5 - 2.2, y6 + H / 2], color="#333", lw=1.1)
ax.annotate("", xy=(x6 - 0.8, y6 + H / 2), xytext=(50, y6 + H / 2), arrowprops=arrow)
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(f"{out}/fig_method_flow_col.png", dpi=300); plt.close(fig)

# ---------- recommendation decision flow ----------
fig, ax = plt.subplots(figsize=(3.5, 4.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
def box(x, y, w, h, text, c, bold=False, fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=c, ec="#333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.25)
def diamond(cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc="#fff4cc", ec="#333", lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=6.6, linespacing=1.2)
box(2, 86, 40, 11, "Hours flagged\nby the detector", "#dbe9f6", True)
box(56, 86, 42, 11, "Group into an incident\n(gaps of at most 2 h)", "#dbe9f6")
ax.annotate("", xy=(55, 91.5), xytext=(43, 91.5), arrowprops=arrow)
diamond(23, 66, 44, 17, "Median CPU ratio\nbelow 0.25?")
diamond(23, 41, 44, 17, "At most 8 h long and\ncost ratio ≥ 1.8?")
box(55, 59, 43, 14, "IDLE WASTE\nStop or rightsize; move\ndata to cheaper storage", "#dbefd9")
box(55, 34, 43, 14, "SPIKE\nFind runaway process; set\nauto-scaling limit + alert", "#fde2e2")
box(55, 12, 43, 13, "DRIFT\nStop forgotten job; add\nlifecycle or shutdown policy", "#e9e1f3")
ax.plot([77, 77, 23], [85.4, 80, 80], color="#333", lw=1.1); ax.annotate("", xy=(23, 74.8), xytext=(23, 80), arrowprops=arrow)
ax.annotate("", xy=(54.4, 66), xytext=(45.2, 66), arrowprops=arrow); ax.text(46.5, 67.6, "Yes", fontsize=6.6)
ax.annotate("", xy=(23, 49.8), xytext=(23, 57.4), arrowprops=arrow); ax.text(25, 52.5, "No", fontsize=6.6)
ax.annotate("", xy=(54.4, 41), xytext=(45.2, 41), arrowprops=arrow); ax.text(46.5, 42.6, "Yes", fontsize=6.6)
ax.plot([23, 23], [32.4, 18], color="#333", lw=1.1); ax.text(25, 26, "No", fontsize=6.6)
ax.annotate("", xy=(54.4, 18), xytext=(23, 18), arrowprops=arrow)
box(2, 0.8, 96, 6.5, "Each incident then gets an estimated waste and a priority (High, Medium, Low)", "#eeeeee", fs=6.4)
ax.annotate("", xy=(76.5, 7.9), xytext=(76.5, 11.3), arrowprops=arrow)
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(f"{out}/fig_decision_flow_col.png", dpi=300); plt.close(fig)
