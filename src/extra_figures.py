"""
Extra figures for the thesis, built from the result files.

    fig_confusion.png           confusion matrices of the four models (mean of five seeds)
    fig_confusion_2x2.png       the same in a 2 x 2 layout for the conference paper
    fig_feature_importance.png  the features XGBoost relies on most
    fig_daily_f1.png            F1-score of each model on every test day
    fig_waste_by_type.png       estimated waste flagged by the recommendation engine
    fig_storage_use.png         storage used by each staff member against the 1 TB quota
    fig_rightsizing.png         busy-hour CPU of every server and the action taken

Run after train_models.py, train_lstm.py, recommend.py and savings.py:
    python src/extra_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import savings
from features import FEATURES, load, split
from xgboost import XGBClassifier

ORDER = ["Threshold baseline", "Isolation Forest", "XGBoost", "LSTM Autoencoder"]
COLOURS = {"Threshold baseline": "#7f7f7f", "Isolation Forest": "#2ca02c", "XGBoost": "#1f77b4", "LSTM Autoencoder": "#ff7f0e"}
TYPE_COLOURS = {"spike": "#d62728", "drift": "#9467bd", "idle": "#8c564b"}


def tidy(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def confusion(layout=(1, 4), size=(12, 3.3), out="figures/fig_confusion.png"):
    runs = pd.concat([pd.read_csv("results/results_models.csv"), pd.read_csv("results/results_lstm.csv")])
    mean = runs.groupby("model")[["tn", "fp", "fn", "tp"]].mean()
    fig, axes = plt.subplots(*layout, figsize=size)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, ORDER):
        r = mean.loc[m]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        share = cm / cm.sum(axis=1, keepdims=True)   # colour by share of each true class
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,.0f}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_title(m, fontsize=10)
    for ax in axes[::layout[1]]:
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def feature_importance():
    train, test = split(load())
    # Same settings as train_models.py (seed 0)
    y = train["is_anomaly"]
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=round((y == 0).sum() / max((y == 1).sum(), 1), 2),
                          subsample=0.8, colsample_bytree=0.8, random_state=0, eval_metric="logloss", n_jobs=-1)
    model.fit(train[FEATURES], y)
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    gain = (gain / gain.sum()).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(gain.index, gain.values * 100, color="#1f77b4", height=0.6)
    for y, v in enumerate(gain.values * 100):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("XGBoost: the 12 most important features (seed 0)")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/fig_feature_importance.png", dpi=300)
    plt.close(fig)
    gain.sort_values(ascending=False).round(4).to_csv("results/feature_importance.csv", header=["gain_share"])


def daily_f1():
    d = pd.read_csv("results/daily_f1.csv", index_col=0)
    # A day without any labelled anomaly has no defined F1 (every model scores 0), so it is left out
    labels = pd.read_csv("results/detections.csv").groupby("day")["is_anomaly"].sum()
    d = d[labels.reindex(d.index).fillna(0) > 0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for m in ORDER:
        ax.plot(d.index, d[m], marker="o", markersize=3, lw=1.5, color=COLOURS[m], label=m)
    ax.set_xlabel("Test day")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Daily F1-score on the test period")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/fig_daily_f1.png", dpi=300)
    plt.close(fig)


def waste_by_type(rate=savings.USD_TO_ZAR):
    rec = pd.read_csv("results/recommendations.csv")
    t = rec.groupby(["detected_type", "priority"])["estimated_waste_usd"].sum().unstack(fill_value=0) * rate
    t = t.reindex(["drift", "spike", "idle"])[["High", "Medium", "Low"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    left = np.zeros(len(t))
    for p, c in zip(t.columns, ["#08519c", "#4292c6", "#9ecae1"]):
        ax.barh(t.index.str.capitalize(), t[p], left=left, color=c, label=f"{p} priority", height=0.55, edgecolor="white", linewidth=1)
        left += t[p].values
    for y, v in enumerate(left):
        ax.text(v + 20, y, f"R{v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, left.max() * 1.2)
    ax.set_xlabel("Estimated waste over the 27-day test period (Rand)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.55))
    ax.set_title("Waste flagged by the recommendation engine, by type and priority")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/fig_waste_by_type.png", dpi=300)
    plt.close(fig)


def storage_and_servers(staff=1000):
    _, storage, servers = savings.company_savings(staff)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(storage["years_of_service"], storage["used_gb"], s=8, alpha=0.6, color="#1f77b4", label="Storage used by one staff member")
    ax.axhline(savings.QUOTA_GB, color="#d62728", lw=1.5, ls="--", label="Quota paid for (1 TB)")
    ax.set_xlabel("Years of service")
    ax.set_ylabel("Storage (GB)")
    ax.set_title(f"Storage used against the 1 TB quota ({staff:,} staff)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/fig_storage_use.png", dpi=300)
    plt.close(fig)

    s = servers.sort_values("p95_cpu").reset_index(drop=True)
    kind = np.where(s["recommended_size"] == "stopped", "Stop", np.where(s["action"] == "Keep", "Keep", "Downsize"))
    colours = {"Stop": "#d62728", "Downsize": "#ff7f0e", "Keep": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k in ["Stop", "Downsize", "Keep"]:
        idx = np.where(kind == k)[0]
        ax.bar(idx, s.loc[idx, "p95_cpu"] * 100, color=colours[k], label=f"{k} ({len(idx)})", width=0.7)
    ax.axhline(savings.TARGET_CPU * 100, color="black", lw=1, ls="--")
    ax.text(len(s) - 0.5, savings.TARGET_CPU * 100 + 2, "Target 70%", ha="right", fontsize=8)
    ax.set_xticks(range(len(s)), s["server_id"], rotation=90, fontsize=7)
    ax.set_ylabel("Busy-hour CPU (p95, %)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Server rightsizing decisions ({staff:,} staff)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/fig_rightsizing.png", dpi=300)
    plt.close(fig)

    tiers = storage["new_quota_gb"].value_counts().sort_index()
    tiers.rename("staff").to_csv("results/storage_tiers.csv")
    s.to_csv("results/server_rightsizing.csv", index=False)


if __name__ == "__main__":
    confusion()
    confusion((2, 2), (6.4, 5.6), "figures/fig_confusion_2x2.png")   # compact version for the paper
    feature_importance()
    daily_f1()
    waste_by_type()
    storage_and_servers()
    print("Saved the extra figures in figures/")
