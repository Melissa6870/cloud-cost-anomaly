"""
Full evaluation of the four models (Objective 4, Research Questions 1 and 2).

Run this AFTER train_models.py and train_lstm.py. It:
  1. combines the 5-seed results into one comparison table and chart
  2. draws ROC curves for all four models on the same test set
  3. tests whether the differences between models are statistically significant
     (Wilcoxon signed-rank test on daily F1-scores, Holm-corrected)
  4. measures model drift: train on days 1-20, then test without retraining on
     days 21-30, 31-60 and 61-90

Usage:
    python src/evaluate.py
"""

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from features import load, split
from train_lstm import run as run_lstm
from train_models import evaluate, isolation_forest, threshold_baseline, xgboost

warnings.filterwarnings("ignore")
RES, FIG = "results", "figures"
ORDER = ["Threshold baseline", "Isolation Forest", "XGBoost", "LSTM Autoencoder"]
COLOURS = dict(zip(ORDER, ["tab:gray", "tab:green", "tab:blue", "tab:orange"]))
METRICS = ["precision", "recall", "f1", "auc_roc", "fpr"]


def predict_all(train, test, seed=0, epochs=15):
    """Run every model once; return {model: DataFrame(day, y, pred, score, type)}."""
    out = {}
    for name, fn in [("Threshold baseline", threshold_baseline), ("Isolation Forest", isolation_forest),
                     ("XGBoost", xgboost)]:
        pred, scores, _ = fn(train, test, seed)
        out[name] = pd.DataFrame({"day": test["day"].values, "y": test["is_anomaly"].values,
                                  "pred": pred, "score": scores, "type": test["anomaly_type"].values})
    print("  training LSTM Autoencoder (takes a few minutes)...")
    _, pred, _, _, _, scores, idx = run_lstm(train, test, seed, epochs, verbose=False)
    out["LSTM Autoencoder"] = pd.DataFrame({"day": test.loc[idx, "day"].values, "y": test.loc[idx, "is_anomaly"].values,
                                            "pred": pred, "score": scores, "type": test.loc[idx, "anomaly_type"].values})
    return out


def comparison_table():
    seeds = pd.concat([pd.read_csv(f"{RES}/results_models.csv"), pd.read_csv(f"{RES}/results_lstm.csv")])
    summary = seeds.groupby("model")[METRICS].agg(["mean", "std"]).reindex(ORDER).fillna(0)
    flat = pd.DataFrame({f"{m}_{s}": summary[m][s] for m in METRICS for s in ["mean", "std"]}).round(4)
    flat.to_csv(f"{RES}/final_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.2
    labels = ["Precision", "Recall", "F1-score", "AUC-ROC", "False-positive rate"]
    x = np.arange(len(METRICS))
    for i, model in enumerate(ORDER):
        ax.bar(x + (i - 1.5) * width, summary.loc[model, (METRICS, "mean")].values, width,
               yerr=summary.loc[model, (METRICS, "std")].values, capsize=2, label=model, color=COLOURS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (mean of 5 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection performance on the held-out test set (days 64-90)", pad=28)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_model_comparison.png", dpi=300)
    return summary


def roc_figure(preds):
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in ORDER:
        p = preds[model]
        fpr, tpr, _ = roc_curve(p["y"], p["score"])
        ax.plot(fpr, tpr, label=f"{model} (AUC {roc_auc_score(p['y'], p['score']):.3f})", color=COLOURS[model])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_roc_curves.png", dpi=300)


def significance(preds):
    """Paired Wilcoxon signed-rank test on F1 per test day (each day = one paired sample)."""
    days = sorted(set.intersection(*[set(p.loc[p["day"] > p["day"].min(), "day"]) for p in preds.values()]))
    daily = pd.DataFrame({m: [f1_score(p.loc[p["day"] == d, "y"], p.loc[p["day"] == d, "pred"], zero_division=0)
                              for d in days] for m, p in preds.items()}, index=days)
    daily.to_csv(f"{RES}/daily_f1.csv")
    rows = []
    for a, b in itertools.combinations(ORDER, 2):
        stat, pval = wilcoxon(daily[a], daily[b])
        rows.append({"model_a": a, "model_b": b, "median_f1_a": daily[a].median(), "median_f1_b": daily[b].median(),
                     "W": stat, "p_value": pval, "days": len(days)})
    tests = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    # Holm correction for testing 6 pairs at once
    m = len(tests)
    adjusted = np.maximum.accumulate([(m - i) * p for i, p in enumerate(tests["p_value"])])
    tests["p_holm"] = np.minimum(adjusted, 1)
    tests["significant_0.05"] = tests["p_holm"] < 0.05
    tests.round(5).to_csv(f"{RES}/wilcoxon_tests.csv", index=False)
    return tests


def drift_analysis(df):
    """Train once on days 1-20, then evaluate without retraining on later windows."""
    train = df[df["day"] <= 20].reset_index(drop=True)
    windows = {"Days 21-30": (21, 30), "Days 31-60": (31, 60), "Days 61-90": (61, 90)}
    rows = []
    for label, (lo, hi) in windows.items():
        test = df[(df["day"] >= lo) & (df["day"] <= hi)].reset_index(drop=True)
        print(f"  drift window {label}...")
        preds = predict_all(train, test)
        for model, p in preds.items():
            m = evaluate(p["y"].values, p["pred"].values, p["score"].values)
            rows.append({"window": label, "model": model, **{k: m[k] for k in METRICS}})
    drift = pd.DataFrame(rows)
    first = drift[drift["window"] == "Days 21-30"].set_index("model")
    drift["f1_drop_pct"] = drift.apply(lambda r: 100 * (first.loc[r["model"], "f1"] - r["f1"]) / first.loc[r["model"], "f1"], axis=1)
    drift["auc_drop_pct"] = drift.apply(lambda r: 100 * (first.loc[r["model"], "auc_roc"] - r["auc_roc"]) / first.loc[r["model"], "auc_roc"], axis=1)
    drift.round(4).to_csv(f"{RES}/drift_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in ORDER:
        part = drift[drift["model"] == model]
        ax.plot(part["window"], part["f1"], marker="o", label=model, color=COLOURS[model])
    ax.set_xlabel("Evaluation window (model trained on days 1-20, no retraining)")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model drift as workload patterns change")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_drift.png", dpi=300)
    return drift


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    pd.set_option("display.width", 140)

    print("1/4 Comparison table and chart")
    summary = comparison_table()
    print(summary.round(3).to_string(), "\n")

    print("2/4 ROC curves (one run of every model on the test set)")
    df = load()
    train, test = split(df)
    preds = predict_all(train.reset_index(drop=True), test.reset_index(drop=True))
    roc_figure(preds)

    print("\n3/4 Wilcoxon signed-rank tests on daily F1")
    tests = significance(preds)
    print(tests[["model_a", "model_b", "median_f1_a", "median_f1_b", "p_value", "p_holm", "significant_0.05"]]
          .round(4).to_string(index=False), "\n")

    print("4/4 Drift analysis")
    drift = drift_analysis(df)
    print(drift.pivot(index="model", columns="window", values="f1").reindex(ORDER).round(3).to_string())
    print("\nF1 drop compared with days 21-30 (%):")
    print(drift.pivot(index="model", columns="window", values="f1_drop_pct").reindex(ORDER).round(1).to_string())

    print(f"\nSaved tables to {RES}/ and figures to {FIG}/:")
    print("  final_comparison.csv, wilcoxon_tests.csv, daily_f1.csv, drift_analysis.csv")
    print("  fig_model_comparison.png, fig_roc_curves.png, fig_drift.png")


if __name__ == "__main__":
    main()
