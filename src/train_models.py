"""
Train and evaluate the rule-based threshold baseline, Isolation Forest and
XGBoost on the same chronological train/test split. Every run is logged in MLflow.

Usage:
    python src/train_models.py
Results:
    results/results_models.csv     one row per model and seed
    results/summary_models.csv     mean and standard deviation per model
    results/recall_by_type.csv     how much of each anomaly type each model catches
"""

import os
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from features import FEATURES, load, split

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]
OUT = "results"


def evaluate(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, scores),
        "fpr": fp / (fp + tn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_baseline(train, test, seed):
    """Industry-style static rules: alert when hourly cost is 50% above the
    7-day average, or when CPU drops below 10% of its normal level."""
    cost_rule = test["cost_usd_ratio"] > 1.5
    idle_rule = test["cpu_util_ratio"] < 0.10
    pred = (cost_rule | idle_rule).astype(int).values
    scores = np.maximum(test["cost_usd_ratio"] / 1.5, 0.10 / (test["cpu_util_ratio"] + 1e-3)).values
    return pred, scores, {"cost_ratio_limit": 1.5, "cpu_ratio_limit": 0.10}


def isolation_forest(train, test, seed):
    """Unsupervised: learns what normal looks like and never sees the labels."""
    contamination = float(train["is_anomaly"].mean())  # expected share of anomalies
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1)
    model.fit(train[FEATURES])
    pred = (model.predict(test[FEATURES]) == -1).astype(int)
    scores = -model.score_samples(test[FEATURES])  # higher = more anomalous
    return pred, scores, {"n_estimators": 200, "contamination": round(contamination, 4)}


def xgboost(train, test, seed):
    """Supervised: learns from labelled examples of normal and anomalous hours."""
    y = train["is_anomaly"]
    weight = (y == 0).sum() / max((y == 1).sum(), 1)  # balances the rare anomaly class
    params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "scale_pos_weight": round(weight, 2)}
    model = XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8, random_state=seed,
                          eval_metric="logloss", n_jobs=-1)
    model.fit(train[FEATURES], y)
    scores = model.predict_proba(test[FEATURES])[:, 1]
    pred = (scores >= 0.5).astype(int)
    return pred, scores, params


MODELS = {
    "Threshold baseline": (threshold_baseline, [0]),  # deterministic, one run is enough
    "Isolation Forest": (isolation_forest, SEEDS),
    "XGBoost": (xgboost, SEEDS),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading data and building features...")
    df = load()
    train, test = split(df)
    print(f"Train: {len(train):,} rows (days 1-63)   Test: {len(test):,} rows (days 64-90)")
    print(f"Anomaly rate  train: {train['is_anomaly'].mean():.2%}   test: {test['is_anomaly'].mean():.2%}\n")

    mlflow.set_experiment("cloud-cost-anomaly")
    rows, by_type = [], []
    for name, (fn, seeds) in MODELS.items():
        for seed in seeds:
            start = time.time()
            pred, scores, params = fn(train, test, seed)
            seconds = time.time() - start
            m = evaluate(test["is_anomaly"].values, pred, scores)
            m.update({"model": name, "seed": seed, "train_test_seconds": round(seconds, 2)})
            rows.append(m)

            for kind in ["spike", "drift", "idle"]:
                mask = (test["anomaly_type"] == kind).values
                by_type.append({"model": name, "seed": seed, "anomaly_type": kind,
                                "recall": pred[mask].mean()})

            with mlflow.start_run(run_name=f"{name} seed {seed}"):
                mlflow.log_params({"model": name, "seed": seed, "features": len(FEATURES), **params})
                mlflow.log_metrics({k: m[k] for k in ["precision", "recall", "f1", "auc_roc", "fpr",
                                                      "train_test_seconds"]})
            print(f"{name:<20} seed {seed}  F1 {m['f1']:.3f}  precision {m['precision']:.3f}  "
                  f"recall {m['recall']:.3f}  FPR {m['fpr']:.3f}  AUC {m['auc_roc']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(f"{OUT}/results_models.csv", index=False)
    summary = results.groupby("model", sort=False)[["precision", "recall", "f1", "auc_roc", "fpr"]].agg(["mean", "std"])
    summary.round(4).to_csv(f"{OUT}/summary_models.csv")
    types = pd.DataFrame(by_type).groupby(["model", "anomaly_type"], sort=False)["recall"].mean().unstack()
    types.round(4).to_csv(f"{OUT}/recall_by_type.csv")

    pd.set_option("display.width", 120)
    print("\nMean over seeds (std in brackets):")
    table = pd.DataFrame({
        k: summary[k]["mean"].map("{:.3f}".format) + " (" + summary[k]["std"].fillna(0).map("{:.3f}".format) + ")"
        for k in ["precision", "recall", "f1", "auc_roc", "fpr"]})
    print(table.to_string())
    print("\nShare of each anomaly type detected (recall):")
    print(types.round(3).to_string())
    print(f"\nSaved results to {OUT}/  |  run 'mlflow ui' to browse the logged runs")


if __name__ == "__main__":
    main()
