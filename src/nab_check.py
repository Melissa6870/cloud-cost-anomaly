"""
Real-data check on the Numenta Anomaly Benchmark (NAB), realAWSCloudwatch set.

NAB contains 17 real AWS CloudWatch metric streams (EC2 CPU, disk, network,
ELB, RDS) recorded every 5 minutes, with anomaly windows labelled by NAB.
This tests whether the four approaches still work on real telemetry
(the role Azure validation had in the proposal).

How each model is applied:
  Threshold baseline  same rules as before, on the metric's ratio to its last day
  Isolation Forest    trained without labels on the first 30% of each stream
  LSTM Autoencoder    trained without labels on the first 30% of each stream
  XGBoost             needs labels, so it is trained on 16 streams and tested on
                      the 17th, repeated for every stream (leave-one-stream-out)

Usage (after cloning NAB into data/raw/NAB):
    python src/nab_check.py
"""

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier

from train_lstm import LSTMAutoencoder
from train_models import evaluate

warnings.filterwarnings("ignore")
NAB = "data/raw/NAB"
DAY = 288  # 5-minute points per day
FEATS = ["value", "z", "ratio", "diff", "roll_std_ratio"]
ORDER = ["Threshold baseline", "Isolation Forest", "XGBoost", "LSTM Autoencoder"]


def load_streams():
    labels = json.load(open(f"{NAB}/labels/combined_windows.json"))
    streams = {}
    for path in sorted(glob.glob(f"{NAB}/data/realAWSCloudwatch/*.csv")):
        key = "realAWSCloudwatch/" + os.path.basename(path)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["y"] = 0
        for start, end in labels.get(key, []):
            df.loc[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end)), "y"] = 1
        # Same idea as the synthetic features: compare each point with the previous day
        prev = df["value"].shift(1)
        mean = prev.rolling(DAY, min_periods=48).mean()
        std = prev.rolling(DAY, min_periods=48).std()
        df["z"] = (df["value"] - mean) / (std + 1e-6)
        df["ratio"] = df["value"] / (mean.abs() + 1e-6)
        df["diff"] = df["value"].diff() / (std + 1e-6)
        df["roll_std_ratio"] = df["value"].rolling(12).std() / (std + 1e-6)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).reset_index(drop=True)
        streams[os.path.basename(path).replace(".csv", "")] = df
    return streams


def lstm_scores(train_x, test_x, seed=0, window=12, epochs=5):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train_x)
    def windows(a):
        a = np.clip(scaler.transform(a), -10, 10).astype(np.float32)
        return torch.tensor(np.stack([a[i - window + 1:i + 1] for i in range(window - 1, len(a))]))
    xtr, xte = windows(train_x), windows(test_x)
    model = LSTMAutoencoder(train_x.shape[1], hidden=32, latent=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 256):
            xb = xtr[perm[i:i + 256]]
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        err_tr = ((model(xtr) - xtr) ** 2)[:, -3:, :].mean(dim=(1, 2)).numpy()
        err_te = ((model(xte) - xte) ** 2)[:, -3:, :].mean(dim=(1, 2)).numpy()
    # first window-1 test points have no full window: give them the median score
    pad = np.full(window - 1, np.median(err_te))
    return err_tr, np.concatenate([pad, err_te])


def main():
    streams = load_streams()
    print(f"Loaded {len(streams)} NAB realAWSCloudwatch streams, "
          f"{sum(len(d) for d in streams.values()):,} points, "
          f"{np.mean([d['y'].mean() for d in streams.values()]):.1%} inside labelled anomaly windows\n")

    pooled = {m: {"y": [], "pred": [], "score": []} for m in ORDER}
    per_stream = []
    for name, df in streams.items():
        cut = int(len(df) * 0.3)
        train, test = df.iloc[:cut], df.iloc[cut:]
        if test["y"].sum() == 0:
            continue
        rate = max(train["y"].mean(), 0.05)
        out = {}

        # Threshold baseline: same rules as on the synthetic data
        score = np.maximum(test["ratio"] / 1.5, 0.10 / (test["ratio"].abs() + 1e-3))
        out["Threshold baseline"] = ((score > 1).astype(int).values, score.values)

        iso = IsolationForest(n_estimators=200, contamination=min(rate, 0.5), random_state=0).fit(train[FEATS])
        out["Isolation Forest"] = ((iso.predict(test[FEATS]) == -1).astype(int), -iso.score_samples(test[FEATS]))

        others = pd.concat([d for n, d in streams.items() if n != name])
        weight = (others["y"] == 0).sum() / max(others["y"].sum(), 1)
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, scale_pos_weight=weight,
                            random_state=0, eval_metric="logloss").fit(others[FEATS], others["y"])
        p = xgb.predict_proba(test[FEATS])[:, 1]
        out["XGBoost"] = ((p >= 0.5).astype(int), p)

        err_tr, err_te = lstm_scores(train[FEATS].values, test[FEATS].values)
        out["LSTM Autoencoder"] = ((err_te > np.quantile(err_tr, 1 - rate)).astype(int), err_te)

        y = test["y"].values
        for model, (pred, sc) in out.items():
            m = evaluate(y, pred, sc)
            per_stream.append({"stream": name, "model": model, **{k: m[k] for k in ["precision", "recall", "f1", "auc_roc", "fpr"]}})
            pooled[model]["y"].append(y)
            pooled[model]["pred"].append(pred)
            pooled[model]["score"].append(sc)
        print(f"  {name:<40} " + "  ".join(f"{m.split()[0]} F1 {evaluate(y, *out[m])['f1']:.2f}" for m in ORDER))

    per_stream = pd.DataFrame(per_stream)
    os.makedirs("results", exist_ok=True)
    per_stream.round(4).to_csv("results/nab_per_stream.csv", index=False)
    summary = per_stream.groupby("model")[["precision", "recall", "f1", "auc_roc", "fpr"]].mean().reindex(ORDER)
    summary.round(4).to_csv("results/nab_summary.csv")
    print("\nMean over NAB streams:")
    print(summary.round(3).to_string())
    print("\nSaved results/nab_per_stream.csv and results/nab_summary.csv")


if __name__ == "__main__":
    main()
