"""
LSTM Autoencoder for cloud cost anomaly detection.

The model reads each resource's telemetry as sliding 24-hour windows and learns
to reconstruct them. It is trained without labels on the training period, so it
learns what normal behaviour looks like. Hours it cannot reconstruct well (high
reconstruction error) are flagged as anomalies.

Usage:
    python src/train_lstm.py                  # 5 seeds, 15 epochs each
    python src/train_lstm.py --seeds 1        # quick single run
Results:
    results/results_lstm.csv          one row per seed
    figures/fig_lstm_loss.png         training loss curve (first seed)
"""

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from features import load, split
from train_models import evaluate

warnings.filterwarnings("ignore")
WINDOW = 24   # hours per sequence
OUT = "results"


class LSTMAutoencoder(nn.Module):
    """Encoder compresses a 24-hour window into a small vector; the decoder
    rebuilds the window from it. Unusual windows rebuild badly."""

    def __init__(self, n_features, hidden=64, latent=16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])                                   # (batch, latent)
        repeated = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


# Relative features only (each hour compared with the resource's own last 7 days).
# Absolute levels are left out: after the workload shift they would look
# abnormal to a model trained on the earlier period and flood it with false alarms.
LSTM_FEATURES = [
    "cost_usd_z", "cost_usd_ratio", "cpu_util_z", "cpu_util_ratio",
    "network_out_mb_z", "network_out_mb_ratio", "disk_ops_z", "disk_ops_ratio",
    "cost_trend_24h", "cost_per_cpu_ratio",
]


def fit_scalers(train):
    """One scaler per resource: an S3 bucket and a large EC2 instance have very
    different normal cost levels, so each is compared with its own history."""
    return {rid: StandardScaler().fit(g[LSTM_FEATURES]) for rid, g in train.groupby("resource_id")}


def make_windows(df, scalers):
    """Sliding windows per resource. Each window is labelled by its LAST hour,
    so every hour (after the first 23 of a resource) gets exactly one score."""
    xs, idx = [], []
    for rid, g in df.groupby("resource_id"):
        values = np.clip(scalers[rid].transform(g[LSTM_FEATURES]), -10, 10).astype(np.float32)
        for end in range(WINDOW - 1, len(g)):
            xs.append(values[end - WINDOW + 1:end + 1])
            idx.append(g.index[end])
    return torch.tensor(np.stack(xs)), np.array(idx)


def window_errors(model, x, batch=2048):
    """Mean squared reconstruction error of the last 6 hours in each window."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            err = ((model(xb) - xb) ** 2)[:, -6:, :].mean(dim=(1, 2))
            out.append(err.numpy())
    return np.concatenate(out)


def run(train, test, seed, epochs, lr=1e-3, batch=256, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scalers = fit_scalers(train)
    x_train, _ = make_windows(train, scalers)
    x_test, test_idx = make_windows(test, scalers)

    model = LSTMAutoencoder(len(LSTM_FEATURES))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(x_train), batch):
            xb = x_train[perm[i:i + batch]]
            optimiser.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(xb)
        losses.append(total / len(x_train))
        if verbose:
            print(f"  seed {seed}  epoch {epoch + 1:>2}/{epochs}  loss {losses[-1]:.4f}")

    # Threshold: flag the same share of hours as the anomaly rate seen in training
    contamination = float(train["is_anomaly"].mean())
    threshold = np.quantile(window_errors(model, x_train), 1 - contamination)
    scores = window_errors(model, x_test)
    pred = (scores > threshold).astype(int)
    y = test.loc[test_idx, "is_anomaly"].values
    types = test.loc[test_idx, "anomaly_type"].values
    params = {"window": WINDOW, "hidden": 64, "latent": 16, "epochs": epochs, "lr": lr,
              "batch": batch, "threshold": round(float(threshold), 5)}
    return evaluate(y, pred, scores), pred, types, losses, params, scores, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print("Loading data and building features...")
    train, test = split(load())
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    mlflow.set_experiment("cloud-cost-anomaly")

    rows = []
    for seed in range(args.seeds):
        start = time.time()
        m, pred, types, losses, params, _, _ = run(train, test, seed, args.epochs)
        m.update({"model": "LSTM Autoencoder", "seed": seed, "train_test_seconds": round(time.time() - start, 2)})
        for kind in ["spike", "drift", "idle"]:
            m[f"recall_{kind}"] = pred[types == kind].mean()
        rows.append(m)

        with mlflow.start_run(run_name=f"LSTM Autoencoder seed {seed}"):
            mlflow.log_params({"model": "LSTM Autoencoder", "seed": seed, "features": len(LSTM_FEATURES), **params})
            for epoch, loss in enumerate(losses):
                mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metrics({k: m[k] for k in ["precision", "recall", "f1", "auc_roc", "fpr",
                                                  "train_test_seconds"]})
        print(f"LSTM Autoencoder  seed {seed}  F1 {m['f1']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  FPR {m['fpr']:.3f}  AUC {m['auc_roc']:.3f}\n")

        if seed == 0:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(range(1, len(losses) + 1), losses, marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean squared reconstruction error")
            ax.set_title("LSTM Autoencoder training loss")
            fig.tight_layout()
            fig.savefig("figures/fig_lstm_loss.png", dpi=300)

    results = pd.DataFrame(rows)
    results.to_csv(f"{OUT}/results_lstm.csv", index=False)
    cols = ["precision", "recall", "f1", "auc_roc", "fpr", "recall_spike", "recall_drift", "recall_idle"]
    print("Mean over seeds (std):")
    for c in cols:
        print(f"  {c:<14} {results[c].mean():.3f} ({results[c].std() if len(results) > 1 else 0:.3f})")
    print(f"\nSaved {OUT}/results_lstm.csv and figures/fig_lstm_loss.png")


if __name__ == "__main__":
    main()
