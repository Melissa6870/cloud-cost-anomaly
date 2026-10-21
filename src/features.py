"""
Shared data loading, feature engineering and train/test split.

Every model uses the same features and the same chronological split, so the
comparison between models is fair (thesis Section 3.4).
"""

import numpy as np
import pandas as pd

DATA = "data/processed/cloud_cost_telemetry.csv"
TRAIN_DAYS = 63  # days 1-63 train (70%), days 64-90 test (30%)

RAW = ["cpu_util", "memory_util", "network_in_mb", "network_out_mb", "disk_ops", "cost_usd"]


def add_features(df):
    """Per-resource features that describe how unusual each hour is compared
    with that resource's own recent history (the previous 7 days)."""
    df = df.sort_values(["resource_id", "timestamp"]).copy()
    g = df.groupby("resource_id")

    for col in ["cost_usd", "cpu_util", "network_out_mb", "disk_ops"]:
        # shift(1) so the current hour never leaks into its own baseline
        mean = g[col].transform(lambda s: s.shift(1).rolling(168, min_periods=24).mean())
        std = g[col].transform(lambda s: s.shift(1).rolling(168, min_periods=24).std())
        df[f"{col}_z"] = (df[col] - mean) / (std + 1e-6)
        df[f"{col}_ratio"] = df[col] / (mean + 1e-6)

    # Cost trend over the last 24 hours (catches slow drift)
    df["cost_trend_24h"] = g["cost_usd"].transform(lambda s: s.pct_change(24))
    # Cost paid per unit of work (high when a resource is idle but still billing)
    df["cost_per_cpu"] = df["cost_usd"] / (df["cpu_util"] + 1)
    df["cost_per_cpu_ratio"] = df["cost_per_cpu"] / (
        g["cost_per_cpu"].transform(lambda s: s.shift(1).rolling(168, min_periods=24).mean()) + 1e-6)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday

    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop each resource's first day, where there is no history to compare with yet
    df = df.dropna(subset=FEATURES).sort_values(["timestamp", "resource_id"]).reset_index(drop=True)
    return df


FEATURES = RAW + [
    "cost_usd_z", "cost_usd_ratio", "cpu_util_z", "cpu_util_ratio",
    "network_out_mb_z", "network_out_mb_ratio", "disk_ops_z", "disk_ops_ratio",
    "cost_trend_24h", "cost_per_cpu", "cost_per_cpu_ratio", "hour", "weekday",
]


def load(path=DATA):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return add_features(df)


def split(df, train_days=TRAIN_DAYS):
    train = df[df["day"] <= train_days]
    test = df[df["day"] > train_days]
    return train, test
