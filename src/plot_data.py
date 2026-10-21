"""
Exploratory figures for the generated dataset (thesis Chapter 4 / paper Section III).

Usage:
    python src/plot_data.py
Saves PNGs (300 dpi) to figures/.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

DATA = "data/processed/cloud_cost_telemetry.csv"
EVENTS = "data/processed/anomaly_events.csv"
OUT = "figures"
COLOURS = {"spike": "tab:red", "drift": "tab:orange", "idle": "tab:purple"}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    events = pd.read_csv(EVENTS, parse_dates=["start", "end"])

    # Fig 1: 90-day cost of one EC2 instance with every anomaly marked
    rid = "ec2-00"
    one = df[df["resource_id"] == rid]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(one["timestamp"], one["cost_usd"], lw=0.6, color="tab:blue", label="Hourly cost")
    for kind, colour in COLOURS.items():
        part = one[one["anomaly_type"] == kind]
        ax.scatter(part["timestamp"], part["cost_usd"], s=6, color=colour, label=kind.capitalize())
    ax.set_xlabel("Date")
    ax.set_ylabel("Hourly cost (USD)")
    ax.set_title(f"Simulated hourly cost of {rid} with labelled anomalies")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cost_timeline.png", dpi=300)

    # Fig 2: one example of each anomaly type, CPU and cost side by side
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex="col")
    for col, kind in enumerate(COLOURS):
        ev = events[(events["anomaly_type"] == kind) & (events["resource_id"].str.startswith("ec2"))].iloc[0]
        pad = pd.Timedelta(hours=48)
        win = df[(df["resource_id"] == ev["resource_id"])
                 & (df["timestamp"] >= ev["start"] - pad) & (df["timestamp"] <= ev["end"] + pad)]
        for row, metric, label in [(0, "cpu_util", "CPU utilisation (%)"), (1, "cost_usd", "Hourly cost (USD)")]:
            ax = axes[row, col]
            ax.plot(win["timestamp"], win[metric], lw=0.8, color="tab:blue")
            ax.axvspan(ev["start"], ev["end"], color=COLOURS[kind], alpha=0.25)
            ax.set_ylabel(label if col == 0 else "")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        axes[0, col].set_title(f"{kind.capitalize()} ({ev['resource_id']})")
    fig.suptitle("Examples of the three injected anomaly types (shaded)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_anomaly_types.png", dpi=300)

    # Fig 3: labelled rows per anomaly type and severity
    counts = df[df["is_anomaly"] == 1].groupby(["anomaly_type", "severity"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(6, 3.5), rot=0)
    ax.set_xlabel("Anomaly type")
    ax.set_ylabel("Labelled hourly records")
    ax.set_title("Labelled anomaly records by type and severity")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_anomaly_counts.png", dpi=300)

    print(f"Saved 3 figures to {OUT}/")


if __name__ == "__main__":
    main()
