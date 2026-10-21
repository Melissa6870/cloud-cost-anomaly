"""
Synthetic cloud cost telemetry generator with labelled anomalies.

Simulates hourly usage and billing telemetry for a set of cloud resources
(EC2, RDS, S3, Lambda, EBS) and injects three labelled anomaly types:

    spike  - sudden CPU and cost jump lasting a few hours (runaway process)
    drift  - cost that creeps upwards over several days (forgotten job)
    idle   - near-zero utilisation while cost stays the same (over-provisioning)

A gradual workload shift is applied in the second half of the period. It is
NOT labelled as an anomaly; it exists so model drift can be measured on the
30, 60 and 90 day windows (Research Question 2).

Usage:
    python src/generate_data.py
    python src/generate_data.py --days 90 --resources 30 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd

# Service profiles: base hourly price (USD) per size unit, typical CPU level,
# and how strongly cost follows usage (0 = flat price, 1 = fully usage-based).
SERVICES = {
    "EC2":    {"price": 0.096, "cpu": 35, "usage_cost": 0.15},
    "RDS":    {"price": 0.171, "cpu": 25, "usage_cost": 0.10},
    "S3":     {"price": 0.023, "cpu": 5,  "usage_cost": 0.80},
    "Lambda": {"price": 0.010, "cpu": 15, "usage_cost": 0.95},
    "EBS":    {"price": 0.011, "cpu": 3,  "usage_cost": 0.30},
}
SERVICE_MIX = ["EC2"] * 12 + ["RDS"] * 4 + ["S3"] * 6 + ["Lambda"] * 5 + ["EBS"] * 3
SIZES = {"small": 1, "medium": 2, "large": 4, "xlarge": 8}

# Number of anomaly events per resource over 90 days, scaled to --days.
EVENTS_PER_90_DAYS = {"spike": 4, "drift": 1, "idle": 1}


def build_resources(n, rng):
    """Create the resource inventory: id, service and instance size."""
    rows = []
    for i in range(n):
        service = SERVICE_MIX[i % len(SERVICE_MIX)]
        size = rng.choice(list(SIZES), p=[0.3, 0.4, 0.2, 0.1])
        rows.append({"resource_id": f"{service.lower()}-{i:02d}", "service": service, "size": size})
    return pd.DataFrame(rows)


def normal_usage(hours, profile, rng, shift_start):
    """Normal behaviour: daily and weekly cycles, slow growth, noise and a
    gradual workload shift after shift_start (unlabelled concept drift)."""
    t = np.arange(hours)
    hour_of_day = t % 24
    day_of_week = (t // 24) % 7

    daily = 1 + 0.35 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)   # busier in working hours
    weekly = np.where(day_of_week >= 5, 0.7, 1.0)                    # quieter at weekends
    growth = 1 + 0.10 * t / hours                                    # 10% organic growth
    shift = 1 + 0.25 * np.clip((t - shift_start) / (hours * 0.2), 0, 1)  # workload shift

    load = daily * weekly * growth * shift
    cpu = profile["cpu"] * load * rng.normal(1, 0.08, hours)
    cpu = np.clip(cpu, 0.5, 100)
    memory = np.clip(cpu * 1.2 + rng.normal(20, 3, hours), 1, 100)
    net_in = np.abs(load * 120 * rng.lognormal(0, 0.25, hours))      # MB per hour
    net_out = np.abs(load * 80 * rng.lognormal(0, 0.25, hours))
    disk_ops = np.abs(load * 900 * rng.lognormal(0, 0.3, hours))
    return load, cpu, memory, net_in, net_out, disk_ops


def hourly_cost(base_price, profile, load, rng):
    """Cost = fixed price share + usage-driven share, with small billing noise."""
    u = profile["usage_cost"]
    return base_price * ((1 - u) + u * load) * rng.normal(1, 0.03, len(load))


def pick_window(hours, length, taken, rng, tries=50):
    """Pick a start hour for an event so it does not overlap earlier events."""
    for _ in range(tries):
        start = int(rng.integers(24, hours - length - 1))
        if not taken[start:start + length].any():
            taken[start:start + length] = True
            return start
    return None


def severity_of(excess_ratio):
    """Severity from how much the cost exceeds normal (1.0 = double the cost)."""
    if excess_ratio >= 2.0:
        return "high"
    if excess_ratio >= 0.75:
        return "medium"
    return "low"


def inject_anomalies(df, hours, rng, scale, events):
    """Inject spike, drift and idle events into one resource's telemetry."""
    taken = np.zeros(hours, dtype=bool)
    rid = df["resource_id"].iat[0]

    for kind, per_90 in EVENTS_PER_90_DAYS.items():
        for _ in range(max(1, round(per_90 * scale))):
            if kind == "spike":
                length = int(rng.integers(2, 7))           # 2 to 6 hours
            elif kind == "drift":
                length = int(rng.integers(48, 121))        # 2 to 5 days
            else:
                length = int(rng.integers(24, 73))         # 1 to 3 days
            start = pick_window(hours, length, taken, rng)
            if start is None:
                continue
            idx = np.arange(start, start + length)
            normal_cost = df["cost_usd"].values[idx].copy()

            if kind == "spike":
                factor = rng.uniform(2.5, 5.0)
                df.loc[idx, "cpu_util"] = np.clip(df.loc[idx, "cpu_util"] * factor, 0, 100)
                df.loc[idx, "network_out_mb"] *= factor
                df.loc[idx, "cost_usd"] *= factor
            elif kind == "drift":
                ramp = np.linspace(1.0, rng.uniform(1.8, 3.0), length)
                df.loc[idx, "cpu_util"] = np.clip(df.loc[idx, "cpu_util"] * (0.5 + 0.5 * ramp), 0, 100)
                df.loc[idx, "disk_ops"] *= ramp
                df.loc[idx, "cost_usd"] *= ramp
            else:  # idle: the resource keeps billing but does no work
                df.loc[idx, "cpu_util"] = rng.uniform(0.2, 2.0, length)
                df.loc[idx, "memory_util"] = rng.uniform(5, 10, length)
                df.loc[idx, "network_in_mb"] *= 0.02
                df.loc[idx, "network_out_mb"] *= 0.02
                df.loc[idx, "disk_ops"] *= 0.02

            excess = df["cost_usd"].values[idx].sum() / normal_cost.sum() - 1
            # Idle waste: the whole cost is wasted, so rate it by duration.
            sev = severity_of(excess) if kind != "idle" else ("high" if length >= 48 else "medium")
            wasted = normal_cost.sum() if kind == "idle" else df["cost_usd"].values[idx].sum() - normal_cost.sum()
            event_id = f"{rid}-{kind}-{start}"
            df.loc[idx, "is_anomaly"] = 1
            df.loc[idx, "anomaly_type"] = kind
            df.loc[idx, "severity"] = sev
            df.loc[idx, "event_id"] = event_id
            events.append({
                "event_id": event_id, "resource_id": rid, "anomaly_type": kind,
                "start": df["timestamp"].iat[start], "end": df["timestamp"].iat[start + length - 1],
                "duration_h": length, "severity": sev,
                "wasted_cost_usd": round(float(wasted), 4),
            })
    return df


def generate(days=90, n_resources=30, seed=42, start_date="2026-01-01"):
    rng = np.random.default_rng(seed)
    hours = days * 24
    timestamps = pd.date_range(start_date, periods=hours, freq="h")
    shift_start = int(hours * 0.5)
    scale = days / 90
    resources = build_resources(n_resources, rng)
    frames, events = [], []

    for _, res in resources.iterrows():
        profile = SERVICES[res["service"]]
        base_price = profile["price"] * SIZES[res["size"]]
        load, cpu, mem, net_in, net_out, disk = normal_usage(hours, profile, rng, shift_start)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "resource_id": res["resource_id"],
            "service": res["service"],
            "size": res["size"],
            "cpu_util": cpu,
            "memory_util": mem,
            "network_in_mb": net_in,
            "network_out_mb": net_out,
            "disk_ops": disk,
            "cost_usd": hourly_cost(base_price, profile, load, rng),
            "is_anomaly": 0,
            "anomaly_type": "normal",
            "severity": "none",
            "event_id": "",
        })
        frames.append(inject_anomalies(df, hours, rng, scale, events))

    data = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "resource_id"])
    num_cols = ["cpu_util", "memory_util", "network_in_mb", "network_out_mb", "disk_ops"]
    data[num_cols] = data[num_cols].round(3)
    data["cost_usd"] = data["cost_usd"].round(5)
    data["day"] = (data["timestamp"] - data["timestamp"].min()).dt.days + 1  # for 30/60/90-day windows
    return data.reset_index(drop=True), pd.DataFrame(events)


def main():
    parser = argparse.ArgumentParser(description="Generate labelled cloud cost telemetry")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--resources", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    data, events = generate(args.days, args.resources, args.seed)
    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, "cloud_cost_telemetry.csv")
    events_path = os.path.join(args.out, "anomaly_events.csv")
    data.to_csv(data_path, index=False)
    events.to_csv(events_path, index=False)

    print(f"Saved {len(data):,} rows to {data_path}")
    print(f"Saved {len(events):,} anomaly events to {events_path}")
    print(f"Resources: {data['resource_id'].nunique()}  Days: {args.days}  Seed: {args.seed}")
    print(f"Anomaly rate: {data['is_anomaly'].mean():.2%}")
    print("\nRows per anomaly type:")
    print(data["anomaly_type"].value_counts().to_string())
    print("\nEvents per type and severity:")
    print(events.groupby(["anomaly_type", "severity"]).size().to_string())
    print(f"\nTotal simulated cost: ${data['cost_usd'].sum():,.2f}  "
          f"(wasted by anomalies: ${events['wasted_cost_usd'].sum():,.2f})")


if __name__ == "__main__":
    main()
