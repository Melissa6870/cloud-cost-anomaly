"""
Company savings calculator (Objective 5: cost optimisation).

Answers the question "how much could a company of this size save?". For a given
number of staff it simulates the cloud resources the company pays for, compares
what is allocated with what is actually used, and prices the difference in Rand.

Three sources of saving are counted:

    storage   - every staff member gets the same reserved quota (like a OneDrive allowance).
                Reserved storage is billed for its full size, not the part that is filled,
                but most people use a small part of it. Quotas are moved to the smallest
                tier that covers each person's use plus headroom.
    servers   - servers are often bought one size too big. Each one is rightsized
                to the smallest size whose capacity covers its busiest hours
                (95th percentile CPU) at the target utilisation. Servers that are
                practically never used are stopped.
    anomalies - the waste the detector finds (spikes, drift, idle periods),
                scaled from results/recommendations.csv to the company's servers.

All numbers are estimates on simulated data. The assumptions are the constants
below and can be changed in the dashboard.

Usage:
    python src/savings.py
Results:
    results/savings_by_company.csv   saving for a small, medium and large company
    figures/fig_savings.png          the same as a chart
"""

import math
import os

import numpy as np
import pandas as pd

from generate_data import SERVICES, SIZES

USD_TO_ZAR = 18.0              # exchange rate assumption (R per US$)
HOURS_PER_MONTH = 730
STORAGE_USD_PER_GB = 0.08      # reserved (provisioned) storage, AWS EBS gp3, per GB-month: billed whether filled or not
QUOTA_GB = 1024                # standard quota per staff member (1 TB)
QUOTA_TIERS_GB = [50, 100, 250, 500, 1024, 2048]
STAFF_PER_SERVER = 50          # one application server per 50 staff
STAFF_PER_DATABASE = 250       # one database server per 250 staff
TARGET_CPU = 0.70              # rightsize so busy hours use at most 70% of the server
IDLE_CPU = 0.05                # p95 CPU below 5%: the server is not really used
HEADROOM = 0.20                # storage quota must leave 20% free space

COMPANIES = {"Small company": 200, "Medium company": 1000, "Large company": 5000}
SIZE_ORDER = list(SIZES)       # small, medium, large, xlarge


def simulate_staff(staff, rng):
    """Storage used by each staff member. Use grows with years of service, as in
    the example of a staff member with 20 years of files versus one with 3."""
    years = np.clip(rng.exponential(7, staff), 0.2, 40)
    used = (5 + 12 * years) * rng.lognormal(0, 0.6, staff)
    return pd.DataFrame({"staff_id": [f"staff-{i:04d}" for i in range(staff)],
                         "years_of_service": years.round(1),
                         "used_gb": np.minimum(used, 2048).round(1)})


def simulate_servers(staff, rng):
    """Servers the company runs, the size it pays for, and how busy each one is."""
    n_app = math.ceil(staff / STAFF_PER_SERVER)
    n_db = math.ceil(staff / STAFF_PER_DATABASE)
    rows = []
    for i, service in enumerate(["EC2"] * n_app + ["RDS"] * n_db):
        size = rng.choice(["medium", "large", "xlarge"], p=[0.4, 0.4, 0.2])
        p95_cpu = 0.02 if rng.random() < 0.06 else rng.beta(2.2, 4)   # about 6% sit idle
        rows.append({"server_id": f"{service.lower()}-{i:03d}", "service": service,
                     "size": size, "p95_cpu": round(float(p95_cpu), 3)})
    return pd.DataFrame(rows)


def server_price_zar(service, size, rate):
    return SERVICES[service]["price"] * SIZES[size] * HOURS_PER_MONTH * rate


def rightsize(servers, target_cpu, rate):
    """Recommend a size for every server and price the monthly saving."""
    out = servers.copy()
    new_sizes, actions = [], []
    for s in out.itertuples():
        needed = SIZES[s.size] * s.p95_cpu / target_cpu
        if s.p95_cpu < IDLE_CPU:
            new_sizes.append("stopped")
            actions.append("Stop: practically unused")
            continue
        new = next((z for z in SIZE_ORDER if SIZES[z] >= needed), s.size)
        if SIZE_ORDER.index(new) > SIZE_ORDER.index(s.size):
            new = s.size   # never recommend a bigger server; flag it instead
        new_sizes.append(new)
        actions.append("Keep" if new == s.size else f"Downsize {s.size} to {new}")
    out["recommended_size"] = new_sizes
    out["action"] = actions
    out["current_cost_zar"] = [server_price_zar(s.service, s.size, rate) for s in out.itertuples()]
    out["new_cost_zar"] = [0.0 if s.recommended_size == "stopped" else server_price_zar(s.service, s.recommended_size, rate)
                           for s in out.itertuples()]
    out["saving_zar"] = out["current_cost_zar"] - out["new_cost_zar"]
    return out


def new_quota(used_gb, headroom):
    need = used_gb * (1 + headroom)
    return next((t for t in QUOTA_TIERS_GB if t >= need), QUOTA_TIERS_GB[-1])


def storage_plan(staff_df, quota_gb, headroom, rate):
    out = staff_df.copy()
    out["allocated_gb"] = np.maximum(quota_gb, out["used_gb"])
    # Heavy users keep what they have: a quota is never raised by this plan.
    out["new_quota_gb"] = np.minimum([new_quota(u, headroom) for u in out["used_gb"]], out["allocated_gb"])
    price = STORAGE_USD_PER_GB * rate
    out["current_cost_zar"] = out["allocated_gb"] * price
    out["new_cost_zar"] = out["new_quota_gb"] * price
    out["saving_zar"] = out["current_cost_zar"] - out["new_cost_zar"]
    return out


def anomaly_waste_per_server_month(rate, results="results"):
    """Monthly anomaly waste per server, from the detector's recommendations on
    the 30 simulated resources over the 27-day test period."""
    path = f"{results}/recommendations.csv"
    if not os.path.exists(path):
        return 0.0
    waste_usd = pd.read_csv(path)["estimated_waste_usd"].sum()
    return waste_usd / 30 / 27 * 30.4 * rate


def company_savings(staff, quota_gb=QUOTA_GB, target_cpu=TARGET_CPU, headroom=HEADROOM,
                    rate=USD_TO_ZAR, seed=42, results="results"):
    """Full calculation for one company. Returns (summary, storage table, server table)."""
    rng = np.random.default_rng(seed)
    storage = storage_plan(simulate_staff(staff, rng), quota_gb, headroom, rate)
    servers = rightsize(simulate_servers(staff, rng), target_cpu, rate)
    anomaly = anomaly_waste_per_server_month(rate, results) * len(servers)

    current = storage["current_cost_zar"].sum() + servers["current_cost_zar"].sum() + anomaly
    saving = {"Reserved storage": storage["saving_zar"].sum(),
              "Server rightsizing": servers.loc[servers["recommended_size"] != "stopped", "saving_zar"].sum(),
              "Stopping idle servers": servers.loc[servers["recommended_size"] == "stopped", "saving_zar"].sum(),
              "Fixing detected anomalies": anomaly}
    total = sum(saving.values())
    summary = {"staff": staff, "servers": len(servers),
               "allocated_tb": storage["allocated_gb"].sum() / 1024, "used_tb": storage["used_gb"].sum() / 1024,
               "current_monthly_zar": current, "saving_monthly_zar": total,
               "saving_yearly_zar": total * 12, "saving_pct": 100 * total / current, **saving}
    return summary, storage, servers


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    rows = []
    for name, staff in COMPANIES.items():
        s, _, _ = company_savings(staff)
        rows.append({"company": name, **s})
        print(f"{name} ({staff} staff): pays R{s['current_monthly_zar']:,.0f}/month, "
              f"could save R{s['saving_monthly_zar']:,.0f}/month ({s['saving_pct']:.0f}%), "
              f"R{s['saving_yearly_zar']:,.0f}/year")
    table = pd.DataFrame(rows)
    table.round(2).to_csv("results/savings_by_company.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = ["Reserved storage", "Server rightsizing", "Stopping idle servers", "Fixing detected anomalies"]
    colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r.company}\n({r.staff:,} staff)" for r in table.itertuples()]
    left = np.zeros(len(table))
    for p, c in zip(parts, colours):
        vals = table[p].values / 1e3
        ax.barh(labels, vals, left=left, color=c, label=p)
        left += vals
    for i, r in enumerate(table.itertuples()):
        ax.text(left[i], i, f"  R{r.saving_monthly_zar / 1e3:,.0f}k ({r.saving_pct:.0f}%)", va="center", fontsize=9)
    ax.set_xlabel("Estimated saving per month (R thousand)")
    ax.set_xlim(0, left.max() * 1.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Estimated monthly saving by company size")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig_savings.png", dpi=300)
    print("\nSaved results/savings_by_company.csv and figures/fig_savings.png")


if __name__ == "__main__":
    main()
