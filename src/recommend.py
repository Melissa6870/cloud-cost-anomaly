"""
Rule-mapped recommendation engine (Objective 5, Research Question 3).

Takes the hours flagged by the best detector (XGBoost), groups consecutive
flagged hours of one resource into incidents, works out what kind of anomaly
each incident looks like from its telemetry, and maps it to a cost-saving action.

    spike -> investigate the runaway process, add an auto-scaling limit / budget alert
    drift -> find and stop the forgotten job, add a lifecycle or scheduling policy
    idle  -> rightsize or stop the resource

The engine never sees the true labels. They are only used afterwards to measure
how often it chose the right action (recommendation accuracy).

Usage:
    python src/recommend.py
Results:
    results/detections.csv           every test hour with the detector's flag and score
    results/recommendations.csv      one row per incident with its action and saving
    results/recommendation_accuracy.csv
"""

import os

import pandas as pd

from features import load, split
from train_models import xgboost

OUT = "results"

# Anomaly type x service -> recommended action
ACTIONS = {
    "spike": {
        "default": "Investigate the runaway process; set an auto-scaling maximum and a budget alert",
        "Lambda": "Check for a retry loop or traffic burst; set reserved concurrency and a budget alert",
        "S3": "Check for unexpected data egress; review bucket access and request rates",
    },
    "drift": {
        "default": "Find and stop the forgotten long-running job; add a scheduled shutdown policy",
        "S3": "Apply an S3 lifecycle policy to move or expire old objects",
        "EBS": "Delete old snapshots and apply a snapshot lifecycle policy",
    },
    "idle": {
        "default": "Rightsize to a smaller instance or stop it outside working hours",
        "RDS": "Stop the idle database or move it to a smaller instance class",
        "S3": "Move data to an infrequent-access storage class",
        "EBS": "Detach and delete the unused volume after taking a snapshot",
    },
}
SIZE_DOWN = {"xlarge": "large", "large": "medium", "medium": "small", "small": "small"}


def classify(incident):
    """Decide the anomaly type from the incident's own telemetry (no labels)."""
    hours = len(incident)
    cpu_ratio = incident["cpu_util_ratio"].median()
    cost_ratio = incident["cost_usd_ratio"].median()
    if cpu_ratio < 0.25:
        return "idle"                       # cost still billed, almost no work done
    if hours <= 8 and cost_ratio >= 1.8:
        return "spike"                      # short and sharp
    return "drift"                          # longer, creeping increase


def incidents_from(detections, gap=2):
    """Group flagged hours of the same resource into incidents (gaps of up to 2 h allowed)."""
    flagged = detections[detections["flag"] == 1].sort_values(["resource_id", "timestamp"])
    groups = []
    for rid, g in flagged.groupby("resource_id"):
        new = g["timestamp"].diff() > pd.Timedelta(hours=gap)
        for _, inc in g.groupby(new.cumsum()):
            groups.append(inc)
    return groups


def recommend(inc):
    service, size = inc["service"].iat[0], inc["size"].iat[0]
    kind = classify(inc)
    action = ACTIONS[kind].get(service, ACTIONS[kind]["default"])
    if kind == "idle" and service in ("EC2", "RDS") and size != "small":
        action += f" (e.g. {size} -> {SIZE_DOWN[size]})"
    # Estimated waste: extra cost above the normal level, or all of it when idle
    baseline = inc["cost_usd"] / inc["cost_usd_ratio"].clip(lower=1e-6)
    waste = inc["cost_usd"].sum() if kind == "idle" else (inc["cost_usd"] - baseline).clip(lower=0).sum()
    hours = len(inc)
    priority = "High" if waste > 1.0 or hours >= 48 else ("Medium" if waste > 0.25 else "Low")
    true_type = inc.loc[inc["anomaly_type"] != "normal", "anomaly_type"]
    return {
        "resource_id": inc["resource_id"].iat[0], "service": service, "size": size,
        "start": inc["timestamp"].min(), "end": inc["timestamp"].max(), "hours": hours,
        "detected_type": kind, "recommended_action": action, "priority": priority,
        "estimated_waste_usd": round(float(waste), 4),
        # evaluation only, never used to choose the action:
        "true_type": true_type.mode().iat[0] if len(true_type) else "false alarm",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading data and running the XGBoost detector on the test period...")
    train, test = split(load())
    pred, scores, _ = xgboost(train, test, seed=0)
    detections = test.assign(flag=pred, score=scores)
    detections[["timestamp", "day", "resource_id", "service", "size", "cpu_util", "cost_usd", "cost_usd_ratio",
                "cpu_util_ratio", "is_anomaly", "anomaly_type", "flag", "score"]].to_csv(f"{OUT}/detections.csv", index=False)

    recs = pd.DataFrame([recommend(inc) for inc in incidents_from(detections)])
    order = {"High": 0, "Medium": 1, "Low": 2}
    recs = recs.sort_values(["priority", "estimated_waste_usd"], key=lambda s: s.map(order) if s.name == "priority" else -s)
    recs.to_csv(f"{OUT}/recommendations.csv", index=False)

    real = recs[recs["true_type"] != "false alarm"]
    acc = (real["detected_type"] == real["true_type"]).mean()
    per_type = (real["detected_type"] == real["true_type"]).groupby(real["true_type"]).mean()
    confusion = pd.crosstab(real["true_type"], real["detected_type"])
    summary = pd.DataFrame({
        "metric": ["incidents", "real incidents", "false alarms", "correct action (all real)"]
                  + [f"correct action ({t})" for t in per_type.index],
        "value": [len(recs), len(real), len(recs) - len(real), round(acc, 4)] + [round(v, 4) for v in per_type.values],
    })
    summary.to_csv(f"{OUT}/recommendation_accuracy.csv", index=False)

    print(f"\n{len(recs)} incidents raised, {len(recs) - len(real)} of them false alarms")
    print(f"Recommendation accuracy on real incidents: {acc:.1%}")
    print("\nTrue type (rows) vs type the engine chose (columns):")
    print(confusion.to_string())
    print(f"\nEstimated waste flagged: ${recs['estimated_waste_usd'].sum():,.2f}")
    print("\nTop 5 recommendations:")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 70)
    print(recs[["resource_id", "hours", "detected_type", "priority", "estimated_waste_usd", "recommended_action"]]
          .head().to_string(index=False))
    print(f"\nSaved {OUT}/recommendations.csv, {OUT}/recommendation_accuracy.csv, {OUT}/detections.csv")


if __name__ == "__main__":
    main()
