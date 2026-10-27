"""Logs every figure and result table to one MLflow run so they can be viewed in the MLflow UI."""
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("cloud-cost-anomaly")

with mlflow.start_run(run_name="Final results and figures"):
    for png in sorted((ROOT / "figures").glob("*.png")):
        mlflow.log_artifact(str(png), artifact_path="figures")
    for csv in sorted((ROOT / "results").glob("*.csv")):
        mlflow.log_artifact(str(csv), artifact_path="results")

    comp = pd.read_csv(ROOT / "results" / "final_comparison.csv", index_col=0)
    for model, row in comp.iterrows():
        key = model.lower().replace(" ", "_")
        mlflow.log_metrics({f"{key}_{m}": row[f"{m}_mean"] for m in ["precision", "recall", "f1", "auc_roc", "fpr"]})

print("Logged figures and results. Open MLflow and look for the run 'Final results and figures'.")