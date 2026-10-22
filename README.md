# AI-Driven Anomaly Detection for Cloud Cost Optimisation

BEngTech Honours research project, Cape Peninsula University of Technology (CPUT).
Author: Melissa Ncube. Supervisor: Mr V. Moyo.

This repository compares four approaches for detecting cost anomalies in cloud
telemetry, namely a rule-based threshold baseline, Isolation Forest, XGBoost and an
LSTM Autoencoder. It then turns the detections into cost-saving recommendations.
Everything is open-source and reproducible with a fixed random seed.

## What is in the repository

| Path | Purpose |
| --- | --- |
| `src/generate_data.py` | Generates the labelled synthetic dataset: 30 resources over 90 days, hourly, with spike, drift and idle anomalies |
| `src/plot_data.py` | Exploratory figures of the dataset |
| `src/features.py` | Shared feature engineering and the chronological train/test split (days 1-63 / 64-90) |
| `src/train_models.py` | Threshold baseline, Isolation Forest and XGBoost (5 seeds each, logged in MLflow) |
| `src/train_lstm.py` | LSTM Autoencoder (5 seeds, logged in MLflow) |
| `src/evaluate.py` | Comparison table, ROC curves, Wilcoxon signed-rank tests and 30/60/90-day drift analysis |
| `src/nab_check.py` | Real-data check on the NAB realAWSCloudwatch benchmark |
| `src/recommend.py` | Rule-mapped recommendation engine and its accuracy |
| `app/dashboard.py` | Streamlit dashboard for anomalies and recommendations |
| `data/processed/` | Generated dataset and anomaly event log |
| `results/` | All result tables (CSV) |
| `figures/` | All figures (PNG, 300 dpi) |

## How to reproduce

Requires Python 3.11.

```
python -m venv .venv
.venv\Scripts\activate            # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/numenta/NAB.git data/raw/NAB

python src/generate_data.py
python src/plot_data.py
python src/train_models.py
python src/train_lstm.py
python src/evaluate.py
python src/nab_check.py
python src/recommend.py
streamlit run app/dashboard.py
```

Run `mlflow ui` to browse every logged experiment run.

## Dataset

The dataset has 64,800 hourly records covering 30 resources (EC2, RDS, S3, Lambda and EBS) over 90 days. About 6.7% of the records belong to one of 180 labelled anomaly events:

- **spike**: a 2 to 6 hour jump in CPU and cost, as from a runaway process
- **drift**: cost creeping up over 2 to 5 days, as from a forgotten job
- **idle**: near-zero utilisation for 1 to 3 days while billing continues, as from an over-provisioned resource

An unlabelled workload shift in the second half of the period is used to measure model drift.

## Licence

Code and data are released for academic use. Please cite the accompanying thesis
and conference paper if you use them.
