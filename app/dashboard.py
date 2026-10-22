"""
Cloud cost anomaly dashboard.

Shows detected anomalies, cost-saving recommendations and the model comparison
for non-technical FinOps users. Run src/recommend.py first, then:

    streamlit run app/dashboard.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

RES, FIG = "results", "figures"

st.set_page_config(page_title="Cloud Cost Anomaly Detection", layout="wide")


@st.cache_data
def load():
    det = pd.read_csv(f"{RES}/detections.csv", parse_dates=["timestamp"])
    recs = pd.read_csv(f"{RES}/recommendations.csv", parse_dates=["start", "end"])
    return det, recs


if not os.path.exists(f"{RES}/recommendations.csv"):
    st.error("No results yet. Run `python src/recommend.py` first, then refresh this page.")
    st.stop()

det, recs = load()

st.title("AI-Driven Cloud Cost Anomaly Detection")
st.caption("Detector: XGBoost  |  Test period: days 64-90 of the simulated telemetry  |  "
           "Recommendations from the rule-mapped engine")

# Headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cloud spend in test period", f"${det['cost_usd'].sum():,.2f}")
c2.metric("Anomaly incidents raised", f"{len(recs)}")
c3.metric("Estimated waste flagged", f"${recs['estimated_waste_usd'].sum():,.2f}")
c4.metric("High-priority actions", f"{(recs['priority'] == 'High').sum()}")

tab_recs, tab_cost, tab_models = st.tabs(["Recommendations", "Cost explorer", "Model comparison"])

with tab_recs:
    st.subheader("Cost-saving recommendations")
    f1, f2 = st.columns(2)
    priorities = f1.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    types = f2.multiselect("Anomaly type", ["spike", "drift", "idle"], default=["spike", "drift", "idle"])
    shown = recs[recs["priority"].isin(priorities) & recs["detected_type"].isin(types)]
    st.dataframe(
        shown[["priority", "resource_id", "service", "detected_type", "start", "hours",
               "estimated_waste_usd", "recommended_action"]],
        hide_index=True, width="stretch",
        column_config={"estimated_waste_usd": st.column_config.NumberColumn("Est. waste (USD)", format="$%.2f"),
                       "detected_type": "Type", "resource_id": "Resource", "recommended_action": "Recommended action"})
    st.download_button("Download recommendations (CSV)", shown.to_csv(index=False), "recommendations.csv")

    by_type = recs.groupby("detected_type")["estimated_waste_usd"].sum().reset_index()
    st.altair_chart(alt.Chart(by_type).mark_bar().encode(
        x=alt.X("estimated_waste_usd:Q", title="Estimated waste (USD)"),
        y=alt.Y("detected_type:N", title="Anomaly type", sort="-x"),
    ).properties(height=160, title="Where the money is being wasted"), width="stretch")

with tab_cost:
    st.subheader("Hourly cost per resource")
    worst = recs.groupby("resource_id")["estimated_waste_usd"].sum().sort_values(ascending=False).index.tolist()
    others = sorted(set(det["resource_id"]) - set(worst))
    rid = st.selectbox("Resource (most wasteful first)", worst + others)
    one = det[det["resource_id"] == rid]
    line = alt.Chart(one).mark_line(strokeWidth=1).encode(
        x=alt.X("timestamp:T", title="Time"), y=alt.Y("cost_usd:Q", title="Hourly cost (USD)"))
    flags = alt.Chart(one[one["flag"] == 1]).mark_circle(size=40, color="red").encode(
        x="timestamp:T", y="cost_usd:Q", tooltip=["timestamp:T", "cost_usd:Q", "cpu_util:Q", "score:Q"])
    st.altair_chart((line + flags).properties(height=320), width="stretch")
    st.caption("Red dots = hours flagged as anomalous by the detector.")
    st.dataframe(recs[recs["resource_id"] == rid][["start", "end", "hours", "detected_type", "priority",
                                                    "estimated_waste_usd", "recommended_action"]],
                 hide_index=True, width="stretch")

with tab_models:
    st.subheader("How the four detection models compare")
    if os.path.exists(f"{RES}/final_comparison.csv"):
        comp = pd.read_csv(f"{RES}/final_comparison.csv", index_col=0)
        means = comp[[c for c in comp.columns if c.endswith("_mean")]]
        means.columns = ["Precision", "Recall", "F1-score", "AUC-ROC", "False-positive rate"]
        st.dataframe(means.style.format("{:.3f}"), width="stretch")
    for name, caption in [("fig_model_comparison.png", "Mean of 5 seeds on the held-out test set"),
                          ("fig_roc_curves.png", "ROC curves"),
                          ("fig_drift.png", "Model drift: trained on days 1-20, never retrained")]:
        if os.path.exists(f"{FIG}/{name}"):
            st.image(f"{FIG}/{name}", caption=caption)
    if os.path.exists(f"{RES}/recommendation_accuracy.csv"):
        st.subheader("Recommendation engine accuracy")
        st.dataframe(pd.read_csv(f"{RES}/recommendation_accuracy.csv"), hide_index=True)
