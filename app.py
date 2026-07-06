"""
IntentGrid — Account Intent & Fit Scoring Engine
--------------------------------------------------
A mini ABM (Account-Based Marketing) platform that scores B2B accounts on
firmographic fit and behavioral intent, predicts conversion likelihood, and
explains each score with SHAP.

Tabs:
1. Overview        - dataset summary
2. Account Explorer - sortable/filterable account table with intent scores
3. Account Heatmap  - classic Fit vs Intent 2x2 ABM matrix
4. Account Deep-Dive - SHAP explanation for a single account ("why this score?")
5. Model Performance - AUC, confusion matrix, global feature importance
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

st.set_page_config(page_title="Account Intent & Fit Scoring Engine", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/accounts.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/xgb_pipeline.joblib")

@st.cache_data
def load_importance():
    return pd.read_csv("model/feature_importance.csv")

df = load_data()
model = load_model()
importance_df = load_importance()

FEATURES_NUM = [
    "employees", "annual_revenue_m", "tech_stack_maturity",
    "website_visits_30d", "pricing_page_visits_30d", "content_downloads_30d",
    "competitor_page_visits_30d", "decision_maker_engaged",
    "num_stakeholders_engaged", "days_since_last_visit",
    "fit_score", "intent_score", "existing_customer"
]
FEATURES_CAT = ["industry", "account_tier"]

df["predicted_probability"] = model.predict_proba(df[FEATURES_NUM + FEATURES_CAT])[:, 1]
df["predicted_probability_pct"] = (df["predicted_probability"] * 100).round(1)

st.title("🎯 IntentGrid — Account Intent & Fit Scoring Engine")
st.caption("A lightweight ABM-style platform for scoring accounts on fit and intent, "
           "predicting conversion likelihood, and explaining every score.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Account Explorer", "Account Heatmap", "Account Deep-Dive", "Model Performance"]
)

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accounts", len(df))
    c2.metric("Opportunities Created", int(df["opportunity_created"].sum()))
    c3.metric("Conversion Rate", f"{df['opportunity_created'].mean()*100:.1f}%")
    c4.metric("Avg Intent Score", f"{df['intent_score'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="industry", color="account_tier",
                            title="Accounts by Industry & Tier", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="predicted_probability_pct", nbins=30,
                             title="Distribution of Predicted Opportunity Probability")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 2: ACCOUNT EXPLORER ----------------
with tab2:
    st.subheader("Prioritized Account List")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        industry_filter = st.multiselect("Industry", options=sorted(df["industry"].unique()))
    with colf2:
        tier_filter = st.multiselect("Tier", options=sorted(df["account_tier"].unique()))
    with colf3:
        min_prob = st.slider("Min predicted probability (%)", 0, 100, 0)

    filtered = df.copy()
    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]
    if tier_filter:
        filtered = filtered[filtered["account_tier"].isin(tier_filter)]
    filtered = filtered[filtered["predicted_probability_pct"] >= min_prob]

    display_cols = ["account_id", "industry", "account_tier", "fit_score",
                     "intent_score", "predicted_probability_pct", "decision_maker_engaged"]
    st.dataframe(
        filtered[display_cols].sort_values("predicted_probability_pct", ascending=False),
        use_container_width=True, height=450
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} accounts, ranked by predicted "
               f"probability of becoming a sales-qualified opportunity.")

# ---------------- TAB 3: ACCOUNT HEATMAP ----------------
with tab3:
    st.subheader("Fit vs. Intent Matrix")
    st.caption("The classic ABM prioritization view: high fit + high intent = "
               "target now. This quadrant view is how sales reps get handed "
               "a prioritized account list.")

    fig = px.scatter(
        df, x="fit_score", y="intent_score", color="account_tier",
        size="employees", hover_data=["account_id", "industry", "predicted_probability_pct"],
        title="Account Prioritization Matrix (Fit vs Intent)",
        labels={"fit_score": "Fit Score", "intent_score": "Intent Score"}
    )
    fig.add_hline(y=df["intent_score"].median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=df["fit_score"].median(), line_dash="dash", line_color="gray")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Quadrant guide**
    - 🟢 **Top-right (high fit, high intent)** → Target Now — hand straight to sales
    - 🟡 **Top-left (low fit, high intent)** → Nurture — engaged but may not be ideal-fit
    - 🟠 **Bottom-right (high fit, low intent)** → Monitor — good fit, not yet in-market
    - ⚪ **Bottom-left (low fit, low intent)** → Deprioritize
    """)

# ---------------- TAB 4: ACCOUNT DEEP-DIVE ----------------
with tab4:
    st.subheader("Why is this account scored this way?")
    account_id = st.selectbox("Select an account", df["account_id"].tolist())
    row = df[df["account_id"] == account_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Probability", f"{row['predicted_probability_pct']}%")
    c2.metric("Fit Score", row["fit_score"])
    c3.metric("Intent Score", row["intent_score"])

    st.write("**Account details**")
    st.json({
        "industry": row["industry"], "tier": row["account_tier"],
        "employees": int(row["employees"]),
        "decision_maker_engaged": bool(row["decision_maker_engaged"]),
        "pricing_page_visits_30d": int(row["pricing_page_visits_30d"]),
        "competitor_page_visits_30d": int(row["competitor_page_visits_30d"]),
        "days_since_last_visit": int(row["days_since_last_visit"]),
    })

    # Per-account SHAP explanation
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    x_row = df[df["account_id"] == account_id][FEATURES_NUM + FEATURES_CAT]
    x_transformed = prep.transform(x_row)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(x_transformed)
    feature_names = prep.get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals[0]
    }).sort_values("shap_value", key=abs, ascending=False).head(8)

    fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                 title="Top factors pushing this account's score up or down",
                 color="shap_value", color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 5: MODEL PERFORMANCE ----------------
with tab5:
    st.subheader("Model Performance")
    with open("model/metrics.txt") as f:
        st.text(f.read())

    fig = px.bar(importance_df.head(10), x="mean_abs_shap", y="feature",
                 orientation="h", title="Global Feature Importance (mean |SHAP value|)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Model: XGBoost classifier predicting `opportunity_created`. "
               "Baseline Logistic Regression included in model/log_reg.joblib for comparison.")
