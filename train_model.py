"""
train_model.py
----------------
Trains a classifier to predict `opportunity_created` (i.e. will this
account convert into a sales-qualified pipeline opportunity?).

This mirrors a real ABM platform's qualification / pipeline-prediction
score: taking account fit + intent signals and producing a single
actionable probability sales reps can prioritize by.

We use XGBoost (handles non-linear interactions well) with a Logistic
Regression baseline for comparison, and SHAP for per-account explainability.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

df = pd.read_csv("data/accounts.csv")

FEATURES_NUM = [
    "employees", "annual_revenue_m", "tech_stack_maturity",
    "website_visits_30d", "pricing_page_visits_30d", "content_downloads_30d",
    "competitor_page_visits_30d", "decision_maker_engaged",
    "num_stakeholders_engaged", "days_since_last_visit",
    "fit_score", "intent_score", "existing_customer"
]
FEATURES_CAT = ["industry", "account_tier"]
TARGET = "opportunity_created"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
], remainder="passthrough")

# ---- Baseline: Logistic Regression ----
log_reg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])
log_reg.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

# ---- Main model: XGBoost ----
xgb_pipeline = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    ))
])
xgb_pipeline.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb_pipeline.predict_proba(X_test)[:, 1])

print(f"Logistic Regression AUC: {lr_auc:.3f}")
print(f"XGBoost AUC:            {xgb_auc:.3f}")
print()
preds = xgb_pipeline.predict(X_test)
print(classification_report(y_test, preds))

# ---- SHAP explainability on the XGBoost model ----
X_test_transformed = xgb_pipeline.named_steps["prep"].transform(X_test)
feature_names = xgb_pipeline.named_steps["prep"].get_feature_names_out()
explainer = shap.TreeExplainer(xgb_pipeline.named_steps["clf"])
shap_values = explainer.shap_values(X_test_transformed)

# Global feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop drivers of intent/opportunity prediction:")
print(importance_df.head(10).to_string(index=False))

# ---- Save everything the Streamlit app needs ----
joblib.dump(xgb_pipeline, "model/xgb_pipeline.joblib")
joblib.dump(log_reg, "model/log_reg.joblib")
importance_df.to_csv("model/feature_importance.csv", index=False)

with open("model/metrics.txt", "w") as f:
    f.write(f"Logistic Regression AUC: {lr_auc:.3f}\n")
    f.write(f"XGBoost AUC: {xgb_auc:.3f}\n")

print("\nSaved model/xgb_pipeline.joblib, model/log_reg.joblib, "
      "model/feature_importance.csv, model/metrics.txt")
