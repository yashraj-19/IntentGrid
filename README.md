# IntentGrid — Account Intent & Fit Scoring Engine

A lightweight Account-Based Marketing (ABM) analytics platform built as an
end-to-end data science project: data simulation → EDA/feature engineering →
classification model → explainability → interactive dashboard.

**Live demo:** _(add your deployed Streamlit link here)_

---

## 1. What this project does

B2B buying decisions are made by committees of 5–10 people, and most of the
research happens anonymously before anyone ever fills out a form. Scoring
individual "leads" misses this — the unit that matters is the *account*.
IntentGrid scores each account on two axes and turns that into a single,
explainable, actionable number:

| Concept | What it captures |
|---|---|
| `fit_score` | Firmographic fit — how closely the account matches an ideal customer profile (size, industry, tech maturity) |
| `intent_score` | Behavioral engagement — pricing-page visits, content downloads, competitor-site visits, decision-maker engagement |
| XGBoost model → `predicted_probability` | Combines fit + intent (with interaction effects) to predict likelihood of becoming a sales-qualified opportunity |
| Per-account SHAP explanation | Answers "why is this account scored this way?" so the score is actionable, not a black box |
| Fit vs Intent heatmap | Classic 2×2 prioritization matrix: Target Now / Nurture / Monitor / Deprioritize |
| Streamlit dashboard | The delivery surface tying it all together |

## 2. Project structure
```
IntentGrid/
├── generate_data.py       # simulates 2,500 B2B accounts w/ firmographic + intent signals
├── train_model.py         # trains Logistic Regression baseline + XGBoost, SHAP explainability
├── app.py                 # Streamlit dashboard (5 tabs, see below)
├── notebooks/eda.ipynb    # exploratory data analysis
├── requirements.txt
├── data/accounts.csv      # generated dataset
└── model/                 # saved model, feature importance, metrics
```

## 3. The pipeline

1. **Data (`generate_data.py`)** — Simulates 2,500 accounts with firmographic
   fields (industry, employee count, revenue, tech stack maturity) and
   behavioral signals over a 30-day window (website visits, pricing page
   visits, content downloads, competitor-site visits, decision-maker
   engagement). Two composite scores are engineered: `fit_score` (firmographic
   fit) and `intent_score` (behavioral engagement) — this fit/intent split is
   the foundational idea behind ABM scoring.
2. **Label** — `opportunity_created` (0/1), generated from a logistic
   combination of fit + intent + noise, so the "ground truth" has a realistic,
   learnable-but-not-perfect signal (AUC ~0.70–0.73, not suspiciously high).
3. **EDA (`notebooks/eda.ipynb`)** — class balance, score distributions,
   conversion rate by industry/tier, correlation heatmap, and a fit-vs-intent
   scatter colored by outcome.
4. **Model (`train_model.py`)** — Logistic Regression baseline vs. XGBoost.
   XGBoost captures non-linear interactions (e.g. a pricing-page visit matters
   much more *combined with* decision-maker engagement) better than a linear
   model.
5. **Explainability** — SHAP `TreeExplainer` gives both global feature
   importance and per-account, per-prediction explanations.
6. **Dashboard (`app.py`)** — 5 tabs: Overview, Account Explorer (filterable
   ranked list), Account Heatmap (fit vs intent quadrant matrix), Account
   Deep-Dive (SHAP bar chart for a single account), Model Performance.

## 4. Run it locally
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python generate_data.py     # creates data/accounts.csv
python train_model.py       # trains model, saves to model/
streamlit run app.py
```

## 5. Honest limitations & how I'd extend it

- The intent data here is **simulated**. Real third-party B2B intent data
  (aggregated, anonymized buyer research across a large publisher network) is
  proprietary to commercial intent-data vendors — a genuinely hard
  data-engineering problem that a synthetic dataset only approximates.
- Given more time, natural next steps: intent *trends* over time (is intent
  rising or falling this week?), CRM integration (Salesforce/HubSpot) to close
  the loop between predicted score and actual pipeline outcome, and a
  next-best-action recommender suggesting which channel/content to use next.
