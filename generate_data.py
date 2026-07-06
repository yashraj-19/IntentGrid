"""
generate_data.py
-----------------
Simulates a B2B "accounts" dataset similar to what a real ABM platform
ingests: firmographic data (who the company is) + behavioral intent
signals (what they're doing that suggests buying interest).

Why synthetic data instead of a downloaded dataset?
- Real third-party intent/ABM data is proprietary to commercial vendors.
- Generating it ourselves means we control the ground truth and can
  clearly explain every feature and the labeling logic.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2500  # number of accounts

INDUSTRIES = ["Technology", "Financial Services", "Healthcare",
              "Manufacturing", "Retail", "Education"]

def generate_accounts(n=N):
    industry = np.random.choice(INDUSTRIES, size=n,
                                 p=[0.30, 0.20, 0.15, 0.15, 0.12, 0.08])

    # Firmographics
    employees = np.random.lognormal(mean=5.5, sigma=1.3, size=n).astype(int)
    employees = np.clip(employees, 5, 50000)

    revenue_m = employees * np.random.uniform(0.08, 0.25, size=n)  # $M, rough proxy
    tech_stack_maturity = np.clip(np.random.normal(5.5, 2, size=n), 1, 10)

    def tier(e):
        if e >= 1000:
            return "Enterprise"
        if e >= 100:
            return "Mid-Market"
        return "SMB"
    account_tier = np.array([tier(e) for e in employees])

    existing_customer = np.random.binomial(1, 0.18, size=n)

    # Behavioral / intent signals (last 30 days)
    base_visits = np.random.poisson(6, size=n)
    website_visits_30d = base_visits + (tech_stack_maturity > 6).astype(int) * np.random.poisson(4, size=n)
    pricing_page_visits_30d = np.random.poisson(1.2, size=n) + (existing_customer == 0).astype(int)
    content_downloads_30d = np.random.poisson(1.5, size=n)
    competitor_page_visits_30d = np.random.poisson(0.8, size=n)  # visiting review/comparison sites
    decision_maker_engaged = np.random.binomial(1, 0.22, size=n)  # a VP/Director+ engaged
    num_stakeholders_engaged = np.random.poisson(1.8, size=n) + decision_maker_engaged
    days_since_last_visit = np.random.exponential(scale=10, size=n).astype(int)
    days_since_last_visit = np.clip(days_since_last_visit, 0, 90)

    df = pd.DataFrame({
        "account_id": [f"ACC{1000+i}" for i in range(n)],
        "industry": industry,
        "employees": employees,
        "annual_revenue_m": revenue_m.round(1),
        "account_tier": account_tier,
        "existing_customer": existing_customer,
        "tech_stack_maturity": tech_stack_maturity.round(1),
        "website_visits_30d": website_visits_30d,
        "pricing_page_visits_30d": pricing_page_visits_30d,
        "content_downloads_30d": content_downloads_30d,
        "competitor_page_visits_30d": competitor_page_visits_30d,
        "decision_maker_engaged": decision_maker_engaged,
        "num_stakeholders_engaged": num_stakeholders_engaged,
        "days_since_last_visit": days_since_last_visit,
    })

    # ---- Composite ABM-style scores (0-100) ----
    # FIT score: how well the account matches an ideal customer profile (firmographic-only)
    fit_raw = (
        (df["employees"].apply(lambda e: min(e, 5000) / 5000)) * 40 +
        (df["tech_stack_maturity"] / 10) * 30 +
        df["industry"].isin(["Technology", "Financial Services"]).astype(int) * 20 +
        (1 - df["existing_customer"]) * 10
    )
    df["fit_score"] = (fit_raw / fit_raw.max() * 100).round(1)

    # INTENT/engagement score: behavior-only
    intent_raw = (
        df["website_visits_30d"] * 1.5 +
        df["pricing_page_visits_30d"] * 6 +
        df["content_downloads_30d"] * 4 +
        df["competitor_page_visits_30d"] * 5 +
        df["decision_maker_engaged"] * 15 +
        df["num_stakeholders_engaged"] * 3 -
        df["days_since_last_visit"] * 0.4
    )
    intent_raw = intent_raw.clip(lower=0)
    df["intent_score"] = (intent_raw / intent_raw.max() * 100).round(1)

    # ---- Label: did this account convert to a sales-qualified opportunity? ----
    # A logistic-style combination of fit + intent + noise -> this is what our
    # model will learn to predict (mirrors a real ABM platform's qualification
    # / pipeline-prediction score).
    z = (
        -6
        + 0.045 * df["fit_score"]
        + 0.05 * df["intent_score"]
        + 0.8 * df["decision_maker_engaged"]
        + np.random.normal(0, 1.1, size=n)
    )
    prob = 1 / (1 + np.exp(-z))
    df["opportunity_created"] = np.random.binomial(1, prob)

    return df


if __name__ == "__main__":
    df = generate_accounts()
    df.to_csv("data/accounts.csv", index=False)
    print(f"Generated {len(df)} accounts -> data/accounts.csv")
    print(df["opportunity_created"].value_counts(normalize=True).round(3))
