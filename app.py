import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# -----------------------------
# Core calculations
# -----------------------------
def compute_dupont_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9

    # ROE
    df["roe"] = df["net_income"] / (df["total_equity"] + eps)

    # Classic DuPont
    df["net_profit_margin"] = df["net_income"] / (df["sales"] + eps)
    df["asset_turnover"] = df["sales"] / (df["total_assets"] + eps)
    df["equity_multiplier"] = df["total_assets"] / (df["total_equity"] + eps)

    # Modified DuPont (5-step)
    df["tax_burden"] = df["net_income"] / (df["ebt"] + eps)          # NI / EBT
    df["interest_burden"] = df["ebt"] / (df["ebit"] + eps)           # EBT / EBIT
    df["operating_margin"] = df["ebit"] / (df["sales"] + eps)        # EBIT / Sales

    # Tax rate estimate
    df["tax_rate_est"] = df["tax_expense"] / (df["ebt"] + eps)
    df["tax_rate_est"] = df["tax_rate_est"].clip(lower=0, upper=0.7)

    # Operating ROA
    df["nopat"] = df["ebit"] * (1 - df["tax_rate_est"])
    df["operating_roa"] = df["nopat"] / (df["total_assets"] + eps)

    # Cost of debt (after tax)
    df["gross_cost_of_debt"] = df["interest_expense"] / (df["total_debt"] + eps)
    df["after_tax_cost_of_debt"] = df["gross_cost_of_debt"] * (1 - df["tax_rate_est"])

    # Spread & leverage
    df["spread"] = df["operating_roa"] - df["after_tax_cost_of_debt"]
    df["leverage_de_ratio"] = df["total_debt"] / (df["total_equity"] + eps)

    # ROE approx from spread relation
    df["roe_from_spread"] = df["operating_roa"] + df["spread"] * df["leverage_de_ratio"]

    return df


def add_ai_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    feature_cols = [
        "roe",
        "net_profit_margin",
        "asset_turnover",
        "equity_multiplier",
        "tax_burden",
        "interest_burden",
        "operating_margin",
        "operating_roa",
        "leverage_de_ratio",
        "spread",
    ]

    feature_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_idx = feature_df.dropna().index

    df["pattern_anomaly_score"] = np.nan
    df["anomaly_score_raw"] = np.nan

    if len(valid_idx) >= 10:
        X = feature_df.loc[valid_idx].values

        iso = IsolationForest(
            n_estimators=300,
            contamination=0.1,
            random_state=42,
        )
        iso.fit(X)

        raw = -iso.score_samples(X)  # higher = more anomalous
        scaled = MinMaxScaler((0, 100)).fit_transform(raw.reshape(-1, 1)).flatten()

        df.loc[valid_idx, "anomaly_score_raw"] = raw
        df.loc[valid_idx, "pattern_anomaly_score"] = scaled

    # Operating quality risk (robust deviation)
    def robust_risk(series: pd.Series) -> pd.Series:
        s = series.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        mad = (s - med).abs().median()
        mad = mad if mad and mad > 0 else 1e-6
        z = ((s - med).abs() / mad).clip(upper=10)
        return pd.Series(MinMaxScaler((0, 100)).fit_transform(z.fillna(0).to_frame()).flatten(), index=series.index)

    df["operating_quality_risk"] = 0.5 * robust_risk(df["operating_margin"]) + 0.5 * robust_risk(df["operating_roa"])
    df["leverage_risk"] = 0.5 * robust_risk(df["equity_multiplier"]) + 0.5 * robust_risk(df["leverage_de_ratio"])

    # Final fraud score
    def fraud_score(row):
        op = row["operating_quality_risk"]
        lev = row["leverage_risk"]
        anom = row["pattern_anomaly_score"]

        if pd.isna(anom):
            return 0.6 * op + 0.4 * lev
        return 0.4 * op + 0.3 * lev + 0.3 * anom

    df["dupont_fraud_score"] = df.apply(fraud_score, axis=1)
    return df


def risk_label(score):
    if pd.isna(score):
        return "N/A"
    if score < 30:
        return "Low"
    if score < 60:
        return "Moderate"
    return "High"


# -----------------------------
# LLM Explanation Agent
# -----------------------------
def llm_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def generate_llm_explanation(row: pd.Series, df_company: pd.DataFrame) -> str | None:
    llm = llm_client()
    if llm is None:
        return None

    df_hist = df_company.sort_values("year").tail(6)
    roe_hist = [{"year": int(r["year"]), "roe_pct": float(r["roe"]) * 100} for _, r in df_hist.iterrows()]

    prompt = ChatPromptTemplate.from_template(
        """
You are a forensic financial analyst. Write a 150–220 word explanation for a company's DuPont results and AI risk score.

Company: {company}
Year: {year}

Metrics:
ROE: {roe:.2f}%
Net Profit Margin: {npm:.2f}%
Asset Turnover: {at:.2f}x
Equity Multiplier: {em:.2f}x
Operating ROA: {op_roa:.2f}%
Debt/Equity: {de:.2f}x

AI risk indicators (0–100):
DuPont Fraud Score: {fraud:.1f}
Operating Quality Risk: {op_risk:.1f}
Leverage Risk: {lev_risk:.1f}
Pattern Anomaly Score: {anom}

Recent ROE history: {roe_history}

Explain:
1) What is driving ROE (margin vs efficiency vs leverage).
2) Whether ROE quality looks operational or leverage-driven.
3) What the score implies (low/moderate/high concern).
4) What a forensic auditor should investigate next (2–3 concrete checks).
Keep it factual; do not claim fraud as fact.
"""
    )

    anom_val = row["pattern_anomaly_score"]
    anom_text = "N/A" if pd.isna(anom_val) else f"{anom_val:.1f}"

    msg = prompt.format_messages(
        company=row["company"],
        year=int(row["year"]),
        roe=float(row["roe"]) * 100,
        npm=float(row["net_profit_margin"]) * 100,
        at=float(row["asset_turnover"]),
        em=float(row["equity_multiplier"]),
        op_roa=float(row["operating_roa"]) * 100,
        de=float(row["leverage_de_ratio"]),
        fraud=float(row["dupont_fraud_score"]),
        op_risk=float(row["operating_quality_risk"]),
        lev_risk=float(row["leverage_risk"]),
        anom=anom_text,
        roe_history=roe_hist,
    )
    return llm(msg).content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Intelligent DuPont Fraud Score", layout="wide")
st.title("🧠 Intelligent DuPont Fraud Score – Multi-Agent Forensic Dashboard")

st.write(
    "Upload multi-year financial data, compute **DuPont + Modified DuPont**, run **AI anomaly detection**, "
    "and generate an **LLM-based forensic explanation**."
)

st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample synthetic data", value=False)

if use_sample and uploaded is None:
    np.random.seed(42)
    companies = ["Alpha Ltd", "Beta Corp", "Gamma Industries"]
    years = list(range(2016, 2025))
    rows = []
    for c in companies:
        base_sales = np.random.randint(800, 1500) * 1e6
        for y in years:
            base_sales *= np.random.uniform(0.95, 1.12)
            sales = base_sales
            ebit = sales * np.random.uniform(0.08, 0.18)
            interest = sales * np.random.uniform(0.01, 0.04)
            ebt = ebit - interest
            tax = max(0, ebt) * np.random.uniform(0.20, 0.30)
            net_income = ebt - tax
            assets = sales * np.random.uniform(0.7, 1.2)
            equity = assets * np.random.uniform(0.35, 0.60)
            debt = max(0, assets - equity)

            rows.append({
                "company": c,
                "industry": "Synthetic",
                "year": y,
                "sales": sales,
                "net_income": net_income,
                "ebit": ebit,
                "ebt": ebt,
                "total_assets": assets,
                "total_equity": equity,
                "total_debt": debt,
                "interest_expense": interest,
                "tax_expense": tax,
            })
    df_raw = pd.DataFrame(rows)
elif uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = None

if df_raw is None:
    st.info("Upload a CSV or tick 'Use sample synthetic data'.")
    st.stop()

required = [
    "company", "industry", "year",
    "sales", "net_income", "ebit", "ebt",
    "total_assets", "total_equity", "total_debt",
    "interest_expense", "tax_expense"
]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df_raw["year"] = pd.to_numeric(df_raw["year"], errors="coerce")
df_raw = df_raw.dropna(subset=["year"])
df_raw["year"] = df_raw["year"].astype(int)

df = compute_dupont_components(df_raw)
df = add_ai_anomaly_scores(df)

st.sidebar.header("Select Company & Year")
company = st.sidebar.selectbox("Company", sorted(df["company"].unique()))
df_company = df[df["company"] == company].sort_values("year")
year = st.sidebar.selectbox("Year", df_company["year"].tolist(), index=len(df_company)-1)
row = df_company[df_company["year"] == year].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROE", f"{row['roe']*100:.2f}%")
col2.metric("DuPont Fraud Score", f"{row['dupont_fraud_score']:.1f}", help="0–100 (higher = more abnormal)")
col2.caption(f"Risk: **{risk_label(row['dupont_fraud_score'])}**")
col3.metric("Operating Quality Risk", f"{row['operating_quality_risk']:.1f}")
col4.metric("Leverage Risk", f"{row['leverage_risk']:.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Classic DuPont", "Modified DuPont", "AI Explanation", "Data Table"])

with tab1:
    st.subheader("Classic 3-step DuPont")
    a, b, c = st.columns(3)
    a.metric("Net Profit Margin", f"{row['net_profit_margin']*100:.2f}%")
    b.metric("Asset Turnover", f"{row['asset_turnover']:.2f}x")
    c.metric("Equity Multiplier", f"{row['equity_multiplier']:.2f}x")

    st.write("Trend (ROE + components)")
    st.line_chart(df_company.set_index("year")[["roe", "net_profit_margin", "asset_turnover", "equity_multiplier"]])

with tab2:
    st.subheader("Modified DuPont (5-step) + Operating vs Leverage")
    a, b, c, d, e = st.columns(5)
    a.metric("Tax Burden (NI/EBT)", f"{row['tax_burden']:.2f}")
    b.metric("Interest Burden (EBT/EBIT)", f"{row['interest_burden']:.2f}")
    c.metric("Operating Margin (EBIT/Sales)", f"{row['operating_margin']*100:.2f}%")
    d.metric("Operating ROA", f"{row['operating_roa']*100:.2f}%")
    e.metric("Debt/Equity", f"{row['leverage_de_ratio']:.2f}x")

    st.write("Trend (Operating ROA, leverage, spread)")
    st.line_chart(df_company.set_index("year")[["operating_roa", "leverage_de_ratio", "spread"]])

with tab3:
    st.subheader("AI + Explanation Agent")
    st.write("Pattern Anomaly Score:", "N/A" if pd.isna(row["pattern_anomaly_score"]) else f"{row['pattern_anomaly_score']:.1f}")

    explanation = generate_llm_explanation(row, df_company)
    if explanation:
        st.write(explanation)
        st.caption("Generated using LangChain + OpenAI (requires OPENAI_API_KEY in Streamlit Secrets).")
    else:
        st.warning("OPENAI_API_KEY not found. Add it in Streamlit Secrets to enable the LLM Explanation Agent.")
        st.write(
            f"- Fraud Score is **{row['dupont_fraud_score']:.1f}** ({risk_label(row['dupont_fraud_score'])}).\n"
            f"- If leverage risk > operating risk, ROE may be leverage-driven; otherwise operating metrics are the bigger concern.\n"
            f"- Use the trend charts to identify the specific years where ROE changed sharply and investigate notes, one-off items, and debt structure."
        )

with tab4:
    st.subheader("Computed dataset for selected company")
    st.dataframe(df_company, use_container_width=True)
