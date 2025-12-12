import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# LangChain + OpenAI for Explanation Agent (Agent 4)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ======================================================
# Agent 2: DuPont & AI Analysis Functions
# ======================================================

def compute_dupont_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic & modified DuPont components and supporting metrics.
    Expects columns:
    sales, net_income, ebit, ebt, total_assets, total_equity, total_debt,
    interest_expense, tax_expense
    """
    df = df.copy()
    eps = 1e-9

    # ROE
    df["roe"] = df["net_income"] / (df["total_equity"] + eps)

    # Classic 3-step DuPont
    df["net_profit_margin"] = df["net_income"] / (df["sales"] + eps)
    df["asset_turnover"] = df["sales"] / (df["total_assets"] + eps)
    df["equity_multiplier"] = df["total_assets"] / (df["total_equity"] + eps)

    # 5-step Modified DuPont
    df["tax_burden"] = df["net_income"] / (df["ebt"] + eps)
    df["interest_burden"] = df["ebt"] / (df["ebit"] + eps)
    df["operating_margin"] = df["ebit"] / (df["sales"] + eps)

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

    # Theoretical ROE from ROA + spread*leverage
    df["roe_from_spread"] = df["operating_roa"] + df["spread"] * df["leverage_de_ratio"]

    return df


def add_ai_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train an IsolationForest on DuPont-related features and add
    anomaly scores and risk components.
    """
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

    if len(valid_idx) < 10:
        df["anomaly_score_raw"] = np.nan
        df["pattern_anomaly_score"] = np.nan
    else:
        X = feature_df.loc[valid_idx].values

        iso = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=42,
        )
        iso.fit(X)

        raw_scores = -iso.score_samples(X)  # invert so higher = more anomalous
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

        df["anomaly_score_raw"] = np.nan
        df["pattern_anomaly_score"] = np.nan

        df.loc[valid_idx, "anomaly_score_raw"] = raw_scores
        df.loc[valid_idx, "pattern_anomaly_score"] = scaled_scores

    # Operating quality risk
    for col, risk_col in [
        ("operating_margin", "operating_quality_risk_margin"),
        ("operating_roa", "operating_quality_risk_roa"),
    ]:
        series = df[col].replace([np.inf, -np.inf], np.nan)
        median = series.median()
        mad = (series - median).abs().median()
        mad = mad if mad > 0 else 1e-6
        z_like = ((series - median).abs() / mad).clip(upper=10)
        df[risk_col] = MinMaxScaler((0, 100)).fit_transform(z_like.fillna(0).to_frame())

    df["operating_quality_risk"] = (
        0.5 * df["operating_quality_risk_margin"] +
        0.5 * df["operating_quality_risk_roa"]
    )

    # Leverage risk
    for col, risk_col in [
        ("equity_multiplier", "leverage_risk_em"),
        ("leverage_de_ratio", "leverage_risk_de"),
    ]:
        series = df[col].replace([np.inf, -np.inf], np.nan)
        median = series.median()
        mad = (series - median).abs().median()
        mad = mad if mad > 0 else 1e-6
        z_like = ((series - median).abs() / mad).clip(upper=10)
        df[risk_col] = MinMaxScaler((0, 100)).fit_transform(z_like.fillna(0).to_frame())

    df["leverage_risk"] = (
        0.5 * df["leverage_risk_em"] +
        0.5 * df["leverage_risk_de"]
    )

    # Final fraud score
    def compute_fraud_score(row):
        op = row.get("operating_quality_risk", np.nan)
        lev = row.get("leverage_risk", np.nan)
        anom = row.get("pattern_anomaly_score", np.nan)

        if pd.isna(anom):
            if pd.isna(op) or pd.isna(lev):
                return np.nan
            return 0.6 * op + 0.4 * lev
        else:
            return 0.4 * op + 0.3 * lev + 0.3 * anom

    df["dupont_fraud_score"] = df.apply(compute_fraud_score, axis=1)

    return df


def risk_label(score):
    if pd.isna(score):
        return "N/A"
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"


# ======================================================
# Agent 4: LLM-based Explanation
# ======================================================

def get_llm():
    """Return a LangChain ChatOpenAI instance if API key is available, else None."""
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        return None
    # You can change the model name if needed
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    return llm


def generate_llm_explanation(row, df_company):
    """
    Use LLM to generate a natural language explanation of
    ROE drivers, anomalies and DuPont Fraud Score.
    """
    llm = get_llm()
    if llm is None:
        return None  # caller will fall back to rule-based explanation

    # Prepare context features
    year = int(row["year"])
    company = row["company"]
    roe = float(row["roe"]) * 100
    fraud_score = float(row["dupont_fraud_score"]) if not pd.isna(row["dupont_fraud_score"]) else None
    op_risk = float(row["operating_quality_risk"]) if not pd.isna(row["operating_quality_risk"]) else None
    lev_risk = float(row["leverage_risk"]) if not pd.isna(row["leverage_risk"]) else None
    patt_score = float(row["pattern_anomaly_score"]) if not pd.isna(row["pattern_anomaly_score"]) else None

    npm = float(row["net_profit_margin"]) * 100
    at = float(row["asset_turnover"])
    em = float(row["equity_multiplier"])

    op_roa = float(row["operating_roa"]) * 100
    lev_de = float(row["leverage_de_ratio"])

    # Basic history: last 5 years roe trend
    df_hist = df_company.sort_values("year").tail(5)
    roe_hist = [
        {"year": int(r["year"]), "roe_pct": float(r["roe"]) * 100}
        for _, r in df_hist.iterrows()
    ]

    prompt = ChatPromptTemplate.from_template(
        """
You are a forensic financial analyst. You are given DuPont and Modified DuPont results
for a company and an AI-based DuPont Fraud Score. Explain what is happening in clear,
professional language (150-220 words), in the style of an equity research + forensic audit note.

Company: {company}
Year: {year}

Key metrics:
- ROE (this year): {roe:.2f} %
- Net Profit Margin: {npm:.2f} %
- Asset Turnover: {at:.2f} x
- Equity Multiplier: {em:.2f} x
- Operating ROA: {op_roa:.2f} %
- Leverage (Debt/Equity): {lev_de:.2f} x

Risk & anomaly indicators:
- DuPont Fraud Score (0–100, higher = more abnormal): {fraud_score}
- Operating Quality Risk (0–100): {op_risk}
- Leverage Risk (0–100): {lev_risk}
- Pattern Anomaly Score (0–100): {patt_score}

Recent ROE history (last few years):
{roe_history}

Tasks:
1. Explain the main drivers of ROE using margins, efficiency, and leverage.
2. Comment on whether ROE seems to come from genuine operating strength or from leverage/capital structure.
3. Interpret the DuPont Fraud Score and risk components, mentioning whether they suggest low, moderate, or high concern.
4. Highlight any red flags or unusual patterns a forensic auditor should investigate further.

Keep it factual, avoid sensational language, and DO NOT recommend buy/sell – only analyze risk and quality.
"""
    )

    messages = prompt.format_messages(
        company=company,
        year=year,
        roe=roe,
        npm=npm,
        at=at,
        em=em,
        op_roa=op_roa,
        lev_de=lev_de,
        fraud_score=fraud_score,
        op_risk=op_risk,
        lev_risk=lev_risk,
        patt_score=patt_score,
        roe_history=roe_hist,
    )

    response = llm(messages)
    return response.content


# ======================================================
# Streamlit App (Agents 1, 2, 3 orchestrated)
# ======================================================

st.set_page_config(
    page_title="Intelligent DuPont Fraud Score",
    layout="wide"
)

st.title("🧠 Intelligent DuPont Fraud Score – Multi-Agent Forensic Dashboard")

st.write(
    """
This app uses a **multi-agent workflow** for forensic financial analysis:

- **Agent 1 – Data Agent:** reads your uploaded dataset.
- **Agent 2 – Analysis Agent:** computes DuPont & Modified DuPont and applies AI anomaly detection.
- **Agent 3 – Dashboard Agent:** builds interactive visualizations and scores.
- **Agent 4 – Explanation Agent:** uses a Large Language Model (via LangChain) to explain the results.

Upload a CSV with multi-year financials to get started.
"""
)

# -----------------------------
# Agent 1: Data Agent – File upload
# -----------------------------

st.sidebar.header("1️⃣ Upload Financial Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (one row per company per year)",
    type=["csv"],
    help=(
        "Required columns: company, industry, year, "
        "sales, net_income, ebit, ebt, total_assets, total_equity, total_debt, "
        "interest_expense, tax_expense"
    ),
)

use_sample = st.sidebar.checkbox("Use sample synthetic data instead", value=False)

if use_sample and uploaded_file is None:
    # Create synthetic dataset
    np.random.seed(42)
    companies = ["Alpha Ltd", "Beta Corp", "Gamma Industries"]
    years = list(range(2016, 2025))
    rows = []
    for c in companies:
        base_sales = np.random.randint(800, 1500) * 1e6
        for y in years:
            growth = np.random.uniform(0.95, 1.12)
            base_sales *= growth
            sales = base_sales
            ebit_margin = np.random.uniform(0.09, 0.17)
            ebit = sales * ebit_margin
            interest = np.random.uniform(0.02, 0.05) * sales
            ebt = ebit - interest
            tax = np.random.uniform(0.20, 0.30) * max(ebt, 0)
            net_income = ebt - tax
            assets = sales * np.random.uniform(0.7, 1.2)
            equity = assets * np.random.uniform(0.35, 0.55)
            debt = assets - equity

            rows.append({
                "company": c,
                "industry": "Synthetic Manufacturing",
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
elif uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = None

if df_raw is None:
    st.info("👈 Upload a CSV or tick **Use sample synthetic data** to see the app in action.")
    st.stop()

required_cols = [
    "company", "industry", "year",
    "sales", "net_income", "ebit", "ebt",
    "total_assets", "total_equity", "total_debt",
    "interest_expense", "tax_expense",
]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in your file: {missing}")
    st.stop()

df_raw["year"] = pd.to_numeric(df_raw["year"], errors="coerce")
df_raw = df_raw.dropna(subset=["year"])
df_raw["year"] = df_raw["year"].astype(int)

# -----------------------------
# Agent 2: Analysis Agent
# -----------------------------

df = compute_dupont_components(df_raw)
df = add_ai_anomaly_scores(df)

# -----------------------------
# Agent 3: Dashboard Agent – UI
# -----------------------------

st.sidebar.header("2️⃣ Select Company & Year")

companies = sorted(df["company"].unique().tolist())
selected_company = st.sidebar.selectbox("Company", companies)

df_company = df[df["company"] == selected_company].sort_values("year")
years = df_company["year"].unique().tolist()
selected_year = st.sidebar.selectbox(
    "Year (for detailed view)",
    years,
    index=len(years) - 1
)
row_selected = df_company[df_company["year"] == selected_year].iloc[0]

st.subheader(f"📊 Overview: {selected_company} – {selected_year}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="ROE",
        value=f"{row_selected['roe'] * 100:,.2f} %",
        help="Return on Equity = Net Income / Equity"
    )

with col2:
    st.metric(
        label="DuPont Fraud Score",
        value=f"{row_selected['dupont_fraud_score']:.1f}"
        if not pd.isna(row_selected["dupont_fraud_score"]) else "N/A",
        help="0–100: higher indicates more abnormal patterns"
    )
    st.caption(f"Risk Level: **{risk_label(row_selected['dupont_fraud_score'])}**")

with col3:
    st.metric(
        label="Operating Quality Risk",
        value=f"{row_selected['operating_quality_risk']:.1f}"
        if not pd.isna(row_selected["operating_quality_risk"]) else "N/A",
        help="0–100: abnormality in operating margins / ROA"
    )

with col4:
    st.metric(
        label="Leverage Risk",
        value=f"{row_selected['leverage_risk']:.1f}"
        if not pd.isna(row_selected["leverage_risk"]) else "N/A",
        help="0–100: abnormality in leverage vs peers/history"
    )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["ROE & Classic DuPont", "Modified DuPont & Operating vs Leverage", "AI & Explanation Agent", "Raw Data"]
)

with tab1:
    st.subheader("🔹 Classic DuPont (3-step)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Net Profit Margin",
            f"{row_selected['net_profit_margin'] * 100:,.2f} %",
            help="Net Income / Sales"
        )
    with c2:
        st.metric(
            "Asset Turnover",
            f"{row_selected['asset_turnover']:.2f} x",
            help="Sales / Total Assets"
        )
    with c3:
        st.metric(
            "Equity Multiplier",
            f"{row_selected['equity_multiplier']:.2f} x",
            help="Total Assets / Total Equity"
        )

    st.markdown("#### ROE and DuPont Components Over Time")
    df_plot = df_company[["year", "roe", "net_profit_margin", "asset_turnover", "equity_multiplier"]].copy()
    st.line_chart(df_plot.set_index("year"), height=320)

with tab2:
    st.subheader("🔹 Modified DuPont (5-step) & Economic Decomposition")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Tax Burden", f"{row_selected['tax_burden']:.2f}", help="Net Income / EBT")
    with c2:
        st.metric("Interest Burden", f"{row_selected['interest_burden']:.2f}", help="EBT / EBIT")
    with c3:
        st.metric("Operating Margin", f"{row_selected['operating_margin'] * 100:,.2f} %", help="EBIT / Sales")
    with c4:
        st.metric("Asset Turnover", f"{row_selected['asset_turnover']:.2f} x")
    with c5:
        st.metric("Equity Multiplier", f"{row_selected['equity_multiplier']:.2f} x")

    st.markdown("#### Operating ROA, Leverage & Spread Over Time")
    df_op = df_company[["year", "operating_roa", "leverage_de_ratio", "spread"]].copy()
    st.line_chart(df_op.set_index("year"), height=320)

with tab3:
    st.subheader("🔹 AI & Explanation Agent")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Pattern Anomaly Score",
            f"{row_selected['pattern_anomaly_score']:.1f}"
            if not pd.isna(row_selected["pattern_anomaly_score"]) else "N/A",
            help="0–100: anomaly in DuPont pattern via IsolationForest"
        )
    with c2:
        st.metric(
            "Operating Quality Risk",
            f"{row_selected['operating_quality_risk']:.1f}"
            if not pd.isna(row_selected["operating_quality_risk"]) else "N/A"
        )
    with c3:
        st.metric(
            "Leverage Risk",
            f"{row_selected['leverage_risk']:.1f}"
            if not pd.isna(row_selected["leverage_risk"]) else "N/A"
        )

    st.markdown("#### DuPont Fraud Score & Risk Components Over Time")

    df_risk = df_company[[
        "year",
        "dupont_fraud_score",
        "operating_quality_risk",
        "leverage_risk",
        "pattern_anomaly_score"
    ]].set_index("year")
    st.line_chart(df_risk, height=320)

    st.markdown("#### Explanation Agent Output")

    explanation = generate_llm_explanation(row_selected, df_company)

    if explanation is not None:
        st.write(explanation)
        st.caption(
            "Generated by LangChain + OpenAI based on DuPont metrics, fraud scores and historical context."
        )
    else:
        # Fallback: simple rule-based explanation
        explanation_lines = []

        if row_selected["dupont_fraud_score"] >= 60:
            explanation_lines.append(
                "- The **DuPont Fraud Score is high**, indicating significant abnormality "
                "in either operating metrics, leverage, or overall DuPont patterns."
            )
        elif row_selected["dupont_fraud_score"] >= 30:
            explanation_lines.append(
                "- The **DuPont Fraud Score is moderate**, with some warning signals "
                "but not extreme anomalies."
            )
        else:
            explanation_lines.append(
                "- The **DuPont Fraud Score is low**, suggesting that ROE behaviour is broadly "
                "consistent with underlying fundamentals."
            )

        if row_selected["operating_quality_risk"] > row_selected["leverage_risk"]:
            explanation_lines.append(
                "- **Operating quality risk exceeds leverage risk**, meaning that margins and operating "
                "ROI behave more unusually than capital structure."
            )
        elif row_selected["leverage_risk"] > row_selected["operating_quality_risk"]:
            explanation_lines.append(
                "- **Leverage risk exceeds operating risk**, indicating that ROE is more dependent on "
                "debt and capital structure than on core operations."
            )

        if not pd.isna(row_selected["pattern_anomaly_score"]) and row_selected["pattern_anomaly_score"] > 60:
            explanation_lines.append(
                "- The DuPont pattern for this year is **statistically anomalous** compared to the overall "
                "dataset, which justifies deeper forensic review."
            )

        if not explanation_lines:
            explanation_lines.append(
                "- The rules did not detect strong red flags for this year, but professional judgment "
                "and detailed audit procedures are still required."
            )

        st.write("\n".join(explanation_lines))
        st.caption(
            "Explanation generated using rule-based logic because no LLM API key was detected."
        )

with tab4:
    st.subheader("🔹 Raw Data & Computed Metrics")
    st.write("Filtered data for this company:")
    st.dataframe(df_company.sort_values("year"), use_container_width=True)
