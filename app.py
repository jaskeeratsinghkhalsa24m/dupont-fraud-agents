import os
import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ============================================================
# AGENT 1: Workbook ingestion + worksheet discovery
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook_sheets(uploaded_file) -> dict:
    """Read all sheets from an .xlsx and return dict {sheet_name: cleaned_df}."""
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheets = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, engine="openpyxl")
        df = df.dropna(how="all").dropna(axis=1, how="all")
        sheets[name] = df
    return sheets


def classify_sheet(df: pd.DataFrame, sheet_name: str) -> str:
    """Heuristic classification: income_statement / balance_sheet / cashflow / other."""
    text = (sheet_name + " " + " ".join(map(str, df.columns))).lower()

    income_kw = ["income", "profit", "loss", "p&l", "p and l", "revenue", "sales", "ebit", "pbt", "pat", "expenses"]
    bs_kw = ["balance", "assets", "liabilities", "equity", "shareholders", "borrowings", "net worth"]
    cf_kw = ["cash", "cashflow", "cash flow", "operating activities", "investing activities", "financing activities"]

    def score(keywords):
        return sum(1 for k in keywords if k in text)

    s_income = score(income_kw)
    s_bs = score(bs_kw)
    s_cf = score(cf_kw)

    best = max(s_income, s_bs, s_cf)
    if best == 0:
        return "other"
    if best == s_income:
        return "income_statement"
    if best == s_bs:
        return "balance_sheet"
    return "cashflow"


def _guess_label_column(df: pd.DataFrame):
    """Pick a likely label/line-item column (mostly text)."""
    best_col, best_ratio = None, -1
    for c in df.columns:
        s = df[c].astype(str)
        text_ratio = (s.str.contains(r"[A-Za-z]", regex=True, na=False)).mean()
        if text_ratio > best_ratio:
            best_ratio = text_ratio
            best_col = c
    return best_col


def _guess_year_columns(df: pd.DataFrame, label_col):
    """Pick year columns based on header containing digits."""
    year_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        cs = str(c)
        if any(ch.isdigit() for ch in cs):
            year_cols.append(c)
    return year_cols


def _standardize_year_header(x):
    """Convert headers like FY 2022, 2022-23, FY22 etc into year int when possible."""
    s = str(x)
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(\d{2})\b", s)
    # FY22 often means 2022 (assume 20xx)
    if m2:
        yy = int(m2.group(1))
        if 0 <= yy <= 99:
            return 2000 + yy
    return None


def extract_timeseries_from_statement(
    df: pd.DataFrame,
    label_map: dict,
    label_col=None,
    year_cols=None
) -> pd.DataFrame:
    """
    Extract year-wise values from a statement where:
      - a label column contains line items
      - other columns represent years
    label_map: canonical_field -> list[keyword]
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if label_col is None:
        label_col = _guess_label_column(df)
    if label_col is None:
        return pd.DataFrame()

    if year_cols is None:
        year_cols = _guess_year_columns(df, label_col)
    if not year_cols:
        return pd.DataFrame()

    # Prepare label text
    labels = df[label_col].astype(str).str.lower().str.strip()

    # Build extraction
    out = {"year": []}
    for canon in label_map.keys():
        out[canon] = []

    # Convert year headers
    year_header_to_year = {}
    for yc in year_cols:
        yr = _standardize_year_header(yc)
        year_header_to_year[yc] = yr

    for yc in year_cols:
        yr = year_header_to_year.get(yc)
        out["year"].append(yr if yr is not None else str(yc))

        for canon, keywords in label_map.items():
            # choose first row match by keyword
            idx = None
            for i, lab in enumerate(labels):
                if any(k in lab for k in keywords):
                    idx = i
                    break
            if idx is None:
                out[canon].append(np.nan)
            else:
                val = df.iloc[idx][yc]
                out[canon].append(pd.to_numeric(val, errors="coerce"))

    res = pd.DataFrame(out)
    # Try to coerce year to int if possible
    res["year"] = pd.to_numeric(res["year"], errors="coerce")
    res = res.dropna(subset=["year"])
    if not res.empty:
        res["year"] = res["year"].astype(int)
    return res


def extraction_quality(df_std: pd.DataFrame, required_fields: list) -> float:
    """Fraction of non-null values across required fields."""
    if df_std.empty:
        return 0.0
    total = len(df_std) * len(required_fields)
    nonnull = df_std[required_fields].notna().sum().sum()
    return nonnull / max(total, 1)


def manual_metric_picker(df: pd.DataFrame, label_col, year_cols, canonical_to_keywords):
    """
    Manual rescue UI: user selects the exact row for each canonical metric.
    Returns a standardized year-wise DataFrame.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    labels = df[label_col].astype(str)
    labels_l = labels.str.lower()

    # Map year headers -> int year
    years = []
    for yc in year_cols:
        yr = _standardize_year_header(yc)
        years.append((yc, yr if yr is not None else str(yc)))

    out = {"year": [y for _, y in years]}
    for canon in canonical_to_keywords.keys():
        out[canon] = [np.nan] * len(years)

    st.markdown("### Manual mapping (rescue)")
    st.caption("Pick the exact row for each metric. This makes the tool work with almost any statement layout.")

    for canon, kws in canonical_to_keywords.items():
        st.markdown(f"**Select row for:** `{canon}`")
        default_search = kws[0] if kws else ""
        query = st.text_input(f"Search label for {canon}", value=default_search, key=f"search_{canon}")

        # Candidate rows based on search
        if query.strip():
            mask = labels_l.str.contains(query.lower(), na=False)
            candidates = labels[mask].head(30).tolist()
        else:
            candidates = labels.head(30).tolist()

        if not candidates:
            candidates = labels.head(30).tolist()

        selected_label = st.selectbox(f"Row label for {canon}", options=candidates, key=f"pick_{canon}")
        # Find first matching index
        idx_list = df.index[labels == selected_label].tolist()
        idx = idx_list[0] if idx_list else None

        if idx is None:
            continue

        # Fill year-wise values
        for j, (yc, yr) in enumerate(years):
            val = df.loc[idx, yc]
            out[canon][j] = pd.to_numeric(val, errors="coerce")

        st.divider()

    res = pd.DataFrame(out)
    res["year"] = pd.to_numeric(res["year"], errors="coerce")
    res = res.dropna(subset=["year"])
    if not res.empty:
        res["year"] = res["year"].astype(int)
    return res


# Canonical label maps
PL_LABELS = {
    "sales": ["revenue", "sales", "turnover", "total income", "income from operations"],
    "ebit": ["ebit", "operating profit", "profit from operations", "operating income"],
    "ebt": ["profit before tax", "pbt", "ebt", "profit before taxation"],
    "net_income": ["profit after tax", "pat", "net profit", "profit for the year"],
    "interest_expense": ["finance cost", "finance costs", "interest", "interest expense", "borrowing cost"],
    "tax_expense": ["tax", "income tax", "tax expense", "taxation"],
}

BS_LABELS = {
    "total_assets": ["total assets", "assets"],
    "total_equity": ["total equity", "shareholders", "shareholder", "net worth", "equity"],
    "total_debt": ["borrowings", "total debt", "loans", "debt"],
}


# ============================================================
# AGENT 2: DuPont + Modified DuPont + AI anomaly scoring
# ============================================================

def compute_dupont_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9

    df["roe"] = df["net_income"] / (df["total_equity"] + eps)

    df["net_profit_margin"] = df["net_income"] / (df["sales"] + eps)
    df["asset_turnover"] = df["sales"] / (df["total_assets"] + eps)
    df["equity_multiplier"] = df["total_assets"] / (df["total_equity"] + eps)

    df["tax_burden"] = df["net_income"] / (df["ebt"] + eps)
    df["interest_burden"] = df["ebt"] / (df["ebit"] + eps)
    df["operating_margin"] = df["ebit"] / (df["sales"] + eps)

    df["tax_rate_est"] = df["tax_expense"] / (df["ebt"] + eps)
    df["tax_rate_est"] = df["tax_rate_est"].clip(lower=0, upper=0.7)

    df["nopat"] = df["ebit"] * (1 - df["tax_rate_est"])
    df["operating_roa"] = df["nopat"] / (df["total_assets"] + eps)

    df["gross_cost_of_debt"] = df["interest_expense"] / (df["total_debt"] + eps)
    df["after_tax_cost_of_debt"] = df["gross_cost_of_debt"] * (1 - df["tax_rate_est"])

    df["spread"] = df["operating_roa"] - df["after_tax_cost_of_debt"]
    df["leverage_de_ratio"] = df["total_debt"] / (df["total_equity"] + eps)

    df["roe_from_spread"] = df["operating_roa"] + df["spread"] * df["leverage_de_ratio"]

    return df


def add_ai_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    feature_cols = [
        "roe", "net_profit_margin", "asset_turnover", "equity_multiplier",
        "tax_burden", "interest_burden", "operating_margin",
        "operating_roa", "leverage_de_ratio", "spread"
    ]

    feature_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_idx = feature_df.dropna().index

    df["pattern_anomaly_score"] = np.nan
    df["anomaly_score_raw"] = np.nan

    if len(valid_idx) >= 10:
        X = feature_df.loc[valid_idx].values
        iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42)
        iso.fit(X)

        raw = -iso.score_samples(X)
        scaled = MinMaxScaler((0, 100)).fit_transform(raw.reshape(-1, 1)).flatten()

        df.loc[valid_idx, "anomaly_score_raw"] = raw
        df.loc[valid_idx, "pattern_anomaly_score"] = scaled

    def robust_risk(series: pd.Series) -> pd.Series:
        s = series.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        mad = (s - med).abs().median()
        mad = mad if mad and mad > 0 else 1e-6
        z = ((s - med).abs() / mad).clip(upper=10)
        return pd.Series(
            MinMaxScaler((0, 100)).fit_transform(z.fillna(0).to_frame()).flatten(),
            index=series.index
        )

    df["operating_quality_risk"] = 0.5 * robust_risk(df["operating_margin"]) + 0.5 * robust_risk(df["operating_roa"])
    df["leverage_risk"] = 0.5 * robust_risk(df["equity_multiplier"]) + 0.5 * robust_risk(df["leverage_de_ratio"])

    def fraud_score(row):
        op, lev, anom = row["operating_quality_risk"], row["leverage_risk"], row["pattern_anomaly_score"]
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


@st.cache_data(show_spinner=True)
def run_analysis(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = compute_dupont_components(df_raw)
    df = add_ai_anomaly_scores(df)
    return df


# ============================================================
# AGENT 4: LLM explanation (LangChain + OpenAI)
# ============================================================

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
You are a forensic financial analyst. Write a 150–220 word explanation based on DuPont and AI anomaly scores.

Company: {company}
Year: {year}

Metrics:
ROE: {roe:.2f}%
Net Profit Margin: {npm:.2f}%
Asset Turnover: {at:.2f}x
Equity Multiplier: {em:.2f}x
Operating ROA: {op_roa:.2f}%
Debt/Equity: {de:.2f}x

AI indicators (0–100):
DuPont Fraud Score: {fraud:.1f}
Operating Quality Risk: {op_risk:.1f}
Leverage Risk: {lev_risk:.1f}
Pattern Anomaly Score: {anom}

Recent ROE history: {roe_history}

Explain:
1) Main drivers of ROE (margin vs efficiency vs leverage).
2) Whether ROE quality looks operational or leverage-driven.
3) What the score implies (low/moderate/high concern).
4) What an auditor should check next (2–3 concrete checks).
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


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Intelligent DuPont Fraud Score (XLSX)", layout="wide")
st.title("🧠 Intelligent DuPont Fraud Score – XLSX Multi-Sheet (Agent Workflow)")

st.write(
    "Upload an **Excel (.xlsx)** file with multiple worksheets. "
    "**Agent 1** discovers sheets and extracts key statement items, "
    "**Agent 2** computes DuPont + anomaly score, "
    "**Agent 3** visualizes it, "
    "**Agent 4** explains it via LangChain + LLM."
)

st.sidebar.header("📥 Upload Excel")
uploaded = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel (.xlsx) file to begin.")
    st.stop()

# Agent 1: load workbook
sheets = load_workbook_sheets(uploaded)
sheet_names = list(sheets.keys())
sheet_types = {n: classify_sheet(sheets[n], n) for n in sheet_names}

st.sidebar.subheader("Agent 1: Worksheet Discovery")
st.sidebar.write(f"Found **{len(sheet_names)}** sheets")

income_candidates = [n for n, t in sheet_types.items() if t == "income_statement"]
bs_candidates = [n for n, t in sheet_types.items() if t == "balance_sheet"]

income_sheet = st.sidebar.selectbox(
    "Select Income Statement / P&L sheet",
    options=sheet_names,
    index=sheet_names.index(income_candidates[0]) if income_candidates else 0
)
bs_sheet = st.sidebar.selectbox(
    "Select Balance Sheet sheet",
    options=sheet_names,
    index=sheet_names.index(bs_candidates[0]) if bs_candidates else 0
)

st.subheader("🧾 Agent 1 Output: Workbook structure")
cA, cB = st.columns(2)
with cA:
    st.write("**Income Statement / P&L Preview**")
    st.caption(f"Selected: {income_sheet} (auto-tag: {sheet_types[income_sheet]})")
    st.dataframe(sheets[income_sheet].head(20), use_container_width=True)

with cB:
    st.write("**Balance Sheet Preview**")
    st.caption(f"Selected: {bs_sheet} (auto-tag: {sheet_types[bs_sheet]})")
    st.dataframe(sheets[bs_sheet].head(20), use_container_width=True)

st.sidebar.subheader("Metadata")
company_name = st.sidebar.text_input("Company name", value="Unknown Company")
industry_name = st.sidebar.text_input("Industry", value="Unknown Industry")

st.sidebar.subheader("Extraction Mode")
mode = st.sidebar.radio("Choose extraction mode", ["Auto (recommended)", "Manual (if auto fails)"], index=0)

# Agent 1: extract
pl_source = sheets[income_sheet]
bs_source = sheets[bs_sheet]

if mode == "Auto (recommended)":
    pl_df = extract_timeseries_from_statement(pl_source, PL_LABELS)
    bs_df = extract_timeseries_from_statement(bs_source, BS_LABELS)

else:
    # Manual config for PL
    st.markdown("## Manual extraction configuration")
    st.caption("First configure P&L, then Balance Sheet.")

    st.markdown("### P&L Manual Config")
    label_col_pl = st.selectbox("P&L label column", options=list(pl_source.columns), index=0)
    year_cols_pl = st.multiselect("P&L year columns", options=list(pl_source.columns), default=[c for c in pl_source.columns[1:5]])
    pl_df = manual_metric_picker(pl_source, label_col_pl, year_cols_pl, PL_LABELS)

    st.markdown("### Balance Sheet Manual Config")
    label_col_bs = st.selectbox("BS label column", options=list(bs_source.columns), index=0)
    year_cols_bs = st.multiselect("BS year columns", options=list(bs_source.columns), default=[c for c in bs_source.columns[1:5]])
    bs_df = manual_metric_picker(bs_source, label_col_bs, year_cols_bs, BS_LABELS)

# Merge standardized dataset
if pl_df.empty or bs_df.empty:
    st.error("Extraction produced no usable year-wise data. Switch to Manual mode and map rows/years explicitly.")
    st.stop()

df_raw = pd.merge(pl_df, bs_df, on="year", how="outer")
df_raw["company"] = company_name
df_raw["industry"] = industry_name

# Required fields for analysis
required_fin_cols = [
    "sales", "net_income", "ebit", "ebt",
    "total_assets", "total_equity", "total_debt",
    "interest_expense", "tax_expense"
]

# Show extraction quality
quality = extraction_quality(df_raw, required_fin_cols)
st.subheader("✅ Agent 1 Output: Standardized dataset")
st.caption(f"Extraction completeness score: **{quality:.0%}** (higher is better). If low, use Manual mode.")
st.dataframe(df_raw.sort_values("year"), use_container_width=True)

# Basic guardrails
if quality < 0.45:
    st.warning(
        "Extraction completeness is low. This usually means your sheet layout/labels differ from the defaults. "
        "Switch to **Manual** extraction mode for precise row selection."
    )

# Agent 2: analysis
df_raw_numeric = df_raw.copy()
for col in required_fin_cols:
    df_raw_numeric[col] = pd.to_numeric(df_raw_numeric[col], errors="coerce")

df_raw_numeric = df_raw_numeric.dropna(subset=["year"])
df_raw_numeric["year"] = df_raw_numeric["year"].astype(int)

df = run_analysis(df_raw_numeric)

# Agent 3: dashboard selection
st.sidebar.header("📌 Select Year")
years = df["year"].sort_values().unique().tolist()
if not years:
    st.error("No valid years after cleaning. Check extraction and ensure year columns are correct.")
    st.stop()

selected_year = st.sidebar.selectbox("Year (detailed view)", years, index=len(years) - 1)
row = df[df["year"] == selected_year].iloc[0]

# Top KPIs
st.subheader(f"📊 Dashboard: {company_name} — {selected_year}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ROE", f"{row['roe']*100:.2f}%")
k2.metric("DuPont Fraud Score", f"{row['dupont_fraud_score']:.1f}")
k2.caption(f"Risk: **{risk_label(row['dupont_fraud_score'])}**")
k3.metric("Operating Quality Risk", f"{row['operating_quality_risk']:.1f}")
k4.metric("Leverage Risk", f"{row['leverage_risk']:.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Classic DuPont", "Modified DuPont", "AI Explanation Agent", "Full Table"])

with tab1:
    st.subheader("Classic 3-step DuPont")
    a, b, c = st.columns(3)
    a.metric("Net Profit Margin", f"{row['net_profit_margin']*100:.2f}%")
    b.metric("Asset Turnover", f"{row['asset_turnover']:.2f}x")
    c.metric("Equity Multiplier", f"{row['equity_multiplier']:.2f}x")

    st.markdown("#### Trend: ROE + DuPont Components")
    st.line_chart(df.set_index("year")[["roe", "net_profit_margin", "asset_turnover", "equity_multiplier"]])

with tab2:
    st.subheader("Modified DuPont (5-step) + Operating vs Leverage")
    a, b, c, d, e = st.columns(5)
    a.metric("Tax Burden (NI/EBT)", f"{row['tax_burden']:.2f}")
    b.metric("Interest Burden (EBT/EBIT)", f"{row['interest_burden']:.2f}")
    c.metric("Operating Margin (EBIT/Sales)", f"{row['operating_margin']*100:.2f}%")
    d.metric("Operating ROA", f"{row['operating_roa']*100:.2f}%")
    e.metric("Debt/Equity", f"{row['leverage_de_ratio']:.2f}x")

    st.markdown("#### Trend: Operating ROA, Leverage, Spread")
    st.line_chart(df.set_index("year")[["operating_roa", "leverage_de_ratio", "spread"]])

with tab3:
    st.subheader("AI + Explanation Agent (LangChain)")
    st.write("Pattern Anomaly Score:", "N/A" if pd.isna(row["pattern_anomaly_score"]) else f"{row['pattern_anomaly_score']:.1f}")

    explanation = generate_llm_explanation(row, df.sort_values("year"))
    if explanation:
        st.write(explanation)
        st.caption("Generated via LangChain + OpenAI. Add OPENAI_API_KEY in Streamlit Secrets.")
    else:
        st.warning("OPENAI_API_KEY not found (or not set in Streamlit Secrets). Using rule-based explanation.")
        lines = []
        lines.append(f"- DuPont Fraud Score is **{row['dupont_fraud_score']:.1f}** ({risk_label(row['dupont_fraud_score'])}).")
        if row["leverage_risk"] > row["operating_quality_risk"]:
            lines.append("- Leverage risk dominates: ROE may be amplified by capital structure rather than operations.")
        else:
            lines.append("- Operating risk dominates: margins/operating ROA look more unusual than leverage.")
        if not pd.isna(row["pattern_anomaly_score"]) and row["pattern_anomaly_score"] > 60:
            lines.append("- The overall DuPont pattern is statistically unusual vs the dataset; audit attention warranted.")
        lines.append("- Next checks: revenue recognition notes, one-off income/expenses, related-party items, debt structure & covenants.")
        st.write("\n".join(lines))

with tab4:
    st.subheader("Full computed table (all years)")
    st.dataframe(df.sort_values("year"), use_container_width=True)
