import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ============================================================
# AGENT 1: XLSX ingestion + layout understanding
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook_sheets(uploaded_file) -> dict:
    """
    Read all sheets as raw grids (header=None) so we can detect internal header rows
    like 'Mar-16 ... Mar-25' even when pandas would call them Unnamed columns.
    """
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheets = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, engine="openpyxl", header=None)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        sheets[name] = df
    return sheets


def classify_sheet(df_raw: pd.DataFrame, sheet_name: str) -> str:
    """
    Heuristic classification using sheet name and a small text scan of top rows.
    """
    sample_text = " ".join(df_raw.head(20).astype(str).fillna("").values.ravel().tolist())
    text = (sheet_name + " " + sample_text).lower()

    income_kw = ["profit", "loss", "p&l", "p and l", "income statement", "revenue", "sales", "operating profit", "pat", "pbt"]
    bs_kw = ["balance sheet", "assets", "liabilities", "equity", "net worth", "borrowings", "reserves", "share capital"]
    cf_kw = ["cash flow", "cashflow", "operating activities", "investing activities", "financing activities"]

    def score(keys):
        return sum(1 for k in keys if k in text)

    si, sb, sc = score(income_kw), score(bs_kw), score(cf_kw)
    best = max(si, sb, sc)
    if best == 0:
        return "other"
    if best == si:
        return "income_statement"
    if best == sb:
        return "balance_sheet"
    return "cashflow"


def detect_year_header_row(df_raw: pd.DataFrame, max_scan_rows: int = 120):
    """
    Detect the internal header row that contains multiple year-like tokens.
    Works for: Mar-16, Mar-17, FY22, 2022, 2022-23 etc.
    Returns: (header_row_index, year_col_indices, year_labels)
    """
    patterns = [
        r"\bmar-\d{2}\b", r"\bjun-\d{2}\b", r"\bsep-\d{2}\b", r"\bdec-\d{2}\b",
        r"\bfy\s?\d{2,4}\b", r"\b20\d{2}\b", r"\b\d{4}-\d{2}\b"
    ]

    best = None  # (row_idx, matches[(col_idx, token)])
    nrows = min(len(df_raw), max_scan_rows)

    for i in range(nrows):
        row = df_raw.iloc[i].astype(str).fillna("").str.lower().map(lambda x: x.strip())
        matches = []
        for j, cell in enumerate(row.tolist()):
            if cell in ["", "nan", "none"]:
                continue
            if any(re.search(p, cell) for p in patterns):
                matches.append((j, cell))

        # Need at least 4 year-like headers to be confident
        if len(matches) >= 4:
            if best is None or len(matches) > len(best[1]):
                best = (i, matches)

    if best is None:
        return None, [], []

    header_row, matches = best
    year_cols = [j for j, _ in matches]
    year_labels = [df_raw.iloc[header_row, j] for j in year_cols]
    return header_row, year_cols, year_labels


def build_statement_table(df_raw: pd.DataFrame):
    """
    Converts a raw statement grid into a clean table by:
      - finding the year header row inside the sheet
      - selecting label column (Narration-like) from the left
      - using year labels as columns
    Returns: (table_df, label_col_name, year_col_names)
    """
    header_row, year_cols_idx, year_labels = detect_year_header_row(df_raw)
    if header_row is None:
        return None, None, None

    # Label column: pick a likely narration col from the first few columns
    # Choose the column with highest text ratio in rows below header_row.
    candidate_cols = list(range(min(5, df_raw.shape[1])))
    best_label_idx, best_ratio = 0, -1
    for c in candidate_cols:
        s = df_raw.iloc[header_row+1:header_row+60, c].astype(str).fillna("")
        text_ratio = (s.str.contains(r"[A-Za-z]", regex=True)).mean()
        if text_ratio > best_ratio:
            best_ratio = text_ratio
            best_label_idx = c

    use_cols = [best_label_idx] + year_cols_idx
    sub = df_raw.iloc[header_row+1:, use_cols].copy()

    # Name columns
    label_col_name = "Narration"
    year_col_names = [str(x).strip() for x in year_labels]
    sub.columns = [label_col_name] + year_col_names

    # Clean narration
    sub[label_col_name] = sub[label_col_name].astype(str).fillna("").str.strip()
    sub = sub[sub[label_col_name] != ""]
    sub = sub[~sub[label_col_name].str.lower().isin(["nan", "none"])]

    # Remove separator blocks
    sub = sub[~sub[label_col_name].str.lower().str.contains(r"ratios|trends|---", na=False)]

    # Numeric coercion for year columns
    for yc in year_col_names:
        sub[yc] = pd.to_numeric(sub[yc], errors="coerce")

    return sub, label_col_name, year_col_names


def year_to_int_from_header(h):
    """
    Convert 'Mar-16' -> 2016, 'FY22' -> 2022, '2022-23' -> 2022, '2022' -> 2022.
    If cannot, return None.
    """
    s = str(h).lower().strip()

    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))

    m = re.search(r"\b(\d{2})\b", s)  # Mar-16
    if m and ("mar" in s or "jun" in s or "sep" in s or "dec" in s or "fy" in s):
        yy = int(m.group(1))
        return 2000 + yy

    m = re.search(r"\b(\d{4})-\d{2}\b", s)
    if m:
        return int(m.group(1))

    return None


def find_lineitem_row(table_df: pd.DataFrame, label_col: str, keywords: list):
    labels = table_df[label_col].astype(str).str.lower()
    for idx, lab in enumerate(labels.tolist()):
        if any(k in lab for k in keywords):
            return idx
    return None


def extract_timeseries(table_df: pd.DataFrame, label_col: str, year_cols: list, keywords: list):
    """
    Extract a year-wise series for a given metric based on keyword match in Narration.
    Returns df: year, value
    """
    idx = find_lineitem_row(table_df, label_col, keywords)
    if idx is None:
        return None

    out = {"year": [], "value": []}
    for yc in year_cols:
        yr = year_to_int_from_header(yc)
        val = table_df.iloc[idx][yc]
        out["year"].append(yr if yr is not None else yc)
        out["value"].append(pd.to_numeric(val, errors="coerce"))
    df = pd.DataFrame(out)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def manual_row_picker(table_df: pd.DataFrame, label_col: str, year_cols: list, canon_name: str):
    """
    Manual rescue: pick exact row label for a metric, return its year-wise series.
    """
    labels = table_df[label_col].astype(str).fillna("")
    show = labels.head(400).tolist()

    st.markdown(f"**Manual pick for:** `{canon_name}`")
    query = st.text_input(f"Search label ({canon_name})", value="", key=f"q_{canon_name}")

    if query.strip():
        mask = labels.str.lower().str.contains(query.lower(), na=False)
        candidates = labels[mask].head(50).tolist()
        if not candidates:
            candidates = show[:50]
    else:
        candidates = show[:50]

    picked = st.selectbox(f"Pick row label for {canon_name}", options=candidates, key=f"pick_{canon_name}")

    idx_list = table_df.index[labels == picked].tolist()
    if not idx_list:
        return None
    idx = table_df.index.get_loc(idx_list[0])

    out = {"year": [], "value": []}
    for yc in year_cols:
        yr = year_to_int_from_header(yc)
        out["year"].append(yr if yr is not None else yc)
        out["value"].append(pd.to_numeric(table_df.iloc[idx][yc], errors="coerce"))

    df = pd.DataFrame(out)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


# ============================================================
# AGENT 2: DuPont + Modified DuPont + AI scoring
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

    # ---- CHANGE: tax_expense is optional; if missing, use fallback tax rate ----
    if "tax_expense" in df.columns and not df["tax_expense"].isna().all():
        df["tax_rate_est"] = df["tax_expense"] / (df["ebt"] + eps)
        df["tax_rate_est"] = df["tax_rate_est"].clip(lower=0, upper=0.7)
    else:
        df["tax_rate_est"] = 0.25

    df["nopat"] = df["ebit"] * (1 - df["tax_rate_est"])
    df["operating_roa"] = df["nopat"] / (df["total_assets"] + eps)

    # ---- CHANGE: safe handling if interest_expense/total_debt are missing ----
    if "interest_expense" in df.columns and "total_debt" in df.columns:
        df["gross_cost_of_debt"] = df["interest_expense"] / (df["total_debt"] + eps)
        df["after_tax_cost_of_debt"] = df["gross_cost_of_debt"] * (1 - df["tax_rate_est"])
    else:
        df["gross_cost_of_debt"] = np.nan
        df["after_tax_cost_of_debt"] = np.nan

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

    if len(valid_idx) >= 8:
        X = feature_df.loc[valid_idx].values
        iso = IsolationForest(n_estimators=250, contamination=0.15, random_state=42)
        iso.fit(X)
        raw = -iso.score_samples(X)  # higher is more anomalous
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
# AGENT 4: LLM explanation (optional)
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
1) Main ROE drivers (margin vs efficiency vs leverage).
2) Whether ROE quality looks operational or leverage-driven.
3) What the score implies (low/moderate/high concern).
4) What an auditor should check next (2–3 concrete checks).
Do not claim fraud as fact; flag as risk signals only.
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
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="DuPont Fraud Score (XLSX Agent)", layout="wide")
st.title("🧠 Intelligent DuPont Fraud Score — XLSX Multi-Sheet Agent")

st.write(
    "Upload an **Excel (.xlsx)** file (like your Tata Steel template). "
    "Agent 1 auto-detects the year header row (e.g., Mar-16…Mar-25), extracts statement line items, "
    "then DuPont + AI anomaly detection runs end-to-end."
)

st.sidebar.header("📥 Upload Excel")
uploaded = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

sheets = load_workbook_sheets(uploaded)
sheet_names = list(sheets.keys())
sheet_types = {n: classify_sheet(sheets[n], n) for n in sheet_names}

st.sidebar.subheader("Agent 1: Sheet selection")
income_candidates = [n for n, t in sheet_types.items() if t == "income_statement"]
bs_candidates = [n for n, t in sheet_types.items() if t == "balance_sheet"]

income_sheet = st.sidebar.selectbox(
    "Select P&L / Profit & Loss sheet",
    options=sheet_names,
    index=sheet_names.index(income_candidates[0]) if income_candidates else 0
)
bs_sheet = st.sidebar.selectbox(
    "Select Balance Sheet sheet",
    options=sheet_names,
    index=sheet_names.index(bs_candidates[0]) if bs_candidates else 0
)

st.sidebar.subheader("Metadata")
company_name = st.sidebar.text_input("Company name", value="Unknown Company")
industry_name = st.sidebar.text_input("Industry", value="Unknown Industry")

st.sidebar.subheader("Extraction mode")
mode = st.sidebar.radio("Mode", ["Auto (agent)", "Manual rescue"], index=0)

pl_raw = sheets[income_sheet]
bs_raw = sheets[bs_sheet]

# Build statement tables
pl_table, pl_label_col, pl_year_cols = build_statement_table(pl_raw)
bs_table, bs_label_col, bs_year_cols = build_statement_table(bs_raw)

st.subheader("🧾 Agent 1 Output: Workbook structure")
c1, c2 = st.columns(2)

with c1:
    st.write("**P&L Raw Preview (top 30 rows)**")
    st.caption(f"Sheet: {income_sheet} (auto-tag: {sheet_types[income_sheet]})")
    st.dataframe(pl_raw.head(30), use_container_width=True)

with c2:
    st.write("**Balance Sheet Raw Preview (top 30 rows)**")
    st.caption(f"Sheet: {bs_sheet} (auto-tag: {sheet_types[bs_sheet]})")
    st.dataframe(bs_raw.head(30), use_container_width=True)

if pl_table is None or bs_table is None:
    st.error(
        "Could not detect the year header row in one or both sheets. "
        "Switch to Manual rescue and choose a better sheet (or use a sheet with Mar-xx / FY / 20xx headers visible)."
    )
    st.stop()

st.markdown("### Agent 1: Detected clean tables (preview)")
cc1, cc2 = st.columns(2)
with cc1:
    st.write("**P&L table preview**")
    st.caption(f"Detected year columns: {pl_year_cols[:8]}{'...' if len(pl_year_cols)>8 else ''}")
    st.dataframe(pl_table.head(25), use_container_width=True)
with cc2:
    st.write("**Balance Sheet table preview**")
    st.caption(f"Detected year columns: {bs_year_cols[:8]}{'...' if len(bs_year_cols)>8 else ''}")
    st.dataframe(bs_table.head(25), use_container_width=True)

# ---------- Auto extraction keywords (tuned for sheets like your screenshot) ----------
PL_KEYWORDS = {
    "sales": ["sales", "revenue", "turnover", "total income", "income from operations"],
    "ebit": ["operating profit", "ebit", "profit from operations", "operating income"],
    "interest_expense": ["interest", "finance cost", "finance costs", "borrowing cost"],
    "ebt": ["profit before tax", "pbt", "ebt", "profit before taxation"],
    "tax_expense": ["tax", "taxation", "income tax"],
    "net_income": ["net profit", "profit after tax", "pat", "profit for the year"],
}

BS_KEYWORDS = {
    "total_assets": ["total assets", "total", "assets"],  # tries total assets first; fallback may hit "Total"
    "total_equity": ["total equity", "net worth", "shareholders", "shareholder", "equity", "reserves"],
    "total_debt": ["borrowings", "total debt", "debt", "loans"],
}

# ---------- Extract series ----------
def series_to_wide(name, s_df):
    if s_df is None or s_df.empty:
        return None
    out = s_df.rename(columns={"value": name}).copy()
    return out[["year", name]]

if mode == "Auto (agent)":
    sales_ts = series_to_wide("sales", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["sales"]))
    ebit_ts = series_to_wide("ebit", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["ebit"]))
    int_ts  = series_to_wide("interest_expense", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["interest_expense"]))
    ebt_ts  = series_to_wide("ebt", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["ebt"]))
    tax_ts  = series_to_wide("tax_expense", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["tax_expense"]))
    pat_ts  = series_to_wide("net_income", extract_timeseries(pl_table, pl_label_col, pl_year_cols, PL_KEYWORDS["net_income"]))

    ta_ts   = series_to_wide("total_assets", extract_timeseries(bs_table, bs_label_col, bs_year_cols, BS_KEYWORDS["total_assets"]))
    te_ts   = series_to_wide("total_equity", extract_timeseries(bs_table, bs_label_col, bs_year_cols, BS_KEYWORDS["total_equity"]))
    td_ts   = series_to_wide("total_debt", extract_timeseries(bs_table, bs_label_col, bs_year_cols, BS_KEYWORDS["total_debt"]))

else:
    st.markdown("## Manual rescue mapping (when auto misses labels)")
    st.caption("Pick the exact row for each metric from the detected tables.")

    st.markdown("### P&L manual picks")
    sales_ts = series_to_wide("sales", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "sales"))
    ebit_ts  = series_to_wide("ebit", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "ebit (Operating Profit)"))
    int_ts   = series_to_wide("interest_expense", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "interest_expense"))
    ebt_ts   = series_to_wide("ebt", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "ebt (Profit before tax)"))
    tax_ts   = series_to_wide("tax_expense", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "tax_expense"))
    pat_ts   = series_to_wide("net_income", manual_row_picker(pl_table, pl_label_col, pl_year_cols, "net_income (Net profit/PAT)"))

    st.markdown("### Balance Sheet manual picks")
    ta_ts    = series_to_wide("total_assets", manual_row_picker(bs_table, bs_label_col, bs_year_cols, "total_assets"))
    te_ts    = series_to_wide("total_equity", manual_row_picker(bs_table, bs_label_col, bs_year_cols, "total_equity"))
    td_ts    = series_to_wide("total_debt", manual_row_picker(bs_table, bs_label_col, bs_year_cols, "total_debt (Borrowings)"))

series_list = [sales_ts, ebit_ts, int_ts, ebt_ts, tax_ts, pat_ts, ta_ts, te_ts, td_ts]
series_list = [s for s in series_list if s is not None and not s.empty]

if not series_list:
    st.error("No metrics could be extracted. Switch to Manual rescue and pick rows explicitly.")
    st.stop()

# Merge all series on year
df_std = series_list[0]
for s in series_list[1:]:
    df_std = pd.merge(df_std, s, on="year", how="outer")

df_std["company"] = company_name
df_std["industry"] = industry_name

# Guard: required columns for DuPont pipeline
required = ["sales", "net_income", "ebit", "ebt", "total_assets", "total_equity", "total_debt", "interest_expense", "tax_expense"]
missing = [c for c in required if c not in df_std.columns]
st.subheader("✅ Agent 1 Output: Standardized dataset (year-wise)")
st.dataframe(df_std.sort_values("year"), use_container_width=True)

if missing:
    st.warning(
        f"Missing fields: {missing}. You can still proceed if most are present, "
        "but for full DuPont you should fill them via Manual rescue."
    )

# Drop years without enough data
df_num = df_std.copy()

# ---- CHANGE: ensure required columns exist (prevents KeyError if extraction missed them) ----
for c in required:
    if c not in df_num.columns:
        df_num[c] = np.nan

for c in required:
    df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

df_num = df_num.dropna(subset=["year"])
df_num["year"] = df_num["year"].astype(int)

# Require at least some core fields
core_needed = ["sales", "net_income", "total_assets", "total_equity"]
if any(df_num.get(c, pd.Series(dtype=float)).isna().all() for c in core_needed):
    st.error("Not enough core fields to run DuPont (need sales, net_income, total_assets, total_equity). Use Manual rescue.")
    st.stop()

# Run analysis
df = run_analysis(df_num)

st.sidebar.header("📌 Select Year")
years = df["year"].sort_values().unique().tolist()
selected_year = st.sidebar.selectbox("Year (detailed view)", years, index=len(years)-1)
row = df[df["year"] == selected_year].iloc[0]

st.subheader(f"📊 Dashboard: {company_name} — {selected_year}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ROE", f"{row['roe']*100:.2f}%")
k2.metric("DuPont Fraud Score", f"{row['dupont_fraud_score']:.1f}")
k2.caption(f"Risk: **{risk_label(row['dupont_fraud_score'])}**")
k3.metric("Operating Quality Risk", f"{row['operating_quality_risk']:.1f}")
k4.metric("Leverage Risk", f"{row['leverage_risk']:.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Classic DuPont", "Modified DuPont", "AI Explanation", "Full Table"])

with tab1:
    st.subheader("Classic 3-step DuPont")
    a, b, c = st.columns(3)
    a.metric("Net Profit Margin", f"{row['net_profit_margin']*100:.2f}%")
    b.metric("Asset Turnover", f"{row['asset_turnover']:.2f}x")
    c.metric("Equity Multiplier", f"{row['equity_multiplier']:.2f}x")

    st.markdown("#### Trend: ROE + DuPont components")
    st.line_chart(df.set_index("year")[["roe", "net_profit_margin", "asset_turnover", "equity_multiplier"]])

with tab2:
    st.subheader("Modified DuPont + Operating vs Leverage")
    a, b, c, d, e = st.columns(5)
    a.metric("Tax Burden (NI/EBT)", f"{row['tax_burden']:.2f}")
    b.metric("Interest Burden (EBT/EBIT)", f"{row['interest_burden']:.2f}")
    c.metric("Operating Margin (EBIT/Sales)", f"{row['operating_margin']*100:.2f}%")
    d.metric("Operating ROA", f"{row['operating_roa']*100:.2f}%")
    e.metric("Debt/Equity", f"{row['leverage_de_ratio']:.2f}x")

    st.markdown("#### Trend: Operating ROA, leverage, spread")
    st.line_chart(df.set_index("year")[["operating_roa", "leverage_de_ratio", "spread"]])

with tab3:
    st.subheader("AI Explanation Agent (LangChain + OpenAI)")
    st.write("Pattern Anomaly Score:", "N/A" if pd.isna(row["pattern_anomaly_score"]) else f"{row['pattern_anomaly_score']:.1f}")

    explanation = generate_llm_explanation(row, df.sort_values("year"))
    if explanation:
        st.write(explanation)
        st.caption("Generated via LangChain + OpenAI. Add OPENAI_API_KEY in Streamlit Secrets.")
    else:
        st.warning("OPENAI_API_KEY not found. Showing rule-based explanation.")
        lines = []
        lines.append(f"- Fraud Score is **{row['dupont_fraud_score']:.1f}** ({risk_label(row['dupont_fraud_score'])}).")
        if row["leverage_risk"] > row["operating_quality_risk"]:
            lines.append("- Leverage risk dominates: ROE could be amplified by capital structure rather than operations.")
        else:
            lines.append("- Operating risk dominates: margin/operating ROA looks more unusual than leverage.")
        if not pd.isna(row["pattern_anomaly_score"]) and row["pattern_anomaly_score"] > 60:
            lines.append("- Overall DuPont pattern is statistically unusual across years in this dataset.")
        lines.append("- Suggested checks: revenue recognition notes, one-off income/expenses, related-party items, debt covenants and refinancing.")
        st.write("\n".join(lines))

with tab4:
    st.subheader("Full computed table (all years)")
    st.dataframe(df.sort_values("year"), use_container_width=True)
