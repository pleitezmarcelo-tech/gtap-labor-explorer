import streamlit as st
import os, json
import pandas as pd
import anthropic
import plotly.express as px
from tools import TOOL_DEFINITIONS, execute_tool

st.set_page_config(page_title="GTAP Labor Data Explorer", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.metric-card { background:#f0f4ff; border-radius:10px; padding:1rem 1.2rem; border-left:4px solid #2E5496; }
.metric-value { font-size:1.6rem; font-weight:700; color:#2E5496; }
.metric-label { font-size:0.8rem; color:#666; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

for k, v in [("messages",[]), ("df",None), ("figures",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

PARQUET_REPO  = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
PARQUET_LOCAL = "gtap_master_with_simulations.parquet"

@st.cache_data(show_spinner="Loading dataset...")
def load_parquet():
    path = PARQUET_REPO if os.path.exists(PARQUET_REPO) else PARQUET_LOCAL
    df = pd.read_parquet(path)
    if "county_fips" in df.columns:
        df["county_fips"] = (df["county_fips"].fillna("").astype(str)
                             .str.strip().str.replace(".0","",regex=False).str.zfill(5))
        df.loc[df["county_fips"]=="00000","county_fips"] = ""
    for col in ["workers_base","workers_sim","workers_change","pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "scenario" not in df.columns:
        df["scenario"] = "baseline"
    return df

st.title("GTAP - Step 3: Full app structure")
st.write("✅ Page config, styles, session state, load function all OK")

with st.sidebar:
    st.markdown("## Configuration")
    if st.session_state.df is None:
        if st.button("Load Data", use_container_width=True):
            st.session_state.df = load_parquet()
            st.rerun()
    else:
        st.success(f"Loaded: {len(st.session_state.df):,} rows")

if st.session_state.df is None:
    st.info("Click Load Data in the sidebar.")
    st.stop()

df = st.session_state.df
st.write(f"✅ DataFrame loaded: {len(df):,} rows, {df['scenario'].nunique()} scenarios")
st.write(f"Columns: {list(df.columns)[:8]}")
