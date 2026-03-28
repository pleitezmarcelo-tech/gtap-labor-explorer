import os, json
import streamlit as st
import pandas as pd
import anthropic
from tools import TOOL_DEFINITIONS, execute_tool

st.set_page_config(page_title="GTAP Labor Data Explorer", page_icon="🌾", layout="wide")

for k, v in [("messages",[]), ("df",None), ("figures",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

PARQUET_REPO  = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
PARQUET_LOCAL = "gtap_master_with_simulations.parquet"

@st.cache_data(show_spinner="Loading dataset...")
def load_parquet():
    path = PARQUET_REPO if os.path.exists(PARQUET_REPO) else PARQUET_LOCAL
    df = pd.read_parquet(path)
    for col in ["workers_base","workers_sim","workers_change","pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "scenario" not in df.columns:
        df["scenario"] = "baseline"
    return df

st.markdown("# 🌾 GTAP Labor Data Explorer")
st.caption("Step 4: With tabs and full structure")

with st.sidebar:
    st.markdown("## Configuration")
    if st.session_state.df is None:
        if st.button("Load Data"):
            st.session_state.df = load_parquet()
            st.rerun()
    else:
        st.success(f"Loaded: {len(st.session_state.df):,} rows")
        if "scenario" in st.session_state.df.columns:
            st.write("Scenarios:", st.session_state.df["scenario"].unique().tolist())

if st.session_state.df is None:
    st.info("Click Load Data in the sidebar.")
    st.stop()

df = st.session_state.df
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Ask Claude"])

with tab1:
    st.write(f"✅ Rows: {len(df):,}")
    st.write(f"✅ Scenarios: {df['scenario'].nunique()}")
    st.write(f"✅ Columns: {list(df.columns)[:6]}")
    
    import plotly.express as px
    fig = px.bar(
        df[df["scenario"]=="baseline"].groupby("gtap_code")["workers_base"]
        .sum().reset_index().sort_values("workers_base",ascending=False).head(10),
        x="workers_base", y="gtap_code", orientation="h",
        title="Top 10 GTAP Sectors"
    )
    st.plotly_chart(fig, key="test_chart")

with tab2:
    st.write("Claude chat tab - coming soon")
