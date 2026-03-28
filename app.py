import streamlit as st
import os
import json
import pandas as pd
import anthropic
import plotly.express as px
from tools import TOOL_DEFINITIONS, execute_tool

st.title("GTAP - Step 2")

# Test session state
for k, v in [("messages",[]), ("df",None), ("figures",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

st.write("✅ Session state OK")

# Test constants
PARQUET_REPO = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
PARQUET_LOCAL = "gtap_master_with_simulations.parquet"
path = PARQUET_REPO if os.path.exists(PARQUET_REPO) else PARQUET_LOCAL
st.write(f"✅ Parquet path: {path}")
st.write(f"✅ File exists: {os.path.exists(path)}")

# Test load
if st.button("Load parquet"):
    df = pd.read_parquet(path)
    st.write(f"✅ Loaded {len(df):,} rows")
    st.write(f"Columns: {list(df.columns)[:6]}")

st.success("Step 2 complete")
