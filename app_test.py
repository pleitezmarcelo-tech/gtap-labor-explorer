import streamlit as st
import pandas as pd
import os

st.title("GTAP Test")
st.write("App is running")
st.write(f"Python version working")

# Test parquet
parquet_path = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
local_path = "gtap_master_with_simulations.parquet"

path = parquet_path if os.path.exists(parquet_path) else local_path

if st.button("Test Load Parquet"):
    try:
        df = pd.read_parquet(path)
        st.success(f"Loaded {len(df):,} rows, columns: {list(df.columns)[:5]}")
    except Exception as e:
        st.error(f"Error: {e}")
