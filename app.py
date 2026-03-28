import streamlit as st

st.title("Import Diagnostics")

steps = []

try:
    import os, json
    steps.append("✅ os, json")
except Exception as e:
    steps.append(f"❌ os/json: {e}")

try:
    import pandas as pd
    steps.append("✅ pandas")
except Exception as e:
    steps.append(f"❌ pandas: {e}")

try:
    import anthropic
    steps.append("✅ anthropic")
except Exception as e:
    steps.append(f"❌ anthropic: {e}")

try:
    import plotly.express as px
    steps.append("✅ plotly.express")
except Exception as e:
    steps.append(f"❌ plotly.express: {e}")

try:
    import plotly.graph_objects as go
    steps.append("✅ plotly.graph_objects")
except Exception as e:
    steps.append(f"❌ plotly.graph_objects: {e}")

try:
    from tools import TOOL_DEFINITIONS, execute_tool
    steps.append("✅ tools")
except Exception as e:
    steps.append(f"❌ tools: {e}")

for s in steps:
    st.write(s)

st.success("All imports attempted")
