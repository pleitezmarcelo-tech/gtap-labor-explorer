"""
app.py
GTAP Labor Data Explorer
Streamlit application with Claude as conversational agent
+ fully interactive dashboard with real-time filters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import anthropic
import json
import os
from tools import TOOL_DEFINITIONS, execute_tool

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTAP Labor Data Explorer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: 700;
        color: #2E5496; margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #2E5496;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #2E5496; }
    .metric-label {
        font-size: 0.8rem; color: #666;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .sidebar-section {
        background: #f8f9fa; border-radius: 8px;
        padding: 0.8rem; margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [("messages", []), ("df", None), ("figures", {})]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── DATA LOADING ──────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1uwwbrY1nOy3Ks3RRvEi75W0L4CODLWM_"

@st.cache_data(show_spinner="Downloading dataset from Google Drive...")
def load_from_gdrive(file_id):
    import requests, io
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"confirm": token}, stream=True)
    df = pd.read_csv(io.BytesIO(response.content), dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].str.zfill(5)
    df["estimated_workers"] = pd.to_numeric(
        df["estimated_workers"].astype(str).str.replace(",", ""),
        errors="coerce"
    ).fillna(0)
    return df

@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    df = pd.read_csv(path, dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].str.zfill(5)
    df["estimated_workers"] = pd.to_numeric(
        df["estimated_workers"].astype(str).str.replace(",", ""),
        errors="coerce"
    ).fillna(0)
    return df


# ── CLAUDE AGENT ──────────────────────────────────────────────────────────────
def run_agent(user_message, df, anthropic_key, usda_key):
    client = anthropic.Anthropic(api_key=anthropic_key)

    history = []
    for msg in st.session_state.messages[-12:]:
        if msg["role"] in ("user", "assistant") and isinstance(msg["content"], str):
            history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": user_message})

    system_prompt = """You are an expert data analyst specializing in U.S. labor
markets and the GTAP (Global Trade Analysis Project) framework. You have access
to a dataset of estimated workers for all 65 GTAP sectors across U.S. counties,
years 2021-2024, disaggregated by skill level and birthplace.

IMPORTANT GUIDELINES:
- Always use tools to answer quantitative questions. Never guess from memory.
- Start with get_dataset_info if the user asks what data is available.
- For geographic questions, ALWAYS use create_map. You CAN create interactive maps.
- For comparisons and trends, use create_chart.
- For specific numbers, use query_dataset.
- You can chain multiple tools in one response.
- Be concise but insightful. Interpret results, do not just report them.
- If users ask for a dashboard with dropdowns and real-time filters, tell them
  to click the "Interactive Dashboard" tab at the top of the app.

GTAP sector codes:
  Crops: pdr=Paddy rice, wht=Wheat, gro=Cereal grains, v_f=Vegetables/fruits/nuts,
  osd=Oil seeds, c_b=Sugar, pfb=Plant fibers, ocr=Crops nec
  Livestock: ctl=Livestock, oap=Animal products, rmk=Raw milk, wol=Wool
  Primary: frs=Forestry, fsh=Fishing, coa=Coal, oil=Oil, gas=Gas
  Industry: cns=Construction, mvh=Motor vehicles, ele=Electronics, tex=Textiles,
  chm=Chemicals, ppp=Paper, omf=Manufactures nec
  Services: trd=Trade, afs=Accommodation/food, edu=Education, hht=Health,
  obs=Business services, osg=Public administration, cmn=Communication,
  ofi=Financial services, ins=Insurance, rsa=Real estate

When presenting tables, format numbers with commas."""

    figures_generated = []
    final_text = ""
    messages_loop = history.copy()

    for iteration in range(6):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages_loop
        )

        for block in response.content:
            if block.type == "text":
                final_text += block.text

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result, is_figure = execute_tool(
                    block.name, block.input, df, usda_key
                )
                if is_figure and result is not None:
                    fig_id = f"fig_{iteration}_{block.id[:8]}"
                    figures_generated.append((fig_id, result))
                    content = json.dumps({
                        "success": True, "figure_id": fig_id,
                        "message": f"Figure created: {block.input.get('title','Chart')}"
                    })
                else:
                    content = json.dumps(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content
                })

        messages_loop.append({"role": "assistant", "content": response.content})
        messages_loop.append({"role": "user", "content": tool_results})

    return final_text, figures_generated


# ── INTERACTIVE DASHBOARD ─────────────────────────────────────────────────────
def render_dashboard(df):
    st.markdown("### Interactive Dashboard")
    st.markdown("Select filters — all charts update in real time.")

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        years = sorted(df["year"].unique().tolist())
        sel_year = st.selectbox("Year", years, index=len(years)-1, key="d_year")
    with c2:
        # Canonical GTAP sector order (65 sectors)
        gtap_order = [
            "pdr","wht","gro","v_f","osd","c_b","pfb","ocr",
            "ctl","oap","rmk","wol","frs","fsh",
            "coa","oil","gas","oxt",
            "cmt","omt","vol","mil","pcr","sgr","ofd","b_t",
            "tex","wap","lea","lum","ppp","chm","bph","rpp",
            "nmm","i_s","nfm","fmp","mvh","otn","ele","eeq","ome","omf","p_c",
            "ely","gdt","wtr","cns",
            "trd","afs","otp","wtp","atp","cmn",
            "ofi","ins","rsa","obs","ros","osg","edu","hht","dwe"
        ]
        # Keep only sectors present in the dataset, in canonical order
        available = df["gtap_code"].unique().tolist()
        sectors = [s for s in gtap_order if s in available]
        # Add any sector in the dataset not covered by the list (safety net)
        sectors += [s for s in available if s not in sectors]

        sector_labels = {
            r["gtap_code"]: f"{r['gtap_code']} — {r['gtap_sector']}"
            for _, r in df[["gtap_code","gtap_sector"]].drop_duplicates().iterrows()
        }
        sel_sector = st.selectbox(
            "GTAP Sector", ["All sectors"] + sectors, key="d_sector",
            format_func=lambda x: sector_labels.get(x, x) if x != "All sectors" else "All sectors"
        )
    with c3:
        skill_opts = ["All"] + sorted(df["skill_level"].dropna().unique().tolist())
        sel_skill = st.selectbox("Skill Level", skill_opts, key="d_skill")
    with c4:
        bp_opts = ["All"] + sorted(df["birthplace_label"].dropna().unique().tolist())
        sel_bp = st.selectbox("Birthplace", bp_opts, key="d_bp")

    # Apply filters
    filt = df[df["year"] == sel_year].copy()
    if sel_sector != "All sectors":
        filt = filt[filt["gtap_code"] == sel_sector]
    if sel_skill != "All":
        filt = filt[filt["skill_level"] == sel_skill]
    if sel_bp != "All":
        filt = filt[filt["birthplace_label"] == sel_bp]

    total = int(filt["estimated_workers"].sum())
    fb    = filt[filt["birthplace_label"]=="Foreign born"]["estimated_workers"].sum()
    sk    = filt[filt["skill_level"]=="Skilled"]["estimated_workers"].sum()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{total:,}",                           f"Estimated Workers {sel_year}"),
        (k2, f"{fb/total*100:.1f}%" if total else "—", "Foreign Born"),
        (k3, f"{sk/total*100:.1f}%" if total else "—", "Skilled"),
        (k4, str(filt["county_fips"].nunique()),       "Counties with Data"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Map + Bar
    map_col, bar_col = st.columns([3, 2])

    with map_col:
        county_agg = (
            filt.groupby(["county_fips","county_name","state"])
            ["estimated_workers"].sum().reset_index()
        )
        county_agg = county_agg[
            county_agg["county_fips"].notna() &
            (county_agg["county_fips"].str.len() == 5) &
            (county_agg["estimated_workers"] > 0)
        ]
        map_title = f"Workers by County — {sel_year}"
        if sel_sector != "All sectors":
            map_title += f" | {sel_sector}"
        if sel_bp != "All":
            map_title += f" | {sel_bp}"

        if len(county_agg) > 0:
            fig_map = px.choropleth(
                county_agg,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                locations="county_fips",
                color="estimated_workers",
                color_continuous_scale="Blues",
                scope="usa",
                hover_data={"county_fips": False, "county_name": True,
                            "state": True, "estimated_workers": ":,"},
                title=map_title,
                labels={"estimated_workers": "Workers",
                        "county_name": "County", "state": "State"}
            )
            fig_map.update_layout(
                margin={"r":0,"t":40,"l":0,"b":0}, height=420,
                coloraxis_colorbar=dict(title="Workers", tickformat=",d"),
                title_font_size=14
            )
            st.plotly_chart(fig_map, use_container_width=True,
                key=f"map_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")
        else:
            st.info("No county data for this combination of filters.")

    with bar_col:
        if sel_sector == "All sectors":
            bar_data = (
                filt.groupby(["gtap_code","gtap_sector"])
                ["estimated_workers"].sum().reset_index()
                .sort_values("estimated_workers", ascending=False).head(15)
            )
            bar_data["label"] = (bar_data["gtap_code"] + " — " +
                                  bar_data["gtap_sector"].str[:18])
            fig_bar = px.bar(
                bar_data, x="estimated_workers", y="label", orientation="h",
                title=f"Top 15 Sectors — {sel_year}",
                labels={"estimated_workers": "Workers", "label": ""},
                color="estimated_workers", color_continuous_scale="Blues"
            )
        else:
            bar_data = (
                filt.groupby("state")["estimated_workers"]
                .sum().reset_index()
                .sort_values("estimated_workers", ascending=False).head(15)
            )
            fig_bar = px.bar(
                bar_data, x="estimated_workers", y="state", orientation="h",
                title=f"Top 15 States — {sel_sector} {sel_year}",
                labels={"estimated_workers": "Workers", "state": "State"},
                color="estimated_workers", color_continuous_scale="Blues"
            )

        fig_bar.update_layout(
            height=420, showlegend=False, plot_bgcolor="white",
            paper_bgcolor="white", coloraxis_showscale=False,
            xaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),
            title_font_size=14, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True,
            key=f"bar_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")

    # Trend + Pie
    t_col, p_col = st.columns(2)

    with t_col:
        trend_base = df.copy()
        if sel_sector != "All sectors":
            trend_base = trend_base[trend_base["gtap_code"] == sel_sector]
        if sel_skill != "All":
            trend_base = trend_base[trend_base["skill_level"] == sel_skill]
        if sel_bp != "All":
            trend_base = trend_base[trend_base["birthplace_label"] == sel_bp]
        trend_data = (trend_base.groupby("year")["estimated_workers"]
                      .sum().reset_index())
        fig_trend = px.line(
            trend_data, x="year", y="estimated_workers",
            title="Trend 2021–2024", markers=True,
            labels={"estimated_workers": "Workers", "year": "Year"},
            color_discrete_sequence=["#2E5496"]
        )
        fig_trend.update_layout(
            height=280, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
            title_font_size=14, margin=dict(t=40, b=30)
        )
        st.plotly_chart(fig_trend, use_container_width=True,
            key=f"trend_{sel_sector}_{sel_skill}_{sel_bp}")

    with p_col:
        split_data = (
            filt.groupby(["skill_level","birthplace_label"])
            ["estimated_workers"].sum().reset_index()
        )
        split_data["group"] = (split_data["skill_level"] + " / " +
                                split_data["birthplace_label"])
        fig_pie = px.pie(
            split_data, values="estimated_workers", names="group",
            title=f"Skill x Birthplace — {sel_year}",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(
            height=280, title_font_size=14,
            margin=dict(t=40, b=10), legend=dict(font_size=11)
        )
        st.plotly_chart(fig_pie, use_container_width=True,
            key=f"pie_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**API Keys**")

    # Read Anthropic key from Streamlit secrets first, then env, then manual input
    default_anthropic = ""
    try:
        default_anthropic = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        default_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")

    if default_anthropic:
        anthropic_key = default_anthropic
        st.success("Anthropic API key configured", icon="✓")
    else:
        anthropic_key = st.text_input(
            "Anthropic API Key", type="password",
            help="Get your key at console.anthropic.com"
        )

    # USDA key always manual (optional)
    usda_key = st.text_input(
        "USDA NASS API Key (optional)", type="password",
        value=os.environ.get("USDA_API_KEY", ""),
        help="Free key at quickstats.nass.usda.gov/api"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Data File**")
    data_source = st.radio(
        "Source",
        ["Google Drive (auto)", "Upload file", "Enter path"],
        label_visibility="collapsed"
    )
    if data_source == "Google Drive (auto)":
        if st.session_state.df is None:
            if st.button("Load from Google Drive", use_container_width=True):
                try:
                    st.session_state.df = load_from_gdrive(GDRIVE_FILE_ID)
                    st.success(f"Loaded: {len(st.session_state.df):,} rows")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.success(f"Loaded: {len(st.session_state.df):,} rows")
    elif data_source == "Upload file":
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            st.session_state.df = load_data(uploaded)
            st.success(f"Loaded: {len(st.session_state.df):,} rows")
    else:
        csv_path = st.text_input("CSV path", value="gtap_complete_master.csv",
                                 label_visibility="collapsed")
        if st.button("Load", use_container_width=True):
            try:
                st.session_state.df = load_data(csv_path)
                st.success(f"Loaded: {len(st.session_state.df):,} rows")
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df_s = st.session_state.df
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Dataset Summary**")
        st.metric("Total Workers (all years)",
                  f"{int(df_s['estimated_workers'].sum()):,}")
        st.metric("GTAP Sectors", df_s["gtap_code"].nunique())
        st.metric("Counties",     df_s["county_fips"].nunique())
        st.metric("Years", f"{int(df_s['year'].min())}–{int(df_s['year'].max())}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example questions for Claude**")
    for ex in [
        "What data is available?",
        "Which sectors have the most foreign-born workers?",
        "Show a map of v_f workers in 2022",
        "Compare skilled vs unskilled in construction",
        "Top 10 counties for agricultural workers",
        "Trend in health sector 2021-2024",
    ]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.figures = {}
        st.rerun()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 GTAP Labor Data Explorer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive dashboard and conversational '
    'analysis for all 65 GTAP sectors across U.S. counties.</div>',
    unsafe_allow_html=True
)

if st.session_state.df is None:
    st.info("Load your `gtap_complete_master.csv` file using the sidebar to get started.")
    st.stop()

df = st.session_state.df

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Interactive Dashboard", "💬 Ask Claude"])

with tab1:
    render_dashboard(df)

with tab2:
    # Top KPIs
    latest = int(df["year"].max())
    df_l   = df[df["year"] == latest]
    tot    = df_l["estimated_workers"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{int(tot):,}", f"Total Workers {latest}"),
        (c2, f"{df_l[df_l['birthplace_label']=='Foreign born']['estimated_workers'].sum()/tot*100:.1f}%",
             f"Foreign Born {latest}"),
        (c3, f"{df_l[df_l['skill_level']=='Skilled']['estimated_workers'].sum()/tot*100:.1f}%",
             f"Skilled {latest}"),
        (c4, str(df["gtap_code"].nunique()), "GTAP Sectors"),
        (c5, f"{df['county_fips'].nunique():,}", "Counties"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])
            if msg.get("figures"):
                for fig_id in msg["figures"]:
                    if fig_id in st.session_state.figures:
                        st.plotly_chart(
                            st.session_state.figures[fig_id],
                            use_container_width=True,
                            key=f"hist_{fig_id}_{i}"
                        )

    # Input
    prefill    = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask Claude about the GTAP labor data...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        if not anthropic_key:
            st.error("Enter your Anthropic API key in the sidebar.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response_text, figures = run_agent(
                        user_input, df, anthropic_key, usda_key
                    )
                    if response_text:
                        st.markdown(response_text)

                    fig_ids = []
                    for fig_id, fig in figures:
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"new_{fig_id}")
                        st.session_state.figures[fig_id] = fig
                        fig_ids.append(fig_id)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text or "(See chart above)",
                        "figures": fig_ids
                    })

                except anthropic.AuthenticationError:
                    st.error("Invalid API key. Check the sidebar.")
                except anthropic.APIConnectionError:
                    st.error("Cannot connect to Anthropic API.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
