"""
app.py
GTAP Labor Data Explorer
Streamlit application with Claude as conversational agent
+ fully interactive dashboard with simulation scenarios.

New CSV structure (gtap_master_with_simulations.csv):
  - scenario = "baseline"  : observed employment 2021-2024
  - scenario = "JPM_sim03" : simulated employment (deportation, variant a)
  - scenario = "JPM_sim03b": simulated employment (deportation, variant b)
  - scenario = "JPM_sim03c": simulated employment (deportation, variant c)
  - scenario = "USMCA_SR"  : simulated employment (short run tariffs)
  - scenario = "USMCA_LR"  : simulated employment (long run trade disengagement)

Key columns:
  workers_base   : baseline employment (all scenarios)
  workers_sim    : simulated employment (simulation scenarios only)
  workers_change : employment change sim - base (simulation scenarios only)
  pct_change     : effective % change (simulation scenarios only)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    .metric-card.negative { border-left-color: #c0392b; background: #fff0ef; }
    .metric-card.positive { border-left-color: #27ae60; background: #f0fff4; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #2E5496; }
    .metric-value.negative { color: #c0392b; }
    .metric-value.positive { color: #27ae60; }
    .metric-label {
        font-size: 0.8rem; color: #666;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .sidebar-section {
        background: #f8f9fa; border-radius: 8px;
        padding: 0.8rem; margin-bottom: 0.8rem;
    }
    .scenario-badge {
        display: inline-block; border-radius: 12px;
        padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── SCENARIO METADATA ─────────────────────────────────────────────────────────
SCENARIO_META = {
    "baseline": {
        "label": "Baseline (Observed)",
        "description": "Observed employment from ACS 2021-2024",
        "horizon": "observed",
        "color": "#2E5496"
    },
    "JPM_sim03": {
        "label": "JPM 2025 — sim03",
        "description": "Deportation shock: 8.3M workers removed",
        "horizon": "Short Run",
        "color": "#c0392b"
    },
    "JPM_sim03b": {
        "label": "JPM 2025 — sim03b",
        "description": "Deportation shock: variant b",
        "horizon": "Short Run",
        "color": "#e74c3c"
    },
    "JPM_sim03c": {
        "label": "JPM 2025 — sim03c",
        "description": "Deportation shock: variant c",
        "horizon": "Short Run",
        "color": "#e67e22"
    },
    "USMCA_SR": {
        "label": "USMCA — Short Run",
        "description": "US tariff increase, low elasticities, no retaliation",
        "horizon": "Short Run",
        "color": "#8e44ad"
    },
    "USMCA_LR": {
        "label": "USMCA — Long Run",
        "description": "Trade disengagement + ICRE FTA, high elasticities",
        "horizon": "Long Run",
        "color": "#2980b9"
    }
}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [("messages", []), ("df", None), ("figures", {})]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── DATA LOADING ──────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1k5aVtkVoBodteUcxC0fP9KJ-GfhKtlbQ"

@st.cache_data(show_spinner="Downloading dataset from Google Drive...", ttl=0)
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
    return _clean_df(pd.read_csv(io.BytesIO(response.content),
                                  dtype={"county_fips": str}))

@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    return _clean_df(pd.read_csv(path, dtype={"county_fips": str}))

def _clean_df(df):
    # Handle county_fips safely
    if "county_fips" in df.columns:
        df["county_fips"] = (df["county_fips"].fillna("").astype(str)
                             .str.strip().str.replace(".0", "", regex=False)
                             .str.zfill(5))
        df.loc[df["county_fips"] == "00000", "county_fips"] = ""

    for col in ["workers_base", "workers_sim", "workers_change",
                "pct_change", "lq", "effective_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    # Legacy support
    if "estimated_workers" in df.columns and "workers_base" not in df.columns:
        df["workers_base"] = pd.to_numeric(
            df["estimated_workers"].astype(str).str.replace(",", ""),
            errors="coerce"
        ).fillna(0)

    if "scenario" not in df.columns:
        df["scenario"] = "baseline"

    return df

def has_simulations(df):
    """Check if dataset contains simulation scenarios."""
    return "scenario" in df.columns and df["scenario"].nunique() > 1

def get_worker_col(df, scenario):
    """Return the appropriate worker column for a given scenario."""
    if scenario == "baseline" or not has_simulations(df):
        return "workers_base"
    return "workers_sim"


# ── CLAUDE AGENT ──────────────────────────────────────────────────────────────
def run_agent(user_message, df, anthropic_key, usda_key):
    client = anthropic.Anthropic(api_key=anthropic_key)

    history = []
    for msg in st.session_state.messages[-12:]:
        if msg["role"] in ("user", "assistant") and isinstance(msg["content"], str):
            history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": user_message})

    has_sims = has_simulations(df)
    sim_note = ""
    if has_sims:
        scenarios = df["scenario"].unique().tolist()
        sim_note = f"""
The dataset also contains simulation scenarios: {scenarios}
For simulation rows (non-baseline), key columns are:
  workers_base   = 2024 baseline employment
  workers_sim    = simulated employment after shock
  workers_change = net change (sim - base)
  pct_change     = effective % change in employment
"""

    system_prompt = f"""You are an expert data analyst specializing in U.S. labor
markets and the GTAP (Global Trade Analysis Project) framework. You have access
to a dataset of estimated workers for all 65 GTAP sectors across U.S. counties,
years 2021-2024, disaggregated by skill level and birthplace.
{sim_note}
GUIDELINES:
- Always use tools for quantitative questions. Never guess.
- Use get_dataset_info first if asked what data is available.
- For geographic questions use create_map.
- For comparisons and trends use create_chart.
- For specific numbers use query_dataset.
- Chain multiple tools when needed.
- If users want interactive dashboards, tell them to use the Dashboard tab.

GTAP sectors: pdr=Paddy rice, wht=Wheat, gro=Cereal grains, v_f=Vegetables/
fruits/nuts, osd=Oil seeds, c_b=Sugar, pfb=Plant fibers, ocr=Crops nec,
ctl=Livestock, oap=Animal products, rmk=Raw milk, frs=Forestry, fsh=Fishing,
coa=Coal, oil=Oil, gas=Gas, cns=Construction, mvh=Motor vehicles,
ele=Electronics, tex=Textiles, chm=Chemicals, trd=Trade, afs=Accommodation,
edu=Education, hht=Health, obs=Business services, osg=Public administration.

Format all numbers with commas."""

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
    has_sims = has_simulations(df)

    st.markdown("### Interactive Dashboard")
    st.markdown("Select filters — all charts update in real time.")

    # ── FILTER ROW ────────────────────────────────────────────────────────────
    if has_sims:
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    else:
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc5 = None

    with fc1:
        # Scenario selector — only for baseline tab
        if has_sims:
            scenario_options = sorted(df["scenario"].unique().tolist())
            scenario_labels = {
                s: SCENARIO_META.get(s, {}).get("label", s)
                for s in scenario_options
            }
            sel_scenario = st.selectbox(
                "Scenario", scenario_options, key="d_scenario",
                format_func=lambda x: scenario_labels.get(x, x)
            )
        else:
            sel_scenario = "baseline"

    with fc2:
        # Year — for simulations only 2024 is available
        if has_sims and sel_scenario != "baseline":
            years = [2024]
        else:
            years = sorted(df[df["scenario"] == "baseline"]["year"]
                           .unique().tolist())
        sel_year = st.selectbox("Year", years,
                                index=len(years)-1, key="d_year")

    with fc3:
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
        available = df["gtap_code"].unique().tolist()
        sectors   = [s for s in gtap_order if s in available]
        sectors  += [s for s in available if s not in sectors]
        sector_labels = {
            r["gtap_code"]: f"{r['gtap_code']} — {r['gtap_sector']}"
            for _, r in df[["gtap_code","gtap_sector"]].drop_duplicates().iterrows()
        }
        sel_sector = st.selectbox(
            "GTAP Sector", ["All sectors"] + sectors, key="d_sector",
            format_func=lambda x: sector_labels.get(x, x)
            if x != "All sectors" else "All sectors"
        )

    with fc4:
        skill_opts = ["All"] + sorted(df["skill_level"].dropna().unique().tolist())
        sel_skill = st.selectbox("Skill Level", skill_opts, key="d_skill")

    if fc5:
        with fc5:
            bp_opts = ["All"] + sorted(df["birthplace"].dropna().unique().tolist()
                                       if "birthplace" in df.columns
                                       else df["birthplace_label"].dropna()
                                       .unique().tolist())
            sel_bp = st.selectbox("Birthplace", bp_opts, key="d_bp")
    else:
        bp_col = "birthplace_label" if "birthplace_label" in df.columns else "birthplace"
        bp_opts = ["All"] + sorted(df[bp_col].dropna().unique().tolist())
        sel_bp = st.selectbox("Birthplace", bp_opts, key="d_bp")

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    bp_col = "birthplace" if "birthplace" in df.columns else "birthplace_label"
    skill_col = "skill_level"

    filt = df[(df["scenario"] == sel_scenario) &
              (df["year"] == sel_year)].copy()
    if sel_sector != "All sectors":
        filt = filt[filt["gtap_code"] == sel_sector]
    if sel_skill != "All":
        filt = filt[filt[skill_col] == sel_skill]
    if sel_bp != "All":
        filt = filt[filt[bp_col] == sel_bp]

    # Choose worker column based on scenario
    wcol = get_worker_col(df, sel_scenario)
    filt_workers = filt[wcol].fillna(0)
    total = int(filt_workers.sum())

    # ── SCENARIO INFO BANNER ──────────────────────────────────────────────────
    if has_sims and sel_scenario in SCENARIO_META:
        meta = SCENARIO_META[sel_scenario]
        color = meta["color"]
        st.markdown(
            f'<div style="background:{color}18; border-left:4px solid {color}; '
            f'padding:8px 14px; border-radius:6px; margin-bottom:12px;">'
            f'<b style="color:{color}">{meta["label"]}</b> — '
            f'{meta["description"]} '
            f'<span style="color:#888; font-size:0.85rem">({meta["horizon"]})</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    if sel_scenario != "baseline" and has_sims and "workers_change" in filt.columns:
        total_change = int(filt["workers_change"].fillna(0).sum())
        pct_avg = filt["pct_change"].fillna(0).mean()
        fb_change = int(filt[filt[bp_col].str.contains("Foreign", na=False)]
                        ["workers_change"].fillna(0).sum())
        sk_change = int(filt[filt[skill_col] == "Unskilled"]
                        ["workers_change"].fillna(0).sum())

        k1, k2, k3, k4 = st.columns(4)
        for col, val, label, neg in [
            (k1, f"{total:,}", f"Simulated Workers {sel_year}", False),
            (k2, f"{total_change:+,}", "Net Employment Change", total_change < 0),
            (k3, f"{fb_change:+,}", "Foreign Born Change", fb_change < 0),
            (k4, f"{sk_change:+,}", "Unskilled Change", sk_change < 0),
        ]:
            cls = "negative" if neg else ("positive" if not neg and
                                           val.startswith("+") else "")
            with col:
                st.markdown(f"""<div class="metric-card {cls}">
                    <div class="metric-value {cls}">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
    else:
        fb  = filt[filt[bp_col].str.contains("Foreign", na=False)][wcol].sum()
        sk  = filt[filt[skill_col] == "Skilled"][wcol].sum()
        k1, k2, k3, k4 = st.columns(4)
        for col, val, label in [
            (k1, f"{total:,}",
                 f"Workers {sel_year}"),
            (k2, f"{fb/total*100:.1f}%" if total else "—", "Foreign Born"),
            (k3, f"{sk/total*100:.1f}%" if total else "—", "Skilled"),
            (k4, str(filt["county_fips"].nunique()), "Counties"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP ───────────────────────────────────────────────────────────────────
    map_col, bar_col = st.columns([3, 2])

    with map_col:
        # For simulations show change map; for baseline show employment map
        if sel_scenario != "baseline" and has_sims and "workers_change" in filt.columns:
            county_agg = (
                filt.groupby(["county_fips", "county_name", "state"])
                ["workers_change"].sum().reset_index()
            )
            county_agg = county_agg[
                county_agg["county_fips"].notna() &
                (county_agg["county_fips"].str.len() == 5)
            ]
            map_color_col   = "workers_change"
            map_color_scale = "RdYlGn"
            map_label       = "Employment Change"
            map_title = f"Employment Change by County — {SCENARIO_META.get(sel_scenario,{}).get('label', sel_scenario)}"
        else:
            county_agg = (
                filt.groupby(["county_fips", "county_name", "state"])
                [wcol].sum().reset_index()
            )
            county_agg = county_agg[
                county_agg["county_fips"].notna() &
                (county_agg["county_fips"].str.len() == 5) &
                (county_agg[wcol] > 0)
            ]
            map_color_col   = wcol
            map_color_scale = "Blues"
            map_label       = "Workers"
            map_title = f"Workers by County — {sel_year}"
            if sel_sector != "All sectors":
                map_title += f" | {sel_sector}"

        if len(county_agg) > 0:
            fig_map = px.choropleth(
                county_agg,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                locations="county_fips",
                color=map_color_col,
                color_continuous_scale=map_color_scale,
                color_continuous_midpoint=0 if "change" in map_color_col else None,
                scope="usa",
                hover_data={"county_fips": False, "county_name": True,
                            "state": True, map_color_col: ":,"},
                title=map_title,
                labels={map_color_col: map_label,
                        "county_name": "County", "state": "State"}
            )
            fig_map.update_layout(
                margin={"r":0,"t":40,"l":0,"b":0}, height=430,
                coloraxis_colorbar=dict(title=map_label, tickformat=",d"),
                title_font_size=14
            )
            st.plotly_chart(fig_map, use_container_width=True,
                key=f"map_{sel_scenario}_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")
        else:
            st.info("No county data for this combination of filters.")

    # ── BAR CHART ─────────────────────────────────────────────────────────────
    with bar_col:
        if sel_scenario != "baseline" and has_sims and "workers_change" in filt.columns:
            # Show top sectors by employment change
            if sel_sector == "All sectors":
                bar_data = (
                    filt.groupby(["gtap_code","gtap_sector"])
                    ["workers_change"].sum().reset_index()
                    .sort_values("workers_change").head(15)
                )
                bar_data["label"] = (bar_data["gtap_code"] + " — " +
                                      bar_data["gtap_sector"].str[:18])
                fig_bar = px.bar(
                    bar_data, x="workers_change", y="label", orientation="h",
                    title="Top 15 Sectors by Employment Change",
                    labels={"workers_change": "Change", "label": ""},
                    color="workers_change",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0
                )
            else:
                bar_data = (
                    filt.groupby("state")["workers_change"]
                    .sum().reset_index()
                    .sort_values("workers_change").head(15)
                )
                fig_bar = px.bar(
                    bar_data, x="workers_change", y="state", orientation="h",
                    title=f"Top 15 States by Employment Change — {sel_sector}",
                    labels={"workers_change": "Change", "state": "State"},
                    color="workers_change",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0
                )
        else:
            if sel_sector == "All sectors":
                bar_data = (
                    filt.groupby(["gtap_code","gtap_sector"])
                    [wcol].sum().reset_index()
                    .sort_values(wcol, ascending=False).head(15)
                )
                bar_data["label"] = (bar_data["gtap_code"] + " — " +
                                      bar_data["gtap_sector"].str[:18])
                fig_bar = px.bar(
                    bar_data, x=wcol, y="label", orientation="h",
                    title=f"Top 15 Sectors — {sel_year}",
                    labels={wcol: "Workers", "label": ""},
                    color=wcol, color_continuous_scale="Blues"
                )
            else:
                bar_data = (
                    filt.groupby("state")[wcol].sum().reset_index()
                    .sort_values(wcol, ascending=False).head(15)
                )
                fig_bar = px.bar(
                    bar_data, x=wcol, y="state", orientation="h",
                    title=f"Top 15 States — {sel_sector} {sel_year}",
                    labels={wcol: "Workers", "state": "State"},
                    color=wcol, color_continuous_scale="Blues"
                )

        fig_bar.update_layout(
            height=430, showlegend=False, plot_bgcolor="white",
            paper_bgcolor="white", coloraxis_showscale=False,
            xaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),
            title_font_size=14, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True,
            key=f"bar_{sel_scenario}_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")

    # ── BOTTOM ROW: TREND + BREAKDOWN ─────────────────────────────────────────
    t_col, p_col = st.columns(2)

    with t_col:
        if has_sims:
            # Compare baseline trend vs simulation point for 2024
            trend_base = df[
                (df["scenario"] == "baseline") &
                (df["gtap_code"] == sel_sector if sel_sector != "All sectors"
                 else pd.Series([True]*len(df)))
            ]
            if sel_sector != "All sectors":
                trend_base = trend_base[trend_base["gtap_code"] == sel_sector]
            trend_data = (trend_base.groupby("year")["workers_base"]
                          .sum().reset_index())
            trend_data["type"] = "Baseline"

            fig_trend = px.line(
                trend_data, x="year", y="workers_base",
                title="Baseline Trend 2021-2024",
                markers=True,
                labels={"workers_base": "Workers", "year": "Year"},
                color_discrete_sequence=["#2E5496"]
            )

            # Add simulation point for 2024 if not baseline
            if sel_scenario != "baseline" and "workers_sim" in filt.columns:
                sim_total = filt["workers_sim"].fillna(0).sum()
                fig_trend.add_scatter(
                    x=[2024], y=[sim_total],
                    mode="markers",
                    marker=dict(size=14, color=SCENARIO_META.get(
                        sel_scenario, {}).get("color", "#e74c3c"),
                        symbol="diamond"),
                    name=SCENARIO_META.get(sel_scenario, {}).get("label", sel_scenario)
                )
        else:
            trend_base = df.copy()
            if sel_sector != "All sectors":
                trend_base = trend_base[trend_base["gtap_code"] == sel_sector]
            trend_data = (trend_base.groupby("year")["workers_base"]
                          .sum().reset_index())
            fig_trend = px.line(
                trend_data, x="year", y="workers_base",
                title="Trend 2021-2024", markers=True,
                labels={"workers_base": "Workers", "year": "Year"},
                color_discrete_sequence=["#2E5496"]
            )

        fig_trend.update_layout(
            height=280, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
            title_font_size=14, margin=dict(t=40, b=30)
        )
        st.plotly_chart(fig_trend, use_container_width=True,
            key=f"trend_{sel_scenario}_{sel_sector}_{sel_skill}_{sel_bp}")

    with p_col:
        if sel_scenario != "baseline" and has_sims and "workers_change" in filt.columns:
            # Skill x Birthplace breakdown of employment CHANGE
            split_data = (
                filt.groupby([skill_col, bp_col])
                ["workers_change"].sum().reset_index()
            )
            split_data["group"] = (split_data[skill_col] + " / " +
                                    split_data[bp_col])
            # Use diverging colors — red for losses, green for gains
            split_data["color"] = split_data["workers_change"].apply(
                lambda x: "#c0392b" if x < 0 else "#27ae60"
            )
            fig_pie = px.bar(
                split_data.sort_values("workers_change"),
                x="workers_change", y="group", orientation="h",
                title=f"Employment Change by Group",
                labels={"workers_change": "Change", "group": ""},
                color="workers_change",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0
            )
            fig_pie.update_layout(
                height=280, title_font_size=14, plot_bgcolor="white",
                paper_bgcolor="white", coloraxis_showscale=False,
                xaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
                margin=dict(t=40, b=10, l=10, r=10)
            )
        else:
            split_data = (
                filt.groupby([skill_col, bp_col])
                [wcol].sum().reset_index()
            )
            split_data["group"] = (split_data[skill_col] + " / " +
                                    split_data[bp_col])
            fig_pie = px.pie(
                split_data, values=wcol, names="group",
                title=f"Skill x Birthplace — {sel_year}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_layout(
                height=280, title_font_size=14,
                margin=dict(t=40, b=10), legend=dict(font_size=11)
            )
        st.plotly_chart(fig_pie, use_container_width=True,
            key=f"pie_{sel_scenario}_{sel_year}_{sel_sector}_{sel_skill}_{sel_bp}")

    # ── SCENARIO COMPARISON (only when simulations are present) ───────────────
    if has_sims:
        st.markdown("---")
        st.markdown("#### Scenario Comparison — 2024")
        st.markdown("Employment change relative to baseline across all scenarios.")

        sim_scenarios = [s for s in df["scenario"].unique() if s != "baseline"]
        comp_data = []
        for scen in sim_scenarios:
            s_filt = df[df["scenario"] == scen]
            if sel_sector != "All sectors":
                s_filt = s_filt[s_filt["gtap_code"] == sel_sector]
            if sel_skill != "All":
                s_filt = s_filt[s_filt[skill_col] == sel_skill]
            if sel_bp != "All":
                s_filt = s_filt[s_filt[bp_col] == sel_bp]
            if "workers_change" in s_filt.columns:
                comp_data.append({
                    "scenario": SCENARIO_META.get(scen, {}).get("label", scen),
                    "workers_change": int(s_filt["workers_change"].fillna(0).sum()),
                    "color": SCENARIO_META.get(scen, {}).get("color", "#888")
                })

        if comp_data:
            comp_df = pd.DataFrame(comp_data).sort_values("workers_change")
            fig_comp = px.bar(
                comp_df, x="workers_change", y="scenario", orientation="h",
                title="Employment Change by Scenario",
                labels={"workers_change": "Net Employment Change",
                        "scenario": ""},
                color="scenario",
                color_discrete_map={
                    r["scenario"]: r["color"] for r in comp_data
                }
            )
            fig_comp.add_vline(x=0, line_dash="dash", line_color="#666")
            fig_comp.update_layout(
                height=250, showlegend=False, plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
                margin=dict(t=40, b=10, l=10, r=10), title_font_size=14
            )
            st.plotly_chart(fig_comp, use_container_width=True,
                key=f"comp_{sel_sector}_{sel_skill}_{sel_bp}")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**API Keys**")
    default_anthropic = ""
    try:
        default_anthropic = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        default_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")

    if default_anthropic:
        anthropic_key = default_anthropic
        st.success("Anthropic API key configured")
    else:
        anthropic_key = st.text_input(
            "Anthropic API Key", type="password",
            help="Get your key at console.anthropic.com"
        )

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
        csv_path = st.text_input(
            "CSV path",
            value="gtap_master_with_simulations.csv",
            label_visibility="collapsed"
        )
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
        try:
    if "scenario" in df_s.columns and "workers_base" in df_s.columns:
        val = int(df_s[(df_s["scenario"]=="baseline") &
                       (df_s["year"]==2024)]["workers_base"].sum())
    elif "workers_base" in df_s.columns:
        val = int(df_s[df_s["year"]==2024]["workers_base"].sum())
    else:
        val = 0
    st.metric("Baseline Workers (2024)", f"{val:,}")
except Exception:
    st.metric("Baseline Workers (2024)", "—")
        st.metric("GTAP Sectors", df_s["gtap_code"].nunique())
        st.metric("Counties",     df_s["county_fips"].nunique())
        if has_simulations(df_s):
            n_scen = df_s["scenario"].nunique() - 1
            st.metric("Simulation Scenarios", n_scen)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example questions for Claude**")
    examples = [
        "What data is available?",
        "Which sectors have the most foreign-born workers?",
        "Show a map of v_f workers in 2022",
        "Compare employment change across all scenarios",
        "Which counties are most affected by the JPM deportation scenario?",
        "Show the impact of USMCA long run on agricultural employment",
        "Foreign born vs US born workers change under JPM sim03",
    ]
    for ex in examples:
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
    '<div class="sub-header">Interactive dashboard with simulation scenarios '
    'and conversational analysis for all 65 GTAP sectors across U.S. counties.</div>',
    unsafe_allow_html=True
)

if st.session_state.df is None:
    st.info("Load your data file using the sidebar to get started. "
            "Use **gtap_master_with_simulations.csv** to include simulation scenarios.")
    st.stop()

df = st.session_state.df

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Interactive Dashboard", "💬 Ask Claude"])

with tab1:
    render_dashboard(df)

with tab2:
    # Top KPIs
    latest = int(df["year"].max())
    df_l   = df[(df["year"] == latest) & (df["scenario"] == "baseline")
                if "scenario" in df.columns else df[df["year"] == latest]]
    tot    = df_l["workers_base"].sum()
    bp_col = "birthplace" if "birthplace" in df_l.columns else "birthplace_label"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{int(tot):,}", f"Baseline Workers {latest}"),
        (c2, f"{df_l[df_l[bp_col].str.contains('Foreign', na=False)]['workers_base'].sum()/tot*100:.1f}%",
             f"Foreign Born {latest}"),
        (c3, f"{df_l[df_l['skill_level']=='Skilled']['workers_base'].sum()/tot*100:.1f}%",
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
