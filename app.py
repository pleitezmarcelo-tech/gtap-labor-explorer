"""
GTAP Labor Data Explorer
Compatible with Python 3.14 and Streamlit 1.55+
"""

import os
import json
import streamlit as st
import pandas as pd
import anthropic
from tools import TOOL_DEFINITIONS, execute_tool

st.set_page_config(
    page_title="GTAP Labor Data Explorer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header { font-size:1.8rem; font-weight:700; color:#2E5496; margin-bottom:0.2rem; }
.sub-header  { font-size:0.95rem; color:#666; margin-bottom:1.5rem; }
.metric-card {
    background:#f0f4ff; border-radius:10px;
    padding:1rem 1.2rem; border-left:4px solid #2E5496; margin-bottom:4px;
}
.metric-card.neg { border-left-color:#c0392b; background:#fff0ef; }
.metric-card.pos { border-left-color:#27ae60; background:#f0fff4; }
.metric-value     { font-size:1.6rem; font-weight:700; color:#2E5496; }
.metric-value.neg { color:#c0392b; }
.metric-value.pos { color:#27ae60; }
.metric-label     { font-size:0.8rem; color:#666; text-transform:uppercase; }
.sidebar-section  { background:#f8f9fa; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("messages",[]), ("df",None), ("figures",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
PARQUET_REPO  = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
PARQUET_LOCAL = "gtap_master_with_simulations.parquet"

GTAP_ORDER = [
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

SCENARIO_META = {
    "baseline":   {"label":"Baseline (Observed)",  "color":"#2E5496"},
    "JPM_sim03":  {"label":"JPM 2025 — sim03",      "color":"#c0392b"},
    "JPM_sim03b": {"label":"JPM 2025 — sim03b",     "color":"#e74c3c"},
    "JPM_sim03c": {"label":"JPM 2025 — sim03c",     "color":"#e67e22"},
    "USMCA_SR":   {"label":"USMCA — Short Run",     "color":"#8e44ad"},
    "USMCA_LR":   {"label":"USMCA — Long Run",      "color":"#2980b9"},
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_parquet():
    path = PARQUET_REPO if os.path.exists(PARQUET_REPO) else PARQUET_LOCAL
    df = pd.read_parquet(path)
    if "county_fips" in df.columns:
        df["county_fips"] = (df["county_fips"].fillna("").astype(str)
                             .str.strip().str.replace(".0","",regex=False).str.zfill(5))
        df.loc[df["county_fips"]=="00000","county_fips"] = ""
    for col in ["workers_base","workers_sim","workers_change","pct_change","lq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "scenario" not in df.columns:
        df["scenario"] = "baseline"
    return df

@st.cache_data(show_spinner="Loading file...")
def load_file(path_or_buffer, name=""):
    n = name if name else str(path_or_buffer)
    df = pd.read_parquet(path_or_buffer) if n.endswith(".parquet") else \
         pd.read_csv(path_or_buffer, dtype={"county_fips":str})
    if "county_fips" in df.columns:
        df["county_fips"] = (df["county_fips"].fillna("").astype(str)
                             .str.strip().str.replace(".0","",regex=False).str.zfill(5))
    for col in ["workers_base","workers_sim","workers_change","pct_change","lq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "scenario" not in df.columns:
        df["scenario"] = "baseline"
    return df

# ── HELPERS ───────────────────────────────────────────────────────────────────
def has_sims(df):
    return "scenario" in df.columns and df["scenario"].nunique() > 1

def wcol(scenario):
    return "workers_base" if scenario == "baseline" else "workers_sim"

def metric_card(val, label, cls=""):
    return (f'<div class="metric-card {cls}">'
            f'<div class="metric-value {cls}">{val}</div>'
            f'<div class="metric-label">{label}</div></div>')

def ordered_sectors(df):
    available = set(df["gtap_code"].dropna().unique())
    ordered = [s for s in GTAP_ORDER if s in available]
    ordered += [s for s in available if s not in ordered]
    return ordered

def sector_label_map(df):
    return {
        r["gtap_code"]: f"{r['gtap_code']} — {r['gtap_sector']}"
        for _, r in df[["gtap_code","gtap_sector"]].drop_duplicates().iterrows()
        if pd.notna(r["gtap_code"]) and pd.notna(r["gtap_sector"])
    }

def bpcol(df):
    return "birthplace" if "birthplace" in df.columns else "birthplace_label"

# ── CLAUDE AGENT ──────────────────────────────────────────────────────────────
def run_agent(user_message, df, api_key, usda_key):
    client = anthropic.Anthropic(api_key=api_key)
    history = []
    for m in st.session_state.messages[-12:]:
        if m["role"] in ("user","assistant") and isinstance(m.get("content"), str):
            history.append({"role": m["role"], "content": m["content"]})
    history.append({"role":"user","content":user_message})

    sims_note = ""
    if has_sims(df):
        sims_note = ("\nDataset includes simulation scenarios. "
                     "workers_sim=simulated, workers_change=net change, pct_change=% change.")

    system = (
        "You are an expert on U.S. labor markets and GTAP. "
        "You have access to estimated workers for 65 GTAP sectors across U.S. counties, "
        f"years 2021-2024, by skill level and birthplace.{sims_note}\n"
        "Always use tools for numbers. Use create_map for maps. "
        "Use create_chart for trends. Use query_dataset for specific figures."
    )

    figures, text, msgs = [], "", history.copy()
    for i in range(6):
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=2048,
            system=system, tools=TOOL_DEFINITIONS, messages=msgs
        )
        for b in resp.content:
            if b.type == "text":
                text += b.text
        if resp.stop_reason != "tool_use":
            break
        tool_results = []
        for b in resp.content:
            if b.type == "tool_use":
                result, is_fig = execute_tool(b.name, b.input, df, usda_key)
                if is_fig and result is not None:
                    fid = f"fig_{i}_{b.id[:8]}"
                    figures.append((fid, result))
                    content = json.dumps({"success":True,"figure_id":fid})
                else:
                    content = json.dumps(result)
                tool_results.append({"type":"tool_result","tool_use_id":b.id,"content":content})
        msgs.append({"role":"assistant","content":resp.content})
        msgs.append({"role":"user","content":tool_results})
    return text, figures

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_dashboard(df):
    import plotly.express as px
    _has_sims = has_sims(df)
    bpc = bpcol(df)
    sk_col = "skill_level"

    st.markdown("### Interactive Dashboard")
    st.caption("All charts update in real time when you change filters.")

    # Sector aggregation toggle
    agg_view = st.radio("Sector view",
        ["65 GTAP sectors", "27 Model sectors"],
        horizontal=True, key="d_agg")
    use_agg = (agg_view == "27 Model sectors") and ("model_sector_code" in df.columns)
    sc_col = "model_sector_code" if use_agg else "gtap_code"
    sd_col = "model_sector_desc" if (use_agg and "model_sector_desc" in df.columns) else "gtap_sector"

    # Filters
    ncols = 5 if _has_sims else 4
    cols = st.columns(ncols)

    with cols[0]:
        if _has_sims:
            scen_opts = sorted(df["scenario"].unique().tolist())
            sel_scen = st.selectbox("Scenario", scen_opts, key="d_scen",
                format_func=lambda x: SCENARIO_META.get(x,{}).get("label",x))
        else:
            sel_scen = "baseline"

    yr_col = cols[1] if _has_sims else cols[0]
    with yr_col:
        if _has_sims and sel_scen != "baseline":
            years = [2024]
        else:
            years = sorted(df[df["scenario"]=="baseline"]["year"].dropna().unique().tolist()) or [2024]
        sel_year = st.selectbox("Year", years, index=len(years)-1, key="d_year")

    sec_col = cols[2] if _has_sims else cols[1]
    sk_c    = cols[3] if _has_sims else cols[2]
    bp_c    = cols[4] if _has_sims else cols[3]

    with sec_col:
        if use_agg:
            msecs = sorted(df[sc_col].dropna().unique().tolist())
            mlmap = {r[sc_col]: f'{r[sc_col]} — {r[sd_col]}'
                for _, r in df[[sc_col,sd_col]].drop_duplicates().iterrows()
                if pd.notna(r[sc_col])}
            sel_sec = st.selectbox("Model Sector", ["All sectors"]+msecs, key="d_sec",
                format_func=lambda x: mlmap.get(x,x) if x!="All sectors" else "All sectors")
        else:
            secs = ordered_sectors(df); slbls = sector_label_map(df)
            sel_sec = st.selectbox("GTAP Sector", ["All sectors"]+secs, key="d_sec",
                format_func=lambda x: slbls.get(x,x) if x!="All sectors" else "All sectors")

    with sk_c:
        sk_opts = ["All"] + sorted(df[sk_col].dropna().unique().tolist())
        sel_sk = st.selectbox("Skill Level", sk_opts, key="d_sk")

    with bp_c:
        bp_opts = ["All"] + sorted(df[bpc].dropna().unique().tolist())
        sel_bp = st.selectbox("Birthplace", bp_opts, key="d_bp")

    # Apply filters
    filt = df[(df["scenario"]==sel_scen) & (df["year"]==sel_year)].copy()
    if sel_sec != "All sectors": filt = filt[filt[sc_col]==sel_sec]
    if sel_sk  != "All":         filt = filt[filt[sk_col]==sel_sk]
    if sel_bp  != "All":         filt = filt[filt[bpc]==sel_bp]

    wc    = wcol(sel_scen)
    total = int(filt[wc].fillna(0).sum()) if wc in filt.columns else 0
    color = SCENARIO_META.get(sel_scen,{}).get("color","#2E5496")

    # Scenario banner
    if _has_sims and sel_scen != "baseline":
        lbl = SCENARIO_META.get(sel_scen,{}).get("label",sel_scen)
        st.markdown(
            f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:8px 14px;border-radius:6px;margin-bottom:10px;">'
            f'<b style="color:{color}">{lbl}</b></div>',
            unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    if sel_scen != "baseline" and _has_sims and "workers_change" in filt.columns:
        chg  = int(filt["workers_change"].fillna(0).sum())
        fb_c = int(filt[filt[bpc].str.contains("Foreign",na=False)]["workers_change"].fillna(0).sum())
        sk_c2 = int(filt[filt[sk_col]=="Unskilled"]["workers_change"].fillna(0).sum())
        for col, val, lbl, neg in [
            (k1, f"{total:,}",   f"Simulated Workers {sel_year}", False),
            (k2, f"{chg:+,}",    "Net Employment Change",   chg<0),
            (k3, f"{fb_c:+,}",   "Foreign Born Change",     fb_c<0),
            (k4, f"{sk_c2:+,}",  "Unskilled Change",        sk_c2<0),
        ]:
            cls = "neg" if neg else ("pos" if str(val).startswith("+") else "")
            with col: st.markdown(metric_card(val,lbl,cls), unsafe_allow_html=True)
    else:
        fb = filt[filt[bpc].str.contains("Foreign",na=False)][wc].sum() if wc in filt.columns else 0
        sk = filt[filt[sk_col]=="Skilled"][wc].sum() if wc in filt.columns else 0
        for col, val, lbl in [
            (k1, f"{total:,}",                              f"Workers {sel_year}"),
            (k2, f"{fb/total*100:.1f}%" if total else "—", "Foreign Born"),
            (k3, f"{sk/total*100:.1f}%" if total else "—", "Skilled"),
            (k4, str(filt["county_fips"].nunique()),         "Counties"),
        ]:
            with col: st.markdown(metric_card(val,lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Map + Bar
    mc, bc = st.columns([3,2])
    is_chg = sel_scen != "baseline" and _has_sims and "workers_change" in filt.columns

    with mc:
        if is_chg:
            cagg = filt.groupby(["county_fips","county_name","state"])["workers_change"].sum().reset_index()
            ccol, cscale, cmid, clbl = "workers_change","RdYlGn",0,"Employment Change"
            mtitle = f"Employment Change — {SCENARIO_META.get(sel_scen,{}).get('label',sel_scen)}"
        else:
            cagg = filt.groupby(["county_fips","county_name","state"])[wc].sum().reset_index() if wc in filt.columns else pd.DataFrame()
            if len(cagg) and wc in cagg.columns:
                cagg = cagg[cagg[wc]>0]
            ccol, cscale, cmid, clbl = wc,"Blues",None,"Workers"
            mtitle = f"Workers by County — {sel_year}"
            if sel_sec != "All sectors": mtitle += f" | {sel_sec}"

        if len(cagg) > 0 and "county_fips" in cagg.columns:
            cagg = cagg[cagg["county_fips"].str.len()==5]

        if len(cagg) > 0:
            kw = {"color_continuous_midpoint":cmid} if cmid is not None else {}
            fig_map = px.choropleth(
                cagg, locations="county_fips", color=ccol,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                color_continuous_scale=cscale, scope="usa",
                hover_data={"county_fips":False,"county_name":True,"state":True,ccol:":,"},
                title=mtitle, labels={ccol:clbl,"county_name":"County","state":"State"}, **kw
            )
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=430,
                coloraxis_colorbar=dict(title=clbl,tickformat=",d"), title_font_size=14)
            st.plotly_chart(fig_map, key=f"map_{sel_scen}_{sel_year}_{sel_sec}_{sel_sk}_{sel_bp}",
                            width="stretch")
        else:
            st.info("No county data for this combination of filters.")

    with bc:
        # Bar chart title based on aggregation level
        sec_title = "Model Sectors" if use_agg else "GTAP Sectors"
        if is_chg:
            if sel_sec == "All sectors":
                bd = (filt.groupby([sc_col, sd_col])["workers_change"]
                      .sum().reset_index().sort_values("workers_change").head(15))
                bd["label"] = bd[sc_col] + " — " + bd[sd_col].str[:20]
                fig_bar = px.bar(bd, x="workers_change", y="label", orientation="h",
                    title=f"Top 15 {sec_title} by Change", color="workers_change",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    labels={"workers_change":"Change","label":""})
            else:
                bd = (filt.groupby("state")["workers_change"]
                      .sum().reset_index().sort_values("workers_change").head(15))
                fig_bar = px.bar(bd, x="workers_change", y="state", orientation="h",
                    title=f"Top 15 States — {sel_sec}", color="workers_change",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    labels={"workers_change":"Change","state":"State"})
        else:
            if wc not in filt.columns:
                fig_bar = px.bar(title="No data")
            elif sel_sec == "All sectors":
                bd = (filt.groupby([sc_col, sd_col])[wc]
                      .sum().reset_index().sort_values(wc, ascending=False).head(15))
                bd["label"] = bd[sc_col] + " — " + bd[sd_col].str[:20]
                fig_bar = px.bar(bd, x=wc, y="label", orientation="h",
                    title=f"Top 15 {sec_title} — {sel_year}", color=wc,
                    color_continuous_scale="Blues", labels={wc:"Workers","label":""})
            else:
                bd = (filt.groupby("state")[wc].sum().reset_index()
                      .sort_values(wc, ascending=False).head(15))
                fig_bar = px.bar(bd, x=wc, y="state", orientation="h",
                    title=f"Top 15 States — {sel_sec} {sel_year}", color=wc,
                    color_continuous_scale="Blues", labels={wc:"Workers","state":"State"})

        fig_bar.update_layout(height=430, showlegend=False, plot_bgcolor="white",
            paper_bgcolor="white", coloraxis_showscale=False,
            xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"), title_font_size=14,
            margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_bar, key=f"bar_{sel_scen}_{sel_year}_{sel_sec}_{sel_sk}_{sel_bp}",
                        width="stretch")

    # Trend + breakdown
    tc, pc = st.columns(2)
    with tc:
        tbase = df[df["scenario"]=="baseline"].copy()
        if sel_sec != "All sectors": tbase = tbase[tbase["gtap_code"]==sel_sec]
        if sel_sk  != "All":         tbase = tbase[tbase[sk_col]==sel_sk]
        if sel_bp  != "All":         tbase = tbase[tbase[bpc]==sel_bp]
        td = tbase.groupby("year")["workers_base"].sum().reset_index()
        fig_tr = px.line(td, x="year", y="workers_base", title="Baseline Trend 2021–2024",
            markers=True, color_discrete_sequence=["#2E5496"],
            labels={"workers_base":"Workers","year":"Year"})
        if sel_scen != "baseline" and _has_sims and "workers_sim" in filt.columns:
            sv = filt["workers_sim"].fillna(0).sum()
            fig_tr.add_scatter(x=[2024], y=[sv], mode="markers",
                marker=dict(size=14,color=color,symbol="diamond"),
                name=SCENARIO_META.get(sel_scen,{}).get("label",sel_scen))
        fig_tr.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
            title_font_size=14, margin=dict(t=40,b=30))
        st.plotly_chart(fig_tr, key=f"tr_{sel_scen}_{sel_sec}_{sel_sk}_{sel_bp}",
                        width="stretch")

    with pc:
        if is_chg:
            sd = filt.groupby([sk_col,bpc])["workers_change"].sum().reset_index()
            sd["group"] = sd[sk_col] + " / " + sd[bpc]
            fig_pc = px.bar(sd.sort_values("workers_change"), x="workers_change", y="group",
                orientation="h", title="Change by Group", color="workers_change",
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                labels={"workers_change":"Change","group":""})
            fig_pc.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False, xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
                title_font_size=14, margin=dict(t=40,b=10,l=10,r=10))
        else:
            wcc = wc if wc in filt.columns else "workers_base"
            if wcc in filt.columns and len(filt):
                sd = filt.groupby([sk_col,bpc])[wcc].sum().reset_index()
                sd["group"] = sd[sk_col] + " / " + sd[bpc]
                fig_pc = px.pie(sd, values=wcc, names="group",
                    title=f"Skill x Birthplace — {sel_year}",
                    color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig_pc = px.pie(title="No data")
            fig_pc.update_layout(height=280, title_font_size=14,
                margin=dict(t=40,b=10), legend=dict(font_size=11))
        st.plotly_chart(fig_pc, key=f"pc_{sel_scen}_{sel_year}_{sel_sec}_{sel_sk}_{sel_bp}",
                        width="stretch")

    # Scenario comparison
    if _has_sims:
        st.markdown("---")
        st.markdown("#### Scenario Comparison — 2024")
        comp = []
        for scen in [s for s in df["scenario"].unique() if s!="baseline"]:
            sf = df[df["scenario"]==scen]
            if sel_sec != "All sectors": sf = sf[sf["gtap_code"]==sel_sec]
            if sel_sk  != "All":         sf = sf[sf[sk_col]==sel_sk]
            if sel_bp  != "All":         sf = sf[sf[bpc]==sel_bp]
            if "workers_change" in sf.columns:
                comp.append({
                    "scenario": SCENARIO_META.get(scen,{}).get("label",scen),
                    "workers_change": int(sf["workers_change"].fillna(0).sum()),
                    "color": SCENARIO_META.get(scen,{}).get("color","#888")
                })
        if comp:
            cdf = pd.DataFrame(comp).sort_values("workers_change")
            fig_cmp = px.bar(cdf, x="workers_change", y="scenario", orientation="h",
                title="Employment Change by Scenario", color="scenario",
                color_discrete_map={r["scenario"]:r["color"] for r in comp},
                labels={"workers_change":"Net Change","scenario":""})
            fig_cmp.add_vline(x=0, line_dash="dash", line_color="#666")
            fig_cmp.update_layout(height=250, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
                margin=dict(t=40,b=10,l=10,r=10), title_font_size=14)
            st.plotly_chart(fig_cmp, key=f"cmp_{sel_sec}_{sel_sk}_{sel_bp}",
                            width="stretch")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**API Keys**")
    api_key = ""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.success("Anthropic API key configured")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY","")
        if api_key:
            st.success("Anthropic API key configured")
        else:
            api_key = st.text_input("Anthropic API Key", type="password",
                help="Get your key at console.anthropic.com")
    usda_key = st.text_input("USDA NASS API Key (optional)", type="password",
        value=os.environ.get("USDA_API_KEY",""))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Data**")
    src = st.radio("Source", ["Auto (built-in)", "Upload file", "Enter path"],
                   label_visibility="collapsed")

    if src == "Auto (built-in)":
        if st.session_state.df is None:
            if st.button("Load Data"):
                try:
                    st.session_state.df = load_parquet()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.success(f"Loaded: {len(st.session_state.df):,} rows")
    elif src == "Upload file":
        up = st.file_uploader("File", type=["csv","parquet"], label_visibility="collapsed")
        if up:
            st.session_state.df = load_file(up, up.name)
            st.rerun()
    else:
        fp = st.text_input("Path", value=PARQUET_LOCAL, label_visibility="collapsed")
        if st.button("Load"):
            try:
                st.session_state.df = load_file(fp, fp)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df_s = st.session_state.df
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Dataset Summary**")
        try:
            if "scenario" in df_s.columns and "year" in df_s.columns and "workers_base" in df_s.columns:
                w = int(df_s[(df_s["scenario"]=="baseline")&(df_s["year"]==2024)]["workers_base"].sum())
            elif "workers_base" in df_s.columns:
                w = int(df_s["workers_base"].sum())
            else:
                w = 0
            st.metric("Baseline Workers (2024)", f"{w:,}")
            if "gtap_code" in df_s.columns:
                st.metric("GTAP Sectors", df_s["gtap_code"].nunique())
            if "county_fips" in df_s.columns:
                st.metric("Counties", df_s["county_fips"].nunique())
            if has_sims(df_s):
                st.metric("Scenarios", df_s["scenario"].nunique()-1)
        except Exception:
            st.write(f"Rows: {len(df_s):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example questions**")
    for ex in [
        "What data is available?",
        "Which sectors have the most foreign-born workers?",
        "Show a map of v_f workers in 2022",
        "Which counties are most affected by JPM sim03?",
        "Compare employment change across all scenarios",
        "Show USMCA long run impact on agriculture",
    ]:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.figures = {}
        st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 GTAP Labor Data Explorer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive dashboard and conversational analysis '
    'for all 65 GTAP sectors across U.S. counties.</div>',
    unsafe_allow_html=True)

if st.session_state.df is None:
    st.info("Click **Load Data** in the sidebar to get started.")
    st.stop()

df = st.session_state.df

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Interactive Dashboard", "💬 Ask Claude"])

with tab1:
    render_dashboard(df)

with tab2:
    bpc2 = bpcol(df)
    latest = int(df["year"].max()) if "year" in df.columns else 2024
    df_l = df[(df["scenario"]=="baseline")&(df["year"]==latest)] if "scenario" in df.columns else df
    tot  = df_l["workers_base"].sum() if "workers_base" in df_l.columns else 0
    fb2  = df_l[df_l[bpc2].str.contains("Foreign",na=False)]["workers_base"].sum() if "workers_base" in df_l.columns else 0
    sk2  = df_l[df_l["skill_level"]=="Skilled"]["workers_base"].sum() if "workers_base" in df_l.columns else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, f"{int(tot):,}",                         f"Baseline {latest}"),
        (c2, f"{fb2/tot*100:.1f}%" if tot else "—",   "Foreign Born"),
        (c3, f"{sk2/tot*100:.1f}%" if tot else "—",   "Skilled"),
        (c4, str(df["gtap_code"].nunique()) if "gtap_code" in df.columns else "—", "GTAP Sectors"),
        (c5, f"{df['county_fips'].nunique():,}" if "county_fips" in df.columns else "—", "Counties"),
    ]:
        with col:
            st.markdown(metric_card(val,lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if isinstance(msg.get("content"), str):
                st.markdown(msg["content"])
            for fid in msg.get("figures",[]):
                if fid in st.session_state.figures:
                    st.plotly_chart(st.session_state.figures[fid],
                        key=f"h_{fid}_{i}", width="stretch")

    prefill    = st.session_state.pop("prefill","")
    user_input = st.chat_input("Ask Claude about the GTAP labor data...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    resp_text, figs = run_agent(user_input, df, api_key, usda_key)
                    if resp_text:
                        st.markdown(resp_text)
                    fids = []
                    for fid, fig in figs:
                        st.plotly_chart(fig, key=f"new_{fid}", width="stretch")
                        st.session_state.figures[fid] = fig
                        fids.append(fid)
                    st.session_state.messages.append({
                        "role":"assistant",
                        "content": resp_text or "(See chart above)",
                        "figures": fids
                    })
                except anthropic.AuthenticationError:
                    st.error("Invalid API key.")
                except Exception as e:
                    st.error(f"Error: {e}")
