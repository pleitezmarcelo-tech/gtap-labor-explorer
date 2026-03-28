"""
GTAP Labor Data Explorer - Production Version
"""
import os, json
import streamlit as st
import pandas as pd
import anthropic
from tools import TOOL_DEFINITIONS, execute_tool

st.set_page_config(page_title="GTAP Labor Data Explorer", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.mh{font-size:1.8rem;font-weight:700;color:#2E5496}
.sh{font-size:.95rem;color:#666;margin-bottom:1.5rem}
.mc{background:#f0f4ff;border-radius:10px;padding:1rem 1.2rem;border-left:4px solid #2E5496;margin-bottom:4px}
.mc.neg{border-left-color:#c0392b;background:#fff0ef}
.mc.pos{border-left-color:#27ae60;background:#f0fff4}
.mv{font-size:1.6rem;font-weight:700;color:#2E5496}
.mv.neg{color:#c0392b}.mv.pos{color:#27ae60}
.ml{font-size:.8rem;color:#666;text-transform:uppercase}
.ss{background:#f8f9fa;border-radius:8px;padding:.8rem;margin-bottom:.8rem}
</style>""", unsafe_allow_html=True)

for k,v in [("messages",[]),("df",None),("figures",{})]:
    if k not in st.session_state: st.session_state[k]=v

PARQUET_REPO  = "/mount/src/gtap-labor-explorer/gtap_master_with_simulations.parquet"
PARQUET_LOCAL = "gtap_master_with_simulations.parquet"
GTAP_ORDER = ["pdr","wht","gro","v_f","osd","c_b","pfb","ocr","ctl","oap","rmk","wol",
    "frs","fsh","coa","oil","gas","oxt","cmt","omt","vol","mil","pcr","sgr","ofd","b_t",
    "tex","wap","lea","lum","ppp","chm","bph","rpp","nmm","i_s","nfm","fmp","mvh","otn",
    "ele","eeq","ome","omf","p_c","ely","gdt","wtr","cns","trd","afs","otp","wtp","atp",
    "cmn","ofi","ins","rsa","obs","ros","osg","edu","hht","dwe"]
SMETA = {
    "baseline":  {"label":"Baseline (Observed)","color":"#2E5496"},
    "JPM_sim03": {"label":"JPM 2025 — sim03","color":"#c0392b"},
    "JPM_sim03b":{"label":"JPM 2025 — sim03b","color":"#e74c3c"},
    "JPM_sim03c":{"label":"JPM 2025 — sim03c","color":"#e67e22"},
    "USMCA_SR":  {"label":"USMCA — Short Run","color":"#8e44ad"},
    "USMCA_LR":  {"label":"USMCA — Long Run","color":"#2980b9"},
}

@st.cache_data(show_spinner="Loading dataset...")
def load_parquet():
    path = PARQUET_REPO if os.path.exists(PARQUET_REPO) else PARQUET_LOCAL
    df = pd.read_parquet(path)
    if "county_fips" in df.columns:
        df["county_fips"] = (df["county_fips"].fillna("").astype(str)
            .str.strip().str.replace(".0","",regex=False).str.zfill(5))
        df.loc[df["county_fips"]=="00000","county_fips"] = ""
    for c in ["workers_base","workers_sim","workers_change","pct_change","lq"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    if "scenario" not in df.columns: df["scenario"]="baseline"
    return df

@st.cache_data(show_spinner="Loading...")
def load_file(p,n=""):
    nm=n if n else str(p)
    df=pd.read_parquet(p) if nm.endswith(".parquet") else pd.read_csv(p,dtype={"county_fips":str})
    if "county_fips" in df.columns:
        df["county_fips"]=(df["county_fips"].fillna("").astype(str)
            .str.strip().str.replace(".0","",regex=False).str.zfill(5))
    for c in ["workers_base","workers_sim","workers_change","pct_change","lq"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    if "scenario" not in df.columns: df["scenario"]="baseline"
    return df

def has_sims(df): return "scenario" in df.columns and df["scenario"].nunique()>1
def wcol(s): return "workers_base" if s=="baseline" else "workers_sim"
def mc(val,lbl,cls=""): return f'<div class="mc {cls}"><div class="mv {cls}">{val}</div><div class="ml">{lbl}</div></div>'
def bpc(df): return "birthplace" if "birthplace" in df.columns else "birthplace_label"
def osecs(df):
    av=set(df["gtap_code"].dropna().unique())
    r=[s for s in GTAP_ORDER if s in av]; r+=[s for s in av if s not in r]; return r
def slmap(df):
    return {r["gtap_code"]:f'{r["gtap_code"]} — {r["gtap_sector"]}'
        for _,r in df[["gtap_code","gtap_sector"]].drop_duplicates().iterrows()
        if pd.notna(r["gtap_code"]) and pd.notna(r["gtap_sector"])}

def run_agent(msg,df,key,ukey):
    client=anthropic.Anthropic(api_key=key)
    hist=[{"role":m["role"],"content":m["content"]}
        for m in st.session_state.messages[-12:]
        if m["role"] in ("user","assistant") and isinstance(m.get("content"),str)]
    hist.append({"role":"user","content":msg})
    sn="" if not has_sims(df) else "\nDataset has simulation scenarios. workers_sim=simulated, workers_change=net change."
    sys=(f"You are an expert on U.S. labor markets and GTAP.{sn}\n"
         "Use tools for all numbers. create_map for maps. create_chart for trends.")
    figs,text,msgs=[],"",[*hist]
    for i in range(6):
        r=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=2048,
            system=sys,tools=TOOL_DEFINITIONS,messages=msgs)
        for b in r.content:
            if b.type=="text": text+=b.text
        if r.stop_reason!="tool_use": break
        tr=[]
        for b in r.content:
            if b.type=="tool_use":
                res,isf=execute_tool(b.name,b.input,df,ukey)
                if isf and res is not None:
                    fid=f"fig_{i}_{b.id[:8]}"; figs.append((fid,res))
                    ct=json.dumps({"success":True,"figure_id":fid})
                else: ct=json.dumps(res)
                tr.append({"type":"tool_result","tool_use_id":b.id,"content":ct})
        msgs.append({"role":"assistant","content":r.content})
        msgs.append({"role":"user","content":tr})
    return text,figs

def render_dashboard(df):
    import plotly.express as px
    hs=has_sims(df); bp=bpc(df); sk="skill_level"
    st.markdown("### Interactive Dashboard")
    st.caption("All charts update in real time.")
    nc=5 if hs else 4; cols=st.columns(nc)
    with cols[0]:
        if hs:
            so=sorted(df["scenario"].unique().tolist())
            ss=st.selectbox("Scenario",so,key="d_scen",
                format_func=lambda x:SMETA.get(x,{}).get("label",x))
        else: ss="baseline"
    yc=cols[1] if hs else cols[0]
    with yc:
        yrs=[2024] if (hs and ss!="baseline") else (sorted(df[df["scenario"]=="baseline"]["year"].dropna().unique().tolist()) or [2024])
        sy=st.selectbox("Year",yrs,index=len(yrs)-1,key="d_year")
    sc2=cols[2] if hs else cols[1]; skc=cols[3] if hs else cols[2]; bpc2=cols[4] if hs else cols[3]
    with sc2:
        secs=osecs(df); slbs=slmap(df)
        sec=st.selectbox("GTAP Sector",["All sectors"]+secs,key="d_sec",
            format_func=lambda x:slbs.get(x,x) if x!="All sectors" else "All sectors")
    with skc:
        sko=["All"]+sorted(df[sk].dropna().unique().tolist())
        ssk=st.selectbox("Skill Level",sko,key="d_sk")
    with bpc2:
        bpo=["All"]+sorted(df[bp].dropna().unique().tolist())
        sbp=st.selectbox("Birthplace",bpo,key="d_bp")
    filt=df[(df["scenario"]==ss)&(df["year"]==sy)].copy()
    if sec!="All sectors": filt=filt[filt["gtap_code"]==sec]
    if ssk!="All": filt=filt[filt[sk]==ssk]
    if sbp!="All": filt=filt[filt[bp]==sbp]
    wc=wcol(ss); tot=int(filt[wc].fillna(0).sum()) if wc in filt.columns else 0
    col=SMETA.get(ss,{}).get("color","#2E5496")
    if hs and ss!="baseline":
        st.markdown(f'<div style="background:{col}18;border-left:4px solid {col};padding:8px 14px;border-radius:6px;margin-bottom:10px;"><b style="color:{col}">{SMETA.get(ss,{}).get("label",ss)}</b></div>',unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    ic=ss!="baseline" and hs and "workers_change" in filt.columns
    if ic:
        chg=int(filt["workers_change"].fillna(0).sum())
        fbc=int(filt[filt[bp].str.contains("Foreign",na=False)]["workers_change"].fillna(0).sum())
        skc2=int(filt[filt[sk]=="Unskilled"]["workers_change"].fillna(0).sum())
        for c,v,l,n in[(k1,f"{tot:,}",f"Simulated {sy}",False),(k2,f"{chg:+,}","Net Change",chg<0),(k3,f"{fbc:+,}","Foreign Born",fbc<0),(k4,f"{skc2:+,}","Unskilled",skc2<0)]:
            cls="neg" if n else ("pos" if str(v).startswith("+") else "")
            with c: st.markdown(mc(v,l,cls),unsafe_allow_html=True)
    else:
        fb=filt[filt[bp].str.contains("Foreign",na=False)][wc].sum() if wc in filt.columns else 0
        skv=filt[filt[sk]=="Skilled"][wc].sum() if wc in filt.columns else 0
        for c,v,l in[(k1,f"{tot:,}",f"Workers {sy}"),(k2,f"{fb/tot*100:.1f}%" if tot else "—","Foreign Born"),(k3,f"{skv/tot*100:.1f}%" if tot else "—","Skilled"),(k4,str(filt["county_fips"].nunique()),"Counties")]:
            with c: st.markdown(mc(v,l),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    mc2,bc2=st.columns([3,2])
    with mc2:
        if ic:
            ca=filt.groupby(["county_fips","county_name","state"])["workers_change"].sum().reset_index()
            cc,cs,cm,cl="workers_change","RdYlGn",0,"Employment Change"
            mt=f"Employment Change — {SMETA.get(ss,{}).get('label',ss)}"
        else:
            ca=filt.groupby(["county_fips","county_name","state"])[wc].sum().reset_index() if wc in filt.columns else pd.DataFrame()
            if len(ca) and wc in ca.columns: ca=ca[ca[wc]>0]
            cc,cs,cm,cl=wc,"Blues",None,"Workers"
            mt=f"Workers by County — {sy}"+(f" | {sec}" if sec!="All sectors" else "")
        if len(ca)>0 and "county_fips" in ca.columns: ca=ca[ca["county_fips"].str.len()==5]
        if len(ca)>0:
            kw={"color_continuous_midpoint":cm} if cm is not None else {}
            fig=px.choropleth(ca,locations="county_fips",color=cc,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                color_continuous_scale=cs,scope="usa",
                hover_data={"county_fips":False,"county_name":True,"state":True,cc:":,"},
                title=mt,labels={cc:cl,"county_name":"County","state":"State"},**kw)
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0},height=430,
                coloraxis_colorbar=dict(title=cl,tickformat=",d"),title_font_size=14)
            st.plotly_chart(fig,key=f"map_{ss}_{sy}_{sec}_{ssk}_{sbp}")
        else: st.info("No county data for this combination.")
    with bc2:
        if ic:
            bd=(filt.groupby(["gtap_code","gtap_sector"] if sec=="All sectors" else ["state"])
                [("workers_change")].sum().reset_index().sort_values("workers_change").head(15))
            if sec=="All sectors": bd["label"]=bd["gtap_code"]+" — "+bd["gtap_sector"].str[:18]
            ycol="label" if sec=="All sectors" else "state"
            fig=px.bar(bd,x="workers_change",y=ycol,orientation="h",
                title="Top 15 by Change",color="workers_change",
                color_continuous_scale="RdYlGn",color_continuous_midpoint=0,
                labels={"workers_change":"Change",ycol:""})
        else:
            if wc not in filt.columns: fig=px.bar(title="No data")
            else:
                bd=(filt.groupby(["gtap_code","gtap_sector"] if sec=="All sectors" else ["state"])
                    [wc].sum().reset_index().sort_values(wc,ascending=False).head(15))
                if sec=="All sectors": bd["label"]=bd["gtap_code"]+" — "+bd["gtap_sector"].str[:18]
                ycol="label" if sec=="All sectors" else "state"
                fig=px.bar(bd,x=wc,y=ycol,orientation="h",
                    title=f"Top 15 — {sy}",color=wc,color_continuous_scale="Blues",
                    labels={wc:"Workers",ycol:""})
        fig.update_layout(height=430,showlegend=False,plot_bgcolor="white",paper_bgcolor="white",
            coloraxis_showscale=False,xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),title_font_size=14,margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig,key=f"bar_{ss}_{sy}_{sec}_{ssk}_{sbp}")
    tc,pc2=st.columns(2)
    with tc:
        tb=df[df["scenario"]=="baseline"].copy()
        if sec!="All sectors": tb=tb[tb["gtap_code"]==sec]
        if ssk!="All": tb=tb[tb[sk]==ssk]
        if sbp!="All": tb=tb[tb[bp]==sbp]
        td=tb.groupby("year")["workers_base"].sum().reset_index()
        fig=px.line(td,x="year",y="workers_base",title="Baseline Trend 2021–2024",
            markers=True,color_discrete_sequence=["#2E5496"],
            labels={"workers_base":"Workers","year":"Year"})
        if ss!="baseline" and hs and "workers_sim" in filt.columns:
            sv=filt["workers_sim"].fillna(0).sum()
            fig.add_scatter(x=[2024],y=[sv],mode="markers",
                marker=dict(size=14,color=col,symbol="diamond"),
                name=SMETA.get(ss,{}).get("label",ss))
        fig.update_layout(height=280,plot_bgcolor="white",paper_bgcolor="white",
            yaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),title_font_size=14,margin=dict(t=40,b=30))
        st.plotly_chart(fig,key=f"tr_{ss}_{sec}_{ssk}_{sbp}")
    with pc2:
        if ic:
            sd=filt.groupby([sk,bp])["workers_change"].sum().reset_index()
            sd["group"]=sd[sk]+" / "+sd[bp]
            fig=px.bar(sd.sort_values("workers_change"),x="workers_change",y="group",
                orientation="h",title="Change by Group",color="workers_change",
                color_continuous_scale="RdYlGn",color_continuous_midpoint=0,
                labels={"workers_change":"Change","group":""})
            fig.update_layout(height=280,plot_bgcolor="white",paper_bgcolor="white",
                coloraxis_showscale=False,xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),
                title_font_size=14,margin=dict(t=40,b=10,l=10,r=10))
        else:
            wcc=wc if wc in filt.columns else "workers_base"
            if wcc in filt.columns and len(filt):
                sd=filt.groupby([sk,bp])[wcc].sum().reset_index()
                sd["group"]=sd[sk]+" / "+sd[bp]
                fig=px.pie(sd,values=wcc,names="group",title=f"Skill x Birthplace — {sy}",
                    color_discrete_sequence=px.colors.qualitative.Set2)
            else: fig=px.pie(title="No data")
            fig.update_layout(height=280,title_font_size=14,margin=dict(t=40,b=10),legend=dict(font_size=11))
        st.plotly_chart(fig,key=f"pc_{ss}_{sy}_{sec}_{ssk}_{sbp}")
    if hs:
        st.markdown("---")
        st.markdown("#### Scenario Comparison — 2024")
        comp=[]
        for scen in [s for s in df["scenario"].unique() if s!="baseline"]:
            sf=df[df["scenario"]==scen]
            if sec!="All sectors": sf=sf[sf["gtap_code"]==sec]
            if ssk!="All": sf=sf[sf[sk]==ssk]
            if sbp!="All": sf=sf[sf[bp]==sbp]
            if "workers_change" in sf.columns:
                comp.append({"scenario":SMETA.get(scen,{}).get("label",scen),
                    "workers_change":int(sf["workers_change"].fillna(0).sum()),
                    "color":SMETA.get(scen,{}).get("color","#888")})
        if comp:
            cdf=pd.DataFrame(comp).sort_values("workers_change")
            fig=px.bar(cdf,x="workers_change",y="scenario",orientation="h",
                title="Employment Change by Scenario",color="scenario",
                color_discrete_map={r["scenario"]:r["color"] for r in comp},
                labels={"workers_change":"Net Change","scenario":""})
            fig.add_vline(x=0,line_dash="dash",line_color="#666")
            fig.update_layout(height=250,showlegend=False,plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(tickformat=",d",gridcolor="#f0f0f0"),margin=dict(t=40,b=10,l=10,r=10),title_font_size=14)
            st.plotly_chart(fig,key=f"cmp_{sec}_{ssk}_{sbp}")

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown('<div class="ss">',unsafe_allow_html=True)
    st.markdown("**API Keys**")
    ak=""
    try: ak=st.secrets["ANTHROPIC_API_KEY"]; st.success("API key configured")
    except:
        ak=os.environ.get("ANTHROPIC_API_KEY","")
        if ak: st.success("API key configured")
        else: ak=st.text_input("Anthropic API Key",type="password")
    ukey=st.text_input("USDA API Key (optional)",type="password",value=os.environ.get("USDA_API_KEY",""))
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="ss">',unsafe_allow_html=True)
    st.markdown("**Data**")
    src=st.radio("Source",["Auto (built-in)","Upload file","Enter path"],label_visibility="collapsed")
    if src=="Auto (built-in)":
        if st.session_state.df is None:
            if st.button("Load Data"): st.session_state.df=load_parquet(); st.rerun()
        else: st.success(f"Loaded: {len(st.session_state.df):,} rows")
    elif src=="Upload file":
        up=st.file_uploader("File",type=["csv","parquet"],label_visibility="collapsed")
        if up: st.session_state.df=load_file(up,up.name); st.rerun()
    else:
        fp=st.text_input("Path",value=PARQUET_LOCAL,label_visibility="collapsed")
        if st.button("Load"):
            try: st.session_state.df=load_file(fp,fp); st.rerun()
            except Exception as e: st.error(f"Error: {e}")
    st.markdown('</div>',unsafe_allow_html=True)
    if st.session_state.df is not None:
        ds=st.session_state.df
        st.markdown('<div class="ss">',unsafe_allow_html=True)
        st.markdown("**Summary**")
        try:
            w=int(ds[(ds["scenario"]=="baseline")&(ds["year"]==2024)]["workers_base"].sum()) if all(c in ds.columns for c in ["scenario","year","workers_base"]) else 0
            st.metric("Baseline 2024",f"{w:,}")
            if "gtap_code" in ds.columns: st.metric("GTAP Sectors",ds["gtap_code"].nunique())
            if "county_fips" in ds.columns: st.metric("Counties",ds["county_fips"].nunique())
            if has_sims(ds): st.metric("Scenarios",ds["scenario"].nunique()-1)
        except: st.write(f"Rows: {len(ds):,}")
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown("---")
    for ex in ["What data is available?","Which sectors have most foreign-born workers?",
        "Show a map of v_f workers in 2022","Compare employment change across scenarios",
        "Which counties are most affected by JPM sim03?","USMCA long run on agriculture"]:
        if st.button(ex,key=f"ex_{ex[:15]}"): st.session_state["prefill"]=ex
    st.markdown("---")
    if st.button("Clear conversation"): st.session_state.messages=[]; st.session_state.figures={}; st.rerun()

st.markdown('<div class="mh">🌾 GTAP Labor Data Explorer</div>',unsafe_allow_html=True)
st.markdown('<div class="sh">Interactive dashboard and conversational analysis for 65 GTAP sectors across U.S. counties.</div>',unsafe_allow_html=True)

if st.session_state.df is None: st.info("Click **Load Data** in the sidebar."); st.stop()
df=st.session_state.df
tab1,tab2=st.tabs(["📊 Interactive Dashboard","💬 Ask Claude"])
with tab1: render_dashboard(df)
with tab2:
    bp2=bpc(df); latest=int(df["year"].max()) if "year" in df.columns else 2024
    dl=df[(df["scenario"]=="baseline")&(df["year"]==latest)] if "scenario" in df.columns else df
    tot=dl["workers_base"].sum() if "workers_base" in dl.columns else 0
    fb2=dl[dl[bp2].str.contains("Foreign",na=False)]["workers_base"].sum() if "workers_base" in dl.columns else 0
    sk2=dl[dl["skill_level"]=="Skilled"]["workers_base"].sum() if "workers_base" in dl.columns else 0
    c1,c2,c3,c4,c5=st.columns(5)
    for c,v,l in[(c1,f"{int(tot):,}",f"Baseline {latest}"),(c2,f"{fb2/tot*100:.1f}%" if tot else "—","Foreign Born"),
        (c3,f"{sk2/tot*100:.1f}%" if tot else "—","Skilled"),
        (c4,str(df["gtap_code"].nunique()) if "gtap_code" in df.columns else "—","GTAP Sectors"),
        (c5,f"{df['county_fips'].nunique():,}" if "county_fips" in df.columns else "—","Counties")]:
        with c: st.markdown(mc(v,l),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    for i,msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if isinstance(msg.get("content"),str): st.markdown(msg["content"])
            for fid in msg.get("figures",[]):
                if fid in st.session_state.figures:
                    st.plotly_chart(st.session_state.figures[fid],key=f"h_{fid}_{i}")
    pf=st.session_state.pop("prefill",""); ui=st.chat_input("Ask Claude about the GTAP labor data...")
    if pf and not ui: ui=pf
    if ui:
        if not ak: st.error("Enter API key in sidebar."); st.stop()
        with st.chat_message("user"): st.markdown(ui)
        st.session_state.messages.append({"role":"user","content":ui})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    rt,figs=run_agent(ui,df,ak,ukey)
                    if rt: st.markdown(rt)
                    fids=[]
                    for fid,fig in figs:
                        st.plotly_chart(fig,key=f"new_{fid}"); st.session_state.figures[fid]=fig; fids.append(fid)
                    st.session_state.messages.append({"role":"assistant","content":rt or "(See chart)","figures":fids})
                except anthropic.AuthenticationError: st.error("Invalid API key.")
                except Exception as e: st.error(f"Error: {e}")
