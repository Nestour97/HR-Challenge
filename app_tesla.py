"""Tesla HR Intelligence — Streamlit App (Tesla black+red theme)
Run: streamlit run app_tesla.py
"""
import os, tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from agent_hr import HRAgent, detect_chart_intents, pick_charts

CORE_TABLES = {"hr_applications"}
st.set_page_config(page_title="Tesla HR Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Tesla Palette
BLACK="#000000"; BLACK2="#0a0a0a"; BLACK4="#1c1c1c"
RED="#E82127"; RED2="#b81920"; RED_DIM="rgba(232,33,39,0.12)"
WHITE="#FFFFFF"; GREY1="#e8e8e8"; GREY2="#999999"; GREY3="#444444"

CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@300;400&display=swap');
*{{box-sizing:border-box}} html,body,[class*="css"]{{font-family:'Inter',sans-serif;font-size:14px;color:{WHITE}}}
.stApp{{background:{BLACK};color:{WHITE}}}
section[data-testid="stSidebar"]{{background:{BLACK2}!important;border-right:1px solid {BLACK4}!important}}
section[data-testid="stSidebar"]>div{{padding-top:0!important}}
.tesla-brand{{background:{BLACK};border-bottom:2px solid {RED};padding:22px 20px 16px;margin:-1rem -1rem 20px;position:relative}}
.tesla-eyebrow{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.35em;text-transform:uppercase;color:{RED};margin-bottom:8px}}
.tesla-wordmark{{font-family:'Rajdhani',sans-serif;font-size:30px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:{WHITE};line-height:1}}
.tesla-wordmark span{{color:{RED}}} .tesla-sub{{font-size:11px;font-weight:300;color:{GREY2};margin-top:6px}}
.t-section{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:.35em;text-transform:uppercase;color:{GREY3};margin:20px 0 8px;padding-bottom:5px;border-bottom:1px solid {BLACK4}}}
.stButton>button{{background:transparent!important;border:1px solid {BLACK4}!important;border-radius:0!important;color:{GREY1}!important;font-family:'Inter',sans-serif!important;font-size:12px!important;font-weight:300!important;text-align:left!important;padding:8px 12px!important;transition:all .15s!important;width:100%!important}}
.stButton>button:hover{{border-color:{RED}!important;color:{RED}!important;background:{RED_DIM}!important;padding-left:16px!important}}
.schema-row{{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid {BLACK4}}}
.schema-name{{font-family:'JetBrains Mono',monospace;font-size:11px;color:{GREY2}}}
.schema-uploaded{{color:{RED};font-family:'JetBrains Mono',monospace;font-size:11px}}
.schema-count{{font-family:'JetBrains Mono',monospace;font-size:10px;background:{BLACK4};color:{RED};padding:1px 8px;border-left:2px solid {RED}}}
div[data-testid="stFileUploader"]{{border:1px dashed {GREY3}!important;border-radius:0!important;background:{BLACK2}!important}}
.main-header{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid {RED};padding-bottom:16px;margin-bottom:32px}}
.main-title{{font-family:'Rajdhani',sans-serif;font-size:44px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{WHITE};line-height:1}}
.main-title span{{color:{RED}}} .main-hint{{font-family:'JetBrains Mono',monospace;font-size:10px;color:{GREY3};letter-spacing:.12em;text-transform:uppercase}}
.msg-user{{background:{BLACK2};border-left:3px solid {RED};padding:12px 18px;margin:8px 0 8px 15%;color:{WHITE};font-size:14px;line-height:1.6}}
.agent-label{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:.3em;text-transform:uppercase;color:{RED};margin:20px 0 10px;display:flex;align-items:center;gap:10px}}
.agent-label::before{{content:'';width:20px;height:1px;background:{RED}}}
.narrative-box{{background:{BLACK2};border-left:3px solid {GREY3};padding:14px 18px;color:{GREY1};font-size:14px;line-height:1.8;font-weight:300;margin-bottom:12px}}
.narrative-box.suggestion{{border-left-color:{RED};background:{RED_DIM};color:{WHITE}}}
.sql-code{{background:#050505;border:1px solid {BLACK4};border-left:3px solid {GREY3};padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#777;white-space:pre-wrap;line-height:1.7}}
.code-box{{background:#040813;border:1px solid {BLACK4};border-left:3px solid {RED};padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#8ec8f6;white-space:pre-wrap;line-height:1.7}}
.error-box{{background:rgba(232,33,39,0.08);border-left:3px solid {RED};padding:12px 16px;color:#ff6b6b;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.6}}
.chart-badge{{display:inline-block;background:{RED_DIM};border:1px solid {RED};color:{RED};font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;padding:2px 10px;margin-bottom:12px}}
[data-testid="stDataFrame"]{{border:1px solid {BLACK4}!important;border-top:2px solid {RED}!important}}
[data-testid="stDataFrame"] th{{background:{BLACK2}!important;color:{RED}!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important;letter-spacing:.1em!important;text-transform:uppercase!important}}
.empty-state{{text-align:center;padding:80px 20px;color:{GREY3}}}
.empty-icon{{font-size:64px;line-height:1;margin-bottom:16px;opacity:.2}}
.empty-title{{font-family:'Rajdhani',sans-serif;font-size:36px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{WHITE};margin-bottom:12px}}
.empty-sub{{font-size:13px;line-height:2.1;font-weight:300;max-width:440px;margin:0 auto;color:{GREY2}}}
.empty-sub em{{color:{RED};font-style:normal}}
hr{{border:none!important;border-top:1px solid {BLACK4}!important;margin:10px 0!important}}
.stSpinner>div{{border-top-color:{RED}!important}}
[data-testid="stChatInput"]>div{{background:{BLACK2}!important;border:1px solid {BLACK4}!important;border-bottom:2px solid {RED}!important;border-radius:0!important}}
[data-testid="stChatInput"] textarea{{color:{WHITE}!important;font-family:'Inter',sans-serif!important;background:transparent!important}}
details summary{{color:{GREY2}!important;font-size:12px!important;font-family:'JetBrains Mono',monospace!important;cursor:pointer!important}}
details summary:hover{{color:{RED}!important}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

CHART_LAYOUT = dict(plot_bgcolor=BLACK2, paper_bgcolor=BLACK, font=dict(color=GREY1, family="Inter", size=12),
    title_font=dict(size=14, color=WHITE, family="Rajdhani"), margin=dict(t=48, b=28, l=28, r=28),
    xaxis=dict(gridcolor=BLACK4, tickfont=dict(color=GREY2, size=11), linecolor=BLACK4),
    yaxis=dict(gridcolor=BLACK4, tickfont=dict(color=GREY2, size=11), linecolor=BLACK4),
    colorway=[RED, "#7fb3d3", "#82c596", "#f0a070", "#b39ddb", "#ffcc80"])
PIE_COLORS = [RED,"#7fb3d3","#82c596","#f0a070","#b39ddb","#ffcc80","#a8d8a8","#ff8a80","#b81920","#555"]

@st.cache_resource
def load_agent():
    try: api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except: api_key = None
    api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return HRAgent(str(Path(__file__).parent / "Data_Challenge.csv"), api_key=api_key)

def render_chart(rows, cfg, money_cols):
    ct = cfg.get("chart","none")
    if ct=="none" or not rows: return
    x, y, title = cfg.get("x"), cfg.get("y"), cfg.get("title","")
    df = pd.DataFrame(rows)
    if not x or x not in df.columns or not y or y not in df.columns: return
    df[y] = pd.to_numeric(df[y], errors="coerce"); df = df.dropna(subset=[y])
    if df.empty: return
    try:
        if ct=="bar":
            fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[RED])
            fig.update_traces(marker_line_width=0)
        elif ct=="line":
            fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=[RED], markers=True)
            fig.update_traces(marker_size=7, line_width=2.5, marker=dict(color=RED, line=dict(color=WHITE,width=1)))
        elif ct=="pie":
            fig = go.Figure(go.Pie(labels=df[x], values=df[y], hole=0.35,
                marker=dict(colors=PIE_COLORS, line=dict(color=BLACK,width=2)), textfont=dict(size=12,color=WHITE)))
            fig.update_layout(title=dict(text=title))
        elif ct=="funnel":
            fig = go.Figure(go.Funnel(y=df[x], x=df[y], textinfo="value+percent initial",
                marker=dict(color=[RED,"#c44","#a33","#822","#611","#400"]),
                connector=dict(line=dict(color=BLACK4,width=1)), textfont=dict(color=WHITE,size=12)))
            fig.update_layout(title=dict(text=title))
        elif ct=="scatter":
            fig = px.scatter(df, x=x, y=y, title=title, color_discrete_sequence=[RED])
            fig.update_traces(marker_size=10, marker_opacity=0.8)
        else: return
        fig.update_layout(**CHART_LAYOUT)
        if ct not in ("pie","funnel") and y in money_cols:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    except: pass

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""<div class="tesla-brand"><div class="tesla-eyebrow">People Analytics</div>
      <div class="tesla-wordmark">TESLA <span>HR</span></div>
      <div class="tesla-sub">Intelligence Platform · Talent Acquisition</div></div>""", unsafe_allow_html=True)
    try: agent = load_agent()
    except Exception as e: st.error(f"Could not start: {e}"); st.stop()

    st.markdown('<div class="t-section">Sample Questions</div>', unsafe_allow_html=True)
    for ex in ["Show the recruitment funnel by stage","How many applications per department?",
               "What is the offer-to-hire conversion rate?","Show monthly hiring trends as a line chart",
               "Which job levels have the most applicants?","What % of applicants reach Onsite?",
               "Applications by quarter as a bar chart","Average time to hire in days?",
               "Which department has the highest hire rate?","Show stage distribution as a pie chart",
               "Top 10 most applied-to job positions","Unique applicants per year"]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.prefill = ex; st.rerun()

    st.markdown('<div class="t-section">Upload Data</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:{GREY3};font-weight:300;margin-bottom:8px;line-height:1.6">Upload any CSV to query alongside the core HR data.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read()); tmp_path = tmp.name
        r = agent.upload_csv(uploaded.name.replace(".csv",""), tmp_path); os.unlink(tmp_path)
        if "error" in r: st.error(r["error"])
        else:
            st.success(f"✓ Loaded `{r['table']}` — {r['rows_loaded']} rows")
            for hint in r.get("join_hints",[]): st.markdown(f'<div style="font-size:10px;color:{RED2};margin-top:2px">🔗 {hint}</div>', unsafe_allow_html=True)

    st.markdown('<div class="t-section">Database Tables</div>', unsafe_allow_html=True)
    for t in agent.get_schema_summary():
        nc = "schema-name" if t["name"] in CORE_TABLES else "schema-uploaded"
        if t["name"] in CORE_TABLES:
            st.markdown(f'<div class="schema-row"><span class="{nc}">{t["name"]}</span><span class="schema-count">{t["rows"]}</span></div>', unsafe_allow_html=True)
        else:
            c1,c2 = st.columns([4,1])
            with c1: st.markdown(f'<div class="schema-row"><span class="{nc}">{t["name"]}</span><span class="schema-count">{t["rows"]}</span></div>', unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"del_{t['name']}"): agent.remove_table(t["name"]); st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✕  Clear conversation", use_container_width=True):
        st.session_state.messages=[]; agent.clear_history(); st.rerun()

# ── Main ──
if "messages" not in st.session_state: st.session_state.messages=[]
if "prefill" not in st.session_state: st.session_state.prefill=""

st.markdown(f"""<div class="main-header"><div class="main-title">HR <span>Intelligence</span></div>
  <div class="main-hint">⚡ query · analyse · visualise</div></div>""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""<div class="empty-state"><div class="empty-icon">⚡</div>
      <div class="empty-title">Ask About Your People</div>
      <div class="empty-sub">Plain English · Typos handled · Follow-ups work<br><br>
        Try <em>"Show the recruitment funnel"</em><br>
        or <em>"What % reach the Onsite stage?"</em><br>
        or <em>"Monthly hiring trend as a line chart"</em><br><br>
        Every answer includes SQL + Python code you can copy.</div></div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"]=="user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="agent-label">Tesla HR Intelligence</div>', unsafe_allow_html=True)
        if msg.get("narrative"):
            is_sug = not msg.get("rows") and not msg.get("error")
            cls = "narrative-box suggestion" if is_sug else "narrative-box"
            icon = "&#128161; " if is_sug else ""
            st.markdown(f'<div class="{cls}">{icon}{msg["narrative"]}</div>', unsafe_allow_html=True)
        cs, cc = st.columns(2)
        with cs:
            if msg.get("sql"):
                with st.expander("⬡  View SQL"):
                    st.markdown(f'<div class="sql-code">{msg["sql"]}</div>', unsafe_allow_html=True)
        with cc:
            if msg.get("code"):
                with st.expander("⬡  View Python Code"):
                    st.markdown(f'<div class="code-box">{msg["code"]}</div>', unsafe_allow_html=True)
        if msg.get("error"):
            st.markdown(f'<div class="error-box">⚠ {msg["error"]}</div>', unsafe_allow_html=True)
        charts, mcols = msg.get("charts",[]), msg.get("money_cols",[])
        if charts and msg.get("rows"):
            if len(charts)>1: st.markdown(f'<div class="chart-badge">⚡ {len(charts)} visualisations generated</div>', unsafe_allow_html=True)
            if len(charts)==1: render_chart(msg["rows"],charts[0],mcols)
            elif len(charts)==2:
                c1,c2=st.columns(2)
                with c1: render_chart(msg["rows"],charts[0],mcols)
                with c2: render_chart(msg["rows"],charts[1],mcols)
            else:
                c1,c2=st.columns(2)
                with c1: render_chart(msg["rows"],charts[0],mcols)
                with c2: render_chart(msg["rows"],charts[1],mcols)
                render_chart(msg["rows"],charts[2],mcols)
        if msg.get("rows"):
            df=pd.DataFrame(msg["rows"])
            for col in mcols:
                if col in df.columns: df[col]=df[col].apply(lambda v: f"${float(v):,.2f}" if v not in (None,"") else v)
            st.dataframe(df,use_container_width=True,hide_index=True)
        elif not msg.get("error") and not msg.get("narrative"): st.info("No results returned.")

pf=""
if st.session_state.prefill: pf=st.session_state.prefill; st.session_state.prefill=""
q=st.chat_input("Ask about stages, departments, trends, conversion rates...")
if not q and pf: q=pf
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.spinner(""): result=agent.ask(q)
    st.session_state.messages.append({"role":"agent","content":q,"sql":result.get("sql"),"code":result.get("code"),
        "rows":result.get("rows",[]),"money_cols":result.get("money_columns",[]),"narrative":result.get("narrative",""),
        "charts":result.get("charts",[]),"error":result.get("error")})
    st.rerun()
