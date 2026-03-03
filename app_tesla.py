"""
Tesla HR Intelligence - Streamlit App
Tesla.com-inspired theme: clean white/black, red accents
Run: streamlit run app_tesla.py
"""
import os, tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from agent_hr import DataAgent, detect_chart_intents, pick_charts

st.set_page_config(
    page_title="Tesla HR Intelligence",
    page_icon="https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tesla.com palette (white-dominant like their actual site) ──────────────────
WHITE   = "#FFFFFF"
OFF_WH  = "#f4f4f4"
LIGHT   = "#e8e8e8"
MID     = "#cccccc"
DARK    = "#393c41"      # Tesla dark grey (body text)
BLACK   = "#000000"
RED     = "#E82127"      # Tesla signature red
RED_D   = "#b81920"
RED_DIM = "rgba(232,33,39,0.08)"
CODE_BG = "#0d1117"
SQL_COL = "#8b949e"
PY_COL  = "#79c0ff"

LOGO_URL = "https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: {DARK};
    background: {WHITE};
}}

/* ── App background ── */
.stApp {{ background: {WHITE}; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {BLACK} !important;
    border-right: none !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* ── Sidebar brand ── */
.t-brand {{
    padding: 28px 20px 20px;
    border-bottom: 1px solid #222;
    margin-bottom: 4px;
}}
.t-brand img {{
    width: 40px;
    display: block;
    margin-bottom: 12px;
    filter: invert(1);
}}
.t-brand-title {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {WHITE} !important;
}}
.t-brand-sub {{
    font-size: 11px;
    color: #888 !important;
    margin-top: 3px;
    letter-spacing: 0.05em;
    font-weight: 300;
}}

/* ── Sidebar section label ── */
.t-section {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #555 !important;
    margin: 20px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #222;
}}

/* ── Sidebar buttons ── */
.stButton > button {{
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 3px !important;
    color: #ccc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 300 !important;
    text-align: left !important;
    padding: 8px 12px !important;
    transition: all 0.15s !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    border-color: {RED} !important;
    color: {WHITE} !important;
    background: rgba(232,33,39,0.1) !important;
}}

/* ── Schema rows ── */
.schema-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1a1a1a;
}}
.schema-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #aaa !important;
}}
.schema-count {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {RED} !important;
    background: #1a1a1a;
    padding: 1px 8px;
    border-left: 2px solid {RED};
}}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {{
    border: 1px dashed #333 !important;
    border-radius: 4px !important;
    background: #111 !important;
}}

/* ── Main area ── */
.main-wrap {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }}

/* ── Page header ── */
.page-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 24px;
    border-bottom: 1px solid {LIGHT};
    margin-bottom: 36px;
}}
.page-header img {{
    width: 32px;
    height: auto;
}}
.page-header-text {{ }}
.page-title {{
    font-size: 22px;
    font-weight: 600;
    color: {BLACK};
    letter-spacing: -0.01em;
    line-height: 1;
}}
.page-sub {{
    font-size: 12px;
    color: #888;
    margin-top: 3px;
    font-weight: 300;
}}

/* ── Upload prompt card ── */
.upload-prompt {{
    background: {OFF_WH};
    border: 1px solid {LIGHT};
    border-radius: 6px;
    padding: 48px 32px;
    text-align: center;
    margin: 60px auto;
    max-width: 560px;
}}
.upload-prompt-icon {{
    font-size: 40px;
    margin-bottom: 16px;
    opacity: 0.4;
}}
.upload-prompt-title {{
    font-size: 18px;
    font-weight: 600;
    color: {BLACK};
    margin-bottom: 8px;
}}
.upload-prompt-sub {{
    font-size: 13px;
    color: #888;
    line-height: 1.7;
}}
.upload-prompt-sub em {{
    color: {RED};
    font-style: normal;
    font-weight: 500;
}}

/* ── User message ── */
.msg-user {{
    background: {OFF_WH};
    border-left: 3px solid {RED};
    border-radius: 0 4px 4px 0;
    padding: 12px 18px;
    margin: 16px 0 16px 10%;
    color: {DARK};
    font-size: 14px;
    line-height: 1.6;
}}

/* ── Agent label ── */
.agent-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {RED};
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.agent-label::before {{
    content: '';
    display: inline-block;
    width: 20px;
    height: 1px;
    background: {RED};
}}

/* ── Narrative box ── */
.narrative-box {{
    background: {WHITE};
    border: 1px solid {LIGHT};
    border-left: 3px solid {DARK};
    border-radius: 0 4px 4px 0;
    padding: 14px 18px;
    color: {DARK};
    font-size: 14px;
    line-height: 1.8;
    font-weight: 400;
    margin-bottom: 14px;
}}
.narrative-box.suggestion {{
    border-left-color: {RED};
    background: {RED_DIM};
}}

/* ── SQL / Code boxes ── */
.sql-code {{
    background: {CODE_BG};
    border: 1px solid #30363d;
    border-left: 3px solid #444;
    border-radius: 0 4px 4px 0;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {SQL_COL};
    white-space: pre-wrap;
    line-height: 1.7;
}}
.code-box {{
    background: {CODE_BG};
    border: 1px solid #30363d;
    border-left: 3px solid {RED};
    border-radius: 0 4px 4px 0;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {PY_COL};
    white-space: pre-wrap;
    line-height: 1.7;
}}

/* ── Error box ── */
.error-box {{
    background: rgba(232,33,39,0.06);
    border: 1px solid rgba(232,33,39,0.2);
    border-left: 3px solid {RED};
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    color: {RED_D};
    font-size: 13px;
    line-height: 1.6;
}}

/* ── Chart badge ── */
.chart-badge {{
    display: inline-block;
    background: {RED_DIM};
    border: 1px solid rgba(232,33,39,0.2);
    color: {RED};
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 12px;
}}

/* ── Data table ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {LIGHT} !important;
    border-top: 2px solid {RED} !important;
    border-radius: 0 !important;
}}
[data-testid="stDataFrame"] th {{
    background: {OFF_WH} !important;
    color: {DARK} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid {LIGHT} !important;
}}

/* ── Misc ── */
hr {{ border: none !important; border-top: 1px solid {LIGHT} !important; margin: 16px 0 !important; }}
.stSpinner > div {{ border-top-color: {RED} !important; }}
[data-testid="stChatInput"] > div {{
    background: #171a20 !important;
    border: 1px solid #2d3139 !important;
    border-bottom: 2px solid {RED} !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.18) !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: #555 !important; }}
[data-testid="stChatInput"] textarea {{
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}}
details summary {{
    color: #666 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    cursor: pointer !important;
}}
details summary:hover {{ color: {RED} !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Plotly theme (light, Tesla-aligned) ───────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    font=dict(color=DARK, family="Inter", size=12),
    title_font=dict(size=14, color=BLACK, family="Inter"),
    margin=dict(t=48, b=28, l=28, r=28),
    xaxis=dict(gridcolor=LIGHT, tickfont=dict(color="#666", size=11),
               linecolor=MID, showgrid=True),
    yaxis=dict(gridcolor=LIGHT, tickfont=dict(color="#666", size=11),
               linecolor=MID, showgrid=True),
    colorway=[RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070", "#b39ddb"],
)
PIE_COLORS = [RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070",
              "#b39ddb", "#ffcc80", "#a8d8a8", "#ff8a80", "#b81920"]


# ── Load agent (cached per session) ───────────────────────────────────────────
@st.cache_resource
def load_agent():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return DataAgent(api_key=api_key)


# ── Chart renderer ─────────────────────────────────────────────────────────────
def render_chart(rows, cfg, money_cols):
    ct = cfg.get("chart", "none")
    if ct == "none" or not rows:
        return
    x, y, title = cfg.get("x"), cfg.get("y"), cfg.get("title", "")
    df = pd.DataFrame(rows)
    if not x or x not in df.columns or not y or y not in df.columns:
        return
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[y])
    if df.empty:
        return
    try:
        if ct == "bar":
            fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[RED])
            fig.update_traces(marker_line_width=0)
        elif ct == "line":
            fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=[RED], markers=True)
            fig.update_traces(marker_size=7, line_width=2.5,
                              marker=dict(color=RED, line=dict(color=WHITE, width=1)))
        elif ct == "pie":
            fig = go.Figure(go.Pie(
                labels=df[x], values=df[y], hole=0.38,
                marker=dict(colors=PIE_COLORS, line=dict(color=WHITE, width=2)),
                textfont=dict(size=12, color=WHITE),
            ))
            fig.update_layout(title=dict(text=title))
        elif ct == "funnel":
            fig = go.Figure(go.Funnel(
                y=df[x], x=df[y],
                textinfo="value+percent initial",
                marker=dict(color=[RED, "#c44", "#a33", "#822", "#611", "#400"]),
                connector=dict(line=dict(color=LIGHT, width=1)),
                textfont=dict(color=WHITE, size=12),
            ))
            fig.update_layout(title=dict(text=title))
        elif ct == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title, color_discrete_sequence=[RED])
            fig.update_traces(marker_size=9, marker_opacity=0.75)
        else:
            return
        fig.update_layout(**CHART_LAYOUT)
        if ct not in ("pie", "funnel") and y in money_cols:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="t-brand">
      <img src="{LOGO_URL}" alt="Tesla" />
      <div class="t-brand-title">HR Intelligence</div>
      <div class="t-brand-sub">Data Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Could not start agent: {e}")
        st.stop()

    # Upload section - FIRST so user uploads before asking
    st.markdown('<div class="t-section">Upload CSV Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#666;font-weight:300;margin-bottom:8px;line-height:1.6">'
        'Upload any CSV. The agent auto-detects all columns and types.</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed",
                                 accept_multiple_files=False)
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Loading..."):
            r = agent.upload_csv(uploaded.name.replace(".csv", ""), tmp_path)
        os.unlink(tmp_path)
        if "error" in r:
            st.error(r["error"])
        else:
            st.success(f"Loaded {r['rows_loaded']:,} rows into `{r['table']}`")
            for hint in r.get("join_hints", []):
                st.markdown(
                    f'<div style="font-size:10px;color:{RED};margin-top:3px">&#128279; {hint}</div>',
                    unsafe_allow_html=True
                )

    # Sample questions (only shown if data is loaded)
    if agent.has_data():
        st.markdown('<div class="t-section">Sample Questions</div>', unsafe_allow_html=True)
        examples = [
            "Show me the top 10 rows",
            "How many records are there?",
            "Show count by stage as a bar chart",
            "What is the distribution by department?",
            "Show the funnel breakdown by stage",
            "What are the unique values in each column?",
            "Show monthly trend as a line chart",
            "What is the conversion rate by stage?",
            "Show top 10 job positions by count",
            "Distribution of job levels as a pie chart",
            "How many unique applicants per year?",
            "Show stage breakdown as a percentage",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.prefill = ex
                st.rerun()

    # Loaded tables
    schema = agent.get_schema_summary()
    if schema:
        st.markdown('<div class="t-section">Loaded Tables</div>', unsafe_allow_html=True)
        for t in schema:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f'<div class="schema-row">'
                    f'<span class="schema-name">{t["name"]}</span>'
                    f'<span class="schema-count">{t["rows"]:,}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c2:
                if st.button("x", key=f"del_{t['name']}", help=f"Remove {t['name']}"):
                    agent.remove_table(t["name"])
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        agent.clear_history()
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <img src="{LOGO_URL}" alt="Tesla" />
  <div class="page-header-text">
    <div class="page-title">HR Intelligence</div>
    <div class="page-sub">Ask questions in plain English &mdash; get SQL, Python code, and charts</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Empty / upload prompt ──────────────────────────────────────────────────────
if not agent.has_data():
    st.markdown("""
    <div class="upload-prompt">
      <div class="upload-prompt-icon">&#128202;</div>
      <div class="upload-prompt-title">Upload your CSV to get started</div>
      <div class="upload-prompt-sub">
        Use the <em>Upload CSV Data</em> panel in the sidebar.<br>
        The agent auto-detects all columns, types, and relationships.<br><br>
        Works with <em>any</em> CSV &mdash; HR, sales, finance, or anything else.
      </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages:
    # Data loaded but no questions yet
    tables = agent.get_schema_summary()
    if tables:
        t = tables[0]
        st.markdown(f"""
        <div class="upload-prompt">
          <div class="upload-prompt-icon">&#9889;</div>
          <div class="upload-prompt-title">Ready &mdash; ask anything</div>
          <div class="upload-prompt-sub">
            <em>{t['name']}</em> loaded with <em>{t['rows']:,} rows</em><br><br>
            Try: <em>"Show count by stage"</em><br>
            or <em>"What is the conversion funnel?"</em><br>
            or <em>"Show monthly trend as a line chart"</em>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Render conversation ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="agent-label">Tesla HR Intelligence</div>', unsafe_allow_html=True)

        if msg.get("narrative"):
            is_sug = not msg.get("rows") and not msg.get("error")
            cls = "narrative-box suggestion" if is_sug else "narrative-box"
            icon = "&#128161; " if is_sug else ""
            st.markdown(f'<div class="{cls}">{icon}{msg["narrative"]}</div>', unsafe_allow_html=True)

        # SQL + Code side by side
        c_sql, c_code = st.columns(2)
        with c_sql:
            if msg.get("sql"):
                with st.expander("View SQL"):
                    st.markdown(f'<div class="sql-code">{msg["sql"]}</div>', unsafe_allow_html=True)
        with c_code:
            if msg.get("code"):
                with st.expander("View Python Code"):
                    st.markdown(f'<div class="code-box">{msg["code"]}</div>', unsafe_allow_html=True)

        if msg.get("error"):
            st.markdown(f'<div class="error-box">&#9888; {msg["error"]}</div>', unsafe_allow_html=True)

        # Up to 3 charts
        charts = msg.get("charts", [])
        mcols  = msg.get("money_cols", [])
        if charts and msg.get("rows"):
            if len(charts) > 1:
                st.markdown(
                    f'<div class="chart-badge">&#9889; {len(charts)} charts generated</div>',
                    unsafe_allow_html=True
                )
            if len(charts) == 1:
                render_chart(msg["rows"], charts[0], mcols)
            elif len(charts) == 2:
                cc1, cc2 = st.columns(2)
                with cc1: render_chart(msg["rows"], charts[0], mcols)
                with cc2: render_chart(msg["rows"], charts[1], mcols)
            else:
                cc1, cc2 = st.columns(2)
                with cc1: render_chart(msg["rows"], charts[0], mcols)
                with cc2: render_chart(msg["rows"], charts[1], mcols)
                render_chart(msg["rows"], charts[2], mcols)

        if msg.get("rows"):
            df = pd.DataFrame(msg["rows"])
            for col in mcols:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: f"${float(v):,.2f}" if v not in (None, "") else v
                    )
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif not msg.get("error") and not msg.get("narrative"):
            st.info("No results returned.")


# ── Chat input ─────────────────────────────────────────────────────────────────
pf = ""
if st.session_state.prefill:
    pf = st.session_state.prefill
    st.session_state.prefill = ""

question = st.chat_input(
    "Ask anything about your data..." if agent.has_data()
    else "Upload a CSV first, then ask questions here..."
)
if not question and pf:
    question = pf

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Analysing..."):
        result = agent.ask(question)
    st.session_state.messages.append({
        "role":      "agent",
        "content":   question,
        "sql":       result.get("sql"),
        "code":      result.get("code"),
        "rows":      result.get("rows", []),
        "money_cols":result.get("money_columns", []),
        "narrative": result.get("narrative", ""),
        "charts":    result.get("charts", []),
        "error":     result.get("error"),
    })
    st.rerun()
