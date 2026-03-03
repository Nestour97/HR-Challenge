"""Tesla HR Intelligence — Premium Data Analytics Platform
Run: streamlit run app_tesla.py
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent_hr import DataAgent, detect_chart_intents, pick_charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Intelligence · Tesla",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
WHITE   = "#FFFFFF"
OFF_WH  = "#F7F7F7"
LIGHT   = "#EBEBEB"
MID     = "#CCCCCC"
SMOKE   = "#999999"
DARK    = "#2B2B2B"
BLACK   = "#0A0A0A"
RED     = "#E82127"
RED_D   = "#B81920"
RED_L   = "#FCE8E9"
CODE_BG = "#0D1117"
SQL_COL = "#8B949E"
PY_COL  = "#79C0FF"
LOGO    = "https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: {WHITE} !important;
    color: {DARK} !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}}

[data-testid="stSidebar"] {{
    background: {OFF_WH} !important;
    border-right: 1px solid {LIGHT} !important;
}}
[data-testid="stSidebar"] > div {{ padding: 0 !important; }}

[data-testid="stHeader"] {{
    background: {WHITE} !important;
    border-bottom: 1px solid {LIGHT} !important;
    height: 0 !important;
}}

/* Hide Streamlit branding */
#MainMenu, footer, [data-testid="stToolbar"] {{ display: none !important; }}

/* Typography */
h1, h2, h3 {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    color: {BLACK};
    letter-spacing: -0.01em;
}}

/* Sidebar sections */
.sb-section {{
    padding: 0 1rem 0.5rem 1rem;
    border-bottom: 1px solid {LIGHT};
    margin-bottom: 0.25rem;
}}
.sb-section-title {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SMOKE};
    padding: 0.85rem 0 0.5rem 0;
}}

/* Metric cards */
.metric-row {{
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}}
.metric-card {{
    flex: 1;
    background: {WHITE};
    border: 1px solid {LIGHT};
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
}}
.metric-label {{
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SMOKE};
    margin-bottom: 0.15rem;
}}
.metric-value {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {BLACK};
    line-height: 1.2;
}}

/* Page header */
.page-header {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 1.1rem 1.5rem;
    border-bottom: 1px solid {LIGHT};
    background: {WHITE};
    position: sticky;
    top: 0;
    z-index: 100;
}}
.page-title {{
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: {BLACK};
    line-height: 1.2;
}}
.page-subtitle {{
    font-size: 0.78rem;
    color: {SMOKE};
    margin-top: 0.1rem;
}}

/* Chat messages */
.chat-container {{ padding: 0 1.5rem; }}

.user-bubble {{
    margin: 1.25rem 0 0.35rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}}
.user-avatar {{
    width: 28px; height: 28px;
    background: {LIGHT};
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 600; color: {DARK};
    flex-shrink: 0; margin-top: 2px;
}}
.user-text {{
    background: {OFF_WH};
    border: 1px solid {LIGHT};
    border-radius: 0 8px 8px 8px;
    padding: 0.55rem 0.85rem;
    font-size: 0.88rem;
    color: {DARK};
    max-width: 100%;
    line-height: 1.5;
}}

.agent-label {{
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {RED};
    margin: 0.5rem 0 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.agent-label::before {{
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: {RED};
    border-radius: 50%;
}}

.narrative-card {{
    background: {WHITE};
    border: 1px solid {LIGHT};
    border-left: 3px solid {MID};
    border-radius: 0 6px 6px 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: {DARK};
    margin-bottom: 0.5rem;
}}
.narrative-card.error {{
    border-left-color: {RED};
    background: {RED_L};
}}

.error-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: {RED_L};
    border: 1px solid #FCA5A5;
    border-radius: 4px;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    color: #991B1B;
    font-family: 'DM Mono', monospace;
    margin-top: 0.35rem;
}}

/* Code blocks */
.code-block {{
    background: {CODE_BG};
    border-radius: 6px;
    padding: 0.85rem 1rem;
    font-family: 'DM Mono', 'Fira Code', monospace;
    font-size: 0.76rem;
    white-space: pre-wrap;
    overflow-x: auto;
    line-height: 1.6;
    margin: 0;
}}

/* Table styling */
[data-testid="stDataFrame"] {{
    border: 1px solid {LIGHT} !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}}

/* Button overrides */
.stButton > button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    border: 1px solid {LIGHT} !important;
    background: {WHITE} !important;
    color: {DARK} !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
    padding: 0.35rem 0.65rem !important;
}}
.stButton > button:hover {{
    border-color: {RED} !important;
    color: {RED} !important;
    background: {RED_L} !important;
}}
.stButton > button[kind="primary"] {{
    background: {RED} !important;
    border-color: {RED_D} !important;
    color: {WHITE} !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: {RED_D} !important;
}}

/* Chat input */
[data-testid="stChatInputTextArea"] textarea {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    background: {OFF_WH} !important;
    border: 1px solid {LIGHT} !important;
    border-radius: 8px !important;
    color: {DARK} !important;
}}

/* File uploader */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed {MID} !important;
    border-radius: 8px !important;
    background: {OFF_WH} !important;
}}

/* Schema table pill */
.schema-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: {WHITE};
    border: 1px solid {LIGHT};
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    color: {DARK};
    margin-bottom: 0.3rem;
    width: 100%;
    justify-content: space-between;
}}
.schema-pill code {{
    font-family: 'DM Mono', monospace;
    color: {RED};
    background: none;
    font-size: 0.75rem;
}}
.schema-rows {{
    color: {SMOKE};
    font-size: 0.72rem;
}}

/* Welcome screen */
.welcome-box {{
    text-align: center;
    padding: 3rem 2rem;
    color: {SMOKE};
}}
.welcome-title {{
    font-size: 1.25rem;
    font-weight: 500;
    color: {DARK};
    margin-bottom: 0.5rem;
}}
.welcome-hint {{
    font-size: 0.85rem;
    line-height: 1.7;
}}
.welcome-hint code {{
    background: {OFF_WH};
    border: 1px solid {LIGHT};
    border-radius: 3px;
    padding: 0.1rem 0.35rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: {DARK};
}}

/* Expander override */
[data-testid="stExpander"] {{
    border: 1px solid {LIGHT} !important;
    border-radius: 6px !important;
    background: {WHITE} !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: {SMOKE} !important;
    padding: 0.4rem 0.75rem !important;
}}

/* Divider */
.t-divider {{
    height: 1px;
    background: {LIGHT};
    margin: 0.75rem 0;
}}

/* Spinner */
[data-testid="stSpinner"] {{ color: {RED} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Chart palette ──────────────────────────────────────────────────────────────
PIE_COLORS = [RED, "#393c41", "#7FB3D3", "#82C596", "#F0A070",
              "#B39DDB", "#FFCC80", "#A8D8A8", "#FF8A80", RED_D]
CHART_LAYOUT = dict(
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    font=dict(color=DARK, family="DM Sans, sans-serif", size=12),
    title_font=dict(size=13, color=BLACK, family="DM Sans, sans-serif"),
    margin=dict(t=44, b=24, l=24, r=24),
    xaxis=dict(gridcolor=LIGHT, tickfont=dict(color=SMOKE, size=11), linecolor=MID, showgrid=True),
    yaxis=dict(gridcolor=LIGHT, tickfont=dict(color=SMOKE, size=11), linecolor=MID, showgrid=True),
    colorway=[RED, "#393c41", "#7FB3D3", "#82C596", "#F0A070", "#B39DDB"],
)


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
                              marker=dict(color=RED, line=dict(color=WHITE, width=1.5)))
        elif ct == "pie":
            fig = go.Figure(go.Pie(
                labels=df[x], values=df[y], hole=0.42,
                marker=dict(colors=PIE_COLORS, line=dict(color=WHITE, width=2)),
                textfont=dict(size=12, color=WHITE),
            ))
            fig.update_layout(title=dict(text=title))
        elif ct == "funnel":
            fig = go.Figure(go.Funnel(
                y=df[x], x=df[y],
                textinfo="value+percent initial",
                marker=dict(color=[RED, "#C44", "#A33", "#822", "#611"]),
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


# ── Load agent ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_agent():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return DataAgent(api_key=api_key)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="padding: 1.1rem 1rem 0.75rem 1rem; border-bottom: 1px solid {LIGHT};">
        <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.25rem;">
            <img src="{LOGO}" width="20" style="object-fit:contain;">
            <span style="font-family:'DM Sans',sans-serif;font-size:0.95rem;font-weight:600;
                         color:{BLACK};letter-spacing:-0.01em;">HR Intelligence</span>
        </div>
        <div style="font-size:0.72rem;color:{SMOKE};letter-spacing:0.01em;">
            Tesla People Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.stop()

    # Upload section
    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-section-title">Data Upload</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding: 0 1rem 0.75rem;">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.76rem;color:{SMOKE};margin-bottom:0.5rem;line-height:1.5;">'
            'Upload CSV or Excel files. Multiple files are auto-joined on shared columns.</div>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "", type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        files = uploaded if isinstance(uploaded, list) else [uploaded]
        for f in files:
            suffix = Path(f.name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            with st.spinner(f"Loading {f.name}..."):
                r = agent.upload_file(Path(f.name).stem, tmp_path)
            os.unlink(tmp_path)
            if "error" in r:
                st.error(f"{f.name}: {r['error']}")
            else:
                total = r.get("rows_loaded", 0)
                st.success(f"✓ {f.name}: {total:,} raw rows ingested")
                cleaning = r.get("cleaning_summary", {})
                if cleaning:
                    lines = "<br>".join(
                        f"<span style='margin-left:0.5rem'>└ <code style='background:none;color:#166534;font-family:monospace'>{k}</code>: {v}</span>"
                        for k, v in cleaning.items()
                    )
                    st.markdown(
                        f"""<div style='margin-top:0.4rem;padding:0.5rem 0.65rem;background:#f0faf4;
                            border:1px solid #86efac;border-radius:5px;font-size:0.73rem;color:#166534;line-height:1.6;'>
                            <strong>⚡ Auto-cleaned</strong><br>{lines}</div>""",
                        unsafe_allow_html=True
                    )

    # Loaded tables
    schema = agent.get_schema_summary()
    if schema:
        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-section-title">Loaded Tables</div>
        </div>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1rem 0.5rem;">', unsafe_allow_html=True)
            for t in schema:
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"""
                    <div class="schema-pill">
                        <code>{t['name']}</code>
                        <span class="schema-rows">{t['rows']:,} rows</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_{t['name']}", help=f"Remove {t['name']}"):
                        agent.remove_table(t["name"])
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Quick stats metrics
        if len(schema) >= 1:
            total_rows = sum(t["rows"] for t in schema)
            total_tables = len(schema)
            st.markdown(f"""
            <div style="padding: 0 1rem 0.75rem;">
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Tables</div>
                        <div class="metric-value">{total_tables}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Rows</div>
                        <div class="metric-value">{total_rows:,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sample questions
    if agent.has_data():
        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-section-title">Sample Questions</div>
        </div>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 0.75rem 0.75rem;">', unsafe_allow_html=True)
            examples = [
                "How many females applied to each job level?",
                "Show the hiring funnel by stage",
                "What % of applicants reach each stage?",
                "Gender split for each stage as bar chart",
                "How many total applications and hires?",
                "Average days to hire by job level",
                "Ethnicity breakdown as pie chart",
                "Monthly application trend as line chart",
                "Top departments by number of hires",
                "Compare male vs female hire rates",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.prefill = ex
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="padding: 0.5rem 1rem 1rem;">', unsafe_allow_html=True)
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <img src="{LOGO}" width="32" style="object-fit:contain;">
    <div>
        <div class="page-title">HR Intelligence</div>
        <div class="page-subtitle">
            Ask questions in plain English — get insights, SQL, Python code, and charts
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────────────────────────
if not agent.has_data():
    st.markdown(f"""
    <div class="welcome-box">
        <div style="font-size:2rem;margin-bottom:1rem;">📊</div>
        <div class="welcome-title">Upload your HR data to get started</div>
        <div class="welcome-hint">
            Use the <strong>Data Upload</strong> panel in the sidebar to load CSV or Excel files.<br>
            Try uploading <code>Tesla_HR_Task_-_Raw.csv</code> and
            <code>Tesla_HR_Task_-_Gender_Ethnicity.csv</code> together —<br>
            they'll be automatically joined on the shared <code>applicant_id</code> column.
        </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages:
    tables = agent.get_schema_summary()
    if tables:
        cols_str = " · ".join(f"`{t['name']}`" for t in tables)
        total = sum(t["rows"] for t in tables)
        st.markdown(f"""
        <div style="padding: 2rem 1.5rem 0.5rem;">
            <div style="font-size:1rem;font-weight:600;color:{BLACK};margin-bottom:0.35rem;">
                Ready to analyze
            </div>
            <div style="font-size:0.85rem;color:{SMOKE};line-height:1.65;">
                {len(tables)} table(s) loaded with <strong>{total:,}</strong> total rows.
                Select a sample question from the sidebar, or ask anything below —
                e.g. <em>"How many females applied to each job level?"</em> or
                <em>"Show the hiring funnel by stage."</em>
            </div>
        </div>
        <div class="t-divider" style="margin: 1rem 1.5rem;"></div>
        """, unsafe_allow_html=True)

# ── Conversation ───────────────────────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="user-avatar">HR</div>
            <div class="user-text">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="agent-label">Tesla HR Intelligence</div>', unsafe_allow_html=True)

        has_error = bool(msg.get("error"))
        cls = "narrative-card error" if has_error else "narrative-card"
        if msg.get("narrative"):
            st.markdown(f'<div class="{cls}">{msg["narrative"]}</div>', unsafe_allow_html=True)

        if msg.get("error"):
            st.markdown(
                f'<div class="error-badge">⚠ {msg["error"]}</div>',
                unsafe_allow_html=True
            )

        # SQL + Code expanders
        c1, c2 = st.columns(2)
        with c1:
            if msg.get("sql"):
                with st.expander("◦ View SQL Query"):
                    st.markdown(
                        f'<pre class="code-block" style="color:{SQL_COL};">{msg["sql"]}</pre>',
                        unsafe_allow_html=True
                    )
        with c2:
            if msg.get("code"):
                with st.expander("◦ View Python Code"):
                    st.markdown(
                        f'<pre class="code-block" style="color:{PY_COL};">{msg["code"]}</pre>',
                        unsafe_allow_html=True
                    )

        # Charts
        charts = msg.get("charts", [])
        mcols = msg.get("money_cols", [])
        rows = msg.get("rows", [])
        if charts and rows:
            if len(charts) == 1:
                render_chart(rows, charts[0], mcols)
            elif len(charts) == 2:
                cc1, cc2 = st.columns(2)
                with cc1: render_chart(rows, charts[0], mcols)
                with cc2: render_chart(rows, charts[1], mcols)
            else:
                cc1, cc2 = st.columns(2)
                with cc1: render_chart(rows, charts[0], mcols)
                with cc2: render_chart(rows, charts[1], mcols)
                render_chart(rows, charts[2], mcols)

        # Data table
        if rows:
            df = pd.DataFrame(rows)
            for col in mcols:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: f"${float(v):,.2f}" if v not in (None, "") else v
                    )
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif not msg.get("error") and not msg.get("narrative"):
            st.info("No results returned.")

        st.markdown('<div class="t-divider"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
pf = ""
if st.session_state.prefill:
    pf = st.session_state.prefill
    st.session_state.prefill = ""

placeholder = (
    "Ask anything about your data — e.g. 'How many females applied to each job level?'"
    if agent.has_data()
    else "Upload data first, then ask questions here..."
)
question = st.chat_input(placeholder)

if not question and pf:
    question = pf

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Analyzing..."):
        result = agent.ask(question)

    st.session_state.messages.append({
        "role": "agent",
        "content": question,
        "sql": result.get("sql"),
        "code": result.get("code"),
        "rows": result.get("rows", []),
        "money_cols": result.get("money_columns", []),
        "narrative": result.get("narrative", ""),
        "charts": result.get("charts", []),
        "error": result.get("error"),
    })
    st.rerun()
