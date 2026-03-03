"""Tesla HR Intelligence - Streamlit App

Professional Tesla-inspired theme: clean white/black, red accents.
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

st.set_page_config(
    page_title="Tesla HR Intelligence",
    page_icon="https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
WHITE = "#FFFFFF"
OFF_WH = "#f4f4f4"
LIGHT = "#e8e8e8"
MID = "#cccccc"
DARK = "#393c41"
BLACK = "#000000"
RED = "#E82127"
RED_D = "#b81920"
RED_DIM = "rgba(232,33,39,0.06)"
CODE_BG = "#0d1117"
SQL_COL = "#8b949e"
PY_COL = "#79c0ff"
LOGO_URL = "https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png"

CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {WHITE} !important;
    color: {DARK} !important;
}}

[data-testid="stSidebar"] {{
    background-color: {OFF_WH} !important;
    border-right: 1px solid {LIGHT};
}}

[data-testid="stHeader"] {{
    background-color: {WHITE} !important;
    border-bottom: 1px solid {LIGHT};
}}

body, .stMarkdown, .stText, .stCode {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Helvetica Neue", Arial, sans-serif;
    color: {DARK};
    line-height: 1.55;
}}

h1, h2, h3, h4 {{
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {BLACK};
}}

.section-header {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {DARK};
    margin-top: 1.25rem;
    margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid {LIGHT};
}}

.response-card {{
    border-radius: 4px;
    padding: 0.75rem 1rem;
    background: {OFF_WH};
    border-left: 3px solid {LIGHT};
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}}

.response-card.has-error {{
    background: #fef8f8;
    border-left-color: {RED};
}}

.user-question {{
    margin: 0.6rem 0 0.3rem 0;
    padding: 0.6rem 0.9rem;
    border-radius: 4px;
    background: {OFF_WH};
    font-size: 0.9rem;
    border-left: 3px solid {MID};
}}

.agent-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    margin-top: 0.75rem;
    margin-bottom: 0.3rem;
}}

.error-notice {{
    margin-top: 0.25rem;
    padding: 0.4rem 0.7rem;
    border-radius: 4px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    font-size: 0.82rem;
    color: #991b1b;
}}

.stButton button[kind="primary"],
.stButton button:hover {{
    background-color: {RED} !important;
    border-color: {RED_D} !important;
    color: {WHITE} !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}}

.stButton button:active {{
    background-color: {RED_D} !important;
}}

[data-testid="stChatInputTextArea"] textarea {{
    background-color: {OFF_WH};
    border: 1px solid {LIGHT};
    border-radius: 4px;
    font-size: 0.9rem;
}}

.stCode, .stMarkdown pre code {{
    background-color: {CODE_BG} !important;
    color: {SQL_COL} !important;
    border-radius: 4px;
    font-size: 0.78rem;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {LIGHT};
    border-radius: 4px;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,
    font=dict(color=DARK, family="Helvetica Neue, Arial, sans-serif", size=12),
    title_font=dict(size=13, color=BLACK, family="Helvetica Neue, Arial, sans-serif"),
    margin=dict(t=48, b=28, l=28, r=28),
    xaxis=dict(
        gridcolor=LIGHT,
        tickfont=dict(color="#666", size=11),
        linecolor=MID,
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor=LIGHT,
        tickfont=dict(color="#666", size=11),
        linecolor=MID,
        showgrid=True,
    ),
    colorway=[RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070", "#b39ddb"],
)

PIE_COLORS = [
    RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070",
    "#b39ddb", "#ffcc80", "#a8d8a8", "#ff8a80", "#b81920",
]

# ── Load agent (cached per session) ──────────────────────────────────────────


@st.cache_resource
def load_agent():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    api_key = (
        api_key
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    return DataAgent(api_key=api_key)


# ── Chart renderer ────────────────────────────────────────────────────────────


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
            fig = px.line(
                df, x=x, y=y, title=title,
                color_discrete_sequence=[RED], markers=True,
            )
            fig.update_traces(
                marker_size=7, line_width=2.5,
                marker=dict(color=RED, line=dict(color=WHITE, width=1)),
            )
        elif ct == "pie":
            fig = go.Figure(
                go.Pie(
                    labels=df[x], values=df[y], hole=0.38,
                    marker=dict(colors=PIE_COLORS, line=dict(color=WHITE, width=2)),
                    textfont=dict(size=12, color=WHITE),
                )
            )
            fig.update_layout(title=dict(text=title))
        elif ct == "funnel":
            fig = go.Figure(
                go.Funnel(
                    y=df[x], x=df[y],
                    textinfo="value+percent initial",
                    marker=dict(color=[RED, "#c44", "#a33", "#822", "#611", "#400"]),
                    connector=dict(line=dict(color=LIGHT, width=1)),
                    textfont=dict(color=WHITE, size=12),
                )
            )
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
            <img src="{LOGO_URL}" width="22">
            <span style="font-weight:600;font-size:1rem;letter-spacing:0.04em;">
                HR Intelligence
            </span>
        </div>
        <div style="font-size:0.78rem;color:#888;letter-spacing:0.02em;">
            Data Analytics Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Could not start agent: {e}")
        st.stop()

    st.markdown(
        '<div class="section-header">Upload Data</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.8rem;color:#777;margin-bottom:0.3rem;">
            Upload CSV or Excel (.xlsx). The agent auto-detects all columns, dates, and joins.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded:
        files = uploaded if isinstance(uploaded, list) else [uploaded]
        for f in files:
            suffix = Path(f.name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            with st.spinner(f"Loading {f.name}..."):
                if hasattr(agent, "upload_file"):
                    r = agent.upload_file(Path(f.name).stem, tmp_path)
                elif hasattr(agent, "upload_csv"):
                    r = agent.upload_csv(Path(f.name).stem, tmp_path)
                else:
                    st.error("DataAgent missing upload method. Check agent_hr.py.")
                    os.unlink(tmp_path)
                    continue

            os.unlink(tmp_path)

            if "error" in r:
                st.error(f"{f.name}: {r['error']}")
            else:
                st.success(f"{f.name}: loaded {r['rows_loaded']:,} rows")
                for t in r.get("tables", []):
                    st.caption(f"Table: {t['table']} ({t['rows_loaded']:,} rows)")
                for t in r.get("tables", []):
                    for hint in t.get("join_hints", []):
                        st.caption(hint)

    if agent.has_data():
        st.markdown(
            '<div class="section-header">Sample Questions</div>',
            unsafe_allow_html=True,
        )
        examples = [
            "Show me the top 10 rows",
            "How many records are there?",
            "Show count by stage as a bar chart",
            "What is the distribution by department?",
            "Show the funnel breakdown by stage",
            "What are the unique values in each column?",
            "Show monthly trend as a line chart",
            "What is the conversion rate by stage?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.prefill = ex
                st.rerun()

    schema = agent.get_schema_summary()
    if schema:
        st.markdown(
            '<div class="section-header">Loaded Tables</div>',
            unsafe_allow_html=True,
        )
        for t in schema:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.82rem;margin-bottom:0.15rem;">
                        <span><code>{t["name"]}</code></span>
                        <span style="color:#888;">{t["rows"]:,} rows</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("x", key=f"del_{t['name']}", help=f"Remove {t['name']}"):
                    agent.remove_table(t["name"])
                    st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:0.75rem;
                margin-bottom:0.5rem;padding-bottom:0.5rem;
                border-bottom:1px solid {LIGHT};">
        <img src="{LOGO_URL}" width="36">
        <div>
            <div style="font-size:1.3rem;font-weight:600;letter-spacing:0.04em;">
                HR Intelligence
            </div>
            <div style="font-size:0.82rem;color:#888;">
                Ask questions in plain English &mdash; get SQL, Python code, and charts
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty / upload prompt ─────────────────────────────────────────────────────

if not agent.has_data():
    st.markdown(
        f"""
        <div style="text-align:center;margin-top:3.5rem;font-size:0.92rem;color:#888;">
            <div style="font-size:1.3rem;font-weight:500;color:{DARK};margin-bottom:0.4rem;">
                Upload your data to get started
            </div>
            Use the <b>Upload Data</b> panel in the sidebar.<br>
            Supports CSV and Excel. The agent auto-detects all columns, types, and relationships.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.messages:
    tables = agent.get_schema_summary()
    if tables:
        t = tables[0]
        st.markdown(
            f"""
            <div style="margin-top:2rem;font-size:0.92rem;color:#888;">
                <div style="font-size:1.1rem;font-weight:500;color:{DARK};
                            margin-bottom:0.25rem;">
                    Ready to analyze
                </div>
                <code>{t['name']}</code> loaded with <b>{t['rows']:,}</b> rows.
                Try a question like
                <code>Show count by stage</code>,
                <code>What is the conversion funnel?</code>, or
                <code>Show monthly trend as a line chart</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Render conversation ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-question">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="agent-label">Tesla HR Intelligence</div>',
            unsafe_allow_html=True,
        )

        if msg.get("narrative"):
            has_error = bool(msg.get("error")) or (
                not msg.get("rows") and not msg.get("error")
            )
            cls = "response-card has-error" if has_error else "response-card"
            st.markdown(
                f'<div class="{cls}">{msg["narrative"]}</div>',
                unsafe_allow_html=True,
            )

        c_sql, c_code = st.columns(2)
        with c_sql:
            if msg.get("sql"):
                with st.expander("View SQL"):
                    st.markdown(
                        f"""<pre style="background:{CODE_BG};color:{SQL_COL};
                                    padding:0.75rem;border-radius:4px;
                                    font-size:0.78rem;white-space:pre-wrap;
                                    overflow-x:auto;">{msg["sql"]}</pre>""",
                        unsafe_allow_html=True,
                    )
        with c_code:
            if msg.get("code"):
                with st.expander("View Python Code"):
                    st.markdown(
                        f"""<pre style="background:{CODE_BG};color:{PY_COL};
                                    padding:0.75rem;border-radius:4px;
                                    font-size:0.78rem;white-space:pre-wrap;
                                    overflow-x:auto;">{msg["code"]}</pre>""",
                        unsafe_allow_html=True,
                    )

        if msg.get("error"):
            st.markdown(
                f'<div class="error-notice">{msg["error"]}</div>',
                unsafe_allow_html=True,
            )

        charts = msg.get("charts", [])
        mcols = msg.get("money_cols", [])
        if charts and msg.get("rows"):
            if len(charts) == 1:
                render_chart(msg["rows"], charts[0], mcols)
            elif len(charts) == 2:
                cc1, cc2 = st.columns(2)
                with cc1:
                    render_chart(msg["rows"], charts[0], mcols)
                with cc2:
                    render_chart(msg["rows"], charts[1], mcols)
            else:
                cc1, cc2 = st.columns(2)
                with cc1:
                    render_chart(msg["rows"], charts[0], mcols)
                with cc2:
                    render_chart(msg["rows"], charts[1], mcols)
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

# ── Chat input ────────────────────────────────────────────────────────────────

pf = ""
if st.session_state.prefill:
    pf = st.session_state.prefill
    st.session_state.prefill = ""

question = st.chat_input(
    "Ask anything about your data..."
    if agent.has_data()
    else "Upload data first, then ask questions here..."
)

if not question and pf:
    question = pf

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Analysing..."):
        result = agent.ask(question)

    st.session_state.messages.append(
        {
            "role": "agent",
            "content": question,
            "sql": result.get("sql"),
            "code": result.get("code"),
            "rows": result.get("rows", []),
            "money_cols": result.get("money_columns", []),
            "narrative": result.get("narrative", ""),
            "charts": result.get("charts", []),
            "error": result.get("error"),
        }
    )
    st.rerun()
