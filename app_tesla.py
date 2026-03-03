"""Tesla HR Intelligence - Streamlit App

Tesla.com-inspired theme: clean white/black, red accents
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

# ── Tesla.com palette ─────────────────────────────────────────────────────────
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
/* ── Base layout ──────────────────────────────────────────── */
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

/* ── Typography ───────────────────────────────────────────── */
body, .stMarkdown, .stText, .stCode {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Helvetica Neue", Arial, sans-serif;
    color: {DARK};
    font-size: 0.9rem;
}}

h1, h2, h3, h4 {{
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: {BLACK};
}}

/* ── Page header ──────────────────────────────────────────── */
.app-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding-bottom: 0.85rem;
    margin-bottom: 1.1rem;
    border-bottom: 2px solid {RED};
}}

.app-header-title {{
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {BLACK};
}}

.app-header-sub {{
    font-size: 0.76rem;
    color: #999;
    letter-spacing: 0.03em;
    margin-top: 0.15rem;
}}

/* ── Sidebar labels ───────────────────────────────────────── */
.sidebar-section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #999;
    margin-top: 1.25rem;
    margin-bottom: 0.4rem;
    padding-bottom: 0.2rem;
    border-bottom: 1px solid {LIGHT};
}}

/* ── Chat messages ────────────────────────────────────────── */
.user-message {{
    padding: 0.55rem 0.85rem;
    border-radius: 4px;
    background: {OFF_WH};
    border: 1px solid {LIGHT};
    font-size: 0.9rem;
    margin: 0.6rem 0 0.15rem 0;
    line-height: 1.55;
}}

.agent-label {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {RED};
    margin-top: 0.9rem;
    margin-bottom: 0.3rem;
}}

/* ── Narrative card ───────────────────────────────────────── */
.narrative-card {{
    padding: 0.7rem 0.9rem;
    border-radius: 4px;
    background: {WHITE};
    border: 1px solid {LIGHT};
    border-left: 3px solid {DARK};
    font-size: 0.88rem;
    line-height: 1.65;
    margin-bottom: 0.4rem;
    color: {DARK};
}}

.narrative-card.no-result {{
    border-left-color: {MID};
    color: #777;
    font-style: italic;
}}

/* ── Error banner ─────────────────────────────────────────── */
.error-banner {{
    padding: 0.45rem 0.8rem;
    border-radius: 4px;
    background: #fff8f8;
    border: 1px solid #f0caca;
    border-left: 3px solid {RED};
    font-size: 0.82rem;
    color: #8b0000;
    margin-top: 0.2rem;
    margin-bottom: 0.3rem;
    line-height: 1.5;
}}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 3px !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.03em !important;
}}

.stButton > button:hover {{
    background-color: {RED} !important;
    border-color: {RED_D} !important;
    color: {WHITE} !important;
}}

/* ── Chat input ───────────────────────────────────────────── */
[data-testid="stChatInputTextArea"] textarea {{
    background-color: {OFF_WH};
    border-radius: 4px;
    font-size: 0.9rem;
}}

/* ── Code blocks ──────────────────────────────────────────── */
.stCode, .stMarkdown pre code {{
    background-color: {CODE_BG} !important;
    color: {SQL_COL} !important;
    border-radius: 4px;
    font-size: 0.78rem;
}}

/* ── Expander headers ─────────────────────────────────────── */
[data-testid="stExpander"] summary {{
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #666 !important;
}}

/* ── Empty-state ──────────────────────────────────────────── */
.empty-state {{
    text-align: center;
    margin-top: 3.5rem;
    padding: 2rem 2.5rem;
    border: 1px solid {LIGHT};
    border-radius: 6px;
    background: {OFF_WH};
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
}}

.empty-state-title {{
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {DARK};
    margin-bottom: 0.5rem;
}}

.empty-state-body {{
    font-size: 0.83rem;
    color: #888;
    line-height: 1.65;
}}

/* ── Turn separator ───────────────────────────────────────── */
.turn-divider {{
    border: none;
    border-top: 1px solid {LIGHT};
    margin: 0.5rem 0;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,
    font=dict(color=DARK, family="system-ui", size=12),
    title_font=dict(size=13, color=BLACK, family="system-ui"),
    margin=dict(t=44, b=28, l=28, r=28),
    xaxis=dict(
        gridcolor=LIGHT,
        tickfont=dict(color="#888", size=11),
        linecolor=MID,
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor=LIGHT,
        tickfont=dict(color="#888", size=11),
        linecolor=MID,
        showgrid=True,
    ),
    colorway=[RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070", "#b39ddb"],
)

PIE_COLORS = [
    RED, "#393c41", "#7fb3d3", "#82c596", "#f0a070",
    "#b39ddb", "#ffcc80", "#a8d8a8", "#ff8a80", "#b81920",
]

# ── Load agent (cached per session) ───────────────────────────────────────────


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
                marker_size=6, line_width=2.5,
                marker=dict(color=RED, line=dict(color=WHITE, width=1.5)),
            )
        elif ct == "pie":
            fig = go.Figure(
                go.Pie(
                    labels=df[x], values=df[y], hole=0.40,
                    marker=dict(colors=PIE_COLORS, line=dict(color=WHITE, width=2)),
                    textfont=dict(size=11, color=WHITE),
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
                    textfont=dict(color=WHITE, size=11),
                )
            )
            fig.update_layout(title=dict(text=title))
        elif ct == "scatter":
            fig = px.scatter(
                df, x=x, y=y, title=title, color_discrete_sequence=[RED]
            )
            fig.update_traces(marker_size=8, marker_opacity=0.75)
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
            <span style="font-weight:700;font-size:0.95rem;letter-spacing:0.08em;
                         text-transform:uppercase;">HR Intelligence</span>
        </div>
        <div style="font-size:0.78rem;color:#999;letter-spacing:0.03em;">
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
        '<div class="sidebar-section-label">Upload Data</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.8rem;color:#888;margin-bottom:0.4rem;">'
        f'CSV or Excel (.xlsx). The agent auto-detects all columns, dates, and joins.'
        f'</div>',
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
                    st.error("DataAgent has no upload method. Check agent_hr.py.")
                    os.unlink(tmp_path)
                    continue

            os.unlink(tmp_path)

            if "error" in r:
                st.error(f"{f.name}: {r['error']}")
            else:
                st.success(f"{f.name}: {r['rows_loaded']:,} rows loaded")
                for t in r.get("tables", []):
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#666;margin-top:0.1rem;">'
                        f'Table: <code>{t["table"]}</code> &nbsp;({t["rows_loaded"]:,} rows)'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                for t in r.get("tables", []):
                    for hint in t.get("join_hints", []):
                        st.markdown(
                            f'<div style="font-size:0.75rem;color:#aaa;margin-top:0.1rem;">'
                            f'{hint}</div>',
                            unsafe_allow_html=True,
                        )

    # Sample questions
    if agent.has_data():
        st.markdown(
            '<div class="sidebar-section-label">Sample Questions</div>',
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
            "Show top 10 job positions by count",
            "Distribution of job levels as a pie chart",
            "How many unique applicants per year?",
            "Show stage breakdown as a percentage",
            "How many female applicants are there?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.prefill = ex
                st.rerun()

    # Loaded tables
    schema = agent.get_schema_summary()
    if schema:
        st.markdown(
            '<div class="sidebar-section-label">Loaded Tables</div>',
            unsafe_allow_html=True,
        )
        for t in schema:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f'<div style="font-size:0.82rem;margin-bottom:0.1rem;">'
                    f'<code>{t["name"]}</code>'
                    f'<span style="color:#aaa;font-size:0.75rem;margin-left:0.4rem;">'
                    f'{t["rows"]:,} rows</span></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("x", key=f"del_{t['name']}", help=f"Remove {t['name']}"):
                    agent.remove_table(t["name"])
                    st.rerun()

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True):
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
    <div class="app-header">
        <img src="{LOGO_URL}" width="32">
        <div>
            <div class="app-header-title">HR Intelligence</div>
            <div class="app-header-sub">
                Natural language queries &mdash; SQL, Python code, and charts
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty / upload prompt ─────────────────────────────────────────────────────

if not agent.has_data():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">No Data Loaded</div>
            <div class="empty-state-body">
                Upload a CSV or Excel file using the <strong>Upload Data</strong>
                panel in the sidebar.<br><br>
                The agent automatically detects columns, data types, dates,
                and table relationships.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.messages:
    tables = agent.get_schema_summary()
    if tables:
        t = tables[0]
        col_count = len(agent.tables_info.get(t["name"], {}).get("columns", []))
        st.markdown(
            f"""
            <div class="empty-state" style="border-left:3px solid {RED};text-align:left;">
                <div class="empty-state-title">Ready</div>
                <div class="empty-state-body">
                    <code>{t['name']}</code> loaded &mdash; {t['rows']:,} rows,
                    {col_count} columns.<br><br>
                    Try: <em>Show count by stage</em> &nbsp;or&nbsp;
                    <em>What is the conversion funnel?</em> &nbsp;or&nbsp;
                    <em>Show monthly trend as a line chart</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Render conversation ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="agent-label">HR Intelligence</div>',
            unsafe_allow_html=True,
        )

        if msg.get("narrative"):
            has_results = bool(msg.get("rows"))
            has_error = bool(msg.get("error"))
            extra_class = "" if (has_results or has_error) else " no-result"
            st.markdown(
                f'<div class="narrative-card{extra_class}">{msg["narrative"]}</div>',
                unsafe_allow_html=True,
            )

        # SQL + Code side by side
        c_sql, c_code = st.columns(2)
        with c_sql:
            if msg.get("sql"):
                with st.expander("SQL Query"):
                    st.markdown(
                        f'<pre style="background:{CODE_BG};color:{SQL_COL};'
                        f'padding:0.75rem;border-radius:4px;'
                        f'font-size:0.78rem;white-space:pre-wrap;margin:0;">'
                        f'{msg["sql"]}</pre>',
                        unsafe_allow_html=True,
                    )
        with c_code:
            if msg.get("code"):
                with st.expander("Python Equivalent"):
                    st.markdown(
                        f'<pre style="background:{CODE_BG};color:{PY_COL};'
                        f'padding:0.75rem;border-radius:4px;'
                        f'font-size:0.78rem;white-space:pre-wrap;margin:0;">'
                        f'{msg["code"]}</pre>',
                        unsafe_allow_html=True,
                    )

        if msg.get("error"):
            st.markdown(
                f'<div class="error-banner">{msg["error"]}</div>',
                unsafe_allow_html=True,
            )

        # Charts
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

        st.markdown('<hr class="turn-divider">', unsafe_allow_html=True)

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
