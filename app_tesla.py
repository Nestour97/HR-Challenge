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

# ── Tesla.com palette (white-dominant like their actual site) ─────────────────
WHITE = "#FFFFFF"
OFF_WH = "#f4f4f4"
LIGHT = "#e8e8e8"
MID = "#cccccc"
DARK = "#393c41"  # Tesla dark grey (body text)
BLACK = "#000000"
RED = "#E82127"  # Tesla signature red
RED_D = "#b81920"
RED_DIM = "rgba(232,33,39,0.08)"
CODE_BG = "#0d1117"
SQL_COL = "#8b949e"
PY_COL = "#79c0ff"
LOGO_URL = "https://pngimg.com/uploads/tesla_logo/tesla_logo_PNG12.png"

# You can drop in any extra CSS here if you like
CSS = """
<style>
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Plotly theme (light, Tesla-aligned) ───────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,
    font=dict(color=DARK, family="Inter", size=12),
    title_font=dict(size=14, color=BLACK, family="Inter"),
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
    RED,
    "#393c41",
    "#7fb3d3",
    "#82c596",
    "#f0a070",
    "#b39ddb",
    "#ffcc80",
    "#a8d8a8",
    "#ff8a80",
    "#b81920",
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
                df,
                x=x,
                y=y,
                title=title,
                color_discrete_sequence=[RED],
                markers=True,
            )
            fig.update_traces(
                marker_size=7,
                line_width=2.5,
                marker=dict(color=RED, line=dict(color=WHITE, width=1)),
            )
        elif ct == "pie":
            fig = go.Figure(
                go.Pie(
                    labels=df[x],
                    values=df[y],
                    hole=0.38,
                    marker=dict(
                        colors=PIE_COLORS, line=dict(color=WHITE, width=2)
                    ),
                    textfont=dict(size=12, color=WHITE),
                )
            )
            fig.update_layout(title=dict(text=title))
        elif ct == "funnel":
            fig = go.Figure(
                go.Funnel(
                    y=df[x],
                    x=df[y],
                    textinfo="value+percent initial",
                    marker=dict(
                        color=[RED, "#c44", "#a33", "#822", "#611", "#400"]
                    ),
                    connector=dict(line=dict(color=LIGHT, width=1)),
                    textfont=dict(color=WHITE, size=12),
                )
            )
            fig.update_layout(title=dict(text=title))
        elif ct == "scatter":
            fig = px.scatter(
                df, x=x, y=y, title=title, color_discrete_sequence=[RED]
            )
            fig.update_traces(marker_size=9, marker_opacity=0.75)
        else:
            return

        fig.update_layout(**CHART_LAYOUT)

        if ct not in ("pie", "funnel") and y in money_cols:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")

        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fail silently for chart errors; the table + SQL are still useful
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
            <img src="{LOGO_URL}" width="24">
            <span style="font-weight:600;font-size:1.05rem;">HR Intelligence</span>
        </div>
        <div style="font-size:0.9rem;color:#555;">Data Analytics Platform</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Could not start agent: {e}")
        st.stop()

    # Upload section - FIRST so user uploads before asking
    st.markdown(
        """
        <h4 style="margin-top:1.5rem;margin-bottom:0.25rem;">Upload Data</h4>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.85rem;color:#555;margin-bottom:0.25rem;">
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
                r = agent.upload_file(Path(f.name).stem, tmp_path)

            os.unlink(tmp_path)

            if "error" in r:
                st.error(f"{f.name}: {r['error']}")
            else:
                st.success(f"{f.name}: loaded {r['rows_loaded']:,} rows")
                for t in r.get("tables", []):
                    st.markdown(
                        f"""
                        <div style="font-size:0.8rem;color:#444;margin-bottom:0.1rem;">
                            ↳ table <code>{t["table"]}</code> ({t["rows_loaded"]:,} rows)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                for t in r.get("tables", []):
                    for hint in t.get("join_hints", []):
                        st.markdown(
                            f"""
                            <div style="font-size:0.75rem;color:#444;margin:0.1rem 0;
                                        padding:0.15rem 0.4rem;border-radius:999px;
                                        background:{RED_DIM};">
                                🔗 {hint}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Sample questions (only shown if data is loaded)
    if agent.has_data():
        st.markdown(
            """
            <h4 style="margin-top:1.5rem;margin-bottom:0.25rem;">Sample Questions</h4>
            """,
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
            """
            <h4 style="margin-top:1.5rem;margin-bottom:0.25rem;">Loaded Tables</h4>
            """,
            unsafe_allow_html=True,
        )
        for t in schema:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.85rem;margin-bottom:0.15rem;">
                        <span><code>{t["name"]}</code></span>
                        <span style="color:#555;">{t["rows"]:,} rows</span>
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
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
        <img src="{LOGO_URL}" width="40">
        <div>
            <div style="font-size:1.4rem;font-weight:600;">HR Intelligence</div>
            <div style="font-size:0.9rem;color:#555;">
                Ask questions in plain English — get SQL, Python code, and charts
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
        <div style="text-align:center;margin-top:3rem;font-size:0.95rem;color:#555;">
            <div style="font-size:2.2rem;margin-bottom:0.5rem;">📊</div>
            <b>Upload your data to get started</b><br>
            Use the <b>Upload Data</b> panel in the sidebar.<br>
            Works with CSV or Excel — the agent auto-detects all columns, types, and relationships.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.messages:
    # Data loaded but no questions yet
    tables = agent.get_schema_summary()
    if tables:
        t = tables[0]
        st.markdown(
            f"""
            <div style="margin-top:2rem;font-size:0.95rem;color:#555;">
                <div style="font-size:2rem;margin-bottom:0.25rem;">⚡</div>
                <b>Ready — ask anything</b><br>
                <code>{t['name']}</code> loaded with {t['rows']:,} rows.<br>
                Try: <code>Show count by stage</code> or
                <code>What is the conversion funnel?</code> or
                <code>Show monthly trend as a line chart</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Render conversation ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="margin:0.5rem 0;padding:0.6rem 0.75rem;
                        border-radius:0.5rem;background:{OFF_WH};">
                {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="font-size:0.8rem;color:#888;margin-top:0.75rem;margin-bottom:0.25rem;">
                Tesla HR Intelligence
            </div>
            """,
            unsafe_allow_html=True,
        )

        if msg.get("narrative"):
            is_sug = not msg.get("rows") and not msg.get("error")
            bg = OFF_WH if not is_sug else RED_DIM
            icon = "💡 " if is_sug else ""
            st.markdown(
                f"""
                <div style="margin-bottom:0.4rem;padding:0.6rem 0.75rem;
                            border-radius:0.5rem;background:{bg};font-size:0.9rem;">
                    {icon}{msg["narrative"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SQL + Code side by side
        c_sql, c_code = st.columns(2)
        with c_sql:
            if msg.get("sql"):
                with st.expander("View SQL"):
                    st.markdown(
                        f"""
                        <pre style="background:{CODE_BG};color:{SQL_COL};
                                    padding:0.75rem;border-radius:0.5rem;
                                    font-size:0.8rem;white-space:pre-wrap;">
{msg["sql"]}
                        </pre>
                        """,
                        unsafe_allow_html=True,
                    )
        with c_code:
            if msg.get("code"):
                with st.expander("View Python Code"):
                    st.markdown(
                        f"""
                        <pre style="background:{CODE_BG};color:{PY_COL};
                                    padding:0.75rem;border-radius:0.5rem;
                                    font-size:0.8rem;white-space:pre-wrap;">
{msg["code"]}
                        </pre>
                        """,
                        unsafe_allow_html=True,
                    )

        if msg.get("error"):
            st.markdown(
                f"""
                <div style="margin-top:0.25rem;padding:0.4rem 0.6rem;
                            border-radius:0.4rem;background:#fff4f4;
                            font-size:0.85rem;color:#b00020;">
                    ⚠ {msg["error"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Up to 3 charts
        charts = msg.get("charts", [])
        mcols = msg.get("money_cols", [])
        if charts and msg.get("rows"):
            if len(charts) > 1:
                st.markdown(
                    f"""
                    <div style="font-size:0.85rem;color:#555;margin:0.4rem 0;">
                        ⚡ {len(charts)} charts generated
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
