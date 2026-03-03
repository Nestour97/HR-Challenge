"""
Tesla HR Insights Agent — Core Logic
Natural language → SQL + Python code + multi-chart output for HR analytics.
"""
from __future__ import annotations
import csv, io, os, re, sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

try:
    from groq import Groq as _Client
    _CLIENT_TYPE = "groq"
except ImportError:
    from openai import OpenAI as _Client
    _CLIENT_TYPE = "openai"

MODEL = "llama-3.3-70b-versatile" if _CLIENT_TYPE == "groq" else "gpt-4o"

BLOCKED_PATTERNS = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+VIEW|DELETE\s+FROM|UPDATE\s+\w+\s+SET|INSERT\s+INTO"
    r"|ALTER\s+TABLE|TRUNCATE\s+TABLE|CREATE\s+TABLE|CREATE\s+VIEW|REPLACE\s+INTO"
    r"|ATTACH\s+DATABASE|DETACH\s+DATABASE)\b",
    re.IGNORECASE | re.DOTALL,
)

BASE_SQL_PROMPT = """You are an expert SQL analyst specialising in HR and talent acquisition data.
Convert natural-language questions into correct, read-only SQLite SELECT queries.

CORE TABLE: hr_applications
  - date_applied        TEXT  (ISO date YYYY-MM-DD)
  - applicant_id        INTEGER
  - stage               TEXT  (Application, Phone Screen, Onsite, Offer, Hired, Rejected)
  - job_level           TEXT  (e.g. S1 (I), M3 ( - ), S4 (IV))
  - department_code     INTEGER
  - job_position_code   INTEGER
  - target_start_date   TEXT  (ISO date or NULL)
  - year_applied        INTEGER
  - month_applied       INTEGER
  - quarter_applied     TEXT  (Q1, Q2, Q3, Q4)

{extra_tables}

KEY HR CONCEPTS:
  - Recruitment funnel: Application -> Phone Screen -> Onsite -> Offer -> Hired
  - Conversion rate: ROUND(100.0 * COUNT(CASE WHEN stage='X' THEN 1 END) / COUNT(*), 1)
  - Time to hire: julianday(target_start_date) - julianday(date_applied)
  - Headcount: COUNT(DISTINCT applicant_id)
  - Hiring velocity: GROUP BY year_applied, month_applied

SQL RULES:
1. Primary table alias: h. Always: FROM hr_applications h
2. Stage fuzzy match: LOWER(h.stage) LIKE '%keyword%'
3. Output ONLY raw SQL — no markdown, no backticks, no explanation.
4. CTEs (WITH ...) are encouraged for multi-step queries.
5. For time trends: GROUP BY h.year_applied, h.month_applied ORDER BY h.year_applied, h.month_applied.
6. NULL-safe: use IS NOT NULL guards where appropriate.
7. Never use CREATE/DROP/UPDATE/INSERT.

FOLLOW-UP: resolve pronouns (them, same, that level) using conversation history.
"""

NARRATIVE_PROMPT = """You are a sharp HR analytics lead at Tesla.
Given a question and results, respond with a crisp insight in 2-4 sentences.
- Data found: lead with the most important number or trend, name specific values.
- Empty result: say filter wasn't matched, suggest existing values.
- Error: explain plainly, no SQL jargon.
- Tone: direct, precise, Tesla-grade. No fluff.
"""

CODE_PROMPT = """You are a Python data analyst. Given a question and the SQL query, write a
clean self-contained pandas + plotly code snippet that performs the same analysis.

The DataFrame `df` is already loaded with these columns:
  date_applied, applicant_id, stage, job_level, department_code,
  job_position_code, target_start_date, year_applied, month_applied, quarter_applied

Rules:
- DO NOT include pd.read_csv(). Assume `df` is already defined.
- Import pandas as pd and plotly.express as px at the top.
- Keep under 25 lines. Use clear variable names and brief comments.
- End with `result` (a DataFrame) and optionally `fig` (a plotly figure).
- Output ONLY raw Python code. No markdown, no backticks, no prose.
"""

CHART_KEYWORDS = {
    "funnel": ["funnel", "pipeline", "conversion", "drop off", "stage breakdown"],
    "trend":  ["trend", "over time", "monthly", "by month", "by year", "by quarter",
               "timeline", "time series", "velocity", "per month", "per year"],
    "bar":    ["bar chart", "bar graph", "column chart", "most", "top", "highest",
               "lowest", "ranking", "compare", "histogram"],
    "pie":    ["pie", "proportion", "share", "percentage", "distribution",
               "breakdown", "composition", "split"],
    "scatter":["scatter", "correlation", "vs ", "versus", "relationship"],
}


def detect_chart_intents(question: str) -> list[str]:
    q = question.lower()
    intents = [ct for ct, patterns in CHART_KEYWORDS.items() if any(p in q for p in patterns)]
    if not intents and any(w in q for w in ("chart", "graph", "visual", "plot", "show me")):
        intents.append("auto")
    return intents


def _is_numeric_col(rows, col):
    checked = 0
    for row in rows[:8]:
        v = str(row.get(col, "")).strip().replace(",", "").replace("$", "").replace("%", "")
        if not v:
            continue
        try:
            float(v); checked += 1
        except ValueError:
            return False
    return checked > 0


def _is_date_col(col):
    return any(k in col.lower() for k in ("date", "month", "year", "quarter", "period", "time"))


def pick_charts(rows, intents, question):
    if not rows:
        return []
    cols = list(rows[0].keys())
    numeric_cols = [c for c in cols if _is_numeric_col(rows, c)]
    text_cols = [c for c in cols if c not in numeric_cols]
    date_cols = [c for c in cols if _is_date_col(c)]
    if not numeric_cols:
        return []
    x = date_cols[0] if date_cols else (text_cols[0] if text_cols else cols[0])
    y = numeric_cols[0]
    y2 = numeric_cols[1] if len(numeric_cols) > 1 else None
    ty = y.replace("_", " ").title()
    tx = x.replace("_", " ").title()
    charts, used = [], set()

    def add(cfg):
        if cfg["chart"] not in used and len(charts) < 3:
            charts.append(cfg); used.add(cfg["chart"])

    for intent in intents:
        if intent == "funnel":
            add({"chart": "funnel", "x": x, "y": y, "title": f"Recruitment Funnel — {ty}"})
        elif intent == "trend" and date_cols:
            add({"chart": "line", "x": date_cols[0], "y": y, "title": f"{ty} Over Time"})
        elif intent == "bar":
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        elif intent == "pie" and len(rows) <= 14:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
        elif intent == "scatter" and y2:
            add({"chart": "scatter", "x": y, "y": y2, "title": f"{ty} vs {y2.replace('_',' ').title()}"})
        elif intent == "auto":
            if len(rows) <= 8:
                add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
            else:
                add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})

    if len(rows) >= 3 and len(charts) < 2:
        if "bar" not in used and text_cols:
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        if date_cols and "line" not in used:
            add({"chart": "line", "x": date_cols[0], "y": y, "title": f"{ty} Over Time"})
        if "pie" not in used and len(rows) <= 12 and text_cols:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Share"})

    return charts[:3]


def _is_money_col(col):
    return any(k in col.lower() for k in ("salary","pay","wage","compensation","bonus","amount","cost","budget","usd","total"))

def _is_safe_sql(sql):
    s = sql.strip().lstrip(";").strip()
    if not s.upper().startswith("SELECT") and not s.upper().startswith("WITH"):
        return False, "Query must start with SELECT or WITH"
    m = BLOCKED_PATTERNS.search(sql)
    if m:
        return False, f"Destructive keyword blocked: {m.group().strip()}"
    if ";" in re.sub(r"'[^']*'", "''", sql).rstrip(";"):
        return False, "Multiple statements not allowed"
    return True, ""

def _read_csv(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))

def _parse_date(raw):
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw

def _infer_logical_type(series):
    s = series.dropna()
    if s.empty: return "str"
    for check, label in [(pd.api.types.is_bool_dtype, "bool"),
                          (pd.api.types.is_integer_dtype, "int"),
                          (pd.api.types.is_float_dtype, "float"),
                          (pd.api.types.is_datetime64_any_dtype, "datetime")]:
        if check(s): return label
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return "float" if (num.dropna() % 1 != 0).mean() > 0.05 else "int"
    try:
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.8: return "datetime"
    except Exception:
        pass
    return "str"


def build_database(hr_csv_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""CREATE TABLE hr_applications (
        date_applied TEXT, applicant_id INTEGER, stage TEXT, job_level TEXT,
        department_code INTEGER, job_position_code INTEGER, target_start_date TEXT,
        year_applied INTEGER, month_applied INTEGER, quarter_applied TEXT
    )""")
    for r in _read_csv(hr_csv_path):
        date_str = _parse_date(r.get("Date Applied", ""))
        start_str = _parse_date(r.get("Target Start Date", ""))
        def si(v):
            try: return int(float(v or 0))
            except: return 0
        ya = ma = qa = None
        if date_str:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
                ya, ma, qa = d.year, d.month, f"Q{(d.month-1)//3+1}"
            except: pass
        cur.execute("INSERT INTO hr_applications VALUES (?,?,?,?,?,?,?,?,?,?)", (
            date_str, si(r.get("Applicant ID")), r.get("STAGE","").strip(),
            r.get("Job Level","").strip(), si(r.get("Department Code")),
            si(r.get("Job Position Code")), start_str, ya, ma, qa,
        ))
    conn.commit()
    return conn


def load_extra_csv(conn, table_name, file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}
    if df.empty:
        return {"error": "File is empty"}
    base = os.path.splitext(os.path.basename(table_name))[0]
    safe = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "uploaded_table"
    logical_types = {col: _infer_logical_type(df[col]) for col in df.columns}
    df.to_sql(safe, conn, if_exists="replace", index=False)
    # relationship detection
    all_tables = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"):
        all_tables[name] = [row[1] for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]
    relationships, join_hints = [], []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique == 0 or nunique > 50000: continue
        sample_list = [v.item() if hasattr(v, "item") else v for v in pd.unique(df[col].dropna())[:200]]
        for ot, oc in all_tables.items():
            if ot == safe or col not in oc: continue
            ph = ",".join(["?"]*len(sample_list))
            try:
                (matches,) = conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{ot}" WHERE "{col}" IN ({ph})', sample_list).fetchone()
                ratio = matches / float(len(sample_list) or 1)
                if ratio >= 0.6:
                    relationships.append({"from_table": safe, "from_col": col, "to_table": ot, "to_col": col, "overlap_ratio": ratio})
                    join_hints.append(f'"{safe}"."{col}" likely joins "{ot}"."{col}" (~{ratio:.0%} overlap)')
            except: pass
    return {
        "table": safe, "original_name": table_name, "columns": list(df.columns),
        "col_types": logical_types, "rows_loaded": int(len(df)),
        "schema_description": ", ".join(f"{c} ({logical_types[c]})" for c in df.columns),
        "join_hints": join_hints, "relationships": relationships,
        "sample": df.head(2).to_dict(orient="records"),
    }


def remove_table(conn, table_name):
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"'); conn.commit(); return True
    except: return False


def list_tables(conn):
    tables = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name").fetchall()
    result = []
    for t in tables:
        try: count = conn.execute(f'SELECT COUNT(*) FROM "{t["name"]}"').fetchone()[0]
        except: count = "?"
        result.append({"name": t["name"], "type": t["type"], "rows": count})
    return result


def get_distinct_values(conn, col):
    try:
        return [str(r[0]) for r in conn.execute(f"SELECT DISTINCT {col} FROM hr_applications WHERE {col} IS NOT NULL ORDER BY {col}").fetchall()]
    except: return []


def _call(client, system, user, max_tokens=400):
    resp = client.chat.completions.create(model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()


def _build_sql_prompt(uploaded_tables):
    if not uploaded_tables:
        return BASE_SQL_PROMPT.format(extra_tables="")
    lines = ["UPLOADED TABLES (use EXACTLY these names):"]
    for tname, info in uploaded_tables.items():
        lines += [f'  TABLE: "{tname}"', f'  COLUMNS: {info.get("schema_description","")}']
        for hint in info.get("join_hints", []):
            lines.append(f"  JOIN HINT: {hint}")
        sample = info.get("sample") or []
        if sample: lines.append(f"  SAMPLE: {sample[0]}")
        lines.append("")
    return BASE_SQL_PROMPT.format(extra_tables="\n".join(lines))


def _generate_sql(question, history, client, uploaded_tables):
    prompt = _build_sql_prompt(uploaded_tables)
    messages = [{"role": "system", "content": prompt}]
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["question"]})
        if turn.get("sql"):
            messages.append({"role": "assistant", "content": turn["sql"]})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=MODEL, max_tokens=700, messages=messages)
    sql = resp.choices[0].message.content.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def _generate_code(question, sql, client):
    user_msg = f"Question: {question}\n\nSQL used:\n{sql}\n\nWrite the equivalent pandas code."
    try:
        code = _call(client, CODE_PROMPT, user_msg, max_tokens=500)
        code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", code).strip()
    except: return ""


def _narrative(question, rows, error, stages, uploaded_tables, client):
    stages_str = ", ".join(stages[:10])
    table_names = ", ".join(uploaded_tables.keys()) if uploaded_tables else "none"
    if error:
        ctx = f"Question: {question}\nError: {error}\nAvailable stages: {stages_str}\nUploaded tables: {table_names}"
    elif not rows:
        ctx = f"Question: {question}\nResult: No rows returned.\nAvailable stages: {stages_str}\nUploaded tables: {table_names}"
    else:
        ctx = f"Question: {question}\nResult ({len(rows)} rows): {str(rows[:5])}\nAvailable stages: {stages_str}"
    try:
        return _call(client, NARRATIVE_PROMPT, ctx, max_tokens=200)
    except: return ""


class HRAgent:
    def __init__(self, hr_csv_path: str, api_key: str | None = None):
        self.conn = build_database(hr_csv_path)
        key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Please set GROQ_API_KEY or OPENAI_API_KEY.")
        self.client = _Client(api_key=key)
        self.history: list[dict] = []
        self.uploaded_tables: dict[str, dict] = {}

    def ask(self, question: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "question": question, "sql": None, "code": None,
            "rows": [], "money_columns": [], "narrative": "",
            "charts": [], "error": None,
        }
        stages = get_distinct_values(self.conn, "stage")
        chart_intents = detect_chart_intents(question)
        try:
            sql = _generate_sql(question, self.history, self.client, self.uploaded_tables)
            result["sql"] = sql
            safe, reason = _is_safe_sql(sql)
            if not safe:
                result["error"] = reason
                result["narrative"] = _narrative(question, [], reason, stages, self.uploaded_tables, self.client)
                return result
            cur = self.conn.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                result["rows"] = rows
                result["money_columns"] = [c for c in cols if _is_money_col(c)]
            result["narrative"] = _narrative(question, result["rows"], None, stages, self.uploaded_tables, self.client)
            if result["rows"]:
                result["charts"] = pick_charts(result["rows"], chart_intents, question)
            result["code"] = _generate_code(question, sql, self.client)
            self.history.append({"question": question, "sql": sql})
        except sqlite3.Error as e:
            err = str(e)
            result["error"] = err
            result["narrative"] = _narrative(question, [], err, stages, self.uploaded_tables, self.client)
        except Exception as e:
            err = str(e)
            result["error"] = err
            result["narrative"] = f"Something went wrong: {err}"
        return result

    def upload_csv(self, table_name: str, file_path: str) -> dict:
        info = load_extra_csv(self.conn, table_name, file_path)
        if "error" not in info:
            self.uploaded_tables[info["table"]] = info
        return info

    def remove_table(self, table_name: str) -> bool:
        ok = remove_table(self.conn, table_name)
        if ok and table_name in self.uploaded_tables:
            del self.uploaded_tables[table_name]
        return ok

    def get_schema_summary(self) -> list[dict]:
        return list_tables(self.conn)

    def clear_history(self) -> None:
        self.history = []
