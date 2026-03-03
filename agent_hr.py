"""
Tesla HR Intelligence — Generic Agent Core
Works with ANY uploaded CSV. Auto-detects schema, generates SQL + Python code + charts.
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

BASE_SQL_PROMPT = """You are an expert SQL analyst. Convert natural-language questions into
correct, read-only SQLite SELECT queries against the available tables.

AVAILABLE TABLES:
{table_descriptions}

SQL RULES:
1. Output ONLY raw SQL - no markdown, no backticks, no explanation.
2. Use only table and column names listed above - copy them exactly (case-sensitive).
3. Fuzzy text matching: LOWER(col) LIKE LOWER('%value%')
4. Percentages: ROUND(100.0 * part / total, 1) via subqueries or window functions.
5. CTEs (WITH ...) are encouraged for multi-step queries.
6. For time trends: GROUP BY year/month columns if available, ORDER BY them.
7. NULL-safe: use IS NOT NULL guards where appropriate.
8. Never use CREATE / DROP / UPDATE / INSERT / DELETE.
9. If a concept is not directly in the schema, pick the closest available column.

FOLLOW-UP: resolve pronouns (them, same, that, those) using conversation history.
"""

NARRATIVE_PROMPT = """You are a sharp data analyst. Given a question and query results,
respond with a crisp insight in 2-4 sentences.
- Data found: lead with the most important number or trend, name specific values.
- Empty result: say the filter matched nothing, suggest what values actually exist.
- Error: explain plainly what went wrong - no SQL jargon.
- Tone: direct, precise, no fluff.
"""

CODE_PROMPT = """You are a Python data analyst. Given a question and the SQL query used to answer it,
write a clean self-contained pandas + plotly code snippet that performs the same analysis.

The DataFrame `df` is already loaded - DO NOT include pd.read_csv().
Import pandas as pd and plotly.express as px at the top.
Keep under 25 lines. Clear variable names and brief inline comments.
End with `result` (a DataFrame) and optionally `fig` (a plotly figure).
Output ONLY raw Python code. No markdown, no backticks, no prose outside comments.
"""

CHART_KEYWORDS = {
    "funnel":  ["funnel", "pipeline", "conversion", "drop off", "stage breakdown", "dropout"],
    "trend":   ["trend", "over time", "monthly", "by month", "by year", "by quarter",
                "timeline", "time series", "velocity", "per month", "per year", "growth"],
    "bar":     ["bar chart", "bar graph", "column chart", "most", "top", "highest",
                "lowest", "ranking", "rank", "compare", "histogram"],
    "pie":     ["pie", "proportion", "share", "percentage", "distribution",
                "breakdown", "composition", "split"],
    "scatter": ["scatter", "correlation", "vs ", "versus", "relationship between"],
}


def detect_chart_intents(question):
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


def pick_charts(rows, intents, question=""):
    if not rows:
        return []
    cols = list(rows[0].keys())
    numeric_cols = [c for c in cols if _is_numeric_col(rows, c)]
    text_cols    = [c for c in cols if c not in numeric_cols]
    date_cols    = [c for c in cols if _is_date_col(c)]
    if not numeric_cols:
        return []
    x  = date_cols[0] if date_cols else (text_cols[0] if text_cols else cols[0])
    y  = numeric_cols[0]
    y2 = numeric_cols[1] if len(numeric_cols) > 1 else None
    ty, tx = y.replace("_", " ").title(), x.replace("_", " ").title()
    charts, used = [], set()

    def add(cfg):
        if cfg["chart"] not in used and len(charts) < 3:
            charts.append(cfg); used.add(cfg["chart"])

    for intent in intents:
        if intent == "funnel":
            add({"chart": "funnel", "x": x, "y": y, "title": f"Funnel - {ty}"})
        elif intent == "trend" and date_cols:
            add({"chart": "line", "x": date_cols[0], "y": y, "title": f"{ty} Over Time"})
        elif intent == "bar":
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        elif intent == "pie" and len(rows) <= 14:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
        elif intent == "scatter" and y2:
            add({"chart": "scatter", "x": y, "y": y2,
                 "title": f"{ty} vs {y2.replace('_',' ').title()}"})
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
    return any(k in col.lower() for k in (
        "salary", "pay", "wage", "compensation", "bonus",
        "amount", "cost", "budget", "usd", "revenue", "total", "price",
    ))


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


def _infer_logical_type(series):
    s = series.dropna()
    if s.empty:
        return "str"
    for check, label in [
        (pd.api.types.is_bool_dtype,           "bool"),
        (pd.api.types.is_integer_dtype,        "int"),
        (pd.api.types.is_float_dtype,          "float"),
        (pd.api.types.is_datetime64_any_dtype, "datetime"),
    ]:
        if check(s):
            return label
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return "float" if (num.dropna() % 1 != 0).mean() > 0.05 else "int"
    try:
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.8:
            return "datetime"
    except Exception:
        pass
    return "str"


def create_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_csv_to_db(conn, table_name, file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}
    if df.empty:
        return {"error": "File is empty"}

    base = os.path.splitext(os.path.basename(table_name))[0]
    safe = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "data"

    logical_types = {col: _infer_logical_type(df[col]) for col in df.columns}

    # Auto-add year/month/quarter helper cols for any date column
    for col in list(df.columns):
        if logical_types[col] in ("datetime", "str") and _is_date_col(col):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed.dt.strftime("%Y-%m-%d")
                    logical_types[col] = "date"
                    df[col + "_year"]    = parsed.dt.year
                    df[col + "_month"]   = parsed.dt.month
                    df[col + "_quarter"] = parsed.dt.quarter.apply(lambda q: f"Q{int(q)}" if pd.notna(q) else None)
                    logical_types[col + "_year"]    = "int"
                    logical_types[col + "_month"]   = "int"
                    logical_types[col + "_quarter"] = "str"
            except Exception:
                pass

    df.to_sql(safe, conn, if_exists="replace", index=False)

    join_hints = _detect_relationships(conn, safe, df)

    schema_lines = [f"  - {col}  ({logical_types.get(col,'str')})" for col in df.columns]
    schema_description = "\n".join(schema_lines)

    sample_values = {}
    for col in df.columns:
        if logical_types.get(col, "str") in ("str", "bool"):
            vals = [str(v) for v in df[col].dropna().unique()[:6]]
            if vals:
                sample_values[col] = vals

    return {
        "table":              safe,
        "original_name":      table_name,
        "columns":            list(df.columns),
        "col_types":          logical_types,
        "rows_loaded":        int(len(df)),
        "schema_description": schema_description,
        "sample_values":      sample_values,
        "join_hints":         join_hints,
        "sample_rows":        df.head(2).to_dict(orient="records"),
    }


def _detect_relationships(conn, new_table, df):
    join_hints = []
    all_tables = {}
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ):
        all_tables[name] = [
            row[1] for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        ]
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique == 0 or nunique > 50000:
            continue
        sample_list = [v.item() if hasattr(v, "item") else v
                       for v in pd.unique(df[col].dropna())[:200]]
        for ot, oc in all_tables.items():
            if ot == new_table or col not in oc:
                continue
            ph = ",".join(["?"] * len(sample_list))
            try:
                (matches,) = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{ot}" WHERE "{col}" IN ({ph})',
                    sample_list,
                ).fetchone()
                ratio = matches / float(len(sample_list) or 1)
                if ratio >= 0.6:
                    join_hints.append(
                        f'"{new_table}"."{col}" can JOIN "{ot}"."{col}" (~{ratio:.0%} overlap)'
                    )
            except Exception:
                pass
    return join_hints


def remove_table(conn, table_name):
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"'); conn.commit(); return True
    except Exception:
        return False


def list_tables(conn):
    tables = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name"
    ).fetchall()
    result = []
    for t in tables:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t["name"]}"').fetchone()[0]
        except Exception:
            count = "?"
        result.append({"name": t["name"], "type": t["type"], "rows": count})
    return result


def _build_sql_prompt(tables_info):
    if not tables_info:
        return BASE_SQL_PROMPT.format(table_descriptions="No tables loaded yet.")
    desc_blocks = []
    for tname, info in tables_info.items():
        block = [f'TABLE: "{tname}"  ({info["rows_loaded"]} rows)', "COLUMNS:"]
        block.append(info["schema_description"])
        sv = info.get("sample_values", {})
        if sv:
            block.append("SAMPLE VALUES (use for filters):")
            for col, vals in list(sv.items())[:8]:
                block.append(f"  {col}: {vals}")
        for hint in info.get("join_hints", []):
            block.append(f"JOIN HINT: {hint}")
        sample = info.get("sample_rows", [])
        if sample:
            block.append(f"EXAMPLE ROW: {sample[0]}")
        desc_blocks.append("\n".join(block))
    return BASE_SQL_PROMPT.format(table_descriptions="\n\n".join(desc_blocks))


def _call(client, system, user, max_tokens=400):
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    return resp.choices[0].message.content.strip()


def _generate_sql(question, history, client, tables_info):
    prompt = _build_sql_prompt(tables_info)
    messages = [{"role": "system", "content": prompt}]
    for turn in history[-6:]:
        messages.append({"role": "user",      "content": turn["question"]})
        if turn.get("sql"):
            messages.append({"role": "assistant", "content": turn["sql"]})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=MODEL, max_tokens=700, messages=messages)
    sql = resp.choices[0].message.content.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def _generate_code(question, sql, col_names, client):
    col_list = ", ".join(col_names[:15])
    user_msg = (
        f"Question: {question}\n\nSQL used:\n{sql}\n\n"
        f"The DataFrame `df` has columns: {col_list}\n\n"
        "Write the equivalent pandas code."
    )
    try:
        code = _call(client, CODE_PROMPT, user_msg, max_tokens=500)
        code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", code).strip()
    except Exception:
        return ""


def _narrative(question, rows, error, tables_info, client):
    table_names = ", ".join(tables_info.keys()) if tables_info else "none"
    if error:
        ctx = f"Question: {question}\nError: {error}\nAvailable tables: {table_names}"
    elif not rows:
        hints = []
        for info in tables_info.values():
            for col, vals in list(info.get("sample_values", {}).items())[:3]:
                hints.append(f"{col}: {vals}")
        ctx = (f"Question: {question}\nResult: No rows returned.\n"
               f"Available tables: {table_names}\nSample values: {'; '.join(hints)}")
    else:
        ctx = f"Question: {question}\nResult ({len(rows)} rows): {str(rows[:5])}"
    try:
        return _call(client, NARRATIVE_PROMPT, ctx, max_tokens=200)
    except Exception:
        return ""


class DataAgent:
    """Generic agent that works with any uploaded CSV(s)."""

    def __init__(self, api_key=None):
        self.conn = create_connection()
        key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Please set GROQ_API_KEY or OPENAI_API_KEY.")
        self.client = _Client(api_key=key)
        self.history = []
        self.tables_info = {}

    def upload_csv(self, table_name, file_path):
        info = load_csv_to_db(self.conn, table_name, file_path)
        if "error" not in info:
            self.tables_info[info["table"]] = info
        return info

    def remove_table(self, table_name):
        ok = remove_table(self.conn, table_name)
        if ok and table_name in self.tables_info:
            del self.tables_info[table_name]
        return ok

    def get_schema_summary(self):
        return list_tables(self.conn)

    def has_data(self):
        return bool(self.tables_info)

    def ask(self, question):
        result = {
            "question": question, "sql": None, "code": None,
            "rows": [], "money_columns": [], "narrative": "",
            "charts": [], "error": None,
        }

        if not self.tables_info:
            result["narrative"] = "No data loaded yet. Please upload a CSV file using the sidebar first."
            return result

        chart_intents = detect_chart_intents(question)

        try:
            sql = _generate_sql(question, self.history, self.client, self.tables_info)
            result["sql"] = sql

            safe, reason = _is_safe_sql(sql)
            if not safe:
                result["error"] = reason
                result["narrative"] = _narrative(question, [], reason, self.tables_info, self.client)
                return result

            cur = self.conn.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                result["rows"] = rows
                result["money_columns"] = [c for c in cols if _is_money_col(c)]
                all_cols = []
                for info in self.tables_info.values():
                    all_cols.extend(info["columns"])
                result["code"] = _generate_code(question, sql, list(dict.fromkeys(all_cols)), self.client)

            result["narrative"] = _narrative(question, result["rows"], None, self.tables_info, self.client)

            if result["rows"]:
                result["charts"] = pick_charts(result["rows"], chart_intents, question)

            self.history.append({"question": question, "sql": sql})

        except sqlite3.Error as e:
            err = str(e)
            result["error"] = err
            result["narrative"] = _narrative(question, [], err, self.tables_info, self.client)
        except Exception as e:
            err = str(e)
            result["error"] = err
            result["narrative"] = f"Something went wrong: {err}"

        return result

    def clear_history(self):
        self.history = []
