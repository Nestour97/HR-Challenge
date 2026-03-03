"""
Tesla HR Intelligence — Generic Agent Core
- Sanitises column names on load (spaces → underscores)
- Passes EXACT column names prominently in every prompt
- Retries SQL up to 2× with the error message fed back to LLM
- Works with ANY CSV
"""
from __future__ import annotations
import os, re, sqlite3, tempfile
from typing import Any
import pandas as pd

try:
    from groq import Groq as _Client
    _CLIENT_TYPE = "groq"
except ImportError:
    from openai import OpenAI as _Client
    _CLIENT_TYPE = "openai"

MODEL = "llama-3.3-70b-versatile" if _CLIENT_TYPE == "groq" else "gpt-4o"

BLOCKED = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+VIEW|DELETE\s+FROM|UPDATE\s+\w+\s+SET|INSERT\s+INTO"
    r"|ALTER\s+TABLE|TRUNCATE|CREATE\s+TABLE|CREATE\s+VIEW|REPLACE\s+INTO"
    r"|ATTACH\s+DATABASE|DETACH\s+DATABASE)\b",
    re.IGNORECASE | re.DOTALL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SQL_SYSTEM = """You are an expert SQLite analyst. Convert the user's natural-language question
into a single read-only SELECT query.

{schema_block}

STRICT RULES:
1. Output ONLY raw SQL — zero markdown, zero backticks, zero explanation.
2. ONLY use the exact column names listed above — copy character-for-character.
3. NEVER invent column names. If you are unsure, pick the closest listed column.
4. Text filters: LOWER(col) LIKE LOWER('%value%')
5. Percentages: ROUND(100.0 * part / total, 1)
6. CTEs are encouraged for multi-step logic.
7. ORDER BY relevant column; add LIMIT where appropriate.
8. Reserved words as column names must be quoted: "Level", "Date", "Group", etc.
9. NEVER use CREATE / DROP / UPDATE / INSERT / DELETE.
"""

RETRY_SYSTEM = """You are an expert SQLite analyst. The query below failed.
Fix ONLY the error — do not change the logic.

{schema_block}

FAILED QUERY:
{failed_sql}

ERROR:
{error}

Output ONLY the corrected raw SQL. No markdown, no backticks, no explanation.
"""

NARRATIVE_SYSTEM = """You are a sharp data analyst. Write a 2-4 sentence insight.
- Data found: lead with the key number or trend, name specific values.
- Empty result: say filter matched nothing, suggest real values that exist.
- Error: explain plainly in plain English — no SQL jargon.
Tone: direct, precise, no fluff."""

CODE_SYSTEM = """You are a Python data analyst. Write a concise pandas + plotly snippet
equivalent to the SQL provided. Rules:
- `df` is already loaded — do NOT call pd.read_csv().
- Import pandas as pd and plotly.express as px at the top.
- Max 25 lines. Clean names, brief inline comments.
- End with `result` (DataFrame) and optionally `fig` (plotly figure).
- Output ONLY raw Python. No markdown, no backticks, no prose outside comments."""

# ─────────────────────────────────────────────────────────────────────────────
# Column name sanitiser
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_col(name: str) -> str:
    """'Date Applied' → 'Date_Applied'  |  'Job Level (Code)' → 'Job_Level_Code'"""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    seen: dict[str, int] = {}
    for col in df.columns:
        new = _sanitise_col(col)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        mapping[col] = new
    return df.rename(columns=mapping)

# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

CHART_KW = {
    "funnel":  ["funnel", "pipeline", "conversion", "drop off", "stage breakdown", "dropout"],
    "trend":   ["trend", "over time", "monthly", "by month", "by year", "by quarter",
                "timeline", "time series", "per month", "per year", "growth"],
    "bar":     ["bar chart", "bar graph", "most", "top", "highest", "lowest",
                "ranking", "rank", "compare", "histogram"],
    "pie":     ["pie", "proportion", "share", "percentage", "distribution",
                "breakdown", "composition", "split"],
    "scatter": ["scatter", "correlation", "vs ", "versus", "relationship between"],
}


def detect_chart_intents(q: str) -> list[str]:
    q = q.lower()
    out = [ct for ct, pats in CHART_KW.items() if any(p in q for p in pats)]
    if not out and any(w in q for w in ("chart", "graph", "visual", "plot", "show")):
        out.append("auto")
    return out


def _is_numeric(rows, col):
    ok = 0
    for r in rows[:10]:
        v = str(r.get(col, "")).replace(",","").replace("$","").replace("%","").strip()
        if not v:
            continue
        try:
            float(v); ok += 1
        except ValueError:
            return False
    return ok > 0


def _is_date_col(col: str) -> bool:
    return any(k in col.lower() for k in ("date","month","year","quarter","period","time"))


def pick_charts(rows, intents, question=""):
    if not rows:
        return []
    cols = list(rows[0].keys())
    num  = [c for c in cols if _is_numeric(rows, c)]
    txt  = [c for c in cols if c not in num]
    dtc  = [c for c in cols if _is_date_col(c)]
    if not num:
        return []
    x  = dtc[0] if dtc else (txt[0] if txt else cols[0])
    y  = num[0]
    y2 = num[1] if len(num) > 1 else None
    ty = y.replace("_", " ").title()
    tx = x.replace("_", " ").title()
    charts, used = [], set()

    def add(cfg):
        if cfg["chart"] not in used and len(charts) < 3:
            charts.append(cfg); used.add(cfg["chart"])

    for intent in intents:
        if   intent == "funnel":                    add({"chart":"funnel","x":x,"y":y,"title":f"Funnel — {ty}"})
        elif intent == "trend" and dtc:             add({"chart":"line","x":dtc[0],"y":y,"title":f"{ty} Over Time"})
        elif intent == "bar":                       add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})
        elif intent == "pie" and len(rows) <= 14:   add({"chart":"pie","x":x,"y":y,"title":f"{ty} Distribution"})
        elif intent == "scatter" and y2:            add({"chart":"scatter","x":y,"y":y2,"title":f"{ty} vs {y2.replace('_',' ').title()}"})
        elif intent == "auto":
            if len(rows) <= 8: add({"chart":"pie","x":x,"y":y,"title":f"{ty} Distribution"})
            else:              add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})

    if len(charts) < 2 and len(rows) >= 3:
        if "bar"  not in used and txt: add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})
        if "line" not in used and dtc: add({"chart":"line","x":dtc[0],"y":y,"title":f"{ty} Over Time"})
        if "pie"  not in used and len(rows) <= 12 and txt: add({"chart":"pie","x":x,"y":y,"title":f"{ty} Share"})

    return charts[:3]

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_type(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty: return "TEXT"
    if pd.api.types.is_bool_dtype(s):    return "INTEGER"
    if pd.api.types.is_integer_dtype(s): return "INTEGER"
    if pd.api.types.is_float_dtype(s):   return "REAL"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return "REAL" if (num.dropna() % 1 != 0).mean() > 0.05 else "INTEGER"
    try:
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.8: return "TEXT"  # stored as ISO string
    except Exception: pass
    return "TEXT"


def _is_money(col: str) -> bool:
    return any(k in col.lower() for k in
               ("salary","pay","wage","compensation","bonus","amount","cost",
                "budget","usd","revenue","total","price"))


def _safe_sql(sql: str):
    s = sql.strip().lstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", s, re.IGNORECASE):
        return False, "Query must start with SELECT or WITH"
    m = BLOCKED.search(sql)
    if m: return False, f"Blocked keyword: {m.group().strip()}"
    if ";" in re.sub(r"'[^']*'", "''", sql).rstrip(";"):
        return False, "Multiple statements not allowed"
    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# Schema block builder (the single most important thing for LLM accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def _schema_block(tables_info: dict) -> str:
    if not tables_info:
        return "No tables loaded. Ask user to upload a CSV first."
    blocks = []
    for tname, info in tables_info.items():
        lines = [
            f'TABLE NAME (use exactly): "{tname}"',
            f'ROW COUNT: {info["rows_loaded"]:,}',
            "COLUMNS (use these exact names — copy character-for-character):",
        ]
        for col, dtype in info["col_types"].items():
            sv = info["sample_values"].get(col)
            sv_str = f"  e.g. {sv[:5]}" if sv else ""
            lines.append(f'  "{col}"  {dtype}{sv_str}')
        for hint in info.get("join_hints", []):
            lines.append(f"JOIN: {hint}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chat(client, system: str, messages: list, max_tokens=700) -> str:
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return resp.choices[0].message.content.strip()


def _clean_sql(raw: str) -> str:
    s = re.sub(r"^```(?:sql)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _generate_sql(question, history, client, tables_info) -> str:
    schema = _schema_block(tables_info)
    system = SQL_SYSTEM.format(schema_block=schema)
    msgs = []
    for t in history[-6:]:
        msgs.append({"role": "user",      "content": t["question"]})
        if t.get("sql"):
            msgs.append({"role": "assistant", "content": t["sql"]})
    msgs.append({"role": "user", "content": question})
    return _clean_sql(_chat(client, system, msgs))


def _retry_sql(question, failed_sql, error, client, tables_info) -> str:
    schema = _schema_block(tables_info)
    system = RETRY_SYSTEM.format(
        schema_block=schema, failed_sql=failed_sql, error=error
    )
    return _clean_sql(_chat(client, system, [{"role": "user", "content": question}]))


def _generate_code(question, sql, col_names, client) -> str:
    col_str = ", ".join(col_names[:20])
    msg = (f"Question: {question}\n\nSQL:\n{sql}\n\n"
           f"DataFrame `df` columns: {col_str}\n\nWrite equivalent pandas code.")
    try:
        raw = _chat(client, CODE_SYSTEM, [{"role":"user","content":msg}], max_tokens=500)
        raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", raw).strip()
    except Exception:
        return ""


def _narrative(question, rows, error, tables_info, client) -> str:
    tnames = ", ".join(tables_info.keys()) if tables_info else "none"
    if error:
        ctx = f"Question: {question}\nError: {error}\nTables: {tnames}"
    elif not rows:
        hints = []
        for info in tables_info.values():
            for col, vals in list(info["sample_values"].items())[:4]:
                hints.append(f'{col}: {vals}')
        ctx = (f"Question: {question}\nResult: 0 rows.\n"
               f"Tables: {tnames}\nSample values: {'; '.join(hints)}")
    else:
        ctx = f"Question: {question}\nResult ({len(rows)} rows): {rows[:5]}"
    try:
        return _chat(client, NARRATIVE_SYSTEM, [{"role":"user","content":ctx}], max_tokens=220)
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# CSV → SQLite loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(conn: sqlite3.Connection, display_name: str, file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}
    if df.empty:
        return {"error": "File appears to be empty"}

    # 1. Sanitise column names (spaces → underscores, etc.)
    df = sanitise_columns(df)

    # 2. Table name from filename
    base = os.path.splitext(os.path.basename(display_name))[0]
    tname = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "data"

    # 3. Detect types
    col_types = {}
    for col in df.columns:
        col_types[col] = _infer_type(df[col])

    # 4. Auto-add year/month/quarter for date columns
    for col in list(df.columns):
        if _is_date_col(col):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed.dt.strftime("%Y-%m-%d")
                    col_types[col] = "TEXT (ISO date)"
                    yr_col = f"{col}_year"
                    mo_col = f"{col}_month"
                    qt_col = f"{col}_quarter"
                    df[yr_col] = parsed.dt.year.astype("Int64")
                    df[mo_col] = parsed.dt.month.astype("Int64")
                    df[qt_col] = parsed.dt.quarter.apply(
                        lambda q: f"Q{int(q)}" if pd.notna(q) else None
                    )
                    col_types[yr_col] = "INTEGER"
                    col_types[mo_col] = "INTEGER"
                    col_types[qt_col] = "TEXT"
            except Exception:
                pass

    # 5. Write to SQLite
    df.to_sql(tname, conn, if_exists="replace", index=False)

    # 6. Sample values for categorical columns
    sample_values = {}
    for col in df.columns:
        if col_types.get(col, "TEXT") in ("TEXT", "TEXT (ISO date)"):
            vals = [str(v) for v in df[col].dropna().unique()[:8]]
            if vals:
                sample_values[col] = vals

    # 7. Join hints
    join_hints = _detect_joins(conn, tname, df)

    return {
        "table":        tname,
        "original":     display_name,
        "columns":      list(df.columns),
        "col_types":    col_types,
        "rows_loaded":  int(len(df)),
        "sample_values": sample_values,
        "join_hints":   join_hints,
    }


def _detect_joins(conn, new_table, df) -> list[str]:
    hints = []
    others = {}
    for (n,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ):
        others[n] = [r[1] for r in conn.execute(f'PRAGMA table_info("{n}")').fetchall()]
    for col in df.columns:
        nu = df[col].nunique(dropna=True)
        if nu == 0 or nu > 100000:
            continue
        sample = [v.item() if hasattr(v,"item") else v for v in pd.unique(df[col].dropna())[:200]]
        for ot, oc in others.items():
            if ot == new_table or col not in oc:
                continue
            ph = ",".join(["?"] * len(sample))
            try:
                (m,) = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{ot}" WHERE "{col}" IN ({ph})',
                    sample
                ).fetchone()
                ratio = m / max(len(sample), 1)
                if ratio >= 0.6:
                    hints.append(f'"{new_table}"."{col}" ~ "{ot}"."{col}" ({ratio:.0%} overlap)')
            except Exception:
                pass
    return hints


def list_tables(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    result = []
    for (n,) in rows:
        try:
            cnt = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
        except Exception:
            cnt = "?"
        result.append({"name": n, "rows": cnt})
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Public Agent class
# ─────────────────────────────────────────────────────────────────────────────

class DataAgent:
    def __init__(self, api_key=None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY.")
        self.client  = _Client(api_key=key)
        self.history: list[dict] = []
        self.tables_info: dict[str, dict] = {}

    # ── CSV management ──────────────────────────────────────────────────
    def upload_csv(self, display_name: str, file_path: str) -> dict:
        info = load_csv(self.conn, display_name, file_path)
        if "error" not in info:
            self.tables_info[info["table"]] = info
        return info

    def remove_table(self, tname: str) -> bool:
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
            self.conn.commit()
            self.tables_info.pop(tname, None)
            return True
        except Exception:
            return False

    def get_schema_summary(self) -> list[dict]:
        return list_tables(self.conn)

    def has_data(self) -> bool:
        return bool(self.tables_info)

    # ── Main ask ────────────────────────────────────────────────────────
    def ask(self, question: str) -> dict[str, Any]:
        out = dict(question=question, sql=None, code=None, rows=[],
                   money_columns=[], narrative="", charts=[], error=None)

        if not self.tables_info:
            out["narrative"] = "No data loaded yet. Please upload a CSV from the sidebar."
            return out

        intents = detect_chart_intents(question)

        # ── Generate + execute SQL, retry once on failure ────────────
        sql = _generate_sql(question, self.history, self.client, self.tables_info)
        out["sql"] = sql

        ok, reason = _safe_sql(sql)
        if not ok:
            out["error"] = reason
            out["narrative"] = _narrative(question, [], reason, self.tables_info, self.client)
            return out

        rows, exec_error = self._run_sql(sql)

        # ── Retry if execution failed ────────────────────────────────
        if exec_error:
            sql2 = _retry_sql(question, sql, exec_error, self.client, self.tables_info)
            rows2, exec_error2 = self._run_sql(sql2)
            if not exec_error2:
                out["sql"] = sql2          # show the fixed SQL
                rows, exec_error = rows2, None
            else:
                out["error"] = exec_error  # show original error if retry also fails

        if exec_error:
            out["narrative"] = _narrative(question, [], exec_error, self.tables_info, self.client)
            return out

        out["rows"] = rows

        # ── Money columns ────────────────────────────────────────────
        if rows:
            out["money_columns"] = [c for c in rows[0].keys() if _is_money(c)]

        # ── Python code ──────────────────────────────────────────────
        all_cols = []
        for info in self.tables_info.values():
            all_cols.extend(info["columns"])
        out["code"] = _generate_code(question, out["sql"], list(dict.fromkeys(all_cols)), self.client)

        # ── Narrative ────────────────────────────────────────────────
        out["narrative"] = _narrative(question, rows, None, self.tables_info, self.client)

        # ── Charts ───────────────────────────────────────────────────
        if rows:
            out["charts"] = pick_charts(rows, intents, question)

        self.history.append({"question": question, "sql": out["sql"]})
        return out

    def _run_sql(self, sql: str):
        try:
            cur = self.conn.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()], None
            return [], None
        except sqlite3.Error as e:
            return [], str(e)

    def clear_history(self):
        self.history = []
