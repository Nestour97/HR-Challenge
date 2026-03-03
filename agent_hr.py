"""Tesla HR Intelligence — Robust Agent Core

Key improvements over v1:
- Column name validation BEFORE and AFTER SQL generation
- Explicit JOIN instructions injected when multiple tables share a key column
- Post-generation SQL rewriting that fixes wrong column names automatically
- Aggressive retry with direct column substitution map
- Table/column introspection from live SQLite so prompt is always 100% accurate
"""

from __future__ import annotations

import os
import re
import sqlite3
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── LLM client ────────────────────────────────────────────────────────────────

try:
    from groq import Groq as _Client
    _CLIENT_TYPE = "groq"
except Exception:
    try:
        from openai import OpenAI as _Client
        _CLIENT_TYPE = "openai"
    except Exception:
        _Client = None
        _CLIENT_TYPE = "none"

MODEL = "llama-3.3-70b-versatile" if _CLIENT_TYPE == "groq" else "gpt-4o"
MAX_RETRIES = 3

BLOCKED = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+VIEW|DELETE\s+FROM|UPDATE\s+\w+\s+SET|INSERT\s+INTO"
    r"|ALTER\s+TABLE|TRUNCATE|CREATE\s+TABLE|CREATE\s+VIEW|REPLACE\s+INTO"
    r"|ATTACH\s+DATABASE|DETACH\s+DATABASE)\b",
    re.IGNORECASE | re.DOTALL,
)

# ── Column sanitiser ──────────────────────────────────────────────────────────

def _sanitise_col(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s or "col").lower()


def sanitise_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for col in df.columns:
        new = _sanitise_col(col)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        mapping[col] = new
    return df.rename(columns=mapping), mapping


# ── Schema introspection (always from live SQLite) ────────────────────────────

def _live_schema(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Read exact table/column info directly from SQLite — always 100% accurate."""
    result: Dict[str, Dict] = {}
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (tname,) in tables:
        cols_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
        cols = [r[1] for r in cols_info]  # col name is index 1
        try:
            cnt = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
        except Exception:
            cnt = 0
        result[tname] = {"columns": cols, "rows": cnt}
    return result


def _live_samples(conn: sqlite3.Connection, tname: str, cols: List[str]) -> Dict[str, List]:
    samples: Dict[str, List] = {}
    for col in cols[:30]:  # limit to first 30 cols
        try:
            rows = conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{tname}" WHERE "{col}" IS NOT NULL LIMIT 6'
            ).fetchall()
            vals = [str(r[0]) for r in rows if r[0] is not None]
            if vals:
                samples[col] = vals
        except Exception:
            pass
    return samples


# ── Join detection ────────────────────────────────────────────────────────────

def _detect_shared_keys(schema: Dict[str, Dict]) -> List[Tuple[str, str, str]]:
    """Find columns shared across tables (candidate JOIN keys).
    Returns list of (table1, table2, column_name).
    """
    table_cols: Dict[str, set] = {t: set(info["columns"]) for t, info in schema.items()}
    tables = list(table_cols.keys())
    joins = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            t1, t2 = tables[i], tables[j]
            shared = table_cols[t1] & table_cols[t2]
            for col in shared:
                joins.append((t1, t2, col))
    return joins


# ── SQL system prompt ─────────────────────────────────────────────────────────

def _build_system_prompt(conn: sqlite3.Connection) -> str:
    schema = _live_schema(conn)
    if not schema:
        return "No tables loaded yet. Ask the user to upload a CSV or Excel file."

    sections = []

    # Schema block
    sections.append("=" * 70)
    sections.append("DATABASE SCHEMA — USE ONLY THESE EXACT NAMES (copy character-for-character)")
    sections.append("=" * 70)

    for tname, info in schema.items():
        sections.append(f'\nTABLE: "{tname}"  |  {info["rows"]:,} rows')
        sections.append("Columns:")
        samples = _live_samples(conn, tname, info["columns"])
        for col in info["columns"]:
            sv = samples.get(col, [])
            sample_str = f"  → samples: {sv}" if sv else ""
            sections.append(f'  "{col}"{sample_str}')

    # JOIN hints
    joins = _detect_shared_keys(schema)
    if joins:
        sections.append("\n" + "=" * 70)
        sections.append("JOIN RELATIONSHIPS (shared columns between tables):")
        sections.append("=" * 70)
        for t1, t2, col in joins:
            sections.append(
                f'  "{t1}" JOIN "{t2}" ON "{t1}"."{col}" = "{t2}"."{col}"'
            )
        sections.append("")
        sections.append("MULTI-TABLE QUERY EXAMPLE:")
        if joins:
            t1, t2, col = joins[0]
            c1 = schema[t1]["columns"][0]
            c2 = schema[t2]["columns"][1] if len(schema[t2]["columns"]) > 1 else schema[t2]["columns"][0]
            sections.append(
                f'  SELECT T1."{c1}", T2."{c2}", COUNT(*) as count\n'
                f'  FROM "{t1}" T1\n'
                f'  JOIN "{t2}" T2 ON T1."{col}" = T2."{col}"\n'
                f'  GROUP BY T1."{c1}", T2."{c2}"\n'
                f'  ORDER BY count DESC;'
            )

    schema_block = "\n".join(sections)

    return f"""You are an expert SQLite analyst for Tesla HR data.
Convert the user's natural-language question into ONE read-only SELECT query.

{schema_block}

═══════════════════════════════════════════════════════════════════════
CRITICAL RULES — VIOLATING ANY OF THESE CAUSES ERRORS:
═══════════════════════════════════════════════════════════════════════

RULE 1 — COLUMN NAMES: Use ONLY the exact column names listed above.
  • Copy them character-for-character including case.
  • DO NOT invent names like "Applicant_ID", "Job", "Status" — these don't exist.
  • If you need applicant identifier → use "applicant_id"
  • If you need job level → use "job_level"
  • If you need stage/status → use "stage"
  • If you need gender → use "gender"
  • If you need ethnicity → use "ethnicity"

RULE 2 — TABLE NAMES: Use ONLY the exact table names listed above.

RULE 3 — JOINs: When a question involves columns from multiple tables,
  always JOIN them using the shared column shown in JOIN RELATIONSHIPS above.

RULE 4 — OUTPUT: Return ONLY raw SQL. No markdown, no backticks, no explanation.

RULE 5 — CASE-INSENSITIVE FILTERS: Always use LOWER(col) LIKE LOWER('%value%')
  for text comparisons.

RULE 6 — READ ONLY: Only SELECT queries. Never DROP, DELETE, UPDATE, INSERT.

RULE 7 — GENDER: Gender values are 'F' (female) and 'M' (male).
  RULE 8 — ETHNICITY: Values are 'F'=Female,'M'=Male for gender; 
  'URM'=Under-represented minority, 'AS'=Asian, 'WH'=White, 'Unk'=Unknown for ethnicity.
"""


def _build_retry_prompt(conn: sqlite3.Connection, failed_sql: str, error: str) -> str:
    schema = _live_schema(conn)
    all_cols = []
    for tname, info in schema.items():
        for col in info["columns"]:
            all_cols.append(f'"{tname}"."{col}"')

    return f"""You are an expert SQLite analyst. A query failed. Fix it.

VALID TABLE.COLUMN REFERENCES (use ONLY these):
{chr(10).join(all_cols)}

FAILED QUERY:
{failed_sql}

ERROR MESSAGE:
{error}

INSTRUCTIONS:
1. Find every identifier in the failed query that does NOT appear in the valid list above.
2. Replace each invalid identifier with the correct one from the valid list.
3. Common mistakes to fix:
   - "Applicant_ID" or "applicantid" → "applicant_id"
   - "Job" or "JobLevel" or "Level" → "job_level"
   - "Status" or "Stage" or "hiring_stage" → "stage"
   - "Gender" → "gender"
   - "Ethnicity" or "Race" → "ethnicity"
   - "Date" or "DateApplied" → "date_applied"
4. If a JOIN is needed, use ON T1."applicant_id" = T2."applicant_id"
5. Return ONLY the corrected raw SQL. No markdown, no backticks.
"""


# ── SQL validation & auto-fix ─────────────────────────────────────────────────

def _extract_identifiers(sql: str) -> List[str]:
    """Extract all identifiers from SQL (table/column names)."""
    # Remove string literals
    clean = re.sub(r"'[^']*'", "''", sql)
    # Find double-quoted identifiers
    dq = re.findall(r'"([^"]+)"', clean)
    # Find unquoted identifiers (after SELECT, FROM, WHERE, JOIN, ON, GROUP BY etc.)
    unquoted = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', clean)
    sql_keywords = {
        'select', 'from', 'where', 'join', 'on', 'group', 'by', 'order',
        'having', 'limit', 'offset', 'as', 'and', 'or', 'not', 'in',
        'like', 'between', 'case', 'when', 'then', 'else', 'end',
        'count', 'sum', 'avg', 'min', 'max', 'distinct', 'inner',
        'left', 'right', 'outer', 'cross', 'union', 'all', 'with',
        'round', 'lower', 'upper', 'null', 'is', 'desc', 'asc',
        'coalesce', 'cast', 'integer', 'text', 'real', 't1', 't2', 't3',
    }
    unquoted_ids = [u for u in unquoted if u.lower() not in sql_keywords]
    return list(set(dq + unquoted_ids))


def _auto_fix_sql(sql: str, conn: sqlite3.Connection) -> str:
    """Attempt to auto-fix obvious column/table name errors.
    
    Handles both bare column refs (Job) and alias-prefixed refs (T1.Job, T2.Applicant_ID).
    """
    schema = _live_schema(conn)

    all_tables = list(schema.keys())
    flat_cols: List[str] = []
    flat_cols_lower_map: Dict[str, str] = {}  # lower -> original

    for tname, info in schema.items():
        for col in info["columns"]:
            flat_cols.append(col)
            flat_cols_lower_map[col.lower()] = col

    KEYWORD_MAP = {
        "job": "job_level", "level": "job_level", "joblevel": "job_level",
        "applicantid": "applicant_id", "candidateid": "applicant_id",
        "status": "stage", "hirestage": "stage",
        "sex": "gender", "race": "ethnicity",
        "dateapplied": "date_applied", "applicationdate": "date_applied",
        "dept": "department_code", "department": "department_code",
    }

    def best_col_match(ident: str) -> Optional[str]:
        """Find the best matching real column for an identifier."""
        if ident in all_tables:
            return None  # it's a table name, not a column
        lower = ident.lower()
        if lower in flat_cols_lower_map:
            return None  # already correct
        # Keyword lookup
        kw = lower.replace("_", "").replace(" ", "")
        if kw in KEYWORD_MAP and KEYWORD_MAP[kw] in flat_cols_lower_map:
            return flat_cols_lower_map[KEYWORD_MAP[kw]]
        # Prefix/part match: "job" matches "job_level"
        for col_lower, col_orig in flat_cols_lower_map.items():
            parts = col_lower.split("_")
            if parts[0] == lower or lower in parts:
                return col_orig
        # Fuzzy with lower threshold
        norm = lower.replace("-", "_").replace(" ", "_")
        matches = get_close_matches(norm, list(flat_cols_lower_map.keys()), n=1, cutoff=0.5)
        if matches:
            return flat_cols_lower_map[matches[0]]
        return None

    fixed = sql

    # Fix alias.Column references: T1.Job → T1."job_level", T2.Applicant_ID → T2."applicant_id"
    def fix_alias_ref(m):
        alias = m.group(1)
        col = m.group(2)
        if col.lower() in flat_cols_lower_map:
            return f'{alias}."{flat_cols_lower_map[col.lower()]}"'
        replacement = best_col_match(col)
        if replacement:
            return f'{alias}."{replacement}"'
        return m.group(0)  # unchanged

    fixed = re.sub(r'\b([A-Za-z_]\w*)\."?([A-Za-z_]\w*)"?', fix_alias_ref, fixed)
    fixed = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', fix_alias_ref, fixed)

    # Fix bare column/table references
    bare_ids = _extract_identifiers(fixed)
    replacements: Dict[str, str] = {}
    for ident in bare_ids:
        replacement = best_col_match(ident)
        if replacement and replacement != ident:
            replacements[ident] = replacement

    for wrong, right in replacements.items():
        fixed = re.sub(r'(?<![.\w])' + re.escape(wrong) + r'(?![.\w])', f'"{right}"', fixed)

    return fixed


# ── Chart helpers ─────────────────────────────────────────────────────────────

CHART_KW = {
    "funnel": ["funnel", "pipeline", "conversion", "drop off", "stage breakdown", "dropout"],
    "trend": ["trend", "over time", "monthly", "by month", "by year", "by quarter",
              "timeline", "time series", "per month", "per year", "growth"],
    "bar": ["bar chart", "bar graph", "most", "top", "highest", "lowest",
            "ranking", "rank", "compare", "histogram"],
    "pie": ["pie", "proportion", "share", "percentage", "distribution",
            "breakdown", "composition", "split"],
    "scatter": ["scatter", "correlation", "vs ", "versus", "relationship between"],
}


def detect_chart_intents(q: str) -> List[str]:
    q = q.lower()
    out = [ct for ct, pats in CHART_KW.items() if any(p in q for p in pats)]
    if not out and any(w in q for w in ("chart", "graph", "visual", "plot", "show")):
        out.append("auto")
    return out


def _is_numeric(rows: List[dict], col: str) -> bool:
    ok = 0
    for r in rows[:10]:
        v = str(r.get(col, "")).replace(",", "").replace("$", "").replace("%", "").strip()
        if not v:
            continue
        try:
            float(v); ok += 1
        except ValueError:
            return False
    return ok > 0


def _is_date_col(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in ("date", "month", "year", "quarter", "period", "time"))


def pick_charts(rows: List[dict], intents: List[str], question: str = "") -> List[dict]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    num = [c for c in cols if _is_numeric(rows, c)]
    txt = [c for c in cols if c not in num]
    dtc = [c for c in cols if _is_date_col(c)]

    if not num:
        return []

    x = dtc[0] if dtc else (txt[0] if txt else cols[0])
    y = num[0]
    y2 = num[1] if len(num) > 1 else None

    ty = y.replace("_", " ").title()
    tx = x.replace("_", " ").title()

    charts: List[dict] = []
    used: set = set()

    def add(cfg: dict):
        if cfg["chart"] not in used and len(charts) < 3:
            charts.append(cfg)
            used.add(cfg["chart"])

    for intent in intents:
        if intent == "funnel":
            add({"chart": "funnel", "x": x, "y": y, "title": f"Funnel — {ty}"})
        elif intent == "trend" and dtc:
            add({"chart": "line", "x": dtc[0], "y": y, "title": f"{ty} Over Time"})
        elif intent == "bar":
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        elif intent == "pie" and len(rows) <= 14:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
        elif intent == "scatter" and y2:
            add({"chart": "scatter", "x": y, "y": y2, "title": f"{ty} vs {y2.replace('_', ' ').title()}"})
        elif intent == "auto":
            if len(rows) <= 8:
                add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
            else:
                add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})

    if len(charts) < 2 and len(rows) >= 3:
        if "bar" not in used and txt:
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        if "line" not in used and dtc:
            add({"chart": "line", "x": dtc[0], "y": y, "title": f"{ty} Over Time"})
        if "pie" not in used and len(rows) <= 12 and txt:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Share"})

    return charts[:3]


# ── Data type helpers ─────────────────────────────────────────────────────────

def _infer_type(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "TEXT"
    if pd.api.types.is_bool_dtype(s):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return "REAL" if (num.dropna() % 1 != 0).mean() > 0.05 else "INTEGER"
    try:
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.8:
            return "TEXT"
    except Exception:
        pass
    return "TEXT"


def _is_money(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in ("salary", "pay", "wage", "compensation", "bonus",
                                "amount", "cost", "budget", "usd", "revenue", "price"))


def _safe_sql(sql: str) -> Tuple[bool, str]:
    s = sql.strip().lstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", s, re.IGNORECASE):
        return False, "Query must start with SELECT or WITH"
    m = BLOCKED.search(sql)
    if m:
        return False, f"Blocked keyword: {m.group().strip()}"
    if ";" in re.sub(r"'[^']*'", "''", sql).rstrip(";"):
        return False, "Multiple statements not allowed"
    return True, ""


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if parsed.notna().mean() >= 0.5:
        return parsed
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.5:
        return parsed
    serial = numeric.dropna()
    if serial.empty:
        return parsed
    mask = serial.between(20000, 60000)
    if mask.mean() < 0.8:
        return parsed
    parsed2 = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    if parsed2.notna().mean() > parsed.notna().mean():
        return parsed2
    return parsed


# ── DataFrame loader ──────────────────────────────────────────────────────────

def _table_name_from(display_name: str) -> str:
    base = os.path.splitext(os.path.basename(display_name))[0]
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "data"


def _load_dataframe(conn: sqlite3.Connection, display_name: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "File is empty"}

    original_columns = list(df.columns)
    df, col_map = sanitise_columns(df)

    tname = _table_name_from(display_name)

    col_types: Dict[str, str] = {col: _infer_type(df[col]) for col in df.columns}

    # Auto-add year/month/quarter for date columns
    for col in list(df.columns):
        if any(k in col.lower() for k in ("date", "time")):
            try:
                parsed = _parse_dates(df[col])
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed.dt.strftime("%Y-%m-%d")
                    col_types[col] = "TEXT (ISO date)"
                    df[f"{col}_year"] = parsed.dt.year.astype("Int64")
                    df[f"{col}_month"] = parsed.dt.month.astype("Int64")
                    df[f"{col}_quarter"] = parsed.dt.quarter.apply(
                        lambda q: f"Q{int(q)}" if pd.notna(q) else None
                    )
                    col_types[f"{col}_year"] = "INTEGER"
                    col_types[f"{col}_month"] = "INTEGER"
                    col_types[f"{col}_quarter"] = "TEXT"
            except Exception:
                pass

    df.to_sql(tname, conn, if_exists="replace", index=False)

    sample_values: Dict[str, list] = {}
    for col in df.columns:
        vals = [str(v) for v in df[col].dropna().unique()[:8]]
        if vals:
            sample_values[col] = vals

    return {
        "table": tname,
        "original": display_name,
        "columns": list(df.columns),
        "col_types": col_types,
        "col_name_map": col_map,
        "rows_loaded": int(len(df)),
        "sample_values": sample_values,
        "join_hints": [],
    }


def load_tabular(conn: sqlite3.Connection, display_name: str, file_path: str) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            return {"error": f"Could not read Excel file: {e}"}
        tables = []
        total = 0
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            info = _load_dataframe(conn, f"{display_name}__{sheet_name}", df)
            if info.get("table"):
                tables.append(info)
                total += info["rows_loaded"]
        if not tables:
            return {"error": "No non-empty sheets found."}
        return {"tables": tables, "rows_loaded": total}

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    if df.empty:
        return {"error": "File is empty"}

    info = _load_dataframe(conn, display_name, df)
    return {"tables": [info], "rows_loaded": info["rows_loaded"]}


def list_tables(conn: sqlite3.Connection) -> List[dict]:
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


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _chat(client, system: str, messages: List[dict], max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return resp.choices[0].message.content.strip()


def _clean_sql(raw: str) -> str:
    s = re.sub(r"^```(?:sql)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _generate_sql(question: str, history: List[dict], client, conn: sqlite3.Connection) -> str:
    system = _build_system_prompt(conn)
    msgs: List[dict] = []
    for t in history[-6:]:
        msgs.append({"role": "user", "content": t["question"]})
        if t.get("sql"):
            msgs.append({"role": "assistant", "content": t["sql"]})
    msgs.append({"role": "user", "content": question})
    raw = _chat(client, system, msgs)
    sql = _clean_sql(raw)
    # Auto-fix obvious mistakes before returning
    return _auto_fix_sql(sql, conn)


def _retry_sql(question: str, failed_sql: str, error: str, client, conn: sqlite3.Connection) -> str:
    system = _build_retry_prompt(conn, failed_sql, error)
    raw = _chat(client, system, [{"role": "user", "content": question}])
    sql = _clean_sql(raw)
    return _auto_fix_sql(sql, conn)


def _generate_code(question: str, sql: str, col_names: List[str], client) -> str:
    CODE_SYSTEM = """You are a Python data analyst.
Write a concise pandas snippet equivalent to the SQL provided.
- `df` is already loaded — do NOT call pd.read_csv().
- Import pandas as pd and plotly.express as px at the top.
- Max 25 lines. End with `result` (DataFrame).
- Output ONLY raw Python. No markdown, no backticks."""
    col_str = ", ".join(col_names[:20])
    msg = f"Question: {question}\n\nSQL:\n{sql}\n\nDataFrame columns: {col_str}\n\nWrite equivalent pandas code."
    try:
        raw = _chat(client, CODE_SYSTEM, [{"role": "user", "content": msg}], max_tokens=500)
        raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", raw).strip()
    except Exception:
        return ""


def _narrative(question: str, rows: List[dict], error: Optional[str],
               conn: sqlite3.Connection, client) -> str:
    NARRATIVE_SYSTEM = """You are a senior HR data analyst presenting findings to Tesla executives.
Write a clear, professional 2-4 sentence insight.
- Data found: lead with the key number or trend, name specific values.
- Empty result: explain the filter matched nothing, suggest valid values from the data.
- Error: explain clearly in plain English without SQL jargon.
- Be direct, precise, and professional. No filler, no emojis."""

    schema = _live_schema(conn)
    tnames = ", ".join(schema.keys())
    if error:
        ctx = f"Question: {question}\nError: {error}\nTables: {tnames}"
    elif not rows:
        ctx = f"Question: {question}\nResult: 0 rows.\nTables: {tnames}"
    else:
        ctx = f"Question: {question}\nResult ({len(rows)} rows): {rows[:5]}"

    try:
        return _chat(client, NARRATIVE_SYSTEM, [{"role": "user", "content": ctx}], max_tokens=220)
    except Exception:
        return ""


# ── Public DataAgent ──────────────────────────────────────────────────────────

class DataAgent:
    def __init__(self, api_key: str | None = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY.")
        if _Client is None:
            raise RuntimeError("Install the `groq` or `openai` Python package.")
        self.client = _Client(api_key=key)
        self.history: List[dict] = []
        self.tables_info: Dict[str, dict] = {}

    def upload_file(self, display_name: str, file_path: str) -> dict:
        info = load_tabular(self.conn, display_name, file_path)
        if "error" in info:
            return info
        for t in info.get("tables", []):
            self.tables_info[t["table"]] = t
        return info

    def upload_csv(self, display_name: str, file_path: str) -> dict:
        return self.upload_file(display_name, file_path)

    def remove_table(self, tname: str) -> bool:
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
            self.conn.commit()
            self.tables_info.pop(tname, None)
            return True
        except Exception:
            return False

    def get_schema_summary(self) -> List[dict]:
        return list_tables(self.conn)

    def has_data(self) -> bool:
        return bool(self.tables_info)

    def ask(self, question: str) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(
            question=question, sql=None, code=None,
            rows=[], money_columns=[], narrative="", charts=[], error=None,
        )

        if not self.tables_info:
            out["narrative"] = "No data loaded yet. Upload a CSV or Excel file from the sidebar."
            return out

        intents = detect_chart_intents(question)
        sql = _generate_sql(question, self.history, self.client, self.conn)
        out["sql"] = sql

        ok, reason = _safe_sql(sql)
        if not ok:
            out["error"] = reason
            out["narrative"] = _narrative(question, [], reason, self.conn, self.client)
            return out

        rows, exec_error = self._run_sql(sql)

        for _attempt in range(MAX_RETRIES):
            if not exec_error:
                break
            sql_retry = _retry_sql(question, out["sql"], exec_error, self.client, self.conn)
            rows_retry, err_retry = self._run_sql(sql_retry)
            out["sql"] = sql_retry
            if not err_retry:
                rows, exec_error = rows_retry, None
            else:
                exec_error = err_retry

        if exec_error:
            out["error"] = exec_error
            out["narrative"] = _narrative(question, [], exec_error, self.conn, self.client)
            return out

        out["rows"] = rows
        if rows:
            out["money_columns"] = [c for c in rows[0].keys() if _is_money(c)]

        all_cols: List[str] = []
        for info in self.tables_info.values():
            all_cols.extend(info["columns"])
        out["code"] = _generate_code(question, out["sql"], list(dict.fromkeys(all_cols)), self.client)
        out["narrative"] = _narrative(question, rows, None, self.conn, self.client)
        if rows:
            out["charts"] = pick_charts(rows, intents, question)

        self.history.append({"question": question, "sql": out["sql"]})
        return out

    def _run_sql(self, sql: str) -> Tuple[List[dict], Optional[str]]:
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
