"""Tesla HR Intelligence — Agent with Auto-Cleaning Silver/Gold Layer

When files are uploaded, the agent automatically:
  Bronze: loads raw file
  Silver: deduplicates, resolves conflicts → one-row-per-application + one-row-per-applicant
  Gold:   joins everything into a single enriched analysis-ready view

The LLM only ever queries Gold — never the messy raw Bronze.
This prevents fan-out (14M counts), duplicates, and demographic conflicts.
"""

from __future__ import annotations
import os, re, sqlite3
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ── LLM client ────────────────────────────────────────────────────────────────
try:
    from groq import Groq as _Client; _CLIENT_TYPE = "groq"
except Exception:
    try:
        from openai import OpenAI as _Client; _CLIENT_TYPE = "openai"
    except Exception:
        _Client = None; _CLIENT_TYPE = "none"

MODEL = "llama-3.3-70b-versatile" if _CLIENT_TYPE == "groq" else "gpt-4o"
MAX_RETRIES = 3
BLOCKED = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+VIEW|DELETE\s+FROM|UPDATE\s+\w+\s+SET|INSERT\s+INTO"
    r"|ALTER\s+TABLE|TRUNCATE|CREATE\s+TABLE|CREATE\s+VIEW|REPLACE\s+INTO"
    r"|ATTACH\s+DATABASE|DETACH\s+DATABASE)\b", re.IGNORECASE | re.DOTALL)

# ── Column sanitiser ──────────────────────────────────────────────────────────
def _sanitise_col(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    return (re.sub(r"_+", "_", s) or "col").lower()

def sanitise_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}; seen: Dict[str, int] = {}
    for col in df.columns:
        new = _sanitise_col(col)
        if new in seen: seen[new] += 1; new = f"{new}_{seen[new]}"
        else: seen[new] = 0
        mapping[col] = new
    return df.rename(columns=mapping), mapping

# ── Table type detection ──────────────────────────────────────────────────────
def _detect_table_type(sanitized_cols: List[str]) -> str:
    cols = set(sanitized_cols)
    if "stage" in cols and "applicant_id" in cols and "job_level" in cols:
        return "raw_events"
    if "applicant_id" in cols and ("gender" in cols or "ethnicity" in cols):
        return "demographics"
    return "generic"

# ── Silver builders ───────────────────────────────────────────────────────────
def _build_silver_applications(conn: sqlite3.Connection, bronze_table: str) -> str:
    cols_info = conn.execute(f'PRAGMA table_info("{bronze_table}")').fetchall()
    existing = {r[1] for r in cols_info}

    def _find(candidates): return next((c for c in candidates if c in existing), None)

    app_id_col  = _find(["applicant_id","candidate_id"])
    stage_col   = _find(["stage","status","hiring_stage"])
    level_col   = _find(["job_level","level","grade"])
    dept_col    = _find(["department_code","dept_code","department"])
    pos_col     = _find(["job_position_code","position_code","job_code"])
    date_col    = _find(["date_applied","applied_date","application_date"])
    start_col   = _find(["target_start_date","start_date"])

    if not all([app_id_col, stage_col, level_col]):
        return bronze_table

    group_cols = [c for c in [app_id_col, level_col, dept_col, pos_col] if c]
    group_str = ", ".join(f'"{c}"' for c in group_cols)

    stage_rank = f"""CASE LOWER(TRIM("{stage_col}"))
        WHEN 'application' THEN 1 WHEN 'recruiter review' THEN 2
        WHEN 'interview' THEN 3 WHEN 'offer' THEN 4 WHEN 'hired' THEN 5 ELSE 0 END"""

    selects = [f'"{c}"' for c in group_cols]
    if date_col:
        selects += [f'MIN("{date_col}") AS date_applied', f'MAX("{date_col}") AS last_stage_date']
    selects.append(f'MAX({stage_rank}) AS max_stage_rank')
    if start_col:
        selects.append(f'MAX("{start_col}") AS target_start_date')
    selects.append(f'COUNT(*) AS event_count')
    selects.append(f"""CASE MAX({stage_rank})
        WHEN 1 THEN 'Application' WHEN 2 THEN 'Recruiter Review'
        WHEN 3 THEN 'Interview' WHEN 4 THEN 'Offer' WHEN 5 THEN 'Hired'
        ELSE 'Unknown' END AS max_stage""")
    selects.append(f'CASE WHEN MAX({stage_rank}) = 5 THEN 1 ELSE 0 END AS is_hired')

    conn.execute('DROP TABLE IF EXISTS "silver_applications"')
    conn.execute(f'CREATE TABLE "silver_applications" AS SELECT {", ".join(selects)} FROM "{bronze_table}" GROUP BY {group_str}')
    conn.commit()
    return "silver_applications"

def _build_silver_demographics(conn: sqlite3.Connection, bronze_table: str) -> str:
    cols_info = conn.execute(f'PRAGMA table_info("{bronze_table}")').fetchall()
    existing = {r[1] for r in cols_info}
    def _find(candidates): return next((c for c in candidates if c in existing), None)
    app_id_col = _find(["applicant_id","candidate_id"])
    if not app_id_col: return bronze_table

    df = pd.read_sql(f'SELECT * FROM "{bronze_table}"', conn)
    df = df.rename(columns={app_id_col: "applicant_id"})
    gender_col = _find(["gender","sex"])
    eth_col = _find(["ethnicity","race"])
    if gender_col and gender_col != "gender": df = df.rename(columns={gender_col: "gender"})
    if eth_col and eth_col != "ethnicity": df = df.rename(columns={eth_col: "ethnicity"})

    rows = []
    for apid, grp in df.groupby("applicant_id"):
        row = {"applicant_id": apid}
        if "gender" in df.columns:
            gvals = [v for v in grp["gender"].dropna().unique() if str(v).strip()]
            row["gender"] = gvals[0] if len(gvals) == 1 else "Unknown"
        if "ethnicity" in df.columns:
            known = [v for v in grp["ethnicity"].dropna().unique()
                     if str(v).strip() and str(v).lower() not in ("unk","unknown","nan")]
            row["ethnicity"] = "Unk" if not known else (known[0] if len(known) == 1 else "Conflicting")
        rows.append(row)

    silver_df = pd.DataFrame(rows)
    conn.execute('DROP TABLE IF EXISTS "silver_demographics"')
    silver_df.to_sql("silver_demographics", conn, index=False)
    conn.commit()
    return "silver_demographics"

def _build_gold_enriched(conn: sqlite3.Connection) -> Optional[str]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "silver_applications" not in tables or "silver_demographics" not in tables:
        return None

    app_cols = [r[1] for r in conn.execute('PRAGMA table_info("silver_applications")').fetchall()]
    demo_cols = [r[1] for r in conn.execute('PRAGMA table_info("silver_demographics")').fetchall()]

    app_id_col = next((c for c in app_cols if "applicant_id" in c), None)
    if not app_id_col: return None

    app_selects = ", ".join(f'A."{c}"' for c in app_cols)
    demo_extras = [c for c in demo_cols if c != "applicant_id"]
    demo_selects = (", " + ", ".join(f'D."{c}"' for c in demo_extras)) if demo_extras else ""

    computed = []
    if "date_applied" in app_cols and "last_stage_date" in app_cols:
        computed.append("CAST(julianday(A.last_stage_date) - julianday(A.date_applied) AS INTEGER) AS days_to_last_stage")
    if "is_hired" in app_cols and "date_applied" in app_cols and "last_stage_date" in app_cols:
        computed.append("CASE WHEN A.is_hired=1 THEN CAST(julianday(A.last_stage_date)-julianday(A.date_applied) AS INTEGER) ELSE NULL END AS days_to_hire")
    if "target_start_date" in app_cols and "last_stage_date" in app_cols:
        computed.append("CASE WHEN A.is_hired=1 AND A.target_start_date IS NOT NULL THEN CAST(julianday(A.target_start_date)-julianday(A.last_stage_date) AS INTEGER) ELSE NULL END AS days_hire_to_start")
    computed_str = (", " + ", ".join(computed)) if computed else ""

    conn.execute('DROP TABLE IF EXISTS "gold_enriched"')
    conn.execute(f"""CREATE TABLE "gold_enriched" AS
        SELECT {app_selects}{demo_selects}{computed_str}
        FROM "silver_applications" A
        LEFT JOIN "silver_demographics" D ON A."{app_id_col}" = D."applicant_id" """)
    conn.commit()
    return "gold_enriched"

# ── Live schema ───────────────────────────────────────────────────────────────
def _live_schema(conn: sqlite3.Connection) -> Dict[str, Dict]:
    result = {}
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    for (tname,) in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()]
        try: cnt = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
        except: cnt = 0
        result[tname] = {"columns": cols, "rows": cnt}
    return result

def _live_samples(conn: sqlite3.Connection, tname: str, cols: List[str]) -> Dict[str, List]:
    samples = {}
    for col in cols[:25]:
        try:
            rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{tname}" WHERE "{col}" IS NOT NULL LIMIT 6').fetchall()
            vals = [str(r[0]) for r in rows if r[0] is not None]
            if vals: samples[col] = vals
        except: pass
    return samples

# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt(conn: sqlite3.Connection) -> str:
    all_schema = _live_schema(conn)
    if not all_schema: return "No tables loaded yet."

    # Prefer gold_enriched → silver_applications/demographics → fallback all
    if "gold_enriched" in all_schema:
        analysis_tables = {"gold_enriched": all_schema["gold_enriched"]}
    elif "silver_applications" in all_schema:
        analysis_tables = {k: v for k, v in all_schema.items() if k.startswith("silver_")}
    else:
        analysis_tables = {k: v for k, v in all_schema.items() if not k.startswith("Tesla_HR_") or True}

    primary_table = list(analysis_tables.keys())[0]

    sections = ["=" * 70,
                "DATABASE SCHEMA — USE ONLY THESE EXACT NAMES (copy character-for-character)",
                "=" * 70]
    for tname, info in analysis_tables.items():
        sections.append(f'\nTABLE: "{tname}"  |  {info["rows"]:,} rows')
        sections.append("Columns (EXACT names, use double quotes):")
        samples = _live_samples(conn, tname, info["columns"])
        for col in info["columns"]:
            sv = samples.get(col, [])
            sections.append(f'  "{col}"' + (f"  → e.g. {sv}" if sv else ""))

    if len(analysis_tables) > 1:
        tnames = list(analysis_tables.keys())
        sections.append("\nJOIN RELATIONSHIPS:")
        for i in range(len(tnames)):
            for j in range(i+1, len(tnames)):
                t1, t2 = tnames[i], tnames[j]
                shared = set(analysis_tables[t1]["columns"]) & set(analysis_tables[t2]["columns"])
                for col in shared:
                    sections.append(f'  "{t1}" JOIN "{t2}" ON "{t1}"."{col}" = "{t2}"."{col}"')

    schema_block = "\n".join(sections)
    return f"""You are an expert SQLite analyst for Tesla HR data.
Convert natural-language questions into ONE read-only SELECT query.

{schema_block}

CRITICAL RULES:
1. Use ONLY the column names listed above, EXACTLY as shown, with DOUBLE QUOTES.
   - NEVER use backtick quoting (`col`) — only double quotes ("col")
   - NEVER invent names like "Job Level", "Applicant ID", "STAGE" — these don't exist
   - "job_level" NOT "Job Level" | "applicant_id" NOT "Applicant ID" | "stage" NOT "STAGE"
2. Default table for queries: "{primary_table}"
3. Gender values: 'F'=female, 'M'=male, 'Unknown'
4. Ethnicity: 'WH'=White, 'AS'=Asian, 'URM'=Under-represented minority, 'Unk'=unknown
5. Stages: 'Application', 'Recruiter Review', 'Interview', 'Offer', 'Hired'
6. is_hired: 1=hired, 0=not hired | max_stage_rank: 1–5
7. Output: ONLY raw SQL. No markdown, no backticks, no explanation.

CORRECT example:
  SELECT "job_level", COUNT(*) AS cnt
  FROM "{primary_table}"
  WHERE "gender" = 'F'
  GROUP BY "job_level"
  ORDER BY cnt DESC;
"""

def _build_retry_prompt(conn: sqlite3.Connection, failed_sql: str, error: str) -> str:
    all_schema = _live_schema(conn)
    valid = [f'"{t}"."{c}"' for t, info in all_schema.items() for c in info["columns"]]
    return f"""Fix this SQLite query.

VALID REFERENCES (double-quoted only, no backticks):
{chr(10).join(valid[:60])}

FAILED SQL: {failed_sql}
ERROR: {error}

FIX RULES:
- Replace ALL backtick quotes with double quotes
- "Job Level"→"job_level", "Applicant ID"→"applicant_id", "STAGE"→"stage"
- "Department Code"→"department_code", "Job Position Code"→"job_position_code"
- Use "gold_enriched" as the primary table if available
Return ONLY corrected SQL. No markdown, no backticks."""

# ── Auto-fix SQL ──────────────────────────────────────────────────────────────
def _auto_fix_sql(sql: str, conn: sqlite3.Connection) -> str:
    schema = _live_schema(conn)
    all_tables = list(schema.keys())
    flat: Dict[str, str] = {}  # lower → original
    for tname, info in schema.items():
        for col in info["columns"]: flat[col.lower()] = col

    FORCE_MAP = {
        "job level": "job_level", "applicant id": "applicant_id",
        "department code": "department_code", "job position code": "job_position_code",
        "target start date": "target_start_date", "date applied": "date_applied",
        "stage": "stage", "gender": "gender", "ethnicity": "ethnicity",
    }
    KEYWORD_MAP = {
        "joblevel": "job_level", "job": "job_level", "level": "job_level",
        "applicantid": "applicant_id", "candidateid": "applicant_id",
        "sex": "gender", "race": "ethnicity", "dept": "department_code",
    }

    def find_col(ident: str) -> Optional[str]:
        if ident in all_tables: return None
        lower = ident.lower()
        if lower in flat: return None  # already correct
        for orig, fixed in FORCE_MAP.items():
            if lower == orig and fixed in flat: return flat[fixed]
        kw = lower.replace("_","").replace(" ","")
        if kw in KEYWORD_MAP and KEYWORD_MAP[kw] in flat: return flat[KEYWORD_MAP[kw]]
        for col_lower, col_orig in flat.items():
            if col_lower.split("_")[0] == lower or lower in col_lower.split("_"): return col_orig
        m = get_close_matches(lower.replace("-","_"), list(flat.keys()), n=1, cutoff=0.5)
        return flat[m[0]] if m else None

    # Fix backtick-quoted identifiers
    def fix_bt(m):
        ident = m.group(1); lower = ident.lower()
        if lower in flat: return f'"{flat[lower]}"'
        r = find_col(ident)
        return f'"{r}"' if r else f'"{ident}"'
    sql = re.sub(r'`([^`]+)`', fix_bt, sql)

    # Fix alias.col references
    def fix_alias(m):
        alias, col = m.group(1), m.group(2).strip('"')
        lower = col.lower()
        if lower in flat: return f'{alias}."{flat[lower]}"'
        r = find_col(col)
        return f'{alias}."{r}"' if r else f'{alias}."{col}"'
    sql = re.sub(r'\b([A-Za-z_]\w*)\."?([A-Za-z_][A-Za-z0-9_ ]*)"?', fix_alias, sql)
    return sql

# ── Chart helpers ─────────────────────────────────────────────────────────────
CHART_KW = {
    "funnel": ["funnel","pipeline","conversion","drop off","stage breakdown"],
    "trend":  ["trend","over time","monthly","by month","by year","by quarter","timeline","time series"],
    "bar":    ["bar chart","bar graph","most","top","highest","lowest","ranking","compare"],
    "pie":    ["pie","proportion","share","percentage","distribution","breakdown","composition","split"],
    "scatter":["scatter","correlation","vs ","versus","relationship between"],
}

def detect_chart_intents(q: str) -> List[str]:
    q = q.lower()
    out = [ct for ct, pats in CHART_KW.items() if any(p in q for p in pats)]
    if not out and any(w in q for w in ("chart","graph","visual","plot","show")): out.append("auto")
    return out

def _is_numeric(rows: List[dict], col: str) -> bool:
    ok = 0
    for r in rows[:10]:
        v = str(r.get(col,"")).replace(",","").replace("$","").replace("%","").strip()
        if not v: continue
        try: float(v); ok += 1
        except ValueError: return False
    return ok > 0

def _is_date_col(col: str) -> bool:
    return any(k in col.lower() for k in ("date","month","year","quarter","period"))

def pick_charts(rows: List[dict], intents: List[str], question: str = "") -> List[dict]:
    if not rows: return []
    cols = list(rows[0].keys())
    num = [c for c in cols if _is_numeric(rows, c)]
    txt = [c for c in cols if c not in num]
    dtc = [c for c in cols if _is_date_col(c)]
    if not num: return []
    x = dtc[0] if dtc else (txt[0] if txt else cols[0])
    y = num[0]; y2 = num[1] if len(num) > 1 else None
    ty = y.replace("_"," ").title(); tx = x.replace("_"," ").title()
    charts: List[dict] = []; used: set = set()

    def add(cfg):
        if cfg["chart"] not in used and len(charts) < 3: charts.append(cfg); used.add(cfg["chart"])

    for intent in intents:
        if intent == "funnel": add({"chart":"funnel","x":x,"y":y,"title":f"Funnel — {ty}"})
        elif intent == "trend" and dtc: add({"chart":"line","x":dtc[0],"y":y,"title":f"{ty} Over Time"})
        elif intent == "bar": add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})
        elif intent == "pie" and len(rows)<=14: add({"chart":"pie","x":x,"y":y,"title":f"{ty} Distribution"})
        elif intent == "scatter" and y2: add({"chart":"scatter","x":y,"y":y2,"title":f"{ty} vs {y2.replace('_',' ').title()}"})
        elif intent == "auto":
            if len(rows)<=8: add({"chart":"pie","x":x,"y":y,"title":f"{ty} Distribution"})
            else: add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})

    if len(charts)<2 and len(rows)>=3:
        if "bar" not in used and txt: add({"chart":"bar","x":x,"y":y,"title":f"{ty} by {tx}"})
        if "line" not in used and dtc: add({"chart":"line","x":dtc[0],"y":y,"title":f"{ty} Over Time"})
        if "pie" not in used and len(rows)<=12 and txt: add({"chart":"pie","x":x,"y":y,"title":f"{ty} Share"})
    return charts[:3]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_money(col: str) -> bool:
    return any(k in col.lower() for k in ("salary","pay","wage","compensation","bonus","amount","cost","budget","revenue","price"))

def _safe_sql(sql: str) -> Tuple[bool, str]:
    s = sql.strip().lstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", s, re.IGNORECASE): return False, "Query must start with SELECT or WITH"
    m = BLOCKED.search(sql)
    if m: return False, f"Blocked keyword: {m.group().strip()}"
    if ";" in re.sub(r"'[^']*'", "''", sql).rstrip(";"): return False, "Multiple statements not allowed"
    return True, ""

def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if parsed.notna().mean() >= 0.5: return parsed
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.5: return parsed
    serial = numeric.dropna()
    if serial.empty or not serial.between(20000, 60000).mean() >= 0.8: return parsed
    parsed2 = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return parsed2 if parsed2.notna().mean() > parsed.notna().mean() else parsed

def _table_name_from(display_name: str) -> str:
    base = os.path.splitext(os.path.basename(display_name))[0]
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "data"

def list_tables(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    result = []
    for (n,) in rows:
        try: cnt = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
        except: cnt = "?"
        result.append({"name": n, "rows": cnt})
    return result

# ── LLM calls ─────────────────────────────────────────────────────────────────
def _chat(client, system: str, messages: List[dict], max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(model=MODEL, max_tokens=max_tokens,
        messages=[{"role":"system","content":system}] + messages)
    return resp.choices[0].message.content.strip()

def _clean_sql(raw: str) -> str:
    s = re.sub(r"^```(?:sql)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r'`([^`]+)`', r'"\1"', s)  # backticks → double quotes
    return s.strip()

def _generate_sql(question: str, history: List[dict], client, conn: sqlite3.Connection) -> str:
    system = _build_system_prompt(conn)
    msgs = []
    for t in history[-6:]:
        msgs.append({"role":"user","content":t["question"]})
        if t.get("sql"): msgs.append({"role":"assistant","content":t["sql"]})
    msgs.append({"role":"user","content":question})
    return _auto_fix_sql(_clean_sql(_chat(client, system, msgs)), conn)

def _retry_sql(question: str, failed_sql: str, error: str, client, conn: sqlite3.Connection) -> str:
    system = _build_retry_prompt(conn, failed_sql, error)
    return _auto_fix_sql(_clean_sql(_chat(client, system, [{"role":"user","content":question}])), conn)

def _generate_code(question: str, sql: str, col_names: List[str], client) -> str:
    CODE_SYS = "Write concise pandas equivalent of the SQL. df is pre-loaded. Import pd, px. Max 25 lines. End with `result`. Output ONLY raw Python."
    msg = f"Question: {question}\n\nSQL:\n{sql}\n\nColumns: {', '.join(col_names[:20])}"
    try:
        raw = _chat(client, CODE_SYS, [{"role":"user","content":msg}], max_tokens=500)
        raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", raw).strip()
    except: return ""

def _narrative(question: str, rows: List[dict], error: Optional[str], conn: sqlite3.Connection, client) -> str:
    NAR_SYS = "Senior HR analyst for Tesla. Write 2-4 sentences of professional insight. Lead with key numbers. No filler, no emojis."
    schema = _live_schema(conn)
    if error: ctx = f"Question: {question}\nError: {error}\nTables: {', '.join(schema.keys())}"
    elif not rows: ctx = f"Question: {question}\nResult: 0 rows.\nTables: {', '.join(schema.keys())}"
    else: ctx = f"Question: {question}\nResult ({len(rows)} rows): {rows[:5]}"
    try: return _chat(client, NAR_SYS, [{"role":"user","content":ctx}], max_tokens=220)
    except: return ""

# ── Public DataAgent ──────────────────────────────────────────────────────────
class DataAgent:
    def __init__(self, api_key: str | None = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key: raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY.")
        if _Client is None: raise RuntimeError("Install the `groq` or `openai` Python package.")
        self.client = _Client(api_key=key)
        self.history: List[dict] = []
        self.tables_info: Dict[str, dict] = {}
        self._raw_table_types: Dict[str, str] = {}

    def upload_file(self, display_name: str, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            try: sheets = pd.read_excel(file_path, sheet_name=None)
            except Exception as e: return {"error": f"Could not read Excel: {e}"}
            tables = []; total = 0
            for sname, df in sheets.items():
                if df is None or df.empty: continue
                info = self._ingest_df(f"{display_name}__{sname}", df)
                if info.get("table"): tables.append(info); total += info["rows_loaded"]
            if not tables: return {"error": "No non-empty sheets found."}
            result = {"tables": tables, "rows_loaded": total}
        else:
            try: df = pd.read_csv(file_path, low_memory=False)
            except Exception as e: return {"error": f"Could not read CSV: {e}"}
            if df.empty: return {"error": "File is empty"}
            info = self._ingest_df(display_name, df)
            result = {"tables": [info], "rows_loaded": info.get("rows_loaded", 0)}

        result["cleaning_summary"] = self._rebuild_clean_layers()
        return result

    def upload_csv(self, display_name: str, file_path: str) -> dict:
        return self.upload_file(display_name, file_path)

    def _ingest_df(self, display_name: str, df: pd.DataFrame) -> dict:
        original_rows = len(df)
        df_san, col_map = sanitise_columns(df)
        for col in list(df_san.columns):
            if any(k in col.lower() for k in ("date","time")):
                try:
                    parsed = _parse_dates(df_san[col])
                    if parsed.notna().mean() > 0.5: df_san[col] = parsed.dt.strftime("%Y-%m-%d")
                except: pass
        tname = _table_name_from(display_name)
        df_san.to_sql(tname, self.conn, if_exists="replace", index=False)
        table_type = _detect_table_type(list(df_san.columns))
        self._raw_table_types[tname] = table_type
        info = {"table": tname, "table_type": table_type, "original": display_name,
                "columns": list(df_san.columns), "rows_loaded": len(df_san),
                "original_rows": original_rows}
        self.tables_info[tname] = info
        return info

    def _rebuild_clean_layers(self) -> dict:
        summary = {}
        event_tables = [t for t, typ in self._raw_table_types.items() if typ == "raw_events"]
        demo_tables  = [t for t, typ in self._raw_table_types.items() if typ == "demographics"]

        if event_tables:
            name = _build_silver_applications(self.conn, event_tables[0])
            if name != event_tables[0]:
                cnt = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                summary["silver_applications"] = f"{cnt:,} unique applications (deduped from raw events)"
                self.tables_info[name] = {"table": name, "table_type": "silver", "rows_loaded": cnt}

        if demo_tables:
            name = _build_silver_demographics(self.conn, demo_tables[0])
            if name != demo_tables[0]:
                cnt = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                summary["silver_demographics"] = f"{cnt:,} unique applicants (conflicts resolved)"
                self.tables_info[name] = {"table": name, "table_type": "silver", "rows_loaded": cnt}

        tables_now = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "silver_applications" in tables_now and "silver_demographics" in tables_now:
            gold = _build_gold_enriched(self.conn)
            if gold:
                cnt = self.conn.execute(f'SELECT COUNT(*) FROM "{gold}"').fetchone()[0]
                summary["gold_enriched"] = f"{cnt:,} rows — main analysis table ✓"
                self.tables_info[gold] = {"table": gold, "table_type": "gold", "rows_loaded": cnt}

        return summary

    def remove_table(self, tname: str) -> bool:
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
            self.conn.commit()
            self.tables_info.pop(tname, None)
            self._raw_table_types.pop(tname, None)
            return True
        except: return False

    def get_schema_summary(self) -> List[dict]: return list_tables(self.conn)
    def has_data(self) -> bool: return bool(self.tables_info)

    def ask(self, question: str) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(question=question, sql=None, code=None,
            rows=[], money_columns=[], narrative="", charts=[], error=None)
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
        for _ in range(MAX_RETRIES):
            if not exec_error: break
            sql_retry = _retry_sql(question, out["sql"], exec_error, self.client, self.conn)
            rows_retry, err_retry = self._run_sql(sql_retry)
            out["sql"] = sql_retry
            if not err_retry: rows, exec_error = rows_retry, None
            else: exec_error = err_retry

        if exec_error:
            out["error"] = exec_error
            out["narrative"] = _narrative(question, [], exec_error, self.conn, self.client)
            return out

        out["rows"] = rows
        if rows: out["money_columns"] = [c for c in rows[0].keys() if _is_money(c)]
        all_cols = list(dict.fromkeys(c for info in self.tables_info.values() for c in info.get("columns", [])))
        out["code"] = _generate_code(question, out["sql"], all_cols, self.client)
        out["narrative"] = _narrative(question, rows, None, self.conn, self.client)
        if rows: out["charts"] = pick_charts(rows, intents, question)
        self.history.append({"question": question, "sql": out["sql"]})
        return out

    def _run_sql(self, sql: str) -> Tuple[List[dict], Optional[str]]:
        try:
            cur = self.conn.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()], None
            return [], None
        except sqlite3.Error as e: return [], str(e)

    def clear_history(self): self.history = []
