"""Tesla HR Intelligence — Generic Agent Core

- Sanitises column names on load (to lower snake_case)
- Applies canonical aliases so the LLM can find columns even if names differ
- Works with CSV and Excel (multi-sheet) uploads
- Passes EXACT column names prominently in every prompt
- Retries SQL up to 2× with the error message fed back to LLM
- Fuzzy-matches user terms to real column names before SQL generation
"""

from __future__ import annotations

import os
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# LLM client setup
# ──────────────────────────────────────────────────────────────────────────────

try:
    from groq import Groq as _Client  # type: ignore

    _CLIENT_TYPE = "groq"
except Exception:  # pragma: no cover
    try:
        from openai import OpenAI as _Client  # type: ignore

        _CLIENT_TYPE = "openai"
    except Exception:  # pragma: no cover
        _Client = None  # type: ignore
        _CLIENT_TYPE = "none"

MODEL = "llama-3.3-70b-versatile" if _CLIENT_TYPE == "groq" else "gpt-4o"

MAX_RETRIES = 2

BLOCKED = re.compile(
    r"\b("
    r"DROP\s+TABLE|DROP\s+VIEW|DELETE\s+FROM|UPDATE\s+\w+\s+SET|INSERT\s+INTO"
    r"|ALTER\s+TABLE|TRUNCATE|CREATE\s+TABLE|CREATE\s+VIEW|REPLACE\s+INTO"
    r"|ATTACH\s+DATABASE|DETACH\s+DATABASE"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

SQL_SYSTEM = """You are an expert SQLite analyst working for a Fortune 500 company.
Convert the user's natural-language question into a single read-only SELECT query.

{schema_block}

{column_hints}

STRICT RULES — FOLLOW EVERY ONE:
1. Output ONLY raw SQL — zero markdown, zero backticks, zero explanation.
2. ONLY use the EXACT table and column names listed above — copy them character-for-character.
   Double-check every column name in your query against the schema before outputting.
3. NEVER invent, guess, or hallucinate column names. The ONLY valid column names are the ones
   listed above. If a column is not in the schema, it does NOT exist.
4. When the user refers to a concept, map it to the CLOSEST matching column:
   - Think about synonyms, abbreviations, and natural language variations.
   - "level" / "job level" / "position level" → look for columns containing "level"
   - "applicant" / "candidate" / "person" → look for columns containing "applicant" or "id"
   - "date" / "when applied" / "application date" → look for columns containing "date"
   - "gender" / "male" / "female" → look for gender-related columns
   - "race" / "ethnicity" / "diversity" → look for ethnicity-related columns
   - "department" / "dept" / "team" / "division" → look for department-related columns
   - "conversion" / "funnel" / "pipeline" → use the stage column with COUNT/GROUP BY
5. Text filters: LOWER(col) LIKE LOWER('%value%') — always case-insensitive.
6. Percentages: ROUND(100.0 * part / total, 1)
7. CTEs are encouraged for multi-step logic (conversion funnels, comparisons, etc.).
8. ORDER BY a relevant column; add LIMIT where appropriate.
9. Always quote column names with double quotes if they could conflict with SQL keywords.
10. NEVER use CREATE / DROP / UPDATE / INSERT / DELETE — only safe SELECT queries.
11. For "how many" questions, use COUNT(*). For "unique" counts, use COUNT(DISTINCT col).
12. For "conversion funnel" or "pipeline": GROUP BY stage, count per stage, order by count DESC.
13. When filtering text values, check the sample values provided and use matching values.
"""

RETRY_SYSTEM = """You are an expert SQLite analyst. A query failed. Fix it.

{schema_block}

IMPORTANT — VALID COLUMN NAMES (use ONLY these):
{all_columns}

FAILED QUERY:
{failed_sql}

ERROR:
{error}

INSTRUCTIONS:
- The error is almost certainly a wrong column or table name.
- Compare every identifier in the failed query against the valid column names above.
- Replace any non-matching identifier with the correct one from the schema.
- Common fixes: "Level" → "job_level", "Applicant_ID" → "applicant_id",
  "Date_Applied" → "date_applied", "Status" → "stage"
- Do NOT introduce any non-SELECT statements.
- Output ONLY the corrected raw SQL. No markdown, no backticks, no explanation.
"""

NARRATIVE_SYSTEM = """You are a senior data analyst presenting findings to executives.
Write a clear, professional 2-4 sentence insight.

Guidelines:
- Data found: lead with the key number or trend, name specific values.
- Empty result: explain the filter matched nothing, suggest valid values from the data.
- Error: explain clearly in plain English without SQL jargon.
- Be direct, precise, and professional. No filler, no emojis, no casual language.
- Use proper business terminology."""

CODE_SYSTEM = """You are a Python data analyst.
Write a concise pandas + plotly snippet equivalent to the SQL provided.

Rules:
- `df` is already loaded — do NOT call pd.read_csv().
- Import pandas as pd and plotly.express as px at the top.
- Max 25 lines. Clean names, brief inline comments.
- End with `result` (DataFrame) and optionally `fig` (plotly figure).
- Output ONLY raw Python. No markdown, no backticks, no prose outside comments.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Column name sanitiser + aliases
# ──────────────────────────────────────────────────────────────────────────────


def _sanitise_col(name: str) -> str:
    """'Date Applied' → 'date_applied' | 'Job Level (Code)' → 'job_level_code'."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s or "col").lower()


def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    return df.rename(columns=mapping)


# Canonical column aliases after sanitisation (all lower snake_case)
ALIASES: Dict[str, set] = {
    "date_applied": {
        "date_applied", "applied_date", "application_date", "date",
        "dateapplied", "date_applied_", "apply_date", "date_of_application",
        "applied_on", "submission_date",
    },
    "applicant_id": {
        "applicant_id", "applicantid", "candidate_id", "candidateid",
        "person_id", "personid", "app_id", "employee_id", "emp_id",
        "id", "applicant_no", "candidate_no",
    },
    "stage": {
        "stage", "application_stage", "status", "hiring_stage",
        "recruitment_stage", "pipeline_stage", "step", "phase",
        "application_status", "candidate_status",
    },
    "job_level": {
        "job_level", "level", "joblevel", "position_level",
        "grade", "job_grade", "role_level", "seniority",
    },
    "department_code": {
        "department_code", "departmentid", "dept_code", "department",
        "dept", "department_id", "dept_id", "division_code",
        "team_code", "business_unit",
    },
    "job_position_code": {
        "job_position_code", "jobcode", "job_position", "position_code",
        "job_code", "role_code", "position", "job_id", "role_id",
        "job_title_code", "position_id",
    },
    "target_start_date": {
        "target_start_date", "start_date", "startdate",
        "expected_start_date", "planned_start_date", "hire_date",
        "joining_date", "onboarding_date",
    },
    "gender": {"gender", "sex", "gender_code", "m_f"},
    "ethnicity": {
        "ethnicity", "race", "ethnic_group", "race_ethnicity",
        "demographic", "ethnic_background",
    },
}


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    rename: Dict[str, str] = {}

    for canonical, variants in ALIASES.items():
        if canonical in cols:
            continue
        hit = next((c for c in cols if c in variants), None)
        if hit:
            rename[hit] = canonical
            cols.remove(hit)
            cols.add(canonical)

    return df.rename(columns=rename) if rename else df


# ──────────────────────────────────────────────────────────────────────────────
# Fuzzy column matching & hint generation
# ──────────────────────────────────────────────────────────────────────────────

SEMANTIC_MAP: Dict[str, List[str]] = {
    "gender": ["male", "female", "women", "woman", "men", "man", "gender", "sex", "m_f"],
    "ethnicity": ["race", "ethnicity", "ethnic", "diversity", "minority", "hispanic",
                   "asian", "african", "caucasian", "white", "black", "latino", "latina"],
    "department": ["department", "dept", "team", "division", "unit", "group", "org"],
    "stage": ["stage", "status", "pipeline", "funnel", "conversion", "step", "phase",
              "hired", "rejected", "screening", "interview", "offer", "applied"],
    "date": ["date", "when", "time", "period", "year", "month", "quarter", "day"],
    "salary": ["salary", "pay", "wage", "compensation", "income", "earning", "money"],
    "level": ["level", "seniority", "grade", "tier", "rank", "position"],
    "applicant": ["applicant", "candidate", "person", "employee", "individual", "worker"],
    "job": ["job", "role", "position", "title", "occupation", "work"],
    "location": ["location", "city", "state", "country", "region", "office", "site"],
    "age": ["age", "old", "young", "senior", "junior", "years"],
    "education": ["education", "degree", "school", "university", "college", "qualification"],
    "experience": ["experience", "years", "tenure", "seniority"],
}


def _normalise_term(term: str) -> str:
    return re.sub(r"[^a-z0-9]", "", term.lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _depluralize(word: str) -> List[str]:
    """Return candidate singular forms for matching purposes."""
    forms = [word]
    if word.endswith("ies") and len(word) > 4:
        forms.append(word[:-3] + "y")
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes") \
       or word.endswith("ches") or word.endswith("shes"):
        forms.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        forms.append(word[:-1])
    return forms


def _semantic_column_match(term: str, columns: List[str]) -> Optional[str]:
    """Use semantic keyword mapping to find a matching column."""
    norm = term.lower().strip()
    candidates = _depluralize(norm)
    matched_concept = None

    for concept, keywords in SEMANTIC_MAP.items():
        if any(c in keywords for c in candidates):
            matched_concept = concept
            break
        if any(c == concept for c in candidates):
            matched_concept = concept
            break
    if not matched_concept:
        return None

    concept_keywords = SEMANTIC_MAP[matched_concept]
    for col in columns:
        col_lower = col.lower()
        col_parts = col_lower.split("_")
        if matched_concept in col_lower or any(matched_concept == p for p in col_parts):
            return col
        for kw in concept_keywords[:4]:
            if kw in col_parts or kw == col_lower:
                return col

    return None


def _fuzzy_match_column(term: str, columns: List[str], threshold: float = 0.55) -> Optional[str]:
    """Find the best-matching column for a natural-language term.

    Checks semantic meaning first, then falls back to string similarity.
    """
    norm_term = _normalise_term(term)
    if not norm_term:
        return None

    for col in columns:
        if norm_term == _normalise_term(col):
            return col

    semantic = _semantic_column_match(term, columns)
    if semantic:
        return semantic

    best_col = None
    best_score = 0.0

    for col in columns:
        norm_col = _normalise_term(col)
        parts = col.lower().split("_")

        if norm_term in norm_col or norm_col in norm_term:
            score = 0.85
        elif any(norm_term == _normalise_term(p) for p in parts):
            score = 0.8
        elif any(norm_term in _normalise_term(p) or _normalise_term(p) in norm_term for p in parts):
            score = 0.7
        else:
            score = _similarity(norm_term, norm_col)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score >= threshold else None


_STOP_WORDS = frozenset({
    "the", "what", "how", "many", "show", "get", "find",
    "are", "there", "have", "has", "with", "from", "for",
    "and", "not", "all", "each", "per", "can", "you",
    "give", "tell", "about", "this", "that", "which",
    "where", "when", "who", "why", "does", "did", "will",
    "was", "were", "been", "being", "would", "could",
    "should", "shall", "may", "might", "must", "our",
    "their", "your", "its", "his", "her", "any", "some",
    "top", "bottom", "first", "last", "by", "as", "in",
    "on", "to", "of", "is", "it", "if", "do", "me",
    "chart", "graph", "bar", "pie", "line", "plot",
    "table", "data", "rows", "count", "total", "sum",
    "average", "avg", "min", "max", "unique", "distinct",
    "please", "need", "want", "see", "look", "like",
    "also", "just", "really", "very", "much", "more",
    "than", "then", "only", "also", "but", "or", "so",
    "up", "down", "out", "off", "over", "into",
})


def _build_column_hints(question: str, tables_info: dict) -> str:
    """Extract potential column references from the question and map to real columns."""
    all_cols = []
    for info in tables_info.values():
        all_cols.extend(info.get("columns", []))
    all_cols = list(dict.fromkeys(all_cols))

    if not all_cols:
        return ""

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", question.lower())
    candidates = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]

    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in _STOP_WORDS and words[i + 1] not in _STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i + 1]}")

    hints = []
    seen = set()

    for phrase in bigrams + candidates:
        match = _fuzzy_match_column(phrase, all_cols)
        if match and match not in seen:
            hints.append(f'  - User said "{phrase}" -> use column "{match}"')
            seen.add(match)

    if not hints:
        return ""

    return "COLUMN MAPPING HINTS (based on the user's question):\n" + "\n".join(hints)


def _get_all_column_list(tables_info: dict) -> str:
    """Build a flat, readable list of every valid column across all tables."""
    parts = []
    for tname, info in tables_info.items():
        cols = ", ".join(f'"{c}"' for c in info.get("columns", []))
        parts.append(f'  Table "{tname}": {cols}')
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────

CHART_KW = {
    "funnel": [
        "funnel",
        "pipeline",
        "conversion",
        "drop off",
        "stage breakdown",
        "dropout",
    ],
    "trend": [
        "trend",
        "over time",
        "monthly",
        "by month",
        "by year",
        "by quarter",
        "timeline",
        "time series",
        "per month",
        "per year",
        "growth",
    ],
    "bar": [
        "bar chart",
        "bar graph",
        "most",
        "top",
        "highest",
        "lowest",
        "ranking",
        "rank",
        "compare",
        "histogram",
    ],
    "pie": [
        "pie",
        "proportion",
        "share",
        "percentage",
        "distribution",
        "breakdown",
        "composition",
        "split",
    ],
    "scatter": [
        "scatter",
        "correlation",
        "vs ",
        "versus",
        "relationship between",
    ],
}


def detect_chart_intents(q: str) -> list[str]:
    q = q.lower()
    out = [ct for ct, pats in CHART_KW.items() if any(p in q for p in pats)]
    if not out and any(w in q for w in ("chart", "graph", "visual", "plot", "show")):
        out.append("auto")
    return out


def _is_numeric(rows, col) -> bool:
    ok = 0
    for r in rows[:10]:
        v = (
            str(r.get(col, ""))
            .replace(",", "")
            .replace("$", "")
            .replace("%", "")
            .strip()
        )
        if not v:
            continue
        try:
            float(v)
            ok += 1
        except ValueError:
            return False
    return ok > 0


def _is_date_col(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in ("date", "month", "year", "quarter", "period", "time"))


def pick_charts(rows, intents, question: str = "") -> list[dict]:
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
            add(
                {
                    "chart": "scatter",
                    "x": y,
                    "y": y2,
                    "title": f"{ty} vs {y2.replace('_', ' ').title()}",
                }
            )
        elif intent == "auto":
            if len(rows) <= 8:
                add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Distribution"})
            else:
                add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})

    # Fallbacks if user didn’t specify explicit chart type
    if len(charts) < 2 and len(rows) >= 3:
        if "bar" not in used and txt:
            add({"chart": "bar", "x": x, "y": y, "title": f"{ty} by {tx}"})
        if "line" not in used and dtc:
            add({"chart": "line", "x": dtc[0], "y": y, "title": f"{ty} Over Time"})
        if "pie" not in used and len(rows) <= 12 and txt:
            add({"chart": "pie", "x": x, "y": y, "title": f"{ty} Share"})

    return charts[:3]


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────


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
        # Heuristic: if most values are integers, treat as INTEGER
        return "REAL" if (num.dropna() % 1 != 0).mean() > 0.05 else "INTEGER"

    try:
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.8:
            # stored as ISO string
            return "TEXT"
    except Exception:
        pass

    return "TEXT"


def _is_money(col: str) -> bool:
    c = col.lower()
    return any(
        k in c
        for k in (
            "salary",
            "pay",
            "wage",
            "compensation",
            "bonus",
            "amount",
            "cost",
            "budget",
            "usd",
            "revenue",
            "total",
            "price",
        )
    )


def _safe_sql(sql: str):
    s = sql.strip().lstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", s, re.IGNORECASE):
        return False, "Query must start with SELECT or WITH"

    m = BLOCKED.search(sql)
    if m:
        return False, f"Blocked keyword: {m.group().strip()}"

    if ";" in re.sub(r"'[^']*'", "''", sql).rstrip(";"):
        return False, "Multiple statements not allowed"

    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# Schema block builder
# ──────────────────────────────────────────────────────────────────────────────


def _schema_block(tables_info: dict) -> str:
    if not tables_info:
        return "No tables loaded.\nAsk user to upload a CSV or Excel file first."

    blocks: List[str] = []

    for tname, info in tables_info.items():
        lines = [
            "=" * 60,
            f'TABLE: "{tname}"  |  ROWS: {info["rows_loaded"]:,}',
            "=" * 60,
            "",
            "COLUMNS (use ONLY these exact names — copy character-for-character):",
            "",
        ]

        col_name_map = info.get("col_name_map", {})
        reverse_map = {v: k for k, v in col_name_map.items()} if col_name_map else {}

        for col, dtype in info["col_types"].items():
            orig = reverse_map.get(col)
            orig_note = f'  (original: "{orig}")' if orig and orig != col else ""
            sv = info["sample_values"].get(col)
            if sv:
                sample_str = ", ".join(repr(v) for v in sv[:6])
                lines.append(f'  "{col}"  ({dtype}){orig_note}  — samples: [{sample_str}]')
            else:
                lines.append(f'  "{col}"  ({dtype}){orig_note}')
        lines.append("")

        for hint in info.get("join_hints", []):
            lines.append(f"JOIN HINT: {hint}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────────────────────


def _chat(client, system: str, messages: list, max_tokens: int = 700) -> str:
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


def _generate_sql(question, history, client, tables_info) -> str:
    schema = _schema_block(tables_info)
    col_hints = _build_column_hints(question, tables_info)
    system = SQL_SYSTEM.format(schema_block=schema, column_hints=col_hints)

    msgs: List[Dict[str, str]] = []
    for t in history[-6:]:
        msgs.append({"role": "user", "content": t["question"]})
        if t.get("sql"):
            msgs.append({"role": "assistant", "content": t["sql"]})

    msgs.append({"role": "user", "content": question})
    return _clean_sql(_chat(client, system, msgs))


def _retry_sql(question, failed_sql, error, client, tables_info) -> str:
    schema = _schema_block(tables_info)
    all_cols = _get_all_column_list(tables_info)
    system = RETRY_SYSTEM.format(
        schema_block=schema,
        all_columns=all_cols,
        failed_sql=failed_sql,
        error=error,
    )
    return _clean_sql(
        _chat(client, system, [{"role": "user", "content": question}])
    )


def _generate_code(question, sql, col_names, client) -> str:
    col_str = ", ".join(col_names[:20])
    msg = (
        f"Question: {question}\n\nSQL:\n{sql}\n\n"
        f"DataFrame `df` columns: {col_str}\n\nWrite equivalent pandas code."
    )

    try:
        raw = _chat(
            client, CODE_SYSTEM, [{"role": "user", "content": msg}], max_tokens=500
        )
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
                hints.append(f"{col}: {vals}")
        ctx = (
            f"Question: {question}\nResult: 0 rows.\n"
            f"Tables: {tnames}\nSample values: {'; '.join(hints)}"
        )
    else:
        ctx = f"Question: {question}\nResult ({len(rows)} rows): {rows[:5]}"

    try:
        return _chat(
            client, NARRATIVE_SYSTEM, [{"role": "user", "content": ctx}], max_tokens=220
        )
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# CSV / Excel → SQLite loader
# ──────────────────────────────────────────────────────────────────────────────


def _parse_dates(series: pd.Series) -> pd.Series:
    """Robust date parser that also handles Excel serial dates."""
    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    # Good enough?
    if parsed.notna().mean() >= 0.5:
        return parsed

    # Try Excel serials (days since 1899-12-30, typically between 20000–60000)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.5:
        return parsed

    serial = numeric.dropna()
    if serial.empty:
        return parsed

    mask = serial.between(20000, 60000)
    if mask.mean() < 0.8:
        return parsed

    parsed2 = pd.to_datetime(
        numeric, unit="D", origin="1899-12-30", errors="coerce"
    )
    if parsed2.notna().mean() > parsed.notna().mean():
        return parsed2
    return parsed


def _load_dataframe(
    conn: sqlite3.Connection, display_name: str, df: pd.DataFrame
) -> dict:
    if df is None or df.empty:
        return {"error": "File appears to be empty", "table": None}

    original_columns = list(df.columns)

    # 1. Sanitise + apply aliases
    df = sanitise_columns(df)
    df = apply_aliases(df)

    col_name_map = dict(zip(original_columns, df.columns))

    # 2. Table name from display_name
    base = os.path.splitext(os.path.basename(display_name))[0]
    tname = (
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", base)).strip("_") or "data"
    )

    # 3. Detect types
    col_types: Dict[str, str] = {}
    for col in df.columns:
        col_types[col] = _infer_type(df[col])

    # 4. Auto-add year/month/quarter for date columns
    for col in list(df.columns):
        if _is_date_col(col):
            try:
                parsed = _parse_dates(df[col])
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

    # 6. Sample values for all columns
    sample_values: Dict[str, list] = {}
    for col in df.columns:
        dtype = col_types.get(col, "TEXT")
        if dtype in ("TEXT", "TEXT (ISO date)"):
            vals = [str(v) for v in df[col].dropna().unique()[:8]]
            if vals:
                sample_values[col] = vals
        elif dtype in ("INTEGER", "REAL"):
            vals = [str(v) for v in df[col].dropna().unique()[:5]]
            if vals:
                sample_values[col] = vals

    # 7. Join hints
    join_hints = _detect_joins(conn, tname, df)

    return {
        "table": tname,
        "original": display_name,
        "columns": list(df.columns),
        "col_types": col_types,
        "col_name_map": col_name_map,
        "rows_loaded": int(len(df)),
        "sample_values": sample_values,
        "join_hints": join_hints,
    }


def load_tabular(conn: sqlite3.Connection, display_name: str, file_path: str) -> dict:
    """Load CSV or Excel into one or more SQLite tables.

    Returns:
        {"tables": [table_info, ...], "rows_loaded": total_rows}
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Excel → one table per sheet
    if ext in (".xlsx", ".xls"):
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            return {"error": f"Could not read Excel file: {e}"}

        tables: List[dict] = []
        total_rows = 0
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            info = _load_dataframe(conn, f"{display_name}__{sheet_name}", df)
            if info.get("table"):
                tables.append(info)
                total_rows += info["rows_loaded"]

        if not tables:
            return {"error": "No non-empty sheets found in Excel file."}

        return {"tables": tables, "rows_loaded": total_rows}

    # Default: CSV / text
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    if df.empty:
        return {"error": "File appears to be empty"}

    info = _load_dataframe(conn, display_name, df)
    return {"tables": [info], "rows_loaded": info["rows_loaded"]}


# Backwards-compatible helper (used by older code paths)
def load_csv(conn: sqlite3.Connection, display_name: str, file_path: str) -> dict:
    """Legacy wrapper for CSV-only behaviour.

    For CSV files it returns a single-table dict (historical shape).
    For Excel files, it returns the first loaded sheet as the primary table.
    """
    res = load_tabular(conn, display_name, file_path)
    if "error" in res:
        return res

    tables = res.get("tables", [])
    if not tables:
        return {"error": "No tables created."}

    if len(tables) == 1:
        return tables[0]

    primary = dict(tables[0])  # shallow copy
    others = ", ".join(t["table"] for t in tables[1:])
    primary["join_hints"] = primary.get("join_hints", []) + [
        f"Additional tables from same file: {others}"
    ]
    return primary


def _detect_joins(conn: sqlite3.Connection, new_table: str, df: pd.DataFrame) -> list[str]:
    hints: List[str] = []
    others: Dict[str, list] = {}

    for (n,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%';"
    ):
        cols = conn.execute(f'PRAGMA table_info("{n}")').fetchall()
        others[n] = [r[1] for r in cols]

    for col in df.columns:
        nu = df[col].nunique(dropna=True)
        if nu == 0 or nu > 100000:
            continue

        sample = [
            v.item() if hasattr(v, "item") else v
            for v in pd.unique(df[col].dropna())[:200]
        ]

        for ot, oc in others.items():
            if ot == new_table or col not in oc:
                continue
            ph = ",".join(["?"] * len(sample))
            try:
                (m,) = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{ot}" '
                    f'WHERE "{col}" IN ({ph})',
                    sample,
                ).fetchone()
                ratio = m / max(len(sample), 1)
                if ratio >= 0.6:
                    hints.append(
                        f'"{new_table}"."{col}" ~ "{ot}"."{col}" '
                        f"({ratio:.0%} overlap)"
                    )
            except Exception:
                pass

    return hints


def list_tables(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    result: List[dict] = []
    for (n,) in rows:
        try:
            cnt = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
        except Exception:
            cnt = "?"
        result.append({"name": n, "rows": cnt})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public Agent class
# ──────────────────────────────────────────────────────────────────────────────


class DataAgent:
    def __init__(self, api_key: str | None = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        key = (
            api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY.")

        if _Client is None:
            raise RuntimeError(
                "Install the `groq` or `openai` Python package to use DataAgent."
            )

        self.client = _Client(api_key=key)  # type: ignore[call-arg]
        self.history: list[dict] = []
        self.tables_info: dict[str, dict] = {}

    # ── File management ────────────────────────────────────────────────

    def upload_file(self, display_name: str, file_path: str) -> dict:
        """Generic upload handler for CSV or Excel."""
        info = load_tabular(self.conn, display_name, file_path)
        if "error" in info:
            return info

        for t in info.get("tables", []):
            self.tables_info[t["table"]] = t
        return info

    def upload_csv(self, display_name: str, file_path: str) -> dict:
        """Backward-compatible wrapper.

        Historically this only worked for CSV; it now simply delegates to
        `upload_file` and, if exactly one table is created, returns the single-
        table dict like before.
        """
        info = self.upload_file(display_name, file_path)
        if "error" in info:
            return info

        tables = info.get("tables")
        if isinstance(tables, list) and len(tables) == 1:
            return tables[0]
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
        out: dict[str, Any] = dict(
            question=question,
            sql=None,
            code=None,
            rows=[],
            money_columns=[],
            narrative="",
            charts=[],
            error=None,
        )

        if not self.tables_info:
            out["narrative"] = (
                "No data loaded yet. Upload a CSV or Excel file from the sidebar."
            )
            return out

        intents = detect_chart_intents(question)

        sql = _generate_sql(question, self.history, self.client, self.tables_info)
        out["sql"] = sql

        ok, reason = _safe_sql(sql)
        if not ok:
            out["error"] = reason
            out["narrative"] = _narrative(
                question, [], reason, self.tables_info, self.client
            )
            return out

        rows, exec_error = self._run_sql(sql)

        for _attempt in range(MAX_RETRIES):
            if not exec_error:
                break
            sql_retry = _retry_sql(
                question, out["sql"], exec_error, self.client, self.tables_info
            )
            rows_retry, err_retry = self._run_sql(sql_retry)
            if not err_retry:
                out["sql"] = sql_retry
                rows, exec_error = rows_retry, None
            else:
                out["sql"] = sql_retry
                exec_error = err_retry

        if exec_error:
            out["error"] = exec_error
            out["narrative"] = _narrative(
                question, [], exec_error, self.tables_info, self.client
            )
            return out

        out["rows"] = rows

        if rows:
            out["money_columns"] = [
                c for c in rows[0].keys() if _is_money(c)
            ]

        all_cols: List[str] = []
        for info in self.tables_info.values():
            all_cols.extend(info["columns"])
        out["code"] = _generate_code(
            question, out["sql"], list(dict.fromkeys(all_cols)), self.client
        )

        out["narrative"] = _narrative(
            question, rows, None, self.tables_info, self.client
        )

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
