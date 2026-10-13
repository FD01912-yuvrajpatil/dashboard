"""
Synthesis + Self-check  (Option B, Phases 3 & 5)
================================================

Two jobs:

1. UNIFY  — take whichever branch ran (retrieval via Genie, or prediction via
   the TabPFN chain) and produce ONE consistent envelope the front-end renders
   the same way. This is the spec's POST /query contract (page 5):
       { answer, sql?, table?, prediction?, uncertainty?, chart? }

2. SELF-CHECK — a light, deterministic check that the answer actually addresses
   the question (the spec's Synthesis self-check). It never blocks the answer;
   it annotates it with a confidence note so the UI can flag a weak match. An
   optional LLM self-check can be enabled, but the default is dependency-free.

Env:
  SELFCHECK_MODE   off | heuristic | llm     (default heuristic)
"""

import os
import re
import logging

log = logging.getLogger("ernest.synthesis")

SELFCHECK_MODE = os.environ.get("SELFCHECK_MODE", "heuristic").lower()

_STOP = set("""a an the of for to in on at by and or is are was were be do does did
we our us you your it this that with from as how many much what which show me
list total sum count average number over next last did get had have please tell
""".split())


def _as_text(v):
    """Coerce any answer-ish value to a clean string so the UI never shows
    '[object Object]'. Mirrors the backend's text coercion as defense in depth."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("content", "text", "value", "string_value", "message", "answer"):
            if k in v:
                got = _as_text(v[k])
                if got:
                    return got
        return ""
    if isinstance(v, (list, tuple)):
        return "\n".join(s for s in (_as_text(x) for x in v) if s)
    return str(v)


# ---------------------------------------------------------------------------
# UNIFY
# ---------------------------------------------------------------------------
def unify(question: str, kind: str, payload: dict, routing: dict) -> dict:
    """
    kind == 'retrieval'  -> payload is the genie_ask() result
    kind == 'prediction' -> payload is the run_prediction() result
    Returns the unified envelope.
    """
    log.info("payload", payload)
    if kind == "prediction":
        env = _unify_prediction(question, payload)
    else:
        env = _unify_retrieval(question, payload)

    env["kind"] = kind
    env["routing"] = routing
    env["question"] = question

    # self-check annotates (never blocks)
    env["self_check"] = self_check(question, env)
    return env


def _unify_retrieval(question, g):
    g = g or {}
    columns = g.get("columns") or []
    rows = g.get("rows") or []
    sql = g.get("sql") or ""
    viz = g.get("viz")            # {title, query_attachment_id, suggested} or None

    # Option B (realistic + SQL-aware): Genie's conversation API returns NO
    # chart-type/encoding spec — its viz attachment carries only a title and the
    # id of the query it came from. So we treat viz as a *signal* ("Genie thinks
    # this is worth charting") and borrow its title, then INFER the chart type and
    # axes from the data shape and the generated SQL. The envelope is the same one
    # drawRetrievalChart() already renders, so the front-end is unchanged.
    chart = _retrieval_chart(question, columns, rows, sql=sql, viz=viz)

    return {
        "answer": _as_text(g.get("text")),
        #"sql": g.get("sql"),
        "table": {
            "columns": columns,
            "rows": rows,
        } if rows else None,
        "prediction": None,
        "uncertainty": None,
        # chart payload the front-end's drawRetrievalChart() understands.
        # None when the result isn't chartable.
        "chart": chart,
        "engine": "genie",
    }


# --- SQL-aware helpers (B-plus) --------------------------------------------
_SQL_GROUPBY_RE = re.compile(r"\bGROUP\s+BY\b(.+?)(?:\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)",
                             re.IGNORECASE | re.DOTALL)
_SQL_ORDERBY_RE = re.compile(r"\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|$)", re.IGNORECASE | re.DOTALL)
_SQL_HAS_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _sql_group_cols(sql):
    """Column/expression names appearing in GROUP BY (the natural label axis)."""
    if not sql:
        return []
    m = _SQL_GROUPBY_RE.search(sql)
    if not m:
        return []
    return _split_sql_cols(m.group(1))


def _sql_order_desc(sql):
    """True if the query is ordered descending — a ranking, best shown as bars."""
    if not sql:
        return False
    m = _SQL_ORDERBY_RE.search(sql)
    return bool(m and re.search(r"\bDESC\b", m.group(1), re.IGNORECASE))


def _split_sql_cols(fragment):
    """Pull bare column names out of a GROUP BY / ORDER BY fragment."""
    cols = []
    for part in fragment.split(","):
        p = part.strip()
        # drop ASC/DESC and table qualifiers; ignore pure ordinals (GROUP BY 1)
        p = re.sub(r"\b(ASC|DESC)\b", "", p, flags=re.IGNORECASE).strip()
        p = p.split()[0] if p.split() else p
        if "." in p:
            p = p.split(".")[-1]
        p = p.strip("`\"[]()")
        if p and not p.isdigit():
            cols.append(p)
    return cols


def _retrieval_chart(question, columns, rows, sql="", viz=None):
    """
    Build a chart spec from a retrieval result when it is chartable: one usable
    label column + at least one numeric column, with a small-enough row count.
    Returns None otherwise so the card just shows the table.

    Option B inputs:
      sql  — Genie's generated SQL. Its GROUP BY picks the label axis and its
             ORDER BY ... DESC signals a ranking (→ bar). (B-plus)
      viz  — Genie's viz attachment {title, suggested}. Used only as a signal
             that Genie wanted a chart, and to borrow its title. (B-realistic)

    Shape (mirrors the forecast 'chart' contract — a self-describing dict):
        { "type": "bar", "labels": [...], "series": [{"name","values"}],
          "label_col": str, "value_cols": [...], "money": bool, "title": str,
          "source": "genie"|"heuristic" }
    """
    if not rows:
        return None

    cols = list(columns) if columns else list(rows[0].keys())
    if not cols:
        return None

    # Classify columns as numeric vs text by sampling the rows.
    sample = rows[:50]
    numeric_cols, text_cols = [], []
    for c in cols:
        vals = [r.get(c) for r in sample if r.get(c) is not None and r.get(c) != ""]
        if not vals:
            continue
        if sum(1 for v in vals if _is_number(v)) >= max(1, int(0.8 * len(vals))):
            numeric_cols.append(c)
        else:
            text_cols.append(c)

    # A numeric column can still be the time axis (e.g. a bare year like 2023).
    # If there's no text label but a numeric column reads as temporal, promote it
    # to the label axis so a year-over-year result charts as a line.
    if not text_cols and numeric_cols:
        for c in list(numeric_cols):
            col_vals = [str(r.get(c)) for r in sample if r.get(c) not in (None, "")]
            if _looks_temporal(c, col_vals):
                numeric_cols.remove(c)
                text_cols.append(c)
                break

    if not numeric_cols or not text_cols:
        return None

    # Too many bars is unreadable; too few rows isn't worth a chart.
    n = len(rows)
    if n < 2 or n > 30:
        return None

    # B-plus: if Genie's SQL grouped by a column we have, use it as the label
    # axis — that's the dimension Genie actually aggregated over. Else first text.
    group_cols = [c for c in _sql_group_cols(sql) if c in text_cols]
    label_col = group_cols[0] if group_cols else text_cols[0]
    # Numeric measures, label column excluded if it was promoted (e.g. a year).
    value_cols = [c for c in numeric_cols if c != label_col][:3]
    if not value_cols:
        return None

    # A distinct label per row keeps the bar chart meaningful.
    labels = [str(r.get(label_col, "")) for r in rows]
    if len(set(labels)) < 2:
        return None

    series = [{
        "name": vc.replace("_", " "),
        "values": [_to_number(r.get(vc)) for r in rows],
    } for vc in value_cols]

    money = any(_looks_money(vc) for vc in value_cols)

    # Chart type: a time/period axis reads best as a line (a trend). But an
    # explicitly ranked query (ORDER BY ... DESC, often with LIMIT) is a
    # top-N comparison — keep that as bars even if the labels look date-ish.
    temporal = _looks_temporal(label_col, labels)
    ranked = _sql_order_desc(sql)
    chart_type = "line" if (temporal and not ranked) else "bar"

    # Title: borrow Genie's viz title when it gave one; else synthesize.
    genie_title = (viz or {}).get("title") if isinstance(viz, dict) else None
    connector = " over " if chart_type == "line" else " by "
    title = (genie_title
             or value_cols[0].replace("_", " ") + connector + label_col.replace("_", " "))

    # Provenance: 'genie' when Genie suggested the viz, else our heuristic.
    source = "genie" if (isinstance(viz, dict) and viz.get("suggested")) else "heuristic"

    return {
        "type": chart_type,
        "labels": labels,
        "series": series,
        "label_col": label_col,
        "value_cols": value_cols,
        "money": money,
        "title": title,
        "source": source,
    }


def _is_number(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _to_number(v):
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _looks_money(col):
    c = (col or "").lower()
    return any(k in c for k in ("revenue", "amount", "cost", "gtm", "price", "sales", "$", "dollar", "spend"))


_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec")
# YYYY-MM / YYYY/MM, YYYY-MM-DD, Qn YYYY / YYYY Qn, or a bare 4-digit year.
_DATEISH_RE = re.compile(
    r"^\s*(\d{4}[-/]\d{1,2}([-/]\d{1,2})?|q[1-4][\s-]?\d{2,4}|\d{2,4}[\s-]?q[1-4]|\d{4})\s*$",
    re.IGNORECASE,
)


def _looks_temporal(label_col, labels):
    """
    True when the label axis is a time/period axis, so the chart should be a
    line. Decided by the column name OR by the actual label values (month names,
    YYYY-MM, quarters, years). Value-based detection means it works even when the
    column is named generically (e.g. 'period').
    """
    name = (label_col or "").lower()
    if any(k in name for k in ("month", "date", "period", "quarter", "year", "week", "day", "time")):
        return True

    vals = [str(v).strip().lower() for v in (labels or []) if str(v).strip()]
    if not vals:
        return False

    def is_time_value(v):
        if v[:3] in _MONTHS:            # 'Jan', 'January', 'Mar-2024'
            return True
        return bool(_DATEISH_RE.match(v))

    hits = sum(1 for v in vals if is_time_value(v))
    return hits >= max(2, int(0.6 * len(vals)))


def _unify_prediction(question, p):
    p = p or {}
    last = (p.get("forecast") or [{}])[-1]
    uncertainty = None
    if last:
        uncertainty = {
            "lower": last.get("lower"),
            "upper": last.get("upper"),
            "central": last.get("point"),
            "as_of": last.get("month"),
            "band": "80%",
        }
    # chart payload the front-end's drawForecastChart() already understands
    chart = {
        "history": p.get("history") or [],
        "forecast": p.get("forecast") or [],
        "target_label": p.get("target_label"),
        "money": p.get("money"),
    }
    return {
        "answer": _as_text(p.get("answer")),
        #"sql": None,
        "table": None,
        "prediction": {
            "target": p.get("target"),
            "target_label": p.get("target_label"),
            "scope": p.get("scope"),
            "entity": p.get("entity"),
            "horizon": p.get("horizon"),
            "forecast": p.get("forecast") or [],
        },
        "uncertainty": uncertainty,
        "chart": chart,
        "engine": p.get("engine"),
    }


# ---------------------------------------------------------------------------
# SELF-CHECK
# ---------------------------------------------------------------------------
def self_check(question: str, env: dict) -> dict:
    """
    Returns {status: 'ok'|'weak'|'empty', note: str, score: float}.
    Heuristic by default; LLM if SELFCHECK_MODE=llm and a key is present.
    """
    if SELFCHECK_MODE == "off":
        return {"status": "ok", "note": "", "score": 1.0}

    answer = (env.get("answer") or "").strip()
    has_table = bool(env.get("table") and env["table"].get("rows"))
    has_pred = bool(env.get("prediction"))

    # empty result
    if not answer and not has_table and not has_pred:
        return {"status": "empty",
                "note": "No answer or data was returned for this question.",
                "score": 0.0}

    if SELFCHECK_MODE == "llm":
        llm = _llm_self_check(question, answer, env)
        if llm is not None:
            return llm
        # fall through to heuristic if LLM unavailable

    # --- heuristic: does the answer engage the question's key terms? ---
    q_terms = _keywords(question)
    a_text = answer.lower()
    if has_pred:
        # predictions restate target/entity; treat as engaged if either appears
        p = env["prediction"]
        for extra in (p.get("target_label"), p.get("entity")):
            if extra:
                a_text += " " + str(extra).lower()
    hits = sum(1 for t in q_terms if t in a_text)
    coverage = hits / len(q_terms) if q_terms else 1.0

    if coverage >= 0.4 or has_table or has_pred:
        return {"status": "ok", "note": "", "score": round(max(coverage, 0.6), 2)}
    return {"status": "weak",
            "note": "The answer may not fully address the question — review the "
                    "underlying data or rephrase.",
            "score": round(coverage, 2)}


def _keywords(text: str):
    words = re.findall(r"[a-zA-Z%]+", (text or "").lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


def _llm_self_check(question, answer, env):
    """Optional LLM self-check. Returns the same dict or None if unavailable."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import requests
        prompt = (
            "Does the ANSWER directly address the QUESTION? Reply with a single "
            "word: YES or NO, then a short reason.\n\n"
            f"QUESTION: {question}\nANSWER: {answer[:1200]}"
        )
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": os.environ.get("PLANNER_MODEL", "claude-sonnet-4-5"),
                  "max_tokens": 100,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=20,
        )
        r.raise_for_status()
        body = r.json()
        txt = "".join(b.get("text", "") for b in body.get("content", [])
                      if b.get("type") == "text").strip()
        ok = txt.lower().startswith("yes")
        return {"status": "ok" if ok else "weak",
                "note": "" if ok else txt,
                "score": 1.0 if ok else 0.3}
    except Exception as e:
        log.warning("LLM self-check failed (%s)", e)
        return None
