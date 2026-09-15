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


# ---------------------------------------------------------------------------
# UNIFY
# ---------------------------------------------------------------------------
def unify(question: str, kind: str, payload: dict, routing: dict) -> dict:
    """
    kind == 'retrieval'  -> payload is the genie_ask() result
    kind == 'prediction' -> payload is the run_prediction() result
    Returns the unified envelope.
    """
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
    return {
        "answer": g.get("text") or "",
        #"sql": g.get("sql"),
        "table": {
            "columns": g.get("columns") or [],
            "rows": g.get("rows") or [],
        } if g.get("rows") else None,
        "prediction": None,
        "uncertainty": None,
        "chart": None,
        "engine": "genie",
    }


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
        "answer": p.get("answer") or "",
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
