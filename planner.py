"""
Planner  (Option B, Phase 1)
============================

Decides whether a natural-language question is a RETRIEVAL question (answer from
data as-is, via Genie) or a PREDICTION question (project a metric forward, via
the Option C TabPFN chain). When it's a prediction, it also extracts the
structured parameters run_prediction() needs.

Design (per the TabPFN spec):
  * Output is a small structured dict, never free text.
  * Hybrid strategy: cheap keyword RULES first; a structured LLM call only for
    the ambiguous middle. Most questions never touch the LLM, keeping latency
    and cost down. The LLM, when used, is constrained to valid targets/scopes
    and to the real entity names we pass in, which defuses the spec's
    "most failure-prone part" (entity/target hallucination).
  * classify() is the single entry point app.py calls.

Env config:
  PLANNER_MODE                 rules | llm | hybrid     (default hybrid)
  PLANNER_LLM_PROVIDER         anthropic | openai       (default anthropic)
  PLANNER_MODEL                model id                 (default claude-sonnet-4-5)
  ANTHROPIC_API_KEY / OPENAI_API_KEY
  PLANNER_CONFIDENCE_THRESHOLD float 0-1                (default 0.8)
"""

import os
import re
import json
import logging

log = logging.getLogger("ernest.planner")

PLANNER_MODE = os.environ.get("PLANNER_MODE", "hybrid").lower()
PLANNER_PROVIDER = os.environ.get("PLANNER_LLM_PROVIDER", "anthropic").lower()
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-sonnet-4-5")
CONF_THRESHOLD = float(os.environ.get("PLANNER_CONFIDENCE_THRESHOLD", "0.8"))

VALID_TARGETS = ["revenue", "gtm", "gtm_pct", "orders"]
VALID_SCOPES = ["portfolio", "client", "shipto"]

# --- lexical signatures -----------------------------------------------------
# Strong prediction cues. Presence of any (with a forward-looking sense) tips
# the question toward prediction.
_PREDICT_WORDS = [
    "predict", "forecast", "forecasted", "project", "projected", "projection",
    "expected", "expect", "will be", "next month", "next quarter", "next year",
    "next few months", "coming months", "going to", "likely to", "trend toward",
    "estimate for", "outlook", "future", "by end of", "by q", "over the next",
]
# Words that push back toward retrieval even if a soft cue appeared.
_RETRIEVE_WORDS = [
    "how many", "how much", "what was", "what were", "last week", "last month",
    "last quarter", "year to date", "ytd", "so far", "to date", "list ",
    "show me", "which ", "top ", "total ", "sum of", "count of", "average",
    "breakdown", "compare", "versus", " vs ", "historical", "past ",
]

# target keyword -> canonical target
_TARGET_HINTS = {
    "revenue": "revenue", "sales": "revenue", "top line": "revenue",
    "gtm %": "gtm_pct", "gtm percent": "gtm_pct", "margin %": "gtm_pct",
    "margin percent": "gtm_pct", "gross margin %": "gtm_pct",
    "gtm": "gtm", "gross to margin": "gtm", "margin dollars": "gtm",
    "order": "orders", "orders": "orders", "po count": "orders",
}

_HORIZON_UNIT = {"month": 1, "months": 1, "quarter": 3, "quarters": 3,
                 "year": 12, "years": 12}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def classify(question: str, entities: dict = None) -> dict:
    """
    entities = {
        "clients":  [<client name>, ...],
        "shiptos":  [{"name": <ship-to>, "client": <client>}, ...],
    }
    Returns one of:
      {"task": "retrieval", "confidence": float, "method": "..."}
      {"task": "prediction", "target","scope","entity","client","horizon",
       "confidence": float, "method": "..."}
    Never raises — on any failure it degrades to retrieval, the safe default.
    """
    entities = entities or {"clients": [], "shiptos": []}
    q = (question or "").strip()
    if not q:
        return {"task": "retrieval", "confidence": 1.0, "method": "empty"}

    if PLANNER_MODE == "llm":
        return _llm_classify(q, entities) or _rules_classify(q, entities)

    # rules first (hybrid + rules modes)
    r = _rules_classify(q, entities)
    if PLANNER_MODE == "rules":
        return r
    if r["confidence"] >= CONF_THRESHOLD:
        return r
    # ambiguous -> escalate to the LLM, fall back to the rules result
    return _llm_classify(q, entities) or r


# ---------------------------------------------------------------------------
# RULES classifier
# ---------------------------------------------------------------------------
def _rules_classify(q: str, entities: dict) -> dict:
    ql = q.lower()

    pred_hits = sum(1 for w in _PREDICT_WORDS if w in ql)
    retr_hits = sum(1 for w in _RETRIEVE_WORDS if w in ql)

    # Decide task + a rough confidence.
    if pred_hits and pred_hits >= retr_hits:
        conf = min(0.95, 0.6 + 0.12 * pred_hits - 0.08 * retr_hits)
        task = "prediction"
    elif retr_hits and not pred_hits:
        conf = min(0.95, 0.7 + 0.08 * retr_hits)
        task = "retrieval"
    elif pred_hits and pred_hits < retr_hits:
        # both present, retrieval stronger -> lean retrieval but low confidence
        conf = 0.55
        task = "retrieval"
    else:
        # no strong cue either way -> default retrieval, low confidence so hybrid
        # mode escalates to the LLM.
        conf = 0.5
        task = "retrieval"

    if task == "retrieval":
        return {"task": "retrieval", "confidence": round(conf, 2), "method": "rules"}

    # prediction: extract parameters deterministically
    params = _extract_params(q, ql, entities)
    return {"task": "prediction", "confidence": round(conf, 2),
            "method": "rules", **params}


def _extract_params(q: str, ql: str, entities: dict) -> dict:
    # target
    target = "revenue"
    for hint, canon in sorted(_TARGET_HINTS.items(), key=lambda kv: -len(kv[0])):
        if hint in ql:
            target = canon
            break

    # horizon: "next 6 months", "next 2 quarters", "by end of year"
    horizon = 3
    m = re.search(r"next\s+(\d+)\s*(month|months|quarter|quarters|year|years)", ql)
    if m:
        horizon = int(m.group(1)) * _HORIZON_UNIT.get(m.group(2), 1)
    elif "next quarter" in ql or "coming quarter" in ql:
        horizon = 3
    elif "next month" in ql:
        horizon = 1
    elif "next year" in ql:
        horizon = 12
    horizon = max(1, min(horizon, 12))

    # entity resolution: match a known client or ship-to name inside the text
    scope, entity, client = "portfolio", None, None
    match = _match_entity(ql, entities)
    if match:
        scope, entity, client = match

    return {"target": target, "scope": scope, "entity": entity,
            "client": client, "horizon": horizon}


def _match_entity(ql: str, entities: dict):
    """Return (scope, entity, client) for the longest matching known name."""
    best = None  # (len, scope, entity, client)
    for c in entities.get("clients", []):
        if c and c.lower() in ql:
            if not best or len(c) > best[0]:
                best = (len(c), "client", c, None)
    for st in entities.get("shiptos", []):
        name = st.get("name") if isinstance(st, dict) else st
        parent = st.get("client") if isinstance(st, dict) else None
        if name and name.lower() in ql:
            if not best or len(name) > best[0]:
                best = (len(name), "shipto", name, parent)
    if best:
        return best[1], best[2], best[3]
    return None


# ---------------------------------------------------------------------------
# LLM classifier (structured output, constrained to valid values)
# ---------------------------------------------------------------------------
def _llm_classify(q: str, entities: dict):
    """
    One structured call. Returns the same dict shape as the rules classifier,
    or None if the provider/key is unavailable (caller falls back to rules).
    """
    client_names = entities.get("clients", [])
    shipto_names = [s.get("name") if isinstance(s, dict) else s
                    for s in entities.get("shiptos", [])]

    system = (
        "You are a query router for a sales BI assistant. Decide if the user's "
        "question is RETRIEVAL (answer from existing data) or PREDICTION "
        "(project a metric into the future). Respond with ONLY a JSON object, "
        "no prose.\n"
        "Schema:\n"
        '{"task":"retrieval"} OR '
        '{"task":"prediction","target":"revenue|gtm|gtm_pct|orders",'
        '"scope":"portfolio|client|shipto","entity":<exact name or null>,'
        '"client":<parent client for a ship-to or null>,"horizon":<int 1-12 months>}\n'
        "Rules: entity MUST be one of the provided names or null. If no entity is "
        "named, scope is portfolio and entity null. Map 'next quarter'->3, "
        "'next month'->1, 'next year'->12."
    )
    user = (
        f"Question: {q}\n\n"
        f"Known clients: {json.dumps(client_names[:200])}\n"
        f"Known ship-tos: {json.dumps(shipto_names[:200])}"
    )

    try:
        if PLANNER_PROVIDER == "openai":
            raw = _call_openai(system, user)
        else:
            raw = _call_anthropic(system, user)
        if not raw:
            return None
        data = _safe_json(raw)
        if not data:
            return None
        return _normalise_llm(data, entities)
    except Exception as e:
        log.warning("LLM planner failed (%s); falling back to rules", e)
        return None


def _normalise_llm(data: dict, entities: dict) -> dict:
    task = (data.get("task") or "retrieval").lower()
    if task != "prediction":
        return {"task": "retrieval", "confidence": 0.9, "method": "llm"}

    target = (data.get("target") or "revenue").lower()
    if target not in VALID_TARGETS:
        target = "revenue"
    scope = (data.get("scope") or "portfolio").lower()
    if scope not in VALID_SCOPES:
        scope = "portfolio"
    entity = data.get("entity")
    client = data.get("client")

    # Guard: the entity must be a real known name; otherwise drop to portfolio.
    known = set(c.lower() for c in entities.get("clients", []))
    known |= set((s.get("name") if isinstance(s, dict) else s).lower()
                 for s in entities.get("shiptos", []))
    if entity and entity.lower() not in known:
        entity, scope = None, "portfolio"

    try:
        horizon = int(data.get("horizon") or 3)
    except (TypeError, ValueError):
        horizon = 3
    horizon = max(1, min(horizon, 12))

    return {"task": "prediction", "target": target, "scope": scope,
            "entity": entity, "client": client, "horizon": horizon,
            "confidence": 0.9, "method": "llm"}


# ---------------------------------------------------------------------------
# provider calls (kept tiny and single-purpose)
# ---------------------------------------------------------------------------
def _call_anthropic(system: str, user: str):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import requests
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": PLANNER_MODEL, "max_tokens": 300, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    parts = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


def _call_openai(system: str, user: str):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    import requests
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
        json={"model": PLANNER_MODEL, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _safe_json(raw: str):
    """Extract a JSON object from a possibly-fenced LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
