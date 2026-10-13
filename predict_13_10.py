"""
Forecast / Prediction chain for the Ernest dashboard  (TabPFN spec, Option C)
=============================================================================

Implements the prediction branch from the BI-Agent-with-TabPFN spec, wired to
the existing Databricks data layer in app.py:

    Planner  ->  Feature Builder  ->  TabPFN Service  ->  Synthesis

Design notes tied to the spec:
  * TabPFN is SELF-HOSTED and NON-NEGOTIABLE: we run the open-source `tabpfn`
    pip package in-process (CPU is fine for dashboard row counts). We NEVER
    call Prior Labs' hosted API.
  * The TabPFN weights are gated on HuggingFace. When TABPFN_ENABLED=true and a
    HF token is present, we use the real TabPFNRegressor. Otherwise we fall back
    to a transparent statistical forecaster (trend + seasonal + residual band)
    so the prototype runs end-to-end today. The interface is identical, so
    switching to real TabPFN is a config flip, not a rewrite.
  * Each step is a small, single-purpose function — not one giant prompt —
    exactly as the spec's risk section recommends.

This module is imported by app.py, which supplies run_sql() and fqtn().
"""

import os
import math
import logging

log = logging.getLogger("ernest.predict")

# TabPFN is optional at runtime. We import lazily inside the service so the app
# boots even when weights aren't provisioned.
TABPFN_ENABLED = os.environ.get("TABPFN_ENABLED", "false").lower() == "true"
TABPFN_HORIZON_MAX = 12


# ---------------------------------------------------------------------------
# 1. PLANNER  — classify the request and resolve target/entity/grain
# ---------------------------------------------------------------------------
# In the full spec this is an LLM call. For a dashboard the target space is
# small and structured (revenue / gtm / gtm_pct / orders, by client or ship-to),
# so a deterministic planner is more reliable and cheaper. The signature is
# LLM-swappable: return the same dict from an LLM call later if desired.

TARGETS = {
    "revenue":  {"label": "Revenue",      "col": "revenue",  "money": True},
    "gtm":      {"label": "GTM $",        "col": "gtm",      "money": True},
    "gtm_pct":  {"label": "GTM %",        "col": "gtm_pct",  "money": False},
    "orders":   {"label": "Orders",       "col": "orders",   "money": False},
}


def plan_prediction(req: dict) -> dict:
    """
    Validate and normalise a prediction request coming from the Forecast page.
    req = {
        target: 'revenue'|'gtm'|'gtm_pct'|'orders',
        scope:  'client'|'shipto'|'portfolio',
        entity: '<client name>' | '<ship-to name>' | None,
        client: '<client name>'        # context when scope=='shipto'
        horizon: <int months ahead>,
    }
    Returns a normalised plan or raises ValueError (the Feature Builder is the
    spec's most failure-prone step, so we validate hard before touching data).
    """
    target = (req.get("target") or "revenue").lower()
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'. Choose one of {list(TARGETS)}.")

    scope = (req.get("scope") or "portfolio").lower()
    if scope not in ("client", "shipto", "portfolio"):
        raise ValueError("scope must be client, shipto or portfolio.")

    horizon = int(req.get("horizon") or 3)
    horizon = max(1, min(horizon, TABPFN_HORIZON_MAX))

    entity = req.get("entity")
    client = req.get("client")
    if scope == "client" and not entity:
        raise ValueError("scope=client requires an entity (client name).")
    if scope == "shipto" and not entity:
        raise ValueError("scope=shipto requires an entity (ship-to name).")

    return {
        "task": "prediction",
        "target": target,
        "target_meta": TARGETS[target],
        "scope": scope,
        "entity": entity,
        "client": client,
        "horizon": horizon,
    }


# ---------------------------------------------------------------------------
# 2. FEATURE BUILDER  — pull the historical series via SQL, encode features
# ---------------------------------------------------------------------------
# Uses the SAME sales/inventory tables and column names as build_dashboard_payload.
# Aggregates the target to a monthly series for the requested entity, then adds
# date-part + lag features (the spec's "simple feature-encoding strategy").

def _month_sort_key(m):
    p = _parse_month(m)
    return (p[1], p[0]) if p else (0, 0)


def build_features(plan: dict, run_sql, fqtn) -> dict:
    """
    Returns {
        months: [ 'M/YYYY', ... ] chronological,
        y:      [ float ... ] target value per month,
        X:      [[month_index, month_of_year, lag1, lag3], ...] feature rows,
        n:      len,
    }
    """
    sales = fqtn(os.environ.get("DBX_SALES_TABLE", "sales_data_updated"))

    # Build the WHERE clause for the requested scope.
    where = ["s.`Extended_Amount` IS NOT NULL"]
    params_note = ""
    if plan["scope"] == "client":
        where.append(f"s.`Sold-To_Party3` = {_sql_str(plan['entity'])}")
    elif plan["scope"] == "shipto":
        where.append(f"s.`Ship-to_party14` = {_sql_str(plan['entity'])}")
        if plan.get("client"):
            where.append(f"s.`Sold-To_Party3` = {_sql_str(plan['client'])}")
    where_sql = " AND ".join(where)

    # Aggregate the chosen target to a monthly series.
    # revenue/gtm/orders are additive; gtm_pct is derived after aggregation.
    agg_sql = f"""
    SELECT
        s.`Month/Year`                              AS month,
        SUM(CAST(s.`Extended_Amount` AS DOUBLE))    AS revenue,
        SUM(CAST(s.`GTM_Dollars`     AS DOUBLE))    AS gtm,
        COUNT(DISTINCT s.`Billing_Document`)        AS orders
    FROM {sales} s
    WHERE {where_sql}
    GROUP BY s.`Month/Year`
    """
    rows = run_sql(agg_sql)

    # Order chronologically and derive the series for the target.
    # Drop rows whose month can't be parsed so a stray value can't distort the
    # series ordering or the lag features.
    rows = [r for r in rows if r.get("month") and _parse_month(r["month"])]
    rows.sort(key=lambda r: _month_sort_key(r["month"]))
    months = [r["month"] for r in rows]

    def val(r):
        if plan["target"] == "revenue":
            return _num(r.get("revenue"))
        if plan["target"] == "gtm":
            return _num(r.get("gtm"))
        if plan["target"] == "orders":
            return _num(r.get("orders"))
        if plan["target"] == "gtm_pct":
            rev = _num(r.get("revenue"))
            return (_num(r.get("gtm")) / rev * 100.0) if rev else 0.0
        return 0.0

    y = [val(r) for r in rows]

    # Feature encoding: month index, month-of-year (seasonality), lag-1, lag-3.
    X = []
    for i, m in enumerate(months):
        moy = _month_of_year(m)
        lag1 = y[i - 1] if i >= 1 else y[i]
        lag3 = y[i - 3] if i >= 3 else y[max(0, i - 1)]
        X.append([float(i), float(moy), float(lag1), float(lag3)])

    return {"months": months, "y": y, "X": X, "n": len(y)}


# ---------------------------------------------------------------------------
# 3. TABPFN SERVICE  — self-hosted inference (real when enabled, else fallback)
# ---------------------------------------------------------------------------

def tabpfn_forecast(feat: dict, horizon: int) -> dict:
    """
    Produce point predictions + an uncertainty band for the next `horizon`
    months. Returns {
        point:   [float ...] length horizon,
        lower:   [float ...] 10th percentile,
        upper:   [float ...] 90th percentile,
        engine:  'tabpfn' | 'statistical-fallback',
    }
    """
    n = feat["n"]
    if n < 4:
        # Not enough history to model; flat-carry the last value with a wide band.
        last = feat["y"][-1] if feat["y"] else 0.0
        pad = abs(last) * 0.25 or 1.0
        return {
            "point": [last] * horizon,
            "lower": [last - pad] * horizon,
            "upper": [last + pad] * horizon,
            "engine": "insufficient-history",
        }

    if TABPFN_ENABLED:
        try:
            log.info("tabpfn enabled")
            return _tabpfn_real(feat, horizon)
        except Exception as e:
            log.warning("TabPFN unavailable (%s); using statistical fallback", e)

    log.info("tabpfn disabled")
    return _statistical_forecast(feat, horizon)


def _tabpfn_real(feat: dict, horizon: int) -> dict:
    """Real self-hosted TabPFN regressor with quantile output."""
    import numpy as np
    from tabpfn import TabPFNRegressor

    X = np.array(feat["X"], dtype=float)
    y = np.array(feat["y"], dtype=float)

    reg = TabPFNRegressor(device=os.environ.get("TABPFN_DEVICE", "cpu"))
    reg.fit(X, y)

    # Build future feature rows, rolling lag features forward as we predict.
    months = feat["months"]
    y_hist = list(feat["y"])
    point, lower, upper = [], [], []
    last_idx = len(months) - 1
    for h in range(1, horizon + 1):
        idx = last_idx + h
        moy = ((_month_of_year(months[-1]) - 1 + h) % 12) + 1
        lag1 = point[-1] if point else y_hist[-1]
        lag3 = (point[-3] if len(point) >= 3
                else y_hist[-(3 - len(point))] if (3 - len(point)) <= len(y_hist) else y_hist[-1])
        xf = np.array([[float(idx), float(moy), float(lag1), float(lag3)]])
        q = reg.predict(xf, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        lo, mid, hi = float(q[0][0]), float(q[1][0]), float(q[2][0])
        point.append(mid); lower.append(lo); upper.append(hi)
    return {"point": point, "lower": lower, "upper": upper, "engine": "tabpfn"}


def _statistical_forecast(feat: dict, horizon: int) -> dict:
    """
    Transparent fallback: linear trend on the month index + additive
    month-of-year seasonal offset, with an uncertainty band from residual
    spread that widens with horizon. Deterministic and dependency-free.
    """
    y = feat["y"]
    months = feat["months"]
    n = len(y)
    xs = list(range(n))

    # Ordinary least squares slope/intercept on the index.
    mx = sum(xs) / n
    my = sum(y) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((xs[i] - mx) * (y[i] - my) for i in range(n)) / denom
    intercept = my - slope * mx

    def trend(i):
        return intercept + slope * i

    # Seasonal offset per month-of-year (mean residual for that calendar month).
    seasonal = {}
    counts = {}
    for i, m in enumerate(months):
        moy = _month_of_year(m)
        seasonal[moy] = seasonal.get(moy, 0.0) + (y[i] - trend(i))
        counts[moy] = counts.get(moy, 0) + 1
    for k in seasonal:
        seasonal[k] /= counts[k]

    # Residual std for the band.
    resid = [y[i] - (trend(i) + seasonal.get(_month_of_year(months[i]), 0.0)) for i in range(n)]
    rstd = (sum(r * r for r in resid) / n) ** 0.5 if n else 0.0

    point, lower, upper = [], [], []
    last_moy = _month_of_year(months[-1])
    for h in range(1, horizon + 1):
        idx = n - 1 + h
        moy = ((last_moy - 1 + h) % 12) + 1
        base = trend(idx) + seasonal.get(moy, 0.0)
        # Band widens ~sqrt(h) like a random-walk forecast interval.
        band = 1.2816 * rstd * math.sqrt(h)   # ~80% interval (10th/90th pct)
        point.append(base)
        lower.append(base - band)
        upper.append(base + band)
    return {"point": point, "lower": lower, "upper": upper, "engine": "statistical-fallback"}


# ---------------------------------------------------------------------------
# 4. SYNTHESIS  — plain-language answer with the uncertainty range
# ---------------------------------------------------------------------------
# The spec insists the answer ALWAYS mentions the uncertainty range, never just
# the number. A deterministic narrator does this reliably; swap for an LLM call
# if richer phrasing is wanted.

def synthesize(plan: dict, feat: dict, fc: dict) -> dict:
    meta = plan["target_meta"]
    horizon = plan["horizon"]
    fut_months = _future_month_labels(feat["months"], horizon)

    def fmt(v):
        if meta["money"]:
            return "${:,.0f}".format(v)
        if plan["target"] == "gtm_pct":
            return "{:.1f}%".format(v)
        return "{:,.0f}".format(v)

    who = ("the portfolio" if plan["scope"] == "portfolio"
           else plan["entity"])
    first = fc["point"][0]
    last = fc["point"][-1]
    lo_last, hi_last = fc["lower"][-1], fc["upper"][-1]

    # Direction vs last actual.
    last_actual = feat["y"][-1] if feat["y"] else 0.0
    delta = (last - last_actual)
    dir_word = "increase" if delta > 0 else "decrease" if delta < 0 else "hold roughly flat"
    pct_txt = ""
    if last_actual:
        pct_txt = " ({:+.0f}% vs the latest month)".format(delta / abs(last_actual) * 100.0)

    engine_note = {
        "tabpfn": "self-hosted TabPFN",
        "statistical-fallback": "trend+seasonal model (TabPFN weights not provisioned in this environment)",
        "insufficient-history": "carry-forward (too few months to model)",
    }.get(fc["engine"], fc["engine"])

    answer = (
        f"{meta['label']} for {who} is projected to {dir_word}{pct_txt} over the next "
        f"{horizon} month(s). By {fut_months[-1]}, the central estimate is {fmt(last)}, "
        f"within a likely range of {fmt(lo_last)} to {fmt(hi_last)} "
        f"(80% band). Forecast produced by {engine_note}."
    )

    series = [{
        "month": fut_months[i],
        "point": round(fc["point"][i], 2),
        "lower": round(fc["lower"][i], 2),
        "upper": round(fc["upper"][i], 2),
    } for i in range(horizon)]

    return {
        "answer": answer,
        "engine": fc["engine"],
        "target": plan["target"],
        "target_label": meta["label"],
        "money": meta["money"],
        "scope": plan["scope"],
        "entity": who,
        "horizon": horizon,
        "history": [{"month": feat["months"][i], "value": round(feat["y"][i], 2)}
                    for i in range(feat["n"])],
        "forecast": series,
    }


# ---------------------------------------------------------------------------
# ORCHESTRATION  — the one function app.py calls
# ---------------------------------------------------------------------------

def run_prediction(req: dict, run_sql, fqtn) -> dict:
    plan = plan_prediction(req)
    feat = build_features(plan, run_sql, fqtn)
    if feat["n"] == 0:
        raise ValueError("No historical data found for that selection.")
    fc = tabpfn_forecast(feat, plan["horizon"])
    return synthesize(plan, feat, fc)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _sql_str(s: str) -> str:
    """Single-quote and escape a string literal for SQL."""
    return "'" + str(s).replace("'", "''") + "'"


_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _norm_year(y):
    """Expand a 2-digit year to 4 digits ('26' -> 2026)."""
    y = int(y)
    if y < 100:
        return 2000 + y
    return y


def _parse_month(m):
    """
    Parse a month cell into (month:int 1-12, year:int), tolerant of the shapes
    that show up in a Month/Year column, including text month names:
        'Feb-26', 'Feb-2026', 'February 2026', '26-Feb',
        'M/YYYY', 'MM/YYYY', 'YYYY/M', 'YYYY-MM', 'YYYY-MM-DD', 'YYYY.MM',
        'YYYYMM'. Returns None if it can't be parsed.
    """
    if m is None:
        return None
    s = str(m).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return None

    import re
    parts = [p for p in re.split(r"[/\-.\s]+", s) if p != ""]

    # --- text-month formats: 'Feb-26', 'February 2026', '26-Feb' ---
    if len(parts) >= 2:
        p0, p1 = parts[0].lower(), parts[1].lower()
        if p0[:4] in _MONTH_NAMES or p0[:3] in _MONTH_NAMES:
            mon = _MONTH_NAMES.get(p0[:4]) or _MONTH_NAMES.get(p0[:3])
            try:
                return (mon, _norm_year(parts[1]))
            except (ValueError, TypeError):
                return None
        if p1[:4] in _MONTH_NAMES or p1[:3] in _MONTH_NAMES:
            mon = _MONTH_NAMES.get(p1[:4]) or _MONTH_NAMES.get(p1[:3])
            try:
                return (mon, _norm_year(parts[0]))
            except (ValueError, TypeError):
                return None

    # --- all-numeric formats ---
    try:
        if len(parts) >= 2:
            a, b = int(parts[0]), int(parts[1])
            if a > 31:                       # YYYY-MM (4-digit year first)
                year, month = a, b
            elif b > 12:                     # M/YY or M/YYYY (year second)
                month, year = a, _norm_year(b)
            elif b >= 1000:                  # M/YYYY
                month, year = a, b
            else:
                # both small & ambiguous: assume month/year (the dashboard format)
                month, year = a, _norm_year(b)
            if 1 <= month <= 12:
                return (month, year)
            if 1 <= year <= 12 and month >= 1000:   # swapped
                return (year, month)
            return None
        if len(parts) == 1 and len(parts[0]) == 6:   # 'YYYYMM'
            year, month = int(parts[0][:4]), int(parts[0][4:])
            if 1 <= month <= 12:
                return (month, year)
    except (ValueError, TypeError):
        return None
    return None


def _month_of_year(m: str) -> int:
    p = _parse_month(m)
    return p[0] if p else 1


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _detect_month_style(sample):
    """Return 'abbr' if the source months look like 'Feb-26', else 'numeric'."""
    if not sample:
        return "numeric"
    s = str(sample).strip().lower()
    head = s.split("-")[0].split("/")[0].split(" ")[0]
    if head[:3] in _MONTH_NAMES:
        return "abbr"
    return "numeric"


def _future_month_labels(months, horizon):
    if not months:
        return [f"+{i+1}" for i in range(horizon)]
    p = _parse_month(months[-1])
    if not p:
        # can't parse the anchor month; fall back to generic offsets
        return [f"+{i+1}" for i in range(horizon)]
    mm, yy = p
    style = _detect_month_style(months[-1])
    out = []
    for _ in range(horizon):
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
        if style == "abbr":
            out.append(f"{_MONTH_ABBR[mm-1]}-{yy % 100:02d}")   # 'Mar-26'
        else:
            out.append(f"{mm}/{yy}")
    return out
