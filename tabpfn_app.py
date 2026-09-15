"""
Ernest Account Intelligence Dashboard — Databricks backend
===========================================================

This Flask app is designed to run as a Databricks App (or any host that can
reach your Databricks workspace). It does three jobs:

  1. Serves the dashboard front-end (static/index.html).
  2. /api/data          -> runs SQL against  workspace.sss  and returns the
                           SAME JSON shape the original static dashboard used,
                           so the existing UI code renders unchanged.
  3. /api/genie/ask     -> proxies a natural-language question to a Databricks
                           Genie space (AI/BI Genie Conversation API) and
                           returns the generated SQL + result rows.

--------------------------------------------------------------------
CONFIG — replace the DUMMY values below (or set them as env vars / Databricks
app resources). When running as a Databricks App, DATABRICKS_HOST and the auth
token are injected automatically, so you usually only set the two IDs.
--------------------------------------------------------------------
"""

import os
import time
import re
import uuid
import logging
import threading
import requests
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ernest")
VERIFY_SSL = os.environ.get("DBX_VERIFY_SSL", "true").lower() != "false"


# ------------------------------------------------------------------ CONFIG ---
CONFIG = {
    # Your Databricks workspace URL, e.g. https://dbc-abc123.cloud.databricks.com
    # On Databricks Apps this is provided automatically via DATABRICKS_HOST.
    "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", "https://REPLACE-WORKSPACE.cloud.databricks.com"),

    # ---- DUMMY IDs — replace with your real values --------------------------
    "WORKSPACE_ID":     os.environ.get("DATABRICKS_WORKSPACE_ID", "1234567890123456"),   # dummy
    "GENIE_SPACE_ID":   os.environ.get("GENIE_SPACE_ID",          "01ef0a1b2c3d4e5f6a7b8c9d0e1f2a3b"),  # dummy
    "WAREHOUSE_ID":     os.environ.get("DATABRICKS_WAREHOUSE_ID", "0123456789abcdef"),   # dummy SQL warehouse

    # Auth token. On Databricks Apps, prefer the injected OAuth token; locally
    # you can drop a PAT here or in the DATABRICKS_TOKEN env var.
    "TOKEN": os.environ.get("DATABRICKS_TOKEN", "dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"),  # dummy

    # Unity Catalog location of your two tables.
    "CATALOG": os.environ.get("DBX_CATALOG", "ernest_dashbord_poc"),
    "SCHEMA":  os.environ.get("DBX_SCHEMA",  "invetory_sales_schema"),

    # Table names inside that schema (change if yours differ).
    "SALES_TABLE":     os.environ.get("DBX_SALES_TABLE",     "sales_data_updated"),
    "INVENTORY_TABLE": os.environ.get("DBX_INVENTORY_TABLE", "inventory_data_curated"),
}

def fqtn(table: str) -> str:
    """Fully-qualified table name: workspace.sss.<table>."""
    return f"`{CONFIG['CATALOG']}`.`{CONFIG['SCHEMA']}`.`{table}`"

def _headers():
    return {
        "Authorization": f"Bearer {CONFIG['TOKEN']}",
        "Content-Type": "application/json",
    }

app = Flask(__name__, static_folder="static")


# ============================================================================
#  SQL EXECUTION  (Databricks SQL Statement Execution API 2.0)
# ============================================================================
def run_sql(statement: str, timeout_s: int = 55):
    """
    Execute a SQL statement on the configured SQL warehouse and return a list
    of dict rows. Uses the synchronous wait first, then polls if needed.
    Docs: https://docs.databricks.com/api/workspace/statementexecution
    """
    url = f"{CONFIG['DATABRICKS_HOST']}/api/2.0/sql/statements"
    payload = {
        "warehouse_id": CONFIG["WAREHOUSE_ID"],
        "statement": statement,
        "wait_timeout": "30s",
        "on_wait_timeout": "CONTINUE",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
    }
    r = requests.post(url, headers=_headers(), json=payload, timeout=60, verify=VERIFY_SSL)
    r.raise_for_status()
    body = r.json()
    stmt_id = body["statement_id"]

    # Poll until the statement leaves the PENDING/RUNNING state.
    deadline = time.time() + timeout_s
    while body["status"]["state"] in ("PENDING", "RUNNING"):
        if time.time() > deadline:
            raise TimeoutError(f"SQL statement {stmt_id} timed out")
        time.sleep(1.0)
        g = requests.get(f"{url}/{stmt_id}", headers=_headers(), timeout=60, verify=VERIFY_SSL)
        g.raise_for_status()
        body = g.json()

    state = body["status"]["state"]
    if state != "SUCCEEDED":
        err = body["status"].get("error", {}).get("message", "unknown error")
        raise RuntimeError(f"SQL failed ({state}): {err}")

    result = body.get("result", {})
    cols = [c["name"] for c in body["manifest"]["schema"]["columns"]]
    rows = result.get("data_array", []) or []
    out = [dict(zip(cols, row)) for row in rows]

    # Handle chunked results (large payloads) via external/next chunk links.
    next_chunk = result.get("next_chunk_internal_link")
    while next_chunk:
        c = requests.get(f"{CONFIG['DATABRICKS_HOST']}{next_chunk}", headers=_headers(), timeout=60, verify=VERIFY_SSL)
        c.raise_for_status()
        cb = c.json()
        for row in cb.get("data_array", []) or []:
            out.append(dict(zip(cols, row)))
        next_chunk = cb.get("next_chunk_internal_link")

    return out


# ============================================================================
#  DATA SHAPING  —  build the exact JSON the dashboard expects
# ============================================================================
def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def build_dashboard_payload():
    """
    Query workspace.sss.sales + inventory and assemble the nested structure:
      { months, clients:{ <client>:{ total_*, ship_tos, monthly, order_type_mix,
                                      shipto_data:{...}, hier_shipto_matrix, all_hiers } },
        insights:{...}, battlecards:{...} }
    The SQL is written against the column names you provided. Adjust the aliases
    on the left of each `AS` only if your physical column names differ.
    """
    sales = fqtn(CONFIG["SALES_TABLE"])
    inv = fqtn(CONFIG["INVENTORY_TABLE"])

    # ---- 1. Line-level sales joined to inventory cost/vendor attributes -----
    #     One row per billing line. Everything else is aggregated in Python
    #     so the nested JSON exactly matches the original file.
    line_sql = f"""
    SELECT
        s.`Sold-To_Party3`                              AS client,
        s.`Ship-to_party14`                             AS ship_to,
        s.`Ship_To_Party_City`                          AS city,
        s.`Ship_To_Party_State`                         AS state,
        s.`Plant1`                                      AS plant,
        s.`Product_hierarchy5`                          AS hier,
        s.`Material9`                                   AS material,
        s.`Material8`                                   AS material_id,
        s.`Month/Year`                              AS month,
        CAST(s.`Net_Invoiced_Qty` AS DOUBLE)            AS qty,
        CAST(s.`Extended_Amount`  AS DOUBLE)            AS revenue,
        CAST(s.`Extended_Cost`    AS DOUBLE)            AS cost,
        CAST(s.`GTM_Dollars`      AS DOUBLE)            AS gtm,
        s.`Billing_Document`                            AS billing_doc,
        i.`Vendor name`                                 AS vendor_name,
        CAST(i.`Vendor Cost`   AS DOUBLE)               AS vendor_cost,
        CAST(i.`Standard Cost` AS DOUBLE)               AS standard_cost,
        CAST(i.`PO Variance %`  AS DOUBLE)              AS po_variance_pct,
        CAST(i.`Standard Cost Variance %` AS DOUBLE)    AS std_cost_variance_pct,
        i.`Fixed Supplier`                              AS fixed_supplier,
        CAST(i.`Last Goods Receipt Date` AS STRING)     AS last_gr_date
    FROM {sales} s
    LEFT JOIN {inv} i
         ON s.`Material9` = i.`Material Description`
          AND s.`Plant0`    = i.`Plant`
    WHERE s.`Extended_Amount` IS NOT NULL
    """
    rows = run_sql(line_sql)

    def month_to_int(month_name):
    # %B handles full names like 'March', %b handles short names like 'Mar'
        for fmt in ('%B', '%b'):
            try:
                return datetime.strptime(month_name.strip(), fmt).month
            except ValueError:
                continue
        raise ValueError(f"Invalid month name: {month_name}")

    # ---- 2. Determine the month axis (sorted chronologically) ---------------
    def month_key(m):
        try:
            mm, yy = m.split("-")
            return (int(yy), int(month_to_int (mm)))
        except Exception:
            return (0, 0)

    months = sorted({r["month"] for r in rows if r.get("month")}, key=month_key)

    # ---- 3. Group rows by client -> ship_to -> hier -> material -------------
    clients = {}
    for r in rows:
        c = r["client"] or "Unknown"
        clients.setdefault(c, []).append(r)

    payload = {"months": months, "clients": {}, "insights": {}, "battlecards": {}}

    for client, crows in clients.items():
        payload["clients"][client] = _shape_client(client, crows, months)
        payload["insights"][client] = _shape_insights(client, crows, months)
        payload["battlecards"][client] = _shape_battlecard(
            client, payload["clients"][client], payload["insights"][client], months
        )

    return payload


def _monthly_series(rows, months):
    by_month = {m: {"revenue": 0.0, "cost": 0.0, "gtm": 0.0, "qty": 0.0, "orders": set()} for m in months}
    for r in rows:
        m = r.get("month")
        if m not in by_month:
            continue
        by_month[m]["revenue"] += _f(r["revenue"])
        by_month[m]["cost"] += _f(r["cost"])
        by_month[m]["gtm"] += _f(r["gtm"])
        by_month[m]["qty"] += _f(r["qty"])
        if r.get("billing_doc"):
            by_month[m]["orders"].add(r["billing_doc"])
    return [
        {
            "month": m,
            "revenue": round(by_month[m]["revenue"], 2),
            "cost": round(by_month[m]["cost"], 2),
            "gtm": round(by_month[m]["gtm"], 2),
            "qty": round(by_month[m]["qty"], 2),
            "orders": len(by_month[m]["orders"]),
        }
        for m in months
    ]


def _shape_client(client, crows, months):
    total_rev = sum(_f(r["revenue"]) for r in crows)
    total_cost = sum(_f(r["cost"]) for r in crows)
    total_gtm = sum(_f(r["gtm"]) for r in crows)
    ship_tos = sorted({r["ship_to"] for r in crows if r.get("ship_to")})

    # order type mix is not in the provided columns; group by plant as a proxy
    # (swap to a real order-type column here if you have one).
    otmix = {}
    for r in crows:
        key = r.get("plant") or "Other"
        otmix[key] = round(otmix.get(key, 0.0) + _f(r["revenue"]), 2)

    shipto_data = {}
    by_st = {}
    for r in crows:
        by_st.setdefault(r["ship_to"], []).append(r)

    hier_shipto = {}
    all_hiers = set()

    for st, strows in by_st.items():
        st_rev = sum(_f(r["revenue"]) for r in strows)
        st_gtm = sum(_f(r["gtm"]) for r in strows)
        # hierarchies
        by_hier = {}
        for r in strows:
            by_hier.setdefault(r["hier"], []).append(r)
        hiers = []
        for h, hrows in by_hier.items():
            hr = sum(_f(x["revenue"]) for x in hrows)
            hg = sum(_f(x["gtm"]) for x in hrows)
            hq = sum(_f(x["qty"]) for x in hrows)
            hiers.append({
                "hier": h, "revenue": round(hr, 2), "gtm": round(hg, 2),
                "gtm_pct": round(hg / hr * 100, 2) if hr else 0,
                "qty": round(hq, 2),
                "orders": len({x.get("billing_doc") for x in hrows if x.get("billing_doc")}),
            })
            all_hiers.add(h)
            hier_shipto.setdefault(h, set()).add(st)
        hiers.sort(key=lambda x: -x["revenue"])
        # materials
        by_mat = {}
        for r in strows:
            by_mat.setdefault(r["material_id"], []).append(r)
        materials = []
        for mid, mrows in by_mat.items():
            mr = sum(_f(x["revenue"]) for x in mrows)
            mg = sum(_f(x["gtm"]) for x in mrows)
            mq = sum(_f(x["qty"]) for x in mrows)
            f0 = mrows[0]
            materials.append({
                "material_id": mid, "material": f0.get("material"),
                "hier": f0.get("hier"),
                "revenue": round(mr, 2), "gtm": round(mg, 2),
                "gtm_pct": round(mg / mr * 100, 2) if mr else 0,
                "qty": round(mq, 2),
                "orders": len({x.get("billing_doc") for x in mrows if x.get("billing_doc")}),
                "vendor_name": f0.get("vendor_name") or "",
                "vendor_cost": _f(f0.get("vendor_cost")),
                "standard_cost": _f(f0.get("standard_cost")),
                "po_variance_pct": _f(f0.get("po_variance_pct")),
                "std_cost_variance_pct": _f(f0.get("std_cost_variance_pct")),
                "fixed_supplier": f0.get("fixed_supplier") or "",
                "last_gr_date": f0.get("last_gr_date") or "",
                "price_series": [],
            })
        materials.sort(key=lambda x: -x["revenue"])

        st_otmix = {}
        for r in strows:
            key = r.get("plant") or "Other"
            st_otmix[key] = round(st_otmix.get(key, 0.0) + _f(r["revenue"]), 2)

        shipto_data[st] = {
            "ship_to": st,
            "city": strows[0].get("city"), "state": strows[0].get("state"),
            "plant": strows[0].get("plant"),
            "revenue": round(st_rev, 2), "gtm": round(st_gtm, 2),
            "gtm_pct": round(st_gtm / st_rev * 100, 2) if st_rev else 0,
            "orders": len({r.get("billing_doc") for r in strows if r.get("billing_doc")}),
            "monthly": _monthly_series(strows, months),
            "hierarchies": hiers,
            "materials": materials,
            "order_type_mix": st_otmix,
        }

    return {
        "total_revenue": round(total_rev, 2),
        "total_gtm": round(total_gtm, 2),
        "total_cost": round(total_cost, 2),
        "avg_gtm_pct": round(total_gtm / total_rev * 100, 2) if total_rev else 0,
        "ship_tos": ship_tos,
        "monthly": _monthly_series(crows, months),
        "order_type_mix": otmix,
        "shipto_data": shipto_data,
        "hier_shipto_matrix": {h: sorted(list(s)) for h, s in hier_shipto.items()},
        "all_hiers": sorted(all_hiers),
    }


def _shape_insights(client, crows, months):
    """Growth, whitespace, margin flags, cost-risk, supplier concentration."""
    last3 = months[-3:] if len(months) >= 3 else months
    prior3 = months[-6:-3] if len(months) >= 6 else []

    # growth by hier
    def hier_window_rev(hier, window):
        return sum(_f(r["revenue"]) for r in crows if r["hier"] == hier and r["month"] in window)
    hiers = sorted({r["hier"] for r in crows})
    growth_all = []
    for h in hiers:
        l3 = hier_window_rev(h, last3)
        p3 = hier_window_rev(h, prior3)
        tot = sum(_f(r["revenue"]) for r in crows if r["hier"] == h)
        if p3 == 0 and l3 > 0:
            pct = 999
        elif p3 == 0:
            pct = 0
        else:
            pct = round((l3 - p3) / p3 * 100, 1)
        growth_all.append({"hier": h, "last3_revenue": round(l3, 2),
                           "prior3_revenue": round(p3, 2), "pct_change": pct,
                           "total_revenue": round(tot, 2)})
    growth_all.sort(key=lambda x: -x["total_revenue"])
    growers = sorted([g for g in growth_all if g["pct_change"] > 5],
                     key=lambda x: -x["pct_change"])[:6]
    decliners = sorted([g for g in growth_all if g["pct_change"] < -5],
                       key=lambda x: x["pct_change"])[:6]

    # margin flags (line/hier below target GTM%) and high margin
    by_st_hier = {}
    for r in crows:
        by_st_hier.setdefault((r["ship_to"], r["hier"]), {"rev": 0.0, "gtm": 0.0})
        by_st_hier[(r["ship_to"], r["hier"])]["rev"] += _f(r["revenue"])
        by_st_hier[(r["ship_to"], r["hier"])]["gtm"] += _f(r["gtm"])
    margin_flags, high_margin = [], []
    for (st, h), v in by_st_hier.items():
        if v["rev"] <= 0:
            continue
        pct = v["gtm"] / v["rev"] * 100
        if pct < 20:
            margin_flags.append({"ship_to": st, "hier": h, "gtm_pct": round(pct, 2),
                                 "revenue": round(v["rev"], 2),
                                 "flag": "critical-low-margin" if pct < 10 else "low-margin"})
        elif pct >= 45:
            high_margin.append({"ship_to": st, "hier": h, "gtm_pct": round(pct, 2),
                                "revenue": round(v["rev"], 2)})
    margin_flags.sort(key=lambda x: x["gtm_pct"])
    high_margin.sort(key=lambda x: -x["gtm_pct"])
    high_margin = high_margin[:8]

    # whitespace: hier bought by peer ship-tos but not by this one
    ship_tos = sorted({r["ship_to"] for r in crows})
    hier_to_sts = {}
    st_hier_rev = {}
    for r in crows:
        hier_to_sts.setdefault(r["hier"], set()).add(r["ship_to"])
        st_hier_rev.setdefault((r["ship_to"], r["hier"]), 0.0)
        st_hier_rev[(r["ship_to"], r["hier"])] += _f(r["revenue"])
    whitespace = []
    for h, buyers in hier_to_sts.items():
        peer_rev = [st_hier_rev[(st, h)] for st in buyers]
        avg_peer = sum(peer_rev) / len(peer_rev) if peer_rev else 0
        for st in ship_tos:
            if st not in buyers:
                whitespace.append({"ship_to": st, "hier": h,
                                   "peer_count": len(buyers),
                                   "avg_peer_revenue": round(avg_peer, 2)})
    whitespace.sort(key=lambda x: -x["peer_count"])
    whitespace = whitespace[:12]

    # cost risk: materials with big variance
    seen = set()
    cost_risk = []
    for r in crows:
        po = _f(r.get("po_variance_pct"))
        sc = _f(r.get("std_cost_variance_pct"))
        if abs(po) > 20 or abs(sc) > 20:
            key = (r["ship_to"], r["material_id"])
            if key in seen:
                continue
            seen.add(key)
            rev = sum(_f(x["revenue"]) for x in crows
                      if x["ship_to"] == r["ship_to"] and x["material_id"] == r["material_id"])
            cost_risk.append({"ship_to": r["ship_to"], "material": r["material"],
                              "material_id": r["material_id"],
                              "po_variance_pct": round(po, 2),
                              "std_cost_variance_pct": round(sc, 2),
                              "revenue": round(rev, 2)})
    cost_risk.sort(key=lambda x: -max(abs(x["po_variance_pct"]), abs(x["std_cost_variance_pct"])))
    cost_risk = cost_risk[:10]

    # supplier concentration: SKU count by vendor
    vendor_sku = {}
    for r in crows:
        v = r.get("vendor_name")
        if v:
            vendor_sku.setdefault(v, set()).add(r["material_id"])
    top_suppliers = sorted([[v, len(s)] for v, s in vendor_sku.items()],
                           key=lambda x: -x[1])[:5]

    # order-type risk proxy by plant share
    order_risk = []
    for st in ship_tos:
        strows = [r for r in crows if r["ship_to"] == st]
        tot = sum(_f(r["revenue"]) for r in strows) or 1
        plant_share = {}
        for r in strows:
            plant_share[r.get("plant") or "Other"] = plant_share.get(r.get("plant") or "Other", 0) + _f(r["revenue"])
        dom = max(plant_share.items(), key=lambda x: x[1]) if plant_share else ("—", 0)
        order_risk.append({"ship_to": st, "jit_pct": 0.0, "stock_pct": 0.0,
                           "dominant_type": dom[0], "dominant_pct": round(dom[1] / tot * 100, 1)})

    return {
        "growth": {"growers": growers, "decliners": decliners, "all": growth_all},
        "margin_flags": margin_flags, "high_margin": high_margin,
        "whitespace": whitespace, "order_risk": order_risk,
        "cost_risk": cost_risk, "top_suppliers": top_suppliers,
        "last3_months": last3, "prior3_months": prior3,
    }


def _shape_battlecard(client, cd, ins, months):
    def money(x):
        return "${:,.0f}".format(x)
    snapshot = [
        f"{money(cd['total_revenue'])} in trailing revenue across {len(cd['ship_tos'])} ship-to location(s), at {cd['avg_gtm_pct']}% blended GTM margin.",
    ]
    if cd["shipto_data"]:
        top = max(cd["shipto_data"].values(), key=lambda s: s["revenue"])
        snapshot.append(f"Largest location: {top['ship_to']} ({top['city']}, {top['state']}) at {money(top['revenue'])}.")
    growth_bullets = [
        (f"{g['hier']}: new line, now {money(g['last3_revenue'])}/qtr — lean in, this is momentum to build on."
         if g["pct_change"] == 999 else
         f"{g['hier']}: +{g['pct_change']:.0f}% vs prior quarter, now {money(g['last3_revenue'])}/qtr — lean in.")
        for g in ins["growth"]["growers"][:4]
    ] or ["No strong growth lines this period."]
    whitespace_bullets = [
        f"{w['ship_to']} does not yet buy {w['hier']}, which {w['peer_count']} sister location(s) purchase (avg {money(w['avg_peer_revenue'])}/location). Cross-sell candidate."
        for w in ins["whitespace"][:5]
    ] or ["No whitespace gaps detected."]
    margin_bullets = [
        f"{m['ship_to']} / {m['hier']}: {m['gtm_pct']}% GTM on {money(m['revenue'])} — {'critically thin, review pricing now.' if m['flag']=='critical-low-margin' else 'below target band.'}"
        for m in ins["margin_flags"][:4]
    ] or ["Margin profile is healthy across the account."]
    risk_bullets = [
        f"{c['material']} at {c['ship_to']}: variance flag (PO {c['po_variance_pct']}% / Std {c['std_cost_variance_pct']}%) — validate vendor cost before next PO."
        for c in ins["cost_risk"][:5]
    ]
    if ins["top_suppliers"]:
        v = ins["top_suppliers"][0]
        risk_bullets.append(f"Supply concentration: {v[0]} supplies {v[1]} of this account's SKUs — single-source risk worth a backup-vendor conversation.")
    plays = [
        "Open with momentum: acknowledge the fastest-growing line and ask what's driving it; volunteer to lock in pricing/service.",
        "Bring a sample/quote for the top whitespace category — sister sites already buy it, so it's a warm cross-sell, not a cold pitch.",
        "Get ahead of thin-margin lines with a value-add (consolidated freight, kitting, spec review) rather than a straight price increase.",
        "Flag cost-variance items with procurement before the next renewal so you know your real cost position.",
        "Use the ship-to drilldown live in the meeting — showing the customer their own multi-site data builds strategic credibility.",
    ]
    return {"snapshot": snapshot, "growth_bullets": growth_bullets,
            "whitespace_bullets": whitespace_bullets, "margin_bullets": margin_bullets,
            "risk_bullets": risk_bullets, "plays": plays}


# ============================================================================
#  GENIE  (AI/BI Genie Conversation API)
# ============================================================================
def genie_ask(question: str):
    """
    Start a Genie conversation, poll for completion, and return the generated
    SQL plus the query result rows.
    Docs: https://docs.databricks.com/api/workspace/genie
    """
    base = f"{CONFIG['DATABRICKS_HOST']}/api/2.0/genie/spaces/{CONFIG['GENIE_SPACE_ID']}"
    headers = {
    "Authorization": f"Bearer {CONFIG['TOKEN']}",
    "Content-Type": "application/json"
}
    # 1. start conversation
    start = requests.post(f"{base}/start-conversation",
                          headers= headers,
                          json={"content": question}, timeout=60, verify=VERIFY_SSL)
    start.raise_for_status()
    sb = start.json()
    conversation_id = sb["conversation_id"]
    message_id = sb["message_id"]

    # 2. poll the message until COMPLETED
    msg_url = f"{base}/conversations/{conversation_id}/messages/{message_id}"
    deadline = time.time() + 90
    status = None
    mb = {}
    while time.time() < deadline:
        m = requests.get(msg_url, headers=_headers(), timeout=60, verify=VERIFY_SSL)
        m.raise_for_status()
        mb = m.json()
        status = mb.get("status")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
            break
        time.sleep(1.5)

    if status != "COMPLETED":
        return {"status": status or "UNKNOWN", "text": mb.get("error", "Genie did not complete."), "sql": None, "rows": []}

    # 3. pull text + any attached query (SQL + results)
    text_parts, sql, rows, cols = [], None, [], []
    for att in mb.get("attachments", []):
        if "text" in att and att["text"].get("content"):
            text_parts.append(att["text"]["content"])
        if "query" in att and att["query"]:
            sql = att["query"].get("query")
            attachment_id = att["attachment_id"]
            # fetch the materialized result for this query attachment
            res_url = f"{msg_url}/attachments/{attachment_id}/query-result"
            rr = requests.get(res_url, headers=_headers(), timeout=60, verify=VERIFY_SSL)
            if rr.ok:
                rb = rr.json().get("statement_response", {})
                man = rb.get("manifest", {})
                cols = [c["name"] for c in man.get("schema", {}).get("columns", [])]
                data = rb.get("result", {}).get("data_array", []) or []
                rows = [dict(zip(cols, row)) for row in data]

    return {
        "status": "COMPLETED",
        "text": "\n".join(text_parts).strip(),
        "sql": sql,
        "columns": cols,
        "rows": rows[:200],
        "conversation_id": conversation_id,
    }


# ============================================================================
#  GOVERNANCE — GUARDRAILS · AUDIT TRAIL · LINEAGE
#  Added to support the three new dashboard tabs. All logic is server-side.
#  When a real SQL warehouse + token are configured the audit trail persists
#  to a Delta table and supports Time Travel; otherwise it runs in-memory so
#  the tabs still work without any Databricks connection.
# ============================================================================

# The Delta table the audit trail writes to / reads from.
AUDIT_TABLE = os.environ.get("DBX_AUDIT_TABLE", "ernest_audit_log")

# Governance thresholds.
HITL_CONF = int(os.environ.get("HITL_CONF", "65"))  # confidence below this -> HITL

# Whether we have enough config to talk to a live warehouse. The uploaded
# CONFIG ships dummy values, so we only treat it as live when the token and
# warehouse id have been overridden away from their dummy defaults.
LIVE = not (
    CONFIG["TOKEN"].startswith("dapiXXXX")
    or CONFIG["WAREHOUSE_ID"] == "0123456789abcdef"
    or CONFIG["DATABRICKS_HOST"].startswith("https://REPLACE-WORKSPACE")
)

# Gold/base tables & views this app is allowed to touch — used for lineage
# matching. Includes the two source tables plus the audit table.
KNOWN_TABLES = [
    CONFIG["SALES_TABLE"],
    CONFIG["INVENTORY_TABLE"],
    AUDIT_TABLE,
]


def audit_fqtn() -> str:
    return fqtn(AUDIT_TABLE)


# ---------------------------------------------------------------- GUARDRAILS -
# Regexes ported from the Vantage client-side guardrail logic.
PII_RE = re.compile(
    r"\bssn\b|social security|\d{3}-\d{2}-\d{4}|credit card|\bdob\b|"
    r"date of birth|customer name|account holder",
    re.IGNORECASE,
)
JB_RE = re.compile(
    r"ignore (all )?previous|system prompt|jailbreak|bypass|exfiltrate|"
    r"reveal your|disregard|dump the schema|drop table|delete from",
    re.IGNORECASE,
)
HV_RE = re.compile(r"\$\s?\d{3,}|discount|approve|credit limit|write[- ]?off|refund",
                   re.IGNORECASE)


def mask_pii(text: str) -> str:
    """Mask SSNs, card numbers and emails before anything reaches a model."""
    text = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN MASKED]", text)
    text = re.sub(r"\b\d{16}\b", "[CARD MASKED]", text)
    text = re.sub(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b", "[CARD MASKED]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL MASKED]", text)
    return text


def guardrail_check(text: str) -> dict:
    """Evaluate a piece of text against the guardrails.

    Returns action (BLOCK | MASK | PASS) plus the individual signals so the
    caller can decide whether to proceed, mask, or route to human review.
    """
    text = text or ""
    blocked = bool(JB_RE.search(text))
    pii = bool(PII_RE.search(text))
    high_value = bool(HV_RE.search(text))
    masked_text = mask_pii(text) if pii else text

    action = "BLOCK" if blocked else "MASK" if pii else "PASS"
    reason = (
        "Prompt-injection / unsafe operation blocked." if blocked else
        "PII detected and masked before any model call." if pii else
        "High-value action — routed to human review." if high_value else
        "Passed all guardrails."
    )
    return {
        "action": action, "blocked": blocked, "pii": pii,
        "high_value": high_value, "masked_text": masked_text,
        "hitl": high_value, "reason": reason,
    }


# --------------------------------------------------------------- AUDIT TRAIL -
AUDIT_LOG = []                      # session rows, newest first
_audit_lock = threading.Lock()
_history_loaded = False


def _trace_id() -> str:
    return "TRC-" + uuid.uuid4().hex[:6].upper()


def _now_hm() -> str:
    return datetime.now().strftime("%H:%M")


def _decision_type(row: dict) -> str:
    decision = (row.get("decision") or "")
    guard = row.get("guard", "PASS")
    if row.get("hitl"):
        if decision.startswith("Approved"):
            return "HITL_APPROVE"
        if decision.startswith("Rejected"):
            return "HITL_REJECT"
        return "HITL_REVIEW"
    if guard == "BLOCK":
        return "GUARDRAIL_BLOCK"
    if guard == "MASK":
        return "PII_MASK"
    return "AI_RESPONSE"


def write_audit_row(row: dict):
    """Fire-and-forget INSERT into the Delta audit table (live mode only)."""
    if not LIVE:
        return
    try:
        def esc(v):
            return str(v).replace("'", "''")[:200]

        conf = row.get("conf")
        sql = f"""INSERT INTO {audit_fqtn()}
          (trace_id, question_text, model_used, guardrail_action, pii_masked,
           hitl_triggered, confidence_score, cost_usd, decision_type, decided_by, decided_at)
        VALUES (
          '{esc(row.get('trace_id') or _trace_id())}',
          '{esc(row.get('decision'))}',
          '{esc(row.get('model') or 'unknown')}',
          '{esc(row.get('guard') or 'PASS')}',
          {str(bool(row.get('pii'))).lower()},
          {str(bool(row.get('hitl'))).lower()},
          {"NULL" if conf is None else int(conf)},
          {float(row.get('cost') or 0):.6f},
          '{_decision_type(row)}',
          '{esc(row.get('decided_by') or 'system')}',
          current_timestamp())"""
        run_sql(sql)
    except Exception as e:
        log.warning("[audit] write failed (suppressed): %s", e)


def add_audit(row: dict) -> dict:
    entry = dict(row)
    entry.setdefault("trace_id", _trace_id())
    entry["time"] = _now_hm()
    entry.setdefault("guard", "PASS")
    entry.setdefault("cost", 0.0)
    with _audit_lock:
        AUDIT_LOG.insert(0, entry)
    write_audit_row(entry)
    return entry


def session_stats() -> dict:
    with _audit_lock:
        rows = list(AUDIT_LOG)
    return {
        "total_decisions": len(rows),
        "hitl_triggered": sum(1 for r in rows if r.get("hitl")),
        "pii_masked": sum(1 for r in rows if r.get("pii")),
        "session_cost": round(sum(float(r.get("cost") or 0) for r in rows), 4),
    }


def _truthy(v):
    return v is True or str(v).lower() == "true"


def _map_delta_row(r: dict) -> dict:
    decided_at = str(r.get("decided_at") or "")
    return {
        "trace_id": r.get("trace_id") or "",
        "decision": r.get("question_text") or "Historical entry",
        "model": r.get("model_used") or "unknown",
        "guard": r.get("guardrail_action") or "PASS",
        "pii": _truthy(r.get("pii_masked")),
        "hitl": _truthy(r.get("hitl_triggered")),
        "conf": r.get("confidence_score"),
        "cost": float(r.get("cost_usd") or 0),
        "time": decided_at[11:16] or "—",
        "historical": True,
    }


def load_audit_history(limit: int = 50):
    """Load historical rows from Delta once; merge in non-duplicates."""
    global _history_loaded
    if not LIVE or _history_loaded:
        return
    _history_loaded = True
    try:
        sql = f"""SELECT trace_id, question_text, model_used, guardrail_action,
                         pii_masked, hitl_triggered, confidence_score, cost_usd,
                         decision_type, decided_at
                  FROM {audit_fqtn()} ORDER BY decided_at DESC LIMIT {int(limit)}"""
        res = run_sql(sql)
        with _audit_lock:
            seen = {r.get("trace_id") for r in AUDIT_LOG if r.get("trace_id")}
            for r in res:
                if (r.get("trace_id") or "") not in seen:
                    AUDIT_LOG.append(_map_delta_row(r))
    except Exception as e:
        log.warning("[audit] history load failed: %s", e)


def audit_versions():
    """DESCRIBE HISTORY for the Time Travel selector."""
    if not LIVE:
        return []
    try:
        out = []
        for r in run_sql(f"DESCRIBE HISTORY {audit_fqtn()}"):
            ver = r.get("version")
            if ver is None:
                continue
            out.append({
                "version": ver,
                "timestamp": str(r.get("timestamp") or "")[:16].replace("T", " "),
                "operation": r.get("operation", ""),
            })
        return out
    except Exception as e:
        log.warning("[tt] history list failed: %s", e)
        return []


def audit_at_version(version: int, limit: int = 50):
    """SELECT ... VERSION AS OF <n> — Delta Time Travel."""
    if not LIVE:
        raise RuntimeError("Time Travel requires a live Databricks connection.")
    sql = f"""SELECT trace_id, question_text, model_used, guardrail_action,
                     pii_masked, hitl_triggered, confidence_score, cost_usd,
                     decision_type, decided_at
              FROM {audit_fqtn()} VERSION AS OF {int(version)}
              ORDER BY decided_at DESC LIMIT {int(limit)}"""
    return [_map_delta_row(r) for r in run_sql(sql)]


# ------------------------------------------------------------------- LINEAGE -
QUERIED_TABLES = set()              # every known table/view touched this session
QUERIED_TABLE_LOG = {}             # { table: [question, ...] }
QUERY_LINEAGE_LOG = []             # [{ question, sql, tables, cols, row_count, ts, audit_id }]
_lineage_lock = threading.Lock()

_SELECT_RE = re.compile(r"SELECT\s+([\s\S]+?)\s+FROM\s", re.IGNORECASE)
_AS_RE = re.compile(r"\s+AS\s+\w+", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r"[\w]+\.")
_AGG_RE = re.compile(r"SUM|COUNT|AVG|ROUND|MIN|MAX|DISTINCT|CAST", re.IGNORECASE)


def parse_columns_from_sql(sql: str):
    """Extract up to 10 selected column names from a SELECT statement."""
    try:
        m = _SELECT_RE.search(sql)
        if not m:
            return []
        cols = []
        for c in m.group(1).split(","):
            c = _AS_RE.sub("", c)
            c = _QUALIFIER_RE.sub("", c)
            c = _AGG_RE.sub("", c)
            c = re.sub(r"[`()\s]", "", c).strip()
            if 0 < len(c) < 40:
                cols.append(c)
        return cols[:10]
    except Exception:
        return []


def track_queried_tables(sql: str, question: str = "", row_count: int = 0, viz_type: str = "Table"):
    """Record which known tables/views + columns a SQL statement touched."""
    if not sql:
        return {"touched": []}
    cols = parse_columns_from_sql(sql)
    touched = []
    low = sql.lower()
    with _lineage_lock:
        for tbl in KNOWN_TABLES:
            if tbl.lower() in low:
                QUERIED_TABLES.add(tbl)
                touched.append(tbl)
                q = (question or "").strip()[:60]
                if q:
                    QUERIED_TABLE_LOG.setdefault(tbl, [])
                    if q not in QUERIED_TABLE_LOG[tbl]:
                        QUERIED_TABLE_LOG[tbl].append(q)
        if touched:
            with _audit_lock:
                audit_id = AUDIT_LOG[0]["trace_id"] if AUDIT_LOG else "—"
            QUERY_LINEAGE_LOG.insert(0, {
                "question": question or "—",
                "sql": sql,
                "tables": touched,
                "cols": cols,
                "row_count": row_count or 0,
                "viz_type": viz_type or "Table",
                "audit_id": audit_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
            })
    return {"touched": touched, "columns": cols}


def lineage_state():
    with _lineage_lock:
        return {
            "active_tables": sorted(QUERIED_TABLES),
            "table_count": len(QUERIED_TABLES),
            "table_questions": dict(QUERIED_TABLE_LOG),
            "log": list(QUERY_LINEAGE_LOG),
            "known_tables": KNOWN_TABLES,
        }


# ============================================================================
#  ROUTES
# ============================================================================
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def api_config():
    """Non-secret config the front-end may want to display."""
    return jsonify({
        "workspace_id": CONFIG["WORKSPACE_ID"],
        "genie_space_id": CONFIG["GENIE_SPACE_ID"],
        "catalog": CONFIG["CATALOG"],
        "schema": CONFIG["SCHEMA"],
    })


@app.route("/api/data")
def api_data():
    try:
        return jsonify(build_dashboard_payload())
    except Exception as e:
        log.exception("data build failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/genie/ask", methods=["POST"])
def api_genie():
    q = (request.get_json(force=True) or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400

    # Guardrail the question before it reaches Genie.
    verdict = guardrail_check(q)
    if verdict["blocked"]:
        add_audit({"decision": "Blocked · " + q[:60], "model": "BLOCKED",
                   "conf": None, "guard": "BLOCK", "pii": False, "hitl": False,
                   "decided_by": "Genie"})
        return jsonify({"status": "BLOCKED", "text": verdict["reason"],
                        "sql": None, "rows": [], "guardrail": verdict})

    try:
        result = genie_ask(verdict["masked_text"])
        # Audit + lineage from the real Genie response.
        add_audit({
            "decision": ("High-value · " if verdict["high_value"] else "") + q[:60],
            "model": "genie",
            "conf": HITL_CONF - 4 if verdict["high_value"] else 95,
            "guard": verdict["action"],
            "pii": verdict["pii"],
            "hitl": verdict["high_value"],
            "decided_by": "Genie",
        })
        if result.get("sql"):
            track_queried_tables(result["sql"], q, len(result.get("rows", [])), "Genie")
        result["guardrail"] = verdict
        return jsonify(result)
    except Exception as e:
        log.exception("genie failed")
        return jsonify({"error": str(e)}), 500


# ---- GUARDRAILS ------------------------------------------------------------
@app.route("/api/guardrails/check", methods=["POST"])
def api_guardrails_check():
    text = (request.get_json(force=True) or {}).get("text", "")
    return jsonify(guardrail_check(text))


# ---- AUDIT TRAIL -----------------------------------------------------------
@app.route("/api/audit", methods=["GET"])
def api_audit_get():
    load_audit_history()
    with _audit_lock:
        rows = list(AUDIT_LOG)
    return jsonify({"rows": rows, "stats": session_stats(), "live": LIVE})


@app.route("/api/audit", methods=["POST"])
def api_audit_post():
    """Append a decision. Guardrails are re-run server-side so the logged
    guard/pii/hitl fields are always consistent with the guardrail engine."""
    body = request.get_json(force=True) or {}
    question = body.get("decision", "") or body.get("question", "")
    verdict = guardrail_check(question)
    row = {
        "decision": (body.get("decision", question) or "")[:200],
        "model": body.get("model", "unknown"),
        "conf": body.get("conf"),
        "guard": body.get("guard", verdict["action"]),
        "pii": body.get("pii", verdict["pii"]),
        "hitl": body.get("hitl", verdict["high_value"]),
        "cost": body.get("cost", 0.0),
        "decided_by": body.get("decided_by", "system"),
    }
    entry = add_audit(row)
    return jsonify({"entry": entry, "stats": session_stats(), "guardrail": verdict})


@app.route("/api/audit/versions")
def api_audit_versions():
    return jsonify({"versions": audit_versions(), "live": LIVE})


@app.route("/api/audit/version/<int:version>")
def api_audit_version(version):
    try:
        return jsonify({"rows": audit_at_version(version), "version": version})
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 400


# ---- LINEAGE ---------------------------------------------------------------
@app.route("/api/lineage/track", methods=["POST"])
def api_lineage_track():
    body = request.get_json(force=True) or {}
    result = track_queried_tables(
        body.get("sql", ""), body.get("question", ""),
        body.get("row_count", 0), body.get("viz_type", "Table"),
    )
    return jsonify({"tracked": result, "state": lineage_state()})


@app.route("/api/lineage")
def api_lineage_get():
    return jsonify(lineage_state())


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
