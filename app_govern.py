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
import logging
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
    try:
        return jsonify(genie_ask(q))
    except Exception as e:
        log.exception("genie failed")
        return jsonify({"error": str(e)}), 500


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
