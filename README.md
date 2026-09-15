# Ernest Account Intelligence Dashboard — Databricks edition

The original dashboard shipped with a large block of data hard-coded into the
HTML. This version keeps the **exact same design and five views** (Overview,
Ship-To Drilldown, Growth & Cross-Sell, Risk Radar, Battle Card) but now:

1. **Pulls data live from Databricks** — `workspace.sss.sales` joined to
   `workspace.sss.inventory` — via the SQL Statement Execution API.
2. **Adds an "Ask Genie" view** — natural-language questions are sent to a
   Databricks AI/BI **Genie space**, which returns generated SQL + live results.

```
ernest_dashboard/
├── app.py               Flask backend: SQL data-shaping + Genie proxy
├── app.yaml             Databricks Apps config (command + env)
├── requirements.txt     Python deps
├── static/
│   └── index.html       The dashboard UI (fetches /api/data, /api/genie/ask)
└── README.md
```

## Where the dummy values live

All of these are in `app.py` under the `CONFIG` dict (and mirrored in
`app.yaml`). **Replace the dummy values with your real ones.**

| Setting | Dummy value | What it is |
|---|---|---|
| `WORKSPACE_ID` | `1234567890123456` | Your Databricks workspace ID |
| `GENIE_SPACE_ID` | `01ef0a1b2c3d4e5f6a7b8c9d0e1f2a3b` | The AI/BI Genie space to query |
| `WAREHOUSE_ID` | `0123456789abcdef` | SQL warehouse that runs the queries |
| `DATABRICKS_HOST` | `https://REPLACE-WORKSPACE…` | Workspace URL |
| `TOKEN` | `dapiXXXX…` | PAT or OAuth token (auto-injected on Databricks Apps) |
| `CATALOG` / `SCHEMA` | `workspace` / `sss` | Where your two tables live |

You can set them either by editing `app.py`, or (preferred) as environment
variables / Databricks app resources — the code reads env vars first.

## Column mapping

The SQL in `build_dashboard_payload()` is written against the exact column
names you provided, e.g. `Sold-To Party3`, `Ship-to party14`,
`Product hierarchy5`, `Extended Amount`, `GTM Dollars`, and the inventory
`Vendor name`, `PO Variance %`, `Standard Cost Variance %`, etc. If any physical
column name differs, change only the string on the **left** of each `AS` in the
`line_sql` query.

> Note on order-type mix: your columns don't include an explicit order-type
> field, so the mix chart currently groups by `Plant`. If you have a real
> order-type column (Stock / JIT / Drop / Dock), swap it into the two
> `otmix` / `st_otmix` blocks.

## Run locally

```bash
pip install -r requirements.txt
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."          # a PAT for local dev
export DATABRICKS_WAREHOUSE_ID="abc123"
export GENIE_SPACE_ID="01ef..."
python app.py            # serves on http://localhost:8000
```

## Deploy as a Databricks App

```bash
databricks apps create ernest-dashboard
databricks sync . "/Workspace/Users/you@co.com/ernest_dashboard"
databricks apps deploy ernest-dashboard \
  --source-code-path "/Workspace/Users/you@co.com/ernest_dashboard"
```

On Databricks Apps, `DATABRICKS_HOST` and an OAuth token are injected
automatically, and the app's service principal is used for auth — so you grant
that principal `SELECT` on `workspace.sss.*`, `CAN USE` on the SQL warehouse,
and `CAN RUN` on the Genie space. Bind the warehouse via the `sql_warehouse`
resource referenced in `app.yaml`.

## API surface

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the dashboard |
| `/api/data` | GET | Runs the SQL, returns the nested dashboard JSON |
| `/api/genie/ask` | POST `{question}` | Genie NL → SQL → rows |
| `/api/config` | GET | Non-secret IDs for display |

## Security note

The browser never sees your Databricks token — all calls go through this
backend, which holds the credential server-side. Don't move the token into
`index.html`; a static page can't hold secrets safely and Databricks APIs
block browser CORS anyway. That's why this ships as a small server, not a lone
HTML file.
