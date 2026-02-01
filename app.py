import csv
import math
import os
import logging
import threading
import secrets
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from databricks import sql  # Databricks SQL Connector
from crypto_utils import get_secret

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "template"   # your repo uses template/ [1](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector)
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Dataset Explorer (Databricks SQL Warehouse Search + Export)")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
logger = logging.getLogger("uvicorn")

# -----------------------------------------------------------------------------
# SECURITY: Basic Auth (LOGIN VALUES ARE PLAIN)
# -----------------------------------------------------------------------------
security = HTTPBasic()

def _ct_eq(a: str, b: str) -> bool:
    return secrets.compare_digest((a or "").encode("utf-8"), (b or "").encode("utf-8"))

def require_login(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Login gate for UI + APIs.
    APP_AUTH_USER and APP_AUTH_PASS are plain env vars on Render (NOT encrypted).
    """
    user = os.getenv("APP_AUTH_USER", "").strip()
    pwd = os.getenv("APP_AUTH_PASS", "").strip()

    if not user or not pwd:
        raise HTTPException(status_code=500, detail="Auth not configured. Set APP_AUTH_USER and APP_AUTH_PASS.")

    if not (_ct_eq(credentials.username, user) and _ct_eq(credentials.password, pwd)):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# -----------------------------------------------------------------------------
# SECURITY: CORS hardened (optional allow list)
# -----------------------------------------------------------------------------
allowed_origins_env = os.getenv("APP_ALLOWED_ORIGINS", "").strip()
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] if allowed_origins_env else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,      # avoid "*" for better protection [1](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# -----------------------------------------------------------------------------
# SECURITY: Security headers
# -----------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"
    return resp

# -----------------------------------------------------------------------------
# Databricks config (ONLY THESE ARE ENCRYPTED)
# -----------------------------------------------------------------------------
DBR_HOSTNAME = get_secret("DATABRICKS_SERVER_HOSTNAME") or get_secret("DATABRICKS_HOST")
DBR_HTTP_PATH = get_secret("DATABRICKS_HTTP_PATH")
DBR_TOKEN = get_secret("DATABRICKS_TOKEN")

# DB_TABLE is PLAIN (NOT encrypted) ✅
DB_TABLE = os.getenv("DB_TABLE", "").strip()

# Column mapping
COL_MACHINE_ID = os.getenv("COL_MACHINE_ID", "native_pin")
COL_COUNTRY = os.getenv("COL_COUNTRY", "country")
COL_STATE = os.getenv("COL_STATE", "state")
COL_CITY = os.getenv("COL_CITY", "city")

_schema_cache_lock = threading.Lock()
_schema_cache: Dict[str, Any] = {"table": None, "columns": None, "columns_lower": None}

def _require_databricks_config():
    if not DBR_HOSTNAME or not DBR_HTTP_PATH or not DBR_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Databricks not configured. Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN."
        )
    if not DB_TABLE:
        raise HTTPException(status_code=500, detail="DB_TABLE not configured. Set DB_TABLE to catalog.schema.table.")

@contextmanager
def db_connection():
    _require_databricks_config()
    conn = None
    try:
        conn = sql.connect(
            server_hostname=DBR_HOSTNAME,
            http_path=DBR_HTTP_PATH,
            access_token=DBR_TOKEN,
        )  # official connection args
        yield conn
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

def _quote_ident(name: str) -> str:
    return f"`{name.replace('`', '')}`"

def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def _get_table_columns() -> Tuple[List[str], Dict[str, str]]:
    with _schema_cache_lock:
        if _schema_cache["table"] == DB_TABLE and _schema_cache["columns"]:
            return _schema_cache["columns"], _schema_cache["columns_lower"]

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {DB_TABLE} LIMIT 0")
            cols = [d[0] for d in (cur.description or [])]
            cols_lower = {c.lower(): c for c in cols}

    with _schema_cache_lock:
        _schema_cache["table"] = DB_TABLE
        _schema_cache["columns"] = cols
        _schema_cache["columns_lower"] = cols_lower

    return cols, cols_lower

def _resolve_col(name_hint: str, cols_lower: Dict[str, str]) -> Optional[str]:
    return cols_lower.get((name_hint or "").lower())

# -----------------------------------------------------------------------------
# JSON safe conversion
# -----------------------------------------------------------------------------
def _json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None

    if isinstance(v, (np.floating, np.integer)):
        vv = v.item()
        if isinstance(vv, float) and (math.isnan(vv) or math.isinf(vv)):
            return None
        return vv

    try:
        import datetime, decimal
        if isinstance(v, (datetime.datetime, datetime.date)):
            return v.isoformat()
        if isinstance(v, decimal.Decimal):
            try:
                return float(v)
            except Exception:
                return str(v)
    except Exception:
        pass

    return v

def rows_to_json_safe(columns: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    return [{columns[i]: _json_safe_value(r[i]) for i in range(len(columns))} for r in rows]

# -----------------------------------------------------------------------------
# Routes (protected)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, _=Depends(require_login)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/ping")
def ping(_=Depends(require_login)):
    return {"ok": True}

@app.get("/health")
def health(_=Depends(require_login)):
    try:
        cols, _ = _get_table_columns()
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 as ok")
                _ = cur.fetchone()
        return {"status": "ok", "source": "databricks", "table": DB_TABLE, "columns": cols}
    except Exception as e:
        logger.exception("Databricks health failed")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# -----------------------------------------------------------------------------
# SEARCH: MACHINE
# -----------------------------------------------------------------------------
@app.get("/api/search/machine")
def search_machine(
    machine_id: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
    _=Depends(require_login),
):
    mid = machine_id.strip()
    if not mid:
        raise HTTPException(400, "Machine ID required.")

    _, cols_lower = _get_table_columns()
    machine_col = _resolve_col(COL_MACHINE_ID, cols_lower)
    if not machine_col:
        raise HTTPException(400, f"Column '{COL_MACHINE_ID}' not found in Databricks table.")

    like_val = f"%{_escape_like(mid.lower())}%"
    where_sql = f"""
      (
        CAST({_quote_ident(machine_col)} AS STRING) = :mid
        OR LOWER(CAST({_quote_ident(machine_col)} AS STRING)) LIKE :mid_like ESCAPE '\\\\'
      )
    """

    count_sql = f"SELECT COUNT(*) as cnt FROM {DB_TABLE} WHERE {where_sql}"
    data_sql = f"SELECT * FROM {DB_TABLE} WHERE {where_sql} LIMIT {int(limit)}"

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, {"mid": mid, "mid_like": like_val})
            matched = int(cur.fetchone()[0])

            cur.execute(data_sql, {"mid": mid, "mid_like": like_val})
            rows = cur.fetchall()
            out_cols = [d[0] for d in cur.description]

    return JSONResponse(content={
        "query": mid,
        "matched_rows": matched,
        "returned_rows": len(rows),
        "columns": out_cols,
        "rows": rows_to_json_safe(out_cols, rows),
    })

# -----------------------------------------------------------------------------
# SEARCH: LOCATION
# -----------------------------------------------------------------------------
@app.get("/api/search/location")
def search_location(
    q: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
    _=Depends(require_login),
):
    query = q.strip()
    if not query:
        raise HTTPException(400, "Location query required.")

    _, cols_lower = _get_table_columns()
    country_col = _resolve_col(COL_COUNTRY, cols_lower)
    state_col = _resolve_col(COL_STATE, cols_lower)
    city_col = _resolve_col(COL_CITY, cols_lower)
    cols_use = [c for c in [country_col, state_col, city_col] if c]

    if not cols_use:
        raise HTTPException(400, "Country/State/City columns not found in Databricks table.")

    like_val = f"%{_escape_like(query.lower())}%"
    ors = [f"LOWER(CAST({_quote_ident(c)} AS STRING)) LIKE :q_like ESCAPE '\\\\'" for c in cols_use]
    where_sql = "(" + " OR ".join(ors) + ")"

    count_sql = f"SELECT COUNT(*) as cnt FROM {DB_TABLE} WHERE {where_sql}"
    data_sql = f"SELECT * FROM {DB_TABLE} WHERE {where_sql} LIMIT {int(limit)}"

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, {"q_like": like_val})
            matched = int(cur.fetchone()[0])

            cur.execute(data_sql, {"q_like": like_val})
            rows = cur.fetchall()
            out_cols = [d[0] for d in cur.description]

    return JSONResponse(content={
        "query": query,
        "matched_rows": matched,
        "returned_rows": len(rows),
        "columns": out_cols,
        "rows": rows_to_json_safe(out_cols, rows),
    })

# -----------------------------------------------------------------------------
# EXPORT: Stream CSV (NO row limits)
# -----------------------------------------------------------------------------
def stream_cursor_as_csv(conn, sql_text: str, params: Dict[str, Any], filename: str, fetch_size: int = 2000):
    def generate():
        with conn.cursor() as cur:
            cur.execute(sql_text, params)
            columns = [d[0] for d in cur.description] if cur.description else []

            buf = StringIO()
            writer = csv.writer(buf)

            writer.writerow(columns)
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

            while True:
                chunk = cur.fetchmany(fetch_size)
                if not chunk:
                    break
                for row in chunk:
                    row_out = []
                    for v in row:
                        v2 = _json_safe_value(v)
                        if isinstance(v2, float) and (math.isnan(v2) or math.isinf(v2)):
                            v2 = ""
                        row_out.append(v2)
                    writer.writerow(row_out)

                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(generate(), media_type="text/csv", headers=headers)

@app.get("/api/export/machine")
def export_machine(machine_id: str = Query(...), _=Depends(require_login)):
    mid = machine_id.strip()
    if not mid:
        raise HTTPException(400, "Machine ID required.")

    _, cols_lower = _get_table_columns()
    machine_col = _resolve_col(COL_MACHINE_ID, cols_lower)
    if not machine_col:
        raise HTTPException(400, f"Column '{COL_MACHINE_ID}' not found in Databricks table.")

    like_val = f"%{_escape_like(mid.lower())}%"
    where_sql = f"""
      (
        CAST({_quote_ident(machine_col)} AS STRING) = :mid
        OR LOWER(CAST({_quote_ident(machine_col)} AS STRING)) LIKE :mid_like ESCAPE '\\\\'
      )
    """
    export_sql = f"SELECT * FROM {DB_TABLE} WHERE {where_sql}"
    filename = f"machine_{mid}_matched.csv".replace(" ", "_")

    conn = sql.connect(server_hostname=DBR_HOSTNAME, http_path=DBR_HTTP_PATH, access_token=DBR_TOKEN)  #
    return stream_cursor_as_csv(conn, export_sql, {"mid": mid, "mid_like": like_val}, filename)

@app.get("/api/export/location")
def export_location(q: str = Query(...), _=Depends(require_login)):
    query = q.strip()
    if not query:
        raise HTTPException(400, "Location query required.")

    _, cols_lower = _get_table_columns()
    country_col = _resolve_col(COL_COUNTRY, cols_lower)
    state_col = _resolve_col(COL_STATE, cols_lower)
    city_col = _resolve_col(COL_CITY, cols_lower)
    cols_use = [c for c in [country_col, state_col, city_col] if c]
    if not cols_use:
        raise HTTPException(400, "Country/State/City columns not found in Databricks table.")

    like_val = f"%{_escape_like(query.lower())}%"
    ors = [f"LOWER(CAST({_quote_ident(c)} AS STRING)) LIKE :q_like ESCAPE '\\\\'" for c in cols_use]
    where_sql = "(" + " OR ".join(ors) + ")"
    export_sql = f"SELECT * FROM {DB_TABLE} WHERE {where_sql}"
    filename = f"location_{query}_matched.csv".replace(" ", "_")

    conn = sql.connect(server_hostname=DBR_HOSTNAME, http_path=DBR_HTTP_PATH, access_token=DBR_TOKEN)  #
    return stream_cursor_as_csv(conn, export_sql, {"q_like": like_val}, filename)

