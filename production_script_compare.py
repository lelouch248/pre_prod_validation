#!/usr/bin/env python3
"""
snowflake_compare_improved.py

- Loop over a list of schemas and compare objects (TABLE, VIEW) between PROD and target env (TST/STG/etc).
- Normalizes DDL more smartly to reduce false positives.
- Produces a Markdown report per run summarizing missing objects and DDL diffs (with one-line summaries).
"""

import os
import re
import sys
import json
from datetime import datetime
from difflib import unified_diff
from dotenv import load_dotenv

# third-party
import pandas as pd
import snowflake.connector

# optional: sql formatting helper (better canonicalization). If not present we fallback.
try:
    import sqlparse
    HAS_SQLPARSE = True
except Exception:
    HAS_SQLPARSE = False

load_dotenv()

# ---------- CONFIG ----------
# Set schema list here or set env var SCHEMA_LIST (comma separated)
SCHEMA_LIST = os.getenv("SCHEMA_LIST", "PUBLIC").split(",")  # default PUBLIC
SCHEMA_LIST = [s.strip() for s in SCHEMA_LIST if s.strip()]

# Objects to compare
SUPPORTED_OBJECTS = ["TABLE", "VIEW"]

# Optional inclusion/exclusion logic (set to None to skip filtering)
# Example: only compare objects in these databases (or None)
INCLUDE_DATABASES = None  # e.g. ["SDM_TST_ONECUSTOMER_DB", "SRC_TST_DATAEDITOR_DB"]
EXCLUDE_OBJECT_PATTERN = None  # regex string to exclude by name if needed

# ---------- CONNECTION UTIL ----------
def get_connection(env_prefix):
    """
    Create a snowflake connection using environment variable prefix.
    env_prefix examples: "PROD", "TST", "STG"
    Required env keys for each prefix:
      {PREFIX}_USER, {PREFIX}_PASSWORD, {PREFIX}_ACCOUNT, {PREFIX}_WAREHOUSE, {PREFIX}_DATABASE, {PREFIX}_ROLE
    """
    user = os.getenv(f"{env_prefix}_USER")
    password = os.getenv(f"{env_prefix}_PASSWORD")
    account = os.getenv(f"{env_prefix}_ACCOUNT")
    warehouse = os.getenv(f"{env_prefix}_WAREHOUSE")
    database = os.getenv(f"{env_prefix}_DATABASE")
    role = os.getenv(f"{env_prefix}_ROLE")

    if not (user and password and account):
        raise ValueError(f"Missing connection env vars for prefix {env_prefix}. Check .env")

    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse or None,
        database=database or None,
        role=role or None,
        client_session_keep_alive=True
    )
    return conn

# ---------- DDL NORMALIZATION ----------
def strip_comments(sql: str) -> str:
    """Remove single-line -- comments and block /* ... */ comments."""
    if not sql:
        return ""
    # remove block comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    # remove line comments
    sql = re.sub(r"--.*?(\r?\n|$)", " ", sql)
    return sql

def replace_env_tokens(sql: str) -> str:
    """Replace environment-specific tokens with a neutral placeholder."""
    # replace _TST_, _PRD_, _DEV_ patterns with _ENV_
    sql = re.sub(r'(_TST_|_PRD_|_DEV_|_PROD_)', '_ENV_', sql, flags=re.I)
    return sql

def remove_quoted_identifiers(sql: str) -> str:
    """Convert "Quoted"."Identifiers" to unquoted form for comparison (lowercased)."""
    if not sql:
        return sql
    # remove double quotes around identifiers
    sql = re.sub(r'"([^"]+)"', r'\1', sql)
    return sql

def canonicalize_fqns(sql: str) -> str:
    """
    Replace fully-qualified names DATABASE.SCHEMA.TABLE or database.schema.table
    with a generic token OBJECT to avoid environment-specific DB/SCHEMA prefixes causing diffs.
    """
    if not sql:
        return sql
    # pattern to match dot-separated names like db.schema.object (3 parts) or schema.object (2)
    # We will replace occurrences of <word>.<word>.<word> with <OBJECT>
    sql = re.sub(r'\b[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+\b', 'OBJECT', sql)
    sql = re.sub(r'\b[A-Za-z0-9_]+\.[A-Za-z0-9_]+\b', 'OBJECT', sql)
    return sql

def whitespace_normalize(sql: str) -> str:
    """Collapse whitespace sequences to single space and normalize punctuation spacing."""
    if not sql:
        return sql
    # convert all whitespace (tabs/newlines) to single space
    sql = re.sub(r'\s+', ' ', sql)
    # remove spaces before commas/parentheses and after open paren
    sql = re.sub(r'\s*,\s*', ',', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    return sql.strip()

def format_with_sqlparse(sql: str) -> str:
    """If sqlparse is available, use it to reformat SQL to a canonical style."""
    if not HAS_SQLPARSE or not sql:
        return sql
    try:
        # Reformat with keyword case upper and strip comments already handled
        formatted = sqlparse.format(sql, keyword_case='upper', reindent=True, strip_whitespace=True)
        # remove leading/trailing whitespace
        return formatted.strip()
    except Exception:
        return sql

def normalize_ddl(ddl: str) -> str:
    """
    Full normalization pipeline returning a canonical string used for comparison.
    The pipeline steps are:
      - strip comments
      - replace env tokens like _TST_/_PRD_
      - remove quotes around identifiers
      - optionally format SQL with sqlparse (if installed)
      - canonicalize fully qualified names
      - whitespace normalization
      - lowercasing (for deterministic compare)
    """
    if ddl is None:
        return ""

    s = ddl
    s = strip_comments(s)
    s = replace_env_tokens(s)
    s = remove_quoted_identifiers(s)
    # optional sqlparse formatting for better canonicalization
    s = format_with_sqlparse(s)
    s = canonicalize_fqns(s)
    s = whitespace_normalize(s)
    # remove any trailing semicolons
    s = s.rstrip(';')
    # finally lower-case for deterministic comparisons
    s = s.lower().strip()
    return s

# ---------- FETCH OBJECTS ----------
def fetch_objects_for_schema(conn, obj_types, database, schema, exclude_pattern=None):
    """
    For a given connection, database and schema, return a dict of objects and their normalized DDL.
    Key format: <OBJECT_TYPE>.<database>.<schema>.<object_name>
    """
    cursor = conn.cursor()
    try:
        if database:
            cursor.execute(f"USE DATABASE IDENTIFIER('{database}')")
        if schema:
            # Use IDENTIFIER if name can have weird chars; otherwise simple USE SCHEMA works.
            cursor.execute(f"USE SCHEMA IDENTIFIER('{schema}')")
    except Exception:
        # fallback to plain USE
        try:
            if database:
                cursor.execute(f"USE DATABASE {database}")
            if schema:
                cursor.execute(f"USE SCHEMA {schema}")
        except Exception:
            pass

    objects = {}

    for obj_type in obj_types:
        try:
            cursor.execute(f"SHOW {obj_type}s IN SCHEMA")
            rows = cursor.fetchall()
            cols = [d[0].lower() for d in cursor.description]
            df = pd.DataFrame(rows, columns=cols)
        except Exception:
            # fallback: try generic SHOW <OBJTYPE>s
            try:
                cursor.execute(f"SHOW {obj_type}s")
                rows = cursor.fetchall()
                cols = [d[0].lower() for d in cursor.description]
                df = pd.DataFrame(rows, columns=cols)
            except Exception:
                df = pd.DataFrame(columns=["name", "database_name", "schema_name"])

        if df.empty:
            # nothing to do for this object type
            continue

        # Optionally filter by the current database/schema, depending on how SHOW results are returned
        if 'database_name' in df.columns:
            df = df[(df['database_name'].str.lower() == (database or '').lower()) &
                    (df['schema_name'].str.lower() == (schema or '').lower())]

        # optional name exclusion
        if exclude_pattern:
            df = df[~df['name'].str.match(exclude_pattern, na=False)]

        for _, row in df.iterrows():
            obj_name = row.get('name')
            if not obj_name:
                continue
            key = f"{obj_type}.{(database or '').upper()}.{(schema or '').upper()}.{obj_name.upper()}"
            try:
                # GET_DDL expects object type and fully qualified name (schema.object) or object name if context set
                # We'll pass just the object name; since we executed USE SCHEMA it's fine.
                cursor.execute(f"SELECT GET_DDL('{obj_type}', '{obj_name}')")
                ddl_row = cursor.fetchone()
                ddl = ddl_row[0] if ddl_row and ddl_row[0] else ""
            except Exception:
                # fallback: attempt with explicit schema.object
                try:
                    cursor.execute(f"SELECT GET_DDL('{obj_type}', '{schema}.{obj_name}')")
                    ddl_row = cursor.fetchone()
                    ddl = ddl_row[0] if ddl_row and ddl_row[0] else ""
                except Exception:
                    ddl = f"-- ERROR: Unable to fetch DDL for {obj_name} --"

            norm = normalize_ddl(ddl)
            objects[key] = {
                "raw_ddl": ddl or "",
                "norm_ddl": norm
            }
        # end for rows

    return objects

# ---------- COMPARISON ----------
def summarize_change_from_diff(diff_text: str) -> str:
    """Return short one-line summary from unified diff text."""
    if not diff_text:
        return "No changes"
    adds = len([l for l in diff_text.splitlines() if l.startswith('+') and not l.startswith('+++')])
    dels = len([l for l in diff_text.splitlines() if l.startswith('-') and not l.startswith('---')])
    if adds and dels:
        return f"Modified ({adds} additions, {dels} deletions)"
    if adds:
        return f"Added ({adds} additions)"
    if dels:
        return f"Removed ({dels} deletions)"
    return "Formatting changes"

def compare_envs(prod_objs: dict, tgt_objs: dict):
    """
    Compare two dicts of objects returned from fetch_objects_for_schema.
    prod_objs and tgt_objs keys: OBJECTTYPE.DATABASE.SCHEMA.NAME
    Values have 'raw_ddl' and 'norm_ddl'
    Returns dict: { missing_in_prod: [...], changed: [(key, diff, summary, raw_prod, raw_tgt)] }
    """
    missing_in_prod = []
    changed = []

    # check each object in target
    for key, tgt_val in tgt_objs.items():
        prod_val = prod_objs.get(key)
        if prod_val is None:
            missing_in_prod.append(key)
            continue
        # Compare normalized DDLs
        prod_norm = prod_val.get('norm_ddl', '')
        tgt_norm = tgt_val.get('norm_ddl', '')

        if prod_norm != tgt_norm:
            # token-aware diff to reduce noise: split on non-alphanum and compare tokens lines
            prod_tokens = re.split(r'(\W)', prod_norm)
            tgt_tokens = re.split(r'(\W)', tgt_norm)
            diff = "\n".join(unified_diff(prod_tokens, tgt_tokens, fromfile='PROD', tofile='TGT', lineterm=''))
            summary = summarize_change_from_diff(diff)
            changed.append((key, diff, summary, prod_val.get('raw_ddl', ''), tgt_val.get('raw_ddl', '')))
    return missing_in_prod, changed

# ---------- REPORT GENERATION (Markdown) ----------
def generate_markdown_report(report_path: str, all_findings: dict, prod_label="PROD", tgt_label="TGT"):
    """
    all_findings is a dict keyed by schema -> { missing: [...], changed: [...] }
    Writes a markdown file with structured sections for each schema.
    """
    lines = []
    lines.append(f"# Snowflake Comparison Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    lines.append(f"- PROD label: `{prod_label}`")
    lines.append(f"- Target label: `{tgt_label}`")
    lines.append("")
    for schema, findings in all_findings.items():
        lines.append(f"## Schema: `{schema}`")
        lines.append("")
        missing = findings.get('missing', [])
        changed = findings.get('changed', [])

        lines.append(f"### Missing in {prod_label} ({len(missing)})")
        lines.append("")
        if missing:
            for k in missing:
                lines.append(f"- ❌ `{k}`")
        else:
            lines.append("- None")
        lines.append("")

        lines.append(f"### Objects with DDL differences ({len(changed)})")
        lines.append("")
        if changed:
            # summary table
            lines.append("| Object | Summary |")
            lines.append("|---|---|")
            for item in changed:
                obj_key, _, summary, _, _ = item
                lines.append(f"| `{obj_key}` | {summary} |")
            lines.append("")
            # detailed diffs
            for obj_key, diff, summary, raw_prod, raw_tgt in changed:
                lines.append(f"#### `{obj_key}` — {summary}")
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Show unified token diff (click)</summary>")
                lines.append("")
                lines.append("```diff")
                lines.append(diff)
                lines.append("```")
                lines.append("</details>")
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Show raw PROD DDL</summary>")
                lines.append("")
                lines.append("```sql")
                lines.append(raw_prod[:100000])  # limit if extremely long
                lines.append("```")
                lines.append("</details>")
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Show raw TGT DDL</summary>")
                lines.append("")
                lines.append("```sql")
                lines.append(raw_tgt[:100000])
                lines.append("```")
                lines.append("</details>")
                lines.append("")
        else:
            lines.append("- None")
        lines.append("---")
        lines.append("")

    md = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return report_path

# ---------- MAIN ----------
def main(target_env_prefix="TST"):
    prod_prefix = "PROD"
    tgt_prefix = target_env_prefix.upper()

    print(f"[INFO] Using schema list: {SCHEMA_LIST}")
    print(f"[INFO] Connecting to {prod_prefix} and {tgt_prefix} ...")

    prod_conn = get_connection(prod_prefix)
    tgt_conn = get_connection(tgt_prefix)

    all_findings = {}

    # By default use database specified in env var for each connection
    prod_db = os.getenv(f"{prod_prefix}_DATABASE")
    tgt_db = os.getenv(f"{tgt_prefix}_DATABASE")

    for schema in SCHEMA_LIST:
        schema = schema.strip()
        print(f"[INFO] Comparing schema: {schema}")

        # Fetch objects per schema
        prod_objs = fetch_objects_for_schema(prod_conn, SUPPORTED_OBJECTS, prod_db, schema, EXCLUDE_OBJECT_PATTERN)
        tgt_objs = fetch_objects_for_schema(tgt_conn, SUPPORTED_OBJECTS, tgt_db, schema, EXCLUDE_OBJECT_PATTERN)

        print(f"  PROD objects fetched: {len(prod_objs)}")
        print(f"  TGT objects fetched:  {len(tgt_objs)}")

        missing, changed = compare_envs(prod_objs, tgt_objs)
        all_findings[schema] = {
            "missing": missing,
            "changed": changed
        }

    # build markdown report
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"snowflake_comparison_report_{tgt_prefix}_{now}.md"
    report_path = os.path.join(os.getcwd(), report_name)
    generate_markdown_report(report_path, all_findings, prod_label=prod_prefix, tgt_label=tgt_prefix)
    print(f"[DONE] Report generated: {report_path}")

    # close connections
    try:
        prod_conn.close()
    except Exception:
        pass
    try:
        tgt_conn.close()
    except Exception:
        pass

    return report_path

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "TST"
    report = main(target)
    print("Report file:", report)