#!/usr/bin/env python3
"""
prod_sql_validator.py

Production-only SQL pre-deployment validator for Snowflake.
Reads credentials from .env, reads a .sql file with multiple statements,
and validates per-statement:
 - local parsing,
 - EXPLAIN USING TEXT (Snowflake compilation),
 - referenced DB/SCHEMA/TABLE/VIEW existence in PROD,
 - column existence (including columns of referenced views, using declared views in the same file when available),
 - environment checks (ensures references point to PROD DB where qualified).
Outputs per-statement results to terminal.
"""

from __future__ import annotations
import os
import re
import sqlparse
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import traceback

# Optional helper libs
try:
    from sql_metadata import Parser as SQLMetadataParser
except Exception:
    SQLMetadataParser = None

from rich.console import Console
from rich.table import Table

# Snowflake connector
import snowflake.connector

console = Console()

# -------------------------
# Utilities / Normalization
# -------------------------
def load_env():
    load_dotenv()
    env = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "deployment_file": os.getenv("DEPLOYMENT_SQL_FILE", "deployment.sql"),
    }
    missing = [k for k in ("account", "user", "password", "database", "schema") if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing required .env values: {missing}")
    return env

def normalize_ident(name: str) -> str:
    if name is None:
        return None
    n = name.strip()
    # remove quoting if present
    if (n.startswith('"') and n.endswith('"')) or (n.startswith('`') and n.endswith('`')):
        n = n[1:-1]
    return n.upper()

def remove_sql_comments(sql: str) -> str:
    # remove /*...*/ and -- ... endline
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--.*?$", "", sql, flags=re.M)
    return sql

# find top-level keyword location (not in quotes/parens)
def find_top_level_keyword(sql: str, keyword: str, start: int = 0) -> int:
    kw = keyword.upper()
    i = start
    depth = 0
    in_single = False
    in_double = False
    while i < len(sql):
        c = sql[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if c == "(":
                depth += 1
            elif c == ")":
                if depth > 0:
                    depth -= 1
        # if at depth 0 and not in quotes, try match keyword
        if depth == 0 and not in_single and not in_double:
            fragment = sql[i:i+len(kw)]
            if fragment.upper() == kw:
                # ensure word boundary
                prev = sql[i-1] if i>0 else " "
                nxt = sql[i+len(kw)] if i+len(kw) < len(sql) else " "
                if not prev.isalnum() and not nxt.isalnum():
                    return i
        i += 1
    return -1

def split_top_level_commas(text: str) -> List[str]:
    items = []
    buf = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "'" and not in_double:
            in_single = not in_single
            buf.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            buf.append(c)
        elif not in_single and not in_double:
            if c == "(":
                depth += 1
                buf.append(c)
            elif c == ")":
                if depth > 0:
                    depth -= 1
                buf.append(c)
            elif c == "," and depth == 0:
                items.append("".join(buf).strip())
                buf = []
            else:
                buf.append(c)
        else:
            buf.append(c)
        i += 1
    if buf:
        items.append("".join(buf).strip())
    return items

def split_sql_statements(sql_text: str) -> List[str]:
    # use sqlparse.split to split on semicolons more intelligently
    statements = [s.strip() for s in sqlparse.split(sql_text) if s and s.strip()]
    return statements

def split_fqdn(raw: str, default_db: str, default_schema: str) -> Tuple[str,str,str]:
    # Accept formats: db.schema.table OR schema.table OR table
    # Keep original quoting behavior tolerant but normalize to uppercase unquoted
    parts = [p.strip() for p in re.split(r"\.(?=(?:[^\"']*\"[^\"']*\")*[^\"']*$)", raw) if p.strip() != ""]
    parts = [normalize_ident(p) for p in parts]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        db = normalize_ident(default_db)
        return db, parts[0], parts[1]
    elif len(parts) == 1:
        db = normalize_ident(default_db)
        schema = normalize_ident(default_schema)
        return db, schema, parts[0]
    else:
        # fallback
        db = normalize_ident(default_db)
        schema = normalize_ident(default_schema)
        return db, schema, normalize_ident(raw)

# -------------------------
# Snowflake metadata helper
# -------------------------
class SnowflakeMeta:
    def __init__(self, conn):
        self.conn = conn
        self._cols_cache: Dict[Tuple[str,str,str], Set[str]] = {}
        self._table_exists_cache: Dict[Tuple[str,str,str], bool] = {}
        self._schema_exists_cache: Dict[Tuple[str,str], bool] = {}
        self._current_db = None
        self._current_schema = None
        self._fill_current()

    def _exec(self, query: str):
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            return cur
        except Exception:
            cur.close()
            raise

    def _fill_current(self):
        try:
            cur = self._exec("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            row = cur.fetchone()
            if row:
                self._current_db, self._current_schema = row[0], row[1]
            cur.close()
        except Exception:
            # ignore; metadata functions will fallback to env defaults
            pass

    def schema_exists(self, db: str, schema: str) -> bool:
        dbu = normalize_ident(db) if db else self._current_db
        schemau = normalize_ident(schema) if schema else self._current_schema
        key = (dbu, schemau)
        if key in self._schema_exists_cache:
            return self._schema_exists_cache[key]
        q = f"SELECT COUNT(*) FROM \"{dbu}\".INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{schemau}'"
        try:
            cur = self._exec(q)
            cnt = cur.fetchone()[0]
            cur.close()
            exists = bool(cnt and int(cnt) > 0)
        except Exception:
            exists = False
        self._schema_exists_cache[key] = exists
        return exists

    def table_exists(self, db: str, schema: str, table: str) -> bool:
        dbu = normalize_ident(db) if db else self._current_db
        schemau = normalize_ident(schema) if schema else self._current_schema
        tableu = normalize_ident(table)
        key = (dbu, schemau, tableu)
        if key in self._table_exists_cache:
            return self._table_exists_cache[key]
        q = f"SELECT COUNT(*) FROM \"{dbu}\".INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '{schemau}' AND TABLE_NAME = '{tableu}'"
        try:
            cur = self._exec(q)
            cnt = cur.fetchone()[0]
            cur.close()
            exists = bool(cnt and int(cnt) > 0)
        except Exception:
            exists = False
        self._table_exists_cache[key] = exists
        return exists

    def get_columns(self, db: str, schema: str, table: str) -> Set[str]:
        dbu = normalize_ident(db) if db else self._current_db
        schemau = normalize_ident(schema) if schema else self._current_schema
        tableu = normalize_ident(table)
        key = (dbu, schemau, tableu)
        if key in self._cols_cache:
            return self._cols_cache[key]
        q = f"SELECT COLUMN_NAME FROM \"{dbu}\".INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schemau}' AND TABLE_NAME = '{tableu}' ORDER BY ORDINAL_POSITION"
        try:
            cur = self._exec(q)
            rows = cur.fetchall()
            cur.close()
            cols = {r[0].upper() for r in rows} if rows else set()
        except Exception:
            cols = set()
        self._cols_cache[key] = cols
        return cols

    def explain_query(self, sql_query: str) -> Tuple[bool,str]:
        # Runs EXPLAIN USING TEXT <sql_query>
        # returns (True, text) on success, (False, err) on error
        sql_query = sql_query.strip().rstrip(";")
        q = f"EXPLAIN USING TEXT {sql_query}"
        cur = self.conn.cursor()
        try:
            cur.execute(q)
            rows = cur.fetchall()
            cur.close()
            # rows usually a list of text rows describing plan
            txt = "\n".join([r[0] for r in rows]) if rows else ""
            return True, txt
        except Exception as e:
            try:
                cur.close()
            except Exception:
                pass
            return False, str(e)

# -------------------------
# SQL parsing & extraction
# -------------------------
CREATE_OBJ_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+|TEMP\s+)?(?P<otype>TABLE|VIEW|MATERIALIZED\s+VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)(?:\.(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)){0,2})",
    flags=re.IGNORECASE | re.DOTALL,
)

def extract_declared_objects_from_statement(stmt: str) -> Optional[Dict]:
    """
    If statement creates a table/view, return dict:
      {"raw_name": "...", "type": "VIEW"|"TABLE", "normalized": "..."}
    else None
    """
    m = CREATE_OBJ_RE.search(stmt)
    if not m:
        return None
    otype = m.group("otype").upper().replace(" ", "_")
    raw = m.group("name")
    norm = normalize_ident(raw)
    return {"raw_name": raw, "type": otype, "normalized": norm}

def extract_select_from_statement(stmt: str) -> Optional[str]:
    """
    If statement contains a SELECT that is part of the DDL (CREATE ... AS SELECT)
    or is itself a SELECT, return the select SQL (starting with SELECT).
    Otherwise None.
    """
    s = remove_sql_comments(stmt)
    # find first top-level SELECT
    idx = find_top_level_keyword(s, "SELECT")
    if idx == -1:
        return None
    # find if SELECT is inside a CREATE ... AS <SELECT>
    # We simply return substring from SELECT to end (strip trailing ;)
    select_sql = s[idx:].strip()
    if select_sql.endswith(";"):
        select_sql = select_sql[:-1].strip()
    return select_sql

def extract_table_references(stmt: str) -> List[str]:
    """
    Try sql_metadata first; else fallback to regex extraction of FROM/JOIN/INTO/UPDATE/TABLE.
    Returns list of raw references as they appear (not normalized).
    """
    s = remove_sql_comments(stmt)
    s = s.strip()
    tables = []
    if SQLMetadataParser:
        try:
            p = SQLMetadataParser(s)
            tables = list(dict.fromkeys(p.tables or []))
            if tables:
                return tables
        except Exception:
            pass

    # fallback regex: FROM|JOIN|INTO|UPDATE|TABLE <identifier>
    pattern = re.compile(r"(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+(?P<name>(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)(?:\.(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)){0,2})", flags=re.IGNORECASE)
    names = [m.group("name") for m in pattern.finditer(s)]
    # dedupe preserving order
    seen = set()
    dedup = []
    for n in names:
        if n not in seen:
            dedup.append(n)
            seen.add(n)
    return dedup

def extract_qualified_column_refs(stmt: str) -> List[Tuple[str,str]]:
    """
    Find qualifier.column patterns like alias.col or table.col
    Returns list of (qualifier_raw, column_raw)
    """
    s = remove_sql_comments(stmt)
    pattern = re.compile(r"(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\b[a-zA-Z_][a-zA-Z0-9_]*\b)\.(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\b[a-zA-Z_][a-zA-Z0-9_]*\b)")
    pairs = []
    for m in pattern.finditer(s):
        raw = m.group(0)
        left, right = raw.rsplit(".",1)
        pairs.append((left.strip(), right.strip()))
    return pairs

def extract_table_aliases(stmt: str) -> Dict[str,str]:
    """
    Return mapping alias -> raw_table_reference
    Looks for patterns: FROM <table_expr> [AS] alias  and JOIN <table_expr> [AS] alias
    """
    s = remove_sql_comments(stmt)
    pattern = re.compile(r"(?:FROM|JOIN)\s+(?P<table>(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)(?:\.(?:(?:\"[^\"]+\")|(?:`[^`]+`)|\w+)){0,2})(?:\s+(?:AS\s+)?(?P<alias>\w+))?", flags=re.IGNORECASE)
    mapping = {}
    for m in pattern.finditer(s):
        table = m.group("table")
        alias = m.group("alias")
        if alias:
            mapping[alias] = table
        else:
            # try to map base table name to itself as alias
            tblname = table.split(".")[-1].strip('"`')
            mapping[tblname] = table
    return mapping

def parse_select_output_columns(select_sql: str,
                                alias_to_table: Dict[str,str],
                                declared_cols_by_obj: Dict[str, List[str]],
                                meta: SnowflakeMeta,
                                default_db: str,
                                default_schema: str) -> Tuple[List[str], List[str]]:
    """
    Parse the select list and return tuple (resolved_output_columns, warnings)
    resolved_output_columns is a list of column names (upper-case).
    Warnings is a list of strings describing unresolvable situations.
    This function will attempt to expand '*' and alias.* using declared_cols_by_obj or Snowflake metadata.
    """
    warnings = []
    out_cols: List[str] = []

    s = remove_sql_comments(select_sql)
    # find SELECT ... FROM top-level
    sel_idx = find_top_level_keyword(s, "SELECT")
    if sel_idx == -1:
        warnings.append("No SELECT found in select_sql")
        return [], warnings
    from_idx = find_top_level_keyword(s, "FROM", sel_idx+6)
    if from_idx == -1:
        # select without from (e.g., SELECT 1)
        select_list_text = s[sel_idx + 6:].strip().rstrip(";")
    else:
        select_list_text = s[sel_idx + 6:from_idx].strip()

    if not select_list_text:
        warnings.append("Empty select list")
        return [], warnings

    items = split_top_level_commas(select_list_text)
    # helper to expand star
    def expand_star_for_alias(alias_token: Optional[str]) -> List[str]:
        expanded: List[str] = []
        nonlocal warnings
        if alias_token is None:
            # global *
            # expand columns from all tables in alias_to_table order
            for alias, tbl in alias_to_table.items():
                cols = get_columns_for_table_ref(tbl)
                if cols:
                    expanded.extend(list(cols))
                else:
                    warnings.append(f"Could not resolve columns for {tbl} when expanding '*'")
            return expanded
        else:
            # alias.*
            if alias_token in alias_to_table:
                tbl = alias_to_table[alias_token]
                cols = get_columns_for_table_ref(tbl)
                if cols:
                    return list(cols)
                else:
                    warnings.append(f"Could not resolve columns for {tbl} when expanding '{alias_token}.*'")
                    return []
            else:
                # alias might be actual table name with no alias
                if alias_token in alias_to_table.values():
                    # find corresponding key
                    for k,v in alias_to_table.items():
                        if v == alias_token:
                            cols = get_columns_for_table_ref(v)
                            return list(cols) if cols else []
                warnings.append(f"Alias/table '{alias_token}' not found for expansion of star")
                return []

    def get_columns_for_table_ref(tbl_raw: str) -> List[str]:
        # tbl_raw might be qualified or raw. try declared columns first (declared_cols_by_obj keys are normalized)
        db, sch, tab = split_fqdn(tbl_raw, default_db, default_schema)
        norm = f"{db}.{sch}.{tab}"
        if norm in declared_cols_by_obj and declared_cols_by_obj[norm]:
            return declared_cols_by_obj[norm]
        # fallback to production metadata
        cols = meta.get_columns(db, sch, tab)
        return sorted(list(cols)) if cols else []

    for item in items:
        # look for alias via AS or last token
        # patterns:
        #   expr AS alias
        #   expr alias
        #   table.*  or *
        it = item.strip()
        # detect '*' or alias.*
        m_star = re.match(r"^(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*\*|(?P<glstar>\*))$", it)
        if m_star:
            alias_token = m_star.group("alias")
            cols = expand_star_for_alias(alias_token)
            out_cols.extend([c.upper() for c in cols])
            continue

        # detect AS alias
        m_as = re.match(r"(?s)^(?P<expr>.+?)\s+AS\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*$", it, flags=re.IGNORECASE)
        if m_as:
            alias = m_as.group("alias").strip()
            out_cols.append(normalize_ident(alias))
            continue

        # detect last token alias without AS (risky)
        m_last_alias = re.match(r"(?s)^(?P<expr>.+?)\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*$", it)
        if m_last_alias:
            expr = m_last_alias.group("expr").strip()
            alias = m_last_alias.group("alias").strip()
            # ensure expr does not look like a simple column reference; even then alias overrides
            out_cols.append(normalize_ident(alias))
            continue

        # otherwise attempt to extract column name if it's simple col or table.col
        # e.g., "t.col", "col", "COALESCE(t.col,0)"
        m_col = re.match(r"(?:(?P<qual>(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))\.)?(?P<col>(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))$", it)
        if m_col:
            col = m_col.group("col")
            out_cols.append(normalize_ident(col))
            continue

        # fallback: can't infer a reliable output column name (e.g. function without alias)
        warnings.append(f"Could not determine output column name for expression: {it[:200]}")
        # generate a deterministic pseudo-name to still have something
        pseudo = "_EXPR_" + str(abs(hash(it)) % (10**8))
        out_cols.append(pseudo.upper())

    # remove duplicates while preserving order
    seen = set()
    final_cols = []
    for c in out_cols:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)
    return final_cols, warnings

# -------------------------
# Orchestration
# -------------------------
@dataclass
class DeclaredObject:
    raw_name: str
    normalized: str
    otype: str
    stmt_index: int
    stmt_sql: str
    output_columns: List[str] = field(default_factory=list)
    output_warnings: List[str] = field(default_factory=list)

def build_declared_objects_map(statements: List[str], default_db: str, default_schema: str) -> Dict[str, DeclaredObject]:
    """
    Scan statements to find declared objects and build DeclaredObject entries.
    normalized key will be 'DB.SCHEMA.OBJECT' (normalized).
    """
    declared = {}
    for i, stmt in enumerate(statements):
        info = extract_declared_objects_from_statement(stmt)
        if not info:
            continue
        raw = info["raw_name"]
        db, sch, obj = split_fqdn(raw, default_db, default_schema)
        normalized_key = f"{db}.{sch}.{obj}"
        declared[normalized_key] = DeclaredObject(
            raw_name=raw,
            normalized=normalized_key,
            otype=info["type"],
            stmt_index=i,
            stmt_sql=stmt,
        )
    return declared

def compute_declared_objects_output_columns(declared: Dict[str, DeclaredObject],
                                            default_db: str,
                                            default_schema: str,
                                            meta: SnowflakeMeta):
    """
    Attempt to compute output columns for declared views/tables that are AS SELECT or CREATE TABLE (...) definitions.
    We do multi-pass expansion to allow views referencing declared views in same file.
    """
    # alias_to_table will vary per statement; compute per statement inside loop
    changed = True
    passes = 0
    max_passes = max(1, len(declared))  # prevent infinite loop
    # declared_cols_by_obj: normalized -> list[str]
    declared_cols_by_obj: Dict[str, List[str]] = {k: v.output_columns for k,v in declared.items()}
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for key, dobj in declared.items():
            if dobj.output_columns:
                continue  # already computed
            select_sql = extract_select_from_statement(dobj.stmt_sql)
            if not select_sql:
                # maybe CREATE TABLE with explicit column list e.g., CREATE TABLE x (a INT, b STRING)
                # attempt to parse column list
                m = re.search(r"\((.*?)\)", dobj.stmt_sql, flags=re.S)
                if m:
                    cols_text = m.group(1)
                    # split by commas top-level
                    cols = split_top_level_commas(cols_text)
                    colnames = []
                    for c in cols:
                        # capture leading identifier name
                        mcn = re.match(r"\s*(?:\"[^\"]+\"|`[^`]+`|([A-Za-z_][A-Za-z0-9_]*))", c)
                        if mcn:
                            name = mcn.group(1) if mcn.group(1) else c.strip().split()[0]
                            colnames.append(normalize_ident(name))
                    dobj.output_columns = colnames
                    declared_cols_by_obj[key] = dobj.output_columns
                    changed = True
                    continue
                else:
                    # no select and no column list - cannot compute
                    dobj.output_warnings.append("No SELECT or explicit column list found; output columns unknown")
                    continue
            # extract alias mapping for FROM/JOIN in the select body
            alias_map = extract_table_aliases(select_sql)
            # parse output columns (this will attempt to query meta for referenced tables if necessary)
            out_cols, warnings = parse_select_output_columns(select_sql, alias_map, declared_cols_by_obj, meta, default_db, default_schema)
            if out_cols:
                dobj.output_columns = out_cols
                dobj.output_warnings.extend(warnings)
                declared_cols_by_obj[key] = out_cols
                changed = True
            else:
                # leave unresolved; maybe next pass will have declared dependency columns available
                dobj.output_warnings.extend(warnings or ["Could not derive output columns this pass"])
        # end for
    # After passes, try to resolve any remaining declared objects by using production metadata when '*' was unresolved
    for key, dobj in declared.items():
        if dobj.output_columns:
            continue
        select_sql = extract_select_from_statement(dobj.stmt_sql)
        if not select_sql:
            continue
        alias_map = extract_table_aliases(select_sql)
        out_cols, warnings = parse_select_output_columns(select_sql, alias_map, declared_cols_by_obj, meta, default_db, default_schema)
        if out_cols:
            dobj.output_columns = out_cols
            dobj.output_warnings.extend(warnings)
    # finally, any still empty will remain with warnings

def validate_statements(statements: List[str], declared: Dict[str, DeclaredObject], env: dict):
    """
    Main validation loop for each statement.
    """
    # connect to snowflake
    conn = snowflake.connector.connect(
        account=env["account"],
        user=env["user"],
        password=env["password"],
        role=env.get("role"),
        warehouse=env.get("warehouse"),
        database=env.get("database"),
        schema=env.get("schema"),
    )
    meta = SnowflakeMeta(conn)
    default_db = normalize_ident(env["database"])
    default_schema = normalize_ident(env["schema"])
    prod_db = normalize_ident(env["database"])

    summary = {
        "statements": len(statements),
        "errors": 0,
        "warnings": 0,
    }

    # helper to print header for each statement
    for i, stmt in enumerate(statements):
        console.rule(f"[yellow]Statement {i+1}[/yellow]")
        short = stmt.strip().replace("\n", " ")
        console.print(f"[bold]Statement {i+1} (first 240 chars):[/bold] {short[:240]}{'...' if len(short)>240 else ''}\n")

        # 0) local parsing (sqlparse)
        try:
            parsed = sqlparse.parse(stmt)
            if not parsed:
                console.print("[red]✖ Local parse failed (sqlparse returned empty).[/red]")
                summary["errors"] += 1
                # continue to next statement
                continue
            else:
                console.print("[green]✔ Local parse OK (basic sqlparse).[/green]")
        except Exception as e:
            console.print(f"[red]✖ Local parsing threw exception: {e}[/red]")
            summary["errors"] += 1
            continue

        # identify declared object if any
        declared_info = extract_declared_objects_from_statement(stmt)
        declared_normalized = None
        if declared_info:
            db, sch, obj = split_fqdn(declared_info["raw_name"], default_db, default_schema)
            declared_normalized = f"{db}.{sch}.{obj}"
            console.print(f"[blue]Declares object:[/blue] {declared_info['type']} {declared_normalized}")

        # 1) environment check for qualified references -> ensure pointing to prod_db
        table_refs = extract_table_references(stmt)
        for tref in table_refs:
            db, sch, tab = split_fqdn(tref, default_db, default_schema)
            if normalize_ident(db) != prod_db:
                console.print(f"[red]✖ Environment mismatch:[/red] reference {tref} (db={db}) does not point to production DB {prod_db}")
                summary["errors"] += 1

        # 2) EXPLAIN using text on SELECT (if present)
        select_sql = extract_select_from_statement(stmt)
        if select_sql:
            ok, out = meta.explain_query(select_sql)
            if ok:
                console.print("[green]✔ EXPLAIN USING TEXT: Snowflake compiled query successfully.[/green]")
            else:
                console.print(f"[red]✖ EXPLAIN / compilation error from Snowflake:[/red] {out}")
                summary["errors"] += 1
        else:
            console.print("[yellow]! No SELECT portion found for EXPLAIN (DDL without embedded SELECT).[/yellow]")

        # 3) For every referenced table - ensure existence in prod unless it's created in this file
        for tref in table_refs:
            db, sch, tab = split_fqdn(tref, default_db, default_schema)
            norm = f"{db}.{sch}.{tab}"
            if norm in declared:
                console.print(f"[yellow]→ Reference '{tref}' will be created in the same file (skipping existence check).[/yellow]")
            else:
                # check schema
                if not meta.schema_exists(db, sch):
                    console.print(f"[red]✖ Schema not found:[/red] {db}.{sch}")
                    summary["errors"] += 1
                    continue
                # check table/view exist
                if not meta.table_exists(db, sch, tab):
                    console.print(f"[red]✖ Table/View not found in PROD:[/red] {db}.{sch}.{tab}")
                    summary["errors"] += 1
                else:
                    console.print(f"[green]✔ Referenced object exists in PROD:[/green] {db}.{sch}.{tab}")

        # 4) Column existence checks
        # build alias map for this statement
        alias_map = extract_table_aliases(stmt)
        # collect qualified references
        qual_cols = extract_qualified_column_refs(stmt)
        # check each qualified ref (qualifier.col)
        for qual_raw, col_raw in qual_cols:
            qual = normalize_ident(qual_raw)
            col = normalize_ident(col_raw)
            # map qualifier to table reference
            mapped_table = None
            # if qualifier is an alias
            if qual_raw in alias_map:
                mapped_table = alias_map[qual_raw]
            else:
                # maybe qualifier is actual table name, find in references
                # check direct matches (unquoted)
                for tref in table_refs:
                    _,_,tname = split_fqdn(tref, default_db, default_schema)
                    if normalize_ident(tname) == qual:
                        mapped_table = tref
                        break
            if not mapped_table:
                console.print(f"[yellow]! Could not map qualifier '{qual_raw}' to any referenced table for column check of '{qual_raw}.{col_raw}'. Skipping.[/yellow]")
                summary["warnings"] += 1
                continue
            # decide where to get columns: declared map or PROD meta
            db, sch, tab = split_fqdn(mapped_table, default_db, default_schema)
            norm_mapped = f"{db}.{sch}.{tab}"
            cols_set = set()
            if norm_mapped in declared:
                # use declared object's projected columns if available
                declared_cols = declared[norm_mapped].output_columns
                if declared_cols:
                    cols_set = set([c.upper() for c in declared_cols])
                else:
                    # try to fetch from PROD fallback
                    cols_set = meta.get_columns(db, sch, tab)
            else:
                cols_set = meta.get_columns(db, sch, tab)
            if not cols_set:
                console.print(f"[yellow]! Could not load columns for {mapped_table} to validate {qual_raw}.{col_raw} (no metadata).[/yellow]")
                summary["warnings"] += 1
            else:
                if col not in cols_set:
                    console.print(f"[red]✖ Column '{col_raw}' not found in {mapped_table}[/red]")
                    summary["errors"] += 1
                else:
                    console.print(f"[green]✔ Column exists: {mapped_table}.{col_raw}[/green]")

        # 5) Unqualified column checks from SELECT list - attempt to map by alias or single-table inference
        # only do this if SELECT exists
        if select_sql:
            sel_idx = find_top_level_keyword(select_sql, "SELECT")
            from_idx = find_top_level_keyword(select_sql, "FROM", sel_idx+6)
            if sel_idx != -1:
                if from_idx == -1:
                    select_list_text = select_sql[sel_idx+6:].strip()
                    from_tables = []
                else:
                    select_list_text = select_sql[sel_idx+6:from_idx].strip()
                    # get list of referenced table refs
                    from_tables = extract_table_references(select_sql)
                items = split_top_level_commas(select_list_text)
                for item in items:
                    it = item.strip()
                    # skip items that are clearly aliased or qualified (checked above)
                    if "." in it:
                        continue
                    # if item is like 'col AS alias' we check the leading expr if it's a column
                    # attempt to extract a simple column reference
                    m = re.match(r"(?:(?:\"[^\"]+\")|(?:`[^`]+`)|([A-Za-z_][A-Za-z0-9_]*))(\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?$", it, flags=re.IGNORECASE)
                    if not m:
                        continue
                    candidate_col = m.group(1) if m.group(1) else None
                    if not candidate_col:
                        continue
                    colu = normalize_ident(candidate_col)
                    # If only one from_table, map to it
                    if len(from_tables) == 1:
                        db, sch, tab = split_fqdn(from_tables[0], default_db, default_schema)
                        cols = set()
                        norm_mapped = f"{db}.{sch}.{tab}"
                        if norm_mapped in declared:
                            cols = set(declared[norm_mapped].output_columns or [])
                        else:
                            cols = meta.get_columns(db, sch, tab)
                        if cols and colu not in cols:
                            console.print(f"[red]✖ Unqualified column '{candidate_col}' not found in {from_tables[0]}[/red]")
                            summary["errors"] += 1
                        elif cols:
                            console.print(f"[green]✔ Unqualified column '{candidate_col}' likely found in {from_tables[0]}[/green]")
                    elif len(from_tables) > 1:
                        # try to find exactly one table that contains this column
                        found_in = []
                        for tref in from_tables:
                            db, sch, tab = split_fqdn(tref, default_db, default_schema)
                            cols = set()
                            norm_mapped = f"{db}.{sch}.{tab}"
                            if norm_mapped in declared:
                                cols = set(declared[norm_mapped].output_columns or [])
                            else:
                                cols = meta.get_columns(db, sch, tab)
                            if cols and colu in cols:
                                found_in.append(tref)
                        if len(found_in) == 0:
                            console.print(f"[red]✖ Unqualified column '{candidate_col}' not found in any referenced tables: {from_tables}[/red]")
                            summary["errors"] += 1
                        elif len(found_in) == 1:
                            console.print(f"[green]✔ Unqualified column '{candidate_col}' mapped to {found_in[0]}[/green]")
                        else:
                            console.print(f"[yellow]! Unqualified column '{candidate_col}' is ambiguous (found in multiple tables): {found_in}[/yellow]")
                            summary["warnings"] += 1

        # 6) If this statement declares an object and that object had computed warnings, surface them
        if declared_normalized and declared_normalized in declared:
            dobj = declared[declared_normalized]
            if dobj.output_warnings:
                console.print(f"[yellow]Declared object '{declared_normalized}' had warnings while deriving output columns:[/yellow]")
                for w in dobj.output_warnings:
                    console.print(f"  - [yellow]{w}[/yellow]")
                    summary["warnings"] += 1

    # done
    console.rule("[bold green]Validation summary[/bold green]")
    console.print(f"Statements processed: {summary['statements']}")
    console.print(f"Errors found: {summary['errors']}")
    console.print(f"Warnings: {summary['warnings']}")
    conn.close()
    return summary

# -------------------------
# Main execution
# -------------------------
def main():
    try:
        env = load_env()
    except Exception as e:
        console.print(f"[red]Failed to load .env or missing keys:[/red] {e}")
        return

    deployment_file = env["deployment_file"]
    if not os.path.exists(deployment_file):
        console.print(f"[red]Deployment SQL file not found:[/red] {deployment_file}")
        return

    with open(deployment_file, "r", encoding="utf-8") as f:
        sql_text = f.read()

    sql_text_clean = remove_sql_comments(sql_text)
    statements = split_sql_statements(sql_text_clean)
    if not statements:
        console.print("[red]No SQL statements found in the deployment file.[/red]")
        return

    # connect early to build metadata (we will close inside validate)
    conn = snowflake.connector.connect(
        account=env["account"],
        user=env["user"],
        password=env["password"],
        role=env.get("role"),
        warehouse=env.get("warehouse"),
        database=env.get("database"),
        schema=env.get("schema"),
    )
    meta = SnowflakeMeta(conn)
    default_db = normalize_ident(env["database"])
    default_schema = normalize_ident(env["schema"])

    # Build declared objects map
    declared = build_declared_objects_map(statements, default_db, default_schema)
    if declared:
        console.print(f"[blue]Found {len(declared)} declared object(s) in file (will derive their output columns when possible).[/blue]")
    else:
        console.print("[yellow]No CREATE TABLE/VIEW statements detected in file (deploy file might be only DML).[/yellow]")

    # compute declared objects output columns (multi-pass)
    compute_declared_objects_output_columns(declared, default_db, default_schema, meta)

    # close this early connection; validate_statements will open new connection
    conn.close()

    # run validations per statement (this creates new connection internally)
    summary = validate_statements(statements, declared, env)

    if summary["errors"] == 0:
        console.print("[bold green]All checks passed — ready to deploy (no errors).[/bold green]")
    else:
        console.print("[bold red]Errors detected. Please fix before deploying.[/bold red]")

if __name__ == "__main__":
    main()
