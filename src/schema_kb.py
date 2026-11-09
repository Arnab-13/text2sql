"""
Builds a structured knowledge base for one database: every column with its
description, every table, and the PK/FK relationships between them. This is
the raw material the retriever will search over -- get this right first,
everything downstream depends on it.
"""

import sqlite3
import pandas as pd
import os


def get_foreign_keys(db_path: str) -> list:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    fks = []
    for t in tables:
        cur.execute(f"PRAGMA foreign_key_list(`{t}`)")
        for row in cur.fetchall():
            fks.append({
                "from_table": t,
                "from_col": row[3],
                "to_table": row[2],
                "to_col": row[4],
            })
    conn.close()
    return fks


def get_primary_keys(db_path: str) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    pks = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info(`{t}`)")
        pk_cols = [row[1] for row in cur.fetchall() if row[5] > 0]
        pks[t] = pk_cols
    conn.close()
    return pks


def get_sample_values(db_path: str, table: str, column: str, n: int = 3) -> list:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT {n}")
        vals = [row[0] for row in cur.fetchall()]
        conn.close()
        return vals
    except Exception:
        return []


def search_value_in_db(db_path: str, candidate: str, max_hits: int = 5) -> list:
    """Search every TEXT column in the database for an exact (case-insensitive)
    match of `candidate`. Returns [{table, column, matched_value}, ...].

    This is deliberately a targeted search, not a reuse of the small random
    sample_values -- a value like 'Alameda' has no reliable chance of showing
    up in a 3-value random sample out of ~58 counties, so we search directly.

    Cost note: this runs one query per TEXT column per candidate word, so it's
    fine for a handful of question-words at inference time, but NOT something
    to run per-training-example at scale without caching -- see retriever.py's
    caller for how we limit which words we search.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    hits = []
    for t in tables:
        cur.execute(f"PRAGMA table_info(`{t}`)")
        cols = cur.fetchall()
        for col_row in cols:
            col_name, col_type = col_row[1], col_row[2]
            if col_type and "CHAR" not in col_type.upper() and "TEXT" not in col_type.upper():
                continue
            try:
                cur.execute(
                    f"SELECT DISTINCT `{col_name}` FROM `{t}` "
                    f"WHERE `{col_name}` = ? COLLATE NOCASE LIMIT 1",
                    (candidate,)
                )
                row = cur.fetchone()
                if row:
                    hits.append({"table": t, "column": col_name, "matched_value": row[0]})
                    if len(hits) >= max_hits:
                        conn.close()
                        return hits
            except Exception:
                continue
    conn.close()
    return hits


def find_value_matches(db_path: str, question: str, min_word_len: int = 4) -> list:
    """Extracts candidate proper-noun-like words from the question (crude but
    effective: capitalized words, length >= min_word_len, skip common question
    words) and searches for each as an exact value in the DB.
    Returns a flat list of {table, column, matched_value, matched_word}.
    """
    import re
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", question)
    skip = {"What", "Which", "How", "Where", "When", "Who", "The", "For", "In", "Of", "Is", "Are"}
    candidates = [w for w in words if len(w) >= min_word_len and w[0].isupper() and w not in skip]

    matches = []
    for word in candidates:
        hits = search_value_in_db(db_path, word)
        for h in hits:
            h["matched_word"] = word
            matches.append(h)
    return matches


def load_descriptions(desc_dir: str) -> dict:
    tables = {}
    for fname in os.listdir(desc_dir):
        if not fname.endswith(".csv"):
            continue
        table_name = fname[:-4]
        path = os.path.join(desc_dir, fname)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        tables[table_name] = df
    return tables


def build_schema_kb(db_root: str, db_id: str) -> dict:
    db_path = f"{db_root}/{db_id}/{db_id}.sqlite"
    desc_dir = f"{db_root}/{db_id}/database_description"

    fks = get_foreign_keys(db_path)
    pks = get_primary_keys(db_path)
    descriptions = load_descriptions(desc_dir) if os.path.exists(desc_dir) else {}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    columns = []
    for t in tables:
        cur.execute(f"PRAGMA table_info(`{t}`)")
        table_cols = cur.fetchall()

        desc_df = descriptions.get(t)

        for col_row in table_cols:
            col_name = col_row[1]
            col_type = col_row[2]

            description = ""
            value_description = ""
            if desc_df is not None:
                match = desc_df[desc_df["original_column_name"].astype(str).str.strip() == col_name]
                if len(match) > 0:
                    row = match.iloc[0]
                    description = str(row.get("column_description", "") or "")
                    value_description = str(row.get("value_description", "") or "")
                    if description == "nan":
                        description = ""
                    if value_description == "nan":
                        value_description = ""

            columns.append({
                "table": t,
                "column": col_name,
                "type": col_type,
                "description": description,
                "value_description": value_description,
                "is_primary_key": col_name in pks.get(t, []),
                "sample_values": get_sample_values(db_path, t, col_name),
            })

    conn.close()
    return {"tables": tables, "columns": columns, "foreign_keys": fks}
