import sqlite3

def get_schema_ddl(db_path: str) -> str:
    """Pull CREATE TABLE statements directly from the sqlite file.
    This is the 'full schema' context for System A/B — no retrieval, everything included."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    statements = [row[0] for row in cur.fetchall()]
    conn.close()
    return ";\n\n".join(statements) + ";"
