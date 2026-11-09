"""
Single source of truth for the prompt format and SQL extraction.

Why this file exists: a fine-tuned model only works well if the prompt it sees
at test time is BYTE-IDENTICAL in format to what it saw during training.
Training scripts and evaluation scripts both import from here, so the two can
never drift apart.
"""

import re
import sqlite3

PROMPT_TEMPLATE = """You are a SQLite expert. Given the database schema and a question, write a single SQL query that answers it.

### Schema:
{schema}

### Question:
{question}

### Evidence:
{evidence}

Return ONLY the SQL query, no explanation, no markdown formatting.

### SQL:
"""


def build_prompt(schema: str, question: str, evidence: str = "") -> str:
    return PROMPT_TEMPLATE.format(schema=schema, question=question, evidence=evidence or "")


def get_schema_ddl(db_path: str) -> str:
    """Full-schema context for Systems A/B: raw CREATE TABLE statements."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    statements = [row[0] for row in cur.fetchall()]
    conn.close()
    return ";\n\n".join(statements) + ";"


def extract_sql(text: str) -> str:
    """Pull the SQL out of raw model output.

    Replaces the earlier `.strip("```sql")` approach, which treated its argument
    as a SET of characters and could eat trailing 's', 'l', 'q' letters off
    real table names (e.g. 'FROM schools' -> 'FROM school').
    """
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
