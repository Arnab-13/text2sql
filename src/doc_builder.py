"""
Converts a schema KB into flat text documents the retriever can index.
One document per column, one per table -- this is what BM25/embeddings
actually search over.
"""

def column_to_doc(col: dict) -> str:
    """Text representation of one column, folding in everything that
    might help a retriever match it to a natural-language question."""
    parts = [f"Table: {col['table']}", f"Column: {col['column']}"]
    if col["description"]:
        parts.append(f"Description: {col['description']}")
    if col["value_description"]:
        parts.append(f"Value meaning: {col['value_description']}")
    if col["is_primary_key"]:
        parts.append("This is a primary key.")
    if col["sample_values"]:
        vals = ", ".join(str(v) for v in col["sample_values"][:3])
        parts.append(f"Example values: {vals}")
    return " | ".join(parts)


def table_to_doc(table_name: str, kb: dict) -> str:
    """Text representation of a whole table: its columns and how it
    connects to other tables via foreign keys."""
    cols_in_table = [c["column"] for c in kb["columns"] if c["table"] == table_name]
    parts = [f"Table: {table_name}", f"Columns: {', '.join(cols_in_table)}"]

    related = [fk for fk in kb["foreign_keys"]
               if fk["from_table"] == table_name or fk["to_table"] == table_name]
    if related:
        rel_strs = []
        for fk in related:
            rel_strs.append(f"{fk['from_table']}.{fk['from_col']} -> {fk['to_table']}.{fk['to_col']}")
        parts.append(f"Relationships: {'; '.join(rel_strs)}")
    return " | ".join(parts)


def build_documents(db_id: str, kb: dict) -> dict:
    """Returns {'column_docs': [{'doc_id', 'text', 'table', 'column'}, ...],
                'table_docs': [{'doc_id', 'text', 'table'}, ...]}
    doc_id is used later to map a retrieval hit back to the actual schema element."""
    column_docs = []
    for col in kb["columns"]:
        column_docs.append({
            "doc_id": f"{db_id}::{col['table']}::{col['column']}",
            "text": column_to_doc(col),
            "db_id": db_id,
            "table": col["table"],
            "column": col["column"],
        })

    table_docs = []
    for t in kb["tables"]:
        table_docs.append({
            "doc_id": f"{db_id}::{t}",
            "text": table_to_doc(t, kb),
            "db_id": db_id,
            "table": t,
        })

    return {"column_docs": column_docs, "table_docs": table_docs}
