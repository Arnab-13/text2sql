"""
Turns retrieved columns into an actual prompt-ready schema context.

Three steps, each solving a distinct problem:
1. Aggregate columns -> tables (a table is "relevant" if any of its columns
   were retrieved; value-matched columns count as strong signal).
2. FK-graph expansion: if the question needs columns from two tables that
   aren't directly connected, find the bridge table(s) needed to JOIN them.
   Without this, the model gets fk-disconnected table fragments and can't
   write a valid join.
3. Render as compact text: full column list per selected table (not just
   the retrieved ones -- a model needs to see a table's real structure to
   use it correctly), with descriptions inline, plus a relationships section.
"""

from collections import deque


def aggregate_columns_to_tables(retrieved_columns: list, top_m_tables: int = 6) -> list:
    """Returns table names ranked by relevance, best-first.
    value_match columns (score=None) are treated as maximally relevant --
    they came from an exact match to real data, which is a strong signal
    even though it's not comparable to the BM25/dense fused score scale.
    """
    VALUE_MATCH_SCORE = 999.0  # sentinel: always outranks text-based scores

    table_scores = {}
    for col in retrieved_columns:
        score = col["score"] if col["score"] is not None else VALUE_MATCH_SCORE
        t = col["table"]
        table_scores[t] = max(table_scores.get(t, -1), score)

    ranked = sorted(table_scores.items(), key=lambda x: -x[1])
    return [t for t, _ in ranked[:top_m_tables]]


def build_fk_graph(foreign_keys: list) -> dict:
    """Undirected adjacency list over tables, built from FK relationships."""
    graph = {}
    for fk in foreign_keys:
        a, b = fk["from_table"], fk["to_table"]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def shortest_path(graph: dict, start: str, end: str) -> list:
    """BFS shortest path between two tables in the FK graph. Returns the
    list of tables on the path (inclusive of start/end), or [] if unreachable."""
    if start == end:
        return [start]
    if start not in graph or end not in graph:
        return []

    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []  # no path found -- tables are in disconnected components


def expand_with_fk_bridges(selected_tables: list, foreign_keys: list) -> list:
    """Given a set of relevant tables, add any bridge tables needed to
    connect them all via foreign keys, so the model can actually write
    JOINs between them. Uses pairwise shortest-path and unions the results
    -- not a true minimal Steiner tree, but good enough for schemas this
    size, and simple enough to reason about and debug.
    """
    graph = build_fk_graph(foreign_keys)
    expanded = set(selected_tables)

    for i, t1 in enumerate(selected_tables):
        for t2 in selected_tables[i + 1:]:
            path = shortest_path(graph, t1, t2)
            expanded.update(path)  # empty list if unreachable -- fine, just adds nothing

    return list(expanded)


def render_schema_context(kb: dict, selected_tables: list) -> str:
    """Renders the final compact schema text for the prompt: one CREATE
    TABLE-style block per selected table, full column list (not just
    retrieved columns -- the model needs the whole table shape to use it
    correctly), with descriptions inline, plus a relationships section.
    """
    lines = []

    cols_by_table = {}
    for col in kb["columns"]:
        cols_by_table.setdefault(col["table"], []).append(col)

    def quote_ident(name: str) -> str:
        """Backtick-quote any identifier with a space or special character,
        matching the convention BIRD's gold SQL actually uses (e.g.
        `Free Meal Count (K-12)`). A plain identifier like CDSCode is left
        unquoted since that matches gold SQL style too -- consistency with
        gold SQL's own quoting habits matters more than always/never quoting."""
        if name.isidentifier():
            return name
        return f"`{name}`"

    for t in selected_tables:
        if t not in cols_by_table:
            continue  # defensive: shouldn't happen, but don't crash the prompt build
        lines.append(f"CREATE TABLE {quote_ident(t)} (")
        n_cols = len(cols_by_table[t])
        for i, col in enumerate(cols_by_table[t]):
            is_last = (i == n_cols - 1)
            piece = f"  {quote_ident(col['column'])} {col['type']}"
            if col["is_primary_key"]:
                piece += " PRIMARY KEY"
            sep = "" if is_last else ","  # comma goes BEFORE the comment, not after
            if col["description"]:
                piece += f"{sep}  -- {col['description']}"
            else:
                piece += sep
            lines.append(piece)
        lines.append(");\n")

    relevant_fks = [
        fk for fk in kb["foreign_keys"]
        if fk["from_table"] in selected_tables and fk["to_table"] in selected_tables
    ]
    if relevant_fks:
        lines.append("-- Relationships:")
        for fk in relevant_fks:
            lines.append(f"-- {fk['from_table']}.{fk['from_col']} = {fk['to_table']}.{fk['to_col']}")

    return "\n".join(lines)


def build_retrieved_context(kb: dict, retrieved_columns: list, top_m_tables: int = 6) -> str:
    """Full pipeline: retrieved columns -> relevant tables -> FK-expanded ->
    rendered schema text. This is the single function the generation step calls."""
    selected = aggregate_columns_to_tables(retrieved_columns, top_m_tables)
    expanded = expand_with_fk_bridges(selected, kb["foreign_keys"])
    return render_schema_context(kb, expanded)
