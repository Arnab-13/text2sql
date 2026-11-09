"""
Execution-accuracy harness for Text-to-SQL evaluation.

Core idea: a predicted SQL query is "correct" if executing it against the
target database returns the same result set as executing the gold SQL,
NOT if the query text matches. This is what lets us compare A/B/C/D fairly
even when there are many syntactically different but semantically correct
ways to write the same query.

Usage:
    from harness import score_dataset, execute_sql

    results = score_dataset(examples, db_root="/path/to/databases")
    print(results["execution_accuracy"], results["valid_sql_rate"])

`examples` is a list of dicts: {"db_id": ..., "gold_sql": ..., "pred_sql": ...}
"""

import sqlite3
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_TIMEOUT_SECONDS = 30


@dataclass
class ExecResult:
    ok: bool
    rows: Optional[list] = None
    error: Optional[str] = None


def _execute_worker(db_path: str, sql: str, conn_queue: mp.Queue):
    """Runs in a separate process so a hanging/malicious query can be killed."""
    try:
        # Read-only: open with a URI and mode=ro so nothing can be mutated.
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        conn_queue.put(("ok", rows))
    except Exception as e:
        conn_queue.put(("error", str(e)))


def execute_sql(db_path: str, sql: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> ExecResult:
    """Execute `sql` against the SQLite file at `db_path` in an isolated,
    time-limited subprocess. Returns ExecResult(ok, rows, error)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_execute_worker, args=(db_path, sql, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return ExecResult(ok=False, error=f"timeout after {timeout}s")

    if q.empty():
        return ExecResult(ok=False, error="process died without result (crash/segfault)")

    status, payload = q.get()
    if status == "ok":
        return ExecResult(ok=True, rows=payload)
    return ExecResult(ok=False, error=payload)


def _normalize_rows(rows: list) -> frozenset:
    """Order-insensitive, duplicate-preserving-ish comparison.

    We compare as a multiset (Counter over stringified rows) rather than a
    plain set, so a query that duplicates a row isn't incorrectly treated as
    equal to one that doesn't. Row order is ignored (most gold SQL doesn't
    enforce a specific order without ORDER BY, and even with ORDER BY, BIRD's
    own eval treats output order permissively unless the question implies it).
    """
    from collections import Counter
    stringified = [tuple(str(v) for v in row) for row in rows]
    return Counter(stringified)


def results_match(gold_rows: list, pred_rows: list) -> bool:
    return _normalize_rows(gold_rows) == _normalize_rows(pred_rows)


def score_example(example: dict, db_root: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Score a single example. Expects keys: db_id, gold_sql, pred_sql.
    Assumes SQLite files live at {db_root}/{db_id}/{db_id}.sqlite
    (BIRD/Spider convention) — adjust `_db_path` if your layout differs."""
    db_path = _db_path(db_root, example["db_id"])

    gold = execute_sql(db_path, example["gold_sql"], timeout)
    pred = execute_sql(db_path, example["pred_sql"], timeout)

    out = {
        "db_id": example["db_id"],
        "gold_sql": example["gold_sql"],
        "pred_sql": example["pred_sql"],
        "gold_valid": gold.ok,
        "pred_valid": pred.ok,
        "gold_error": gold.error,
        "pred_error": pred.error,
        "correct": False,
    }
    # Pass through any extra keys (question_id, difficulty, etc.) so callers
    # can slice results afterward without re-scoring.
    for k, v in example.items():
        if k not in ("db_id", "gold_sql", "pred_sql"):
            out[k] = v

    if not gold.ok:
        # Gold itself failed to execute -- this is a data problem, flag it
        # loudly rather than silently scoring it as incorrect.
        out["skip_reason"] = "gold_sql_failed"
        return out

    if not pred.ok:
        out["correct"] = False
        return out

    out["correct"] = results_match(gold.rows, pred.rows)
    return out


def _db_path(db_root: str, db_id: str) -> str:
    p = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(
            f"No sqlite file at {p}. Check db_root/layout matches your dataset's actual structure."
        )
    return str(p)


def score_dataset(examples: list, db_root: str, timeout: int = DEFAULT_TIMEOUT_SECONDS,
                   n_workers: int = 1) -> dict:
    """Score a whole list of examples. Set n_workers > 1 to parallelize across
    a process pool once you trust the harness (start with 1 for debugging)."""
    per_example = []

    if n_workers <= 1:
        for ex in examples:
            per_example.append(score_example(ex, db_root, timeout))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(score_example, ex, db_root, timeout) for ex in examples]
            per_example = [f.result() for f in futures]

    scored = [r for r in per_example if r.get("skip_reason") != "gold_sql_failed"]
    skipped = [r for r in per_example if r.get("skip_reason") == "gold_sql_failed"]

    n = len(scored)
    ex_acc = sum(r["correct"] for r in scored) / n if n else 0.0
    valid_rate = sum(r["pred_valid"] for r in scored) / n if n else 0.0

    return {
        "execution_accuracy": ex_acc,
        "valid_sql_rate": valid_rate,
        "n_scored": n,
        "n_skipped_bad_gold": len(skipped),
        "per_example": per_example,
    }
