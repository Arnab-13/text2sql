import os
import sqlite3
import tempfile
from harness import score_dataset


def build_toy_db(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            revenue REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        INSERT INTO customers VALUES (1, 'Alice', 'USA');
        INSERT INTO customers VALUES (2, 'Bob', 'UK');
        INSERT INTO customers VALUES (3, 'Carla', 'USA');
        INSERT INTO orders VALUES (1, 1, 100.0);
        INSERT INTO orders VALUES (2, 1, 50.0);
        INSERT INTO orders VALUES (3, 2, 200.0);
        INSERT INTO orders VALUES (4, 3, 300.0);
    """)
    conn.commit()
    conn.close()


def main():
    tmpdir = tempfile.mkdtemp()
    db_root = os.path.join(tmpdir, "databases")
    db_id = "toy_store"
    db_dir = os.path.join(db_root, db_id)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{db_id}.sqlite")
    build_toy_db(db_path)

    gold_sql = (
        "SELECT c.name, SUM(o.revenue) as total_rev "
        "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.name ORDER BY total_rev DESC"
    )

    examples_gold_as_pred = [{"db_id": db_id, "gold_sql": gold_sql, "pred_sql": gold_sql}]

    examples_wrong = [
        {"db_id": db_id, "gold_sql": gold_sql, "pred_sql": "SELECT * FROM nonexistent_table"},
        {"db_id": db_id, "gold_sql": gold_sql, "pred_sql": "SELECT name FROM customers"},
    ]

    equivalent_sql = (
        "select c.name, sum(o.revenue) as total_rev from orders o "
        "join customers c on o.customer_id = c.customer_id "
        "group by c.name order by total_rev desc"
    )
    examples_equivalent = [{"db_id": db_id, "gold_sql": gold_sql, "pred_sql": equivalent_sql}]

    examples_bad_gold = [{"db_id": db_id, "gold_sql": "SELECT * FROM table_that_does_not_exist", "pred_sql": gold_sql}]

    r1 = score_dataset(examples_gold_as_pred, db_root)
    r2 = score_dataset(examples_wrong, db_root)
    r3 = score_dataset(examples_equivalent, db_root)
    r4 = score_dataset(examples_bad_gold, db_root)

    print("=== Case 1: gold-as-prediction ===")
    print(f"  execution_accuracy = {r1['execution_accuracy']:.3f}  (expect 1.000)")
    assert r1["execution_accuracy"] == 1.0

    print("=== Case 2: wrong/broken predictions ===")
    print(f"  execution_accuracy = {r2['execution_accuracy']:.3f}  (expect 0.000)")
    assert r2["execution_accuracy"] == 0.0

    print("=== Case 3: semantically-equivalent query ===")
    print(f"  execution_accuracy = {r3['execution_accuracy']:.3f}  (expect 1.000)")
    assert r3["execution_accuracy"] == 1.0

    print("=== Case 4: invalid gold SQL ===")
    print(f"  n_scored = {r4['n_scored']}  n_skipped_bad_gold = {r4['n_skipped_bad_gold']}  (expect 0 and 1)")
    assert r4["n_skipped_bad_gold"] == 1

    print("\nALL CHECKS PASSED -- harness meets the Week 1 exit criterion.")


if __name__ == "__main__":
    main()
