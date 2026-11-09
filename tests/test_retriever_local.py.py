import sys
sys.path.insert(0, "/home/claude/text2sql/retriever")
from retriever import SchemaRetriever

# Synthetic docs mimicking real BIRD structure (frpm + schools tables from
# california_schools), including deliberate DISTRACTOR columns from an
# unrelated table to check the retriever doesn't just grab everything.
column_docs = [
    {"doc_id": "d1", "text": "Table: frpm | Column: Free Meal Count (K-12) | Description: number of students receiving free meals K-12 | Example values: 120, 45, 300", "table": "frpm", "column": "Free Meal Count (K-12)"},
    {"doc_id": "d2", "text": "Table: frpm | Column: Enrollment (K-12) | Description: total K-12 enrollment | Example values: 500, 200, 800", "table": "frpm", "column": "Enrollment (K-12)"},
    {"doc_id": "d3", "text": "Table: frpm | Column: County Name | Description: name of the county | Example values: Alameda, Los Angeles, Fresno", "table": "frpm", "column": "County Name"},
    {"doc_id": "d4", "text": "Table: schools | Column: School Name | Description: name of the school | Example values: Lincoln High, Washington Elementary", "table": "schools", "column": "School Name"},
    {"doc_id": "d5", "text": "Table: schools | Column: CDSCode | Description: unique school code, primary key | Example values: 001, 002", "table": "schools", "column": "CDSCode"},
    # distractors: unrelated table about satscores, should rank low for this question
    {"doc_id": "d6", "text": "Table: satscores | Column: AvgScrMath | Description: average SAT math score | Example values: 500, 450, 600", "table": "satscores", "column": "AvgScrMath"},
    {"doc_id": "d7", "text": "Table: satscores | Column: NumTstTakr | Description: number of test takers | Example values: 100, 200", "table": "satscores", "column": "NumTstTakr"},
]

retriever = SchemaRetriever(column_docs)

# The real BIRD dev question we already saw:
question = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"

results = retriever.retrieve_columns(question, top_k=5)

print(f"Question: {question}\n")
print("Top 5 retrieved columns:")
for r in results:
    print(f"  score={r['score']:.4f}  {r['table']}.{r['column']}")

# Sanity checks
top_tables = set(r["table"] for r in results[:3])
print(f"\nTop-3 tables: {top_tables}")
assert "frpm" in top_tables, "FAIL: frpm table (has the actual answer columns) should be in top results"
print("\nPASS: the table containing the actual gold columns (Free Meal Count, Enrollment, County Name) is retrieved.")