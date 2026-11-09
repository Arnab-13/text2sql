"""
Schema-aware retriever. Given a natural-language question and a database's
document set (from doc_builder.py), returns the tables/columns most likely
relevant -- using hybrid search (BM25 + dense embeddings), not just one method
alone, since keyword and semantic matching catch different failure modes:
BM25 catches exact column-name mentions, embeddings catch paraphrases.
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


_embedder = None

def get_embedder():
    """Lazy-load so importing this module doesn't force a model download."""
    global _embedder
    if _embedder is None:
        # small, fast, good enough for schema text -- not a place to spend your GPU budget
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


import re

def tokenize(text: str) -> list:
    """Lowercase and strip punctuation (backticks, parens, commas, etc.) so
    tokens match cleanly. We deliberately keep hyphens inside words (e.g.
    'k-12' stays one token) since splitting it into 'k' and '12' would lose
    a meaningful unit -- but strip surrounding punctuation like backticks,
    parens, and equals signs that add noise without meaning."""
    text = text.lower()
    text = re.sub(r"[`(),=/|:]", " ", text)
    return text.split()


class SchemaRetriever:
    def __init__(self, column_docs: list):
        """column_docs: list of {doc_id, text, table, column, ...} from doc_builder.py"""
        self.docs = column_docs
        self.texts = [d["text"] for d in column_docs]

        # BM25 index
        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense embeddings, computed once and cached
        self.embeddings = get_embedder().encode(self.texts, show_progress_bar=False, convert_to_numpy=True)
        # normalize for cosine similarity via dot product
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def bm25_rank(self, query: str) -> list:
        """Returns doc indices ranked best-first by BM25 score."""
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores))

    def dense_rank(self, query: str) -> list:
        """Returns doc indices ranked best-first by embedding cosine similarity."""
        q_emb = get_embedder().encode([query], convert_to_numpy=True)[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        sims = self.embeddings @ q_emb
        return list(np.argsort(-sims))

    def hybrid_rank(self, query: str, k_rrf: int = 60) -> list:
        """Reciprocal Rank Fusion: combine BM25 and dense rankings without
        needing to normalize/compare raw scores across the two methods
        (which are on incomparable scales). Each doc's fused score is the
        sum of 1/(k + rank) across both rankings -- a doc ranked highly by
        EITHER method gets a boost, which is exactly the point of hybrid search.
        Returns (doc_index, fused_score) sorted best-first.
        """
        bm25_order = self.bm25_rank(query)
        dense_order = self.dense_rank(query)

        fused_scores = {}
        for rank, idx in enumerate(bm25_order):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(dense_order):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (k_rrf + rank + 1)

        ranked = sorted(fused_scores.items(), key=lambda x: -x[1])
        return ranked

    def retrieve_columns(self, query: str, top_k: int = 15, value_matches: list = None) -> list:
        """Returns column doc dicts, each with a 'score' and 'source' field.

        Design: text-based ranking (BM25+dense fusion) and value-matching are
        UNIONED, not blended into one competing score. Blending them (an
        earlier version of this code) caused a real bug: value matches for an
        ambiguous word like 'Alameda' (5 columns contain that literal value)
        scored so consistently high they filled the entire top_k budget and
        pushed out a column (Enrollment (K-12)) that only text/semantic
        matching could find. Keeping the two candidate pools separate means
        one strong signal can never silently starve the other.
        """
        ranked = self.hybrid_rank(query)[:top_k]
        results = []
        seen = set()
        for idx, score in ranked:
            doc = dict(self.docs[idx])
            doc["score"] = score
            doc["source"] = "text"
            results.append(doc)
            seen.add((doc["table"], doc["column"]))

        if value_matches:
            doc_lookup = {(d["table"], d["column"]): idx for idx, d in enumerate(self.docs)}
            for vm in value_matches:
                key = (vm["table"], vm["column"])
                if key in seen or key not in doc_lookup:
                    continue
                idx = doc_lookup[key]
                doc = dict(self.docs[idx])
                doc["score"] = None  # not from the ranked pool, no comparable score
                doc["source"] = "value_match"
                doc["matched_value"] = vm["matched_value"]
                doc["matched_word"] = vm["matched_word"]
                results.append(doc)
                seen.add(key)

        return results
