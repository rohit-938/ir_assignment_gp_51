"""
indexing.py
───────────
Index construction for IR Assignment 1.
Covers:
  - Standard inverted index (with TF, DF)
  - Index serialization (save / load)
  - Index statistics
  - Boolean retrieval (AND / OR / NOT)
"""

import time
import math
import json
import re

from preprocessing import preprocess
from collections import defaultdict

import pandas as pd

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _get_tokens(text: str, **kwargs) -> list[str]:
    """Run preprocessing pipeline and return final tokens."""
    tokens = preprocess(text, **kwargs)
    return tokens["final_tokens"]


# ─────────────────────────────────────────────
#  1. INVERTED INDEX  (with TF)
# ─────────────────────────────────────────────


def build_inverted_index(
    documents: dict[str, str],
    lowercase=True,
    rm_stopwords=True,
    do_lemma=False,
    do_stem=False,
) -> dict[str, dict[str, int]]:
    """
    Build an inverted index with term frequency.

    Returns:
        {term: {doc_id: tf, ...}}

    Example:
        index["retrieval"] → {"doc1": 3, "doc3": 1}
    """

    index = defaultdict(lambda: defaultdict(int))

    for doc_id, text in documents.items():
        tokens = _get_tokens(
            text,
            lowercase=lowercase,
            rm_stopwords=rm_stopwords,
            do_lemma=do_lemma,
            do_stem=do_stem,
        )
        for t in tokens:
            index[t][doc_id] += 1

    return {term: dict(doc_map) for term, doc_map in index.items()}


def build_doc_lengths(documents: dict[str, str], **kwargs) -> dict[str, int]:
    """Return {doc_id: total_token_count} for each document."""

    lengths = {}

    for doc_id, text in documents.items():
        tokens = _get_tokens(text, **kwargs)
        lengths[doc_id] = len(tokens)
    return lengths


# ─────────────────────────────────────────────
#  2. TF-IDF WEIGHTS
# ─────────────────────────────────────────────


def compute_tfidf(
    index: dict[str, dict[str, int]], doc_count
) -> dict[str, dict[str, int]]:
    """
    Compute TF-IDF weights for every (term, doc) pair.

    TF  = term_freq / doc_length  (not used here — raw tf kept simple)
    IDF = log(N / df)

    Returns:
        {term: {doc_id: tfidf_score}}
    """

    tfidf = {}

    for term, doc_map in index.items():
        df = len(doc_map)
        idf = math.log(doc_count / df) if df else 0.0
        tfidf[term] = {doc_id: round(tf * idf, 6) for doc_id, tf in doc_map.items()}
    return tfidf


# ─────────────────────────────────────────────
#  BOOLEAN
# ─────────────────────────────────────────────

BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}


def _docs_for_term(term: str, index: dict[str, dict[str, int]]) -> set[str]:
    """Return the posting-list document IDs for a term."""
    postings = index.get(term, {})
    if isinstance(postings, dict):
        return set(postings.keys())
    return set(postings)


def boolean_and(terms: list[str], index: dict[str, dict[str, int]]) -> set[str]:
    """
    AND retrieval — return docs containing ALL terms.

    Example:
        boolean_and(["cat", "dog"], index) → {"doc2", "doc5"}
    """
    if not terms:
        return set()

    sets = [_docs_for_term(t, index) for t in terms]
    st = sets[0].copy()
    for s in sets[1:]:
        st &= s
    return st


def boolean_or(terms: list[str], index: dict[str, dict[str, int]]) -> set[str]:
    """
    OR retrieval — return docs containing ANY term.

    Example:
        boolean_or(["cat", "dog"], index) → {"doc1", "doc2", "doc5"}
    """
    result = set()
    for t in terms:
        result |= _docs_for_term(t, index)
    return result


def boolean_not(
    term: str, index: dict[str, dict[str, int]], all_docs: set[str]
) -> set[str]:
    """
    NOT retrieval — return docs NOT containing the term.

    Example:
        boolean_not("cat", index, all_docs) → {"doc1", "doc3"}
    """

    has_terms = _docs_for_term(term, index)
    return all_docs - has_terms


def _tokenize_boolean_query(query: str) -> list[str]:
    """Tokenize a boolean query, keeping parentheses as separate tokens."""
    raw_tokens = re.findall(r"\(|\)|[^\s()]+", query)
    tokens = []

    for token in raw_tokens:
        upper = token.upper()
        if upper in BOOLEAN_OPERATORS:
            tokens.append(upper)
        else:
            tokens.append(token.lower())

    return tokens


def parse_boolena_query(query: str, index: dict[str, dict[str, int]], all_docs) -> dict:
    """
    Parse and evaluate a boolean query string.
    Supports AND, OR, NOT, nested parentheses, and mixed operators.

    Operator precedence:
        1. NOT
        2. AND
        3. OR

    Supported formats:
        "cat AND dog"
        "cat OR dog"
        "NOT cat"
        "cat AND NOT dog"
        "(cat OR dog) AND NOT fish"
        "cat OR dog AND fish"

    Args:
        query:    raw boolean query string
        index:    inverted index
        all_docs: set of all doc IDs in collection

    Returns dict with:
        query        — original query
        operator     — detected operator (AND/OR/NOT/COMPLEX/INVALID)
        terms        — extracted search terms
        results      — matching doc IDs
        steps        — processing log
    """

    steps = []
    query = query.strip()
    results = set()
    operator = "AND"
    all_docs = set(all_docs)

    tokens = _tokenize_boolean_query(query)
    steps.append(f"Raw tokens : {tokens}")

    if not tokens:
        steps.append("Operator  : AND")
        steps.append("Terms     : []")
        steps.append("Results   : []")
        return {
            "query": query,
            "operator": operator,
            "terms": [],
            "results": results,
            "steps": steps,
        }

    terms = [
        token
        for token in tokens
        if token not in BOOLEAN_OPERATORS and token not in {"(", ")"}
    ]
    used_ops = set()
    implicit_and = False
    pos = 0

    def current():
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected=None):
        nonlocal pos
        token = current()
        if expected is not None and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
        pos += 1
        return token

    def starts_operand(token):
        return token is not None and token not in {"AND", "OR", ")"}

    def parse_primary():
        token = current()

        if token == "(":
            consume("(")
            value = parse_or()
            if current() != ")":
                raise ValueError("Missing closing parenthesis")
            consume(")")
            return value

        if token is None:
            raise ValueError("Unexpected end of query")

        if token in {"AND", "OR", ")"}:
            raise ValueError(f"Unexpected token '{token}'")

        consume()
        return _docs_for_term(token, index)

    def parse_not():
        if current() == "NOT":
            used_ops.add("NOT")
            consume("NOT")
            return all_docs - parse_not()
        return parse_primary()

    def parse_and():
        nonlocal implicit_and
        left = parse_not()

        while True:
            token = current()

            if token == "AND":
                used_ops.add("AND")
                consume("AND")
                left &= parse_not()
            elif starts_operand(token):
                used_ops.add("AND")
                implicit_and = True
                left &= parse_not()
            else:
                break

        return left

    def parse_or():
        left = parse_and()

        while current() == "OR":
            used_ops.add("OR")
            consume("OR")
            left |= parse_and()

        return left

    try:
        results = parse_or()
        if pos != len(tokens):
            raise ValueError(f"Unexpected token '{current()}'")
    except ValueError as exc:
        operator = "INVALID"
        steps.append(f"Error     : {exc}")
        steps.append("Results   : []")
        return {
            "query": query,
            "operator": operator,
            "terms": terms,
            "results": set(),
            "steps": steps,
        }

    if not used_ops:
        operator = "AND"
    elif used_ops == {"AND"}:
        operator = "AND"
    elif used_ops == {"OR"}:
        operator = "OR"
    elif used_ops == {"NOT"}:
        operator = "NOT"
    elif used_ops == {"AND", "NOT"}:
        operator = "AND"
    else:
        operator = "COMPLEX"

    operators_text = ", ".join(sorted(used_ops)) if used_ops else "none"
    steps.append(f"Operator  : {operator}")
    steps.append(f"Operators : {operators_text}")
    steps.append(f"Terms     : {terms}")
    if implicit_and:
        steps.append("Implicit AND applied between adjacent terms/groups")
    steps.append(f"Results   : {sorted(results)}")

    return {
        "query": query,
        "operator": operator,
        "terms": terms,
        "results": results,
        "steps": steps,
    }


# ─────────────────────────────────────────────
#  4. INDEX STATISTICS
# ─────────────────────────────────────────────


def index_statistics(index: dict[str, dict[str, int]], documents: dict[str, str]) -> dict:
    """
    Compute summary statistics for the index.

    Returns dict with:
        vocab_size        — number of unique terms
        total_postings    — total (term, doc) pairs
        avg_df            — average document frequency
        max_df_term       — term with highest doc frequency
        min_df_term       — term with lowest doc frequency
        singleton_terms   — terms appearing in only 1 document
        top_10_terms      — 10 most frequent terms by DF
    """
    vocab_size = len(index)
    total_postings = sum(len(docs) for docs in index.values())
    avg_df = total_postings / vocab_size if vocab_size else 0

    sorted_by_df = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)
    max_df_term = sorted_by_df[0][0] if sorted_by_df else ""
    min_df_term = sorted_by_df[-1][0] if sorted_by_df else ""
    singleton_terms = sum(1 for docs in index.values() if len(docs) == 1)
    top_10 = [(t, len(d)) for t, d in sorted_by_df[:10]]

    return {
        "vocab_size": vocab_size,
        "total_postings": total_postings,
        "avg_df": round(avg_df, 2),
        "max_df_term": max_df_term,
        "min_df_term": min_df_term,
        "singleton_terms": singleton_terms,
        "top_10_terms": top_10,
        "doc_count": len(documents),
    }


def index_to_dataframe(index: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Convert inverted index to a displayable DataFrame."""
    rows = []
    for term, doc_map in sorted(index.items()):
        rows.append(
            {
                "Term": term,
                "Document Frequency": len(doc_map),
                "Term Frequency": sum(doc_map.values()),
                "Posting List": ", ".join(
                    f"{doc}({tf})" for doc, tf in sorted(doc_map.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def stats_to_dataframe(stats: dict) -> pd.DataFrame:
    """Convert index statistics to a displayable DataFrame."""
    rows = [
        {"Metric": "Vocabulary Size", "Value": stats["vocab_size"]},
        {"Metric": "Total Postings", "Value": stats["total_postings"]},
        {"Metric": "Document Count", "Value": stats["doc_count"]},
        {"Metric": "Average Document Frequency", "Value": stats["avg_df"]},
        {"Metric": "Max DF Term", "Value": stats["max_df_term"]},
        {"Metric": "Min DF Term", "Value": stats["min_df_term"]},
        {"Metric": "Singleton Terms", "Value": stats["singleton_terms"]},
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  5. SAVE / LOAD INDEX
# ─────────────────────────────────────────────


def save_index(index: dict, filepath: str) -> None:
    """
    Serialize the index to a JSON file.
    Sets inside postings are converted to sorted lists.
    """
    serializable = {}
    for term, doc_map in index.items():
        if isinstance(doc_map, set):
            serializable[term] = sorted(doc_map)
        elif isinstance(doc_map, dict):
            serializable[term] = {
                doc_id: (sorted(v) if isinstance(v, (set, list)) else v)
                for doc_id, v in doc_map.items()
            }
        else:
            serializable[term] = doc_map

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Index saved → {filepath}")


def load_index(filepath: str) -> dict:
    """Load an index from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# # ─────────────────────────────────────────────
# #  QUICK SELF-TEST  (run: python indexing.py)
# # ─────────────────────────────────────────────

# if __name__ == "__main__":

#     sample = {
#         "doc1": "information retrieval is the activity of obtaining relevant information resources.",
#         "doc2": "a search engine is a well-known software system designed to carry out web searches.",
#         "doc3": "natural language processing is a subfield of linguistics and artificial intelligence.",
#         "doc4": "stemming reduces inflected words to their word stem or root form.",
#         "doc5": "lemmatization groups inflected forms of a word so they can be analysed as a single item.",
#     }

#     print("=" * 60)
#     print("INDEXING SELF-TEST")
#     print("=" * 60)

#     # Build index
#     print("\nBuilding inverted index (stemmed)...")
#     start = time.perf_counter()
#     index = build_inverted_index(sample, rm_stopwords=True, do_stem=True)
#     ms = round((time.perf_counter() - start) * 1000, 3)
#     print(f"  Built in {ms} ms  |  {len(index)} terms")

#     # Top 10
#     print("\nTop 10 terms by DF:")
#     sorted_idx = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)
#     for term, docs in sorted_idx[:10]:
#         print(f"  {term:15s} df={len(docs)}  postings={dict(docs)}")

#     # Statistics
#     print("\nIndex statistics:")
#     stats = index_statistics(index, sample)
#     for k, v in stats.items():
#         if k != "top_10_terms":
#             print(f"  {k:25s}: {v}")

#     # Boolean retrieval
#     print("\nBoolean queries:")
#     all_docs = set(sample.keys())
#     for q in ["information AND retrieval", "stem OR lemma", "NOT search"]:
#         result = parse_boolena_query(q, index, all_docs)
#         print(f"  '{q}' → {sorted(result['results'])}")

#     # TF-IDF
#     print("\nTF-IDF (top 5 terms in doc1):")
#     tfidf = compute_tfidf(index, len(sample))
#     doc1_tfidf = {
#         term: scores["doc1"] for term, scores in tfidf.items() if "doc1" in scores
#     }
#     for term, score in sorted(doc1_tfidf.items(), key=lambda x: x[1], reverse=True)[:5]:
#         print(f"  {term:15s}: {score}")
