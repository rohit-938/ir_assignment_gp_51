"""
phrase_query.py
───────────────
Phrase query processing for IR Assignment 1.
Covers:
  - Biword Index construction + querying
  - Positional Index construction + querying
  - Side-by-side comparison (including false positive demo)
  - Results as DataFrames for Streamlit display
"""

import time
from preprocessing import preprocess, to_lowercase
import re
import pandas as pd
from collections import defaultdict

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _tokenzie_query(text: str) -> list[str]:
    """
    Lowercase and tokenize a query string into clean word tokens.
    Strips punctuation-only tokens.

    Example:
        _tokenize_query("Information Retrieval System")
        → ['information', 'retrieval', 'system']
    """
    result = preprocess(
        text, lowercase=True, rm_stopwords=False, do_lemma=False, do_stem=False
    )
    return [t for t in result["lower_tokens"] if re.search(r"[a-z0-9]", t)]


def _preprocesss_doc(text: str) -> list[str]:
    """
    Tokenize and lowercase a document for index building.
    Stop words are kept so positional offsets stay accurate.

    Returns a flat list of lowercase word tokens.
    """
    result = preprocess(
        text, lowercase=True, do_lemma=False, do_stem=False, rm_stopwords=False
    )
    return [t for t in result["lower_tokens"] if re.search(r"[a-z0-9]", t)]


# ─────────────────────────────────────────────
#  1. Biword Index
# ─────────────────────────────────────────────


def build_biword_index(documents: dict[str, str]) -> dict[str, set[str]]:
    """
    Build a biword index from a document collection.

    A biword index pairs every consecutive token into a bigram.
    For tokens [t1, t2, t3, t4]:
        bigrams → (t1,t2), (t2,t3), (t3,t4)

    Args:
        documents: {doc_id: raw_text}

    Returns:
        {biword_phrase: {doc_id1, doc_id2, ...}}

    Example:
        doc = "information retrieval system"
        index → {
            "information retrieval": {"doc1"},
            "retrieval system":      {"doc1"},
        }
    """

    index = defaultdict(set)

    for doc_id, text in documents.items():
        tokens = _preprocesss_doc(text)
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i+1]}"
            index[biword].add(doc_id)

    return dict(index)


def query_biword(query: str, index: dict[str, set[str]]) -> dict:
    """
    Answer a phrase query using the biword index.

    Process:
        1. Tokenize query → [t1, t2, t3]
        2. Extract biwords → ["t1 t2", "t2 t3"]
        3. Look up each biword in the index → posting sets
        4. Intersect all posting sets → candidate docs

    Args:
        query: raw phrase query string
        index: biword index from build_biword_index()

    Returns dict with:
        query_tokens    — tokenized query
        biwords         — extracted biword pairs
        postings        — {biword: {doc_ids}} for each biword
        results         — final intersected doc IDs
        missing_biwords — biwords not found in index
        steps           — human-readable processing log
        elapsed_ms      — time took
    """

    start = time.perf_counter()
    tokens = _tokenzie_query(query)
    steps = []

    if len(tokens) < 2:
        return {
            "query_tokens": tokens,
            "biwords": [],
            "postings": {},
            "results": set(),
            "missing_biwords": [],
            "steps": ["Query must have at least 2 words for biword matching."],
            "elapsed_ms": 0.0,
        }
    # Step 1 — extract biwords from query
    biwords = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    steps.append(f"Query Tokens : {tokens}")
    steps.append(f"Biwords : {biwords}")

    postings = {}
    missing_biwords = []

    # Step 2 — look up each biword
    for w in biwords:
        if w in index:
            postings[w] = index[w]
            steps.append(f"Found '{w}' -> {sorted(index[w])}")
        else:
            postings[w] = set()
            missing_biwords.append(w)
            steps.append(f"NOT FOUND '{w} -> No Documents")

    # Step 3 — intersect all posting sets
    if missing_biwords:
        result_docs = set()
        steps.append(f"Result : Missing biwords {biwords}")
    else:
        sets = [postings[w] for w in biwords]
        result_docs = sets[0].copy()
        for s in sets[1:]:
            result_docs &= s
        steps.append(f"Intersection : {sorted(result_docs)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)

    return {
        "query_tokens": tokens,
        "biwords": biwords,
        "postings": postings,
        "results": result_docs,
        "missing_biwords": missing_biwords,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
    }


def biwords_index_to_dataframe(index: dict[str, set[str]]) -> pd.DataFrame:
    """Convert a biword index to a displayable DataFrame."""

    rows = [
        {
            "Biwords": w,
            "Document Frequency": len(docs),
            "Posting List": ", ".join(sorted(docs)),
        }
        for w, docs in sorted(index.items())
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  2. POSITIONAL INDEX
# ─────────────────────────────────────────────


def build_positional_index(
    documents: dict[str, str],
) -> dict[str, dict[str, list[int]]]:
    """
    Build a positional index from a document collection.

    Stores the exact position of every token in every document.
    Position counting starts at 0.

    Args:
        documents: {doc_id: raw_text}

    Returns:
        {term: {doc_id: [pos1, pos2, ...]}}

    Example:
        doc = "the cat sat on the cat mat"
        index["cat"] → {"doc1": [1, 5]}
    """

    index = defaultdict(lambda: defaultdict(list))

    for doc_id, text in documents.items():
        tokens = _preprocesss_doc(text)

        for p, token in enumerate(tokens):
            index[token][doc_id].append(p)

    return {term: dict(doc_map) for term, doc_map in index.items()}


def query_postional(query: str, index: dict[str, dict[str, list[int]]]) -> dict:
    """
    Answer a phrase query using the positional index.

    Process:
        1. Tokenize query → [t1, t2, t3]
        2. Look up each term → positional postings
        3. Find documents containing ALL query terms
        4. In those docs, check for consecutive positions
           (pos of t2 = pos of t1 + 1, pos of t3 = pos of t1 + 2, etc.)

    Args:
        query: raw phrase query string
        index: positional index from build_positional_index()

    Returns dict with:
        query_tokens  — tokenized query
        postings      — {term: {doc_id: [positions]}} for each query term
        candidates    — docs containing all terms (before position check)
        results       — docs where terms appear consecutively
        false_positives — docs eliminated by position check
        steps         — human-readable processing log
        elapsed_ms    — time took
    """

    start = time.perf_counter()
    tokens = _tokenzie_query(query)
    steps = []

    if len(tokens) < 1:
        return {
            "query_tokens": [],
            "postings": {},
            "candidates": set(),
            "results": set(),
            "false_positives": set(),
            "steps": ["⚠ Empty query."],
            "elapsed_ms": 0.0,
        }

    steps.append(f"Query tokens : {tokens}")

    # Step 1 — look up each term
    postings = {}
    missing_terms = []

    for term in tokens:
        if term in index:
            postings[term] = index[term]
            steps.append(f"'{term}' found in {list(index[term].keys())}")
        else:
            postings[term] = {}
            missing_terms.append(term)
            steps.append(f"'{term}' NOT in index")

    if missing_terms:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        return {
            "query_tokens": tokens,
            "postings": postings,
            "candidates": set(),
            "results": set(),
            "false_positives": set(),
            "steps": steps + [f"Result: ∅ (missing terms: {missing_terms})"],
            "elapsed_ms": elapsed_ms,
        }

    # Step 2 — candidate docs (contain ALL query terms)
    doc_sets = [set(postings[t].keys()) for t in tokens]
    candidates = doc_sets[0].copy()
    for s in doc_sets[1:]:
        candidates &= s

    steps.append(f"Candidate docs (all terms present): {sorted(candidates)}")

    # Step 3 — positional check for consecutive positions
    results = set()
    false_positives = set()

    for doc_id in candidates:
        # Get positions of first query token
        start_positions = postings[tokens[0]][doc_id]

        # For each starting position, check if all subsequent
        # tokens appear at offset +1, +2, +3 ...
        phrase_found = False
        for start_pos in start_positions:
            match = True
            for offset, term in enumerate(tokens[1:], start=1):
                required_pos = start_pos + offset
                if required_pos not in postings[term].get(doc_id, []):
                    match = False
                    break
            if match:
                phrase_found = True
                steps.append(
                    f"'{doc_id}': phrase found starting at position {start_pos} ✓"
                )
                break

        if phrase_found:
            results.add(doc_id)
        else:
            false_positives.add(doc_id)
            steps.append(
                f"'{doc_id}': terms present but NOT consecutive — false positive eliminated ✗"
            )

    steps.append(f"Final results : {sorted(results)}")
    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)

    return {
        "query_tokens": tokens,
        "postings": postings,
        "candidates": candidates,
        "results": results,
        "false_positives": false_positives,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
    }


def positional_index_to_dataframe(
    index: dict[str, dict[str, list[int]]],
) -> pd.DataFrame:
    """Convert a positional index to a displayable DataFrame."""

    rows = []

    for t, doc_map in sorted(index.items()):
        for doc_id, p in sorted(doc_map.items()):
            rows.append(
                {
                    "Term": t,
                    "Document": doc_id,
                    "Positions": str(p),
                    "Document frequency": len(p),
                }
            )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  3. COMPARISON — biword vs positional
# ─────────────────────────────────────────────


def compare_indexes(
    query: str,
    documents: dict[str, str],
    biword_idx: dict[str, set[str]],
    pos_idx: dict[str, dict[str, list[int]]],
):
    """
    Run the same query through both indexes and return a
    side-by-side comparison DataFrame.

    Highlights false positives caught by the positional index
    but missed by the biword index.

    Returns DataFrame with columns:
        Document | In Biword Results | In Positional Results | Note
    """
    bw_result = query_biword(query, biword_idx)
    pos_result = query_postional(query, pos_idx)

    bw_docs = bw_result["results"]
    pos_docs = pos_result["results"]
    fp_docs = pos_result["false_positives"]

    all_docs = sorted(documents.keys())
    rows = []

    for doc_id in all_docs:
        in_bw = doc_id in bw_docs
        in_pos = doc_id in pos_docs
        is_fp = doc_id in fp_docs

        if is_fp:
            note = "⚠ False positive — biword matched but terms not consecutive"
        elif in_bw and in_pos:
            note = "✅ True match — both indexes agree"
        elif in_pos and not in_bw:
            note = "ℹ Positional only — biword missed this doc"
        elif in_bw and not in_pos:
            note = "⚠ Biword only — positional rejected (not consecutive)"
        else:
            note = "—"

        rows.append(
            {
                "Document": doc_id,
                "Biword Match": "✅" if in_bw else "❌",
                "Positional Match": "✅" if in_pos else "❌",
                "Note": note,
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  4. FALSE POSITIVE DEMO
# ─────────────────────────────────────────────


def false_positive_demo() -> dict:
    """
    Demonstrate a case where biword index returns a false positive
    that the positional index correctly rejects.

    Returns a dict with:
        documents    — the demo document set
        query        — the phrase query used
        explanation  — what happened and why
        biword_idx   — the biword index built
        pos_idx      — the positional index built
        bw_result    — biword query result
        pos_result   — positional query result
    """
    # Carefully crafted so biword finds a match but
    # the words are NOT actually adjacent as a phrase
    documents = {
        "demo_doc1": (
            "information systems are used for retrieval of data. "
            "retrieval information is key in modern search."
        ),
        "demo_doc2": ("information retrieval is the core of any search engine."),
    }

    query = "information retrieval"

    biword_idx = build_biword_index(documents)
    pos_idx = build_positional_index(documents)

    bw_result = query_biword(query, biword_idx)
    pos_result = query_postional(query, pos_idx)

    explanation = (
        "demo_doc1 contains both 'information' and 'retrieval' "
        "but they are NOT adjacent — 'retrieval' appears before 'information' "
        "in the second sentence. The biword index finds the biword "
        "'information retrieval' only in demo_doc2. "
        "The positional index verifies consecutive positions and correctly "
        "returns only demo_doc2 as a true match."
    )

    return {
        "documents": documents,
        "query": query,
        "explanation": explanation,
        "biword_idx": biword_idx,
        "pos_idx": pos_idx,
        "bw_result": bw_result,
        "pos_result": pos_result,
    }


# # ─────────────────────────────────────────────
# #  QUICK SELF-TEST  (run: python phrase_query.py)
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
#     print("PHRASE QUERY SELF-TEST")
#     print("=" * 60)

#     # ── Build indexes ──
#     print("\nBuilding biword index...")
#     bw_idx = build_biword_index(sample)
#     print(f"  Biword index size: {len(bw_idx)} biwords")

#     print("\nBuilding positional index...")
#     pos_idx = build_positional_index(sample)
#     print(f"  Positional index size: {len(pos_idx)} terms")

#     # ── Biword query ──
#     query = "information retrieval"
#     print(f"\n── Biword query: '{query}'")
#     bw_result = query_biword(query, bw_idx)
#     for step in bw_result["steps"]:
#         print(f"  {step}")
#     print(f"  Results  : {sorted(bw_result['results'])}")
#     print(f"  Time     : {bw_result['elapsed_ms']} ms")

#     # ── Positional query ──
#     print(f"\n── Positional query: '{query}'")
#     pos_result = query_postional(query, pos_idx)
#     for step in pos_result["steps"]:
#         print(f"  {step}")
#     print(f"  Results         : {sorted(pos_result['results'])}")
#     print(f"  False positives : {sorted(pos_result['false_positives'])}")
#     print(f"  Time            : {pos_result['elapsed_ms']} ms")

#     # ── Comparison table ──
#     print(f"\n── Comparison table")
#     comp_df= compare_indexes(query, sample, bw_idx, pos_idx)
#     print(comp_df.to_string(index=False))

#     # ── False positive demo ──
#     print("\n── False positive demonstration")
#     demo = false_positive_demo()
#     print(f"  Query     : '{demo['query']}'")
#     print(f"  Biword    : {sorted(demo['bw_result']['results'])}")
#     print(f"  Positional: {sorted(demo['pos_result']['results'])}")
#     print(f"  Explanation: {demo['explanation']}")
