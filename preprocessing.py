"""
preprocessing.py
────────────────
Text preprocessing pipeline for IR Assignment 1.
Covers:
  - Tokenization
  - Lowercasing
  - Hyphen handling
  - Stop word removal
  - Stemming  (Porter + Snowball)
  - Lemmatization (WordNet)
  - Inverted index construction
  - Stemming vs Lemmatization comparison (TF-IDF cosine similarity)
"""

import re
import time
from collections import defaultdict
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  Download required NLTK data (silent if already present)
# ─────────────────────────────────────────────

for pkg in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
]:
    nltk.download(pkg, quiet=True)

_porter = PorterStemmer()
_snowball = SnowballStemmer("english")
_lemmer = WordNetLemmatizer()
_stop_world_en = set(stopwords.words("english"))

# ─────────────────────────────────────────────
#  1. TOKENIZATION
# ─────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """
    Split text into tokens using NLTK word_tokenize.
    Keeps punctuation as separate tokens.

    Example:
        tokenize("Hello, world!") → ['Hello', ',', 'world', '!']
    """
    return word_tokenize(text)


def simple_tokenize(text: str) -> list[str]:
    """
    Whitespace-based tokenizer — faster, less precise.
    Strips leading/trailing punctuation from each token.

    Example:
        tokenize_simple("Hello, world!") → ['Hello', 'world']
    """
    return [
        t.strip(string.punctuation) for t in text.split() if t.strip(string.punctuation)
    ]


# ─────────────────────────────────────────────
#  2. LOWERCASING
# ─────────────────────────────────────────────


def to_lowercase(tokens: list[str]) -> list[str]:
    """
    Convert every token to lowercase.

    Example:
        to_lowercase(['Hello', 'World']) → ['hello', 'world']
    """
    return [t.lower() for t in tokens]


# ─────────────────────────────────────────────
#  3. HYPHEN HANDLING
# ─────────────────────────────────────────────


def handle_hypen(tokens: list[str]) -> list[str]:
    """
    For each hyphenated token (e.g. 'well-known'):
      - keep the original compound  ('well-known')
      - add each individual part    ('well', 'known')
    This ensures both the compound and parts are searchable.

    Example:
        handle_hyphens(['well-known', 'dog']) → ['well-known', 'well', 'known', 'dog']
    """
    result = []

    for t in tokens:
        if "-" in t:
            result.append(t)  ## keep compound
            result.extend(t.split("-"))  # add split parts
        else:
            result.append(t)

    return result


# ─────────────────────────────────────────────
#  4. STOP WORD REMOVAL
# ─────────────────────────────────────────────


def get_stopwords() -> set[str]:
    """Return the default English stop word set."""
    return _stop_world_en.copy()


def remove_stopwords(
    tokens: list[str], extra_stopwords: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """
    Remove English stop words from token list.
    Optionally pass extra domain-specific stop words.

    Returns:
        (filtered_tokens, removed_tokens)
    """

    sw = get_stopwords()

    if extra_stopwords:
        sw.update(extra_stopwords)

    filtered = [t for t in tokens if t.lower() not in sw]
    removed = [t for t in tokens if t.lower() in sw]

    return filtered, removed


# ─────────────────────────────────────────────
#  5. STEMMING
# ─────────────────────────────────────────────


def porter_stem(tokens: list[str]) -> list[str]:
    """
    Apply Porter stemming to each token.

    Example:
        stem_porter(['running', 'runs', 'easily']) → ['run', 'run', 'easili']
    """

    return [_porter.stem(t) for t in tokens]


def snowball_stem(tokens: list[str]) -> list[str]:
    """
    Apply Snowball (English) stemming — slightly less aggressive than Porter.

    Example:
        stem_snowball(['generously']) → ['generous']
    """
    return [_snowball.stem(t) for t in tokens]


def stem_with_map(
    tokens: list[str], stemmer: str = "porter"
) -> tuple[list[str], dict[str, str]]:
    """
    Stem tokens and return both the stemmed list and an
    original → stem mapping for display purposes.

    Args:
        tokens:  list of raw tokens
        stemmer: 'porter' or 'snowball'

    Returns:
        (stemmed_tokens, mapping_dict)
    """
    fn = _porter.stem if stemmer == "porter" else _snowball.stem
    stems = [fn(t) for t in tokens]
    mapping = {origin: stem for origin, stem in zip(tokens, stems) if origin != stem}

    return stems, mapping


# ─────────────────────────────────────────────
#  6. LEMMATIZATION
# ─────────────────────────────────────────────


def lemmatize(tokens: list[str], pos: str = "n") -> list[str]:
    """
    Lemmatize tokens using WordNetLemmatizer.

    Args:
        tokens: list of tokens
        pos:    part-of-speech tag — 'n' noun, 'v' verb, 'a' adjective

    Example:
        lemmatize(['running','better','geese'], pos='v') → ['run','better','geese']
    """
    return [_lemmer.lemmatize(t, pos=pos) for t in tokens]


def lemmatize_with_map(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Lemmatize and return both the lemmatized list and an
    original → lemma mapping for display purposes.
    Tries noun, then verb POS for best coverage.

    Returns:
        (lemmatized_tokens, mapping_dict)
    """
    result = []
    mapping = {}
    fn = _lemmer.lemmatize
    for t in tokens:
        lemma = fn(t, pos="n")
        if lemma == t:
            lemma = fn(t, pos="v")  # retry as verb
        result.append(lemma)
        if lemma != t:
            mapping[t] = lemma
    return result, mapping


# ─────────────────────────────────────────────
#  7. FULL PIPELINE
# ─────────────────────────────────────────────


def preprocess(
    text: str,
    lowercase: bool = True,
    handle_hyph: bool = True,
    rm_stopwords: bool = True,
    do_stem: bool = False,
    do_lemma: bool = False,
    stemmer: str = "porter",
) -> dict:
    """
    Run the full preprocessing pipeline on a single text string.

    Args:
        text:         raw input text
        lowercase:    apply lowercasing
        handle_hyph:  expand hyphenated tokens
        rm_stopwords: remove English stop words
        do_stem:      apply stemming
        do_lemma:     apply lemmatization
        stemmer:      'porter' or 'snowball' (used when do_stem=True)

    Returns a dict with every intermediate stage:
    {
        'raw_tokens':    [...],
        'lower_tokens':  [...],
        'hyph_tokens':   [...],
        'clean_tokens':  [...],   # after stop word removal
        'removed_sw':    [...],   # stop words that were removed
        'final_tokens':  [...],   # after stem / lemma
        'stem_map':      {...},   # original → stem  (if stem)
        'lemma_map':     {...},   # original → lemma (if lemma)
    }
    """
    result = {}
    # Step 1 - tokenize

    tokens = tokenize(text)
    result["raw_tokens"] = tokens[:]

    # Step 2 - lowercase

    if lowercase:
        tokens = to_lowercase(tokens)
    result["lower_tokens"] = tokens[:]

    # Step 3 - hypen handling

    if handle_hyph:
        tokens = handle_hypen(tokens)
    result["hyph_token"] = tokens[:]

    # step 4 - remove punctuation-only tokens

    tokens = [t for t in tokens if re.search(r"[a-zA-Z0-9]", t)]

    # step 5 - remove stopwords

    removed_sw = []

    if rm_stopwords:
        tokens, removed_sw = remove_stopwords(tokens)
    result["clean_tokens"] = tokens[:]
    result["removed_sw"] = removed_sw

    # step 6 - stem or lemmatize

    stem_map = {}
    lemma_map = {}

    if do_lemma and do_stem:
        tokens, lemma_map = lemmatize_with_map(tokens)
        tokens, stem_map = stem_with_map(tokens, stemmer)
    elif do_stem:
        tokens, stem_map = stem_with_map(tokens, stemmer)
    elif do_lemma:
        tokens, lemma_map = lemmatize_with_map(tokens)

    result["final_tokens"] = tokens[:]
    result["stem_map"] = stem_map
    result["lemma_map"] = lemma_map

    return result


# ─────────────────────────────────────────────
#  8. INVERTED INDEX
# ─────────────────────────────────────────────


def build_inverted_index(
    documents: dict[str, str], **preprocess_kwargs
) -> dict[str, set[str]]:
    """
    Build an inverted index from a dict of {doc_id: text}.

    Args:
        documents:          {filename: raw_text}
        preprocess_kwargs:  passed directly to preprocess()

    Returns:
        {term: {doc_id1, doc_id2, ...}}

    Example:
        index = build_inverted_index({"d1": "cat sat", "d2": "cat ran"})
        index["cat"] → {"d1", "d2"}
    """
    index = defaultdict(set)
    for doc_id, text in documents.items():
        result = preprocess(text, **preprocess_kwargs)
        for term in result["final_tokens"]:
            index[term].add(doc_id)
    return dict(index)


def index_to_df(index: dict[str, set[str]]) -> pd.DataFrame:
    """
    Convert an inverted index to a displayable DataFrame.

    Returns columns: Term | Document Frequency | Posting List
    """
    rows = []
    for term, docs in sorted(index.items()):
        rows.append(
            {
                "Term": term,
                "Doc Frequency": len(docs),
                "Posting List": ", ".join(sorted(docs)),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  9. STEMMING vs LEMMATIZATION COMPARISON
# ─────────────────────────────────────────────


def compare_stem_vs_lemma(
    documents: dict[str, str], queries: list[str]
) -> pd.DataFrame:
    """
    Compare stemming vs lemmatization using TF-IDF cosine similarity.

    For each query:
      - Retrieve docs using stemmed index
      - Retrieve docs using lemmatized index
      - Compute cosine similarity between query vector and each doc vector
      - Report avg similarity for each approach

    Args:
        documents: {doc_id: raw_text}
        queries:   list of query strings

    Returns:
        DataFrame with columns:
        Query | Stem Matches | Lemma Matches | Stem Avg Similarity | Lemma Avg Similarity | Better
    """

    # build both indexs

    stem_index = build_inverted_index(documents, do_stem=True, do_lemma=False)
    lemma_index = build_inverted_index(documents, do_stem=False, do_lemma=True)

    def corpus_texts(do_lemma, do_stem):
        result = []
        for t in documents.values():
            s = preprocess(t, do_stem, do_lemma)
            result.append(" ".join(s["final_tokens"]))
        return result

    stem_text_corpus = corpus_texts(do_lemma=False, do_stem=True)
    lemma_text_corpus = corpus_texts(do_lemma=True, do_stem=False)

    stem_vector = TfidfVectorizer()
    lemma_vector = TfidfVectorizer()

    stem_matrix = stem_vector.fit_transform(stem_text_corpus)
    lemma_matrix = lemma_vector.fit_transform(lemma_text_corpus)

    doc_ids = list(documents.keys())

    rows = []

    for q in queries:
        stem_query = preprocess(q, do_lemma=False, do_stem=True)
        lemma_query = preprocess(q, do_lemma=True, do_stem=False)

        stem_query_text = " ".join(stem_query["final_tokens"])
        lemma_query_text = " ".join(lemma_query["final_tokens"])

        try:
            s_query_vector = stem_vector.fit_transform(stem_query_text)
            l_query_vector = lemma_vector.fit_transform(lemma_query_text)

            s_similarity = cosine_similarity(s_query_vector, stem_matrix)[0]
            l_similaruty = cosine_similarity(l_query_vector, lemma_matrix)[0]
        except Exception:
            s_similarity = np.zeros(len(doc_ids))
            l_similaruty = np.zeros(len(doc_ids))

        # Matched docs (sim > 0)
        s_matches = [doc_ids[i] for i, v in enumerate(s_similarity) if v > 0]
        l_matches = [doc_ids[i] for i, v in enumerate(l_similaruty) if v > 0]

        s_avg = (
            float(np.mean(s_similarity[s_similarity > 0]))
            if any(s_similarity > 0)
            else 0.0
        )

        l_avg = (
            float(np.mean(l_similaruty[l_similaruty > 0]))
            if any(l_similaruty > 0)
            else 0.0
        )

        better = (
            "Stemming"
            if s_avg > l_avg
            else ("Lemmatization" if l_avg > s_avg else "Tie")
        )

        rows.append(
            {
                "Query": q,
                "Stem matches": len(s_matches),
                "Lemma matches": len(l_matches),
                "Stem Avg. Similarity": round(s_avg, 4),
                "Lemma Avg. Similarity": round(l_avg, 4),
                "Better": better,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  10. UTILITY — timing wrapper
# ─────────────────────────────────────────────


def timed_preprocess(text: str, **kwargs) -> tuple[dict, float]:
    """
    Run preprocess() and return (stages_dict, elapsed_ms).
    Useful for benchmarking in the Streamlit UI.
    """

    start = time.perf_counter()
    result = preprocess(text, **kwargs)
    elasped_ms = (time.perf_counter() - start) * 1000
    return result, round(elasped_ms, 3)


# # ─────────────────────────────────────────────────────────────────────────────
# #  QUICK SELF-TEST  (run: python preprocessing.py)
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     sample = {
#         "doc1": "Information retrieval is the activity of obtaining relevant information resources.",
#         "doc2": "A search engine is a well-known software system designed to carry out web searches.",
#         "doc3": "Natural language processing is a subfield of linguistics and artificial intelligence.",
#         "doc4": "Stemming reduces inflected words to their word stem or root form.",
#         "doc5": "Lemmatization groups inflected forms of a word so they can be analysed as a single item.",
#     }

#     print("=" * 60)
#     print("PREPROCESSING SELF-TEST")
#     print("=" * 60)

#     # Single doc pipeline
#     result, ms = timed_preprocess(
#         sample["doc1"],
#         lowercase=True,
#         handle_hyph=True,
#         rm_stopwords=True,
#         do_stem=True,
#     )
#     print(f"\nDoc1 pipeline ({ms} ms)")
#     print(f"  Raw tokens   : {result['raw_tokens'][:8]}")
#     print(f"  Clean tokens : {result['clean_tokens'][:8]}")
#     print(f"  Final tokens : {result['final_tokens'][:8]}")
#     print(f"  Stem map     : {result['stem_map']}")

#     # Inverted index
#     print("\nInverted index (top 10 terms):")
#     index = build_inverted_index(
#         sample, lowercase=True, rm_stopwords=True, do_stem=True
#     )
#     for term, docs in list(index.items())[:10]:
#         print(f"  {term:15s} → {sorted(docs)}")

#     # Stem vs Lemma comparison
#     print("\nStem vs Lemma comparison:")
#     queries = ["information retrieval", "searching documents", "reducing words"]
#     df = compare_stem_vs_lemma(sample, queries)
#     print(df.to_string(index=False))
