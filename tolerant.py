"""
tolerant.py
───────────
Tolerant retrieval for IR Assignment 1.
Covers:
  - K-gram index (k=3)
  - Wildcard query resolution via k-gram intersection
  - Spelling correction via edit distance (rapidfuzz)
  - Phonetic correction (Soundex + Metaphone via jellyfish)
  - Combined tolerant search pipeline
"""

import re
from collections import defaultdict

import pandas as pd
import jellyfish
from rapidfuzz import process, fuzz

from preprocessing import preprocess


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _get_vocabulary(
    documents:    dict[str, str],
    rm_stopwords: bool = True,
    do_stem:      bool = False,
    do_lemma:     bool = False,
) -> list[str]:
    """Extract sorted unique vocabulary from document collection."""
    vocab = set()
    for text in documents.values():
        result = preprocess(
            text,
            lowercase    = True,
            rm_stopwords = rm_stopwords,
            do_stem      = do_stem,
            do_lemma     = do_lemma,
        )
        vocab.update(result["final_tokens"])
    return sorted(vocab)


def _get_tokens(text: str, **kwargs) -> list[str]:
    """Run preprocessing and return final tokens."""
    return preprocess(text, **kwargs)["final_tokens"]


# ─────────────────────────────────────────────
#  1. K-GRAM INDEX
# ─────────────────────────────────────────────

def get_kgrams(term: str, k: int = 3) -> list[str]:
    """
    Extract all k-grams from a term.
    Pads with $ markers: $term$ for boundary awareness.

    Example (k=3):
        get_kgrams("information") →
        ['$in', 'inf', 'nfo', 'for', 'orm', 'rma', 'mat', 'ati', 'tio', 'ion', 'on$']
    """
    padded = f"${term}$"
    return [padded[i:i + k] for i in range(len(padded) - k + 1)]


def build_kgram_index(
    vocabulary: list[str],
    k: int = 3,
) -> dict[str, set[str]]:
    """
    Build a k-gram index from a vocabulary list.

    Returns:
        {kgram: {term1, term2, ...}}

    Example:
        index["inf"] → {"inform", "information", "informative"}
    """
    index = defaultdict(set)
    for term in vocabulary:
        for kg in get_kgrams(term, k):
            index[kg].add(term)
    return dict(index)


# ─────────────────────────────────────────────
#  2. WILDCARD QUERY
# ─────────────────────────────────────────────

def wildcard_to_kgrams(pattern: str, k: int = 3) -> list[str]:
    """
    Convert a wildcard pattern to k-grams for index lookup.

    Handles:
        prefix*   → "inf*"    → k-grams of "$inf"
        *suffix   → "*tion"   → k-grams of "tion$"
        in*fix    → "in*tion" → k-grams of both parts

    Example:
        wildcard_to_kgrams("inf*")   → ['$in', 'inf']
        wildcard_to_kgrams("*tion")  → ['tio', 'ion', 'on$']
    """
    parts  = pattern.split("*")
    kgrams = []
    for i, part in enumerate(parts):
        if not part:
            continue
        # add boundary markers where appropriate
        if i == 0:
            padded = f"${part}"
        elif i == len(parts) - 1:
            padded = f"{part}$"
        else:
            padded = part
        kgrams.extend(get_kgrams(padded, k) if len(padded) >= k else [])
    return list(set(kgrams))


def query_wildcard(
    pattern:     str,
    kgram_index: dict[str, set[str]],
    vocabulary:  list[str],
    k:           int = 3,
) -> dict:
    """
    Resolve a wildcard query using the k-gram index.

    Process:
        1. Extract k-grams from the wildcard pattern
        2. Look up each k-gram in the index → candidate terms
        3. Intersect all candidate sets
        4. Filter candidates using regex to remove false matches

    Args:
        pattern:     wildcard string e.g. "inf*", "*tion", "s*em"
        kgram_index: built k-gram index
        vocabulary:  full vocabulary list (for regex post-filtering)
        k:           gram size

    Returns dict with:
        pattern          — original wildcard pattern
        query_kgrams     — k-grams extracted from pattern
        candidates       — terms after k-gram intersection
        results          — terms after regex verification
        steps            — processing log
    """
    steps = []
    steps.append(f"Pattern      : {pattern}")

    # Step 1 — extract k-grams from pattern
    query_kgrams = wildcard_to_kgrams(pattern, k)
    steps.append(f"Query k-grams: {query_kgrams}")

    if not query_kgrams:
        # no k-grams extractable — return all vocab matching regex
        regex   = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
        results = [t for t in vocabulary if re.match(regex, t)]
        return {
            "pattern":      pattern,
            "query_kgrams": [],
            "candidates":   results,
            "results":      results,
            "steps":        steps + [f"No k-grams — regex fallback: {len(results)} terms"],
        }

    # Step 2 — look up each k-gram and intersect
    sets = []
    for kg in query_kgrams:
        if kg in kgram_index:
            sets.append(kgram_index[kg])
            steps.append(f"  '{kg}' → {len(kgram_index[kg])} terms")
        else:
            sets.append(set())
            steps.append(f"  '{kg}' → not in index")

    if not sets or all(len(s) == 0 for s in sets):
        return {
            "pattern":      pattern,
            "query_kgrams": query_kgrams,
            "candidates":   [],
            "results":      [],
            "steps":        steps + ["No candidates found."],
        }

    candidates = sets[0].copy()
    for s in sets[1:]:
        candidates &= s
    candidates = sorted(candidates)
    steps.append(f"After intersection: {len(candidates)} candidates → {candidates}")

    # Step 3 — regex post-filter to remove k-gram false matches
    regex   = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    results = [t for t in candidates if re.match(regex, t)]
    steps.append(f"After regex filter: {len(results)} results → {results}")

    return {
        "pattern":      pattern,
        "query_kgrams": query_kgrams,
        "candidates":   candidates,
        "results":      results,
        "steps":        steps,
    }


def kgram_index_to_dataframe(
    kgram_index: dict[str, set[str]],
    top_n: int = 50,
) -> pd.DataFrame:
    """Convert k-gram index to displayable DataFrame (top N by coverage)."""
    rows = sorted(
        [{"K-gram": kg, "Terms": len(terms), "Vocabulary": ", ".join(sorted(terms)[:10])}
         for kg, terms in kgram_index.items()],
        key=lambda x: x["Terms"], reverse=True
    )
    return pd.DataFrame(rows[:top_n])


# ─────────────────────────────────────────────
#  3. SPELLING CORRECTION — EDIT DISTANCE
# ─────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    Classic DP implementation.

    Example:
        edit_distance("infomation", "information") → 1
    """
    m, n = len(s1), len(s2)
    dp   = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def spelling_correction(
    query:      str,
    vocabulary: list[str],
    top_n:      int = 5,
    max_dist:   int = 2,
) -> dict:
    """
    Suggest spelling corrections for a query term using
    rapidfuzz (fast Levenshtein) against the vocabulary.

    Args:
        query:      potentially misspelled query term
        vocabulary: list of known correct terms
        top_n:      number of suggestions to return
        max_dist:   maximum edit distance allowed

    Returns dict with:
        query        — original misspelled query
        in_vocab     — True if query already in vocabulary
        suggestions  — list of (term, score, edit_dist) tuples
        best_match   — top suggestion
        steps        — processing log
    """
    steps   = []
    query_l = query.lower().strip()
    steps.append(f"Query        : '{query_l}'")

    # Check if already in vocab
    if query_l in vocabulary:
        steps.append("✔ Query is already in vocabulary — no correction needed.")
        return {
            "query":       query_l,
            "in_vocab":    True,
            "suggestions": [(query_l, 100, 0)],
            "best_match":  query_l,
            "steps":       steps,
        }

    steps.append(f"Not in vocab — searching {len(vocabulary)} terms...")

    # Use rapidfuzz for fast candidate retrieval
    raw_matches = process.extract(
        query_l,
        vocabulary,
        scorer   = fuzz.ratio,
        limit    = top_n * 3,
        score_cutoff = 50,
    )

    # Re-rank by edit distance
    ranked = []
    for term, score, _ in raw_matches:
        dist = edit_distance(query_l, term)
        if dist <= max_dist:
            ranked.append((term, score, dist))

    ranked.sort(key=lambda x: (x[2], -x[1]))   # sort by dist then score
    suggestions = ranked[:top_n]

    best_match = suggestions[0][0] if suggestions else query_l
    steps.append(f"Top suggestions: {[(t, d) for t, _, d in suggestions[:3]]}")
    steps.append(f"Best match     : '{best_match}'")

    return {
        "query":       query_l,
        "in_vocab":    False,
        "suggestions": suggestions,
        "best_match":  best_match,
        "steps":       steps,
    }


def spelling_correction_table(
    suggestions: list[tuple],
) -> pd.DataFrame:
    """Convert spelling suggestions to a displayable DataFrame."""
    return pd.DataFrame(
        [{"Suggested Term": t, "Similarity Score": s, "Edit Distance": d}
         for t, s, d in suggestions]
    )


# ─────────────────────────────────────────────
#  4. PHONETIC CORRECTION
# ─────────────────────────────────────────────

def soundex(term: str) -> str:
    """Return Soundex code for a term using jellyfish."""
    return jellyfish.soundex(term)


def metaphone(term: str) -> str:
    """Return Metaphone code for a term using jellyfish."""
    return jellyfish.metaphone(term)


def build_phonetic_index(
    vocabulary:  list[str],
    method:      str = "soundex",
) -> dict[str, list[str]]:
    """
    Build a phonetic index grouping vocabulary by phonetic code.

    Args:
        vocabulary: list of terms
        method:     'soundex' or 'metaphone'

    Returns:
        {phonetic_code: [term1, term2, ...]}
    """
    fn    = soundex if method == "soundex" else metaphone
    index = defaultdict(list)
    for term in vocabulary:
        try:
            code = fn(term)
            index[code].append(term)
        except Exception:
            pass
    return dict(index)


def phonetic_correction(
    query:          str,
    phonetic_index: dict[str, list[str]],
    method:         str = "soundex",
) -> dict:
    """
    Find phonetically similar terms to a query.

    Args:
        query:          query term (possibly misspelled)
        phonetic_index: built phonetic index
        method:         'soundex' or 'metaphone'

    Returns dict with:
        query          — original query
        code           — phonetic code of query
        matches        — list of phonetically similar terms
        steps          — processing log
    """
    steps   = []
    query_l = query.lower().strip()
    fn      = soundex if method == "soundex" else metaphone

    try:
        code = fn(query_l)
    except Exception:
        code = ""

    steps.append(f"Query         : '{query_l}'")
    steps.append(f"Method        : {method}")
    steps.append(f"Phonetic code : {code}")

    matches = phonetic_index.get(code, [])
    steps.append(f"Matches       : {matches}")

    return {
        "query":   query_l,
        "code":    code,
        "matches": matches,
        "steps":   steps,
    }


def phonetic_index_to_dataframe(
    phonetic_index: dict[str, list[str]],
    top_n: int = 30,
) -> pd.DataFrame:
    """Convert phonetic index to displayable DataFrame."""
    rows = sorted(
        [{"Code": code, "Terms": len(terms), "Vocabulary": ", ".join(sorted(terms))}
         for code, terms in phonetic_index.items() if len(terms) > 1],
        key=lambda x: x["Terms"], reverse=True
    )
    return pd.DataFrame(rows[:top_n])


# ─────────────────────────────────────────────
#  5. COMBINED TOLERANT SEARCH PIPELINE
# ─────────────────────────────────────────────

def tolerant_search(
    query:          str,
    documents:      dict[str, str],
    inv_index:      dict[str, dict[str, int]],
    kgram_index:    dict[str, set[str]],
    phonetic_idx_s: dict[str, list[str]],
    phonetic_idx_m: dict[str, list[str]],
    vocabulary:     list[str],
    k:              int = 3,
    max_dist:       int = 2,
) -> dict:
    """
    Full tolerant retrieval pipeline for a single query term.

    Steps:
        1. Check if query is a wildcard (contains *)
           → use k-gram wildcard resolution
        2. Check if query is in vocabulary
           → exact match, retrieve directly
        3. Run spelling correction
           → suggest corrections + retrieve docs for best match
        4. Run phonetic correction
           → find phonetically similar terms + retrieve their docs

    Returns dict with:
        query            — original query
        is_wildcard      — bool
        wildcard_result  — result from wildcard query (if *)
        exact_match      — bool
        spell_result     — spelling correction result
        phonetic_soundex — phonetic (soundex) result
        phonetic_meta    — phonetic (metaphone) result
        final_terms      — resolved terms used for retrieval
        retrieved_docs   — final set of matched doc IDs
    """
    query_l = query.lower().strip()

    # ── Wildcard path ──
    if "*" in query_l:
        wc_result = query_wildcard(query_l, kgram_index, vocabulary, k)
        matched_terms = wc_result["results"]
        docs = set()
        for term in matched_terms:
            docs |= set(inv_index.get(term, {}).keys())
        return {
            "query":            query_l,
            "is_wildcard":      True,
            "wildcard_result":  wc_result,
            "exact_match":      False,
            "spell_result":     None,
            "phonetic_soundex": None,
            "phonetic_meta":    None,
            "final_terms":      matched_terms,
            "retrieved_docs":   docs,
        }

    # ── Exact match ──
    if query_l in inv_index:
        docs = set(inv_index[query_l].keys())
        return {
            "query":            query_l,
            "is_wildcard":      False,
            "wildcard_result":  None,
            "exact_match":      True,
            "spell_result":     None,
            "phonetic_soundex": None,
            "phonetic_meta":    None,
            "final_terms":      [query_l],
            "retrieved_docs":   docs,
        }

    # ── Spelling correction ──
    spell_result = spelling_correction(query_l, vocabulary, max_dist=max_dist)
    best_spell   = spell_result["best_match"]

    # ── Phonetic correction ──
    ph_soundex = phonetic_correction(query_l, phonetic_idx_s, method="soundex")
    ph_meta    = phonetic_correction(query_l, phonetic_idx_m, method="metaphone")

    # ── Collect all resolved terms ──
    final_terms = set()
    final_terms.add(best_spell)
    final_terms.update(ph_soundex["matches"])
    final_terms.update(ph_meta["matches"])
    final_terms = [t for t in final_terms if t in inv_index]

    # ── Retrieve docs ──
    docs = set()
    for term in final_terms:
        docs |= set(inv_index.get(term, {}).keys())

    return {
        "query":            query_l,
        "is_wildcard":      False,
        "wildcard_result":  None,
        "exact_match":      False,
        "spell_result":     spell_result,
        "phonetic_soundex": ph_soundex,
        "phonetic_meta":    ph_meta,
        "final_terms":      list(final_terms),
        "retrieved_docs":   docs,
    }


# ─────────────────────────────────────────────
#  QUICK SELF-TEST  (run: python tolerant.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from indexing import build_inverted_index

    sample = {
        "doc1": "Information retrieval is the activity of obtaining relevant information resources.",
        "doc2": "A search engine is a well-known software system designed to carry out web searches.",
        "doc3": "Natural language processing is a subfield of linguistics and artificial intelligence.",
        "doc4": "Stemming reduces inflected words to their word stem or root form.",
        "doc5": "Lemmatization groups inflected forms of a word so they can be analysed as a single item.",
    }

    print("=" * 60)
    print("TOLERANT RETRIEVAL SELF-TEST")
    print("=" * 60)

    # Build vocab and indexes
    vocab     = _get_vocabulary(sample, rm_stopwords=True)
    inv_index = build_inverted_index(sample, rm_stopwords=True)
    kg_index  = build_kgram_index(vocab, k=3)
    ph_sdx    = build_phonetic_index(vocab, method="soundex")
    ph_meta   = build_phonetic_index(vocab, method="metaphone")

    print(f"\nVocabulary     : {len(vocab)} terms")
    print(f"K-gram index   : {len(kg_index)} k-grams")

    # Wildcard query
    print("\n── Wildcard: 'inf*'")
    wc = query_wildcard("inf*", kg_index, vocab)
    for s in wc["steps"]:
        print(f"  {s}")
    print(f"  Results: {wc['results']}")

    print("\n── Wildcard: '*ing'")
    wc2 = query_wildcard("*ing", kg_index, vocab)
    print(f"  Results: {wc2['results']}")

    # Spelling correction
    print("\n── Spelling: 'infomation'")
    sp = spelling_correction("infomation", vocab)
    for s in sp["steps"]:
        print(f"  {s}")

    print("\n── Spelling: 'retreival'")
    sp2 = spelling_correction("retreival", vocab)
    print(f"  Best match: '{sp2['best_match']}'")
    print(f"  Suggestions: {[(t,d) for t,_,d in sp2['suggestions'][:3]]}")

    # Phonetic correction
    print("\n── Phonetic (soundex): 'serch'")
    ph = phonetic_correction("serch", ph_sdx, method="soundex")
    for s in ph["steps"]:
        print(f"  {s}")

    # Combined pipeline
    print("\n── Combined tolerant search: 'infomation'")
    result = tolerant_search(
        "infomation", sample, inv_index, kg_index,
        ph_sdx, ph_meta, vocab
    )
    print(f"  Exact match  : {result['exact_match']}")
    print(f"  Best spell   : {result['spell_result']['best_match']}")
    print(f"  Final terms  : {result['final_terms']}")
    print(f"  Retrieved    : {result['retrieved_docs']}")

    print("\n── Combined tolerant search: 'inf*'")
    result2 = tolerant_search(
        "inf*", sample, inv_index, kg_index,
        ph_sdx, ph_meta, vocab
    )
    print(f"  Wildcard     : {result2['is_wildcard']}")
    print(f"  Matched terms: {result2['final_terms']}")
    print(f"  Retrieved    : {result2['retrieved_docs']}")