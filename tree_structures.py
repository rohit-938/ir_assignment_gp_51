"""
tree_structures.py
──────────────────
Dictionary search structures for IR Assignment 1.
Covers:
  - Binary Search Tree (BST)
  - B-Tree (via SortedList — order-guaranteed)
  - Timing benchmarks for both
  - Side-by-side comparison DataFrame
"""

import time
import pandas as pd

from sortedcontainers import SortedList
import random

# ─────────────────────────────────────────────
#  1. Binary search tree (BST)
# ─────────────────────────────────────────────


class BSTNode:
    """Single node in a Binary Search Tree."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: str) -> None:
        self.key: str = key
        self.left: BSTNode | None = None
        self.right: BSTNode | None = None


class BST:
    """
    Binary Search Tree for vocabulary term lookup.

    Supports:
        insert(key)        — insert a term
        search(key)        — search, returns (found: bool, steps: int)
        delete(key)        — remove a term
        inorder()          — sorted list of all terms
        height()           — tree height
        size()             — number of nodes
    """

    def __init__(self) -> None:
        self.root: BSTNode | None = None
        self._size: int = 0

    # ─────────────────────────────────────────────
    #  1. Insert
    # ─────────────────────────────────────────────

    def insert(self, key: str) -> None:
        if self.root is None:
            self.root = BSTNode(key)
            self._size += 1
        else:
            inserted = self._insert(self.root, key)
            if inserted:
                self._size += 1

    def _insert(self, node: BSTNode, key: str) -> bool:
        if key == node.key:
            return False

        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
                return True
            return self._insert(node.left, key)
        else:
            if node.right is None:
                node.right = BSTNode(key)
                return True
            return self._insert(node.right, key)

    # ─────────────────────────────────────────────
    #  2. Search
    # ─────────────────────────────────────────────

    def search(self, key: str) -> tuple[bool, int]:
        """
        Search for a term.

        Returns:
            (found: bool, comparisons: int)
        """
        return self._search(self.root, key, 0)

    def _search(self, node: BSTNode | None, key: str, steps: int) -> tuple[bool, int]:

        if node is None:
            return False, steps
        steps += 1
        if key == node.key:
            return True, steps
        if node.key > key:
            return self._search(node.left, key, steps)
        return self._search(node.right, key, steps)

    def delete(self, key: str) -> None:
        self.root, deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1

    def _delete(self, node: BSTNode | None, key: str) -> tuple[BSTNode | None, bool]:

        if node is None:
            return None, False

        deleted = False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            # Two children — replace with in-order successor
            successor = self._min_node(node.right)
            node.key = successor.key
            node.right, _ = self._delete(node.right, successor.key)
        return node, deleted

    def _min_node(self, node: BSTNode) -> BSTNode:
        while node.left:
            node = node.left
        return node

    # ─────────────────────────────────────────────
    # 3. Traversal
    # ─────────────────────────────────────────────

    def inorder(self) -> list[str]:
        """Return all keys in sorted order."""
        results = []
        self._inorder(self.root, results)
        return results

    def _inorder(self, node: BSTNode | None, results: list[str]) -> None:
        if node:
            self._inorder(node.left, results)
            results.append(node.key)
            self._inorder(node.right, results)

    # ─────────────────────────────────────────────
    #  4. Metrics
    # ─────────────────────────────────────────────

    def height(self) -> int:
        return self._height(self.root)

    def _height(self, node: BSTNode | None) -> int:
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def size(self) -> int:
        return self._size

    def is_balanced(self) -> bool:
        """Check if height ≈ log2(size) — balanced tree."""
        import math

        if self._size == 0:
            return True
        ideal = math.log2(self._size + 1)
        return self.height() <= ideal * 2


# ─────────────────────────────────────────────
#  2. B-TREE  (via SortedList)
# ─────────────────────────────────────────────


class BTree:
    """
    B-Tree backed by sortedcontainers.SortedList.

    SortedList maintains sorted order and provides
    O(log n) search, insert, and delete — equivalent
    to a balanced B-Tree for dictionary lookups.

    Supports:
        insert(key)        — insert a term
        search(key)        — returns (found: bool, index: int)
        delete(key)        — remove a term
        range_query(lo,hi) — all terms between lo and hi
        size()             — number of stored terms
    """

    def __init__(self):
        self._data = SortedList()

    def insert(self, key: str) -> None:
        """Insert a term (duplicates ignored)."""
        if key not in self._data:
            self._data.add(key)

    def search(self, key: str) -> tuple[bool, int]:
        """
        Search for a term using binary search.

        Returns:
            (found: bool, index_position: int)
        """
        idx = self._data.bisect_left(key)
        found = idx < len(self._data) and self._data[idx] == key
        return found, idx

    def delete(self, key: str) -> None:
        """Remove a term if present."""
        try:
            self._data.remove(key)
        except ValueError:
            pass

    def range_query(self, lo: str, hi: str) -> list[str]:
        """
        Return all terms between lo and hi (inclusive).

        Example:
            range_query("inf", "inz") → ["inform", "information", "informative"]
        """
        left = self._data.bisect_left(lo)
        right = self._data.bisect_right(hi)
        return list(self._data[left:right])

    def prefix_search(self, prefix: str) -> list[str]:
        """Return all terms starting with prefix."""
        lo = prefix
        # hi = prefix with last char incremented
        hi = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else ""
        return self.range_query(lo, hi) if prefix else list(self._data)

    def size(self) -> int:
        return len(self._data)

    def all_terms(self) -> list[str]:
        return list(self._data)


# ─────────────────────────────────────────────
#  3. BUILD FROM VOCABULARY
# ─────────────────────────────────────────────


def build_bst_from_vocab(vocabulary: list[str]) -> BST:
    """
    Insert all terms from vocabulary into a BST.
    Terms are shuffled first to reduce worst-case height.
    """
    tree = BST()
    terms = vocabulary[:]
    random.shuffle(terms)  # shuffling prevents O(n) degenerate case
    for term in terms:
        tree.insert(term)
    return tree


def build_btree_from_vocab(vocabulary: list[str]) -> BTree:
    """Insert all terms from vocabulary into a B-Tree."""
    tree = BTree()
    for term in vocabulary:
        tree.insert(term)
    return tree


def extract_vocabulary(
    documents: dict[str, str],
    lowercase: bool = True,
    rm_stopwords: bool = True,
    do_stem: bool = False,
    do_lemma: bool = False,
) -> list[str]:
    """
    Extract a sorted unique vocabulary list from a document collection
    using the same preprocessing pipeline as the inverted index.
    """
    from preprocessing import preprocess

    vocab = set()
    for text in documents.values():
        result = preprocess(
            text,
            lowercase=lowercase,
            rm_stopwords=rm_stopwords,
            do_stem=do_stem,
            do_lemma=do_lemma,
        )
        vocab.update(result["final_tokens"])
    return sorted(vocab)


# ─────────────────────────────────────────────
#  4. BENCHMARK
# ─────────────────────────────────────────────


def benchmark(
    queries: list[str],
    bst: BST,
    btree: BTree,
    n_repeats: int = 100,
) -> pd.DataFrame:
    """
    Benchmark BST vs B-Tree search time for a list of queries.

    Each query is run n_repeats times; average time is reported
    in milliseconds to get stable measurements.

    Args:
        queries:   list of query terms to search
        bst:       built BST
        btree:     built B-Tree
        n_repeats: number of repetitions per query

    Returns:
        DataFrame with columns:
        Query | Found (BST) | BST Time (ms) | BST Steps |
        Found (B-Tree) | B-Tree Time (ms) | Faster
    """
    rows = []

    for query in queries:
        # ── BST timing ──
        bst_times = []
        bst_found, bst_steps = False, 0
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            bst_found, bst_steps = bst.search(query)
            bst_times.append(time.perf_counter() - t0)
        bst_avg_ms = round((sum(bst_times) / n_repeats) * 1000, 6)

        # ── B-Tree timing ──
        bt_times = []
        bt_found = False
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            bt_found, _ = btree.search(query)
            bt_times.append(time.perf_counter() - t0)
        bt_avg_ms = round((sum(bt_times) / n_repeats) * 1000, 6)

        faster = (
            "B-Tree"
            if bt_avg_ms < bst_avg_ms
            else "BST" if bst_avg_ms < bt_avg_ms else "Tie"
        )

        rows.append(
            {
                "Query": query,
                "Found (BST)": "✅" if bst_found else "❌",
                "BST Time (ms)": bst_avg_ms,
                "BST Steps": bst_steps,
                "Found (B-Tree)": "✅" if bt_found else "❌",
                "B-Tree Time (ms)": bt_avg_ms,
                "Faster": faster,
            }
        )

    return pd.DataFrame(rows)


def tree_stats(bst: BST, btree: BTree) -> pd.DataFrame:
    """
    Return a comparison of structural properties of both trees.
    """
    import math

    n = bst.size()
    rows = [
        {"Property": "Vocabulary size", "BST": n, "B-Tree": btree.size()},
        {"Property": "Tree height", "BST": bst.height(), "B-Tree": "O(log n)"},
        {
            "Property": "Ideal log₂(n) height",
            "BST": round(math.log2(n + 1), 2) if n else 0,
            "B-Tree": "—",
        },
        {"Property": "Is balanced", "BST": bst.is_balanced(), "B-Tree": "Always"},
        {
            "Property": "Search complexity",
            "BST": "O(h) worst O(n)",
            "B-Tree": "O(log n) guaranteed",
        },
        {"Property": "Insert complexity", "BST": "O(h)", "B-Tree": "O(log n)"},
    ]
    return pd.DataFrame(rows)


# # ─────────────────────────────────────────────
# #  QUICK SELF-TEST  (run: python tree_structures.py)
# # ─────────────────────────────────────────────

# if __name__ == "__main__":

#     sample = {
#         "doc1": "information retrieval is the activity of obtaining relevant information.",
#         "doc2": "a search engine is a well-known software system designed for web searches.",
#         "doc3": "natural language processing is a subfield of linguistics and intelligence.",
#         "doc4": "stemming reduces inflected words to their word stem or root form.",
#         "doc5": "lemmatization groups inflected forms of a word into a single item.",
#     }

#     print("=" * 60)
#     print("TREE STRUCTURES SELF-TEST")
#     print("=" * 60)

#     # Build vocabulary
#     vocab = extract_vocabulary(sample, rm_stopwords=True, do_stem=True)
#     print(f"\nVocabulary size: {len(vocab)} terms")
#     print(f"Sample terms   : {vocab[:10]}")

#     # Build trees
#     print("\nBuilding BST...")
#     bst = build_bst_from_vocab(vocab)
#     print(f"  Size   : {bst.size()}")
#     print(f"  Height : {bst.height()}")
#     print(f"  Balanced: {bst.is_balanced()}")

#     print("\nBuilding B-Tree...")
#     btree = build_btree_from_vocab(vocab)
#     print(f"  Size : {btree.size()}")

#     # Search
#     test_queries = ["inform", "retriev", "xyz_missing", "search", "stem"]
#     print(f"\nBenchmark ({len(test_queries)} queries, 100 repeats each):")
#     df = benchmark(test_queries, bst, btree)
#     print(df.to_string(index=False))

#     # Stats
#     print("\nTree structural comparison:")
#     print(tree_stats(bst, btree).to_string(index=False))

#     # Range query demo
#     print("\nB-Tree range query 'inf' → 'inz':")
#     print(f"  {btree.range_query('inf', 'inz')}")

#     print("\nB-Tree prefix search 'stem':")
#     print(f"  {btree.prefix_search('stem')}")
