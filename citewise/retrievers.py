"""Retrievers over a list of chunks. All return (chunk_index, score) best-first."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .text import tokenize


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[int, float]]: ...


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]  # score desc, then index asc: deterministic
    return [(int(i), float(scores[i])) for i in idx]


class TfidfRetriever:
    """Naive-RAG baseline: TF-IDF vectors, cosine similarity."""

    def __init__(self, texts: Sequence[str]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.m = self.vec.fit_transform(texts)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.vec.transform([query])
        return _top_k((self.m @ q.T).toarray().ravel(), k)


class BM25Retriever:
    """Okapi BM25, with per-(term, chunk) weights precomputed into a sparse matrix.

    score(q, d) = sum over unique query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))      (never negative)

    A query is then a sparse column sum, which keeps search fast without an
    external search engine.
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for d, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                tid = vocab.setdefault(t, len(vocab))
                counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                rows.append(d)
                cols.append(tid)
                tfs.append(c)
        self.vocab = vocab
        n = len(docs)
        dl = np.array([len(t) for t in docs], dtype=float)
        avgdl = dl.mean() if n and dl.mean() > 0 else 1.0
        rows_a, cols_a, tf = np.array(rows), np.array(cols), np.array(tfs, dtype=float)
        df = np.bincount(cols_a, minlength=len(vocab)) if len(cols_a) else np.zeros(len(vocab))
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * dl[rows_a] / avgdl) if len(rows_a) else np.array([])
        w = self.idf[cols_a] * tf * (k1 + 1) / (tf + norm) if len(rows_a) else np.array([])
        self.w = sparse.csc_matrix((w, (rows_a, cols_a)), shape=(n, len(vocab)))
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.n)
        return np.asarray(self.w[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return _top_k(self.scores(query), k)


class LsaRetriever:
    """Dense retrieval with LSA embeddings (TF-IDF projected by truncated SVD).

    It stands in for a neural embedding model so the benchmark runs offline
    with no model download. Swap in any embedder with the same ``search``
    signature; the hybrid fusion below doesn't care where vectors come from.
    """

    def __init__(self, texts: Sequence[str], dims: int = 256, seed: int = 0):
        self.vec = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, min_df=2
        )
        x = self.vec.fit_transform(texts)
        dims = max(1, min(dims, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.emb = normalize(self.svd.fit_transform(x))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = normalize(self.svd.transform(self.vec.transform([query])))
        return _top_k(self.emb @ q.ravel(), k)


def rrf(rankings: Sequence[Sequence[tuple[int, float]]], k: int = 60, weights=None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank from 1.

    Uses ranks, not raw scores, so BM25 scores (unbounded) and cosine scores
    (in [-1, 1]) can be combined without calibrating one against the other.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, (i, _) in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + w / (k + r)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    """BM25 for exact terms (names, codes, numbers) + dense for paraphrase, fused by RRF."""

    def __init__(self, texts: Sequence[str], depth: int = 100, rrf_k: int = 60, dense_weight: float = 1.0, dims: int = 256):
        self.bm25 = BM25Retriever(texts)
        self.dense = LsaRetriever(texts, dims=dims)
        self.depth, self.rrf_k, self.dense_weight = depth, rrf_k, dense_weight

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        fused = rrf(
            [self.bm25.search(query, self.depth), self.dense.search(query, self.depth)],
            k=self.rrf_k,
            weights=[1.0, self.dense_weight],
        )
        return fused[:k]
