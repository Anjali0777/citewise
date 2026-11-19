"""The citewise pipeline: structured chunks with contextual headers, BM25, cited answers."""

from __future__ import annotations

from dataclasses import dataclass

from .answer import NOT_FOUND, Answer, answer
from .chunking import Chunk, structured_chunks
from .retrievers import BM25Retriever


@dataclass
class Hit:
    chunk: Chunk
    title: str
    score: float


class Index:
    """In-memory document index.

    Each chunk is indexed as "<document title>. <chunk text>". The header gives
    a chunk the context its own sentences often leave out: a paragraph from
    the travel policy that only says "the limit is $75 per night" still
    matches "hotel limit in the travel policy". The stored chunk text and its
    character span stay unchanged, so citations point at the original words.
    """

    def __init__(self, target_words: int = 120, overlap_sentences: int = 1, min_score: float = 0.5):
        self.target, self.overlap, self.min_score = target_words, overlap_sentences, min_score
        self.titles: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.chunks: list[Chunk] = []
        self._bm25: BM25Retriever | None = None

    def add(self, doc_id: str, title: str, text: str) -> int:
        if doc_id in self.texts:
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self.titles[doc_id], self.texts[doc_id] = title, text
        new = structured_chunks(doc_id, text, self.target, self.overlap)
        self.chunks.extend(new)
        self._bm25 = None  # rebuilt lazily on the next search
        return len(new)

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever([f"{self.titles[c.doc_id]}. {c.text}" for c in self.chunks])
        return self._bm25

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        return [
            Hit(self.chunks[i], self.titles[self.chunks[i].doc_id], s)
            for i, s in self.bm25.search(query, k)
            if s >= self.min_score
        ]

    def idf(self, token: str) -> float:
        b = self.bm25
        return float(b.idf[b.vocab[token]]) if token in b.vocab else 0.0

    def ask(self, question: str, k: int = 4, llm=None) -> tuple[Answer, list[Hit]]:
        hits = self.search(question, k)
        if not hits:
            return Answer(NOT_FOUND, [], "none"), []
        return answer(question, [h.chunk.text for h in hits], llm=llm, idf=self.idf), hits
