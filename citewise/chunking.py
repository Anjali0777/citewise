"""Two ways to cut documents into retrievable chunks.

Every chunk keeps its character span in the source document, so an answer can
be cited back to the exact passage and so the benchmark can check whether a
chunk really contains the gold answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import split_paragraphs, split_sentences


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def fixed_chunks(doc_id: str, text: str, size: int = 200, overlap: int = 0) -> list[Chunk]:
    """Naive baseline: windows of ``size`` words, ignoring sentences and paragraphs."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = []
    step = size - overlap
    for i in range(0, len(words), step):
        window = words[i : i + size]
        s, e = window[0][0], window[-1][1]
        out.append(Chunk(doc_id, s, e, text[s:e]))
        if i + size >= len(words):
            break
    return out


def structured_chunks(
    doc_id: str, text: str, target: int = 120, overlap_sentences: int = 1
) -> list[Chunk]:
    """Structure-aware chunks.

    * Never cross a paragraph boundary: a paragraph is usually one topic, and a
      chunk that straddles two topics matches neither query well.
    * Never cut a sentence in half: half a sentence often loses the subject
      ("It was renamed in 1905") or the answer.
    * Paragraphs longer than ``target`` words are split into sentence windows
      that overlap by ``overlap_sentences``, so a fact stated across a
      sentence boundary still lands whole in at least one chunk.
    """
    out: list[Chunk] = []
    for ps, pe in split_paragraphs(text):
        sents = split_sentences(text[ps:pe], base=ps)
        if not sents:
            continue
        lens = [len(text[s:e].split()) for s, e in sents]
        i = 0
        while i < len(sents):
            j, words = i, 0
            while j < len(sents) and (words == 0 or words + lens[j] <= target):
                words += lens[j]
                j += 1
            s, e = sents[i][0], sents[j - 1][1]
            out.append(Chunk(doc_id, s, e, text[s:e]))
            if j >= len(sents):
                break
            i = max(i + 1, j - overlap_sentences)
    return out
