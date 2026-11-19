"""Turn retrieved chunks into a grounded, cited answer.

With ANTHROPIC_API_KEY set, an LLM writes the answer from numbered sources and
must cite them as [1], [2]. Citations are then checked: a citation to a source
that wasn't supplied is removed, and an answer with no valid citation is
replaced by the extractive fallback. Without a key, the extractive fallback
returns the best-matching source sentence verbatim, so it can't hallucinate.
If retrieval found nothing relevant, the answer says so instead of guessing.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .text import split_sentences, tokenize

NOT_FOUND = "I couldn't find this in the indexed documents."


def extract_answer_sentence(question: str, contexts: list[str], idf=None) -> str:
    """Pick the context sentence that best matches the question.

    Score = sum of IDF of query terms the sentence contains, lightly
    normalised by sentence length, with a small penalty per rank so ties go to
    the higher-ranked context. Rare terms ("1905", "Tesla") count for far more
    than common ones ("year", "city"). ``idf`` maps a token to its weight;
    without it every term counts 1. Returns "" if no sentence overlaps.
    """
    q = set(tokenize(question))
    weight = idf or (lambda t: 1.0)
    best, best_score = "", 0.0
    for rank, ctx in enumerate(contexts):
        for s, e in split_sentences(ctx):
            sent = ctx[s:e]
            toks = tokenize(sent)
            overlap = q.intersection(toks)
            if not overlap:
                continue
            score = sum(weight(t) for t in overlap) / len(toks) ** 0.15 - 0.05 * rank
            if score > best_score:
                best, best_score = sent, score
    return best


_CITE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop citations outside 1..n_sources. Returns cleaned text and cited ids."""
    cited: list[int] = []

    def keep(m):
        i = int(m.group(1))
        if 1 <= i <= n_sources:
            if i not in cited:
                cited.append(i)
            return m.group(0)
        return ""

    cleaned = _CITE.sub(keep, answer)
    cleaned = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
    return cleaned, cited


@dataclass
class Answer:
    text: str
    citations: list[int] = field(default_factory=list)  # 1-based source numbers
    mode: str = "extractive"


def build_prompt(question: str, sources: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
    return (
        "Answer the question using only the numbered sources. Cite every claim "
        "with its source number in brackets, like [2]. If the sources don't "
        f"contain the answer, reply exactly: {NOT_FOUND}\n\n"
        f"Sources:\n{numbered}\n\nQuestion: {question}"
    )


def _call_claude(prompt: str, key: str, model: str) -> str:
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
        ).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.load(r)
    return "".join(b.get("text", "") for b in body.get("content", []))


def answer(question: str, sources: list[str], llm=None, idf=None) -> Answer:
    """``llm`` is any callable prompt -> text (injected in tests)."""
    if not sources:
        return Answer(NOT_FOUND, [], "none")
    if llm is None and os.environ.get("ANTHROPIC_API_KEY"):
        key = os.environ["ANTHROPIC_API_KEY"]
        model = os.environ.get("CITEWISE_MODEL", "claude-sonnet-5-5")
        llm = lambda p: _call_claude(p, key, model)  # noqa: E731
    if llm is not None:
        raw = llm(build_prompt(question, sources)).strip()
        if raw == NOT_FOUND:
            return Answer(NOT_FOUND, [], "llm")
        text, cited = validate_citations(raw, len(sources))
        if cited:
            return Answer(text, cited, "llm")
        # uncited LLM text is not trusted; fall through to extractive
    sent = extract_answer_sentence(question, sources, idf)
    if not sent:
        return Answer(NOT_FOUND, [], "extractive")
    src = next(i for i, s in enumerate(sources, 1) if sent in s)
    return Answer(f"{sent} [{src}]", [src], "extractive")
