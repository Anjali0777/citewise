"""Tokenizing and sentence splitting, with character offsets kept everywhere."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# A short list on purpose: dropping too much hurts questions like "who was the
# first ...". These are the words that carry no retrieval signal in practice.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were what which who whom whose why how when where with did do does
    this these those there their they he she his her him i we you your our""".split()
)


def _singular(t: str) -> str:
    """Light plural folding so "hotels" matches "hotel". Deliberately minimal:
    aggressive stemmers merge unrelated words ("university" / "universe")."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_singular(t) for t in toks]


# A candidate sentence end is . ! or ? (plus any closing quote or bracket),
# then whitespace, then an uppercase letter, digit or opening quote.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
# ...unless the word before the period is an abbreviation: initialisms such as
# "U.S." or "J.R.R.", single initials ("John F. Kennedy"), or common titles.
_ABBREV_TAIL = re.compile(r"(?:\b(?:[A-Za-z]\.){2,}|\b[A-Z]\.|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Inc|Ltd|Co|Corp|vs|No|Gen|Gov|Sen|Rev|Fig|approx|e\.g|i\.e)\.)$")


def split_sentences(text: str, base: int = 0) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences, offset by ``base``.

    Spans cover the text with no gaps except the whitespace between sentences.
    """
    spans = []
    start = 0
    for m in _SENT_END.finditer(text):
        tail = m.group(0)
        end = m.start() + (len(tail) - len(tail.lstrip("\"')]")))
        if _ABBREV_TAIL.search(text[max(start, m.start() - 12) : m.start()]):
            continue
        if text[start:end].strip():
            spans.append((base + start, base + end))
        start = m.end()
    if text[start:].strip():
        spans.append((base + start, base + len(text.rstrip())))
    return spans


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Spans of paragraphs separated by one or more blank lines."""
    spans = []
    for m in re.finditer(r"[^\n](?:.|\n(?!\s*\n))*", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans
