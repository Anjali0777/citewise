"""Turn SQuAD v1.1 dev into a retrieval benchmark.

Each of the 48 Wikipedia articles becomes one document (its paragraphs joined
by blank lines), so the retriever has to find the right passage inside long
documents, the way it would over a company's wiki or policy library. Every
question keeps the character span of its gold answer inside that document.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dev-v1.1.json")
URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"


@dataclass
class Question:
    qid: str
    text: str
    doc: int
    spans: list[tuple[int, int]]  # gold answer spans in document coordinates
    answers: list[str]


def load(path: str = DATA):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Download it with:\n  curl -L -o {path} {URL}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    docs, titles, questions = [], [], []
    for di, art in enumerate(data):
        parts, offset = [], 0
        for p in art["paragraphs"]:
            ctx = p["context"].strip()
            lead = len(p["context"]) - len(p["context"].lstrip())
            for qa in p["qas"]:
                spans, answers = [], []
                for a in qa["answers"]:
                    s = offset + a["answer_start"] - lead
                    spans.append((s, s + len(a["text"])))
                    answers.append(a["text"])
                questions.append(Question(qa["id"], qa["question"], di, spans, answers))
            parts.append(ctx)
            offset += len(ctx) + 2
        doc = "\n\n".join(parts)
        docs.append(doc)
        titles.append(art["title"])
    for q in questions:  # sanity: spans point at the answer text
        s, e = q.spans[0]
        assert docs[q.doc][s:e] == q.answers[0], (q.qid, docs[q.doc][s:e], q.answers[0])
    return titles, docs, questions
