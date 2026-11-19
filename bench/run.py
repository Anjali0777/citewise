"""Benchmark retrieval pipelines on SQuAD-as-documents.

    python -m bench.run              # full run, writes results/bench.md
    python -m bench.run --quick      # 1,000 test questions

Questions from even-numbered articles are the tuning split (used to pick
parameters); the tables report the held-out odd-numbered articles. The corpus
always holds all 48 documents, so every query searches everything.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from citewise.answer import extract_answer_sentence
from citewise.chunking import Chunk, fixed_chunks, structured_chunks
from citewise.retrievers import BM25Retriever, HybridRetriever, LsaRetriever, TfidfRetriever

from .squad import Question, load

BUDGET = 300  # words of context handed to the generator


def hit(chunk: Chunk, doc: int, spans) -> bool:
    if int(chunk.doc_id) != doc:
        return False
    return any(chunk.start <= s and e <= chunk.end for s, e in spans)


def within_budget(ranked, chunks, budget=BUDGET):
    out, used = [], 0
    for i, _ in ranked:
        w = chunks[i].n_words
        if out and used + w > budget:
            break
        out.append(i)
        used += w
    return out


@dataclass
class Row:
    name: str
    n_chunks: int
    r1: float
    r5: float
    mrr: float
    rbudget: float
    ans: float
    ms: float


def evaluate(name, chunks, retriever, questions: list[Question], idf=None) -> Row:
    r1 = r5 = rr = rb = ans = 0
    t0 = time.perf_counter()
    for q in questions:
        ranked = retriever.search(q.text, 10)
        hits = [hit(chunks[i], q.doc, q.spans) for i, _ in ranked]
        r1 += hits[0] if hits else 0
        r5 += any(hits[:5])
        rr += next((1 / (j + 1) for j, h in enumerate(hits) if h), 0)
        ctx = within_budget(ranked, chunks)
        rb += any(hit(chunks[i], q.doc, q.spans) for i in ctx)
        sent = extract_answer_sentence(q.text, [chunks[i].text for i in ctx], idf)
        ans += bool(sent) and any(a.lower() in sent.lower() for a in q.answers)
    ms = (time.perf_counter() - t0) * 1000 / len(questions)
    n = len(questions)
    return Row(name, len(chunks), r1 / n, r5 / n, rr / n, rb / n, ans / n, ms)


def build_chunks(docs, kind, **kw):
    fn = fixed_chunks if kind == "fixed" else structured_chunks
    return [c for i, d in enumerate(docs) for c in fn(str(i), d, **kw)]


def pipelines(docs, titles) -> list[tuple[str, list[Chunk], Callable]]:
    fixed = build_chunks(docs, "fixed", size=200)
    fixed_ov = build_chunks(docs, "fixed", size=200, overlap=50)
    struct = build_chunks(docs, "structured", target=120, overlap_sentences=1)
    ft, fot, st = [c.text for c in fixed], [c.text for c in fixed_ov], [c.text for c in struct]
    headed = [f"{titles[int(c.doc_id)].replace('_', ' ')}. {c.text}" for c in struct]
    return [
        ("A. Naive: 200-word chunks, TF-IDF", fixed, lambda: TfidfRetriever(ft)),
        ("B. 200-word chunks + 50 overlap, TF-IDF", fixed_ov, lambda: TfidfRetriever(fot)),
        ("C. Structured chunks, TF-IDF", struct, lambda: TfidfRetriever(st)),
        ("D. Structured chunks, BM25", struct, lambda: BM25Retriever(st)),
        ("E. Structured chunks, dense (LSA) only", struct, lambda: LsaRetriever(st)),
        ("F. Structured, hybrid BM25 + LSA (RRF, dense weight 0.1)", struct, lambda: HybridRetriever(st, dense_weight=0.1)),
        ("**G. Structured + title headers, BM25 (citewise)**", struct, lambda: BM25Retriever(headed)),
    ]


def table(rows: list[Row]) -> str:
    lines = [
        f"| pipeline | chunks | recall@1 | recall@5 | MRR@10 | recall @ {BUDGET} words | answer sentence found | ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.n_chunks:,} | {r.r1:.1%} | {r.r5:.1%} | {r.mrr:.3f} | {r.rbudget:.1%} | {r.ans:.1%} | {r.ms:.2f} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--split", choices=["test", "tune"], default="test")
    args = ap.parse_args()

    titles, docs, questions = load()
    parity = 1 if args.split == "test" else 0
    qs = [q for q in questions if q.doc % 2 == parity]
    if args.quick:
        qs = random.Random(0).sample(qs, 1000)
    print(f"{len(docs)} documents, {len(qs):,} {args.split} questions")

    # One corpus-wide IDF table for the answer extractor, so every pipeline's
    # answer stage is identical and only retrieval differs.
    # IDF is computed over passages (structured chunks), not whole articles:
    # with only 48 articles, most terms appear in many of them and IDF flattens.
    corpus_idf = BM25Retriever([c.text for c in build_chunks(docs, "structured")])
    idf = lambda t: float(corpus_idf.idf[corpus_idf.vocab[t]]) if t in corpus_idf.vocab else 0.0  # noqa: E731

    rows = []
    last = None
    for name, chunks, make in pipelines(docs, titles):
        r = make()
        row = evaluate(name, chunks, r, qs, idf)
        print(table([row]).splitlines()[-1], flush=True)
        rows.append(row)
        last = (chunks, r)

    # Answer-stage ablation on the final retriever
    plain = evaluate("G with term-count sentence picker", last[0], last[1], qs, None)
    md = table(rows)
    md += (
        "\n\n**Answer stage, same retrieval (G):** term-count sentence picker "
        f"{plain.ans:.1%} vs IDF-weighted {rows[-1].ans:.1%} of questions answered by the chosen sentence."
    )
    print("\n" + md)
    if not args.quick and args.split == "test":
        os.makedirs("results", exist_ok=True)
        with open("results/bench.md", "w") as f:
            f.write(
                f"# Results\n\nSQuAD v1.1 dev as 48 long documents; {len(qs):,} held-out questions "
                f"(odd-numbered articles). Python {'.'.join(map(str, __import__('sys').version_info[:3]))}, "
                f"numpy {np.__version__}.\n\n{md}\n"
            )


if __name__ == "__main__":
    main()
