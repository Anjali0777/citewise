import math
import random

import numpy as np
import pytest

from citewise.answer import NOT_FOUND, answer, extract_answer_sentence, validate_citations
from citewise.chunking import fixed_chunks, structured_chunks
from citewise.retrievers import BM25Retriever, HybridRetriever, rrf
from citewise.text import split_sentences, tokenize

DOC = (
    "The travel policy covers all employees. Hotels are reimbursed up to $175 per night in most cities.\n\n"
    "Meals are reimbursed at actual cost. Alcohol is never reimbursed. Receipts are required above $25.\n\n"
    "International trips need director approval. Book flights at least 14 days ahead when possible."
)


def random_doc(rng: random.Random) -> str:
    words = ["alpha", "beta", "gamma", "delta", "Omega", "plant", "wafer", "yield", "1905", "U.S."]
    paras = []
    for _ in range(rng.randint(1, 6)):
        sents = []
        for _ in range(rng.randint(1, 12)):
            n = rng.randint(1, 30)
            sents.append(rng.choice("ABCDEFG") + " ".join(rng.choice(words) for _ in range(n)) + rng.choice(".?!"))
        paras.append(" ".join(sents))
    return "\n\n".join(paras)


# ---------- chunking ----------

@pytest.mark.parametrize("seed", range(200))
def test_structured_chunks_cover_every_sentence_and_keep_offsets(seed):
    rng = random.Random(seed)
    doc = random_doc(rng)
    target = rng.randint(5, 80)
    chunks = structured_chunks("d", doc, target=target, overlap_sentences=rng.randint(0, 2))
    for c in chunks:
        assert doc[c.start : c.end] == c.text  # offsets are exact
        assert "\n\n" not in c.text  # never crosses a paragraph
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    for s, e in split_sentences(doc):
        assert set(range(s, e)) <= covered  # no sentence is lost
        assert any(c.start <= s and e <= c.end for c in chunks)  # and none is cut in half


@pytest.mark.parametrize("seed", range(100))
def test_fixed_chunks_cover_every_word(seed):
    doc = random_doc(random.Random(seed))
    size = random.Random(seed).randint(1, 50)
    chunks = fixed_chunks("d", doc, size=size, overlap=size // 3)
    assert all(doc[c.start : c.end] == c.text for c in chunks)
    joined = set()
    for c in chunks:
        joined.update(range(c.start, c.end))
    assert all(i in joined for i, ch in enumerate(doc) if not ch.isspace())


def test_structured_chunk_does_not_split_sentence():
    chunks = structured_chunks("d", DOC, target=8)
    assert all(c.text.rstrip().endswith((".", "!", "?")) for c in chunks)


def test_sentence_splitter_keeps_abbreviation_together():
    s = "The U.S. Army opened the base in 1905. It closed later."
    spans = split_sentences(s)
    assert [s[a:b] for a, b in spans] == ["The U.S. Army opened the base in 1905.", "It closed later."]


# ---------- BM25 ----------

def brute_bm25(docs, query, k1=1.2, b=0.75):
    toks = [tokenize(d) for d in docs]
    n, avg = len(toks), sum(map(len, toks)) / len(toks)
    out = []
    for t in toks:
        s = 0.0
        for term in set(tokenize(query)):
            df = sum(term in d for d in toks)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        out.append(s)
    return np.array(out)


@pytest.mark.parametrize("seed", range(50))
def test_bm25_matches_brute_force(seed):
    rng = random.Random(seed)
    vocab = "wafer yield fab memory dram nand litho etch clean test ship".split()
    docs = [" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 40))) for _ in range(rng.randint(2, 30))]
    query = " ".join(rng.choice(vocab + ["missing"]) for _ in range(rng.randint(1, 5)))
    np.testing.assert_allclose(BM25Retriever(docs).scores(query), brute_bm25(docs, query), rtol=1e-9, atol=1e-12)


def test_bm25_unknown_query_returns_zero_scores():
    r = BM25Retriever(["a wafer", "a die"])
    assert r.search("zzz", 2) == [(0, 0.0), (1, 0.0)]


# ---------- fusion ----------

def test_rrf_prefers_items_ranked_well_by_both():
    a = [(1, 9.0), (2, 8.0), (3, 7.0)]
    b = [(3, 0.9), (1, 0.8), (4, 0.7)]
    fused = [i for i, _ in rrf([a, b])]
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_rrf_ignores_score_scale():
    a = [(1, 1e9), (2, 1.0)]
    b = [(2, 0.5), (1, 0.4)]
    a_small = [(1, 2.0), (2, 1.0)]
    assert rrf([a, b]) == rrf([a_small, b])


def test_hybrid_runs():
    texts = [DOC.split("\n\n")[i] for i in range(3)] * 3
    assert len(HybridRetriever(texts, dims=2).search("hotel nightly limit", 3)) == 3


# ---------- answers ----------

def test_validate_citations_drops_unknown_sources():
    text, cited = validate_citations("Hotels cap at $175 [1][7]. Meals at cost [2] [0].", 2)
    assert cited == [1, 2]
    assert "[7]" not in text and "[0]" not in text


def test_uncited_llm_answer_falls_back_to_extractive():
    src = ["Hotels are reimbursed up to $175 per night.", "Alcohol is never reimbursed."]
    a = answer("What is the hotel limit per night?", src, llm=lambda p: "It is $175.")
    assert a.mode == "extractive" and a.citations == [1] and "$175" in a.text


def test_llm_answer_with_valid_citation_is_kept():
    a = answer("hotel limit?", ["Hotels up to $175."], llm=lambda p: "$175 per night [1].")
    assert a.mode == "llm" and a.citations == [1]


def test_llm_not_found_is_respected():
    a = answer("parking?", ["Hotels up to $175."], llm=lambda p: NOT_FOUND)
    assert a.text == NOT_FOUND and a.citations == []


def test_prompt_numbers_sources_and_demands_citations():
    seen = {}
    answer("q?", ["first", "second"], llm=lambda p: seen.setdefault("p", p) and "x [2]")
    assert "[1] first" in seen["p"] and "[2] second" in seen["p"] and "Cite" in seen["p"]


def test_extractor_prefers_rare_terms_with_idf():
    ctx = ["The city has many parks. Tesla moved to the city in 1884."]
    idf = {"city": 0.1, "tesla": 5.0, "move": 1.0, "1884": 5.0, "parks": 1.0}.get
    s = extract_answer_sentence("When did Tesla move to the city?", ctx, lambda t: idf(t, 0.0))
    assert s.startswith("Tesla")
