# Citewise

> Retrieval-augmented question answering over long documents, with answers cited to exact source passages. Benchmarked against naive RAG on 5,173 held-out SQuAD questions: 56% → 90% recall of the answer passage at the same 300-word context budget.

## The problem

Most "chat with your documents" demos cut files into fixed-size pieces, embed them, and send the top few to an LLM. When that answers wrong, the cause is usually retrieval, not the model: the passage with the answer never reached the prompt, or reached it cut in half. An enterprise assistant over policies, runbooks or specs also has to show *where* each answer came from, and say "not found" instead of guessing.

So the questions this project measures are:

1. How often does the passage containing the answer actually make it into the LLM's context?
2. Which pipeline choices move that number, at a fixed context budget?
3. Can every answer be traced to an exact span of a source document?

## What it does

```
 documents ──► structure-aware chunks ──► BM25 index ──► top chunks ──► answer
               (paragraph + sentence       (title header    within a      LLM with numbered sources
                boundaries, exact          per chunk)       word budget   + citation check, or
                char offsets)                                             extractive fallback
```

- **Chunking** (`citewise/chunking.py`): never crosses a paragraph, never cuts a sentence, overlaps long paragraphs by one sentence. Every chunk keeps its character span, so a citation points at the original words.
- **Contextual headers** (`citewise/index.py`): each chunk is indexed as "document title. chunk text", so a paragraph that never names its topic still matches queries about it.
- **Retrieval** (`citewise/retrievers.py`): BM25 implemented on a sparse matrix (no search engine needed). TF-IDF, LSA dense retrieval and reciprocal-rank-fusion hybrid are included as baselines and as the plug-in point for a neural embedding model.
- **Answers** (`citewise/answer.py`): with `ANTHROPIC_API_KEY` set, the LLM gets numbered sources and must cite them as `[n]`. Citations to sources that weren't supplied are stripped, and an answer with no valid citation is rejected in favour of the extractive fallback, which returns the best-matching source sentence verbatim. Weak retrieval returns "I couldn't find this in the indexed documents."
- **REST API** (`citewise/api.py`, FastAPI): `POST /documents`, `POST /ask`, `GET /health`. Docker image included.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                     # 367 tests
curl -L -o data/dev-v1.1.json https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
python -m bench.run                     # full benchmark, about 1 minute; writes results/bench.md

uvicorn citewise.api:app                # or: docker build -t citewise . && docker run -p 8000:8000 citewise
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"travel","title":"Travel Policy","text":"Hotels are reimbursed up to $175 per night.\n\nAlcohol is never reimbursed."}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What is the hotel limit?"}'
```

## Benchmark

SQuAD v1.1 dev has 10,570 questions about 48 Wikipedia articles, each with the exact character span of its answer. Each article becomes **one long document** (about 5,300 words), so a retriever has to find the right passage inside long documents the way it would in a policy library, not just pick a paragraph. A retrieved chunk counts only if it contains the full gold answer span from the right document; the answer appearing somewhere else doesn't count.

Parameters were tuned on questions from the even-numbered articles. The table is the **held-out odd-numbered articles** (5,173 questions), searched against all 48 documents. Full output: [`results/bench.md`](results/bench.md).

| pipeline | recall@1 | recall@5 | recall @ 300 words | answer sentence found | ms/query |
|---|---:|---:|---:|---:|---:|
| A. Naive: 200-word chunks, TF-IDF | 55.3% | 82.6% | 55.5% | 48.0% | 1.22 |
| B. 200-word chunks + 50 overlap, TF-IDF | 59.4% | 85.2% | 59.5% | 51.6% | 1.40 |
| C. Structured chunks, TF-IDF | 63.8% | 85.3% | 79.5% | 65.8% | 1.46 |
| D. Structured chunks, BM25 | 79.2% | 92.6% | 89.2% | 71.9% | 0.38 |
| E. Structured chunks, dense (LSA) only | 44.4% | 69.8% | 62.3% | 52.6% | 3.31 |
| F. Structured, hybrid BM25 + LSA (RRF) | 73.6% | 90.9% | 86.3% | 70.1% | 3.78 |
| **G. Structured + title headers, BM25 (citewise)** | **79.3%** | **92.9%** | **89.7%** | **72.2%** | **0.38** |

- **Recall @ 300 words** is the fair comparison: chunks are taken in rank order until 300 words of context are used, so a pipeline can't win just by using bigger chunks.
- **Answer sentence found** runs the same extractive answer step on every pipeline's context and checks whether the chosen sentence contains the gold answer. It's an end-to-end proxy for what an LLM would have to work with.

## What mattered, and what didn't

- **Chunking was the biggest single win** (A → C: +24 points at 300 words). Fixed windows cut answers in half and fill the budget with neighbouring, unrelated text. Overlap (B) helps a little but costs 32% more chunks.
- **BM25 over TF-IDF was the second** (C → D: +10 points, and nearly 4× faster). Term-frequency saturation and length normalisation matter when chunk lengths vary.
- **Dense retrieval with LSA made things worse**, alone (E) and fused (F). LSA stands in for a neural embedding model so the benchmark runs offline, and it isn't one: it blurs exact names and numbers that SQuAD questions repeat. Hybrid search is worth it only with a strong embedder, so the fusion code stays as the plug-in point and the default is BM25.
- **Title headers gave a small, consistent gain** (D → G: +0.5 points at 300 words). SQuAD questions usually name their subject, so the gain should be larger on enterprise documents whose paragraphs don't repeat the document's topic.
- **The answer step needs IDF weighting.** On the same retrieval, picking the sentence by IDF-weighted overlap finds the answer for 72.2% of questions, against 70.4% for raw term counts.
- **Light plural folding** in the tokenizer ("hotels" matches "hotel") added 0.8 points of recall@1 and matters most for short queries typed by users.
- **Tried and dropped:** a sentence-level reranker (+0.9 recall@1 before title headers, nothing after) and blending document-level BM25 into chunk scores (+0.6 at best). Neither earned its complexity. Details in [DESIGN.md](DESIGN.md).

## How it's tested

- **Chunking properties** over 300 random documents: chunk offsets reproduce the text exactly, no chunk crosses a paragraph, every sentence lands whole in some chunk, fixed windows cover every word.
- **BM25 against a brute-force implementation** of the formula on 50 random corpora (match to 1e-9).
- **Fusion:** RRF is invariant to score scale and favours items both rankers like.
- **Grounding:** citations to unknown sources are stripped, an uncited LLM answer falls back to extractive, an explicit "not found" is respected, and the prompt numbers its sources. The LLM is injected as a function, so these run without an API key.
- **API:** answers cite a source whose `start`/`end` reproduce its text exactly, unrelated questions return "not found", re-adding a document replaces its chunks, bad input returns 422.

## Limitations

- SQuAD questions were written by people looking at the paragraph, so they share a lot of wording with it. That favours lexical retrieval; on paraphrased enterprise queries a neural embedder plus hybrid fusion would likely matter more.
- The LLM answer path is unit-tested with a stub but not benchmarked here: the benchmark has to run offline and for free. Measuring answer faithfulness with a real model is the next step.
- The index lives in memory and rebuilds after each added document. Fine for thousands of documents, not for millions.

## Next steps

- Plug in a neural embedding model behind `HybridRetriever` and rerun the table.
- Benchmark LLM answers for correctness and citation faithfulness.
- Persist the index (SQLite or Postgres with pgvector) and add incremental updates.

Data: SQuAD v1.1 (Rajpurkar et al., 2016), CC BY-SA 4.0, downloaded at benchmark time and not included in the repo. Built with AI pair-programming (Claude). MIT licensed.
