# Design: Citewise

## Problem

A RAG system can only answer from what retrieval hands the model. If the passage with the answer isn't in the context, or arrives cut in half, no prompt fixes it. The goal is to maximise the chance that the answer passage reaches the model **within a fixed context budget**, and to make every answer traceable to an exact span of a source document.

## Evaluation design

- **Corpus:** SQuAD v1.1 dev, each of the 48 articles joined into one long document (about 5,300 words), so retrieval happens inside long documents.
- **Gold:** the character span of each question's answer in its document. A chunk counts as a hit only if it is from the right document and contains the whole span. String matching alone would credit chunks that mention "Denver" for the wrong reason.
- **Budget:** chunks are added in rank order until 300 words are used. Comparing at recall@k would reward bigger chunks; comparing at equal words doesn't.
- **Split:** parameters (chunk size, BM25 k1 and b, fusion weight, reranker settings) were chosen on questions from even-numbered articles. Reported numbers come from odd-numbered articles, while the index always holds all 48 documents.
- **End-to-end proxy:** the same extractive sentence picker runs on every pipeline's context. It measures whether the context is usable, not just whether it technically contains the span.

## Approaches considered

Tuning-split results (1,500 sampled questions) unless noted.

| Change | Effect | Kept? |
|---|---|---|
| Structured chunks (paragraph + sentence boundaries) instead of 200-word windows | +24 points recall at 300 words (held-out) | Yes |
| Chunk target 60 / 90 / 120 / 160 words | Recall at 300 words 84.9 / 84.7 / 84.8 / 84.1%; recall@1 rises with size | 120: best recall@1/budget balance |
| BM25 instead of TF-IDF cosine | +10 points (held-out), nearly 4× faster | Yes |
| BM25 k1/b grid (0.9–1.5, 0.4–0.9) | Within ±0.5 points | Defaults 1.2 / 0.75 |
| LSA dense retrieval, alone | −27 points vs BM25 (held-out) | No |
| Hybrid BM25 + LSA via RRF, dense weight 0.1–1.0 | −2 to −14 points; lower weight hurts less | Code kept as plug-in point, off by default |
| Title header prepended to each chunk's indexed text | +1.5 points at 300 words and +1.6 recall@5 on tuning split; +0.5 at 300 words held-out | Yes |
| Sentence-level rerank of top 20–50 (measured with an earlier tokenizer) | +0.9 recall@1 on plain BM25; no gain once title headers were on | No |
| Blend document-level BM25 into chunk score (earlier tokenizer) | +0.6 recall@1 at best | No |
| IDF-weighted answer-sentence picker instead of term counts | +1.8 points answer sentence found (held-out) | Yes |
| Plural folding in the tokenizer ("hotels" → "hotel") | +0.8 recall@1 for BM25 (held-out); needed for short user queries | Yes |

### Why LSA hurt

SQuAD questions reuse the passage's exact names, numbers and dates. LSA compresses the vocabulary into 256 topics, so "1905" and "1906", or two people from the same field, can land close together. It finds the right *topic* and loses the right *passage*. RRF then lets those near-misses displace correct BM25 results. A neural embedder keeps much more lexical detail and handles paraphrase, which is where hybrid search earns its cost. That couldn't be measured offline, so it's left as the first next step rather than claimed.

### Why chunk structure matters so much

A 200-word window starting mid-paragraph often contains the end of one topic and the start of another. It scores moderately for both and well for neither, and when the answer sits at the boundary it is split across two chunks, neither of which contains the full span. Paragraph-bounded chunks match one topic each, so the top results are denser in relevant text and more of the 300-word budget goes to the answer's neighbourhood.

## Grounding

- Sources are numbered in the prompt; the model must cite `[n]`.
- `validate_citations` removes citations outside `1..n`. If no valid citation remains, the LLM text is discarded and the extractive answer is used, because an uncited claim can't be checked.
- The extractive answer is a verbatim sentence from a source, so it can be wrong but can't invent facts.
- A minimum BM25 score gates retrieval; below it the API returns "not found" instead of answering from weak matches.

## Decision

Structured chunks with title headers, BM25, a 300-word context budget, and cited answers with validation. It's the most accurate pipeline measured, the fastest (0.38 ms per query over 3,179 chunks), and has no model dependency, so it runs anywhere, including inside a locked-down enterprise network.

## Scaling

- **10×** (tens of thousands of documents): rebuild the BM25 matrix incrementally (append rows, recompute IDF lazily) instead of on every add.
- **100×**: move the inverted index to Postgres full-text search or OpenSearch, keep chunking and citation logic unchanged, and add a neural embedder with an ANN index (pgvector/HNSW) behind the existing RRF fusion.
