# Results

SQuAD v1.1 dev as 48 long documents; 5,173 held-out questions (odd-numbered articles). Python 3.13.18, numpy 2.5.3.

| pipeline | chunks | recall@1 | recall@5 | MRR@10 | recall @ 300 words | answer sentence found | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Naive: 200-word chunks, TF-IDF | 1,292 | 55.3% | 82.6% | 0.670 | 55.5% | 48.0% | 1.22 |
| B. 200-word chunks + 50 overlap, TF-IDF | 1,701 | 59.4% | 85.2% | 0.705 | 59.5% | 51.6% | 1.40 |
| C. Structured chunks, TF-IDF | 3,179 | 63.8% | 85.3% | 0.732 | 79.5% | 65.8% | 1.46 |
| D. Structured chunks, BM25 | 3,179 | 79.2% | 92.6% | 0.850 | 89.2% | 71.9% | 0.38 |
| E. Structured chunks, dense (LSA) only | 3,179 | 44.4% | 69.8% | 0.554 | 62.3% | 52.6% | 3.31 |
| F. Structured, hybrid BM25 + LSA (RRF, dense weight 0.1) | 3,179 | 73.6% | 90.9% | 0.812 | 86.3% | 70.1% | 3.78 |
| **G. Structured + title headers, BM25 (citewise)** | 3,179 | 79.3% | 92.9% | 0.852 | 89.7% | 72.2% | 0.38 |

**Answer stage, same retrieval (G):** term-count sentence picker 70.4% vs IDF-weighted 72.2% of questions answered by the chosen sentence.
