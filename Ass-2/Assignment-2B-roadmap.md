# Assignment 2B: RAG Pipeline roadmap

Plan for building the 2B notebook (20 marks) on the Type 2 Diabetes corpus from Assignment 1. This is a how-to guide, not the solution.

## At a glance

Build one notebook, `Ass-2/Assignment2B.ipynb`, in the same style as `Ass-1/Assignment1A.ipynb`.

| Part | What you build | Marks | Step |
| --- | --- | --- | --- |
| Setup | Notebook, folders, libraries | 0 | 0 |
| Before A | 10 queries with answer spans | 0 (needed for A, B, C) | 1 |
| A | Dense search (FAISS) and chunk size | 5 | 2 |
| B | Dense vs BM25 vs hybrid (RRF) | 4 | 3 |
| C | Cross-encoder reranking | 4 | 4 |
| D1 | Grounded answers with citations | 3 | 5 |
| D2 | LLM judge, spot-check, cause, fix | 4 | 6 |
| Submit | .ipynb with outputs and HTML export | 0 | 7 |

**Reuse from Assignment 1**

- `Ass-1/domain_corpus/*.txt`: the 14 cleaned documents, the document collection.
- `Ass-1/instruction_eval.jsonl` (93 pairs): pick the 10 queries and expected answers here.
- `Ass-1/instruction_source_pairs.json`: each pair's source passage and file name, which makes marking answer spans quick.
- The generator: `mistralai/Mistral-7B-v0.1` with the CPT checkpoint and QLoRA adapter in `persistent/`.

**Where each step runs**

- Steps 0 to 4 (setup, spans, search, BM25, reranking) can run on any machine.
- Part D (generating answers and judging them) runs on the lab server, where the Assignment 1 model files live.

**Brief typos to handle openly**

- The A2 text says 100 and 400 tokens, but the table header says 100 and 500. Run 100 and 400 as the text says, add a 500 row, and write one line explaining why.
- "Total chunks (1M)" looks like a stray label.

## Notebook layout (follow Assignment 1)

Assignment 1 uses one pattern throughout, so copy it:

- A setup cell that finds the storage folder, sets the cache folder, and checks the GPU and dtype (`detect_env()` / `setup()`).
- A cell that creates the working folders and checks that inputs exist.
- A versions cell that prints library and GPU versions.
- For each part: a markdown cell with **Goal** and **What we do**, then code cells, then an Inference cell whose text is built from the real numbers.
- Every code cell starts with a header comment: "What this cell does", "Why", "Output".
- A `print_table()` helper for every table, and `md_safe()` for printing model answers in markdown.

Suggested folders:

```
Ass-2/
  Assignment2B.ipynb
  Assignment2B.html          <- export at the end
  requirements.txt
  data/answer_spans.json     <- 10 queries and spans (Step 1)
  outputs/                   <- tables, rankings, answers, judge results
```

Read the corpus from `../Ass-1/domain_corpus/` rather than copying it.

**Libraries** (add to `Ass-2/requirements.txt`)

| Library | Used for | Part |
| --- | --- | --- |
| `sentence-transformers` | Embeddings and `CrossEncoder` | A, C |
| `faiss-cpu` (or `faiss-gpu`) | `IndexFlatIP` vector index | A |
| `transformers` | Tokenizers; generator and judge models | A, D |
| `rank-bm25` | `BM25Okapi` keyword index | B |
| `peft`, `bitsandbytes`, `accelerate`, `torch` | Load the CPT model and QLoRA adapter in 4-bit | D |
| `pandas` | Result tables and CSV exports | all |
| `nbconvert` | HTML export | submit |

**Test:** run the setup and versions cells on the target machine and confirm the GPU, the 14 corpus files, and the adapter folder are all found.

## Step 1: Pick 10 queries and mark answer spans

Every hit-rate number in Parts A to C depends on these spans. A chunk is a hit if it contains the span.

1. Choose 10 pairs from `instruction_eval.jsonl` that come from different documents and ask for a specific fact (a number, a drug, a threshold).
2. Look up each pair's passage in `instruction_source_pairs.json` (match on `instruction`) to get its `source` file.
3. Copy the shortest exact sentence or phrase that answers the question, usually 10 to 30 words. Short spans are fairer; long spans are more likely to split across a chunk boundary.
4. Assert that each span appears in the cleaned corpus after whitespace normalisation.
5. Save `data/answer_spans.json` with `id`, `query`, `expected_answer`, `source_file`, `span`.
6. Show the 10 rows in a table in the notebook.

```python
def norm(t): return " ".join(t.lower().split())
def is_hit(chunk_text, span): return norm(span) in norm(chunk_text)
```

**Test:** the assert passes for all 10 spans, and each span appears exactly once in the corpus.

## Step 2: Part A, dense search and chunk size (5 marks)

**Embedding model.** `all-MiniLM-L6-v2` truncates at 256 tokens, so 400-token chunks would be cut off. Recommended: switch to `BAAI/bge-base-en-v1.5` (768-dim, 512-token limit). Print `model.max_seq_length` and the vector size to prove it, and write one sentence saying you switched and why.

**A1.1 Reusable dense search (1 mark).** Write `build_dense_index(chunks, model, batch_size=64)`: encode with `normalize_embeddings=True`, then `faiss.IndexFlatIP(dim)` and `index.add(...)`. Normalised vectors make inner product equal cosine similarity. Write `dense_search(query, k)` to encode the query the same way and return `(rank, chunk_id, score)`.

**A1.2 Report the setup (1 mark).** Print model name, vector size, similarity measure, batch size, chunk count, and index build time (timed with `time.perf_counter()`) for the final chunk size.

**A2.1 Chunk at 100 and 400 tokens, no overlap (2 marks).**

- Chunk per document so no chunk spans two files. Keep `doc`, `chunk_id`, `text`, `n_tokens`.
- Tokenise with the embedding model's tokenizer and slice the original text using offsets (`return_offsets_mapping=True`). Decoding token IDs lowercases the text, and the generator needs the original wording.
- For each size, build the index, run the 10 queries at k=3, and compute Hit rate@3 and tokens sent per query (the top 3 chunks' `n_tokens`, averaged).
- Fill the table, including the 500 row.

**A2.2 Choose the size (1 mark).** Two or three sentences weighing hit rate, tokens per query, and the generator's context. The Assignment 1 adapter was trained with `SFT_MAX_LEN = 512`, so three 400-token chunks is far more context than it saw in training. Let the numbers decide and quote them.

**Test:** run one query by hand and read its top 3 chunks; check that `max(n_tokens) <= size`.

## Step 3: Part B, dense vs BM25 vs hybrid (4 marks)

Use the chunk size from A2. All three methods return top 5 and share one `evaluate(search_fn)` helper.

- **B1 Dense (1 mark).** Reuse `dense_search(query, k=5)`.
- **B2 BM25 (1 mark).** `BM25Okapi` over lowercased tokens (`re.findall(r"[a-z0-9]+", text)`). Keep domain terms like `hba1c` and `sglt2` intact.
- **B3 Hybrid with RRF (1 mark).** Take the top 20 from each method, then fuse:

```latex
\mathrm{RRF}(d) = \frac{1}{60 + r_{\text{dense}}(d)} + \frac{1}{60 + r_{\text{BM25}}(d)}
```

Ranks start at 1, and a chunk missing from one list gets nothing from it. Keep the top 5. Write `rrf_fuse(dense_ids, bm25_ids, k=60, top=5)` and unit-test it on a toy example.

**Table columns**

| Column | How |
| --- | --- |
| Avg query latency (ms) | One warm-up call, then time each of the 10 queries and average. Hybrid includes both searches and fusion. |
| Hit rate@1 (%) | Share of queries whose rank-1 chunk contains the span |
| Hit rate@3 (%) | Share with the span in any of the top 3 |

- **B4 Best method (1 mark).** Two or three sentences naming the winner, with one query as evidence: show where the answer chunk ranked under each method.

**Test:** print per-query ranks of the answer chunk for each method. Every hit rate should be a multiple of 10%.

## Step 4: Part C, reranking with a cross-encoder (4 marks)

- **C1 Rerank (1 mark).** Take the best method from B, retrieve the top 10, and score each (query, chunk) pair with `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`. Keep the top 3.
- **C2 Compare (1 mark).** For each query, show the top 3 before and after reranking in a table. Rank change rate is the share of queries whose rank-1 chunk changes.
- **C3 Measure (1 mark).** Fill the table: reranking time per query (ms, timing only `predict`, after a warm-up), rank change rate, hit rate@1 before (from B) and after.
- **C4 Did it help? (1 mark).** Two or three sentences weighing the hit@1 change against the added milliseconds. ms-marco was trained on web search, not clinical text, so mention that if it explains a weak result.

**Test:** the reranked top 3 must be a subset of the original top 10. Keep `final_chunks[qid]` for Part D.

## Step 5: Part D1, grounded answers (3 marks)

The generator is the Assignment 1 model with the CPT checkpoint and QLoRA adapter. Load it with the same 4-bit setup as Assignment 1 B2, and adapt `generate_instruct()` to take chunks.

- **Prompt (1 mark, show it).** One template for all queries:

```text
Answer the question using ONLY the numbered chunks below.
Cite the chunk number you used after each fact, for example [2].
If the chunks do not contain the answer, reply exactly: Not found in the documents

Chunks:
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}

Question: {question}
Answer:
```

Keep the Assignment 1 marker style so the adapter stays on-format, and stop generation at a newline or `Instruction:`.

- **All 10 answers (1 mark).** Use the 3 reranked chunks from Part C. Print query, answer, and cited chunks.
- **"Not in corpus" query (1 mark).** Add one question whose answer is not in the documents. Show the reply is `Not found in the documents`.
- **Fallback.** If the model ignores the chunks or never says "Not found", switch to Qwen2.5 7B Instruct or Llama 3.1 8B Instruct. Name the model and show one example of the failure.

**Test:** every answer with a claim carries at least one `[n]`, where n is 1 to 3.

## Step 6: Part D2, LLM judge (4 marks)

Use a judge model different from the generator, such as `Qwen/Qwen2.5-7B-Instruct`. Name it, and set `do_sample=False` so verdicts repeat.

- **D2.1 Judge prompt and results (1 mark).** One prompt that returns fixed lines per answer:

```text
You are grading an answer about Type 2 Diabetes.

Question: {question}
Expected answer (reference): {expected}
Retrieved chunks:
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}
Model answer: {answer}

CORRECT: one of CORRECT, PARTLY CORRECT, WRONG.
  CORRECT = the model answer gives the same key fact as the reference.
  PARTLY CORRECT = it gives part of the key fact, or the right fact with a missing detail.
  WRONG = it gives a different fact, or says "Not found" when the reference has an answer.
SUPPORTED: YES or NO. YES only if every statement in the model answer appears in the chunks above.
REASON: one sentence. If SUPPORTED is NO, quote the unsupported statement.
```

Show one table: query, correct, supported, reason. Fill the summary counts with `Counter`. Parse with a regex on `^CORRECT:`, `^SUPPORTED:`, `^REASON:`; if a line fails to parse, print the raw output and count it separately.

- **D2.2 Spot-check (1 mark).** Pick 3 answers and read their chunks yourself. Say how many verdicts you agree with, and explain one disagreement.
- **D2.3 Find the cause (1 mark).** Pick one answer that is wrong or unsupported and show its 3 chunks. Decide whether retrieval caused it (the needed fact is in none of the chunks; compare with the Step 1 span) or generation did (the fact is in a chunk but unused, or the model added content).
- **D2.4 Suggest one fix (1 mark).** Match the fix to the cause: a different chunk size if the span was split, hybrid search if an exact term was missed, more chunks if the fact sat at rank 4 or 5, or a stricter prompt if the model invented content. Explain why it would help.

**Test:** the parser reads all 10 verdicts with no fallbacks, and a second run gives the same verdicts.

## Step 7: Submission checklist

Run the notebook top to bottom with outputs saved, then run `jupyter nbconvert --to html Assignment2B.ipynb`.

- [ ] Notebook `.ipynb` with all outputs visible
- [ ] HTML export of the same notebook
- [ ] Part A: 10 spans; A1 model, vector size, similarity, batch size, build time; A2 table (100 and 400, plus 500 with a note); chosen size and why
- [ ] Part B: Dense, BM25, Hybrid table; best method and why, with an example query
- [ ] Part C: top 3 before and after for each query; reranking table; whether it helped
- [ ] Part D1: prompt, all 10 answers, "Not found" query and reply; generator model name, plus the failure example if you switched
- [ ] Part D2: judge model name and prompt; per-query table and summary counts; spot-check of 3 verdicts; one failed answer with its chunks, cause, and fix

## Sources

- Assignment brief: `Ass-2/Assignment 2B — RAG Pipeline.pdf`
- Structure reference: `Ass-1/Assignment1A.ipynb`
- Reused data: `Ass-1/domain_corpus/`, `Ass-1/instruction_eval.jsonl`, `Ass-1/instruction_source_pairs.json`

## Not verified here

- The Ass-1 README points to `Ass-1/SPRINT-PLAN.md` for the team's execution sequence. That file was not in the clone I read.
