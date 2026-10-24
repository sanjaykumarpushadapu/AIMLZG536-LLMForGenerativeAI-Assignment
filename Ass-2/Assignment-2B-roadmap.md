# Assignment 2B: RAG Pipeline roadmap

A step-by-step plan for the 2B notebook (20 marks) on the Type 2 Diabetes corpus from Assignment 1. Each step lists its inputs, what to do, what the notebook must show, and how to check it. This is a guide, not the solution.

Audited twice against `Ass-2/Assignment 2B — RAG Pipeline.pdf` and the Assignment 1 files in this repo on 2026-10-24.

## At a glance

Build one notebook, `Ass-2/Assignment2B.ipynb`, in the same style as `Ass-1/Assignment1A.ipynb`, and run it top to bottom on the **lab server**.

| Part | What you build | Marks | Step |
| --- | --- | --- | --- |
| Setup | Notebook, folders, libraries, model paths | 0 | 0 |
| Before A | 10 queries with expected answers and answer spans | 0 (A, B, C depend on it) | 1 |
| A1 | One reusable dense search function, and its setup report | 2 | 2 |
| A2 | Chunk size test (100 and 400 tokens) and the choice | 3 | 3 |
| B | Dense vs BM25 vs hybrid (RRF), top 5 | 4 | 4 |
| C | Cross-encoder reranking, top 10 to top 3 | 4 | 5 |
| D1 | Grounded answers with citations and "Not found" | 3 | 6 |
| D2 | LLM judge, spot-check, cause, fix | 4 | 7 |
| Submit | .ipynb with all outputs, plus HTML export | 0 | 8 |

### Where it runs

The brief asks for one notebook with all outputs visible, so the graded run happens in one place: the **lab server**, where the Assignment 1 model files are. You can write and try Steps 0 to 5 on any machine first (they need only the corpus and small models), but do the final top-to-bottom run on the lab server.

The Assignment 1 generator is not in git (`cpt_ckpt/` and `adapter/` are in `Ass-1/.gitignore`). On the lab server they sit next to the Assignment 1 notebook:

- CPT model: `Ass-1/persistent/cpt_ckpt` (from `../Ass-1/persistent/cpt_ckpt` in the 2B notebook)
- QLoRA adapter: `Ass-1/persistent/adapter`
- Base model id: `mistralai/Mistral-7B-v0.1`

These are the Assignment 1 paths (`PERSIST = BASE / "persistent"` in Assignment 1 cell 8, `CPT_CKPT` in cell 29, `ADAPTER_DIR` in cell 49). The 2B paths cell sets `CPT_DIR` and `ADAPTER_DIR` to the same folders. On the lab server the CPT checkpoint and adapter sit in the same folder as in Assignment 1, confirmed by the project owner, so the paths need no change.

**Hugging Face access.** `meta-llama/Llama-3.1-8B-Instruct` is a gated model: it needs an `HF_TOKEN` from an account that has accepted Meta's licence. `Qwen/Qwen2.5-7B-Instruct`, `BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` are not gated. The Assignment 1 setup cell already reads `HF_TOKEN` from the environment.

### Fixes to the brief to handle openly

- The A2 text says 100 and 400 tokens, but the table header says 100 and 500. Run 100 and 400 as the text says, add a 500 row, and write one line explaining why.
- "Total chunks (1M)" in the A2 table is a stray label. Use the header "Total chunks".
- The B question about the best method is numbered "4" under three numbered steps. Treat it as B4.

## Model choices and where each comes from

Checked against every assignment file in the clone: the 2B brief, `Ass-1/briefs/Assignment 1A - CPT and SFT.pdf`, `Ass-1/briefs/Enterprise_Variants_All_Assignments.pdf`, the README, `Ass-1/requirements.txt` and the Assignment 1 notebook. No file **requires** a specific embedding model, reranker or judge. The 2B brief gives examples, and the variants guide says "Students are encouraged to experiment with different models other than recommended" (How to use this guide, point 5). Variant 1 adds no model guidance of its own ("refer directly to each assignment document").

| Role | Model | Source |
| --- | --- | --- |
| Generator (fixed) | `mistralai/Mistral-7B-v0.1` + CPT + QLoRA adapter (Assignment 1) | **Brief:** 2B Overview, "your fine-tuned model as the answer generator", and D1.2 "give your Assignment 1 model the top 3 chunks". Mistral-7B-v0.1 is `MODEL_ID` in `Assignment1A.ipynb` cell 18. **Decision:** the generator stays Mistral. Assignment 1 is already submitted, so there is no switch to Qwen or Llama for generation. |
| Embedding model | `BAAI/bge-base-en-v1.5` | **Our choice, allowed by the brief.** The brief's example is `all-MiniLM-L6-v2` (Part A, "Before you start"), but its 256-token limit cannot hold a 400-token chunk. A2 says to "switch to a model with a longer limit or replace that size", and say which. Switching keeps the 100 and 400 rows the brief asks for. |
| Vector index | FAISS `IndexFlatIP` | **Brief:** A1.1, "builds a FAISS IndexFlatIP index (or another vector index)". |
| Keyword search | BM25 (`rank-bm25`, `BM25Okapi`) | **Brief:** B2, "build a BM25 index". The library is our choice. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **Brief:** C1, "for example cross-encoder/ms-marco-MiniLM-L-6-v2". |
| Judge | `Qwen/Qwen2.5-7B-Instruct` (option: `meta-llama/Llama-3.1-8B-Instruct`, gated) | **Brief:** D2, "Use a judge model different from your generator, for example Llama 3.1 8B Instruct or Qwen2.5 7B Instruct". Qwen differs from the Mistral generator, so it is kept. |

**Stale comment to know about.** `Ass-1/requirements.txt` line 3 says `Model: stanford-crfm/BioMedLM`, but the notebook's `MODEL_ID` and title both say `mistralai/Mistral-7B-v0.1`. The notebook is what produced the adapter, so the generator is Mistral.

## Important: which file is the Assignment 1 evaluation set

**Do not use `Ass-1/instruction_eval.jsonl` from the repo.** It holds 93 short pairs, and none of its questions appear in the dataset the Assignment 1 model was trained and evaluated on. It does not match the notebook's own output.

The real Assignment 1 eval set is the **last 80 rows of `Ass-1/instruction_dataset.jsonl`**. The B1 cell of `Assignment1A.ipynb` writes train pairs first and eval pairs after, and its output says "eval split (20% of passages): 80 pairs from 10 passages". The three held-out questions printed in the Assignment 1 notebook ("When should glycaemic control, co-morbidities…", "Why must diabetes education be tailored…", "What role does physical activity play…") are the first three of those 80 rows.

These 80 pairs come from 10 passages that share no passage with the training data. Each passage appears word for word, exactly once, in the cleaned corpus:

| Passage id | Source file |
| --- | --- |
| 2 | hse-ireland-nutrition-dietetic-type2-diabetes-2020.txt |
| 6 | icmr-type2-diabetes-guidelines-2018.txt |
| 7 | icmr-type2-diabetes-guidelines-2018.txt |
| 8 | icmr-type2-diabetes-guidelines-2018.txt |
| 9 | idf-atlas-global-diabetes-prevalence-2021.txt |
| 15 | idf-clinical-practice-recommendations-2025.txt |
| 16 | moh-malaysia-type2-diabetes-cpg-2020.txt |
| 18 | moh-malaysia-type2-diabetes-cpg-2020.txt |
| 35 | sign116-management-of-diabetes-qrg.txt |
| 41 | sign154-pharmacological-glycaemic-control.txt |

Taking one query per passage gives exactly 10 queries, and you already know which passage holds each answer.

## Step 0: Set up the notebook the Assignment 1 way (no marks, everything depends on it)

**Inputs:** `Ass-1/Assignment1A.ipynb` (to copy cells from), `Ass-1/domain_corpus/*.txt`, `Ass-1/instruction_dataset.jsonl`, `Ass-1/instruction_source_pairs.json`, the model folders above.

**Do:**

1. Create `Ass-2/Assignment2B.ipynb` with a title cell: assignment name, domain (Type 2 Diabetes), variant 1, generator model.
2. Copy cell 1 of Assignment 1 (`detect_env()` and `setup()`): it finds the storage folder, sets `HF_HOME` before `transformers` is imported, and picks the GPU and dtype.
3. Add a folder cell that creates `data/` and `outputs/` and checks the inputs: 14 corpus files, `instruction_dataset.jsonl` with 400 rows, `instruction_source_pairs.json` with 50 passages, and both model folders. Print OK or MISSING for each, like Assignment 1 cell 3.
4. Add an install cell and a versions cell (Python, packages, torch, GPU), like Assignment 1 cells 5 and 6.
5. Copy the helpers `print_table()` (cell 9), `md_safe()` and `working` (cell 18).
6. Fix the random seed (`SEED = 42`) for `random`, `numpy` and `torch`.

Use the Assignment 1 conventions in every part:

- A markdown cell per part with **Goal**, **What we do**, and the marks.
- A small heading for every numbered step in the brief, with its marks, for example "A1.1 Dense search function (1 mark)". The brief grades each numbered step on its own, so the marker should find each one at a glance.
- A header comment on every code cell: "What this cell does", "Why", "Output".
- An Inference cell at the end of each part that builds its text from the computed numbers, so the words always match the results.
- Save every table to `outputs/` as CSV or JSON.

**Libraries** (list them in `Ass-2/requirements.txt`):

| Library | Used for | Part |
| --- | --- | --- |
| `sentence-transformers` | Embedding model and `CrossEncoder` | A, B, C |
| `faiss-cpu` | `IndexFlatIP` vector index (a few thousand vectors, so CPU is fast enough and installs with pip) | A, B |
| `transformers` | Tokenizers, generator and judge models | A, D |
| `rank-bm25` | `BM25Okapi` keyword index | B |
| `peft`, `bitsandbytes`, `accelerate`, `torch` | Load the CPT model and adapter in 4-bit | D |
| `pandas`, `numpy` | Tables and scores | all |
| `nbconvert` | HTML export | submit |

**Show:** the environment line, the input checks, the versions table.

**Check:** every input check prints OK on the lab server before you write any RAG code.

## Step 1: Pick 10 queries and mark answer spans (needed for A, B, C)

**Inputs:** the last 80 rows of `instruction_dataset.jsonl`, `instruction_source_pairs.json`, the corpus.

**Do:**

1. Load the 80 eval rows and map each question to its passage through `instruction_source_pairs.json` (match on `instruction`; every eval question is there).
2. Pick **one question per eval passage**, so the 10 queries cover all 10 passages. Prefer questions with one clear fact (a number, a drug, a threshold, a named step).
3. Keep the pair's `response` as the **expected answer**. Part D2 judges against it.
4. Mark the **answer span**: the shortest exact sentence or phrase in the passage that answers the question, usually 10 to 30 words. Copy it from the passage, not from the response, because the responses were written in the model's own words.
5. Check each span appears in the cleaned corpus file after whitespace normalisation, and appears only once in the corpus.
6. Save `data/answer_spans.json` with `qid`, `query`, `expected_answer`, `source_file`, `passage_id`, `span`.

```python
def norm(t): return " ".join(t.lower().split())
def is_hit(chunk_text, span): return norm(span) in norm(chunk_text)
```

Use `is_hit` for every hit rate in Parts A, B and C. A span split across two chunks counts as a miss, which is a real effect of chunk size. Shorter spans make this less likely.

**Show:** a table of the 10 queries with source file, span and expected answer. This is a graded checklist item ("Answer span marked for each of the 10 queries").

**Check:** all 10 asserts pass, 10 different passages are used, and each span occurs exactly once in the corpus.

## Step 2: Part A1, dense search (2 marks)

**Inputs:** the corpus, the embedding model.

**Choose the embedding model first.** `all-MiniLM-L6-v2` cuts input at 256 tokens, so 400-token chunks would be truncated. The brief asks you to switch models or change the size, and say which. Recommended: switch to `BAAI/bge-base-en-v1.5` (768 dimensions, 512-token limit), so 100, 400 and 500 all fit with room for the 2 special tokens. Write one sentence saying you switched and why, and print `model.max_seq_length`.

**A1.1 One reusable function (1 mark).** The brief asks for **one** function that embeds the chunks, builds the index, and returns the top k for a query. For example:

```python
def dense_search(chunks, queries, k, model, batch_size=64, cache={}):
    """Embed the chunks, build a FAISS IndexFlatIP, return the top k chunks for each query."""
    # 1. If this chunk set is not cached: encode with normalize_embeddings=True,
    #    build faiss.IndexFlatIP(dim), index.add(...), record the build time.
    # 2. Encode the queries the same way, index.search(q, k).
    # 3. Return, per query, a list of (rank, chunk_id, score, text), ranks starting at 1.
```

The cache keeps Parts A, B and C from rebuilding the same index. Normalised vectors make the inner product equal cosine similarity. bge v1.5 accepts a query prefix (`"Represent this sentence for searching relevant passages: "`); use it or not, but say which, and keep it the same everywhere.

**A1.2 Report the setup (1 mark).** The brief asks for this "for your final chunk set", so print it **after** A2 picks the size: model name, vector size, similarity measure, batch size, number of chunks, and index build time (encoding plus `index.add`, timed with `time.perf_counter()`).

**Show:** the function, and the A1.2 table after A2.

**Check:** a query copied word for word from a chunk returns that chunk at rank 1 with a score near 1.0.

## Step 3: Part A2, chunk size (3 marks)

**Inputs:** the corpus, `dense_search` from A1, the 10 queries and spans.

**A2.1 Chunk and test (2 marks).**

1. Chunk each document separately, so no chunk spans two files.
2. Use fixed-size chunks with no overlap, at 100 and 400 tokens (plus 500, see the typo note).
3. Count tokens with the **embedding model's** tokenizer: `tok(text, add_special_tokens=False, return_offsets_mapping=True)`. Slice the original text with the offsets, because decoding token ids lowercases it and the generator needs the original wording.
4. Keep `doc`, `chunk_id`, `text`, `n_tokens` for each chunk.
5. For each size, run `dense_search` on the 10 queries with k=3.
6. Hit rate@3 = share of queries with the span in any of the top 3. Tokens sent per query = the sum of `n_tokens` of the top 3, averaged over the 10 queries.

**Table to fill:**

| Chunk size (tokens) | Total chunks | Hit rate@3 (%) | Tokens sent per query (top 3) |
| --- | --- | --- | --- |
| 100 | | | |
| 400 | | | |
| 500 (header typo) | | | |

**A2.2 Choose the size (1 mark).** Two or three sentences weighing hit rate, tokens per query, and how much context the Assignment 1 model can take. Facts to use, from the Assignment 1 notebook: the adapter was trained with `SFT_MAX_LEN = 512` on prompts with no context (cell 49), and the model's context window is 4,096 tokens (printed by cell 18). Three 400-token chunks plus the prompt come to roughly 1,300 tokens, which fits the window but is well beyond the 512 tokens the adapter saw in training. Measure the full Part D prompt length with the **Mistral** tokenizer for your chosen size, since it counts differently from the embedding tokenizer. Let the numbers decide, and quote them.

**Show:** the table, the chosen size with the reason, and then the A1.2 report.

**Check:** the chunk count for 100 is about 4 times the count for 400, and `max(n_tokens) <= size` for every size.

## Step 4: Part B, dense vs BM25 vs hybrid (4 marks)

**Inputs:** the chunks at the chosen size, the 10 queries and spans.

All three methods return the top 5 and share one `evaluate(search_fn)` helper, so the rows are comparable.

1. **B1 Dense (1 mark).** Call `dense_search(..., k=5)`.
2. **B2 BM25 (1 mark).** Build `BM25Okapi` once over the chunks. Tokenise chunks and queries the same way: lowercase, `re.findall(r"[a-z0-9]+", text)`. This keeps domain terms like `hba1c` and `sglt2` whole. Per query: `bm25.get_scores(tokens)`, then `np.argsort(-scores)[:k]`.
3. **B3 Hybrid with RRF (1 mark).** Take the top **20** from dense and from BM25, then score each chunk:

```latex
\mathrm{RRF}(d) = \frac{1}{60 + r_{\text{dense}}(d)} + \frac{1}{60 + r_{\text{BM25}}(d)}
```

Ranks start at 1, and a chunk missing from one list gets nothing from that list. Sort by score and keep the top 5. Write it as `rrf_fuse(dense_ids, bm25_ids, k=60, top=5)` and test it on a toy case: a chunk ranked 2nd and 3rd should beat a chunk ranked 1st in only one list.

**Table to fill:**

| Method | Average query latency (ms) | Hit rate@1 (%) | Hit rate@3 (%) |
| --- | --- | --- | --- |
| Dense | | | |
| BM25 | | | |
| Hybrid (RRF) | | | |

How to compute the columns:

- **Latency:** the time to return the top 5 for one query, averaged over the 10 queries. Run one warm-up query first and leave it out. Dense includes encoding the query. Hybrid includes both top-20 searches and the fusion.
- **Hit rate@1:** share of queries whose rank-1 chunk contains the span.
- **Hit rate@3:** share with the span in any of the top 3.

4. **B4 Best method (1 mark).** Two or three sentences naming the method that works best for your queries and why, using **one query** as the example. Show where the answer chunk ranked under each of the three methods for that query.

**Show:** the table, a per-query table of the answer chunk's rank under each method, and the B4 text.

**Check:** every hit rate is a multiple of 10%, and the hybrid top 5 only contains chunks from the two top-20 lists.

## Step 5: Part C, reranking (4 marks)

**Inputs:** the best method from B4, the chunks, the 10 queries and spans.

1. **C1 Rerank (1 mark).** Retrieve the top **10** with the best method. Score each (query, chunk text) pair with `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)` using `ce.predict(pairs)`. Sort by score and keep the top 3. Say whether it ran on GPU or CPU.
2. **C2 Compare rankings (1 mark).** For each query, show the top 3 before reranking (the best method's own order) and after. Show the chunk id, the first 80 characters, and a hit marker. **Rank change rate** = share of the 10 queries whose first result changes.
3. **C3 Measure (1 mark).** Time only the `predict` call, after one warm-up, and average over the 10 queries. On GPU, call `torch.cuda.synchronize()` before stopping the timer, so the time is not cut short.

| Measure | Value |
| --- | --- |
| Reranking time per query (ms) | |
| Rank change rate (%) | |
| Hit rate@1 before reranking (from Part B, same method) | |
| Hit rate@1 after reranking | |

4. **C4 Did it help? (1 mark).** Two or three sentences that weigh the hit rate@1 before and after against the time added. If the gain is small, note that ms-marco was trained on web search, not clinical text.

Save the reranked top 3 per query as `final_chunks[qid]`. Part D uses exactly these.

**Show:** the before and after table for all 10 queries, the measure table, and the C4 text.

**Check:** the after list is always a subset of the top 10, and "hit rate@1 before" equals the value in the Part B table for the same method.

## Step 6: Part D1, grounded answers (3 marks)

**Inputs:** `final_chunks`, the 10 queries, the Assignment 1 generator.

**Load the generator.** Load the CPT model from `../Ass-1/persistent/cpt_ckpt` in 4-bit with the same `BitsAndBytesConfig` as Assignment 1 B2, add the adapter with `PeftModel.from_pretrained(model, "../Ass-1/persistent/adapter")`, and call `model.eval()`. Load the tokenizer from the adapter folder (`AutoTokenizer.from_pretrained("../Ass-1/persistent/adapter")`): Assignment 1 saved it there after setting the chat template (cells 49 and 50), so the `Instruction: … Response:` format comes with it. Use greedy decoding (`do_sample=False`) and about 150 new tokens.

1. **Prompt (1 mark).** Show the prompt in the notebook. One template for all queries:

```text
Answer the question using ONLY the numbered chunks below.
Cite the chunk number you used after each fact, for example [2].
If the chunks do not contain the answer, reply exactly: Not found in the documents

Chunks:
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}

Question: {question}
```

Put this whole text in the user turn, so the model sees `Instruction: <prompt> Response:` as in Assignment 1. Cut the output at the next `Instruction:`, as `generate_instruct()` does.

2. **All 10 answers (1 mark).** For each query, pass the 3 reranked chunks, labelled [1] to [3]. Show a table: query, answer, chunks cited.
3. **The not-in-corpus query (1 mark).** Add one question whose answer is not in the 14 documents (for example, about a sport, or a drug the corpus never mentions). Run it through the same retrieval, reranking and prompt. Show its 3 chunks and the reply `Not found in the documents`.

**If the Assignment 1 model does not follow the rules.** The generator is kept as Mistral in every case; there is no switch. If it ignores the chunks, never cites, or never says "Not found", report the D1 check honestly:

- Count the failures (answers with no citation, and the out-of-corpus query if it does not return "Not found").
- Show one or two failure examples: the Mistral answer next to the chunks it ignored.
- Say in the D1 text what the failures mean for the answers, and do not describe the answers as grounded where they are not.

**Show:** the prompt, the generator name (Mistral), the 10 answers, the not-in-corpus query with its reply, the citation count, and the failure examples.

**Check:** each answer either cites at least one of [1], [2] or [3], or is exactly "Not found in the documents". Where an answer does not meet this, the failure is reported, not hidden.

## Step 7: Part D2, LLM judge (4 marks)

**Inputs:** the 10 answers, their 3 chunks, the expected answers from Step 1.

**Choose the judge.** It must differ from the generator. The generator is Mistral, so use `Qwen/Qwen2.5-7B-Instruct`. `meta-llama/Llama-3.1-8B-Instruct` is an option only (gated: needs `HF_TOKEN`, see "Where it runs"). Free the generator first (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`), load the judge in 4-bit, and use its own chat template. Use greedy decoding (`do_sample=False`), which is temperature 0. Name the judge in the notebook.

1. **D2.1 Judge prompt and results (1 mark).** One prompt that checks both things and follows every hint in the brief:

```text
You are grading one answer about Type 2 Diabetes.

Question: {question}
Expected answer (reference): {expected}
Retrieved chunks:
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}
Model answer: {answer}

Check 1, CORRECT: compare the model answer with the expected answer.
  CORRECT = it states the same key fact as the expected answer.
  PARTLY CORRECT = it states only part of the key fact, or the right fact with a detail missing or wrong.
  WRONG = it states a different fact, contradicts the expected answer, or says "Not found in the documents".
Check 2, SUPPORTED: can every statement in the model answer be found in the 3 chunks above?
  YES = every statement is in the chunks. NO = at least one statement is not.
  A "Not found in the documents" reply makes no claims, so it is SUPPORTED: YES.
REASON: one sentence. If SUPPORTED is NO, quote the unsupported statement.

Reply in exactly this format, with nothing else:
CORRECT: <CORRECT | PARTLY CORRECT | WRONG>
SUPPORTED: <YES | NO>
REASON: <one sentence>
```

Run it on the 10 answers (not on the not-in-corpus query). Parse the three lines with regular expressions on `^CORRECT:`, `^SUPPORTED:` and `^REASON:`. Show one table with a row per query: query, correct, supported, reason. Then fill the summary:

| Check | Count out of 10 |
| --- | --- |
| Correct | |
| Partly correct | |
| Wrong | |
| Supported by the chunks | |

2. **D2.2 Spot-check (1 mark).** Pick 3 answers, ideally with different verdicts. Read each answer and its 3 chunks yourself, and write your own CORRECT and SUPPORTED. Say how many of the judge's verdicts you agree with (out of 3), and explain one disagreement if there is one.
3. **D2.3 Find the cause (1 mark).** Pick one answer the judge marked WRONG or SUPPORTED: NO. Show its 3 chunks and say which caused it:
    - **Retrieval:** the needed fact is not in any of the 3 chunks. Check with `is_hit` against the Step 1 span.
    - **Generation:** the fact is in a chunk, but the model did not use it, or added something the chunks do not say.

    If no answer is wrong or unsupported, pick the weakest one (partly correct), and say so.
4. **D2.4 Suggest a fix (1 mark).** Propose one change that matches the cause, and explain why it would help:
    - The span was split across chunks: a different chunk size, or overlap.
    - An exact term was missed: hybrid or BM25 search.
    - The fact ranked 4th or 5th: rerank more candidates, or pass more chunks.
    - The model ignored or added content: stricter prompt wording, or an instruct model.

**Show:** the judge name, the judge prompt, the per-query table, the summary counts, the spot-check, the failed answer with its 3 chunks, the cause and the fix.

**Check:** all 10 verdicts parse with no fallback. Running the judge twice gives the same verdicts.

## Step 8: Export and submit

1. Restart the kernel and run all cells on the lab server.
2. Make sure every output is visible: no errors, no empty cells, no cleared outputs.
3. Export: `jupyter nbconvert --to html Ass-2/Assignment2B.ipynb`.
4. Open the HTML and check that every table and model answer shows.

### Submission checklist (from the brief)

**Files**

- [ ] Notebook (.ipynb) with all outputs visible (Step 8)
- [ ] HTML export of the same notebook (Step 8)

**Part A**

- [ ] Answer span marked for each of the 10 queries (Step 1)
- [ ] A1: embedding model name, vector size, similarity measure, batch size, index build time (Steps 2 and 3)
- [ ] A2: chunk size table (100 and 400 tokens) and the chosen size with its reason (Step 3)

**Part B**

- [ ] Search comparison table: Dense, BM25, Hybrid (Step 4)
- [ ] Best method and why, with an example query (Step 4)

**Part C**

- [ ] Top 3 before and after reranking for each query (Step 5)
- [ ] Reranking table: rank change rate, time per query, hit rate@1 before and after (Step 5)
- [ ] Whether reranking helped, weighing accuracy against time (Step 5)

**Part D**

- [ ] D1: generation prompt, all 10 answers, the "Not found" query and its answer (Step 6)
- [ ] D1: generator model used (Mistral, the Assignment 1 model), with the D1 check results and any failure examples reported honestly (Step 6)
- [ ] D2: judge model name and judge prompt (Step 7)
- [ ] D2: judge results table per query, and the summary counts (Step 7)
- [ ] D2: spot-check of 3 verdicts, with one disagreement explained if any (Step 7)
- [ ] D2: one failed answer with its 3 chunks, its cause and a suggested fix (Step 7)

## Sources

- Assignment brief: `Ass-2/Assignment 2B — RAG Pipeline.pdf`
- Structure and code to reuse: `Ass-1/Assignment1A.ipynb` (cells 1, 3, 5, 6, 9, 18, 49, 50, 51)
- Data: `Ass-1/domain_corpus/`, `Ass-1/instruction_dataset.jsonl` (last 80 rows = eval set), `Ass-1/instruction_source_pairs.json`
- Model files on the lab server: `Ass-1/persistent/cpt_ckpt`, `Ass-1/persistent/adapter`
- This roadmap: `Ass-2/Assignment-2B-roadmap.md` in the clone, and the published copy

## Not verified here

- The model folders could not be checked from this clone, because they are not in git. Confirm they exist on the lab server in Step 0.
- The Hugging Face model facts (bge-base-en-v1.5: 768 dimensions and a 512-token limit; all-MiniLM-L6-v2: 256 tokens) are from the model cards, not checked from here. The notebook must print them (Step 2), and the printed values are what count.
- The Ass-1 README points to `Ass-1/SPRINT-PLAN.md` for the team's execution sequence. That file is not in the clone.
