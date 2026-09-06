# Assignment 1A — Execution Plan

## At a Glance

- **Total marks:** 15 → Part A (CPT) = 10, Part B (QLoRA) = 5
- **Team:** 4 people (our own split — the brief doesn't require this size)
- **Deadline:** not stated in either brief — confirm on Canvas / Ops mail before locking the schedule
- **Effort:** ~2.5–3 working days
- Checkboxes throughout are for tracking — tick them as your group finishes each item.
- **Locked:** Medical & Clinical Literature (Variant 1, default) · `microsoft/biogpt-large`

## One-Page Overview

| Step | What happens | Marks | Owner | Output file |
|---|---|---|---|---|
| 1 | Extract + clean domain PDFs | 2 | P1 | `domain_corpus/*.txt` |
| 2 | Tokenize + pack into training data | 2 | P2 | `packed_train.parquet`, `packed_eval.parquet` |
| 3 | Load model + inspect architecture | 2 | P3 | `baseline_generations.json` |
| 4 | Run CPT training | 2 | P2 | `cpt_ckpt/` |
| 5 | Evaluate perplexity + forgetting | 2 | P3 | `ppl_results.json` |
| B1 | Build instruction dataset | 2 | P1 | `instruction_dataset.jsonl` |
| B2 | QLoRA fine-tune one adapter | 2 | P4 | `adapter/` |
| B3 | Final 3-way comparison | 1 | P4 | write-up in notebook |

Flow: Steps 1→2→3→4→5 build and evaluate the base CPT model (Part A). B1→B2→B3 then turn that
CPT model into an instruction-following one (Part B). Step 4's checkpoint is the hinge — nothing
in Step 5 or Part B can start before it's saved.

---

## Before You Start

### Accounts & setup

- [ ] 🕒 Hugging Face account + access token — start now, this has lead time
- [ ] 🕒 Accept licenses for any gated model you might use — start now
  - Gated: `mistralai/Mistral-7B-v0.1`, `meta-llama/Meta-Llama-3-8B`, `google/gemma-7b`
  - Ungated (no waiting): `Qwen/Qwen2.5-*`, `HuggingFaceTB/SmolLM2-*`, `TinyLlama/TinyLlama-1.1B-*`, `openai-community/gpt2-*`, `microsoft/biogpt-large`, `stanford-crfm/BioMedLM`
- [ ] Compute confirmed: T4 (16 GB, free Colab) or A100/L40S (BITS remote lab)
- [ ] Persistent storage set up — a Colab session disk does not survive a disconnect, and Step 4's checkpoint must survive
- [ ] External LLM access arranged, only if B1 will use synthetic generation
- [ ] Libraries installed **and imports verified**: `transformers`, `peft`, `bitsandbytes`, `trl`, `torch`, a PDF extractor (`pypdf`/`PyMuPDF`/`pdfplumber`), `pyarrow`/`pandas`, `matplotlib`, `langdetect`
  - ⚠️ `bitsandbytes` fails at *import*, not install — check this before Step 1 starts, not during Step 4

### Decisions to lock (45 min, whole team — changing these later costs a re-run)

- [x] **Variant + domain** — **V1 (Default), Medical & Clinical Literature.** Open choice of medical/clinical PDFs; no fixed sources, no disclaimer requirement (that's V4 only).
- [x] **Model** — **`microsoft/biogpt-large`** (347M, ungated). Confirmed from its HF config:

  | Field | Value |
  |---|---|
  | Context window (`max_position_embeddings`) | 2048 |
  | Hidden size | 1600 |
  | Decoder layers | 48 |
  | Attention heads | 25 |
  | Head dim (1600 / 25) | 64 |
  | Vocab size | 57,717 |
  | `bos_token_id` / `eos_token_id` | 0 / 2 |

  ⚠️ Two things this model changes versus a Llama-family pick:
  - **No chat template.** BioGPT is a plain causal LM, not instruction-tuned — `tok.chat_template` will be empty. B2 must set one manually and document it (the brief explicitly allows this).
  - **LoRA target module names differ.** BioGPT's attention layer is `q_proj` / `k_proj` / `v_proj` / **`out_proj`** — not `o_proj`. Adapter B (`q_proj`, `v_proj`) works unchanged; Adapter C needs `target_modules=["q_proj","v_proj","out_proj"]`.
- [ ] **3 domain prompts** — used in Step 3, Step 4, and B3
- [ ] **3 general prompts** — used in Step 5B (e.g. capital of France / water boils at / speed of light)
- [ ] **Instruction-dataset size** — the brief only says "suitable size." Lock it at **500 pairs → 400 train / 100 eval**.
- [ ] **Filenames** — exactly as listed in the Step-by-Step section below

---

## Who Owns What

| Person | Focus | Steps | Marks |
|---|---|---|---|
| **P1** | Corpus & instruction data | Step 1, B1 | 4 |
| **P2** | Packing & CPT training | Step 2, Step 4 | 4 |
| **P3** | Model audit & evaluation | Step 3, Step 5 | 4 |
| **P4** | QLoRA & submission | B2, B3, notebook assembly | 3 + integration |

## Schedule

| Block | Effort | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| **1 — Setup** | 3–4 h | Step 1, first 5 files | write Step 2 script | Step 3 complete | verify imports, notebook skeleton |
| **2 — Train** | 1 day | Step 1 done, hand off | Step 2, then Step 4 | Step 5A base PPL, Step 5 done | assemble Steps 1–3 |
| **3 — Tune** | 1 day | B1 done | — | write Step 5 inferences | B2, then B3 |
| **4 — Ship** | ½ day | inferences | inferences | inferences | run notebook top-to-bottom, export HTML |

- **Block 1 gate:** run a 20-step CPT test on 5 documents before closing Block 1. The output is
  throwaway — the point is catching environment bugs while there's still time to fix them.
- **Step 4 is the hard deadline inside the schedule.** Step 5, B2, and B3 cannot start until its
  checkpoint is saved, so it must finish in Block 2, no slipping into Block 3.

---

## Step-by-Step

Each step below follows the same shape: what it needs, what it produces, the tasks, what to
report, and how to know you're done.

### Step 1 — Data Collection, Extraction & Cleaning
**2 marks · Owner: P1**

- **Needs:** raw PDFs you've downloaded
- **Produces:** `domain_corpus/*.txt`, `cleaning_stats.json`

**Tasks:**
- [ ] Download 8–10+ domain PDFs from your variant's sources into `raw_pdfs/`
- [ ] Extract text page-by-page (`pypdf`/`PyMuPDF`), join pages, write one `.txt` per PDF into `domain_corpus/`
- [ ] Record the raw document count
- [ ] Length filter — drop documents under ~1,000 characters; record the count after
- [ ] Deduplication — exact hash, then near-duplicate via MinHash/shingles; record the count after
- [ ] Language filter — keep English only (`langdetect`); record the count after
- [ ] Add one more cleaning step of your own (the brief invites this: "not confined to the
  following") — e.g. strip repeated headers/footers — and justify it; that justification is the
  "inference" the rubric rewards. `scripts/step1_extract_clean.py` already implements this as its
  own tracked stage, measured in characters removed since it doesn't drop whole documents.
- [ ] Write `cleaning_stats.json` with every stage's count

**Report:**
- [ ] Document counts before and after each step
- [ ] Which step had the greatest impact on corpus size, and a short paragraph on why

**Done when:** cleaned `.txt` files exist, stats are recorded, and the write-up names which
filter dominated and why that fits the domain.

---

### Step 2 — Tokenization & Packed Dataset
**2 marks · Owner: P2**

- **Needs:** Step 1's `.txt` files
- **Produces:** `packed_train.parquet`, `packed_eval.parquet`, `pack_stats.json`

**Tasks:**
- [ ] Load the tokenizer with `AutoTokenizer.from_pretrained(MODEL_ID)` — the same id as Step 3, never a custom-trained one
- [ ] For each `.txt`: wrap its token ids with BOS at the start and EOS at the end
- [ ] Concatenate every document's ids into one flat stream
- [ ] Slice the stream into fixed-length chunks equal to the model's context window (`config.max_position_embeddings` — 2048 for `biogpt-large`); drop the remainder; no padding
- [ ] Hold out 10% of the chunks as the eval split (Step 5A needs this exact split, unseen in training)
- [ ] Save both splits as Parquet

**Report:**
- [ ] Total token count
- [ ] Average document length in tokens
- [ ] Total number of packed sequences

**Done when:** both Parquet files exist and all three figures are printed.

---

### Step 3 — Model Loading & Architecture Inspection
**2 marks · Owner: P3**

- **Needs:** the locked model id
- **Produces:** `baseline_generations.json`

**Tasks:**
- [ ] Load with `AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)`
- [ ] On T4: also call `model.gradient_checkpointing_enable()`. On A100: full precision or bf16, either is fine
- [ ] Count trainable parameters
- [ ] Audit the architecture from `model.config`: decoder layers, attention heads, hidden size, head dim (expect 48 / 25 / 1600 / 64 for `biogpt-large`)
- [ ] Confirm `model.lm_head.out_features == config.vocab_size` (57,717 for `biogpt-large`)
- [ ] Generate on the 3 locked domain prompts and save the text — this is your "before" baseline
- [ ] While the base model is still loaded, also generate on the 3 general prompts (saves a reload later for Step 5B)

**Report:**
- [ ] Trainable parameter count
- [ ] Layers, heads, hidden size, head dim
- [ ] The `lm_head` confirmation
- [ ] The baseline generations

**Done when:** `baseline_generations.json` holds 6 outputs (3 domain + 3 general) and the audit
numbers are printed.

---

### Step 4 — CPT Training Loop & Loss Analysis
**2 marks · Owner: P2**

- **Needs:** Step 2's Parquet files + Step 3's loaded model
- **Produces:** `cpt_ckpt/`, the loss curve plot

**Tasks:**
- [ ] Wrap the packed dataset in a PyTorch `Dataset` class (`__getitem__` returns `input_ids` and `labels`, same tensor for causal LM)
- [ ] Configure `TrainingArguments`: LR ≈ `2e-5`, warmup, `bf16=True`, gradient accumulation for a sensible effective batch, `logging_steps=1`
- [ ] Add a loss callback — subclass `TrainerCallback`, override `on_log`, collect `logs["loss"]`
- [ ] Run `Trainer(...).train()`
- [ ] Check the first logged loss immediately: **2–4 is correct**, **~10.8 means the model loaded with random weights** — stop and fix the loading rather than training through it
- [ ] Plot loss vs. training step and mark where it plateaus
- [ ] Save both `model` and `tokenizer` to `cpt_ckpt/` on persistent storage

**Report:**
- [ ] The loss curve plot
- [ ] The starting loss
- [ ] Where it plateaus

**Done when:** `cpt_ckpt/` holds both model and tokenizer, and the plateau is marked on the plot.

---

### Step 5 — Evaluation: Perplexity & Catastrophic Forgetting
**2 marks · Owner: P3**

- **Needs:** `packed_eval.parquet` + `cpt_ckpt/`
- **Produces:** `ppl_results.json`, `forgetting_table.md`

**5A — Domain perplexity**

`PPL = exp( −(1/N) Σ log P(tᵢ | t₁…tᵢ₋₁) )`

**Tasks:**
- [ ] Run one eval loop with `model.eval()` and `torch.no_grad()` — no gradients, no training
- [ ] Accumulate token-level cross-entropy over `packed_eval.parquet`, then `ppl = exp(total_loss / total_tokens)`
- [ ] Run it once for the base model, once for the CPT model, same split both times
- [ ] Compute `reduction% = (base − cpt) / base × 100`

**Report:** base PPL · CPT PPL · percentage drop (a healthy run drops domain PPL 10–40%; lower = success)

**5B — Catastrophic forgetting**

**Tasks:**
- [ ] Generate on the 3 general prompts with both models (base outputs already saved in Step 3)
- [ ] Build a side-by-side table with a verdict column:

| Prompt | Base output | CPT output | Verdict |
|---|---|---|---|
| The capital of France is… | … | … | Retained / Degraded |

- [ ] If outputs degrade badly: cut the learning rate 10× or halve `max_steps` and re-run — this usually keeps the domain gain

**Done when:** both perplexities, the percentage drop, the 3-row verdict table, and a paragraph
on whether the trade-off was worth it all exist.

---

### B1 — Instruction Dataset Creation
**2 marks · Owner: P1**

- **Needs:** Step 1's `.txt` files
- **Produces:** `instruction_dataset.jsonl`, `instruction_train.jsonl`, `instruction_eval.jsonl`

**Tasks:**
- [ ] Chunk the cleaned text into ~50 readable passages
  - ⚠️ Check this against Step 1's actual corpus size once it's done — 8–10 short/medium PDFs
    may not yield 50 distinct, decent-length passages. If not, collect more source PDFs or scale
    the pair target down rather than forcing thin passages (see the risk register).
- [ ] Generate 10 instruction/response pairs per passage → 500 total, by hand/heuristic or via an external LLM using: *"Read the text below and generate 10 instruction-response pairs in JSON format based ONLY on this text. Each entry must have instruction and response keys."*
- [ ] If synthetic, save the exact prompt template — required in the submission
- [ ] Validate every JSONL line has `instruction` and `response` keys, and that responses come from your domain text, not invented
  - *(V4 clinical variant only: every response must carry the educational-use disclaimer)*
- [ ] Shuffle and split 80/20 → 400 train / 100 eval

**Report:**
- [ ] Train and eval counts
- [ ] The exact generation prompt template

**Done when:** all three JSONL files validate and the counts are printed.

---

### B2 — QLoRA Fine-Tuning
**2 marks · Owner: P4**

- **Needs:** `cpt_ckpt/` + B1's training split
- **Produces:** `adapter/`

**Tasks:**
- [ ] Set up 4-bit quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`
- [ ] Load **`cpt_ckpt/`** (not the original base model) with that config
- [ ] Call `prepare_model_for_kbit_training(model)`
- [ ] Pick one adapter and state why:

  | Adapter | r | α | Target modules | Effect |
  |---|---|---|---|---|
  | A — low | 8 | 16 | `q_proj`, `v_proj` | Faster; may underfit |
  | **B — balanced (default)** | **16** | **32** | **`q_proj`, `v_proj`** | Good quality/cost |
  | C — high | 32 | 32 | `q_proj`, `v_proj`, `out_proj`\* | Best quality; slower, more VRAM |

  \* the brief's table says `o_proj` — that's the Llama-family name. On `biogpt-large` the same module is called `out_proj`; use that name or Adapter C will silently match nothing.

- [ ] Format every pair with the model's chat template (`tok.apply_chat_template(...)`); if there's no template, set `tok.chat_template` yourself and document it
- [ ] Train with `SFTTrainer` on the 400-pair training split
- [ ] Save the adapter

**Report:**
- [ ] The adapter config chosen and the reasoning
- [ ] Training loss

**Done when:** the adapter loads onto `cpt_ckpt/` and generates without error.

---

### B3 — Evaluation Analysis
**1 mark · Owner: P4**

- **Needs:** B2's trained adapter
- **Produces:** written analysis in the notebook

**Tasks:**
- [ ] Run the adapter on the same 3 locked domain prompts from Step 3
- [ ] Lay out all three stages side by side:

| Prompt | Base (Step 3) | CPT (Step 4) | CPT + adapter (B2) |
|---|---|---|---|

- [ ] Write the observations — did the base model ramble? Did CPT get fluent in domain vocabulary but still complete rather than answer? Did the adapter make it actually respond to the instruction? Naming that progression is the point of the assignment.

**Report:**
- [ ] The three-way comparison table
- [ ] Observations that name the behavioural shift, not just "the output got better"

**Done when:** the table and the write-up both exist and name the shift explicitly.

---

## Submission

| File | Contents |
|---|---|
| `Assignment1A.ipynb` | Notebook with outputs — Steps 1–5, B1–B3, inferences throughout |
| `Assignment1A.html` | Exported HTML of the notebook, with outputs |
| `instruction_dataset.jsonl` | Final instruction/response pairs |
| `domain_corpus/*.txt` | Cleaned text files from Step 1 |
| *(custom variant only)* | Custom-variant template, in the V2–V6 format |

**Final checks:**
- [ ] Restart the kernel and run top to bottom
- [ ] Every step has a written inference (the brief calls this mandatory twice)
- [ ] Every figure listed above is visible in the output

**Optional, no marks:** a chat loop that routes queries to different adapters via `model.set_adapter()`.

---

## Risk Register

| Risk | Signal | Response |
|---|---|---|
| Gated model not approved | 401/403 on load | Accept the license at hour zero, or switch to an ungated model |
| `bitsandbytes` import fails | ImportError at runtime | Match the wheel to the CUDA build — catch this in Block 1 |
| Model initialised randomly | Start loss ≈10.8, not 2–4 | Fix `from_pretrained`; never train through it |
| Tokenizer mismatch | `lm_head` dim ≠ vocab size | One tokenizer id everywhere, always the model's own |
| OOM during CPT | CUDA OOM | Smaller model, gradient checkpointing, lower batch / higher grad accumulation |
| Catastrophic forgetting | 5B outputs clearly degrade | Cut LR 10× or halve `max_steps`, re-run |
| No chat template (certain — `biogpt-large` ships none) | `SFTTrainer`/`apply_chat_template` errors in B2 | Set `tok.chat_template` manually before B2 and document exactly what you set |
| LoRA target module typo | Adapter C matches zero modules, trains nothing | Use `out_proj` (not `o_proj`) for `biogpt-large`'s attention output projection |
| Corpus too small | PPL drop under 10% | Collect more documents, or train more epochs |
| Too few passages for 500 pairs | B1 can't reach ~50 distinct passages from 8–10 PDFs | Collect more source PDFs, allow shorter/overlapping passages, or scale the pair target down and document why |
| Checkpoint lost | Session disconnect | Save `cpt_ckpt/` to mounted persistent storage |
| GPU contention | Queue builds up near the deadline | Step 4 finishing in Block 2 is non-negotiable |

---

## Downstream Note

The Enterprise Variants guide confirms **2A, 2B, and 2C** reuse the `.txt` corpus and
`instruction_dataset.jsonl` built here. It doesn't name **1B** directly, but 1B is grouped with
1A in the same guide, so it most likely reuses this corpus too — **confirm with the instructor**
rather than assuming. Either way, clean this corpus properly once: at least three later
assignments depend on it.
