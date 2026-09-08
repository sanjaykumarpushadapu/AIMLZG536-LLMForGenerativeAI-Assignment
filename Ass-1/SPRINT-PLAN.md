# Assignment 1A — Execution Plan

## At a Glance

- **Total marks:** 15 → Part A (CPT) = 10, Part B (QLoRA) = 5
- **Team:** 4 people (our own split — the brief doesn't require this size)
- **Deadline:** not stated in either brief — confirm on Canvas / Ops mail before locking the schedule
- **Effort:** ~2.5–3 working days
- **Ownership model:** contiguous work areas rather than scattered task assignments. P1 owns
  Steps 1–3, P2 owns Steps 4–5, P3 owns B1–B2, and P4 owns B3 plus final integration/QA.
  P4's release work is counted as a real deliverable.
- Checkboxes throughout are for tracking — tick them as your group finishes each item.
- **Chosen:** Variant 1, Medical & Clinical Literature (Type 2 Diabetes) · `microsoft/biogpt-large`
  Variant 1 permits any domain from the model-selection table, so this is a valid team choice,
  not a requirement imposed by the Enterprise Variants guide.
- **Current notebook status:** Steps 1–3 are represented; Step 4 through B3 are still pending
  and must be completed before submission.

## One-Page Overview

| Step | What happens | Marks | Owner | Output file |
|---|---|---|---|---|
| 1 | Extract + clean domain PDFs | 2 | P1 | `domain_corpus/*.txt` |
| 2 | Tokenize + pack into training data | 2 | P1 | `packed_train.parquet`, `packed_eval.parquet` |
| 3 | Load model + inspect architecture | 2 | P1 | `baseline_generations.json` |
| 4 | Run CPT training | 2 | P2 | `cpt_ckpt/` |
| 5 | Evaluate perplexity + forgetting | 2 | P2 | `ppl_results.json` |
| B1 | Build instruction dataset | 2 | P3 | `instruction_dataset.jsonl` |
| B2 | QLoRA fine-tune one adapter | 2 | P3 | `adapter/` |
| B3 | Final 3-way comparison | 1 | P4 | write-up in notebook |

Flow: Steps 1→2→3→4→5 build and evaluate the base CPT model (Part A). B1→B2→B3 then turn that
CPT model into an instruction-following one (Part B). Step 4's checkpoint is the hinge — nothing
in Step 5 or Part B can start before it's saved.

## Ordered Execution Sequence

The team follows one dependency-ordered pipeline. A person prepares their next package only
while waiting for the required handoff; no downstream package is treated as complete before its
input gate is signed off.

| Order | Work package | Owner | Required handoff / gate |
|---:|---|---|---|
| 1 | Step 1 — Extract and clean the domain PDFs | **P1** | P1 signs off `domain_corpus/*.txt` and `cleaning_stats.json` before continuing |
| 2 | Step 2 — Tokenize and pack the corpus | **P1** | P1 signs off both Parquet splits and `pack_stats.json` before continuing |
| 3 | Step 3 — Load, audit, and baseline the model | **P1** | P1 signs off architecture values and six baseline generations; then P2 may start |
| 4 | Step 4 — Run CPT and save the checkpoint | **P2** | P2 signs off loss evidence and persistent `cpt_ckpt/` before continuing |
| 5 | Step 5 — Evaluate PPL and catastrophic forgetting | **P2** | P2 signs off `ppl_results.json`, forgetting table, and written inference; then P3 may start B2 |
| 6 | B1 — Build and validate the instruction dataset | **P3** | P3 signs off `instruction_dataset.jsonl`, train/eval splits, and validation |
| 7 | B2 — Train one QLoRA adapter from `cpt_ckpt/` | **P3** | P3 signs off the adapter configuration, training loss, and load/generate smoke test |
| 8 | B3 — Compare base, CPT, and CPT+adapter | **P4** | P4 signs off the three-way table and behavioral analysis |
| 9 | Release integration and final QA | **P4**, with all owners | Each owner signs their section; P4 runs the notebook top-to-bottom and exports HTML |

**Dependency rule:** the main handoff chain is P1 Steps 1–3 → P2 Steps 4–5 → P3 B1–B2 →
P4 B3/QA. P3 may prepare B1 after P1 has produced cleaned text, but B2 cannot start until
both `cpt_ckpt/` and the validated B1 training split exist.

---

## Before You Start

### Accounts & setup

- [ ] 🕒 Hugging Face account + access token — start now, this has lead time
- [ ] 🕒 Accept licenses for any gated model you might use — start now
  - Gated: `mistralai/Mistral-7B-v0.1`, `meta-llama/Meta-Llama-3-8B`, `google/gemma-7b`
  - Ungated (no waiting): `Qwen/Qwen2.5-*`, `HuggingFaceTB/SmolLM2-*`, `TinyLlama/TinyLlama-1.1B-*`, `openai-community/gpt2-*`, `microsoft/biogpt-large`, `stanford-crfm/BioMedLM`
- [x] **Compute target confirmed:** T4 GPU (Google Colab; the selected runtime reports about
  14.6 GB usable VRAM)
  - Enable gradient checkpointing and use small per-device batches with gradient accumulation.
  - Verify `torch.cuda.is_bf16_supported()` in the runtime before loading/training. The brief
    specifies BF16 for T4; if the selected runtime cannot execute it reliably, stop and agree
    an FP16 fallback before producing graded results.
- [ ] **Persistent storage for final run:** optional during development, but required before
  the final CPT/QLoRA run because a Colab session disk does not survive disconnects. Set
  `USE_GOOGLE_DRIVE = True` in the notebook when persistent Drive storage is needed.
- [ ] External LLM access arranged, only if B1 will use synthetic generation
- [ ] Libraries installed **and imports verified**: `transformers`, `peft`, `bitsandbytes`, `trl`, `torch`, a PDF extractor (`pypdf`/`PyMuPDF`/`pdfplumber`), `pyarrow`/`pandas`, `matplotlib`, `langdetect`
  - ⚠️ `bitsandbytes` fails at *import*, not install — check this before Step 1 starts, not during Step 4

### Decisions to lock (45 min, whole team — changing these later costs a re-run)

- [x] **Variant + domain** — **V1 (Default), Medical & Clinical Literature: Type 2 Diabetes.**
  V1 is open choice within the model-selection table; use authoritative diabetes guidance and
  record the source and license for every PDF. No V4 clinical-protocol disclaimer is required.
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

### Brief alignment checks

- **Assignment 1A:** Part A is 10 marks, Part B is 5 marks, for 15 marks total. Detailed
  inferences are mandatory, and the submission must include the notebook with output, exported
  HTML, `instruction_dataset.jsonl`, and cleaned `domain_corpus/*.txt`.
- **Enterprise Variants guide:** the pipeline and rubric remain fixed across variants. The
  cleaned `.txt` corpus and `instruction_dataset.jsonl` are explicitly reusable by 2A, 2B,
  and 2C, so they should be treated as shared project assets.
- **Model consistency:** use the same `microsoft/biogpt-large` tokenizer throughout. BioGPT has
  no built-in chat template, so the manually defined template must be documented in B2.

## Who Owns What

The team uses contiguous ownership blocks so each person can follow one coherent area instead
of switching between unrelated steps. The marks do not divide evenly under this requested
grouping: P1 owns Steps 1–3 (6 marks), P2 owns Steps 4–5 (4 marks), P3 owns B1–B2 (4 marks),
and P4 owns B3 plus the integration/QA deliverable. Every person documents inferences,
validates outputs, and signs off their handoff.

| Person | Lead work packages | Effort points | Handoff / peer-review duty |
|---|---|---:|---|
| **P1** | Step 1 → Step 2 → Step 3: data, packing, model audit, baseline | 6 | Hand off the complete Part A input/baseline package to P2; review P2's CPT evidence |
| **P2** | Step 4 → Step 5: CPT training, PPL, forgetting | 4 | Receive P1's package, hand off `cpt_ckpt/` and evaluation tables to P3/P4; review P3's dataset evidence |
| **P3** | B1 → B2: instruction data, QLoRA adapter | 4 | Receive P2's checkpoint, hand off validated dataset and adapter to P4; review P1's baseline evidence |
| **P4** | B3 → QA: three-way analysis, notebook assembly, HTML export | 1 + integration | Receive P3's adapter, complete release checks, and collect all owner sign-offs |

**Fairness rule:** P4 owns the mechanical merge, restart-and-run-all check, and HTML export,
but every person owns the correctness and written inference of their own sections. No one can
claim the final integration point without collecting all four peer-review sign-offs.

## Schedule

| Block | Effort | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| **1 — P1 Part A preparation** | 1 day | Complete Steps 1–3 and hand off the full baseline package | Prepare CPT configuration | Prepare B1/B2 environment | Verify T4 environment and notebook skeleton |
| **2 — P2 CPT and evaluation** | 1 day | Answer data handoff questions | Complete Steps 4–5 and hand off checkpoint/evaluation | Prepare instruction-data inputs | Prepare B3 comparison structure |
| **3 — P3 Part B** | 1 day | Review P2's evaluation evidence | Review P3's training inputs | Complete B1–B2 and hand off adapter | Prepare final notebook assembly |
| **4 — P4 release** | ½ day + QA | Sign off Part A | Sign off CPT/evaluation | Sign off dataset/adapter | Complete B3, run top-to-bottom, collect sign-offs, export HTML |

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
  "inference" the rubric rewards. The notebook's `Pipeline` class (Step 1 setup cells) already implements this as its
  own tracked stage, measured in characters removed since it doesn't drop whole documents.
- [ ] Write `cleaning_stats.json` with every stage's count

**Report:**
- [ ] Document counts before and after each step
- [ ] Which step had the greatest impact on corpus size, and a short paragraph on why

**Done when:** cleaned `.txt` files exist, stats are recorded, and the write-up names which
filter dominated and why that fits the domain.

---

### Step 2 — Tokenization & Packed Dataset
**2 marks · Owner: P1**

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
**2 marks · Owner: P1**

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
**2 marks · Owner: P2**

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
**2 marks · Owner: P3**

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
**2 marks · Owner: P3**

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
`instruction_dataset.jsonl` built here. The guide does not explicitly define a reuse rule for
1B, so do not assume one without checking the 1B brief. Either way, clean this corpus properly
once: at least three later assignments depend on it.
