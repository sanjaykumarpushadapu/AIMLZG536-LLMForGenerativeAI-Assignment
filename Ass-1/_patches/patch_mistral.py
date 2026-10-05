import json, sys
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
C = nb["cells"]
def src(i): return "".join(C[i]["source"])
def setsrc(i, s):
    C[i]["source"] = s.splitlines(keepends=True)
def rep(i, old, new, count=1):
    s = src(i)
    assert old in s, f"cell {i}: text not found: {old[:60]!r}"
    setsrc(i, s.replace(old, new, count))

if "CPT_MODE" in src(29):
    print("already patched"); sys.exit(0)

# ---- cell 0
rep(0, "**Model:** `stanford-crfm/BioMedLM`", "**Model:** `mistralai/Mistral-7B-v0.1`")

# ---- cell 18: model + context cap
rep(18, 'MODEL_ID = "stanford-crfm/BioMedLM"', 'MODEL_ID = "mistralai/Mistral-7B-v0.1"')
rep(18, "CONTEXT = cfg.max_position_embeddings\n",
"""CONTEXT_CAP = 1024   # some models report a very long window (32,768 here); training on that is far too heavy for one shared GPU
CONTEXT = min(cfg.max_position_embeddings, CONTEXT_CAP)
""")
rep(18, 'print(f"Context window   : {CONTEXT:,} tokens")',
'print(f"Context window   : {CONTEXT:,} tokens" + (f"  (the config allows {cfg.max_position_embeddings:,}; capped at {CONTEXT_CAP:,} to fit the GPU)" if CONTEXT < cfg.max_position_embeddings else ""))')

# ---- cell 24: base copy only when needed
s = src(24)
a = s.index("# Save a bf16 copy of the base model")
setsrc(24, s[:a] + '''# Later steps reload the base model. If the model repo has safetensors files (loaded piece by piece,
# so little CPU RAM is used) they are read straight from the Hugging Face cache. Otherwise (for example a
# big single .bin file) a bf16 copy is saved here in small pieces, because reloading the original
# would need too much RAM on a small pod.
from huggingface_hub import try_to_load_from_cache
BASE_BF16_DIR = PERSIST / "base_bf16"
_has_safetensors = any(isinstance(try_to_load_from_cache(MODEL_ID, f), str)
                       for f in ("model.safetensors", "model.safetensors.index.json"))
if _has_safetensors:
    BASE_SRC = MODEL_ID
    print("Base source      : model repo (safetensors, loaded in pieces) - no extra copy needed")
else:
    if not (BASE_BF16_DIR / "config.json").exists():
        model.save_pretrained(BASE_BF16_DIR, safe_serialization=True, max_shard_size="1GB")
        tok.save_pretrained(BASE_BF16_DIR)
    BASE_SRC = str(BASE_BF16_DIR)
    print(f"Base bf16 copy   : {BASE_BF16_DIR}")
''')

# ---- cell 27: text
rep(27, "so every parameter is trainable. This is full-parameter CPT, unlike the adapter training in Part B.",
        "so every parameter can be trained. Step 4 decides from the free GPU memory whether CPT updates all of them or only a LoRA adapter.")
rep(27, "The count is about {total_params / 1e6:.0f}M.",
        "The count is about {_size_txt}.")
rep(27, "# Build the explanation text.",
    '_size_txt = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M"\n# Build the explanation text.')

# ---- cell 28: recipe
s = src(28)
a = s.index("**Recipe.**")
setsrc(28, s[:a] + "**Recipe.** The config cell below picks the training mode from the GPU memory that is free: full-parameter CPT (FP32 weights and AdamW states for every parameter) when that fits, otherwise LoRA-based CPT on a bf16 copy of the model, where the adapter is merged back into the weights before saving, so Step 5 and Part B still use one normal checkpoint. Gradient checkpointing keeps activation memory small, and we use BF16 autocast, a small learning rate with cosine decay and warmup, and a short run. The mode, batch size, accumulation, learning rate and epochs are printed in the config cell and in the observations after training.")

# ---- cell 29: config
setsrc(29, '''# ---------------------------------------------------------------------------
# Step 4 configuration: training mode, hyperparameters and where checkpoints live.
# ---------------------------------------------------------------------------
import math
import shutil
import torch

CPT_BATCH_SIZE = 4           # sequences per forward pass; lower this if a larger model runs out of GPU memory
CPT_GRAD_ACCUM = 2           # effective batch = BATCH_SIZE x GRAD_ACCUM sequences per step
CPT_EPOCHS = 3               # a few passes; many repeats over a small corpus cause forgetting
CPT_MAX_GRAD_NORM = 1.0
CPT_ADAM_BETA2 = 0.95        # 0.95 instead of the default 0.999: the optimizer adapts faster after a bad batch (common for LLM training)
CPT_SAVE_EVERY = 40          # optimizer steps between resumable checkpoints

# Training mode. Full-parameter CPT with standard AdamW keeps FP32 weights, gradients and two
# optimizer states for every parameter: about 16 bytes per parameter (8-bit AdamW: about 10),
# plus activations. Use the memory that is free right now (plus what this notebook already holds),
# because the GPU may be shared with other jobs. If even the 8-bit variant does not fit with
# some margin, train LoRA adapters on the bf16 model instead and merge them afterwards.
CPT_MODE_OVERRIDE = None     # set to "full" or "lora" to force a mode
_need_full_gb = total_params * 16 / 1e9
_need_8bit_gb = total_params * 10 / 1e9
if torch.cuda.is_available():
    _avail_gb = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_allocated()) / 1e9
else:
    _avail_gb = L.GPU_MEM_GB
if _need_full_gb * 1.3 < _avail_gb:
    CPT_MODE, CPT_OPTIM = "full", "adamw_torch"
elif _need_8bit_gb * 1.3 < _avail_gb:
    CPT_MODE, CPT_OPTIM = "full", "adamw_bnb_8bit"
else:
    CPT_MODE, CPT_OPTIM = "lora", "adamw_torch"
if CPT_MODE_OVERRIDE:
    CPT_MODE = CPT_MODE_OVERRIDE
    CPT_OPTIM = "adamw_torch" if CPT_MODE == "lora" else CPT_OPTIM
print(f"Full CPT would need about {_need_full_gb:.1f} GB (8-bit AdamW {_need_8bit_gb:.1f} GB); available on the GPU {_avail_gb:.1f} GB "
      f"-> mode: {CPT_MODE}, optimizer: {CPT_OPTIM}")

if CPT_MODE == "lora":
    # LoRA-based CPT: a larger adapter than in Part B (it has to absorb a whole domain, not a format),
    # on every linear layer of the transformer blocks. The learning rate is higher than for full training.
    CPT_LORA_R, CPT_LORA_ALPHA, CPT_LORA_DROPOUT = 64, 128, 0.05
    CPT_LR = 1e-4
    CPT_WARMUP_FRACTION = 0.10
    print(f"LoRA CPT         : r={CPT_LORA_R}, alpha={CPT_LORA_ALPHA}, all linear layers, learning rate {CPT_LR:g}")
else:
    CPT_LR = 5e-6                # gentle: CPT should adapt the model, not overwrite it. 2e-5 blew up in one run (loss spiked near step 100), so it was lowered
    CPT_WARMUP_FRACTION = 0.20   # longer warmup avoids early loss spikes

# Resumable checkpoints. Full mode: weights (FP32) plus optimizer states, which is large for a
# big model, so they are only kept if the disk has room. LoRA mode: only the small adapter and its states.
_free_gb = shutil.disk_usage(".").free / 1e9
_final_gb = total_params * 2 / 1e9                       # final model is saved in BF16
if CPT_MODE == "lora":
    _ckpt_gb = 3.0
    CPT_SAVE_CKPT = _free_gb > _ckpt_gb + _final_gb + 5
else:
    _bytes_per_param = 4 + (2 if CPT_OPTIM == "adamw_bnb_8bit" else 8)
    _ckpt_gb = total_params * _bytes_per_param / 1e9
    CPT_SAVE_CKPT = (_free_gb > _ckpt_gb + _final_gb + 5) and _ckpt_gb < 8   # skip very large checkpoints: writing them can crash a lab pod
print(f"Disk free {_free_gb:.1f} GB; resumable checkpoint ~{_ckpt_gb:.1f} GB + final model ~{_final_gb:.1f} GB "
      f"-> resumable checkpoints: {'on' if CPT_SAVE_CKPT else 'off (checkpoint too large to write safely)'}")

CPT_CKPT = PERSIST / "cpt_ckpt"          # final model + tokenizer, read by Step 5 and B2
CPT_RUN_DIR = PERSIST / "cpt_run"        # resumable checkpoints during training
CPT_SUMMARY_PATH = CPT_CKPT / "training_summary.json"   # written last = save completed
CPT_LOSS_LOG = "loss_log.json"

print(f"Checkpoint       : {CPT_CKPT.resolve()}")
print(f"Resumable run dir: {CPT_RUN_DIR.resolve()}")
''')

# ---- cell 32
setsrc(32, src(32).replace('''def build_cpt_model():
    """Load MODEL_ID with FP32 master weights; bf16 autocast happens inside the Trainer."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **{DTYPE_KW: torch.float32}, device_map={"": DEVICE})
    m.config.use_cache = False               # conflicts with gradient checkpointing
    return m.to(DEVICE)''', '''def build_cpt_model():
    """Full mode: FP32 master weights. LoRA mode: bf16 base model plus trainable LoRA adapters.
    bf16 autocast happens inside the Trainer in both cases."""
    if CPT_MODE == "lora":
        from peft import LoraConfig, get_peft_model
        m = AutoModelForCausalLM.from_pretrained(BASE_SRC, **{DTYPE_KW: torch.bfloat16}, device_map={"": DEVICE})
        m.config.use_cache = False           # conflicts with gradient checkpointing
        m.enable_input_require_grads()       # lets gradients reach the adapters through checkpointed blocks
        return get_peft_model(m, LoraConfig(r=CPT_LORA_R, lora_alpha=CPT_LORA_ALPHA, lora_dropout=CPT_LORA_DROPOUT,
                                            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **{DTYPE_KW: torch.float32}, device_map={"": DEVICE})
    m.config.use_cache = False               # conflicts with gradient checkpointing
    return m.to(DEVICE)'''))
assert "CPT_MODE" in src(32)

# ---- cell 33
rep(33, '''    total_params = sum(p.numel() for p in cpt_model.parameters())
    trainable_params = sum(p.numel() for p in cpt_model.parameters() if p.requires_grad)
''', '''    trainable_params = sum(p.numel() for p in cpt_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in cpt_model.parameters())
    if CPT_MODE == "lora":
        total_params -= trainable_params     # report the size of the base model; the adapter is extra
''')
rep(33, '''"optimizer": "adamw_torch",
        "precision": "FP32 master weights with bf16 autocast",''',
'''"mode": CPT_MODE,
        "optimizer": CPT_OPTIM,
        "precision": ("bf16 base weights with FP32 LoRA adapters and bf16 autocast" if CPT_MODE == "lora"
                      else "FP32 master weights with bf16 autocast"),
        "lora_r": CPT_LORA_R if CPT_MODE == "lora" else None,
        "lora_alpha": CPT_LORA_ALPHA if CPT_MODE == "lora" else None,
        "est_full_training_gb": round(_need_full_gb, 1),
        "gpu_available_gb": round(_avail_gb, 1),''')

# ---- cell 35: merge before saving
rep(35, '''    cpt_model.to(torch.bfloat16)             # one uniform BF16 checkpoint, same dtype as the base''',
'''    if CPT_MODE == "lora":
        # Fold the adapter into the base weights so the saved model is an ordinary checkpoint.
        # The loss on a few training sequences is compared before and after merging, because merging
        # in bf16 rounds the weights and the model must behave the same afterwards.
        _ids = torch.stack([torch.as_tensor(train_ds[i]["input_ids"]) for i in range(4)]).to(DEVICE)
        cpt_model.eval()
        with torch.no_grad():
            merge_loss_before = float(cpt_model(input_ids=_ids, labels=_ids).loss)
        cpt_model = cpt_model.merge_and_unload()
        with torch.no_grad():
            merge_loss_after = float(cpt_model(input_ids=_ids, labels=_ids).loss)
        print(f"Loss on 4 training sequences: adapter {merge_loss_before:.4f} -> merged {merge_loss_after:.4f}")
        assert abs(merge_loss_after - merge_loss_before) < 0.1, "merging changed the model too much"
        cpt_run["merge_loss_before"], cpt_run["merge_loss_after"] = merge_loss_before, merge_loss_after
    cpt_model.to(torch.bfloat16)             # one uniform BF16 checkpoint, same dtype as the base''')

# ---- cell 37: setup text + start-loss wording
rep(37, "_start_ok = 2 <= _l[\"start_loss\"] <= 4", "_start_ok = 2 <= _l[\"start_loss\"] <= 4\n_start_low = _l[\"start_loss\"] < 2   # below the range: the model already predicts this text well")
rep(37, "# Sentence about peak GPU memory", '''# Sentence about how the model was trained (full or LoRA, from the run settings).
if _r.get("mode") == "lora":
    _setup = (f"`{MODEL_ID}` has {_r['total_params']:,} parameters. Full-parameter CPT needs about {_r['est_full_training_gb']:.0f} GB of GPU memory "
              f"(weights, gradients and optimizer states) and only {_r['gpu_available_gb']:.0f} GB were free on this shared GPU, so CPT was done with LoRA adapters "
              f"(rank {_r['lora_r']}, alpha {_r['lora_alpha']}, on every linear layer) on the bf16 model. {_r['trainable_params']:,} parameters "
              f"({100 * _r['trainable_params'] / _r['total_params']:.2f}% of the model) were trained, and the adapter was then merged into the weights "
              f"(loss on a few training sequences {_r['merge_loss_before']:.3f} before and {_r['merge_loss_after']:.3f} after merging)." if "merge_loss_before" in _r else
              f"`{MODEL_ID}` has {_r['total_params']:,} parameters. Full-parameter CPT did not fit in the free GPU memory, so CPT was done with LoRA adapters "
              f"(rank {_r['lora_r']}, alpha {_r['lora_alpha']}, every linear layer; {_r['trainable_params']:,} trainable parameters) and merged into the weights afterwards.")
    _setup += " This differs from full-parameter CPT: the model can move less far from its starting point, which also limits forgetting. We used bf16 weights with BF16 autocast, AdamW and gradient checkpointing."
else:
    _setup = (f"`{MODEL_ID}` has {_r['total_params']:,} parameters and all {_r['trainable_params']:,} were trained (full-parameter CPT). "
              f"We used FP32 weights with BF16 autocast (as the brief asks), AdamW and gradient checkpointing.")
# Sentence about peak GPU memory''')
rep(37, '''`{MODEL_ID}` has {_r['total_params']:,} parameters and all {_r['trainable_params']:,} were trained (full-parameter CPT). We used FP32 weights with BF16 autocast (as the brief asks), AdamW and gradient checkpointing.{_peak}''', "{_setup}{_peak}")
rep(37, '''{"inside" if _start_ok else "outside"} the 2–4 range the brief gives for a pretrained model.''',
        '''{"inside" if _start_ok else ("below" if _start_low else "above")} the 2–4 range the brief gives for a pretrained model.''')
rep(37, '''{"So the pretrained weights loaded correctly. Clinical guideline text is somewhat out of domain for this model, so CPT has room to improve it." if _start_ok else "This is outside the expected range, so the model loading should be checked."}''',
        '''{"So the pretrained weights loaded correctly. Clinical guideline text is somewhat out of domain for this model, so CPT has room to improve it." if _start_ok else ("A low start means the pretrained model already predicts this text well (a stronger model finds clinical text easier), and it is far from random, so the weights loaded correctly. There is less room for CPT to improve it." if _start_low else "This is above the expected range, so the model loading should be checked.")}''')

# ---- cell 47: markdown
rep(47, "The brief's table lists `q_proj` and `v_proj`, but layer names differ between models, so they are not used here.",
        "The brief's table lists `q_proj` and `v_proj`. Layer names differ between model families, so the code takes the attention layers peft uses for this model type; for a model with separate query and value layers (such as Mistral) these are `q_proj` and `v_proj`, as in the brief.")

# ---- cell 48: padding side
rep(48, "if tok.pad_token is None:\n    tok.pad_token = tok.eos_token      # batched training needs a pad token",
        "if tok.pad_token is None:\n    tok.pad_token = tok.eos_token      # batched training needs a pad token\ntok.padding_side = \"right\"            # pad after the text for training")

# ---- cell 50: wording
rep(50, "which is expected for a small model with little training data.", "which is expected with this little training data.")
rep(50, "The LoRA target layers are {ADAPTER_TARGETS}, chosen from the model type (the brief's table names `q_proj`/`v_proj`, but layer names depend on the model).",
        "The LoRA target layers are {ADAPTER_TARGETS}, chosen from the model type (the brief's table names `q_proj`/`v_proj`; layer names depend on the model family).")

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
