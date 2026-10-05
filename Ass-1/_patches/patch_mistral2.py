import json, sys
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
C = nb["cells"]
def src(i): return "".join(C[i]["source"])
def setsrc(i, s): C[i]["source"] = s.splitlines(keepends=True)
def rep(i, old, new):
    s = src(i); assert old in s, f"cell {i}: not found: {old[:70]!r}"
    setsrc(i, s.replace(old, new, 1))
if "CPT_VAL_FRACTION" in src(29):
    print("already patched"); sys.exit(0)

# ---- cell 29: gentler LoRA settings + validation split
rep(29, "CPT_EPOCHS = 3               # a few passes; many repeats over a small corpus cause forgetting",
        "CPT_EPOCHS = 3               # a few passes; many repeats over a small corpus cause forgetting\nCPT_VAL_FRACTION = 0.06      # last part of the training sequences, held back to watch the loss on unseen text during training")
rep(29, '''    CPT_LORA_R, CPT_LORA_ALPHA, CPT_LORA_DROPOUT = 64, 128, 0.05
    CPT_LR = 1e-4
    CPT_WARMUP_FRACTION = 0.10''',
'''    # A first run with r=64, learning rate 1e-4 and 3 epochs cut the training loss a lot but made the
    # held-out perplexity worse (overfitting a small corpus), so the adapter is smaller, the rate lower,
    # and the run shorter. The validation loss below is watched during training and the best point is kept.
    CPT_LORA_R, CPT_LORA_ALPHA, CPT_LORA_DROPOUT = 32, 64, 0.1
    CPT_LR = 3e-5
    CPT_EPOCHS = 2
    CPT_WARMUP_FRACTION = 0.10
    CPT_SAVE_EVERY = 20''')

# ---- cell 31: progress with time left
rep(31, "from transformers import TrainerCallback\n", "import time as _time\nfrom transformers import TrainerCallback\n")
rep(31, '''        self.history: list[dict] = []
''', '''        self.history: list[dict] = []
        self._t0 = None
''')
rep(31, '''        self.history.append(entry)''', '''        if self._t0 is None:
            self._t0, self._s0 = _time.time(), entry["step"]
        self.history.append(entry)''')
rep(31, '''              f"loss {entry['loss']:.4f}  lr {entry['learning_rate'] or 0:.2e}  grad_norm {grad}")''',
'''              f"loss {entry['loss']:.4f}  lr {entry['learning_rate'] or 0:.2e}  grad_norm {grad}")
        # Every 10 steps say how far the run is and how long it will probably take to finish.
        done = entry["step"] - self._s0
        if entry["step"] % 10 == 0 and done > 0 and state.max_steps:
            left_min = (_time.time() - self._t0) / done * (state.max_steps - entry["step"]) / 60
            print(f"  --- progress: step {entry['step']} of {state.max_steps}, about {left_min:.0f} min left ---")''')

# ---- cell 33
rep(33, "steps_per_epoch = math.ceil(len(train_ds) / (CPT_BATCH_SIZE * CPT_GRAD_ACCUM))",
'''# The last CPT_VAL_FRACTION of the training sequences are only used to measure the loss on unseen text during
# training (Step 5A uses different, whole held-out documents and stays untouched).
from torch.utils.data import Subset
_n_val = max(4, round(len(train_ds) * CPT_VAL_FRACTION))
train_part = Subset(train_ds, range(len(train_ds) - _n_val))
val_part = Subset(train_ds, range(len(train_ds) - _n_val, len(train_ds)))
steps_per_epoch = math.ceil(len(train_part) / (CPT_BATCH_SIZE * CPT_GRAD_ACCUM))''')
rep(33, "        save_total_limit=1,  # keep only the latest checkpoint",
'''        save_total_limit=2,  # latest checkpoint plus the best one
        per_device_eval_batch_size=CPT_BATCH_SIZE,
        eval_strategy="steps",  # measure the validation loss while training
        eval_steps=CPT_SAVE_EVERY,
        load_best_model_at_end=bool(CPT_SAVE_CKPT),  # go back to the step with the lowest validation loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,''')
rep(33, "trainer = Trainer(model=cpt_model, args=args, train_dataset=train_ds, callbacks=[loss_cb])",
'''trainer = Trainer(model=cpt_model, args=args, train_dataset=train_part, eval_dataset=val_part, callbacks=[loss_cb])

    # Validation loss before any training (the adapter starts at zero, so this is the starting model).
    base_val_loss = None
    if not last_ckpt:
        print("Measuring the validation loss before training ...")
        base_val_loss = float(trainer.evaluate()["eval_loss"])
        print(f"Validation loss before training: {base_val_loss:.4f}")
    print("Training started. A progress line is printed every 10 steps.")''')
rep(33, "    loss_history = loss_cb.history\n",
'''    loss_history = loss_cb.history
    val_history = [{"step": int(h["step"]), "eval_loss": float(h["eval_loss"])}
                   for h in trainer.state.log_history if "eval_loss" in h]
    best_val = min((v["eval_loss"] for v in val_history), default=None)
    best_val_step = int(trainer.state.best_global_step) if getattr(trainer.state, "best_global_step", None) else None
''')
rep(33, '        "train_sequences": len(train_ds),',
'''        "train_sequences": len(train_part),
        "val_sequences": len(val_part),
        "base_val_loss": base_val_loss,
        "val_history": val_history,
        "best_val_loss": best_val,
        "best_val_step": best_val_step,
        "best_kept": bool(CPT_SAVE_CKPT),''')

# ---- cell 34: plot validation loss
rep(34, "ax.set_xlabel(f\"optimizer step",
'''_vh = cpt_run.get("val_history") or []
if _vh:
    ax.plot([v["step"] for v in _vh], [v["eval_loss"] for v in _vh], color="tab:orange", marker="o", linewidth=2,
            label="validation loss (unseen text)")
ax.set_xlabel(f"optimizer step''')

# ---- cell 35: progress prints
rep(35, "    if CPT_MODE == \"lora\":\n        # Fold", "    if CPT_MODE == \"lora\":\n        print(\"Merging the adapter into the model weights ...\")\n        # Fold")
rep(35, "    CPT_CKPT.mkdir(parents=True, exist_ok=True)\n", "    CPT_CKPT.mkdir(parents=True, exist_ok=True)\n    print(\"Saving the model (about 14 GB in small pieces, this takes a few minutes) ...\")\n")

# ---- cell 37: text
rep(37, "We kept the learning rate small and the run short so the model learns the domain without losing its general knowledge, which Step 5B checks.",
        "We kept the run short and watched the loss on held-back sequences (below) so the model learns the domain without losing its general knowledge, which Step 5B checks.")
rep(37, "# Sentence about peak GPU memory", '''# Sentence about the validation loss measured during training.
if _r.get("val_history"):
    _vl = [v["eval_loss"] for v in _r["val_history"]]
    _vtxt = (f"\\n\\n**Validation loss.** The last {_r['val_sequences']} training sequences were not trained on; their loss was measured every "
             f"{CPT_SAVE_EVERY} steps" + (f" and was {_r['base_val_loss']:.3f} before training" if _r.get("base_val_loss") is not None else "") +
             f". It was {_vl[0]:.3f} at the first check, lowest at {_r['best_val_loss']:.3f}" +
             (f" (step {_r['best_val_step']})" if _r.get("best_val_step") else "") + f", and {_vl[-1]:.3f} at the end. " +
             ("The model from the lowest validation loss was kept. " if _r.get("best_kept") else "") +
             ("A validation loss that rises while the training loss keeps falling means the model starts to memorise the training text. " if _vl[-1] > _r["best_val_loss"] + 0.01 else
              "The validation loss did not rise at the end, so there is no sign of memorising the training text. ") +
             "Step 5A uses different, whole held-out documents, so it is not affected by this choice.")
else:
    _vtxt = ""
# Sentence about peak GPU memory''')
rep(37, "{_plateau}{_warm}", "{_plateau}{_warm}{_vtxt}")

# ---- progress messages elsewhere
rep(24, "model = AutoModelForCausalLM.from_pretrained(MODEL_ID,", "print(\"Loading the model. The first time it is downloaded (about 14 GB), which can take 30-60 minutes; later runs take a minute or two.\")\nmodel = AutoModelForCausalLM.from_pretrained(MODEL_ID,")
rep(40, "base_eval_model = AutoModelForCausalLM", "print(\"Loading the base and CPT models (about a minute each) ...\")\nbase_eval_model = AutoModelForCausalLM")
rep(49, "# Load the CPT model (from Part A) in 4-bit on GPU 0.", "print(\"Loading the CPT model in 4-bit (about a minute) ...\")\n# Load the CPT model (from Part A) in 4-bit on GPU 0.")
rep(49, "# Train and time it.\nt0 = time.time()", "# Train and time it.\nprint(\"Adapter training started; the progress bar below shows the steps (a few minutes).\")\nt0 = time.time()")

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
