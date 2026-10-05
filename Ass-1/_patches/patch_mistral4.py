import json, sys, ast
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
def put(i, s): C[i]["source"] = s.splitlines(keepends=True)
if "scoring {label}" in S(40):
    print("already patched"); sys.exit(0)
s = S(40)
def rep(old, new):
    global s
    assert old in s, old[:60]; s = s.replace(old, new, 1)
rep("def perplexity(m, seqs):", "def perplexity(m, seqs, label=None):")
rep("        for ids in seqs:\n", "        for k, ids in enumerate(seqs, 1):\n")
rep("            total_tokens += targets.numel()  # count the tokens that were predicted\n",
    "            total_tokens += targets.numel()  # count the tokens that were predicted\n"
    "            if label and (k % 10 == 0 or k == len(seqs)):   # progress line, rewritten in place\n"
    "                print(f\"\\r  scoring {label}: {k} of {len(seqs)} sequences\", end=\"\", flush=True)\n"
    "    if label:\n        print()\n")
rep('with working("Loading the model"):\n    base_eval_model', 'with working("Loading the base model"):\n    base_eval_model')
rep('with working("Loading the model"):\n    cpt_eval_model', 'with working("Loading the CPT model"):\n    cpt_eval_model')
rep("perplexity(base_eval_model, eval_seqs)", 'perplexity(base_eval_model, eval_seqs, "the base model")')
rep("perplexity(cpt_eval_model, eval_seqs)", 'perplexity(cpt_eval_model, eval_seqs, "the CPT model")')
put(40, s)
for c in C:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).split("\n") if not l.strip().startswith(("%", "!"))))
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
