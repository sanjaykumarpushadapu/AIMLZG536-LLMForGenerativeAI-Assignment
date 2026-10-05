import json, sys, ast
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
if "for the general-text check" in S(42):
    print("already patched"); sys.exit(0)
s = S(42)
a = 'with working("Loading the model"):\n    _m = AutoModelForCausalLM.from_pretrained(BASE_SRC'
assert a in s
s = s.replace(a, 'with working("Loading the base model for the general-text check"):\n    _m = AutoModelForCausalLM.from_pretrained(BASE_SRC', 1)
a = 'with working("Loading the model"):\n    _m = AutoModelForCausalLM.from_pretrained(CPT_CKPT'
assert a in s
s = s.replace(a, 'with working("Loading the CPT model for the general-text check"):\n    _m = AutoModelForCausalLM.from_pretrained(CPT_CKPT', 1)
s = s.replace("gen_base, _, gen_tokens = perplexity(_m, general_seqs)", 'gen_base, _, gen_tokens = perplexity(_m, general_seqs, "the base model")')
s = s.replace("gen_cpt, _, _ = perplexity(_m, general_seqs)", 'gen_cpt, _, _ = perplexity(_m, general_seqs, "the CPT model")')
C[42]["source"] = s.splitlines(keepends=True)
for c in C:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).split("\n") if not l.strip().startswith(("%", "!"))))
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
