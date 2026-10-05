import json, sys, ast
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
if "within noise" in S(43):
    print("already patched"); sys.exit(0)
s = S(43)
def rep(old, new):
    global s
    assert old in s, old[:70]; s = s.replace(old, new, 1)
rep("if _red >= 10:\n    _ppl_text", '''if abs(_red) < 2:
    _ppl_text = (f"a **{_red:+.1f}%** change, which is within noise. The base model already predicts these documents well "
                 f"(PPL {_pr['base_ppl']:.2f}), so CPT on this small corpus left little to improve. This is below the 10-40% the brief calls typical.")
elif _red >= 10:
    _ppl_text''')
rep('''{"so CPT made the model less surprised by guideline text, which means it learned the vocabulary and phrasing of this domain." if _red > 0 else "so this does not confirm domain adaptation."}''',
    '''{"so CPT made the model less surprised by guideline text, which means it learned the vocabulary and phrasing of this domain." if _red >= 2 else "so this does not show a measurable domain adaptation."}''')
rep("if _red > 0 and _n_deg == 0:\n    _worth", '''if abs(_red) < 2 and _n_deg == 0:
    _worth = ("CPT changed the model very little: perplexity on held-out guideline text is about the same and general behaviour was kept. "
              "A strong base model leaves little room for CPT on a small corpus, so the gain is negligible here, but nothing was lost.")
elif _red > 0 and _n_deg == 0:
    _worth''')
C[43]["source"] = s.splitlines(keepends=True)
for c in C:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).split("\n") if not l.strip().startswith(("%", "!"))))
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
