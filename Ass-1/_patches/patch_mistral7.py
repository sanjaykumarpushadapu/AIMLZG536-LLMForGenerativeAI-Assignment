import json, sys, ast
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
def put(i, s): C[i]["source"] = s.splitlines(keepends=True)
if "if f.is_file() and not f.is_symlink()" in S(18):
    print("already patched"); sys.exit(0)
def rep(i, old, new):
    s = S(i); assert old in s, (i, old[:70]); put(i, s.replace(old, new, 1))
# folder size: do not count the symlinks of the Hugging Face cache twice
rep(18, 'for f in _Path(folder).rglob("*") if f.is_file())', 'for f in _Path(folder).rglob("*") if f.is_file() and not f.is_symlink())')
rep(24, 'with working("Loading the model (downloading it the first time)"', 'with working("Loading the model (it is downloaded the first time)"')
# step 5 wording when the change is tiny
rep(43, 'f"{_overlap:.1f}% of the 8-word phrases in the held-out documents also appear in the training documents, so the drop is not explained by copied text."',
        'f"{_overlap:.1f}% of the 8-word phrases in the held-out documents also appear in the training documents, so copied text does not explain the result."')
rep(43, 'f"{_overlap:.1f}% of the 8-word phrases in the held-out documents also appear in the training documents, so part of the drop may come from shared wording."',
        'f"{_overlap:.1f}% of the 8-word phrases in the held-out documents also appear in the training documents, so part of the result may come from shared wording."')
rep(43, '(small models are weak at plain facts and repeat themselves)', '(base models are often weak at plain facts and repeat themselves)')
rep(43, '''+ "The small learning rate and short run adapted the model without clearly erasing what it could already do.")''',
        '''+ ("CPT changed the model very little, and nothing it could already do was clearly erased." if abs(_red) < 2 else
                       "The small learning rate and short run adapted the model without clearly erasing what it could already do."))''')
# B2 text: eval loss trend
rep(50, '''{"The eval loss fell, so the adapter generalises to held-out pairs." if len(_s['eval_loss_per_epoch']) > 1 and _s['eval_loss_per_epoch'][-1] <= _s['eval_loss_per_epoch'][0] else "The eval loss did not fall, so the adapter may be overfitting or underfitting."}''',
        '''{_eval_text}''')
rep(50, "# Short name for the training summary, used in the text below.\n_s = sft_summary\n",
'''# Short name for the training summary, used in the text below.
_s = sft_summary
_ev = _s["eval_loss_per_epoch"]
_best = _ev.index(min(_ev)) if _ev else 0
if len(_ev) > 1 and _best < len(_ev) - 1:
    _eval_text = (f"The eval loss was lowest after epoch {_best + 1} ({_ev[_best]:.3f}) and rose a little afterwards (to {_ev[-1]:.3f}), "
                  "so the last epoch starts to overfit slightly. The differences are small, and the loss is still below its first value." if _ev[-1] <= _ev[0] else
                  f"The eval loss was lowest after epoch {_best + 1} ({_ev[_best]:.3f}) and was higher at the end ({_ev[-1]:.3f}), so the adapter overfits a little.")
elif len(_ev) > 1 and _ev[-1] <= _ev[0]:
    _eval_text = "The eval loss fell, so the adapter generalises to held-out pairs."
else:
    _eval_text = "The eval loss did not fall, so the adapter may be overfitting or underfitting."
''')
for c in C:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).split("\n") if not l.strip().startswith(("%", "!"))))
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
