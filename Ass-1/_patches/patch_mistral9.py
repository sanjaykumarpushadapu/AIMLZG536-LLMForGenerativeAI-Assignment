import json, sys
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
def put(i, s): C[i]["source"] = s.splitlines(keepends=True)
if "no_repeat_ngram_size" not in S(50):
    print("already patched"); sys.exit(0)
s = S(50)
a = "        # no_repeat_ngram_size=4 stops the model repeating the same 4 words.\n"
assert a in s
s = s.replace(a, "        # No repeat-blocking setting is used: it also looks at the prompt, so it stopped answers that copy words from the\n        # question (for example 'countries and territories') in the middle of a word.\n", 1)
b = "do_sample=False, no_repeat_ngram_size=4,"
assert b in s
s = s.replace(b, "do_sample=False,", 1)
put(50, s)
t = S(46); old = "For a small model and a short QLoRA run, a few hundred"
assert old in t
put(46, t.replace(old, "For a short QLoRA run, a few hundred", 1))
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
