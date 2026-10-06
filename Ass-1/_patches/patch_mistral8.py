import json, sys
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8")); C = nb["cells"]
s = "".join(C[43]["source"])
old = 'f"a **{_red:+.1f}%** change, which is within noise.'
new = 'f"a **{abs(_red):.1f}%** {\'drop\' if _red > 0 else \'rise\'}, which is within noise.'
if old not in s:
    print("already patched or not found"); sys.exit(0)
C[43]["source"] = s.replace(old, new, 1).splitlines(keepends=True)
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
