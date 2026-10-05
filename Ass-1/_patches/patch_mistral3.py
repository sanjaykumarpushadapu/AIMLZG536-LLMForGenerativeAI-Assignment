import json, re, ast, sys
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
def put(i, s): C[i]["source"] = s.splitlines(keepends=True)
if "class working" in S(18):
    print("already patched"); sys.exit(0)
helper = '''# Slow steps (loading or saving a 14 GB model) print one line that updates every few seconds, with the time
# passed and, when a folder is given, how much data is there so far. This shows the step is still running.
import threading
import time as _time_mod
from pathlib import Path as _Path


def _folder_gb(folder):
    """Size of everything inside a folder in GB (0 if it does not exist yet)."""
    try:
        return sum(f.stat().st_size for f in _Path(folder).rglob("*") if f.is_file()) / 1024**3
    except OSError:
        return 0.0


class working:
    """Context manager: shows 'label ... 2m10s' (and the folder size, if given) while the code inside runs."""

    def __init__(self, label, folder=None, expected_gb=None):
        """Remember the message and, optionally, the folder to measure and its expected final size."""
        self.label, self.folder, self.expected_gb = label, folder, expected_gb

    def _line(self):
        """Build the status text for the current moment."""
        secs = int(_time_mod.time() - self.t0)
        text = f"{self.label} ... {secs // 60}m{secs % 60:02d}s"
        if self.folder is not None:
            gb = _folder_gb(self.folder)
            text += f"  ({gb:.1f} of ~{self.expected_gb:.1f} GB)" if self.expected_gb else f"  ({gb:.1f} GB so far)"
        return text

    def _run(self):
        """Background loop: rewrite the status line every 5 seconds until the step ends."""
        while not self._stop.wait(5):
            print("\\r" + self._line() + "   ", end="", flush=True)

    def __enter__(self):
        """Start the status line."""
        self.t0, self._stop = _time_mod.time(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        """Stop the status line and print the final time."""
        self._stop.set()
        self._thread.join()
        print("\\r" + self._line().replace(" ...", " ... done in", 1) + "   ")
        return False


'''
s = S(18); m = "def repeat_share"; assert m in s
put(18, s.replace(m, helper + m, 1))

# drop the plain "Loading ..." prints that the status line replaces
for i in (24, 40, 49):
    s = S(i)
    s = re.sub(r'^print\("Loading the[^\n]*\n', '', s, flags=re.M)
    put(i, s)

pat = re.compile(r'^(\s*)(\w+) = AutoModelForCausalLM\.from_pretrained\(.*$', re.M)
for i in (24, 40, 42, 49, 50):
    s = S(i)
    put(i, pat.sub(lambda m: f'{m.group(1)}with working("Loading the model"):\n{m.group(1)}    {m.group(0).strip()}', s))
s = S(24)
s = s.replace('with working("Loading the model"):', '''_hub_dir = _Path(__import__("huggingface_hub").constants.HF_HUB_CACHE) / ("models--" + MODEL_ID.replace("/", "--"))
try:
    from huggingface_hub import HfApi
    _expected = sum(f.size or 0 for f in HfApi().model_info(MODEL_ID, files_metadata=True).siblings
                    if f.rfilename.endswith(".safetensors")) / 1024**3 or None
except Exception:
    _expected = None
with working("Loading the model (downloading it the first time)", folder=_hub_dir, expected_gb=_expected):''', 1)
put(24, s)

s = S(33); old = "    cpt_model = build_cpt_model()\n"; assert old in s
put(33, s.replace(old, '    with working("Loading the model for training"):\n        cpt_model = build_cpt_model()\n'))
s = S(35)
old = '        cpt_model = cpt_model.merge_and_unload()\n'; assert old in s
s = s.replace(old, '        with working("Merging the adapter"):\n            cpt_model = cpt_model.merge_and_unload()\n')
old = '    cpt_model.save_pretrained(CPT_CKPT, safe_serialization=True, max_shard_size="1GB")   # small shards keep CPU RAM low\n'; assert old in s
s = s.replace(old, '    with working("Saving the model", folder=CPT_CKPT, expected_gb=total_params * 2 / 1024**3):\n        cpt_model.save_pretrained(CPT_CKPT, safe_serialization=True, max_shard_size="1GB")   # small shards keep CPU RAM low\n')
s = re.sub(r'^\s*print\("(Saving the model|Merging the adapter)[^\n]*\n', '', s, flags=re.M)
put(35, s)

for i, c in enumerate(C):
    if c["cell_type"] == "code":
        t = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.strip().startswith(("%", "!")))
        ast.parse(t)
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched")
