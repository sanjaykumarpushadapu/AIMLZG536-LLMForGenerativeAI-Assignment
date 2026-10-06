#!/bin/bash
# usage: bash apply_all.sh Assignment1A.ipynb   (runs the three patches in order; each skips if already applied)
set -e
d=$(dirname "$0")
for p in patch_mistral.py patch_mistral2.py patch_mistral3.py patch_mistral4.py patch_mistral5.py patch_mistral6.py patch_mistral7.py; do python3 "$d/$p" "$1"; done
