"""Hugging Face cache checks. The HF cache must live on local disk (/content/hf_cache), not on Google Drive:
the cache stores snapshots as symlinks to blobs, and Drive's FUSE mount does not support symlinks, which leaves
files that fail with `SafetensorError: header too large`. `repair()` finds and deletes such broken files so the
next from_pretrained() downloads them again."""
from __future__ import annotations

import json
import os
import struct


def check_safetensors(path: str) -> str:
    """Return '' if the file has a valid safetensors header, else the reason it is broken."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return f"too small ({size} bytes)"
            n = struct.unpack("<Q", head)[0]
            if n > min(size - 8, 100 * 2 ** 20):
                return f"header length {n} > file size {size} (partial download or symlink stub)"
            json.loads(f.read(n))
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


def repair(cache_root: str, verbose: bool = True) -> list:
    """Delete broken *.safetensors files (and dangling snapshot links) under cache_root. Returns deleted paths."""
    bad = []
    if not os.path.isdir(cache_root):
        return bad
    for root, _, files in os.walk(cache_root):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p) and not os.path.exists(p):
                bad.append((p, "dangling link"))
            elif f.endswith(".safetensors") or ("/blobs/" in p and not f.endswith(".incomplete")
                                                and os.path.exists(p) and os.path.getsize(p) > 2 ** 20):
                why = check_safetensors(p) if f.endswith(".safetensors") or _looks_safetensors(p) else ""
                if why:
                    bad.append((p, why))
            elif f.endswith(".incomplete"):
                bad.append((p, "incomplete download"))
    for p, why in bad:
        real = os.path.realpath(p)
        for q in {p, real}:
            try:
                os.remove(q)
            except OSError:
                pass
        if verbose:
            print(f"[hf-cache] removed broken file {p}: {why}")
    if verbose:
        print(f"[hf-cache] checked {cache_root}: {len(bad)} broken file(s) removed")
    return [p for p, _ in bad]


def _looks_safetensors(p: str) -> bool:
    # blobs have hash names; a safetensors blob starts with a u64 length followed by '{'
    try:
        with open(p, "rb") as f:
            h = f.read(9)
        return len(h) == 9 and h[8:9] == b"{"
    except OSError:
        return False
