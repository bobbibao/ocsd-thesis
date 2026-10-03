"""CPU test of the global prompt P_g with the caption (OCSDConfig.use_caption): the caption is appended only when it
adds something to the object phrases + background, and the object / background token groups do not move.
python tests/test_prompt.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import open_clip
from tiny import make_clip_tokenizer
from ocsd.sketch import Scene, SceneObject, caption_adds_info, make_caption, scene_global_prompt

W = os.environ.get("PROMPT_DIR", "/tmp/ocsd_prompt")
tok = make_clip_tokenizer(os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz"),
                          os.path.join(W, "tok"))
tok.add_tokens(["<o0>", "<o1>", "<o2>"])


def scene(phrases, bg, caption=None):
    m = np.zeros((8, 8), bool)
    objs = [SceneObject(p.split()[-1], p, (0, 0, 4, 4), m, m) for p in phrases]
    return Scene("t", m, objs, bg, caption if caption is not None else make_caption(phrases, bg))


# QuickDraw-style caption (built from the phrases): nothing is appended
qd = scene(["brown horse", "white sheep", "white sheep"], "in a green meadow")
assert not caption_adds_info(qd)
for ids in (None, ["<o0>", "<o1>", "<o2>"]):
    a = scene_global_prompt(tok, qd, ids)
    b = scene_global_prompt(tok, qd, ids, caption=True)
    assert a == b, (a, b)

# COCO-style caption: appended after the background as group 'cap', other groups unchanged
coco = scene(["person", "horse"], "on a sandy beach", "A man riding a horse on a sandy beach.")
assert caption_adds_info(coco)
for ids in (None, ["<o0>", "<o1>"]):
    a = scene_global_prompt(tok, coco, ids)
    b = scene_global_prompt(tok, coco, ids, caption=True)
    print(b.text, b.groups)
    assert b.text.startswith(a.text) and b.text.endswith("riding a horse on a sandy beach")
    assert {k: v for k, v in b.groups.items() if k != "cap"} == a.groups
    assert b.groups["cap"] and min(b.groups["cap"]) > max(a.groups["bg"])
    full = tok(b.text).input_ids
    assert [full[i] for i in b.groups["cap"]] == tok("A man riding a horse on a sandy beach", add_special_tokens=False).input_ids

# a caption longer than the 77-token window is cut; the object groups stay inside it
long = scene(["person", "horse"], "on a sandy beach", "a man " + "riding a very large brown horse " * 20)
b = scene_global_prompt(tok, long, None, caption=True)
assert max(b.groups["cap"]) <= 75 and b.groups["obj1"] == scene_global_prompt(tok, long, None).groups["obj1"]
print("PROMPT TEST OK")
