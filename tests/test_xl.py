"""CPU tests of OCSD-XL with a tiny random SDXL-shaped stack: engine packing of the pooled embedding, text_time
conditioning, ControlNet, region attention and energy on the SDXL block layout, identity tokens in both text
encoders, every arm end to end, resume, evaluation and report. python tests/test_xl.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import open_clip
import torch
from tiny import FakeVision, make_engine_xl
from ocsd import ocsd_xl, xl_study
from ocsd.data import load_scene
from ocsd.engine import ID_TOKENS

W = os.environ.get("XL_DIR", "/tmp/ocsd_xl_test")
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
S = 128

# ---------------------------------------------------------------- engine
eng = make_engine_xl(os.path.join(W, "tok"), BPE, S)
emb = eng.encode(["a photo of a red car", "a dog"])
assert emb.shape == (2, 78, 64), emb.shape
assert torch.count_nonzero(emb[:, -1, eng.pool_dim:]) == 0 and torch.count_nonzero(emb[:, -1, :eng.pool_dim]) > 0
z = eng.randn(0, 1, S, S)
assert z.shape == (1, 4, S // 2, S // 2)
ts = eng.set_timesteps(4)
control = eng.control_tensor(np.zeros((S, S, 3), np.uint8))
e1 = eng.cfg_eps(z, ts[0], emb[:1], emb[1:], 6.0)
e2 = eng.cfg_eps(z, ts[0], emb[:1], emb[1:], 6.0, control, 1.0)
assert e1.shape == z.shape and torch.isfinite(e1).all() and not torch.allclose(e1, e2)
assert eng.compute["unet_sample_calls"] == 4 and eng.compute["controlnet_sample_calls"] == 2
# identity tokens exist in both tokenizers at the same position and are restored by reset_identity
layers = eng.embedding_layers()
assert len(layers) == 2 and layers[0][2] == layers[1][2]
assert eng.tokenizer("a <o0> dog").input_ids == eng.tokenizer_2("a <o0> dog").input_ids
for _, layer, rows in layers:
    layer.weight.data[rows[0]] += 1
eng.reset_identity()
for (_, layer, rows), orig in zip(layers, eng._orig_id_embs):
    assert torch.equal(layer.weight.data[rows], orig)
# the pooled row changes the prediction (text_time conditioning is wired)
emb2 = emb.clone(); emb2[:, -1, :eng.pool_dim] += 1
assert not torch.allclose(eng.unet_eps(z, ts[0], emb[:1]), eng.unet_eps(z, ts[0], emb2[:1]))
print("engine OK")

# ---------------------------------------------------------------- scene scaling
scene_dir = os.path.join(ROOT, "assets/heldout_v1/quickdraw/ho_qd_3_simple_000")
scene = load_scene(scene_dir)
up = ocsd_xl.upscale_scene(scene, 1024)
assert up.sketch.shape == (1024, 1024) and up.objects[0].mask.shape == (1024, 1024)
assert np.allclose(np.array(up.objects[0].box), 2 * np.array(scene.objects[0].box))
assert abs((up.sketch < 128).mean() - (scene.sketch < 128).mean()) < 0.02 and scene.sketch.shape == (512, 512)
for o, u in zip(scene.objects, up.objects):
    ys, xs = np.where(u.sketch < 128)
    assert u.box[0] - 4 <= xs.min() and xs.max() <= u.box[2] + 4 and (u.sketch < 128).any()
print("upscale OK")

# ---------------------------------------------------------------- every arm end to end at tiny size
ocsd_xl.XL_COMMON.update(height=S, width=S, steps=6, obj_steps=4)
ocsd_xl.XL_OCSD.update(tau=2)
ocsd_xl.ARMS["ocsd_xl_id"] = ("ocsd", dict(use_identity=True, S1=3, S2=3, lora_scale=0.5, lora_rank=4))
arms = list(ocsd_xl.ARMS)
sids = ["ho_qd_3_simple_000", "ho_qd_5_medium_000"]
p = dict(name="test", title="test", drive_root="xl_test", scene_assets="assets/heldout_v1", scenes=sids, seeds=[0, 1],
         arms=arms, models={}, size=S, review_salt="t", reference_arm="xl_controlnet")
vis = FakeVision()
xl_study.generate(W, p, eng=eng, vis=vis)
dirs = xl_study.Dirs(W, p)
missing = [(a, s, k) for a in arms for s in sids for k in (0, 1) if not dirs.png(a, s, k).exists()]
state = json.load(open(dirs.results / "status.json"))
assert not missing and not state["failures"], (missing, state["failures"][:1])
meta = json.load(open(dirs.png("ocsd_xl", sids[0], 0).with_suffix(".json")))
assert meta["cfg"]["blend_mode"] == "joint" and meta["cfg"]["height"] == S and meta["compute"]["unet_sample_calls"] > 0
assert meta["blend_steps"] > 0 and meta["m2"]["m2_time"] >= 0 and "energy" in meta
assert json.load(open(dirs.png("ocsd_xl_id", sids[0], 0).with_suffix(".json")))["m3_time"] > 0
assert json.load(open(dirs.png("ocsd_xl_region", sids[0], 0).with_suffix(".json")))["blend_steps"] == 0
import cv2
assert cv2.imread(str(dirs.hires / "ocsd_xl" / f"{sids[0]}_s0.jpg")).shape[0] == S
# no state leaks: the ControlNet baseline is identical before and after the OCSD arms ran
a = cv2.imread(str(dirs.png("xl_controlnet", sids[0], 0)))
b = ocsd_xl.to_eval_size(ocsd_xl.run_arm("xl_controlnet", eng, vis, ocsd_xl.upscale_scene(scene, S), 0, None))
assert np.abs(a[..., ::-1].astype(int) - b.astype(int)).max() <= 1
# resume: nothing left to do, and the images are not rewritten
before = {str(q): os.path.getmtime(q) for q in (dirs.outputs / "quickdraw").glob("*/*.png")}
calls = dict(eng.compute)
xl_study.generate(W, p, eng=eng, vis=vis)
assert eng.compute == calls and before == {str(q): os.path.getmtime(q) for q in (dirs.outputs / "quickdraw").glob("*/*.png")}
print("generation + resume OK:", len(before), "images")


class FakePick:
    def score(self, img, text): return float(img.mean() / 10)


xl_study.evaluate(W, p, vis=vis, picker=FakePick())
out = xl_study.report(W, p)
text = open(out["summary"]).read()
assert all(a in text for a in arms) and "pickscore" in text and "sketch_chamfer" in text
summary = json.load(open(dirs.results / "summary.json"))
assert all(r["complete"] for r in summary["summary"]) and summary["paired"]
key = json.load(open(dirs.results / "review_key.json"))
assert len(key) == 4 and all(sorted(v.values()) == sorted(arms) for v in key.values())
assert len(list((dirs.results / "review").glob("*.jpg"))) == 4 and os.path.getsize(out["review_archive"]) > 1000
print("evaluate + report OK")

print("ALL XL TESTS PASSED")
