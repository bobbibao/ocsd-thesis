"""CPU test of the OCSD-XL paper benchmark path (assets/xl_paper_v1): the 72 scenes are copied from a fake Drive
benchmark and checked, a wrong benchmark is refused, blind sheets are made only for the locked subset, and the whole
confirmation path (tests/test_xl_confirm.py) runs on two of the scenes, one of them a single-object scene.
python tests/test_xl_paper.py"""
import json, os, runpy, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cv2
import numpy as np
from ocsd import xl_study

DRIVE = os.environ.get("XLP_DRIVE", "/tmp/ocsd_xl_paper_drive")
SRC = os.path.join(DRIVE, "data/benchmarks/quickdraw")
COPY = os.path.join(ROOT, "assets/paper_v3/quickdraw")
MANIFEST = os.path.join(ROOT, "assets/paper_v3/manifest.json")
if os.path.exists(COPY) or os.path.exists(MANIFEST):
    sys.exit("assets/paper_v3 holds imported scenes; this test writes there. Remove them first.")
shutil.rmtree(DRIVE, ignore_errors=True)
expected = json.load(open(os.path.join(ROOT, "assets/paper_v3/expected_scenes.json")))["scenes"]
H3 = os.path.join(ROOT, "assets/heldout_v3/quickdraw")
base = {}
for name in sorted(os.listdir(H3)):
    base.setdefault(json.load(open(os.path.join(H3, name, "scene.json")))["n_obj"], name)
try:
    # a fake benchmark with the expected 72 scenes, built from committed held-out scenes cut to the expected object count
    for sid, want in expected.items():
        src = os.path.join(H3, base[max(k for k in base if k <= max(want["n_obj"], 3))] if want["n_obj"] <= max(base)
                           else base[max(base)])
        meta, d = json.load(open(os.path.join(src, "scene.json"))), os.path.join(SRC, sid)
        os.makedirs(d)
        n = want["n_obj"]
        idx = [i % meta["n_obj"] for i in range(n)]
        masks = np.load(os.path.join(src, "masks.npz"))["masks"][idx]
        sketch = np.full((512, 512), 255, np.uint8)
        for j, i in enumerate(idx):
            obj = cv2.imread(os.path.join(src, f"obj_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
            sketch = np.minimum(sketch, obj)
            cv2.imwrite(os.path.join(d, f"obj_{j:02d}.png"), obj)
        cv2.imwrite(os.path.join(d, "sketch.png"), sketch)
        np.savez(os.path.join(d, "masks.npz"), masks=masks)
        meta.update(sid=sid, n_obj=n, count_bin=want["count_bin"], complexity=want["complexity"],
                    objects=[meta["objects"][i] for i in idx],
                    relations=[r for r in meta["relations"] if r[0] < min(n, meta["n_obj"]) and r[1] < min(n, meta["n_obj"])])
        json.dump(meta, open(os.path.join(d, "scene.json"), "w"))

    try:
        xl_study.load_protocol("xl_paper_v1")
        raise AssertionError("a protocol that imports its scenes must be given the Drive root")
    except ValueError:
        pass
    p = xl_study.load_protocol("xl_paper_v1", DRIVE)
    ident = p["scene_identity"]
    assert len(p["scenes"]) == 72 and len(os.listdir(COPY)) == 72 and ident["same_as_earlier_run"] is False
    assert len(ident["dataset_id"]) == 64 and p["scene_import"]["earlier_run_dataset_id"] != ident["dataset_id"]
    # the dataset identity is the one ocsd.provenance.dataset_identity gives for the same files in a data root
    from ocsd.config import Paths
    from ocsd.provenance import dataset_identity
    probe = Paths(os.path.join(DRIVE, "probe"), run_id="r"); probe.makedirs()
    shutil.copytree(SRC, os.path.join(probe.benchmarks, "quickdraw"), dirs_exist_ok=True)
    assert dataset_identity(probe) == ident["dataset_id"]
    assert xl_study.load_protocol("xl_paper_v1", DRIVE)["scene_identity"] == ident          # second import: same manifest
    rv = p["review"]
    assert rv["seeds"] == [0] and len(rv["scenes"]) == 36 == len(xl_study.review_scenes(p)) and xl_study.review_seeds(p) == [0]
    cells = {}
    for sid in rv["scenes"]: cells[sid.rsplit("_", 1)[0]] = cells.get(sid.rsplit("_", 1)[0], 0) + 1
    assert len(cells) == 12 and set(cells.values()) == {3}
    assert xl_study.xl_images_left(DRIVE, "xl_paper_v1") == 72 * 2 * 4
    print("scene import + lock OK:", ident)

    # a benchmark whose scene differs from the expected one is refused
    bad = os.path.join(SRC, "qd_5_simple_004", "scene.json")
    keep = open(bad).read()
    meta = json.loads(keep); meta["count_bin"] = "3"
    json.dump(meta, open(bad, "w"))
    try:
        xl_study.load_protocol("xl_paper_v1", DRIVE)
        raise AssertionError("changed benchmark scene was accepted")
    except ValueError as exc:
        assert "qd_5_simple_004" in str(exc)
    open(bad, "w").write(keep)
    print("changed benchmark refused OK")

    # full path on a single-object scene (reviewed) and a three-object scene (not reviewed)
    os.environ.update(XLC_PROTOCOL="xl_paper_v1", XLC_DRIVE=DRIVE, XLC_SIDS="qd_1_complex_003,qd_3_complex_006",
                      XLC_DIR=os.environ.get("XLC_DIR", "/tmp/ocsd_xl_paper_test"))
    runpy.run_path(os.path.join(ROOT, "tests/test_xl_confirm.py"), run_name="__main__")
    print("ALL XL PAPER TESTS PASSED")
finally:
    shutil.rmtree(COPY, ignore_errors=True)
    if os.path.exists(MANIFEST): os.remove(MANIFEST)
