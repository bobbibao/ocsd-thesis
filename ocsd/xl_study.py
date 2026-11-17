"""OCSD-XL studies: locked scenes x seeds x arms on SDXL, resumable, scored with the evaluator of every earlier run.

Layout under <drive>/<protocol root>/:
  outputs/quickdraw/<arm>/<sid>_s<seed>.png|json   512 px image used for scoring (area-downsampled) + log
  hires/<arm>/<sid>_s<seed>.jpg                    the 1024 px image
  m2/<sid>_s<seed>.jpg                             the M2 objects placed on a white canvas (diagnostic)
  results/                                          status.json, logs/, per-image CSVs, summary.md|json, review/, ZIPs

Protocols live in assets/<name>/protocol.json and are checked against the code before anything runs.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .provenance import atomic_json, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "quickdraw"
METRICS = ["opr", "miou", "count_acc", "ra", "joint_layout_success", "sketch_chamfer", "facing_acc", "clip",
           "obj_clip", "pickscore", "time_s"]
LOWER_IS_BETTER = {"sketch_chamfer", "time_s"}


# ---------------------------------------------------------------------------------------------- protocol
def arm_digest(arms):
    from .ocsd_xl import ROUND, arm_config, arm_m2, m2_config
    d = {a: (arm_config(a)[0], arm_config(a)[1].to_dict()) for a in arms}
    if ROUND != "v1":       # round 1 was locked before M2 variants existed; its digest must not change
        d["__m2__"] = {a: (arm_m2(a), m2_config(arm_m2(a)).to_dict()) for a in arms}
    return digest(d)


def import_scenes(p, drive_root):
    """Protocols with "scene_import": the scenes are not committed. Copy them from the benchmark on Drive, check them
    against the committed list of expected scenes (ocsd.heldout) and return what identifies the copied set."""
    from . import heldout
    spec = p["scene_import"]
    heldout.use(spec["heldout_version"])
    files = heldout.import_paper_scenes(Path(drive_root) / spec["source"])
    if sorted({rel.split("/")[1] for rel in files}) != sorted(p["scenes"]):
        raise ValueError("Imported scenes differ from the locked scene list")
    # the identity that the earlier locked run recorded for its installed scenes (ocsd.provenance.dataset_identity
    # over <data>/benchmarks_v2/quickdraw/...)
    dataset_id = digest({"benchmarks_v2/" + rel: h for rel, h in files.items()})
    return dict(manifest_sha256=file_hash(ROOT / p["scene_assets"] / "manifest.json"), dataset_id=dataset_id,
                same_as_earlier_run=dataset_id == spec.get("earlier_run_dataset_id"))


def load_protocol(name: str, drive_root=None) -> dict:
    """Read assets/<name>/protocol.json and refuse to run when scenes or arm settings differ from the lock."""
    from .engine_xl import XL_MODELS
    p = json.loads((ROOT / "assets" / name / "protocol.json").read_text())
    p["name"] = name
    from . import ocsd_xl
    ocsd_xl.use_round(p.get("round", "v1"))
    if p.get("scene_import"):
        if drive_root is None:
            raise ValueError(f"Protocol {name} copies its scenes from Drive: pass the Drive root")
        p["scene_identity"] = import_scenes(p, drive_root)
    scenes = ROOT / p["scene_assets"] / SPLIT
    manifest = json.loads((ROOT / p["scene_assets"] / "manifest.json").read_text())
    for sid in p["scenes"]:
        for rel, h in manifest.items():
            if rel.split("/")[1] == sid and file_hash(ROOT / p["scene_assets"] / rel) != h:
                raise ValueError(f"Scene file differs from its manifest: {rel}")
        if not (scenes / sid / "scene.json").exists():
            raise FileNotFoundError(f"Scene {sid} not found in {scenes}")
    if p["arm_config_sha256"] != arm_digest(p["arms"]):
        raise ValueError("Arm settings in ocsd/ocsd_xl.py differ from the locked protocol")
    if p["models"] != XL_MODELS:
        raise ValueError("Model repositories differ from the locked protocol")
    if p.get("sd15_config_sha256") and p["sd15_config_sha256"] != sd15_digest(p):
        raise ValueError("SD 1.5 method settings differ from the locked protocol")
    return p


def all_arms(p):
    """Every arm that is scored and reviewed: the SDXL arms, the SD 1.5 arms generated in this run and the arms whose
    images are imported from an earlier locked run."""
    return list(p["arms"]) + list(p.get("sd15_arms", [])) + list(p.get("imported_arms", {}))


def scene_dirs(p):
    return [str(ROOT / p["scene_assets"] / SPLIT / sid) for sid in p["scenes"]]


class Dirs:
    def __init__(self, drive_root, p):
        self.root = Path(drive_root) / p["drive_root"]
        self.outputs, self.hires, self.m2 = self.root / "outputs", self.root / "hires", self.root / "m2"
        self.results = self.root / "results"
        for d in (self.outputs / SPLIT, self.hires, self.m2, self.results / "logs"):
            d.mkdir(parents=True, exist_ok=True)

    def png(self, arm, sid, seed):
        return self.outputs / SPLIT / arm / f"{sid}_s{seed}.png"


def _status(dirs, **kw):
    path = dirs.results / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(path, state)
    return state


def _signature(p, arm, sid, seed):
    from .ocsd_xl import arm_config
    kind, cfg = arm_config(arm)
    return digest(dict(protocol=p["name"], arm=arm, kind=kind, cfg=cfg.to_dict(), sid=sid, seed=seed,
                       models=p["models"]))


def _done(dirs, p, arm, sid, seed):
    png = dirs.png(arm, sid, seed)
    try:
        meta = json.loads(png.with_suffix(".json").read_text())
        return meta["signature"] == _signature(p, arm, sid, seed) and meta["sha256"] == file_hash(png)
    except (OSError, ValueError, KeyError):
        return False


def xl_images_left(drive_root, name):
    """How many SDXL images of a protocol are not on Drive yet (no model or scene is loaded). The notebook uses it:
    scoring and the SD 1.5 arms also run on a smaller GPU once this is 0."""
    p = json.loads((ROOT / "assets" / name / "protocol.json").read_text())
    out = Path(drive_root) / p["drive_root"] / "outputs" / SPLIT
    return sum(not ((out / a / f"{sid}_s{seed}.png").exists() and (out / a / f"{sid}_s{seed}.json").exists())
               for a in p["arms"] for sid in p["scenes"] for seed in p["seeds"])


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


# ---------------------------------------------------------------------------------------------- generation
def generate(drive_root, p, eng=None, vis=None, max_minutes=None, log=print):
    """Scene-major loop, so the first scene exercises every arm within minutes. An arm that fails three images in a
    row is switched off for the rest of the run (the others continue); a later Run all tries it again."""
    import cv2
    import torch
    from .data import load_scene
    from .ocsd_xl import ARMS, arm_m2, compose_preview, make_objects, run_arm, to_eval_size, upscale_scene
    dirs = Dirs(drive_root, p)
    arms, seeds = list(p["arms"]), list(p["seeds"])
    jobs = [(sdir, seed) for sdir in scene_dirs(p) for seed in seeds]
    total = len(jobs) * len(arms)
    todo = {(sdir, seed): [a for a in arms if not _done(dirs, p, a, Path(sdir).name, seed)] for sdir, seed in jobs}
    left = sum(len(v) for v in todo.values())
    log(f"[xl] {p['name']}: {total - left}/{total} images already complete, {left} to generate")
    _status(dirs, stage="generate", total=total, complete=total - left, failures=[])
    if not left:
        return eng, vis
    if eng is None:
        from .engine_xl import EngineXL
        eng = EngineXL.from_pretrained(cache_dir=os.environ.get("HF_HOME"), models=p["models"])
    if vis is None:
        from .vision import Vision
        vis = Vision(str(eng.device), cache_dir=os.environ.get("HF_HOME"))
    start = time.time()
    streak, off, failures, complete = {}, set(), [], total - left
    for sdir, seed in jobs:
        arms_left = [a for a in todo[(sdir, seed)] if a not in off]
        if not arms_left:
            continue
        if max_minutes and (time.time() - start) / 60 > max_minutes:
            log(f"[xl] time budget of {max_minutes} min reached; Run all again to continue")
            break
        scene = load_scene(sdir)
        sid = scene.sid
        scene_xl = upscale_scene(scene, int(p.get("size", 1024)))
        if hasattr(vis, "set_scene"): vis.set_scene(scene_xl)      # CPU tests only (fake detector)
        m2_sets = {}
        for variant in sorted({arm_m2(a) for a in arms_left if ARMS[a][0] != "controlnet"}):
            m2_log = {}
            try:
                before = dict(eng.compute)
                m2_sets[variant] = (make_objects(eng, vis, scene_xl, seed, m2_log, variant), m2_log)
                m2_log["compute"] = eng.compute_delta(before)
                tag = "" if variant == "a" else f"_m2{variant}"
                cv2.imwrite(str(dirs.m2 / f"{sid}_s{seed}{tag}.jpg"),
                            cv2.cvtColor(to_eval_size(compose_preview(scene_xl, m2_sets[variant][0])),
                                         cv2.COLOR_RGB2BGR))
            except Exception as exc:
                failures.append(dict(arm=f"M2-{variant}", sid=sid, seed=seed, error=repr(exc),
                                     traceback=traceback.format_exc()))
                log(f"[xl] M2 ({variant}) FAILED for {sid} seed {seed}: {exc!r}")
                _status(dirs, failures=failures[-20:])
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        for arm in arms_left:
            objs, m2_log = m2_sets.get(arm_m2(arm), (None, {}))
            if objs is None and ARMS[arm][0] != "controlnet":
                continue
            run_log = {}
            t0 = time.time()
            try:
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                img = run_arm(arm, eng, vis, scene_xl, seed, objs, run_log)
                if img.dtype != np.uint8 or img.ndim != 3 or not np.isfinite(img).all():
                    raise RuntimeError("Generated image is not a finite uint8 RGB array")
                if float(img.std()) < 2.0:
                    raise RuntimeError("Generated image is blank (half-precision overflow?)")
                png = dirs.png(arm, sid, seed)
                png.parent.mkdir(parents=True, exist_ok=True)
                (dirs.hires / arm).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dirs.hires / arm / f"{sid}_s{seed}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(png), cv2.cvtColor(to_eval_size(img), cv2.COLOR_RGB2BGR))
                run_log.setdefault("scene_time", time.time() - t0)
                atomic_json(png.with_suffix(".json"), _jsonable(dict(
                    method=arm, sid=sid, seed=seed, n_obj=scene.n, m2=m2_log, **run_log,
                    wall_s=time.time() - t0, signature=_signature(p, arm, sid, seed), sha256=file_hash(png))))
                streak[arm] = 0
                complete += 1
                log(f"[xl] {complete}/{total} {sid} s{seed} {arm}: {time.time() - t0:.0f} s "
                    f"| elapsed {(time.time() - start) / 60:.0f} min")
            except Exception as exc:
                streak[arm] = streak.get(arm, 0) + 1
                failures.append(dict(arm=arm, sid=sid, seed=seed, error=repr(exc), traceback=traceback.format_exc(),
                                     m3={k: v for k, v in run_log.items() if k.startswith("m3")}))
                log(f"[xl] FAILED {sid} s{seed} {arm}: {exc!r}")
                traceback.print_exc()
                if streak[arm] >= 3:
                    off.add(arm)
                    log(f"[xl] arm {arm} failed 3 images in a row; switched off for this run")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            _status(dirs, complete=complete, failures=failures[-20:], arms_off=sorted(off),
                    minutes=(time.time() - start) / 60)
    (dirs.results / "logs" / "failures.json").write_text(json.dumps(failures, indent=1))
    return eng, vis


# ---------------------------------------------------------------------------------------------- SD 1.5 arms
def import_legacy(drive_root, p, log=print):
    """Copy the images of an earlier locked run on the same scenes and seeds (protocol "imported_arms": arm -> folder
    under the Drive root) into this study, so that one evaluator scores every arm and the blind sheets show them all.
    Returns the (arm, sid, seed) triples that are not on Drive."""
    import shutil
    dirs = Dirs(drive_root, p)
    missing, record = [], {}
    for arm, src in p.get("imported_arms", {}).items():
        for sid in p["scenes"]:
            for seed in p["seeds"]:
                a, b = Path(drive_root) / src / f"{sid}_s{seed}.png", dirs.png(arm, sid, seed)
                ok = a.exists()
                if ok and a.with_suffix(".json").exists():      # the earlier run logged each image's checksum
                    try:
                        ok = json.loads(a.with_suffix(".json").read_text()).get("sha256", file_hash(a)) == file_hash(a)
                    except ValueError:
                        ok = False
                if not ok:
                    if not b.exists(): missing.append((arm, sid, seed))
                    continue
                if not b.exists() or file_hash(a) != file_hash(b):
                    b.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(a, b)
                    if a.with_suffix(".json").exists(): shutil.copy2(a.with_suffix(".json"), b.with_suffix(".json"))
                record[f"{arm}/{sid}_s{seed}"] = file_hash(b)
    atomic_json(dirs.results / "imported.json", dict(sources=p.get("imported_arms", {}), sha256=record,
                                                     missing=[list(m) for m in missing]))
    if p.get("imported_arms"):
        log(f"[xl] imported {len(record)} images of {sorted(p['imported_arms'])} from the earlier run; "
            f"{len(missing)} not found on Drive")
    return missing


def sd15_configs(p):
    """The SD 1.5 methods with the settings of the earlier locked comparison (tier + frozen paper_v2 tuning)."""
    from .config import ExperimentConfig
    from .stages import tier_baselines, tier_variants
    E = ExperimentConfig(tier=p["sd15_tier"])
    variants, base = tier_variants(E)
    return E, variants, base, tier_baselines(E)


def sd15_digest(p):
    _, variants, _, baselines = sd15_configs(p)
    names = list(p.get("sd15_arms", [])) + list(p.get("sd15_fallback", []))
    return digest({m: (variants[m].to_dict() if m in variants else [baselines[m][0], baselines[m][1].to_dict()])
                   for m in names})


def generate_sd15(drive_root, p, missing=(), eng=None, vis=None, max_minutes=None, log=print):
    """Generate the SD 1.5 arms of this study with the legacy runner: the arms in "sd15_arms" (not part of the
    earlier run) and, only for images that could not be imported, the arms in "sd15_fallback"."""
    dirs = Dirs(drive_root, p)
    want = {(m, sid, s) for m in p.get("sd15_arms", []) for sid in p["scenes"] for s in p["seeds"]}
    want |= {m for m in missing if m[0] in p.get("sd15_fallback", [])}
    want = {(m, sid, s) for m, sid, s in want
            if not (dirs.png(m, sid, s).exists() and dirs.png(m, sid, s).with_suffix(".json").exists())}
    if not want:
        return eng, vis
    log(f"[xl] SD 1.5 arms: {len(want)} images to generate ({sorted({m for m, _, _ in want})})")
    _status(dirs, stage="generate_sd15", sd15_todo=len(want))
    from .runner import Runner
    E, variants, base, baselines = sd15_configs(p)
    if eng is None:
        from .engine import Engine
        eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=os.environ.get("HF_HOME"), load_adapter=False)
    if vis is None:
        from .vision import Vision
        vis = Vision(str(eng.device), cache_dir=os.environ.get("HF_HOME"))
    runner = Runner(eng, vis, str(dirs.outputs), E.backbone, variants=variants, base_cfg=base, baselines=baselines,
                    cache_dir=os.environ.get("HF_HOME"))
    items = [(SPLIT, sdir, sorted((m, s) for m, sid, s in want if sid == Path(sdir).name)) for sdir in scene_dirs(p)]
    stats = runner.run_items([x for x in items if x[2]], max_minutes=max_minutes)
    _status(dirs, sd15=stats)
    return eng, vis


# ---------------------------------------------------------------------------------------------- evaluation
class PickScore:
    """PickScore v1 (Kirstain et al., NeurIPS 2023): CLIP-H fine-tuned on human preferences between generated
    images. Reported as 100 x cosine(image, caption); higher = preferred. A learned proxy, not a human rating."""
    MODEL, PROCESSOR = "yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(self, device="cuda", cache_dir=None):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.PROCESSOR, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(self.MODEL, cache_dir=cache_dir).to(device).eval()
        self.device = device

    def score(self, img, text):
        import torch
        from PIL import Image
        with torch.no_grad():
            a = self.proc(images=[Image.fromarray(img)], return_tensors="pt").to(self.device)
            b = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            ia, tb = self.model.get_image_features(**a), self.model.get_text_features(**b)
            ia = ia.pooler_output if hasattr(ia, "pooler_output") else ia
            tb = tb.pooler_output if hasattr(tb, "pooler_output") else tb
            ia, tb = torch.nn.functional.normalize(ia.float(), dim=-1), torch.nn.functional.normalize(tb.float(), dim=-1)
            return float(100 * (ia * tb).sum())


def _cached_rows(csv, key_cols):
    import pandas as pd
    old = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=key_cols + ["png_sha256"])
    return {tuple(r[k] for k in key_cols): r for r in old.to_dict("records")}


def evaluate(drive_root, p, vis=None, picker=None, log=print):
    """OWLv2 / CLIP metrics (ocsd.metrics, unchanged), SAM sketch fidelity and PickScore for every image present."""
    import cv2
    import pandas as pd
    from .data import load_scene
    from .metrics import evaluate_images
    from .sketch_fidelity import image_fidelity
    dirs = Dirs(drive_root, p)
    _status(dirs, stage="evaluate")
    if vis is None:
        from .vision import Vision
        vis = Vision("cuda", cache_dir=os.environ.get("HF_HOME"))
    arms, seeds = all_arms(p), list(p["seeds"])
    evaluate_images(str(dirs.outputs), "", SPLIT, arms, seeds, vis, str(dirs.results), scene_dirs=scene_dirs(p))
    extra_csv = dirs.results / SPLIT / "extra_metrics.csv"
    cached = _cached_rows(extra_csv, ["method", "sid", "seed"])
    rows, pick_error, fresh = [], None, 0
    for sdir in scene_dirs(p):
        if fresh >= 50:         # checkpoint: rows computed so far + the cached rows of the scenes not reached yet
            seen = {(r["method"], r["sid"], r["seed"]) for r in rows}
            pd.DataFrame(rows + [r for k, r in cached.items() if k not in seen]).to_csv(extra_csv, index=False)
            fresh = 0
        scene = load_scene(sdir)
        if hasattr(vis, "set_scene"): vis.set_scene(scene)
        for arm in arms:
            for seed in seeds:
                png = dirs.png(arm, scene.sid, seed)
                if not png.exists():
                    continue
                h = file_hash(png)
                old = cached.get((arm, scene.sid, seed))
                if old is not None and old["png_sha256"] == h and not (picker is not None and pd.isna(old.get("pickscore"))):
                    rows.append(old)
                    continue
                img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
                _, fid = image_fidelity(img, scene, lambda im, box: vis.sam(im, box))
                row = dict(method=arm, sid=scene.sid, seed=seed, png_sha256=h, **fid, pickscore=float("nan"))
                if picker is not None and pick_error is None:
                    try:
                        row["pickscore"] = picker.score(img, scene.caption)
                    except Exception as exc:        # a quality proxy must not block the structural metrics
                        pick_error = repr(exc)
                        log(f"[xl] PickScore unavailable: {pick_error}")
                rows.append(row)
                fresh += 1
    pd.DataFrame(rows).to_csv(extra_csv, index=False)
    _status(dirs, stage="evaluated", pickscore_error=pick_error)
    return vis


# ---------------------------------------------------------------------------------------------- report
def _boot(values, n=10000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values): return (float("nan"),) * 3
    means = np.random.default_rng(seed).choice(values, (n, len(values))).mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_table(dirs, p):
    import pandas as pd
    frames = [pd.read_csv(dirs.results / SPLIT / f"per_image_{a}.csv") for a in all_arms(p)
              if (dirs.results / SPLIT / f"per_image_{a}.csv").exists()]
    if not frames:
        raise RuntimeError("No evaluated images")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.sid.isin(p["scenes"]) & df.seed.isin(p["seeds"])]
    extra = pd.read_csv(dirs.results / SPLIT / "extra_metrics.csv").drop(columns=["png_sha256"])
    return df.merge(extra, on=["method", "sid", "seed"], how="left", validate="one_to_one")


def review_order(p, sid, seed):
    order = all_arms(p)
    random.Random(f"{sid}:{seed}:{p['review_salt']}").shuffle(order)
    return dict(zip("ABCDEFGHIJKLMNOP", order))


def review_scenes(p):
    """Scenes with a blind sheet: all of them, or the subset fixed in the protocol ("review")."""
    return [sid for sid in p["scenes"] if sid in set(p.get("review", {}).get("scenes", p["scenes"]))]


def review_seeds(p):
    return [s for s in p["seeds"] if s in p.get("review", {}).get("seeds", p["seeds"])]


def make_review(dirs, p, panel=384, columns=6):
    """Blind sheets: input sketch, then every arm under a letter (order shuffled per sheet; key kept separately)."""
    import textwrap
    from PIL import Image, ImageDraw
    from .data import load_scene
    from .direct_canary import review_font
    font, plain = review_font(18)
    out = dirs.results / "review"
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for n, sdir in enumerate([d for d in scene_dirs(p) if Path(d).name in review_scenes(p)], 1):
        scene = load_scene(sdir)
        for seed in review_seeds(p):
            mapping = {k: a for k, a in review_order(p, scene.sid, seed).items() if dirs.png(a, scene.sid, seed).exists()}
            key[f"{scene.sid}_s{seed}"] = mapping
            cells = [("Input sketch", Image.fromarray(scene.sketch).convert("RGB"))] + [
                (label, Image.open(dirs.png(arm, scene.sid, seed)).convert("RGB")) for label, arm in mapping.items()]
            rows = -(-len(cells) // columns)
            head, label_h = 86, 26
            sheet = Image.new("RGB", (panel * columns, head + rows * (panel + label_h)), "white")
            draw = ImageDraw.Draw(sheet)
            text = f"Sheet {n:02d} | {scene.sid} seed {seed} | {scene.n} objects | " + scene.caption + \
                   " | Objects: " + ", ".join(o.phrase for o in scene.objects)
            for i, line in enumerate(textwrap.wrap(plain(text), width=panel * columns // 10)[:3]):
                draw.text((10, 6 + i * 24), line, fill="black", font=font)
            for i, (label, im) in enumerate(cells):
                x, y = (i % columns) * panel, head + (i // columns) * (panel + label_h)
                draw.text((x + 8, y + 2), label, fill="black", font=font)
                sheet.paste(im.resize((panel, panel), Image.Resampling.LANCZOS), (x, y + label_h))
            sheet.save(out / f"{n:02d}_{scene.sid}_seed{seed}.jpg", quality=88)
    atomic_json(dirs.results / "review_key.json", key)
    return key


def report(drive_root, p, log=print):
    import pandas as pd
    dirs = Dirs(drive_root, p)
    df = load_table(dirs, p)
    expected = len(p["scenes"]) * len(p["seeds"])
    metrics = [m for m in METRICS if m in df]
    rows = []
    for arm in all_arms(p):
        d = df[df.method == arm]
        r = dict(method=arm, images=len(d), complete=len(d) == expected)
        for k in metrics:
            r[k], r[k + "_lo"], r[k + "_hi"] = _boot(d.groupby("sid")[k].mean().to_numpy())
        rows.append(r)
    summary = pd.DataFrame(rows)
    by_bin = df.groupby(["method", "count_bin"])[[k for k in ("opr", "miou", "count_acc", "sketch_chamfer") if k in df]] \
        .mean().reset_index()
    ref = p.get("reference_arm")
    paired = []
    if ref and (df.method == ref).any():
        for arm in all_arms(p):
            if arm == ref: continue
            for k in metrics:
                t = df.groupby(["method", "sid"])[k].mean().unstack("method")
                if arm not in t or ref not in t: continue
                t = t.dropna(subset=[arm, ref])
                mu, lo, hi = _boot((t[arm] - t[ref]).to_numpy())
                paired.append(dict(a=arm, b=ref, metric=k, scenes=len(t), diff=mu, lo=lo, hi=hi))
    tables = dirs.results / "tables"
    tables.mkdir(exist_ok=True)
    summary.to_csv(tables / "summary.csv", index=False)
    by_bin.to_csv(tables / "by_count_bin.csv", index=False)
    pd.DataFrame(paired).to_csv(tables / "paired_vs_reference.csv", index=False)
    df.to_csv(tables / "per_image.csv", index=False)
    key = make_review(dirs, p)
    lines = [f"# {p['title']}", "", f"Protocol: assets/{p['name']}/protocol.json. {len(p['scenes'])} scenes x seeds "
             f"{p['seeds']}; means over scenes (seeds averaged) with 95% bootstrap CIs over scenes.",
             "SDXL images are generated at 1024 px and scored at 512 px. sketch_chamfer and time_s: lower is better.", "",
             "| arm | images | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for r in rows:
        lines.append(f"| {r['method']} | {r['images']}/{expected} | " + " | ".join(
            f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" for k in metrics) + " |")
    lines += ["", "## By object count", "", by_bin.round(3).to_markdown(index=False)]
    if paired:
        lines += ["", f"## Paired differences against {ref} (a - b)", "", "| a | metric | diff [95% CI] |", "|---|---|---|"]
        lines += [f"| {r['a']} | {r['metric']} | {r['diff']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] |" for r in paired]
    (dirs.results / "summary.md").write_text("\n".join(lines) + "\n")
    atomic_json(dirs.results / "summary.json", _jsonable(dict(protocol=p["name"], summary=rows, paired=paired,
                                                              by_count_bin=by_bin.to_dict("records"))))
    bundle, review_zip = dirs.results / f"ocsd_{p['name']}_results.zip", dirs.results / f"REVIEW_{p['name']}.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
        for path in sorted(dirs.results.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".md", ".txt", ".log") and "review" not in path.parts:
                z.write(path, path.relative_to(dirs.results).as_posix())
        for path in sorted((dirs.outputs / SPLIT).glob("*/*.json")):
            z.write(path, "outputs/" + path.relative_to(dirs.outputs / SPLIT).as_posix())
    with ZipFile(review_zip, "w", ZIP_DEFLATED) as z:
        for path in sorted((dirs.results / "review").iterdir()):
            z.write(path, "review/" + path.name)
        shown = {f"{sid}_s{seed}" for sid in review_scenes(p) for seed in review_seeds(p)}
        for path in sorted(dirs.m2.glob("*.jpg")):
            if any(path.name.startswith(x) for x in shown):
                z.write(path, "m2/" + path.name)
    # the optional GitHub export (ocsd.sync) only carries logs/tables/figures: give it the sheets too
    import shutil
    fig = dirs.results / "figures" / "review_sheets"
    fig.mkdir(parents=True, exist_ok=True)
    for path in sorted((dirs.results / "review").iterdir()):
        shutil.copy2(path, fig / path.name)
    _status(dirs, stage="reported", sheets=len(key))
    log(f"[xl] results ZIP: {bundle}\n[xl] review ZIP: {review_zip}")
    return dict(summary=str(dirs.results / "summary.md"), archive=str(bundle), review_archive=str(review_zip))


# ---------------------------------------------------------------------------------------------- entry point
def main(drive_root, protocol="xl_dev_v1", max_minutes=None):
    """Generate -> evaluate -> report. Everything is resumable; logs go to <root>/results/logs/run.log."""
    from .runlog import _Tee
    p = load_protocol(protocol, drive_root)
    dirs = Dirs(drive_root, p)
    log_file = (dirs.results / "logs" / "run.log").open("a")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, log_file), _Tee(old_err, log_file)
    try:
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {protocol} =====", flush=True)
        try:
            import subprocess
            head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            head = None
        _status(dirs, protocol=protocol, code=head, error=None, scene_identity=p.get("scene_identity"))
        if p.get("scene_identity"): print("[xl] scenes:", p["scene_identity"], flush=True)
        import torch
        t_start = time.time()
        missing = import_legacy(drive_root, p)
        eng, vis = generate(drive_root, p, max_minutes=max_minutes)
        del eng
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if p.get("sd15_arms") or missing:
            try:        # a failure here must not lose the SDXL images' scores; Run all again retries it
                left = max(5.0, max_minutes - (time.time() - t_start) / 60) if max_minutes else None
                eng15, vis = generate_sd15(drive_root, p, missing, vis=vis, max_minutes=left)
                del eng15
            except Exception as exc:
                print(f"[xl] SD 1.5 arms FAILED ({exc!r}); continuing with the images that exist", flush=True)
                traceback.print_exc()
                _status(dirs, sd15_error=repr(exc), sd15_traceback=traceback.format_exc())
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        picker = None
        try:
            picker = PickScore("cuda" if torch.cuda.is_available() else "cpu", cache_dir=os.environ.get("HF_HOME"))
        except Exception as exc:
            print(f"[xl] PickScore could not be loaded ({exc!r}); the column stays empty", flush=True)
        evaluate(drive_root, p, vis=vis, picker=picker)
        out = report(drive_root, p)
        print(open(out["summary"]).read())
        return out
    except Exception as exc:
        _status(dirs, stage="failed", error=repr(exc), traceback=traceback.format_exc())
        (dirs.results / "logs" / "LATEST_ERROR.txt").write_text(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_file.close()
