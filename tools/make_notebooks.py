"""Sinh các notebook Colab từ mã (để notebook luôn khớp với gói ocsd). python tools/make_notebooks.py"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")
os.makedirs(OUT, exist_ok=True)

SETUP = r'''#@title 1. Cấu hình (chỉ sửa ô này)
REPO_URL   = "https://github.com/bobbibao/ocsd-thesis.git"  #@param {type:"string"}
BRANCH     = "main"                                             #@param {type:"string"}
DRIVE_ROOT = "/content/drive/MyDrive/KLTN_OCSD"                 #@param {type:"string"}
TIER       = "pilot"   #@param ["pilot", "paper", "full"]
BACKBONE   = "sd15"    #@param ["sd15", "sd21"]
MAX_MINUTES = 600      #@param {type:"number"}
SKIP_JOBS  = ""        #@param {type:"string"}
HF_TOKEN   = ""        #@param {type:"string"}
# pilot: kiểm tra nhanh (~1-2 giờ GPU); paper: đủ số liệu cho khóa luận; full: cho bài báo (cần thêm GPU).
# Có thể bấm "Run all" nhiều lần: ảnh/độ đo đã có trên Drive sẽ được bỏ qua và chạy tiếp phần còn lại.
# SKIP_JOBS: comma-separated job prefixes to leave out, e.g. "pareto" or "pareto,E3_quickdraw_power" (GPU budget).'''

MOUNT = r'''#@title 2. Gắn Google Drive, kiểm tra GPU, lấy mã nguồn mới nhất
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
CODE = "/content/KLTN_OCSD"
if not REPO_URL:
    # chưa có repo: dùng bản mã nguồn đã tải lên Drive (thư mục DRIVE_ROOT/code)
    CODE = os.path.join(DRIVE_ROOT, "code")
elif os.path.exists(CODE):
    subprocess.run(f"git -C {CODE} fetch -q origin {BRANCH} && git -C {CODE} reset -q --hard origin/{BRANCH}", shell=True, check=True)
else:
    subprocess.run(f"git clone -q -b {BRANCH} {REPO_URL} {CODE}", shell=True, check=True)
print("Mã nguồn:", CODE, subprocess.run(f"git -C {CODE} log -1 --format='%h %s' 2>/dev/null", shell=True, capture_output=True, text=True).stdout)
sys.path.insert(0, CODE)
# drop modules imported by an earlier run in this kernel, so the code just pulled is what runs
for _m in [m for m in sys.modules if m == "ocsd" or m.startswith("ocsd.")]: del sys.modules[_m]
# HF cache on LOCAL disk: Drive cannot store the cache's symlinks (broken files -> "SafetensorError: header too large").
# Models re-download each new session (~2-4 min on Colab).
os.environ["HF_HOME"] = "/content/hf_cache"
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN'''

INSTALL = r'''#@title 3. Install libraries (keeps Colab's PyTorch), start the Drive run log
LOGS = os.path.join(DRIVE_ROOT, "results", "logs"); os.makedirs(LOGS, exist_ok=True)
# Colab ships torchao 0.10, which peft rejects when adding LoRA ("incompatible version of torchao"); OCSD does not use it
subprocess.run("pip uninstall -y -q torchao", shell=True)
r = subprocess.run(f"pip install -q -r {CODE}/requirements-colab.txt", shell=True, capture_output=True, text=True)
open(os.path.join(LOGS, "pip_install.log"), "w").write(r.stdout + "\n" + r.stderr)
print(r.stdout[-2000:], r.stderr[-3000:])
if r.returncode != 0:
    raise RuntimeError(f"pip install failed; see {LOGS}/pip_install.log")
import importlib, ocsd; importlib.reload(ocsd)
from ocsd import hfcache
hfcache.repair(os.environ["HF_HOME"])   # delete broken model files so they download again
from ocsd import runlog
runlog.start(os.path.join(DRIVE_ROOT, "results"), CODE)   # every print below also goes to results/logs/run_*.log
from ocsd.config import Paths, ExperimentConfig, TIERS
P = Paths(DRIVE_ROOT); P.makedirs()
E = ExperimentConfig(backbone=BACKBONE, tier=TIER, skip_jobs=SKIP_JOBS)
E.save(os.path.join(P.results, "experiment_config.json"))
print(E)'''

def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.metadata = {"accelerator": "GPU", "colab": {"gpuType": "L4", "machine_shape": "hm", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("MD:") else nbf.v4.new_code_cell(c) for c in cells]
    path = os.path.join(OUT, name)
    nbf.write(n, path)
    print("wrote", path)


# ============================================================================ RUN_ALL
nb([
    "MD:# OCSD – Chạy toàn bộ thực nghiệm (Run all)\n\n"
    "Khóa luận: *Sinh ảnh cảnh nhất quán đối tượng từ phác thảo và văn bản bằng mô hình khuếch tán* – "
    "Lê Hoàng Bảo (21090331), Thái Bá Cường (21050681); GVHD: TS. Nguyễn Thanh Chuyên.\n\n"
    "**Cách dùng:** Runtime → Change runtime type → GPU (L4 hoặc A100) → sửa ô 1 → Runtime → Run all.\n\n"
    "Notebook chạy 4 giai đoạn: (A) dựng bộ dữ liệu, (B) sinh ảnh cho E1–E4 và khảo sát α, (C) đánh giá, "
    "(D) xuất bảng/hình/tóm tắt. Mọi kết quả nằm trong `DRIVE_ROOT/results` "
    "(`summary.md`, `summary.json`, `tables/`, `figures/`, `progress.json`). "
    "Nếu Colab ngắt kết nối, chỉ cần Run all lại: phần đã xong được bỏ qua.",
    SETUP, MOUNT, INSTALL,
    "MD:## A. Bộ dữ liệu\nQuickDraw-Scenes (tổng hợp, 12 ô: số đối tượng 1/3/5/8+ × độ phức tạp đơn giản/trung bình/phức tạp) "
    "và COCO-Sketch (ảnh thật COCO val2017, phác thảo PiDiNet, mặt nạ thật). Lần đầu mất ~10–20 phút (tải COCO 1 GB).",
    r'''#@title A. Build benchmarks (skipped if already built for this tier)
from ocsd import stages
with runlog.stage("A_data"):
    stages.setup_data(P, TIER, with_coco=True)
from IPython.display import Image, display
f = os.path.join(P.results, "figures", "benchmark_examples.png")
if os.path.exists(f): display(Image(f, width=900))''',
    "MD:## B. Sinh ảnh\nThứ tự: E3 QuickDraw (mọi baseline + OCSD-lite trên toàn bộ tập) → OCSD và Zhang et al. "
    "(cần học định danh, trên tập con phân tầng) → E3 COCO → E4 cắt bỏ → khảo sát α. "
    "`MAX_MINUTES` giới hạn thời gian của ô này; chạy lại để tiếp tục.",
    "MD:## T. Tuning\nPaper/full tiers only, on the pilot scenes, which are then left out of the evaluation: "
    "(1) α × LoRA strength, (2) how M5(b) energy guidance is applied, (3) which of M5(a) / M5(b) to keep, jointly "
    "with α. Writes `results/tuning/tuned.json`; finished phases are skipped (phase 3 alone is about 1 A100 hour).",
    r'''#@title T. Tune OCSD on the pilot scenes (finished phases are skipped)
import torch
eng = vis = None
with runlog.stage("T_tune"):
    eng, vis = stages.tune(P, E, max_minutes=MAX_MINUTES)''',
    r'''#@title B1. Plan and remaining images
for j in stages.experiment_plan(P, E):
    print(f"{j['name']:22s} {len(j['scenes']):4d} scenes x {len(j['methods']):2d} methods x {len(j['seeds'])} seeds -> {stages._count_todo(P, j)} images left")''',
    r'''#@title B2. Generate images (resumable: finished images are skipped)
with runlog.stage("B_generate"):
    eng, vis = stages.generate(P, E, max_minutes=MAX_MINUTES, eng=eng, vis=vis)''',
    "MD:## C. Đánh giá\nOPR, OCE, độ chính xác đếm, mIoU, RA bằng bộ phát hiện **OWLv2** (khác Grounding DINO mà phương pháp "
    "dùng bên trong, để tránh tối ưu hóa theo chính độ đo); CLIP score toàn ảnh và cấp đối tượng; ID-Sim (DINOv2); FID/KID; LPIPS.",
    r'''#@title C. Metrics (only new images are evaluated; FID reused when unchanged)
if eng is not None:
    del eng; torch.cuda.empty_cache()
with runlog.stage("C_evaluate"):
    vis = stages.evaluate(P, E, vis=vis, fid=True)''',
    "MD:## D. Báo cáo",
    r'''#@title D. Tables, figures, statistics, summary
with runlog.stage("D_report"):
    summary = stages.report(P, E)
    from IPython.display import Markdown, display
    display(Markdown(open(os.path.join(P.results, "summary.md")).read()))
    for f in summary["figures"]:
        display(Image(f, width=900))''',
    r'''#@title E. Run status (also saved to results/logs/status.json; errors in results/logs/errors/)
runlog.summary()
if os.path.exists(os.path.join(LOGS, "LATEST_ERROR.txt")):
    print("\nMost recent error:\n" + open(os.path.join(LOGS, "LATEST_ERROR.txt")).read()[-3000:])''',
    r'''#@title F. Send logs and results to GitHub (branch colab-results) so Claude can read them
from ocsd import sync
try:
    sync.push_results(P.results, sync.colab_token())
except Exception as e:
    print("[sync] failed:", e)''',
], "RUN_ALL.ipynb")

# ============================================================================ push logs only
nb([
    "MD:# Send the latest logs/results from Drive to GitHub (no GPU needed)\n\n"
    "Use this when a run already finished (or crashed) and Claude needs to read the logs. "
    "Requires the Colab secret `GH_TOKEN` (key icon on the left).",
    SETUP,
    r'''import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
CODE = "/content/KLTN_OCSD"
if os.path.exists(CODE):
    subprocess.run(f"git -C {CODE} fetch -q origin {BRANCH} && git -C {CODE} reset -q --hard origin/{BRANCH}", shell=True, check=True)
else:
    subprocess.run(f"git clone -q -b {BRANCH} {REPO_URL} {CODE}", shell=True, check=True)
sys.path.insert(0, CODE)
from ocsd import sync
sync.push_results(os.path.join(DRIVE_ROOT, "results"), sync.colab_token())''',
], "00_push_logs.ipynb")

# ============================================================================ demo từng mô-đun
nb([
    "MD:# OCSD – Minh họa từng mô-đun trên một cảnh\n\nDùng để tạo hình cho Chương 3–4: phác thảo và thông tin đối tượng (M1), "
    "ứng viên đối tượng (M2), học định danh (M3), ảnh tiền cảnh và quá trình trộn (M4), bản đồ chú ý và kết quả có/không M5, "
    "dãy α. Chạy sau khi RUN_ALL đã dựng bộ dữ liệu.",
    SETUP, MOUNT, INSTALL,
    r'''#@title Chọn cảnh
from ocsd.data import list_scenes, load_scene
SCENE_ID = ""  #@param {type:"string"}
dirs = list_scenes(P.benchmarks, "quickdraw", count_bins=["3"], complexities=["medium"])
sdir = next((d for d in dirs if d.endswith(SCENE_ID)), dirs[0]) if SCENE_ID else dirs[0]
scene = load_scene(sdir); print(scene.sid, "|", scene.caption); print(scene.relations)
FIG = os.path.join(P.results, "figures", "demo_" + scene.sid); os.makedirs(FIG, exist_ok=True)''',
    r'''#@title M1: phác thảo, mặt nạ, hộp, quan hệ
import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(scene.sketch, cmap="gray"); ax[0].set_title("Phác thảo cảnh")
col = plt.cm.tab10(np.arange(scene.n) % 10)
ov = np.ones(scene.sketch.shape + (3,))
for i, o in enumerate(scene.objects): ov[o.mask] = col[i][:3]
ax[1].imshow(ov); ax[1].imshow(scene.sketch, cmap="gray", alpha=0.5); ax[1].set_title("Mặt nạ m_i")
ax[2].imshow(scene.sketch, cmap="gray")
for i, o in enumerate(scene.objects):
    x0, y0, x1, y1 = o.box
    ax[2].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color=col[i], lw=2))
    ax[2].text(x0, y0-4, f"{i}: {o.phrase}", color=col[i], fontsize=9)
ax[2].set_title("Hộp b_i: " + "; ".join(f"{i}{r}{j}" for i, j, r in scene.relations[:6]), fontsize=8)
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/m1.png", dpi=200); plt.show()''',
    r'''#@title Nạp mô hình
import torch
from ocsd.engine import Engine
from ocsd.vision import Vision
from ocsd.config import OCSDConfig
eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16)
vis = Vision("cuda")
cfg = OCSDConfig()''',
    r'''#@title M2: ứng viên và ứng viên được chọn
from ocsd.method import object_branch
objs = object_branch(eng, vis, scene, cfg, seed=0)
fig, ax = plt.subplots(2, scene.n, figsize=(3.2*scene.n, 6.4), squeeze=False)
for i, r in enumerate(objs):
    ax[0, i].imshow(r.sketch_crop, cmap="gray"); ax[0, i].set_title(scene.objects[i].phrase)
    shown = r.img.copy(); shown[~r.mask] = (0.35*shown[~r.mask] + 165).astype(np.uint8)
    ax[1, i].imshow(shown); ax[1, i].set_title(f"điểm chọn {r.score:.2f}")
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/m2.png", dpi=200); plt.show()''',
    r'''#@title M3 + M4: học định danh, ảnh tiền cảnh, dựng cảnh (có / không M5)
from ocsd.method import compose_foreground, learn_identity, Prepared, generate
fg = compose_foreground(scene, objs, cfg)
log = {}
tokens = learn_identity(eng, scene, objs, cfg, seed=0, log=log); print(log)
prep = Prepared(objs, fg, tokens)
from ocsd.methods import OCSD_VARIANTS, run_baseline
res = {"ControlNet": run_baseline("controlnet", eng, vis, scene, 0),
       "OCSD không M5": generate(eng, vis, scene, prep, OCSD_VARIANTS["abl_no_m5"], 0),
       "OCSD": generate(eng, vis, scene, prep, OCSD_VARIANTS["ocsd"], 0)}
fig, ax = plt.subplots(1, 2 + len(res), figsize=(4*(2+len(res)), 4))
ax[0].imshow(scene.sketch, cmap="gray"); ax[0].set_title("Phác thảo")
ax[1].imshow(fg[0]); ax[1].set_title("Ảnh tiền cảnh x_fg (M4)")
for a, (k, v) in zip(ax[2:], res.items()): a.imshow(v); a.set_title(k)
for a in ax: a.axis("off")
plt.suptitle(scene.caption); plt.tight_layout(); plt.savefig(f"{FIG}/m4_m5.png", dpi=200); plt.show()''',
    r'''#@title Bản đồ chú ý chéo của token định danh (có / không chú ý giới hạn theo vùng)
from ocsd.attention import token_maps
from ocsd.sketch import scene_global_prompt
from ocsd.method import _lat_mask
gp = scene_global_prompt(eng.tokenizer, scene, tokens)
emb = eng.encode([gp.text]); eot = eng.eot_index(gp.text)
ts = eng.set_timesteps(cfg.steps); t = ts[int(len(ts)*(1-cfg.alpha))]
z = eng.q_sample(eng.to_latent(fg[0]), eng.randn(0), t)
lh = cfg.height // 8
om = torch.stack([_lat_mask(o.mask, lh, lh, eng.device) for o in scene.objects])
eng.lora(True)
fig, ax = plt.subplots(2, scene.n, figsize=(3*scene.n, 6), squeeze=False)
for row, bias in enumerate([False, True]):
    eng.ctrl.set_regions(om, [gp.groups.get(f"obj{i}", []) for i in range(scene.n)], gp.groups.get("bg", []))
    eng.ctrl.bias_enabled, eng.ctrl.lambda_t = bias, cfg.lambda0
    eng.ctrl.store, eng.ctrl.store_res = True, (lh // 4) ** 2; eng.ctrl.reset_maps()
    with torch.no_grad(): eng.unet_eps(z, t, emb, cond_flags=[True])
    maps = token_maps(eng.ctrl.aggregated_map(0), [gp.groups[f"id{i}"] for i in range(scene.n)], eot).cpu()
    eng.ctrl.store = False; eng.ctrl.clear_regions()
    for i in range(scene.n):
        ax[row, i].imshow(maps[i], cmap="jet"); ax[row, i].set_title(("có" if bias else "không") + f" M5(a): {scene.objects[i].cls}", fontsize=8); ax[row, i].axis("off")
eng.lora(False)
plt.tight_layout(); plt.savefig(f"{FIG}/attn.png", dpi=200); plt.show()''',
    r'''#@title Dãy alpha
alphas = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
fig, ax = plt.subplots(1, len(alphas), figsize=(3.2*len(alphas), 3.4))
for a, al in zip(ax, alphas):
    a.imshow(generate(eng, vis, scene, prep, OCSD_VARIANTS["ocsd"].replace(alpha=al, use_verify=False), 0)); a.set_title(f"α = {al}"); a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/alpha.png", dpi=200); plt.show()
eng.reset_identity()''',
    r'''#@title Nhất quán định danh khi đổi nền (dùng lại M1–M3)
import copy
bgs = ["on a city street", "in a green meadow", "on a sandy beach", "on a snowy field"]
tokens = learn_identity(eng, scene, objs, cfg, seed=0); prep = Prepared(objs, fg, tokens)
fig, ax = plt.subplots(1, len(bgs), figsize=(4*len(bgs), 4))
for a, bg in zip(ax, bgs):
    sc2 = copy.copy(scene); sc2.bg = bg
    a.imshow(generate(eng, vis, sc2, prep, OCSD_VARIANTS["ocsd"], 0)); a.set_title(bg); a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/identity_bg.png", dpi=200); plt.show()
eng.reset_identity()''',
], "01_demo_modules.ipynb")

# ============================================================================ ứng dụng minh họa
nb([
    "MD:# OCSD – Ứng dụng minh họa (Gradio)\n\nVẽ phác thảo (hoặc tải ảnh), nhập mô tả từng đối tượng theo thứ tự trái → phải "
    "và mô tả nền; hệ thống tự phân tách đối tượng (M1) rồi sinh ảnh bằng OCSD-lite (không cần huấn luyện, nhanh) hoặc "
    "OCSD đầy đủ. Bảng kiểm tra cho biết số đối tượng phát hiện được.",
    SETUP, MOUNT, INSTALL,
    r'''#@title Khởi chạy ứng dụng
!pip install -q gradio
from ocsd.app import launch
launch(backbone=E.backbone, share=True)''',
], "02_app_gradio.ipynb")
