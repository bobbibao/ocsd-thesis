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
TIER       = "pilot"   #@param ["foreground_screen", "clean_screen", "count_screen", "instance_screen", "semantic_screen", "repair_screen", "screen", "pilot", "paper", "full"]
BACKBONE   = "sd15"    #@param ["sd15", "sd21"]
MAX_MINUTES = 600      #@param {type:"number"}
SKIP_JOBS  = ""        #@param {type:"string"}
RUN_ID     = ""        #@param {type:"string"}
PUSH_RESULTS = True    #@param {type:"boolean"}
# pilot: kiểm tra nhanh (~1-2 giờ GPU); paper: đủ số liệu cho khóa luận; full: cho bài báo (cần thêm GPU).
# Có thể bấm "Run all" nhiều lần: ảnh/độ đo đã có trên Drive sẽ được bỏ qua và chạy tiếp phần còn lại.
# SKIP_JOBS: comma-separated job prefixes to leave out, e.g. "pareto" or "pareto,E3_quickdraw_power" (GPU budget).'''

MOUNT = r'''#@title 2. Gắn Google Drive, kiểm tra GPU, chọn mã nguồn
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True)
if gpu.returncode:
    raise RuntimeError("Select a GPU runtime before Run all")
print(gpu.stdout)
CODE = "/content/KLTN_OCSD"
if not REPO_URL:
    # chưa có repo: dùng bản mã nguồn đã tải lên Drive (thư mục DRIVE_ROOT/code)
    CODE = os.path.join(DRIVE_ROOT, "code")
elif os.path.exists(CODE):
    if not RUN_ID:
        dirty = subprocess.run(["git", "-C", CODE, "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        if dirty.strip():
            raise RuntimeError("Checkout has local changes. Preserve them; use a fresh runtime for current GitHub code, or set RUN_ID to resume.")
        subprocess.run(["git", "-C", CODE, "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", CODE, "merge", "--ff-only", "FETCH_HEAD"], check=True)
    print("Using checkout (explicit RUN_ID preserves its version):", CODE)
else:
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, CODE], check=True)
print("Mã nguồn:", CODE, subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
sys.path.insert(0, CODE)
# drop modules imported by an earlier run in this kernel, so the selected code is what runs
for _m in [m for m in sys.modules if m == "ocsd" or m.startswith("ocsd.")]: del sys.modules[_m]
# HF cache on LOCAL disk: Drive cannot store the cache's symlinks (broken files -> "SafetensorError: header too large").
# Models re-download each new session (~2-4 min on Colab).
os.environ["HF_HOME"] = "/content/hf_cache"
# Public models do not require a token; optional HF authentication can use Colab secrets.'''

INSTALL = r'''#@title 3. Install libraries (keeps Colab's PyTorch), start the Drive run log
LOGS = os.path.join(DRIVE_ROOT, "results", "logs"); os.makedirs(LOGS, exist_ok=True)
# Colab ships torchao 0.10, which peft rejects when adding LoRA ("incompatible version of torchao"); OCSD does not use it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(CODE, "requirements-colab.txt")], capture_output=True, text=True)
open(os.path.join(LOGS, "pip_install.log"), "w").write(r.stdout + "\n" + r.stderr)
print(r.stdout[-2000:], r.stderr[-3000:])
if r.returncode != 0:
    raise RuntimeError(f"pip install failed; see {LOGS}/pip_install.log")
import importlib, ocsd; importlib.reload(ocsd)
from ocsd import hfcache
hfcache.repair(os.environ["HF_HOME"])   # delete broken model files so they download again
from ocsd import runlog
from ocsd.config import Paths, ExperimentConfig, TIERS
EXP_ROOT = DRIVE_ROOT
if globals().get("RUN_MODE") in ("heldout", "heldout2", "heldout3"):
    # Locked held-out comparison: own data root, fixed tier and generation budget
    # (docs/HELDOUT_PROTOCOL.md, HELDOUT2_PROTOCOL.md, HELDOUT3_PROTOCOL.md)
    HELDOUT_SET = {"heldout": "v1", "heldout2": "v2", "heldout3": "v3"}[RUN_MODE]
    os.environ["OCSD_HELDOUT"] = HELDOUT_SET
    from ocsd import heldout as _heldout; _cfg = _heldout.use(HELDOUT_SET)
    EXP_ROOT, TIER = os.path.join(DRIVE_ROOT, "heldout_" + HELDOUT_SET), _cfg["tier"]
    MAX_MINUTES = 300 if HELDOUT_SET == "v3" else 180
P = Paths(EXP_ROOT); P.makedirs()
E = ExperimentConfig(backbone=BACKBONE, tier=TIER, skip_jobs=SKIP_JOBS,
                     extra_detectors=() if TIER in ("screen", "repair_screen", "semantic_screen", "instance_screen", "count_screen", "clean_screen", "foreground_screen", "heldout_screen", "heldout_final") else ("owlv2", "owlv2d", "detr"))
from ocsd.provenance import start_run
P, RUN = start_run(P, E, run_id=RUN_ID)
RUN_ID = P.run_id
runlog.start(P.results, CODE)
LOGS = os.path.join(P.results, "logs")
E.save(os.path.join(P.results, "experiment_config.json"))
print("RUN_ID:", RUN_ID, "results:", P.results)
print(E)'''

DIRECT_MOUNT = MOUNT.replace(
    '#@title 2. Gắn Google Drive, kiểm tra GPU, chọn mã nguồn',
    '#@title 2. Gắn Drive, kiểm tra loại runtime, lấy mã nguồn mới'
).replace(
    'gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True)\n'
    'if gpu.returncode:\n    raise RuntimeError("Select a GPU runtime before Run all")\nprint(gpu.stdout)',
    '''import shutil
if RUN_MODE not in ("direct_preflight", "direct_canary", "direct_dev", "silhouette_dev", "naturalness_dev", "heldout", "heldout2", "heldout3", "experiment"):
    raise ValueError("Unknown RUN_MODE")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
if RUN_MODE == "direct_preflight":
    if RUN_ID:
        raise ValueError("CPU preflight uses a separate log: leave RUN_ID empty")
    if gpu is not None and gpu.returncode == 0:
        raise RuntimeError("Bước này dùng CPU: Runtime > Change runtime type > Hardware accelerator: None, rồi Run all")
    print("CPU preflight: kiểm tra trọng số thật; không sinh ảnh, không cần GH_TOKEN.")
else:
    if RUN_MODE in ("direct_canary", "direct_dev", "silhouette_dev", "naturalness_dev") and RUN_ID:
        raise ValueError("Direct runs resume compatible images automatically: leave RUN_ID empty")
    if gpu is None or gpu.returncode:
        raise RuntimeError("Select an L4/A100 GPU runtime for this mode")
    print(gpu.stdout)''')

DIRECT_PREFLIGHT = r'''#@title 3. Kiểm tra mô hình trực tiếp trên CPU và tải ZIP log
if RUN_MODE == "direct_preflight":
    import json, uuid
    from pathlib import Path
    from zipfile import ZipFile, ZIP_DEFLATED
    from google.colab import files
    DIRECT_OUT = Path(DRIVE_ROOT) / "results" / "direct_preflight" / ("attempt_" + uuid.uuid4().hex[:12])
    DIRECT_OUT.mkdir(parents=True)
    setup_log = DIRECT_OUT / "setup.log"
    print("Cài thư viện; dùng lại cache nếu còn. Tải mới: khoảng 13,6 GB checkpoint + 1,71 GB CLIP, chưa gồm tệp tạm.", flush=True)
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                               os.path.join(CODE, "requirements-direct-preflight.txt")], capture_output=True, text=True)
    setup_log.write_text(installed.stdout + "\n" + installed.stderr)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    checked = subprocess.run([sys.executable, os.path.join(CODE, "tools", "preflight_direct.py"),
                              "--cache", "/content/ocsd-direct-cache", "--out", str(DIRECT_OUT),
                              "--setup-log", str(setup_log), "--setup-returncode", str(installed.returncode)], env=env)
    latest = DIRECT_OUT / "latest_preflight.json"
    if latest.exists():
        result = json.loads(latest.read_text())
        archive = result["archive"]
        print("Kết quả kiểm tra:", result["status"])
    else:
        # Preserve even an OS kill/import crash; never download an older successful run.
        archive = str(DIRECT_OUT / "ocsd_direct_preflight_incomplete.zip")
        (DIRECT_OUT / "incomplete.json").write_text(json.dumps(dict(status="incomplete", returncode=checked.returncode,
            message="Process stopped before the report completed. Review logs/RAM; no GPU run is recommended."), indent=2))
        with ZipFile(archive, "w", ZIP_DEFLATED) as z:
            for p in DIRECT_OUT.rglob("*"):
                if p.is_file() and p.suffix in (".log", ".json"):
                    z.write(p, str(p.relative_to(DIRECT_OUT)))
        print("Kiểm tra chưa hoàn tất; đã giữ log để tìm lỗi.")
    print("Gửi file ZIP này vào cuộc trò chuyện:", archive)
    print("PASS chỉ xác nhận tương thích CPU. Chưa chứng minh ảnh đẹp hơn; chưa chuyển sang GPU/paper/full.")
    try:
        files.download(archive)
    except Exception as exc:
        print("Không tự tải được; lấy file theo đường dẫn Drive ở trên:", exc)
else:
    print("Bỏ qua CPU preflight trong chế độ này.")'''

DIRECT_CANARY = r'''#@title 3b. Thử GPU tối đa 4 ảnh và tải PNG so sánh + ZIP
if RUN_MODE == "direct_canary":
    import json, uuid
    from pathlib import Path
    from google.colab import files
    CANARY_OUT = Path(DRIVE_ROOT) / "results" / "direct_canary"
    attempt = CANARY_OUT / ("attempt_" + uuid.uuid4().hex[:12]); attempt.mkdir(parents=True)
    print("Canary: 1 cảnh mới × 2 seed × 2 nhánh = 4 ảnh; tối đa 600 giây cho tiến trình sau cài thư viện.", flush=True)
    print("Tải mới khoảng 13,6 GB + 1,71 GB. Chỉ sinh phần chưa hoàn tất; không tự thử lại nếu lỗi.", flush=True)
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                               os.path.join(CODE, "requirements-direct-preflight.txt")], capture_output=True, text=True)
    (attempt / "setup.log").write_text(installed.stdout + "\n" + installed.stderr)
    script = os.path.join(CODE, "tools", "run_direct_canary.py")
    interrupted = False
    try:
        checked = subprocess.run([sys.executable, script, "--cache", "/content/ocsd-direct-cache",
            "--out", str(CANARY_OUT), "--attempt", str(attempt), "--setup-returncode", str(installed.returncode)],
            env=dict(os.environ, PYTHONUNBUFFERED="1"), timeout=600)
        interrupted = not (attempt / "result.json").exists()
    except subprocess.TimeoutExpired:
        interrupted = True
        print("Đã dừng tại giới hạn 600 giây; giữ lại ảnh đã hoàn tất và log.")
    if interrupted:
        subprocess.run([sys.executable, script, "--attempt", str(attempt), "--finalize-only", "--interrupted"], timeout=60)
    result_path = attempt / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"Không đóng gói được; giữ thư mục này để kiểm tra: {attempt}")
    result = json.loads(result_path.read_text())
    print("Trạng thái:", result["status"], "| ZIP:", result["archive"])
    if result["status"] == "complete":
        from IPython.display import Image, display
        for seed in (0, 1):
            display(Image(filename=str(Path(result["directory"]) / f"SO_SANH_seed{seed}.png"), width=1152))
        print("Đọc run/HUONG_DAN.txt; điền run/REVIEW_TEMPLATE.txt hoặc trả lời trong chat. Complete chưa phải chất lượng đạt.")
    else:
        print("Gửi ZIP để kiểm tra lỗi/ngân sách; chưa tự tăng số ảnh, số bước hoặc chuyển paper/full.")
    print("Lượt GPU đã dừng. Tải ZIP rồi ngắt runtime GPU khi không dùng để tránh thời gian nhàn rỗi.")
    try:
        files.download(result["archive"])
    except Exception as exc:
        print("Lấy ZIP theo đường dẫn Drive ở trên:", exc)
else:
    print("Bỏ qua canary GPU trong chế độ này.")'''


DIRECT_DEV = r'''#@title 3c. Development study (8 dev scenes x 2 seeds), then download ZIPs
if RUN_MODE in ("direct_dev", "silhouette_dev", "naturalness_dev"):
    import json, uuid
    from pathlib import Path
    from google.colab import files
    DEV_VERSION = {"direct_dev": "v1", "silhouette_dev": "v2", "naturalness_dev": "v3"}[RUN_MODE]
    DEV_OUT = Path(DRIVE_ROOT) / "results" / {"v1": "direct_dev", "v2": "direct_dev2", "v3": "direct_dev3"}[DEV_VERSION]
    DEV_SECONDS = 3000 if DEV_VERSION == "v3" else 2400
    attempt = DEV_OUT / ("attempt_" + uuid.uuid4().hex[:12]); attempt.mkdir(parents=True)
    if DEV_VERSION == "v3":
        print("Locked naturalness study: 32 InstanceDiffusion images + 32 SDXL-Refiner images + SAM sketch fidelity,", flush=True)
        print("at most 3000 s. Extra downloads: SDXL refiner ~6 GB, SAM ViT-H ~2.6 GB.", flush=True)
    else:
        print("Locked development study: 48 images, at most 2400 s after library install. Completed images are reused.", flush=True)
    print("May download ~13.6 GB checkpoint + 1.71 GB CLIP if the runtime cache is empty. No automatic retries.", flush=True)
    reqs = ["-r", os.path.join(CODE, "requirements-direct-preflight.txt")]
    if DEV_VERSION == "v3": reqs += ["-r", os.path.join(CODE, "requirements-direct-refiner.txt")]
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], capture_output=True, text=True)
    (attempt / "setup.log").write_text(installed.stdout + "\n" + installed.stderr)
    script = os.path.join(CODE, "tools", "run_direct_dev.py")
    interrupted = False
    try:
        subprocess.run([sys.executable, script, "--cache", "/content/ocsd-direct-cache",
            "--out", str(DEV_OUT), "--attempt", str(attempt), "--setup-returncode", str(installed.returncode), "--version", DEV_VERSION],
            env=dict(os.environ, PYTHONUNBUFFERED="1"), timeout=DEV_SECONDS)
        interrupted = not (attempt / "result.json").exists()
    except subprocess.TimeoutExpired:
        interrupted = True
        print(f"Stopped at the {DEV_SECONDS} s limit; completed images and logs are kept.")
    if interrupted:
        subprocess.run([sys.executable, script, "--attempt", str(attempt), "--finalize-only", "--interrupted"], timeout=120)
    result_path = attempt / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"Could not package; keep this folder for inspection: {attempt}")
    result = json.loads(result_path.read_text())
    print("Status:", result["status"], "| full ZIP:", result["archive"])
    downloads = [result["archive"]]
    if result["status"] == "complete":
        print("Review ZIP (16 PNG sheets + HOW_TO_REVIEW.txt + REVIEW_TEMPLATE.txt):", result["review_archive"])
        downloads.append(result["review_archive"])
        print("'complete' means all images exist; it is not a quality pass.")
    else:
        print("Send the ZIP so the failure can be checked; do not change settings or rerun with more images.")
    print("GPU work finished. Disconnect the GPU runtime after downloading to avoid idle compute units.")
    for path in downloads:
        try:
            files.download(path)
        except Exception as exc:
            print("Get it from the Drive path above:", path, exc)
else:
    print("Skipping the development study in this mode.")'''


DIRECT_HELDOUT = r'''#@title 3d. Held-out comparison, part 1: InstanceDiffusion-core arms (72 images), then the other methods below
if RUN_MODE in ("heldout", "heldout2", "heldout3"):
    import json, uuid
    from pathlib import Path
    HELDOUT_SET = {"heldout": "v1", "heldout2": "v2", "heldout3": "v3"}[RUN_MODE]
    os.environ["OCSD_HELDOUT"] = HELDOUT_SET
    HELDOUT_DIRECT = Path(DRIVE_ROOT) / ("heldout_" + HELDOUT_SET) / "direct"
    attempt = HELDOUT_DIRECT / ("attempt_" + uuid.uuid4().hex[:12]); attempt.mkdir(parents=True)
    arms = {"v1": "(box, scribble)", "v2": "(box, silhouette)", "v3": "(InstanceDiffusion box, OCSD v3)"}[HELDOUT_SET]
    print(f"Locked held-out {HELDOUT_SET}: 18 scenes x 2 seeds x {arms} = 72 images, at most 3600 s. Completed images are reused.", flush=True)
    reqs = ["-r", os.path.join(CODE, "requirements-direct-preflight.txt")]
    if HELDOUT_SET == "v3": reqs += ["-r", os.path.join(CODE, "requirements-direct-refiner.txt")]
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], capture_output=True, text=True)
    (attempt / "setup.log").write_text(installed.stdout + "\n" + installed.stderr)
    script = os.path.join(CODE, "tools", "run_direct_heldout.py")
    interrupted = False
    try:
        subprocess.run([sys.executable, script, "--cache", "/content/ocsd-direct-cache", "--out", str(HELDOUT_DIRECT),
            "--attempt", str(attempt), "--setup-returncode", str(installed.returncode)],
            env=dict(os.environ, PYTHONUNBUFFERED="1"), timeout=3600)
        interrupted = not (attempt / "result.json").exists()
    except subprocess.TimeoutExpired:
        interrupted = True
    if interrupted:
        subprocess.run([sys.executable, script, "--attempt", str(attempt), "--finalize-only", "--interrupted"], timeout=120)
    result = json.loads((attempt / "result.json").read_text())
    print("Direct arms:", result["status"], "| log ZIP:", result["archive"])
    if result["status"] != "complete":
        try:
            from google.colab import files; files.download(result["archive"])
        except Exception as exc:
            print("Get the log ZIP from Drive:", exc)
        raise RuntimeError("Direct arms incomplete. Run all again to resume (completed images are kept), or send the log ZIP.")
    import gc; gc.collect()
    print("Direct arms complete. Continuing with the other methods (resumable).")
else:
    print("Skipping the held-out comparison in this mode.")'''

def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.metadata = {"accelerator": "GPU", "colab": {"gpuType": "L4", "machine_shape": "hm", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
    if name in ("00_push_logs.ipynb", "03_export_screen_review.ipynb"):
        n.metadata.pop('accelerator', None)
        n.metadata['colab'] = {'provenance': []}
    if name == "RUN_ALL.ipynb":
        # Keep the user's single entry point. No legacy stage may run in preflight mode,
        # even when its dependency check fails and Colab continues to later cells.
        guarded, experiment = [], False
        for c in cells:
            if c == INSTALL:
                experiment = True
            if experiment and not c.startswith("MD:"):
                title, body = c.split('\n', 1)
                c = title + '\nif RUN_MODE in ("experiment", "heldout", "heldout2", "heldout3"):\n' + '\n'.join('    ' + line for line in body.splitlines())
            guarded.append(c)
        cells = guarded
    cells = [c.replace("start_run(P, E, run_id=RUN_ID)", f'start_run(P, E, run_id=RUN_ID, notebook="notebooks/{name}")') for c in cells]
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("MD:") else nbf.v4.new_code_cell(c) for c in cells]
    path = os.path.join(OUT, name)
    import hashlib
    old = nbf.read(path, as_version=4) if os.path.exists(path) else None
    if old:
        by_title = {c.source.split("\n")[0]: c.id for c in old.cells if c.get("id")}
        for c in n.cells:
            c.id = by_title.get(c.source.split("\n")[0], hashlib.sha256(c.source.encode()).hexdigest()[:12])
    nbf.write(n, path)
    print("wrote", path)


# ============================================================================ RUN_ALL
nb([
    "MD:# OCSD – Chạy toàn bộ thực nghiệm (Run all)\n\n"
    "Khóa luận: *Sinh ảnh cảnh nhất quán đối tượng từ phác thảo và văn bản bằng mô hình khuếch tán* – "
    "Lê Hoàng Bảo (21090331), Thái Bá Cường (21050681); GVHD: TS. Nguyễn Thanh Chuyên.\n\n"
    "**Cập nhật 03/11/2026:** đã nhận đủ `foreground_screen` của run `20261103T035056Z_dc40e4b5`. "
    "Đã nhận review: A–D đủ số lượng/hình dáng; A/B chính xác nhất nhưng chưa tự nhiên. **Không cần chấm lại hay chạy lại phép thử foreground.** "
    "[Xem A–D bằng PNG, không cần HTML](https://raw.githubusercontent.com/bobbibao/ocsd-thesis/main/docs/reviews/foreground_20261103_ABCD.png). "
    "Hướng mới: điều kiện hóa trực tiếp từng đối tượng bằng nét phác thảo với lõi InstanceDiffusion chính thức. "
    "Đây là ứng viên đang kiểm chứng, chưa có kết quả chứng minh chất lượng ảnh tốt hơn.\n\n"
    "**Update 2026-11-04:** the canary and the 8-scene development study are done and reviewed; do not rerun them "
    "([dev results](https://github.com/bobbibao/ocsd-thesis/blob/main/docs/DIRECT_DEV_20261104_RESULTS.md)). "
    "Held-out v1, the silhouette dev study and held-out v2 are done; do not rerun them "
    "([held-out v2 results](https://github.com/bobbibao/ocsd-thesis/blob/main/docs/HELDOUT2_20261104_RESULTS.md)).\n\n"
    "**Next step (2026-11-04):** Runtime → Change runtime type → **GPU L4 or A100** → keep `RUN_MODE=\"naturalness_dev\"`, `RUN_ID=\"\"` → **Run all**. "
    "This is the locked naturalness dev study for the silhouette model on the 8 dev scenes × 2 seeds: silhouette on all steps vs "
    "layout steps only, each without/with the upstream SDXL refiner = 64 images, then SAM sketch fidelity. "
    "Expect about 30–45 min; it resumes if stopped. Download the two ZIPs at the end (or leave them on Drive in `results/direct_dev3`). "
    "No manual review needed; Claude reviews it. "
    "[Protocol](https://github.com/bobbibao/ocsd-thesis/blob/main/docs/DIRECT_DEV3_PROTOCOL.md).\n\n"
    "Chỉ khi chủ động tái lập thí nghiệm cũ: chọn `RUN_MODE=\"experiment\"` và GPU L4/A100. "
    "Luồng này chạy 4 giai đoạn: (A) dựng bộ dữ liệu, (B) sinh ảnh cho E1–E4 và khảo sát α, (C) đánh giá, "
    "(D) xuất bảng/hình/tóm tắt. Mọi kết quả nằm trong `DRIVE_ROOT/results/runs/<run_id>/` "
    "(`summary.md`, `summary.json`, `tables/`, `figures/`, `progress.json`). "
    "Lưu RUN_ID để resume. Chỉ tái sử dụng kết quả cùng code, cấu hình, thư viện và dữ liệu. Xem docs/EXPERIMENT_RUNBOOK.md.",
    SETUP.replace('REPO_URL   =', 'RUN_MODE   = "naturalness_dev"  #@param ["naturalness_dev", "heldout3", "heldout2", "silhouette_dev", "heldout", "direct_dev", "direct_canary", "direct_preflight", "experiment"]\nREPO_URL   =').replace('TIER       = "pilot"', 'TIER       = "foreground_screen"').replace('MAX_MINUTES = 600', 'MAX_MINUTES = 20'),
    DIRECT_MOUNT, DIRECT_PREFLIGHT, DIRECT_CANARY, DIRECT_DEV, DIRECT_HELDOUT, INSTALL,
    "MD:## Các ô dưới chỉ chạy khi RUN_MODE=experiment\nPhép thử cũ `foreground_screen`: 1 cảnh xe đạp × 4 đối chứng "
    "= 4 ảnh cuối. So sánh clean cũ, anchor có nền cảnh, mask học được dạng cứng và dạng mềm. "
    "BiRefNet-lite chạy CPU, tải khoảng 224 MB một lần và kiểm tra SHA-256. "
    "Không gửi ảnh ra dịch vụ ngoài. Cùng mẫu M2, vị trí, vùng sửa và số bước diffusion; "
    "chi phí tách nền ghi riêng. Đã chạy và nhận review: người dùng ưu tiên hai nhánh tách nền, count/shape đạt trên cảnh này, nhưng chưa tự nhiên. "
    "Xem docs/FOREGROUND_20261103_RESULTS.md. Không chạy lại sweep này hoặc paper/full theo kết quả hiện tại. "
    "Chỉ dùng RUN_ID mới nếu chủ động tái lập phép thử; đây chưa phải model đạt chất lượng.",
    "MD:## A. Bộ dữ liệu\nQuickDraw-Scenes (tổng hợp, 12 ô: số đối tượng 1/3/5/8+ × độ phức tạp đơn giản/trung bình/phức tạp) "
    "và COCO-Sketch (ảnh thật COCO val2017, phác thảo PiDiNet, mặt nạ thật). Lần đầu mất ~10–20 phút (tải COCO 1 GB).",
    r'''#@title A. Build benchmarks (skipped if already built for this tier)
from ocsd import stages
with RUN.stage("A_data"):
    stages.setup_data(P, TIER, with_coco=TIER not in ("screen", "repair_screen", "semantic_screen", "instance_screen", "count_screen", "clean_screen", "foreground_screen", "heldout_screen", "heldout_final"))
    RUN.seal_data()
    RUN.plan(E)
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
with RUN.stage("T_tune"):
    eng, vis = stages.tune(P, E, max_minutes=MAX_MINUTES)''',
    r'''#@title B1. Plan and remaining images
for j in stages.experiment_plan(P, E):
    print(f"{j['name']:22s} {len(j['scenes']):4d} scenes x {len(j['methods']):2d} methods x {len(j['seeds'])} seeds -> {stages._count_todo(P, j)} images left")''',
    r'''#@title B2. Generate images (resumable: finished images are skipped)
with RUN.stage("B_generate"):
    eng, vis = stages.generate(P, E, max_minutes=MAX_MINUTES, eng=eng, vis=vis)''',
    "MD:## C. Đánh giá\nOPR, OCE, độ chính xác đếm, mIoU, RA bằng bộ phát hiện **OWLv2** (khác Grounding DINO mà phương pháp "
    "dùng bên trong, để tránh tối ưu hóa theo chính độ đo); CLIP score toàn ảnh và cấp đối tượng; ID-Sim (DINOv2); FID/KID; LPIPS.",
    r'''#@title C. Metrics (only new images are evaluated; FID reused when unchanged)
if eng is not None:
    del eng
from ocsd.methods import _GLIGEN
_GLIGEN.clear()
import gc
gc.collect(); torch.cuda.empty_cache()
with RUN.stage("C_evaluate"):
    vis = stages.evaluate(P, E, vis=vis, fid=TIER not in ("screen", "repair_screen", "semantic_screen", "instance_screen", "count_screen", "clean_screen", "foreground_screen", "heldout_screen", "heldout_final"))''',
    "MD:## D. Báo cáo",
    r'''#@title D. Tables, figures, statistics, summary
with RUN.stage("D_report"):
    summary = stages.report(P, E)
RUN.finish(E)  # checksum only after the stage has written its final status
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
if PUSH_RESULTS:
    sync.try_push_results(P.results, sync.colab_token())
else:
    print("Results retained on Drive:", P.results)''',
    r'''#@title G. Held-out mode: download the results ZIP and the blind review ZIP
if RUN_MODE in ("heldout", "heldout2", "heldout3"):
    from google.colab import files
    tag = RUN_MODE
    for name in (f"ocsd_{tag}_results.zip", f"REVIEW_{tag}.zip"):
        path = os.path.join(P.results, name)
        print(path)
        try:
            files.download(path)
        except Exception as exc:
            print("Get it from Drive:", exc)
    print("GPU work finished. Disconnect the GPU runtime after downloading.")''',
], "RUN_ALL.ipynb")

# ============================================================================ push logs only
nb([
    "MD:# Send the latest logs/results from Drive to GitHub (no GPU needed)\n\n"
    "Use this when a run already finished (or crashed) and Claude needs to read the logs. "
    "Select RUN_ID (blank = latest), set PUSH_RESULTS=True and provide the Colab secret `GH_TOKEN` "
    "(key icon on the left) to upload. REPO_URL blank uses DRIVE_ROOT/code.",
    SETUP,
    r'''import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
CODE = "/content/KLTN_OCSD"
if not REPO_URL:
    CODE = os.path.join(DRIVE_ROOT, "code")
elif os.path.exists(CODE):
    print("Using existing checkout:", CODE)
else:
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, CODE], check=True)
sys.path.insert(0, CODE)
from ocsd import sync
RESULTS = sync.results_for_run(DRIVE_ROOT, RUN_ID)
print("Selected results:", RESULTS)
if PUSH_RESULTS:
    sync.try_push_results(RESULTS, sync.colab_token())
else:
    print("Set PUSH_RESULTS=True to upload this run, including its manifest")''',
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
from ocsd.app import launch
launch(backbone=E.backbone, share=True)''',
], "02_app_gradio.ipynb")


# CPU-only export, deliberately independent of the inference environment and GitHub credentials.
nb([
    "MD:# Export completed screen evidence (CPU only)\n\nBlank RUN_ID selects the latest run. "
    "Downloads result tables, final images, and repair diagnostics after checking manifest hashes. "
    "No model download or inference. Attach the metrics ZIP first, then all image ZIPs; each is below 32 MiB. No GitHub token is needed.",
    r'''from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import drive, files
import json, hashlib, re

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/KLTN_OCSD')
RUN_ID = "" #@param {type:"string"}
if not RUN_ID:
    RUN_ID = json.loads((ROOT / 'results/latest_run.json').read_text())['run_id']
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_ID), 'Invalid RUN_ID'
manifest_path = ROOT / 'results' / 'runs' / RUN_ID / 'run_manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'complete', 'Run is incomplete: resume before interpreting results'
assert manifest['contract']['experiment']['tier'] in ('screen', 'repair_screen', 'semantic_screen', 'instance_screen', 'count_screen', 'clean_screen', 'foreground_screen')
methods = {'gligen', 'gligen_bon', 'grounded_repair', 'grounded_repair_no_m2', 'grounded_repair_sparse', 'semantic_repair', 'gligen_layout5', 'gligen_semantic5', 'instance_repair', 'instance_repair_focus', 'instance_repair_count', 'instance_repair_count_global', 'instance_repair_count_init', 'instance_repair_count_blend', 'instance_repair_count_clean', 'instance_repair_count_context', 'instance_repair_count_matte_hard', 'instance_repair_count_matte_soft'}
selected = []
for relative, digest in manifest['artifacts'].items():
    rel = Path(relative)
    result = (rel.parts[:3] == ('results', 'runs', RUN_ID)
              and rel.suffix.lower() in {'.csv', '.json', '.md', '.txt', '.log', '.png'})
    output = (rel.parts[:3] == ('outputs', 'runs', RUN_ID) and len(rel.parts) >= 6
              and rel.parts[3] == 'quickdraw' and rel.parts[4] in methods)
    if not (result or output):
        continue
    p = (ROOT / rel).resolve()
    assert p.is_relative_to(ROOT.resolve()), 'Invalid artifact path'
    assert p.is_file(), f'Missing: {p}'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Changed: {p}'
    selected.append((p, relative))
assert selected, 'No artifacts found'
# Bound uncompressed size conservatively; each ZIP is independently readable.
MAX_PART_BYTES = 24 * 2**20
archives = []
def export_parts(label, entries):
    part, group = 1, []
    manifest_size = manifest_path.stat().st_size
    size = manifest_size
    def write_group(group, part):
        archive = f'/content/ocsd_review_{RUN_ID}_{label}_{part:02d}.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
            z.write(manifest_path, 'run_manifest.json')
            for p, relative in group:
                z.write(p, relative)
        assert Path(archive).stat().st_size < 32 * 2**20, 'Archive exceeds transfer limit'
        archives.append(archive)
    for p, relative in sorted(entries, key=lambda entry: entry[1]):
        item_size = p.stat().st_size + len(relative.encode('utf-8')) * 2 + 512
        assert manifest_size + item_size < MAX_PART_BYTES, f'Single artifact too large: {relative}'
        if group and size + item_size > MAX_PART_BYTES:
            write_group(group, part)
            part += 1
            group, size = [], manifest_size
        group.append((p, relative))
        size += item_size
    if group:
        write_group(group, part)

export_parts('metrics', [(p, rel) for p, rel in selected if p.suffix.lower() != '.png'])
export_parts('images', [(p, rel) for p, rel in selected if p.suffix.lower() == '.png'])
print('Exported existing files; no model execution:', len(selected), 'run:', RUN_ID)
print('Upload metrics first, then every image part. If browser blocks downloads, use Colab Files:')
for archive in archives:
    print(archive, round(Path(archive).stat().st_size / 2**20, 2), 'MiB')
    files.download(archive)''',
], "03_export_screen_review.ipynb")
