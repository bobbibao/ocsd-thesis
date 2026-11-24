"""Write notebooks/06_ocsd_m4_one_run.ipynb (the one-run M4 study, ocsd/m4_study.py). python tools/make_m4_notebook.py"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANCH = "claude/project-thread-5zyzae"

INTRO = f"""# OCSD with harmonised M4: one run, from tuning to the paper benchmark

OCSD on **Stable Diffusion 1.5 with all five modules M1–M5 (M3 on)**. Only M4 (scene construction) is rebuilt:
one scene trajectory in which the M2 objects are re-imposed through a halo-free, feathered mask with the trajectory's own
noise, and then released smoothly (low frequencies last), so the last steps redraw edges, contact shadows and lighting.

One **Run all** does everything, in this order (protocol locked in `assets/m4_study_v1/protocol.json`, see
`docs/M4_STUDY_PROTOCOL.md`):
1. **Development** on all 36 scenes of the paper_v2 *tuning split* × seeds 0/1: the original OCSD + 8 M4 settings,
   then one combined setting built from the changes that helped.
2. **Freeze**: a rule written before the run picks one setting (OPR + mIoU + RA, PickScore not below the original OCSD)
   and writes it to `KLTN_OCSD/m4_study_v1/locked.json`. It is never re-chosen.
3. **Paper benchmark**: the frozen setting on the 72 evaluation scenes × seeds 0/1, against the original OCSD, GLIGEN,
   Collage, ControlNet and OCSD-v2 (same scenes and seeds; their earlier images and scores are reused, nothing is
   regenerated), with pre-registered paired sign tests (Holm). The primary results are saved at this point.
4. **Extra baselines** (last, optional): T2I-Adapter and Zhang et al. (2025) on the same scenes and seeds.

**How to run:** Runtime → Change runtime type → **A100 GPU** (an L4 also works, about twice as slow) → **Run all**.
About 8–9 hours on an A100 for the primary results, plus about 1.5 hours for the extra baselines (A100 uses roughly
11–12 compute units per hour). Keep this browser tab open. If Colab disconnects, open the notebook again and **Run all**:
everything already on Drive is kept and the run continues where it stopped. At the end two ZIPs download (also on Drive
in `KLTN_OCSD/m4_study_v1/paper/results/`).
"""

SETTINGS = f'''#@title 1. Settings
BRANCH     = "{BRANCH}"  #@param {{type:"string"}}
DRIVE_ROOT = "/content/drive/MyDrive/KLTN_OCSD"   #@param {{type:"string"}}
MAX_HOURS  = 23      #@param {{type:"number"}}
RUN_EXTRAS = True    #@param {{type:"boolean"}}
# MAX_HOURS: the run stops cleanly after this many hours (Run all again to continue).
# RUN_EXTRAS: also generate T2I-Adapter and Zhang et al. (2025) after the primary results (about 1.5 hours).'''

SETUP = r'''#@title 2. Drive, GPU check, code, libraries
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
print(gpu.stdout)
if gpu.returncode:
    raise RuntimeError("No GPU: Runtime > Change runtime type > A100 GPU, then Run all")
if int(gpu.stdout.strip().split(",")[-1]) < 22000:
    raise RuntimeError("This run needs at least 22 GB of GPU memory (A100 or L4). Runtime > Change runtime type > A100 GPU")
CODE = "/content/KLTN_OCSD_m4"
if os.path.exists(CODE):
    subprocess.run(["git", "-C", CODE, "fetch", "-q", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", CODE, "checkout", "-q", "-B", BRANCH, "FETCH_HEAD"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, "https://github.com/bobbibao/ocsd-thesis.git", CODE], check=True)
print("Code:", subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
sys.path.insert(0, CODE)
for _m in [m for m in sys.modules if m == "ocsd" or m.startswith("ocsd.")]: del sys.modules[_m]
os.environ["HF_HOME"] = "/content/hf_cache"      # model cache on local disk (Drive breaks its symlinks)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(CODE, "requirements-colab.txt")], capture_output=True, text=True)
print(r.stdout[-1000:], r.stderr[-2000:])
if r.returncode:
    raise RuntimeError("pip install failed (see the text above)")
from ocsd import hfcache
hfcache.repair(os.environ["HF_HOME"])'''

PREFLIGHT = r'''#@title 3. Preflight (no model, a few seconds): the lock matches the code and every input is on Drive
from ocsd import m4_study
PRE = m4_study.preflight(DRIVE_ROOT)
print("Left to do:", PRE["left"])
for n in PRE["notes"]: print("Note:", n)
print("Preflight OK")'''

RUN = r'''#@title 4. Run everything (development -> freeze -> paper benchmark -> extra baselines)
import time
t0 = time.time()
try:
    OUT = m4_study.main(DRIVE_ROOT, max_hours=MAX_HOURS, run_extras=RUN_EXTRAS)
except Exception:
    print("\nThe run stopped with an error. The traceback is also on Drive:",
          os.path.join(DRIVE_ROOT, "m4_study_v1", "logs", "LATEST_ERROR.txt"))
    raise
print(f"Session time: {(time.time() - t0) / 3600:.1f} h; state:", m4_study.stage_left(DRIVE_ROOT))'''

SHOW = r'''#@title 5. Results and downloads
import glob
from IPython.display import Image, Markdown, display
base = os.path.join(DRIVE_ROOT, "m4_study_v1")
for f in ("SELECTION.md", "paper/results/summary.md"):
    if os.path.exists(os.path.join(base, f)):
        display(Markdown(open(os.path.join(base, f)).read()))
sheets = sorted(glob.glob(os.path.join(base, "paper/results/review/*.jpg")))[:2]
for s in sheets: display(Image(s, width=1400))
left = m4_study.stage_left(DRIVE_ROOT)
if left["paper_images_left"] or not left["locked"]:
    print("Not finished yet: Run all again to continue.", left)
else:
    from google.colab import files
    for z in sorted(glob.glob(os.path.join(base, "paper/results/*.zip"))):
        try:
            files.download(z)
        except Exception as exc:
            print("Get it from Drive:", z, exc)
    print("Done. Disconnect the GPU runtime (Runtime > Disconnect and delete runtime).")'''


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO)] + [nbf.v4.new_code_cell(c) for c in (SETTINGS, SETUP, PREFLIGHT, RUN, SHOW)]
    nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    out = os.path.join(ROOT, "notebooks", "06_ocsd_m4_one_run.ipynb")
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
