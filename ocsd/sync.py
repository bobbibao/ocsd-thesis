"""Push small result files (logs, status, tracebacks, tables, summary, figures) from Colab to the GitHub branch
`colab-results`, so Claude can read every run's outcome directly. Needs a GitHub token stored as the Colab secret
GH_TOKEN (fine-grained, repository bobbibao/ocsd-thesis, permission Contents: read and write)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time

INCLUDE = ["logs", "tables", "figures", "summary.md", "summary.json", "progress.json", "budget_estimate.json",
           "benchmark_counts.json", "experiment_config.json", "*/per_image_*.csv", "*/quality_fid_kid.csv"]
MAX_FILE_MB = 20


def _run(cmd, cwd=None, secret=""):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(secret, "***") if secret else r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd.replace(secret, '***') if secret else cmd}\n{out[-2000:]}")
    return out


def push_results(results_dir: str, token: str, repo: str = "bobbibao/ocsd-thesis", branch: str = "colab-results",
                 work: str = "/content/_results_sync", remote_url: str = "") -> str:
    if not token and not remote_url:
        print("[sync] no GH_TOKEN secret: results stay on Drive only")
        return ""
    url = remote_url or f"https://x-access-token:{token}@github.com/{repo}.git"
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    _run("git init -q && git checkout -q -b " + branch, cwd=work)
    _run(f"git remote add origin {url}", cwd=work, secret=token or "\0")
    # start from the branch if it exists (keeps history of runs)
    try:
        _run(f"git fetch -q --depth 1 origin {branch} && git reset -q --soft FETCH_HEAD", cwd=work, secret=token or "\0")
    except RuntimeError:
        pass
    dst = os.path.join(work, "results")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for pat in INCLUDE:
        for src in glob.glob(os.path.join(results_dir, pat)):
            rel = os.path.relpath(src, results_dir)
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for f in files:
                        p = os.path.join(root, f)
                        if os.path.getsize(p) <= MAX_FILE_MB * 2 ** 20:
                            t = os.path.join(dst, os.path.relpath(p, results_dir))
                            os.makedirs(os.path.dirname(t), exist_ok=True)
                            shutil.copy2(p, t)
                            n += 1
            elif os.path.getsize(src) <= MAX_FILE_MB * 2 ** 20:
                os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
                shutil.copy2(src, os.path.join(dst, rel))
                n += 1
    msg = time.strftime("Colab results %Y-%m-%d %H:%M:%S")
    _run('git add -A && git -c user.name="colab" -c user.email="colab@users.noreply.github.com" '
         f'commit -q -m "{msg}" --allow-empty', cwd=work)
    _run(f"git push -q origin HEAD:{branch}", cwd=work, secret=token or "\0")
    print(f"[sync] pushed {n} files to {repo}@{branch}")
    return branch


def colab_token() -> str:
    try:
        from google.colab import userdata
        return userdata.get("GH_TOKEN") or ""
    except Exception:
        return os.environ.get("GH_TOKEN", "")
