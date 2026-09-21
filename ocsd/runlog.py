"""Run logging for Colab: every run writes a full log, per-stage status and Python tracebacks to Drive
(results/logs/), so failures can be diagnosed without copying anything from the notebook.

  results/logs/run_<timestamp>.log   full stdout/stderr of the run (tee'd)
  results/logs/status.json           per-stage: state (ok/failed/skipped), time, error summary
  results/logs/errors/<stage>_<ts>.txt   full traceback + environment for each failure
  results/logs/LATEST_ERROR.txt      copy of the most recent traceback
  results/logs/env.json              GPU, library versions, git commit
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import traceback

_STATE = {"log_dir": None, "log_path": None, "tee": None}


class _Tee:
    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
            self.f.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.f.flush()
        except Exception:
            pass

    def __getattr__(self, k):
        return getattr(self.stream, k)


def start(results_dir: str, code_dir: str = "") -> str:
    """Start (or continue) logging for this Colab session. Safe to call more than once."""
    d = os.path.join(results_dir, "logs")
    os.makedirs(os.path.join(d, "errors"), exist_ok=True)
    if _STATE["log_path"] is None:
        path = os.path.join(d, time.strftime("run_%Y%m%d_%H%M%S.log"))
        f = open(path, "a", buffering=1)
        _STATE.update(log_dir=d, log_path=path, tee=f)
        # wrap the CURRENT streams (in Colab these are the notebook's output streams)
        sys.stdout = _Tee(sys.stdout.stream if isinstance(sys.stdout, _Tee) else sys.stdout, f)
        sys.stderr = _Tee(sys.stderr.stream if isinstance(sys.stderr, _Tee) else sys.stderr, f)
    env = environment(code_dir)
    json.dump(env, open(os.path.join(d, "env.json"), "w"), indent=1)
    print(f"[log] writing to {_STATE['log_path']}")
    print("[env]", json.dumps(env))
    return _STATE["log_path"]


def environment(code_dir: str = "") -> dict:
    env = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0])
    for mod in ("torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "huggingface_hub",
                "torchmetrics", "controlnet_aux", "numpy"):
        try:
            import importlib.metadata as md
            env[mod] = md.version(mod.replace("_", "-") if mod == "controlnet_aux" else mod)
        except Exception as e:
            env[mod] = f"not installed: {type(e).__name__}"
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"], env["vram_gb"] = p.name, round(p.total_memory / 2 ** 30, 1)
        else:
            env["gpu"] = "none"
    except Exception:
        pass
    if code_dir:
        env["git"] = subprocess.run(f"git -C {code_dir} log -1 --format='%h %s'", shell=True, capture_output=True,
                                    text=True).stdout.strip()
    return env


def _status_path():
    return os.path.join(_STATE["log_dir"], "status.json")


def read_status() -> dict:
    if _STATE["log_dir"] and os.path.exists(_status_path()):
        try:
            return json.load(open(_status_path()))
        except Exception:
            return {}
    return {}


def _write_status(stage, **kw):
    s = read_status()
    s[stage] = dict(s.get(stage, {}), **kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(s, open(_status_path(), "w"), indent=1)


def log_exception(stage: str, exc: BaseException = None, context: str = "") -> str:
    """Write a traceback file for `stage` (also usable inside loops for non-fatal errors)."""
    tb = traceback.format_exc() if exc is None else "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ts = time.strftime("%Y%m%d_%H%M%S")
    body = f"stage: {stage}\ncontext: {context}\ntime: {ts}\nlog: {_STATE['log_path']}\n\n{tb}\n\nenv: " \
           f"{json.dumps(environment())}\n"
    if _STATE["log_dir"]:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stage)[:60]
        p = os.path.join(_STATE["log_dir"], "errors", f"{safe}_{ts}.txt")
        open(p, "a").write(body + "\n" + "=" * 80 + "\n")
        open(os.path.join(_STATE["log_dir"], "LATEST_ERROR.txt"), "w").write(body)
        return p
    return ""


@contextlib.contextmanager
def stage(name: str, raise_errors: bool = False):
    """Run a notebook stage: record start/end/duration; on error write the traceback to Drive and continue
    (so later stages still run on whatever exists), unless raise_errors=True."""
    t0 = time.time()
    print(f"\n{'=' * 20} STAGE {name} {'=' * 20}")
    if _STATE["log_dir"]:
        _write_status(name, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        yield
    except KeyboardInterrupt:
        if _STATE["log_dir"]:
            _write_status(name, state="interrupted", minutes=round((time.time() - t0) / 60, 1))
        raise
    except BaseException as e:
        p = log_exception(name, e)
        if _STATE["log_dir"]:
            _write_status(name, state="failed", minutes=round((time.time() - t0) / 60, 1),
                          error=f"{type(e).__name__}: {e}"[:500], traceback_file=p)
        print(f"\n!!! STAGE {name} FAILED: {type(e).__name__}: {e}\n!!! traceback saved to {p}\n")
        traceback.print_exc()
        if raise_errors:
            raise
    else:
        if _STATE["log_dir"]:
            _write_status(name, state="ok", minutes=round((time.time() - t0) / 60, 1), error=None)
        print(f"{'=' * 20} STAGE {name} OK ({(time.time() - t0) / 60:.1f} min) {'=' * 20}")


def mark_skipped(name: str, reason: str):
    print(f"[skip] {name}: {reason}")
    if _STATE["log_dir"]:
        _write_status(name, state="skipped", reason=reason)


def summary():
    s = read_status()
    print("\nStage status:")
    for k, v in s.items():
        print(f"  {k:24s} {v.get('state'):12s} {v.get('minutes', '')!s:>6} min  {v.get('error') or v.get('reason') or ''}")
    return s
