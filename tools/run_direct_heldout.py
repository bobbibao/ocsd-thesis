"""Run or package the locked held-out InstanceDiffusion arms; finalization uses no model imports."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.heldout import run_direct as run, finalize

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache'); p.add_argument('--out'); p.add_argument('--attempt', required=True)
    p.add_argument('--setup-returncode', type=int, default=0)
    p.add_argument('--finalize-only', action='store_true'); p.add_argument('--interrupted', action='store_true')
    a = p.parse_args()
    if a.finalize_only: result = finalize(a.attempt, a.interrupted)
    else:
        if not a.cache or not a.out: p.error('--cache and --out are required for generation')
        result = run(a.cache, a.out, a.attempt, a.setup_returncode)
    sys.exit(0 if result['status'] == 'complete' else 1)
