"""Build the locked held-out v3 scenes (18 new QuickDraw-Scenes for the final OCSD v3 comparison).

Same generator and cells as held-out v1 (3, 5 and 8+ objects x simple/medium/complex, 2 per cell),
new seed 20261107. Scenes that reuse a QuickDraw drawing from held-out v1, v2 or the dev scenes are
rejected and rebuilt with the next seed offset, so no drawing appears in two sets.
Usage: python tools/build_heldout_v3.py <quickdraw ndjson dir> <scratch dir>
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.data import build_quickdraw_scenes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets/heldout_v3'
SEED = 20261107


def used_keys():
    keys = set()
    for path in [*(ROOT/'assets/heldout_v1/quickdraw').glob('*/scene.json'), *(ROOT/'assets/heldout_v2/quickdraw').glob('*/scene.json')]:
        keys |= {str(o['qd_key']) for o in json.loads(path.read_text())['objects']}
    for spec in json.loads((ROOT/'assets/direct_dev_v1/scenes.json').read_text())['scenes']:
        keys |= {str(o['source']['key_id']) for o in spec['objects']}
    return keys


def main(qd_dir, scratch):
    scratch = Path(scratch); used = used_keys(); chosen = {}
    for offset in range(20):
        build = scratch/f'build_{offset}'
        build_quickdraw_scenes(qd_dir, str(build), per_cell=2, seed=SEED+offset)
        for sdir in sorted(build.iterdir()):
            meta = json.loads((sdir/'scene.json').read_text())
            if meta['n_obj'] == 1 or meta['sid'] in chosen: continue
            keys = {str(o['qd_key']) for o in meta['objects']}
            if keys & used: continue
            chosen[meta['sid']] = (sdir, SEED+offset); used |= keys
        if len(chosen) == 18: break
    if len(chosen) != 18: raise RuntimeError('Could not build 18 disjoint scenes')
    if OUT.exists(): shutil.rmtree(OUT)
    manifest = {}
    for sid, (sdir, seed) in sorted(chosen.items()):
        new = 'h3_' + sid; dst = OUT/'quickdraw'/new; shutil.copytree(sdir, dst)
        meta = json.loads((dst/'scene.json').read_text())
        meta.update(sid=new, heldout_set='heldout_v3', source_build=dict(
            generator='ocsd.data.build_quickdraw_scenes', seed=seed, per_cell=2, original_sid=sid,
            quickdraw_pool='first 3000 records per class (same pool as the paper benchmark, new seed)',
            disjoint_from=['heldout_v1', 'heldout_v2', 'direct_dev_v1']))
        (dst/'scene.json').write_text(json.dumps(meta, indent=1))
    for path in sorted((OUT/'quickdraw').rglob('*')):
        if path.is_file():
            manifest[path.relative_to(OUT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(len(chosen), 'scenes,', len(manifest), 'files')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
