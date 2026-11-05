"""Write the locked paper-benchmark v3 protocol (assets/paper_v3/protocol.json).

It copies the held-out v3 protocol (frozen OCSD v3, same arms, same tuning, same analysis) and changes only the
scene set: the 72 paper_v2 evaluation scenes (assets/paper_v3/expected_scenes.json), imported from the paper
benchmark on Drive at run time. Usage: python tools/lock_paper_v3.py
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from ocsd.direct_instances import sampling_work
    v3 = json.loads((ROOT/'assets/heldout_v3/protocol.json').read_text())
    scenes = json.loads((ROOT/'assets/paper_v3/expected_scenes.json').read_text())['scenes']
    calls = sum(2*2*sampling_work(s['n_obj'], v3['steps'], v3['mis']) for s in scenes.values())
    p = dict(v3)
    p.update(
        schema='ocsd-paper-v3',
        stage='confirmation on the paper benchmark: frozen OCSD v3 vs the original OCSD and the paper_v2 baselines',
        scenes='assets/paper_v3/expected_scenes.json (72 paper_v2 evaluation scenes; manifest written at import)',
        heldout_v3_protocol='assets/heldout_v3/protocol.json',
        direct_max_seconds=14400, direct_max_unet_sample_calls=calls,
        experiment=dict(v3['experiment'], max_minutes=360, tuned='assets/paper_v3/tuned_paper_v2.json (copy of results/paper_v2/tuning/tuned.json)'),
        analysis=v3['analysis'] + [
            'Same tests, families and claim rule as held-out v3, over 72 scenes (18 per count bin 1/3/5/8+).',
            'Secondary: by count bin; held-out v3 and this run are reported separately (no pooling).',
            'Blind analyst review: the 24 sheets of the scenes numbered 003 (one per cell x 2 seeds), fixed before output.'],
        limits=v3['limits'] + [
            'These scenes were the paper_v2 evaluation set; paper_v2 tuning used the separate tuning split (000-002), and no OCSD v3 '
            'setting was chosen on them. They are not new to the reader, so held-out v3 remains the out-of-sample test.',
            'COCO-Sketch is not included (QuickDraw only).'],
        locked_note='Locked before any paper-v3 output. OCSD v3 settings, arms and legacy tuning are those of held-out v3, unchanged.')
    (ROOT/'assets/paper_v3/protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    print('locked paper v3: 72 scenes, direct UNet calls', calls)


if __name__ == '__main__':
    main()
