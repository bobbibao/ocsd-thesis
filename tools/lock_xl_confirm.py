"""Write assets/xl_confirm_v1/protocol.json: the locked confirmation of the two frozen OCSD-XL settings on the 18
held-out v3 scenes (docs/XL_CONFIRM_PROTOCOL.md). Run once, before any image of this study exists.
python tools/lock_xl_confirm.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.engine_xl import XL_MODELS
from ocsd import ocsd_xl
from ocsd.xl_study import arm_digest, sd15_digest

ROOT = Path(__file__).resolve().parents[1]
ocsd_xl.use_round('c1')
ASSETS = 'assets/heldout_v3'
OUT = ROOT/'assets/xl_confirm_v1'
SRC = 'heldout_v3/outputs/runs/20261105T015848Z_8ed9c617/quickdraw/'   # run audited in docs/HELDOUT3_20261105_RESULTS.md
PRIMARY = [
    dict(id='P1', a='ocsd_xl_faithful', b='ocsd', metric='opr', better=1),
    dict(id='P2', a='ocsd_xl_faithful', b='ocsd', metric='miou', better=1),
    dict(id='P3', a='ocsd_xl_natural', b='ocsd', metric='naturalness', better=1),
    dict(id='P4', a='ocsd_xl_natural', b='ocsd', metric='opr', better=1),
]


def main():
    arms = list(ocsd_xl.ARMS)
    p = dict(
        schema='ocsd-xl-study-v2', round='c1',
        title='OCSD-XL confirmation on held-out v3 (two frozen settings vs OCSD v1/v2 and baselines)',
        stage='confirmation: out-of-sample test of two settings frozen after development round 3',
        drive_root='xl_confirm_v1', scene_assets=ASSETS,
        scenes=sorted(q.name for q in (ROOT/ASSETS/'quickdraw').iterdir()), seeds=[0, 1], size=ocsd_xl.SIZE,
        arms=arms, arm_config_sha256=arm_digest(arms), models=XL_MODELS,
        arm_settings={a: dict(kind=ocsd_xl.ARMS[a][0], overrides=ocsd_xl.ARMS[a][1]) for a in arms},
        common_settings=ocsd_xl.XL_COMMON, ocsd_xl_settings=dict(ocsd_xl.XL_OCSD),
        frozen_from={'ocsd_xl_natural': 'xl_dev_v3 arm ocsd_xl_w04 (settings identical)',
                     'ocsd_xl_faithful': 'xl_dev_v3 arm ocsd_xl_w08_noenergy (settings identical)'},
        # SD 1.5 arms: OCSD-v2 is generated in this run with the frozen paper_v2 tuning; the others are the images of
        # the locked held-out v3 run of 2026-11-05 (same scenes, seeds and settings), re-scored here by one evaluator.
        sd15_tier='heldout_final', sd15_arms=['ocsd_v2'],
        imported_arms={m: SRC + m for m in ('ocsd', 'gligen', 'collage', 'controlnet', 'ocsd_v3', 'direct_box')},
        sd15_fallback=['ocsd', 'gligen', 'collage', 'controlnet'],
        reference_arm='ocsd', review_salt='ocsd-xl-confirm-v1',
        evaluation=dict(detector='OWLv2', det_thr=0.3, match_iou=0.1, image_size=512,
                        sketch_fidelity='SAM outline vs raw sketch ink', quality_proxy='PickScore v1 (secondary)',
                        naturalness='blind analyst rating 0 / 1 / 2 on all 36 sheets (every arm on each sheet, order '
                                    'shuffled), written before review_key.json or any table is opened'),
        analysis=dict(
            unit='scene (18); per-scene value = mean over seeds 0 and 1',
            test='exact two-sided sign test over scenes; Holm over the four primary tests; paired bootstrap 95% CIs',
            primary=PRIMARY,
            claims=['C-faithful "OCSD-XL (faithful) keeps objects better than the original OCSD": P1 or P2 has Holm '
                    'p < 0.05 in its favour and neither is significantly worse.',
                    'C-natural "OCSD-XL (natural) looks more natural than the original OCSD without losing objects": '
                    'P3 has Holm p < 0.05 in its favour and P4 is not significantly worse (Holm p >= 0.05 or in favour).',
                    'An "improved OCSD" is reported in the thesis only for a setting whose claim holds, with its '
                    'trade-off stated; if neither holds the SDXL line stays a negative result.'],
            secondary='Both settings against ocsd_v2, xl_controlnet, xl_collage, gligen, collage, controlnet, ocsd_v3 '
                      'and direct_box on OPR, mIoU, sketch chamfer, naturalness and PickScore: differences with '
                      'bootstrap CIs and unadjusted sign tests, labelled exploratory.',
            tool='tools/analyze_xl_confirm.py (committed with this lock)'),
        phase2='Only if a claim holds: the same two settings on the 72 paper scenes, with its own lock. Not part of '
               'this run.',
        limits=['The naturalness rater is the analyst who developed the settings; sheets are blind and shuffled, but '
                'SDXL images can be recognisable by their look.',
                'OCSD-XL runs without M3 (identity learning does not train on SDXL in this code); the original OCSD '
                'uses M3.', 'Part of any difference to the SD 1.5 methods is the backbone; xl_controlnet and '
                'xl_collage are the same-backbone references.',
                'Held-out v3 was used on 2026-11-05 to test OCSD v3; no OCSD-XL setting was chosen or run on it.'])
    p['sd15_config_sha256'] = sd15_digest(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(p, indent=1) + '\n')
    print('locked', OUT/'protocol.json', p['arm_config_sha256'][:12], p['sd15_config_sha256'][:12], len(p['scenes']), 'scenes')


if __name__ == '__main__':
    main()
