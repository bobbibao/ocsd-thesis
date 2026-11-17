"""Write assets/xl_paper_v1/protocol.json: the locked confirmation of the two frozen OCSD-XL settings on the 72
paper benchmark scenes (docs/XL_PAPER_PROTOCOL.md). Run once, before any image of this study exists.
python tools/lock_xl_paper.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.engine_xl import XL_MODELS
from ocsd import ocsd_xl
from ocsd.xl_study import arm_digest, sd15_digest

ROOT = Path(__file__).resolve().parents[1]
ocsd_xl.use_round('c1')
ASSETS = 'assets/paper_v3'
OUT = ROOT/'assets/xl_paper_v1'
RUN = '20261105T085721Z_c08740c2'                        # run audited in docs/PAPER3_20261105_RESULTS.md
SRC = f'paper_v3/outputs/runs/{RUN}/quickdraw/'
CONFIRM = json.loads((ROOT/'assets/xl_confirm_v1/protocol.json').read_text())


def main():
    arms = list(ocsd_xl.ARMS)
    expected = json.loads((ROOT/ASSETS/'expected_scenes.json').read_text())['scenes']
    scenes = sorted(expected)
    # Blind sheets: half of the benchmark, fixed here by scene number (3 of the 6 scenes of each of the 12 cells), seed 0.
    review = [sid for sid in scenes if sid.rsplit('_', 1)[1] in ('003', '004', '005')]
    earlier = json.loads((ROOT/'docs/paper3_20261105/run_manifest.json').read_text())
    assert earlier['run_id'] == RUN and len(scenes) == 72 and len(review) == 36
    # the settings and the tests are those of the first confirmation, unchanged
    assert arms == CONFIRM['arms'] and arm_digest(arms) == CONFIRM['arm_config_sha256']
    p = dict(
        schema='ocsd-xl-study-v3', round='c1',
        title='OCSD-XL on the paper benchmark (72 QuickDraw-Scenes): two frozen settings vs OCSD v1/v2 and baselines',
        stage='second confirmation: the settings, arms, tests and claim rules of assets/xl_confirm_v1, on the paper benchmark',
        drive_root='xl_paper_v1', scene_assets=ASSETS,
        scene_import=dict(heldout_version='v4', source='data/benchmarks/quickdraw',
                          expected='assets/paper_v3/expected_scenes.json',
                          earlier_run=RUN, earlier_run_dataset_id=earlier['dataset_id'],
                          note='The scenes are not committed. They are copied from the paper benchmark on Drive at run '
                               'time and must match the expected list; the run records whether the copied files have '
                               'the dataset identity that the 2026-11-05 run recorded.'),
        scenes=scenes, seeds=[0, 1], size=ocsd_xl.SIZE,
        arms=arms, arm_config_sha256=arm_digest(arms), models=XL_MODELS,
        arm_settings={a: dict(kind=ocsd_xl.ARMS[a][0], overrides=ocsd_xl.ARMS[a][1]) for a in arms},
        common_settings=ocsd_xl.XL_COMMON, ocsd_xl_settings=dict(ocsd_xl.XL_OCSD),
        frozen_from={'ocsd_xl_natural': 'xl_dev_v3 arm ocsd_xl_w04; unchanged since assets/xl_confirm_v1',
                     'ocsd_xl_faithful': 'xl_dev_v3 arm ocsd_xl_w08_noenergy; unchanged since assets/xl_confirm_v1'},
        # SD 1.5 arms: OCSD-v2 is generated in this run with the frozen paper_v2 tuning; the others are the images of
        # the locked paper benchmark run of 2026-11-05 (same scenes, seeds and settings), re-scored here.
        sd15_tier='heldout_final', sd15_arms=['ocsd_v2'],
        imported_arms={m: SRC + m for m in ('ocsd', 'gligen', 'collage', 'controlnet', 'ocsd_v3', 'direct_box')},
        sd15_fallback=['ocsd', 'gligen', 'collage', 'controlnet'],
        reference_arm='ocsd', review_salt='ocsd-xl-paper-v1',
        review=dict(scenes=review, seeds=[0]),
        evaluation=dict(detector='OWLv2', det_thr=0.3, match_iou=0.1, image_size=512,
                        sketch_fidelity='SAM outline vs raw sketch ink', quality_proxy='PickScore v1 (secondary)',
                        naturalness='blind analyst rating 0 / 1 / 2 on 36 sheets: the scenes numbered 003 to 005 (3 per '
                                    'cell, 9 per object-count bin), seed 0, every arm on each sheet in shuffled order, '
                                    'written before review_key.json or any table is opened'),
        analysis=dict(
            unit='scene; per-scene value = mean over seeds 0 and 1 (72 scenes) for the automatic metrics, and the '
                 'seed 0 rating of the 36 reviewed scenes for naturalness',
            test='exact two-sided sign test over scenes; Holm over the four primary tests; paired bootstrap 95% CIs',
            primary=CONFIRM['analysis']['primary'],
            claims=['C-faithful "OCSD-XL (faithful) keeps objects better than the original OCSD": P1 or P2 has Holm '
                    'p < 0.05 in its favour and neither is significantly worse.',
                    'C-natural "OCSD-XL (natural) looks more natural than the original OCSD without losing objects": '
                    'P3 has Holm p < 0.05 in its favour and P4 is not significantly worse (Holm p >= 0.05 or in favour).',
                    'A claim is reported as confirmed on the paper benchmark only if it holds here. The two test sets '
                    'are reported side by side and never pooled; a claim that holds on one set and not on the other is '
                    'reported as such.'],
            secondary='Both settings against ocsd_v2, xl_controlnet, xl_collage, gligen, collage, controlnet, ocsd_v3 '
                      'and direct_box on OPR, mIoU, sketch chamfer, naturalness and PickScore, and the means by '
                      'object-count bin: differences with bootstrap CIs and unadjusted sign tests, labelled exploratory.',
            tool='tools/analyze_xl_confirm.py <results> <ratings> xl_paper_v1 (unchanged since the first confirmation)'),
        limits=['The naturalness rater is the analyst who developed the settings and who has seen OCSD-XL images of '
                'other scenes; sheets are blind and shuffled, but SDXL images can be recognisable by their look.',
                'Naturalness is rated on half of the scenes and one seed, fixed in this lock.',
                'OCSD-XL runs without M3 (identity learning does not train on SDXL in this code); the original OCSD '
                'uses M3.', 'Part of any difference to the SD 1.5 methods is the backbone; xl_controlnet and '
                'xl_collage are the same-backbone references.',
                'The 72 scenes are the evaluation scenes of results/paper_v2 and of the 2026-11-05 run: the SD 1.5 '
                'methods and OCSD v3 have been reported on them before. No OCSD-XL image of these scenes exists before '
                'this run, and no OCSD-XL setting was chosen on them.',
                'OCSD-XL has not been run on single-object scenes before (18 of the 72).'])
    p['sd15_config_sha256'] = sd15_digest(p)
    assert p['sd15_config_sha256'] == CONFIRM['sd15_config_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(p, indent=1) + '\n')
    print('locked', OUT/'protocol.json', p['arm_config_sha256'][:12], p['sd15_config_sha256'][:12], len(p['scenes']),
          'scenes,', len(review), 'reviewed')


if __name__ == '__main__':
    main()
