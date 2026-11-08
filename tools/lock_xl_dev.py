"""Write assets/xl_dev_v1/protocol.json: the locked OCSD-XL development study. Run once, before any SDXL output.
python tools/lock_xl_dev.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.engine_xl import XL_MODELS
from ocsd.ocsd_xl import ARMS, SIZE, XL_COMMON, XL_OCSD, arm_config
from ocsd.xl_study import arm_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets/xl_dev_v1'
# Development scenes: the first scene of every cell of held-out v1 (3 / 5 / 8+ objects x simple / medium / complex).
# Held-out v1 was already used for the November studies, so it is development data now. Held-out v3 and the 72 paper
# scenes are not touched here; they stay for the confirmation.
SCENES = sorted(p.name for p in (ROOT/'assets/heldout_v1/quickdraw').iterdir() if p.name.endswith('_000'))
CANDIDATES = ['ocsd_xl', 'ocsd_xl_a02', 'ocsd_xl_a06', 'ocsd_xl_w04', 'ocsd_xl_noenergy', 'ocsd_xl_sep', 'ocsd_xl_id']


def main():
    arms = list(ARMS)
    protocol = dict(
        schema='ocsd-xl-study-v1',
        title='OCSD-XL development study (M1-M5 on SDXL, 1024 px)',
        stage='development: chooses the OCSD-XL configuration; not a held-out result',
        drive_root='xl_dev_v1', scene_assets='assets/heldout_v1', scenes=SCENES, seeds=[0, 1], size=SIZE,
        arms=arms, arm_config_sha256=arm_digest(arms), models=XL_MODELS,
        arm_settings={a: dict(kind=ARMS[a][0], overrides=ARMS[a][1]) for a in arms},
        common_settings=XL_COMMON, ocsd_xl_settings={k: v for k, v in XL_OCSD.items()},
        centre_config=arm_config('ocsd_xl')[1].to_dict(),
        reference_arm='xl_controlnet', review_salt='ocsd-xl-dev-v1',
        evaluation=dict(detector='OWLv2 (independent of the Grounding DINO used inside M2 / M5(d))', det_thr=0.3,
                        match_iou=0.1, image_size=512, sketch_fidelity='SAM outline vs raw sketch ink',
                        quality_proxy='PickScore v1 (secondary; a learned preference model, not a human rating)'),
        decision_rule=dict(
            candidates=CANDIDATES,
            ablations_only=['ocsd_xl_norepair', 'ocsd_xl_region'],
            baselines=['xl_controlnet', 'xl_collage'],
            structure='S = mean over scenes of (OPR + mIoU) / 2, OWLv2, seeds averaged',
            naturalness='N = mean blind rating (0 broken or pasted, 1 fake but readable, 2 natural) over the 18 sheets, '
                        'written by the analyst before opening review_key.json',
            steps=['1. A candidate must have all 18 images.',
                   '2. Admissible: N >= 1.0 and S >= (best S among candidates) - 0.03.',
                   '3. Choose the admissible candidate with the highest N; within 0.1 of N prefer the lower sketch '
                   'chamfer, then the lower time per image.',
                   '4. Go to confirmation only if the chosen arm has S >= S of xl_controlnet (same backbone and '
                   'ControlNet); otherwise report that M2-M5 add nothing on SDXL.',
                   '5. If no candidate is admissible, nothing is frozen; the next development round uses the '
                   'held-out v2 scenes.'],
            note='PickScore, CLIP, count accuracy, relations and facing are reported but do not enter the rule.'),
        confirmation_plan='Frozen OCSD-XL vs original OCSD (v1), OCSD-v2, GLIGEN, Collage, ControlNet (SD 1.5), '
                          'InstanceDiffusion and the SDXL baselines on the 18 held-out v3 scenes and the 72 paper '
                          'scenes; protocol locked in a separate file before that run.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=1) + '\n')
    print('locked', OUT/'protocol.json', protocol['arm_config_sha256'][:12], len(SCENES), 'scenes', len(arms), 'arms')


if __name__ == '__main__':
    main()
