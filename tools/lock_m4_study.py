"""Write assets/m4_study_v1/protocol.json: the locked one-run study of OCSD with harmonised M4 (SD 1.5, M1-M5, M3 on),
docs/M4_STUDY_PROTOCOL.md. Run once, before any image of this study exists.
python tools/lock_m4_study.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.config import BACKBONES
from ocsd.m4_study import STUDY, arm_digest, baseline_digest
from ocsd.methods import M4_ARMS, M4H
from ocsd import xl_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / STUDY
TIER = 'heldout_final'          # SD 1.5 paper settings + the frozen paper_v2 tuning (assets/heldout_v3/tuned_paper_v2.json)
XLP = json.loads((ROOT / 'assets/xl_paper_v1/protocol.json').read_text())
CELLS = [f'qd_{n}_{c}' for n in ('1', '3', '5', '8p') for c in ('complex', 'medium', 'simple')]


def main():
    # the earlier SD 1.5 methods are unchanged by this study's code (same digest as the 2026-11-17 lock)
    assert xl_study.sd15_digest(XLP) == XLP['sd15_config_sha256'], 'SD 1.5 method settings changed'
    dev_scenes = sorted(f'{c}_{i:03d}' for c in CELLS for i in range(3))   # all 36 scenes of the paper_v2 tuning split
    candidates = list(M4_ARMS)
    dev_arms = ['ocsd'] + candidates
    paper_scenes = list(XLP['scenes'])
    review = [s for s in paper_scenes if s.rsplit('_', 1)[1] in ('003', '004', '005')]
    assert len(dev_scenes) == 36 and len(paper_scenes) == 72 and len(review) == 36
    assert not set(dev_scenes) & set(paper_scenes)
    imported = {m: f'xl_paper_v1/outputs/quickdraw/{m}' for m in ('ocsd', 'gligen', 'collage', 'controlnet', 'ocsd_v2')}
    extras = ['t2i_adapter', 'zhang2025']
    fallback = list(imported)
    primary = [dict(id='P1', a='ocsd_final', b='ocsd', metric='opr', better=1),
               dict(id='P2', a='ocsd_final', b='ocsd', metric='miou', better=1),
               dict(id='P3', a='ocsd_final', b='ocsd', metric='ra', better=1),
               dict(id='P4', a='ocsd_final', b='ocsd', metric='pickscore', better=1)]
    p = dict(
        schema='ocsd-m4-study-v1',
        title='OCSD with harmonised M4 (SD 1.5, M1-M5, M3 on): one-run development, freeze and paper benchmark',
        drive_root=STUDY, sd15_tier=TIER,
        models={k: BACKBONES['sd15'][k] for k in ('sd', 'controlnet', 't2i_adapter', 'gligen')},
        review_salt='ocsd-m4-study-v1',
        m4_centre=M4H,
        dev=dict(
            stage='development: chooses the M4 configuration; never reported as a result',
            source='data/benchmarks/quickdraw', tuning_split='data/benchmarks/tuning_split.json',
            scenes=dev_scenes, seeds=[0, 1], arms=dev_arms, reference='ocsd',
            arm_settings={a: dict(M4H, **M4_ARMS[a]) for a in candidates},
            arm_config_sha256=arm_digest(dev_arms, TIER),
            selection=dict(
                candidates=candidates,
                text='Per arm: means over the 36 development scenes (seeds averaged) of OPR, mIoU and RA (OWLv2) and PickScore. '
                     'score = mean(OPR, mIoU, RA) + 0.05 x (PickScore - PickScore of the original OCSD). Eligible: '
                     'at least 95% of the images exist and PickScore >= that of the original OCSD (not less natural '
                     'by the proxy). The eligible arm with the highest score is frozen; ties go to the earlier arm in '
                     'the candidate list. If no arm is eligible, the complete arm with the highest score is frozen and '
                     'this is reported. If PickScore is unavailable, the score is the structure mean alone.',
                pickscore_weight=0.05, min_pickscore_gain=0.0, min_complete=0.95),
            combo=dict(
                centre='ocsd_m4',
                factor_groups=[['ocsd_m4_a02', 'ocsd_m4_a04'], ['ocsd_m4_rel40'], ['ocsd_m4_w06'],
                               ['ocsd_m4_lora_late'], ['ocsd_m4_ctx']],
                factor_overrides={a: M4_ARMS[a] for a in M4_ARMS if a not in ('ocsd_m4', 'ocsd_m4_hard')},
                text='Second development step, after the one-factor arms are scored: every one-factor arm that is '
                     'eligible and scores above the centre contributes its change (of the two alpha arms only the '
                     'better one). If at least two changes qualify, the centre with all of them is generated as '
                     'ocsd_m4_combo on the same development scenes and seeds and joins the candidates of the '
                     'selection rule. The plan is written once to dev/combo.json.')),
        paper=dict(
            stage='confirmation of the frozen arm on the paper benchmark',
            scene_assets='assets/paper_v3', scene_import=XLP['scene_import'], scenes=paper_scenes, seeds=[0, 1],
            imported_arms=imported, fallback=fallback, reference='ocsd',
            reuse_scores_from='xl_paper_v1/results',
            review=dict(scenes=review, seeds=[0]),
            primary=primary,
            claim_rules=[
                dict(id='C-structure', text='OCSD with harmonised M4 keeps objects, layout and relations better than '
                     'the original OCSD', significant_any=['P1', 'P2', 'P3'], not_worse=['P1', 'P2', 'P3', 'P4']),
                dict(id='C-natural', text='OCSD with harmonised M4 is preferred by the learned human-preference model '
                     '(PickScore) over the original OCSD without losing objects', significant_any=['P4'],
                     not_worse=['P1', 'P2'])],
            claims=['Unit: scene (72); per-scene value = mean over seeds 0 and 1. Exact two-sided sign tests, Holm over '
                    'the four primary tests P1-P4 (ocsd_final vs the original OCSD on OPR, mIoU, RA, PickScore); '
                    'paired bootstrap 95% CIs; Wilcoxon p reported alongside.',
                    'C-structure holds if P1, P2 or P3 is significant (Holm p < 0.05) in favour of ocsd_final and none '
                    'of P1-P4 is significantly worse.',
                    'C-natural holds if P4 is significant in favour of ocsd_final and neither P1 nor P2 is '
                    'significantly worse.',
                    'Comparisons with OCSD-v2, GLIGEN, Collage, ControlNet, T2I-Adapter and Zhang et al. (2025) are '
                    'secondary: Holm over the secondary family is reported, and no claim of beating a baseline is made '
                    'unless that adjusted p is below 0.05.',
                    'Blind naturalness ratings (0 broken or pasted / 1 fake but readable / 2 natural) of the 36 sheets '
                    '(scenes 003-005, seed 0) are made after the run by the analyst, before review_key.json is opened; '
                    'they are secondary.'],
            secondary_vs=['ocsd_v2', 'gligen', 'collage', 'controlnet', 't2i_adapter', 'zhang2025'],
            secondary_metrics=[['opr', 1], ['miou', 1], ['ra', 1], ['count_acc', 1], ['sketch_chamfer', -1],
                               ['pickscore', 1], ['clip', 1]]),
        extras=dict(arms=extras, config_sha256=baseline_digest(extras + fallback, TIER),
                    note='generated in this run after the primary report, same scenes and seeds; settings = paper_v2 '
                         'tuned values (T2I-Adapter scale 0.8, Zhang et al. alpha 0.1)'),
        limits=['Development and selection use automatic metrics only (OWLv2 structure + PickScore); no human looks at '
                'development images before the freeze.',
                'The 72 paper scenes have been used to report the SD 1.5 methods before (paper_v2, 2026-11-05, '
                '2026-11-17); no image of the harmonised M4 exists on them before this run and no setting is chosen '
                'on them.',
                'QuickDraw-Scenes only; COCO-Sketch is not part of this study.',
                'PickScore is a learned proxy of human preference, not a human rating.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps(p, indent=1) + '\n')
    print('locked', OUT / 'protocol.json', p['dev']['arm_config_sha256'][:12], p['extras']['config_sha256'][:12])


if __name__ == '__main__':
    main()
