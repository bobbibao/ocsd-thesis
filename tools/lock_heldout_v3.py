"""Write the locked held-out v3 protocol once the dev v3 rule has frozen the OCSD v3 configuration.

Usage: python tools/lock_heldout_v3.py <dev3 run.json> <kept arm: sil_full|sil_layout|sil_full_refine|sil_layout_refine>
The kept arm's options and (if refined) the refiner revision recorded by the dev run are copied verbatim.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(dev_run, kept):
    from ocsd.data import load_scene
    from ocsd.direct_instances import sampling_work
    dev = json.loads(Path(dev_run).read_text())
    dev_protocol = dev['contract']['protocol']
    arm = next(a for a in dev_protocol['arms'] if a['name'] == kept)
    refine = bool(arm.get('refine_of'))
    if refine and not dev.get('refiner_revision'): raise ValueError('Dev run has no refiner revision')
    v2 = json.loads((ROOT/'assets/heldout_v2/protocol.json').read_text())
    tuned = json.loads((ROOT/'assets/heldout_v3/tuned_paper_v2.json').read_text())
    sids = sorted({k.split('/')[1] for k in json.loads((ROOT/'assets/heldout_v3/manifest.json').read_text())})
    scenes = [load_scene(str(ROOT/'assets/heldout_v3/quickdraw'/sid)) for sid in sids]
    calls = sum(2*2*sampling_work(s.n, 50, .36) for s in scenes)
    ocsd_v3 = dict(name='ocsd_v3', scribble=False, fusion='official_mean', silhouette=True,
                   silhouette_layout_only=bool(arm.get('silhouette_layout_only')), refine=refine)
    protocol = dict(
        schema='ocsd-heldout-v3',
        stage='final locked comparison: complete OCSD v3 vs the original OCSD (paper_v2) and the strongest baselines',
        scenes='assets/heldout_v3/manifest.json', seeds=[0, 1],
        ocsd_v3_selected_by=dict(dev_run=dev['run_id'], kept_arm=kept, rule='assets/direct_dev_v3/protocol.json decision_rules',
                                 review='docs/direct_dev3/analyst_review_blind.txt'),
        arms=[
            dict(name='ocsd_v3', family='OCSD v3: OCSD instance parser + sketch-silhouette parser + InstanceDiffusion core'
                 + (' + SDXL-Refiner cascade' if refine else ''), role='proposed (complete model)'),
            dict(name='direct_box', family='InstanceDiffusion (external, Wang et al. CVPR 2024), boxes + phrases',
                 role='ablation: same generator without the OCSD sketch parser'),
            dict(name='ocsd', family='original OCSD (thesis/paper_v2): M2 objects, identity learning S1=S2=100, region attention, energy',
                 role='primary comparator; paper settings + frozen paper_v2 tuning (assets/heldout_v3/tuned_paper_v2.json)'),
            dict(name='gligen', family='GLIGEN boxes (SD 1.4)', role=f"baseline; gligen_beta {tuned['baselines']['gligen']['gligen_beta']} (paper_v2 tuned)"),
            dict(name='collage', family='OCSD M2 objects + ControlNet + SDEdit', role=f"baseline; best paper_v2 baseline (strength {tuned['baselines']['collage']['collage_strength']})"),
            dict(name='controlnet', family='SD 1.5 ControlNet scribble', role=f"baseline; cn_scale {tuned['baselines']['controlnet']['cn_scale']} (paper_v2 tuned)")],
        direct_arms=[dict(name='direct_box', scribble=False, fusion='official_mean'), ocsd_v3],
        steps=50, mis=.36, alpha=.8, guidance=7.5, negative=v2['negative'], precision=v2['precision'],
        direct_max_seconds=3600, direct_max_unet_sample_calls=calls,
        **(dict(refiner_repo='stabilityai/stable-diffusion-xl-refiner-1.0', refiner_revision=dev['refiner_revision'],
                refiner_strength=dev_protocol['refiner_strength'], refiner_steps=dev_protocol['refiner_steps']) if refine else {}),
        experiment=dict(tier='heldout_final', cfg=dict(steps=30, obj_steps=20, K=2, S1=100, S2=100), max_minutes=300,
                        tuned='assets/heldout_v3/tuned_paper_v2.json (copy of results/paper_v2/tuning/tuned.json)'),
        model_provenance=v2['model_provenance'], accepted_cpu_run=v2['accepted_cpu_run'], accepted_cpu_report=v2['accepted_cpu_report'],
        primary_endpoint='Per scene (seeds averaged): OPR and mIoU (OWLv2, same evaluator as paper_v2) and sketch_chamfer (SAM, lower is better).',
        primary_tests=['ocsd_v3 vs ocsd: opr', 'ocsd_v3 vs ocsd: miou', 'ocsd_v3 vs ocsd: sketch_chamfer'],
        key_secondary_tests=['ocsd_v3 vs gligen and vs collage on the same three metrics (Holm over six)'],
        analysis=['Unit = scene (18); per-scene value = mean over seeds 0/1.',
                  'Primary: exact two-sided sign test over scenes, Holm over the three primary tests; paired bootstrap 95% CIs.',
                  'Claim rule: "OCSD v3 improves over the original OCSD" only if at least one primary test has Holm p < 0.05 in its '
                  'favour and no primary metric is significantly worse; otherwise report the result as it is.',
                  'Secondary (CIs only): count accuracy, OCE, relations, facing accuracy, found rate, CLIP, sampling work and time '
                  'for all pairs; blind analyst review of 36 six-panel sheets (structure + naturalness 0/1/2), committed before unblinding.'],
        limits=['InstanceDiffusion and the SDXL refiner are external pretrained components (credited); OCSD v3 contributes the '
                'sketch instance/silhouette parsing, the layout-only release rule and the pipeline.',
                'Original OCSD and the baselines are SD 1.4/1.5 based; OCSD v3 uses the InstanceDiffusion (SD 1.5) checkpoint, so part '
                'of any difference is the generator. direct_box isolates the parser contribution on the same generator.',
                'Chamfer is partly circular for silhouette-conditioned arms; OPR/mIoU are not.',
                'Review is by the analyst (Bobby delegated), blind to arm labels; not a user study.'],
        fidelity=v2['fidelity'], silhouette_parser=v2['silhouette_parser'],
        locked_note='Locked before any held-out v3 output. Scenes: tools/build_heldout_v3.py (seed 20261107; QuickDraw drawings '
                    'disjoint from held-out v1, v2 and dev). Legacy methods use the paper tier settings and the frozen paper_v2 tuning, never re-tuned.')
    (ROOT/'assets/heldout_v3/protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('locked: ocsd_v3 =', ocsd_v3, '| direct UNet calls', calls)


if __name__ == '__main__':
    main(*sys.argv[1:3])
