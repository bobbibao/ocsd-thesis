"""Exploratory GLIGEN + protected local repair; no claim of pretrained quality improvement."""
import json
import os
import time

import numpy as np

from .matching import consistency, verify_pass, verify_score
from .method import Prepared, compose_foreground, repair_scene, _cuda_peak
from .methods import _gligen
from .provenance import atomic_json

METHODS = ('controlnet', 'gligen', 'gligen_bon', 'collage', 'ocsd_v2',
           'grounded_repair_no_m2', 'grounded_repair')
REPAIR_METHODS = ('gligen', 'gligen_bon', 'grounded_repair', 'grounded_repair_sparse')
CLIP_TOLERANCE = 0.5  # fixed prospectively; CLIP is a method guard, not an independent quality endpoint


def accept_repair(before, after, baseline_clip, candidate_clip):
    """A verifier guard, NOT a guarantee about OWLv2 or human-perceived quality."""
    kept = {i for i, (j, _) in enumerate(before['matched']) if j >= 0}
    found = {i for i, (j, _) in enumerate(after['matched']) if j >= 0}
    if not kept <= found:
        return False, 'lost_object'
    if after['oce_c'] > before['oce_c'] or after['miou'] < before['miou']:
        return False, 'count_or_layout_regressed'
    if np.isfinite(before['ra']) and (not np.isfinite(after['ra']) or after['ra'] < before['ra']):
        return False, 'relations_regressed'
    if not np.isfinite(candidate_clip) or not np.isfinite(baseline_clip):
        return False, 'invalid_clip'
    if candidate_clip < baseline_clip - CLIP_TOLERANCE:
        return False, 'clip_regressed'
    if verify_score(after) <= verify_score(before):
        return False, 'no_verifier_gain'
    return True, 'accepted'


def grounded_repair(eng, vis, scene, seed, cfg, backbone, cache_dir, log, get_objects, use_m2=True, sparse=False, diagnose=None):
    """Start from GLIGEN, lazily prepare M2 only for missing objects, preserve/reject local edits.

    get_objects returns a scene-aligned M2 set (None for unrequested indices in sparse mode).
    Sparse mode passes original missing indices to the callback. It is never called if the initial image passes,
    only extras need removal, or the no-M2 ablation is selected. No identity training is performed.
    """
    t0 = time.time()
    log['scene_spec'] = dict(caption=scene.caption, bg=scene.bg,
                             objects=[dict(index=i, cls=o.cls, phrase=o.phrase, box=o.box)
                                      for i, o in enumerate(scene.objects)])
    best = _gligen(eng, scene, seed, cfg, backbone, cache_dir)
    classes = [o.cls for o in scene.objects]
    dets = vis.gdino(best, classes, thr=cfg.verify_thr)
    c = consistency(scene, dets, size=cfg.height)
    history = [dict(kind='gligen', score=verify_score(c), opr=c['opr'], oce_c=c['oce_c'])]
    prep = Prepared(None, None, None)
    initial_clip = None
    for attempt in range(cfg.R if cfg.use_verify else 0):
        if verify_pass(c):
            break
        if use_m2 and prep.objs is None and any(j < 0 for j, _ in c['matched']):
            missing_ids = [i for i, (j, _) in enumerate(c['matched']) if j < 0]
            objs = get_objects(missing_ids) if sparse else get_objects()
            prep = Prepared(objs, compose_foreground(scene, objs, cfg), None)
        emit = (lambda name, image: diagnose(f'attempt_{attempt + 1}_{name}', image)) if diagnose else None
        if emit:
            emit('before', best)
        candidate = repair_scene(eng, scene, prep, cfg, best, c, dets, seed + 100003 * (attempt + 1),
                                 cfg.lambda0, log, protect_matched=True, **({"diagnose": emit} if emit else {}))
        if candidate is None:
            break
        if emit:
            emit("candidate", candidate)
        d2 = vis.gdino(candidate, classes, thr=cfg.verify_thr)
        c2 = consistency(scene, d2, size=cfg.height)
        if initial_clip is None:
            initial_clip = float(vis.clip.score([best], [scene.caption])[0])
        clip = float(vis.clip.score([candidate], [scene.caption])[0])
        accept, reason = accept_repair(c, c2, initial_clip, clip)
        history.append(dict(missing_ids=[i for i, (j, _) in enumerate(c['matched']) if j < 0],
                            matched_before=c['matched'], matched_after=c2['matched'],
                            detections_before=dets, detections_after=d2, kind='protected_repair', score=verify_score(c2), opr=c2['opr'],
                            oce_c=c2['oce_c'], clip=clip, accepted=accept, reason=reason))
        if accept:
            best, c, dets = candidate, c2, d2
        else:
            break  # do not spend another attempt after a rejected edit
    log.update(tries=len(history), verify=history,
               scene_time=time.time()-t0-log.get('m2_lookup_seconds', 0), peak_gb=_cuda_peak(),
               experimental=True, m2_scope="missing_only" if sparse else "full_scene", compute_includes_m2=True, clip_guard_tolerance=CLIP_TOLERANCE)
    return best


def screen_report(P, E):
    """Report the complete fixed screening population; never auto-escalate to a paper run."""
    from . import report as R
    from .stages import experiment_plan
    df = R.load_per_image(P.results, 'quickdraw')
    expected = {(m, os.path.basename(d), int(s)) for j in experiment_plan(P, E)
                for m in j['methods'] for d in j['scenes'] for s in j['seeds']}
    found = [(r.method, r.sid, int(r.seed)) for r in df.itertuples()]
    if not expected or set(found) != expected or len(found) != len(set(found)):
        raise RuntimeError('Incomplete screening population; resume before interpreting results')
    df['standalone_time_s'] = df['time_s'].fillna(0)
    for name in ('m2_s', 'm3_s'):
        if name in df:
            df['standalone_time_s'] += df[name].fillna(0)
    ps = R.per_scene(df)
    summary = R.summarize(ps, metrics=['opr', 'oce_c', 'miou', 'ra', 'clip', 'standalone_time_s'])
    ref = 'grounded_repair_sparse' if E.tier == 'repair_screen' else 'grounded_repair'
    others = ['gligen', 'gligen_bon', 'grounded_repair'] if E.tier == 'repair_screen' else [
        'gligen', 'gligen_bon', 'collage', 'ocsd_v2', 'grounded_repair_no_m2']
    paired = R.paired_tests(ps, ref=ref, others=others)
    tables = os.path.join(P.results, 'tables'); os.makedirs(tables, exist_ok=True)
    summary.to_csv(os.path.join(tables, 'screen_means.csv'), index=False)
    paired.to_csv(os.path.join(tables, 'screen_paired.csv'), index=False)
    # A screening signal can stop a weak direction, but cannot establish superiority on held-out data.
    base = summary.set_index('method')
    good = all(base.loc['grounded_repair', 'opr'] > base.loc[b, 'opr'] and
               base.loc['grounded_repair', 'miou'] >= base.loc[b, 'miou'] and
               base.loc['grounded_repair', 'oce_c'] <= base.loc[b, 'oce_c']
               for b in (('gligen', 'gligen_bon') if E.tier == 'repair_screen' else ('gligen', 'gligen_bon', 'collage')))
    decision = 'review_for_heldout_validation' if good else 'do_not_scale_up'
    if E.tier == 'repair_screen':
        # Development efficiency screen, never a statistical non-inferiority claim.
        old, new = base.loc['grounded_repair'], base.loc['grounded_repair_sparse']
        good = (new.opr >= old.opr and new.miou >= old.miou and new.oce_c <= old.oce_c
                and new.standalone_time_s < old.standalone_time_s)
        decision = 'review_sparse_quality_and_cost' if good else 'do_not_scale_up'
    result = dict(tier=E.tier, decision=decision, research_claim='unproven',
                  n_images=len(df), n_scenes=df.sid.nunique(), figures=[],
                  tables=['screen_means', 'screen_paired'])
    atomic_json(os.path.join(P.results, 'summary.json'), result)
    with open(os.path.join(P.results, 'summary.md'), 'w') as f:
        f.write('# Exploratory screen — not paper evidence\n\n' + summary.round(4).to_markdown(index=False))
        f.write('\n\nDecision: **' + decision + '**. No automatic paper/full run.\n\n')
        f.write('CLIP is used by the method guard and is not an independent quality test. '
                'Inspect images blind, paired OWLv2 results and total work before another run. '
                'Twelve scenes and one seed cannot establish general superiority. '
                'All these scenes must remain in development/tuning, never final held-out evaluation.\n')
    return result
