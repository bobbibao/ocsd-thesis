"""Experimental semantic routing. CLIP probes are method heuristics, never evaluation truth."""
import copy
import re
import time

import numpy as np

from .matching import consistency, verify_score, verify_pass
from .method import Prepared, compose_foreground, repair_scene, _cuda_peak
from .methods import _gligen

METHODS = ('gligen_layout5', 'gligen_semantic5', 'grounded_repair_sparse', 'semantic_repair')
COLORS = ('black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'golden')
MATERIALS = ('wooden', 'metal', 'plastic', 'glass', 'stone')
# Fixed before this experiment. Unknown attributes are explicitly unsupported.
POLICY = dict(version=1, samples=5, max_local_objects=1, contrast_margin=0.0, caption_tolerance=0.5)


def crop(image, box):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, min(w-1, int(np.floor(x0)))), max(0, min(h-1, int(np.floor(y0))))
    x1, y1 = max(x0+1, min(w, int(np.ceil(x1)))), max(y0+1, min(h, int(np.ceil(y1))))
    return image[y0:y1, x0:x1]


class Probe:
    """Contrast requested background/color/material against fixed alternatives; no trained calibration."""
    def __init__(self, vis, scene, cfg):
        from .data import BACKGROUNDS
        self.vis, self.scene, self.cfg = vis, scene, cfg
        self.text_cache = {}
        self.backgrounds = [scene.bg] + sorted(set(BACKGROUNDS) - {scene.bg})
        self.attributes = []
        for i, obj in enumerate(scene.objects):
            phrase = obj.phrase.lower()
            for group in (COLORS, MATERIALS):
                hits = [word for word in group if re.search(r'\b' + word + r'\b', phrase)]
                if len(hits) == 1:
                    word = hits[0]
                    alternatives = [phrase] + [re.sub(r'\b'+word+r'\b', other, phrase) for other in group if other != word]
                    self.attributes.append((i, word, alternatives))

    def scores(self, images, texts):
        key = tuple(texts)
        if key not in self.text_cache:
            self.text_cache[key] = self.vis.clip.text_emb(texts)
        return (100 * (self.vis.clip.image_emb(images) @ self.text_cache[key].T)).detach().cpu().numpy()

    def __call__(self, image):
        dets = self.vis.gdino(image, [o.cls for o in self.scene.objects], thr=self.cfg.verify_thr)
        c = consistency(self.scene, dets, size=self.cfg.height)
        scores = self.scores([image], ['a photo '+b for b in self.backgrounds])[0]
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite semantic background probe')
        margin = float(scores[0] - max(scores[1:]))
        attrs = []
        for i, word, alternatives in self.attributes:
            # Fixed requested boxes avoid changing the crop with the detector's selected instance.
            values = self.scores([crop(image, self.scene.objects[i].box)], ['a photo of a '+p for p in alternatives])[0]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite semantic attribute probe')
            attrs.append(dict(object_id=i, attribute=word, margin=float(values[0]-max(values[1:])),
                              requested=alternatives[0], preferred=alternatives[int(values.argmax())]))
        caption = float(self.scores([image], [self.scene.caption])[0,0])
        if not np.isfinite(caption):
            raise ValueError('Nonfinite caption probe')
        missing = [i for i,(j,_) in enumerate(c['matched']) if j < 0]
        return dict(check=c, detections=dets, missing_ids=missing, attributes=attrs,
                    background=dict(requested=self.scene.bg, preferred=self.backgrounds[int(scores.argmax())], margin=margin),
                    caption_score=caption)


def issues(probe):
    attrs = {(a['object_id'], a['attribute']) for a in probe['attributes'] if a['margin'] <= POLICY['contrast_margin']}
    return attrs, probe['background']['margin'] <= POLICY['contrast_margin']


def route(probe, cfg):
    attrs, background = issues(probe)
    targets = sorted(set(probe['missing_ids']) | {i for i,_ in attrs})
    repair_steps = cfg.steps - min(max(int(round((1-cfg.repair_strength)*cfg.steps)), 0), cfg.steps-1)
    # Count batch-weighted denoiser calls, summing UNet + ControlNet; not FLOPs or wall-time equality.
    local_work = 2*cfg.steps + 4*cfg.K*cfg.obj_steps*len(targets) + 4*repair_steps
    cap = 2*cfg.steps*POLICY['samples']
    if background or len(targets) > POLICY['max_local_objects']:
        return 'global', targets, local_work, cap
    if targets:
        return ('local' if local_work <= cap else 'global'), targets, local_work, cap
    if not verify_pass(probe['check']):
        return 'global', targets, local_work, cap
    return 'keep', [], local_work, cap


def accept(before, after):
    a, b = before['check'], after['check']
    kept = {i for i,(j,_) in enumerate(a['matched']) if j >= 0}
    found = {i for i,(j,_) in enumerate(b['matched']) if j >= 0}
    if not kept <= found:
        return False, 'lost_object'
    if b['oce_c'] > a['oce_c'] or b['miou'] < a['miou']:
        return False, 'count_or_layout_regressed'
    if np.isfinite(a['ra']) and (not np.isfinite(b['ra']) or b['ra'] < a['ra']):
        return False, 'relations_regressed'
    old_attr, old_bg = issues(before); new_attr, new_bg = issues(after)
    if not new_attr <= old_attr or (new_bg and not old_bg):
        return False, 'new_semantic_issue'
    if after['caption_score'] < before['caption_score'] - POLICY['caption_tolerance']:
        return False, 'caption_regressed'
    if len(new_attr)+int(new_bg) < len(old_attr)+int(old_bg) or verify_score(b) > verify_score(a):
        return True, 'accepted_proxy_gain'
    return False, 'no_proxy_gain'


def generate(kind, eng, vis, scene, seed, cfg, backbone, cache_dir, log, get_objects, diagnose, make_gligen=None):
    t0 = time.time()
    if cfg.R != 1 or not cfg.use_scene_cn or cfg.use_identity or cfg.use_energy or cfg.bo_n != POLICY['samples']:
        raise ValueError('Semantic screen requires R=1, bo_n=5, scene ControlNet, no identity training or energy guidance')
    probe = Probe(vis, scene, cfg)
    make = make_gligen or (lambda s: _gligen(eng, scene, s, cfg, backbone, cache_dir))
    best = make(seed)
    state = probe(best)
    action, targets, local_work, cap = route(state, cfg)
    log.update(policy=dict(POLICY), experimental=True, semantic_proxy_not_evaluation=True,
               route=action, targets=targets, sampling_work_cap=cap, planned_local_work=local_work,
               scene_spec=dict(caption=scene.caption, bg=scene.bg,
                               objects=[dict(index=i, cls=o.cls, phrase=o.phrase, box=o.box) for i,o in enumerate(scene.objects)]))
    history = [dict(kind='initial', probe=state)]
    diagnose('initial', best)
    if kind in ('gligen_layout5','gligen_semantic5') or action == 'global':
        log['route'] = 'global'
        # Same five GLIGEN seeds in both controls and the global branch. Never early-stop the controls.
        for attempt in range(1, POLICY['samples']):
            candidate = make(seed+100003*attempt)
            new = probe(candidate)
            if kind == 'gligen_layout5':
                accepted = verify_score(new['check']) > verify_score(state['check'])
                reason = 'layout_score' if accepted else 'no_layout_gain'
            else:
                accepted, reason = accept(state, new)
            diagnose(f'global_{attempt}_candidate', candidate)
            history.append(dict(kind='global', accepted=accepted, reason=reason, probe=new))
            if accepted:
                best, state = candidate, new
    elif action == 'local':
        objs = get_objects(targets)
        prep = Prepared(objs, compose_foreground(scene, objs, cfg), None)
        repair_check = copy.deepcopy(state['check'])
        # Only the selected target is released; other verified objects remain protected.
        for i in targets:
            repair_check['matched'][i] = (-1, 0.)
        emit = lambda name,image: diagnose('local_'+name,image)
        candidate = repair_scene(eng, scene, prep, cfg, best, repair_check, state['detections'],
                                 seed+100003, cfg.lambda0, log, protect_matched=True,
                                 replacement_ids=targets, diagnose=emit)
        if candidate is None:
            history.append(dict(kind='local', accepted=False, reason='empty_edit_mask'))
        else:
            diagnose('local_candidate',candidate)
            new = probe(candidate)
            accepted,reason = accept(state,new)
            history.append(dict(kind='local', accepted=accepted, reason=reason, probe=new))
            if accepted:
                best,state = candidate,new
    wall = time.time()-t0
    log.update(verify=history, tries=len(history), execution_wall_time_s=wall,
               scene_time=wall-log.get('m2_lookup_seconds',0)+log.get('gligen_reused_seconds',0),
               compute_includes_m2=True, peak_gb=_cuda_peak())
    return best
