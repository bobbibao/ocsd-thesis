"""Conservative routing for surplus detections inside one requested instance region."""
import numpy as np

METHODS = ('gligen_layout5', 'instance_repair_focus', 'instance_repair_count_global', 'instance_repair_count')


def _area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


def _intersection(a,b):
    return _area((max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])))


def choose_count_target(scene, check, dets):
    """Only complete scenes; surplus must be distinct from every matched detection.

    An extra must have > half its box inside exactly one requested box of its
    class. Overlapping/contained duplicate detections are not evidence to edit.
    No detector thresholds are tuned on the twelve reviewed scenes.
    """
    if any(j < 0 for j,_ in check['matched']) or check['oce_c'] <= 0:
        return None, []
    used={j for j,_ in check['matched']}
    owned={}
    counts=scene.class_counts()
    for jd,d in enumerate(dets):
        box=d['box']
        if jd in used or not np.isfinite(box).all() or _area(box)<=0:
            continue
        if sum(x['cls']==d['cls'] for x in dets)<=counts.get(d['cls'],0):
            continue
        # Conservative across classes too: a second label on one object is not an extra.
        if any(_intersection(box,dets[j]['box'])/max(min(_area(box),_area(dets[j]['box'])),1e-9)>=.5 for j in used):
            continue
        owners=[i for i,o in enumerate(scene.objects) if o.cls==d['cls'] and _intersection(box,o.box)/_area(box)>.5]
        if len(owners)==1:
            owned.setdefault(owners[0],[]).append(jd)
    if not owned:return None, []
    target=min(owned,key=lambda i:(-len(owned[i]),i))
    return target, owned[target]
