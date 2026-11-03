"""One-instance repair ablation: isolate local conditioning, not semantic reranking."""
import time
import numpy as np
from .matching import consistency, verify_score
from .method import Prepared, compose_foreground, protect_regions, repair_scene, _cuda_peak
from .screening import accept_repair

METHODS = ('gligen_layout5', 'grounded_repair_sparse', 'instance_repair', 'instance_repair_focus')


def choose_target(scene, check, dets, size):
    """Prefer a missing target whose requested box has the largest unprotected fraction; index breaks ties."""
    candidates = []
    for i,(j,_) in enumerate(check['matched']):
        if j >= 0:
            continue
        box = np.zeros((size,size),bool)
        x0,y0,x1,y1 = scene.objects[i].box
        x0,y0=max(0,int(np.floor(x0))),max(0,int(np.floor(y0)))
        x1,y1=min(size,int(np.ceil(x1))),min(size,int(np.ceil(y1)))
        box[y0:y1,x0:x1] = True
        visible=protect_regions(box,[o.mask for o in scene.objects],check,dets)
        fraction=float(visible.sum()/max(box.sum(),1))
        candidates.append(dict(index=i,visible_fraction=fraction))
    viable=[x for x in candidates if x['visible_fraction']>0]
    selected=max(viable,key=lambda x:(x['visible_fraction'],-x['index']))['index'] if viable else None
    return selected,candidates


def generate(kind,eng,vis,scene,seed,cfg,backbone,cache_dir,log,get_objects,diagnose,make_gligen):
    if cfg.R != 1 or cfg.use_energy or cfg.use_region_attn or cfg.use_identity or not cfg.use_scene_cn:
        raise ValueError('Instance screen requires one attempt, scene CN, no energy/region attention/identity')
    t0=time.time();best=make_gligen(seed)
    dets=vis.gdino(best,[o.cls for o in scene.objects],thr=cfg.verify_thr)
    check=consistency(scene,dets,size=cfg.height)
    target,coverage=choose_target(scene,check,dets,cfg.height)
    from .clean_repair import FLAGS as CLEAN_FLAGS
    from .foreground import CONTEXT_METHODS, REFINED_METHODS, predict_foreground, place_foreground
    FLAGS={**CLEAN_FLAGS,**{m:(True,True) for m in CONTEXT_METHODS}}
    clean_flags=FLAGS.get(kind,(False,False))
    count_route=False; extra_ids=[]
    if target is None and (kind in ('instance_repair_count','instance_repair_count_global') or kind in FLAGS):
        from .count_repair import choose_count_target
        target,extra_ids=choose_count_target(scene,check,dets)
        count_route=target is not None
    focused=kind in ('instance_repair_focus','instance_repair_count') or kind in FLAGS
    # Both new arms retain the existing experimental missing-object behavior.
    if not count_route and kind=='instance_repair_count_global':focused=True
    repair_check=check
    if count_route:
        repair_check=dict(check,matched=list(check['matched']))
        repair_check['matched'][target]=(-1,0.)
    repair_steps=cfg.steps-min(max(int(round((1-cfg.repair_strength)*cfg.steps)),0),cfg.steps-1)
    planned=2*cfg.steps+4*cfg.K*cfg.obj_steps+4*repair_steps
    cap=10*cfg.steps
    log.update(experimental=True,route='keep' if target is None else 'count_replacement' if count_route else 'single_instance',target=target,
               target_coverage=coverage,sampling_work_cap=cap,planned_local_work=planned,
               policy_version=2 if kind.startswith('instance_repair_count') else 1,focused=focused,
               count_extra_ids=extra_ids,
               scene_spec=dict(caption=scene.caption,bg=scene.bg,
                               objects=[dict(index=i,cls=o.cls,phrase=o.phrase,box=o.box) for i,o in enumerate(scene.objects)]))
    history=[dict(kind='initial',check=check,detections=dets)]
    diagnose('initial',best)
    if target is not None:
        if planned>cap:
            raise ValueError('Requested instance repair exceeds the fixed sampler-work budget')
        objs=get_objects([target]);placements={}
        prep=Prepared(objs,compose_foreground(scene,objs,cfg,placements=placements) if count_route and kind in REFINED_METHODS else compose_foreground(scene,objs,cfg),None)
        foreground_override=None
        if count_route and kind in REFINED_METHODS:
            probability=predict_foreground(objs[target].img,cache_dir,log)
            foreground_override=place_foreground(objs[target].img,probability,objs[target].mask,objs[target].info,
                                                 placements[target],cfg.height,prep.fg[1][target],hard=kind.endswith('_hard'))
            diagnose('refined_probability',np.repeat((probability*255).round().astype(np.uint8)[...,None],3,axis=2))
            log['foreground_placement']=list(placements[target])
        # Keep proposals in the review export: cache-only images were previously omitted.
        if target < len(objs) and hasattr(objs[target],'img'):
            diagnose('proposal',objs[target].img)
            diagnose('proposal_mask',np.repeat((objs[target].mask.astype(np.uint8)*255)[...,None],3,axis=2))
        if prep.fg is not None:
            diagnose('placed_proposal',prep.fg[0])
            diagnose('placed_proposal_mask',np.repeat((prep.fg[1][target].astype(np.uint8)*255)[...,None],3,axis=2))
        emit=lambda name,img:diagnose('local_'+name,img)
        candidate=repair_scene(eng,scene,prep,cfg,best,repair_check,dets,seed+100003,cfg.lambda0,log,
                               protect_matched=True,replacement_ids=[target],isolate_replacement=True,
                               focus_ids=[target] if focused else None,diagnose=emit,
                               **(dict(replacement_extra_ids=extra_ids) if count_route else {}),
                               **(dict(clean_init=clean_flags[0],clean_blend=clean_flags[1]) if count_route and kind in FLAGS else {}),
                               **(dict(context_anchor=True,foreground_override=foreground_override) if count_route and kind in CONTEXT_METHODS else {}))
        if candidate is None:
            history.append(dict(kind='local',accepted=False,reason=log.get('repair_skip_reason','empty_edit_mask')))
        else:
            diagnose('local_candidate',candidate)
            after_dets=vis.gdino(candidate,[o.cls for o in scene.objects],thr=cfg.verify_thr)
            after=consistency(scene,after_dets,size=cfg.height)
            clips=vis.clip.score([best,candidate],[scene.caption,scene.caption])
            accepted,reason=accept_repair(check,after,float(clips[0]),float(clips[1]))
            if count_route and accepted and after['oce_c'] >= check['oce_c']:
                accepted,reason=False,'count_not_improved'
            history.append(dict(kind='local',accepted=accepted,reason=reason,check=after,detections=after_dets,
                                score=verify_score(after),clip_before=float(clips[0]),clip_after=float(clips[1])))
            if accepted:best=candidate
    wall=time.time()-t0
    log.update(verify=history,tries=len(history),execution_wall_time_s=wall,
               scene_time=wall-log.get('m2_lookup_seconds',0)+log.get('gligen_reused_seconds',0)+log.get('foreground_reused_seconds',0),
               compute_includes_m2=True,peak_gb=_cuda_peak())
    return best
