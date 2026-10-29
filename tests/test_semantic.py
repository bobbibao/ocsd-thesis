"""Prospective semantic routing contract on CPU; no pretrained quality evidence."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
import test_screening
from ocsd import semantic as S, stages
from ocsd.config import ExperimentConfig, Paths
from ocsd.matching import consistency


class Semantic(unittest.TestCase):
    def setUp(self):
        test_screening.Screening.setUp(self)
        self.cfg=self.cfg.replace(R=1,steps=30,obj_steps=20,K=2,bo_n=5,use_identity=False,use_scene_cn=True,
                                  use_energy=False,use_region_attn=False)

    def state(self, missing=False, bg=False, attr=False):
        c=consistency(self.scene,self.dets[:1] if missing else self.dets,size=32)
        return dict(check=c,detections=self.dets[:1] if missing else self.dets,missing_ids=[1] if missing else [],
                    attributes=[dict(object_id=1,attribute='brown',margin=-1 if attr else 1)],
                    background=dict(margin=-1 if bg else 1),caption_score=30.)

    def test_routes_and_work_budget(self):
        self.assertEqual(S.route(self.state(),self.cfg)[0],'keep')
        action,targets,work,cap=S.route(self.state(missing=True),self.cfg)
        self.assertEqual((action,targets,work,cap),('local',[1],280,300))
        self.assertEqual(S.route(self.state(bg=True),self.cfg)[0],'global')
        self.assertEqual(S.route(self.state(attr=True),self.cfg)[0],'local')
        self.assertEqual(S.route(self.state(attr=True),self.cfg.replace(K=3))[0],'global')
        state=self.state(missing=True);state['attributes']=[dict(object_id=0,attribute='black',margin=-1)]
        self.assertEqual(S.route(state,self.cfg)[0],'global')

    def test_accept_requires_semantic_or_layout_gain_and_protects_objects(self):
        before=self.state(attr=True);after=self.state()
        self.assertTrue(S.accept(before,after)[0])
        self.assertEqual(S.accept(after,after),(False,'no_proxy_gain'))
        self.assertEqual(S.accept(after,before),(False,'new_semantic_issue'))
        self.assertEqual(S.accept(before,self.state(missing=True))[1],'lost_object')
        bad=copy.deepcopy(after);bad['caption_score']=28
        self.assertEqual(S.accept(before,bad)[1],'caption_regressed')
        bad=copy.deepcopy(after);bad['check']['miou']=0.
        self.assertEqual(S.accept(before,bad)[1],'count_or_layout_regressed')

    def test_probe_records_contrasts_and_unsupported_attributes(self):
        self.scene.objects[1].phrase='brown dog'
        class Clip:
            def image_emb(self, images):return torch.tensor([[1.,0.]]*len(images))
            def text_emb(self,texts):
                return torch.tensor([[1.,0.] if t in ('a photo park','a photo of a brown dog',self_caption) else [0.,1.] for t in texts])
        self_caption=self.scene.caption
        vis=SimpleNamespace(gdino=Mock(return_value=self.dets),clip=Clip())
        probe=S.Probe(vis,self.scene,self.cfg)
        out=probe(self.image)
        self.assertEqual(len(out['attributes']),1)
        self.assertEqual(out['attributes'][0]['preferred'],'brown dog')
        self.assertEqual(out['background']['preferred'],'park')
        self.assertFalse(S.issues(out)[1])
        self.assertEqual(out['missing_ids'],[])

    def test_fixed_global_pool_and_guarded_local_rejection(self):
        for kind in ('gligen_layout5','gligen_semantic5','semantic_repair'):
            initial=self.state(bg=True);better=self.state()
            probe=Mock(side_effect=[initial]+[better]*4);get=Mock();emit=Mock();log={}
            with patch.object(S,'Probe',return_value=probe),patch.object(S,'_gligen',return_value=self.image) as gligen:
                S.generate(kind,None,None,self.scene,7,self.cfg,'sd15',None,log,get,emit)
            self.assertEqual([c.args[2] for c in gligen.call_args_list],[7+100003*i for i in range(5)])
            get.assert_not_called();self.assertEqual(len(log['verify']),5)
        before=self.state(attr=True);bad=self.state(missing=True)
        get=Mock(return_value=[None,object()]);log={};emit=Mock()
        with patch.object(S,'Probe',return_value=Mock(side_effect=[before,bad])),patch.object(S,'_gligen',return_value=self.image),\
             patch.object(S,'compose_foreground',return_value=None),patch.object(S,'repair_scene',return_value=self.image+30) as repair:
            out=S.generate('semantic_repair',None,None,self.scene,0,self.cfg,'sd15',None,log,get,emit)
        self.assertIs(out,self.image);get.assert_called_once_with([1]);self.assertEqual(log['verify'][-1]['reason'],'lost_object')
        self.assertEqual(repair.call_args.kwargs['replacement_ids'],[1])
        self.assertEqual(repair.call_args.args[5]['matched'][1],(-1,0.))
        self.assertEqual(repair.call_args.args[5]['matched'][0],before['check']['matched'][0])
        self.assertEqual(emit.call_args.args[0],'local_candidate')

    def test_real_tiny_replacement_mask_protects_other_objects_and_resumes(self):
        import open_clip
        from tiny import make_engine
        from ocsd.runner import Runner
        from ocsd.method import object_branch
        with tempfile.TemporaryDirectory() as d:
            eng=make_engine(d,str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'),32)
            cfg=self.cfg.replace(steps=2,obj_steps=2,K=1)
            vis=SimpleNamespace(gdino=Mock(return_value=self.dets),clip=SimpleNamespace(score=Mock(return_value=[30.])))
            runner=Runner(eng,vis,str(Path(d)/'out'),variants={},baselines={'semantic_repair':('semantic_repair',cfg)})
            probe=Mock(side_effect=[self.state(attr=True),self.state()])
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(S,'Probe',return_value=probe),\
                 patch.object(S,'_gligen',return_value=self.image),\
                 patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw:object_branch(eng,None,*a,**kw)):
                items=[('quickdraw','unused',[('semantic_repair',0)])]
                first=runner.run_items(items,verbose=False);second=runner.run_items(items,verbose=False)
            self.assertEqual((first['done'],first['failed'],second['skipped']),(1,0,1))
            path=Path(runner.img_path('quickdraw','semantic_repair','s',0));meta=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(meta['route'],'local');self.assertTrue(meta['verify'][-1]['accepted'])
            from PIL import Image
            mask=np.array(Image.open(path.parent/'_diagnostics/s_s0/local_edit_mask.png'))[...,0]>0
            candidate=np.array(Image.open(path.parent/'_diagnostics/s_s0/local_candidate.png'))
            self.assertTrue(mask[20:30,20:30].all());self.assertFalse(mask[2:12,2:12].any())
            self.assertTrue(np.array_equal(candidate[~mask],self.image[~mask]))
            self.assertLessEqual(meta['standalone_sampling_work'],meta['sampling_work_cap'])
            # A cache hit must still be charged the original M2 sampler work.
            path.with_suffix('.json').unlink()
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(S,'Probe',return_value=Mock(side_effect=[self.state(attr=True),self.state()])),\
                 patch.object(S,'_gligen',return_value=self.image),patch('ocsd.runner.object_branch',side_effect=AssertionError('cache miss')):
                cached=runner.run_items(items,verbose=False)
            self.assertEqual(cached['done'],1)
            reused=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(reused['standalone_sampling_work'],meta['standalone_sampling_work'])
            self.assertEqual(reused['m2_compute_in_this_call'].get('unet_sample_calls',0),0)
            # Two five-sample controls share five actual GLIGEN calls, but each is charged all five.
            baselines={m:(m,cfg) for m in ('gligen_layout5','gligen_semantic5')}
            control=Runner(eng,vis,str(Path(d)/'controls'),variants={},baselines=baselines)
            def fake_gligen(*args):
                eng.compute['gligen_unet_sample_calls']=eng.compute.get('gligen_unet_sample_calls',0)+2*cfg.steps
                return self.image.copy()
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(S,'Probe',return_value=Mock(side_effect=lambda _:self.state(bg=True))),\
                 patch.object(S,'_gligen',side_effect=fake_gligen) as gligen:
                result=control.run_items([('quickdraw','unused',[(m,0) for m in baselines])],verbose=False)
            self.assertEqual(result['done'],2);self.assertEqual(gligen.call_count,5)
            cached_meta=json.loads(Path(control.img_path('quickdraw','gligen_semantic5','s',0)).with_suffix('.json').read_text())
            self.assertEqual(cached_meta['standalone_sampling_work'],20)
            self.assertEqual(cached_meta['compute']['gligen_unet_sample_calls'],0)
            self.assertEqual(cached_meta['gligen_compute_cached']['gligen_unet_sample_calls'],20)



    def test_plan_report_blinding_and_complete_population(self):
        from ocsd.semantic_report import report
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            paths=Paths(d);e=ExperimentConfig(tier='semantic_screen')
            scenes=[f'/bench/qd_{n}_{c}_000' for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            with patch.object(stages,'list_scenes',return_value=scenes):plan=stages.experiment_plan(paths,e)
            self.assertEqual(len(plan[0]['methods'])*len(plan[0]['scenes']),48)
            self.assertEqual(set(plan[0]['methods']),set(S.METHODS))
            self.assertIn('semantic_repair',stages._gen_signatures(e)['quickdraw'])
            rows=[]
            for method in S.METHODS:
                for scene in scenes:
                    sid=Path(scene).name;p=Path(paths.outputs)/'quickdraw'/method/f'{sid}_s0.png';p.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(self.image).save(p)
                    p.with_suffix('.json').write_text(json.dumps(dict(compute={'unet_sample_calls':10},m2_compute={'unet_sample_calls':6},m2_compute_in_this_call={},scene_spec={'caption':'a cat'})))
                    rows.append(dict(method=method,sid=sid,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',opr=.5,oce_c=1,miou=.4,ra=.5,clip=30,time_s=2,m2_s=1))
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):
                result=report(paths,e)
            self.assertEqual(result['decision'],'requires_blind_review')
            key=json.loads((Path(paths.results)/'review_key.json').read_text())
            self.assertEqual(len(key),12);self.assertTrue(all(set(v.values())==set(S.METHODS) for v in key.values()))
            ratings=pd.read_csv(Path(paths.results)/'review_blind/ratings.csv')
            self.assertEqual(len(ratings),48);self.assertTrue(ratings.presence.isna().all());self.assertNotIn('method',ratings)
            work=pd.read_csv(Path(paths.results)/'tables/sampling_work.csv');self.assertTrue((work.sampling_work==16).all())
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows[:-1])):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):report(paths,e)

if __name__=='__main__':unittest.main(verbosity=2)
