"""Isolate conditioning changes without pretrained quality claims."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from PIL import Image
import test_screening
from ocsd import instance_repair as I, stages
from ocsd.config import ExperimentConfig,Paths
from ocsd.matching import consistency
from ocsd.method import Prepared,repair_scene,object_branch
from ocsd.runner import Runner

class Instance(unittest.TestCase):
    def setUp(self):
        test_screening.Screening.setUp(self)
        for obj in self.scene.objects:
            obj.sketch=np.where(obj.mask,0,255).astype(np.uint8)
        self.scene.sketch=np.minimum.reduce([o.sketch for o in self.scene.objects])
        self.cfg=self.cfg.replace(R=1,steps=2,obj_steps=2,K=1,bo_n=5,use_identity=False,
                                  use_energy=False,use_region_attn=False,use_scene_cn=True)

    def test_joint_endpoint_does_not_hide_one_bad_object(self):
        from ocsd.matching import joint_layout_success
        c=consistency(self.scene,self.dets,size=32)
        self.assertEqual(joint_layout_success(c),1.)
        c['matched'][1]=(1,.49)
        self.assertEqual(joint_layout_success(c),0.)
        c['matched'][1]=(1,1.);c['oce_c']=1
        self.assertEqual(joint_layout_success(c),0.)
        c['oce_c']=0;c['ra']=0.
        self.assertEqual(joint_layout_success(c),0.)

    def test_target_choice_is_geometry_only_and_stable(self):
        check=consistency(self.scene,[],size=32)
        target,coverage=I.choose_target(self.scene,check,[],32)
        self.assertEqual(target,0);self.assertEqual(len(coverage),2)
        check=consistency(self.scene,self.dets[:1],size=32)
        self.assertEqual(I.choose_target(self.scene,check,self.dets[:1],32)[0],1)
        dets=[dict(cls='cat',box=(0,0,32,32),score=.99)]
        check=dict(matched=[(0,1.),(-1,0.)])
        self.assertIsNone(I.choose_target(self.scene,check,dets,32)[0])

    def test_real_pair_keeps_masks_seed_work_and_resume_equal(self):
        import open_clip
        from tiny import make_engine
        from ocsd import semantic
        with tempfile.TemporaryDirectory() as d:
            eng=make_engine(d,str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'),32)
            vis=SimpleNamespace(gdino=Mock(side_effect=[self.dets[:1],self.dets,self.dets[:1],self.dets]),
                                clip=SimpleNamespace(score=Mock(return_value=np.array([30.,30.]))))
            methods=['instance_repair','instance_repair_focus']
            runner=Runner(eng,vis,str(Path(d)/'out'),variants={},baselines={m:(m,self.cfg) for m in methods})
            def initial(*args):
                eng.compute['gligen_unet_sample_calls']=eng.compute.get('gligen_unet_sample_calls',0)+4
                return self.image.copy()
            items=[('quickdraw','unused',[(m,0) for m in methods])]
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(semantic,'_gligen',side_effect=initial) as gen,\
                 patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw:object_branch(eng,None,*a,**kw)):
                first=runner.run_items(items,verbose=False);second=runner.run_items(items,verbose=False)
            self.assertEqual((first['done'],first['failed'],second['skipped']),(2,0,2));self.assertEqual(gen.call_count,1)
            metas=[];masks=[]
            for m in methods:
                p=Path(runner.img_path('quickdraw',m,'s',0));meta=json.loads(p.with_suffix('.json').read_text());metas.append(meta)
                self.assertTrue(meta['verify'][-1]['accepted']);self.assertEqual(meta['m2_object_ids'],[1])
                mask=np.array(Image.open(p.parent/'_diagnostics/s_s0/local_edit_mask.png'))[...,0]>0;masks.append(mask)
                candidate=np.array(Image.open(p.parent/'_diagnostics/s_s0/local_candidate.png'))
                self.assertTrue(np.array_equal(candidate[~mask],self.image[~mask]))
            self.assertTrue(np.array_equal(*masks));self.assertFalse(masks[0][2:12,2:12].any())
            self.assertEqual(metas[0]['standalone_sampling_work'],metas[1]['standalone_sampling_work'])
            self.assertEqual(metas[0]['m2_key'],metas[1]['m2_key'])
            self.assertNotIn('focus_conditioning',metas[0])
            focus=metas[1]['focus_conditioning'];self.assertEqual(focus['original_index'],1)
            self.assertIn('single dog',focus['prompt']);self.assertNotIn('cat',focus['prompt']);self.assertIn('cat',focus['negative_prompt'])
            self.assertEqual(focus['control'],'target_sketch_only')
            controls=[]
            for m in methods:
                base=Path(runner.img_path('quickdraw',m,'s',0)).parent/'_diagnostics/s_s0'
                controls.append(np.array(Image.open(base/'local_conditioning_control.png')))
            self.assertTrue(controls[0][2:12,2:12].all())
            self.assertFalse(controls[1][2:12,2:12].any())
            self.assertTrue(controls[1][20:30,20:30].all())

            # Unselected missing objects must also stay outside the editable region.
            self.scene.objects[0].cls='bird'
            self.scene.objects[1].cls='duck';self.scene.objects[1].phrase='duck'
            empty=consistency(self.scene,[],size=32);diagnostics={};focus_log={}
            repair_scene(eng,self.scene,Prepared(None,None,None),self.cfg,self.image,empty,[],7,self.cfg.lambda0,
                         protect_matched=True,replacement_ids=[1],focus_ids=[1],isolate_replacement=True,log=focus_log,
                         diagnose=lambda name,img:diagnostics.setdefault(name,img))
            self.assertNotIn('bird',focus_log['focus_conditioning']['negative_prompt'])
            self.assertFalse(diagnostics['edit_mask'][2:12,2:12].any())
            self.assertTrue(diagnostics['edit_mask'][20:30,20:30].all())

    def test_rejection_keeps_initial_and_saves_candidate(self):
        vis=SimpleNamespace(gdino=Mock(return_value=self.dets[:1]),clip=SimpleNamespace(score=Mock(return_value=[30.,30.])))
        log={};emit=Mock();get=Mock(return_value=[None,object()])
        with patch.object(I,'compose_foreground',return_value=None),patch.object(I,'repair_scene',return_value=self.image+30):
            out=I.generate('instance_repair_focus',None,vis,self.scene,0,self.cfg,'sd15',None,log,get,emit,lambda _:self.image)
        self.assertIs(out,self.image);get.assert_called_once_with([1]);self.assertFalse(log['verify'][-1]['accepted'])
        self.assertEqual(emit.call_args.args[0],'local_candidate')

    def test_plan_report_requires_sketch_and_has_ablation(self):
        from ocsd.semantic_report import report
        with tempfile.TemporaryDirectory() as d:
            p=Paths(d);e=ExperimentConfig(tier='instance_screen')
            scenes=[]
            for n in ('1','3','5','8p'):
                for c in ('simple','medium','complex'):
                    path=Path(d)/'bench'/f'qd_{n}_{c}_000';path.mkdir(parents=True)
                    Image.fromarray(self.scene.sketch).save(path/'sketch.png');scenes.append(str(path))
            with patch.object(stages,'list_scenes',return_value=scenes):plan=stages.experiment_plan(p,e)
            self.assertEqual(set(plan[0]['methods']),set(I.METHODS));self.assertEqual(len(scenes)*len(plan[0]['methods']),48)
            self.assertIn('instance_repair_focus',stages._gen_signatures(e)['quickdraw'])
            rows=[]
            for m in I.METHODS:
                for directory in scenes:
                    sid=Path(directory).name;out=Path(p.outputs)/'quickdraw'/m/f'{sid}_s0.png';out.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(self.image).save(out);out.with_suffix('.json').write_text(json.dumps({'compute':{},'scene_spec':{'caption':'test'}}))
                    rows.append(dict(method=m,sid=sid,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',opr=.5,oce_c=1,miou=.4,ra=.5,clip=30,time_s=2,joint_layout_success=0.))
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):
                result=report(p,e);self.assertEqual(result['research_claim'],'unproven')
                (Path(scenes[0])/'sketch.png').unlink()
                with self.assertRaisesRegex(FileNotFoundError,'sketch'):report(p,e)
            ratings=pd.read_csv(Path(p.results)/'review_blind/ratings.csv');self.assertIn('shape',ratings);self.assertIn('relations',ratings)
            paired=pd.read_csv(Path(p.results)/'tables/screen_paired.csv');self.assertEqual(set(paired.ref),{'instance_repair_focus'})

if __name__=='__main__':unittest.main(verbosity=2)
