"""Count-aware replacement: no quality claim from synthetic CPU tests."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from PIL import Image
import test_instance_repair
from ocsd import instance_repair as I, stages
from ocsd.count_repair import choose_count_target, METHODS
from ocsd.matching import consistency
from ocsd.config import Paths,ExperimentConfig
from ocsd.runner import Runner
from ocsd.method import object_branch

class Count(unittest.TestCase):
    def setUp(self):
        test_instance_repair.Instance.setUp(self)
        self.dets=[dict(cls='cat',box=(2,2,6,12),score=.9),self.dets[1],dict(cls='cat',box=(7,2,12,12),score=.8)]

    def test_route_rejects_duplicates_missing_ambiguous_and_outside(self):
        c=consistency(self.scene,self.dets,size=32)
        self.assertEqual(choose_count_target(self.scene,c,self.dets),(0,[0]))
        dup=self.dets[:2]+[dict(cls='cat',box=(2,2,6,12),score=.8)]
        self.assertIsNone(choose_count_target(self.scene,consistency(self.scene,dup,size=32),dup)[0])
        outside=[self.dets[0],self.dets[1],dict(cls='cat',box=(14,2,18,12),score=.8)]
        self.assertIsNone(choose_count_target(self.scene,consistency(self.scene,outside,size=32),outside)[0])
        self.assertIsNone(choose_count_target(self.scene,consistency(self.scene,self.dets[::2],size=32),self.dets[::2])[0])
        import copy
        ambiguous=copy.deepcopy(self.scene)
        ambiguous.objects[1].cls='cat';ambiguous.objects[1].box=(2,2,12,12)
        candidates=[dict(cls='cat',box=b,score=.9) for b in [(2,2,4,12),(10,2,12,12),(6,2,8,12)]]
        self.assertIsNone(choose_count_target(ambiguous,dict(matched=[(0,.2),(1,.2)],oce_c=1),candidates)[0])

    def test_missing_branch_stays_focused_in_both_new_arms(self):
        good=[dict(cls=o.cls,box=o.box,score=.9) for o in self.scene.objects]
        for method in ('instance_repair_count','instance_repair_count_global'):
            vis=SimpleNamespace(gdino=Mock(side_effect=[good[:1],good]),clip=SimpleNamespace(score=Mock(return_value=[30.,30.])))
            log={}
            with patch.object(I,'compose_foreground',return_value=None),patch.object(I,'repair_scene',return_value=self.image+10) as repair:
                I.generate(method,None,vis,self.scene,0,self.cfg,'sd15',None,log,Mock(return_value=[]),Mock(),lambda _:self.image)
            self.assertEqual(repair.call_args.kwargs['focus_ids'],[1])
            self.assertNotIn('replacement_extra_ids',repair.call_args.kwargs)
            self.assertEqual(log['route'],'single_instance')


    def test_actual_repair_shared_cache_protection_count_guard_and_resume(self):
        import open_clip
        from tiny import make_engine
        from ocsd import semantic
        good=[dict(cls=o.cls,box=o.box,score=.9) for o in self.scene.objects]
        with tempfile.TemporaryDirectory() as tmp:
            eng=make_engine(tmp,str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'),32)
            vis=SimpleNamespace(gdino=Mock(side_effect=[self.dets,good,self.dets,good]),clip=SimpleNamespace(score=Mock(return_value=[30.,30.])))
            methods=['instance_repair_count_global','instance_repair_count']
            runner=Runner(eng,vis,str(Path(tmp)/'out'),variants={},baselines={m:(m,self.cfg) for m in methods})
            def initial(*args):
                eng.compute['gligen_unet_sample_calls']=eng.compute.get('gligen_unet_sample_calls',0)+4
                return self.image.copy()
            items=[('quickdraw','unused',[(m,0) for m in methods])]
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(semantic,'_gligen',side_effect=initial) as gen,patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw:object_branch(eng,None,*a,**kw)):
                done=runner.run_items(items,verbose=False);resume=runner.run_items(items,verbose=False)
            self.assertEqual((done['done'],done['failed'],resume['skipped']),(2,0,2));self.assertEqual(gen.call_count,1)
            logs=[];masks=[]
            for m in methods:
                path=Path(runner.img_path('quickdraw',m,'s',0));log=json.loads(path.with_suffix('.json').read_text());logs.append(log)
                self.assertEqual(log['route'],'count_replacement');self.assertEqual(log['m2_object_ids'],[0]);self.assertTrue(log['verify'][-1]['accepted'])
                diag=path.parent/'_diagnostics/s_s0';mask=np.array(Image.open(diag/'local_edit_mask.png'))[...,0]>0;masks.append(mask)
                candidate=np.array(Image.open(diag/'local_candidate.png'))
                self.assertTrue(np.array_equal(candidate[~mask],self.image[~mask]));self.assertFalse(mask[20:30,20:30].any())
            self.assertTrue(np.array_equal(*masks));self.assertEqual(logs[0]['m2_key'],logs[1]['m2_key']);self.assertEqual(logs[0]['standalone_sampling_work'],logs[1]['standalone_sampling_work'])
            self.assertNotIn('focus_conditioning',logs[0]);self.assertIn('focus_conditioning',logs[1])
        # Better IoU alone cannot accept a count replacement if count is unchanged.
        after=[dict(cls='cat',box=(2,2,12,12),score=.9),good[1],dict(cls='cat',box=(14,2,18,12),score=.8)]
        vis=SimpleNamespace(gdino=Mock(side_effect=[self.dets,after]),clip=SimpleNamespace(score=Mock(return_value=[30.,30.])))
        log={}
        with patch.object(I,'compose_foreground',return_value=None),patch.object(I,'repair_scene',return_value=self.image+10):
            result=I.generate('instance_repair_count',None,vis,self.scene,0,self.cfg,'sd15',None,log,Mock(return_value=[]),Mock(),lambda _:self.image)
        self.assertIs(result,self.image);self.assertEqual(log['verify'][-1]['reason'],'count_not_improved')

    def test_plan_is_four_image_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Paths(tmp);e=ExperimentConfig(tier='count_screen')
            scenes=[str(Path(tmp)/f'qd_{n}_{c}_000') for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            with patch.object(stages,'list_scenes',return_value=scenes):plan=stages.experiment_plan(p,e)
            self.assertEqual(len(plan[0]['scenes']),1);self.assertEqual(set(plan[0]['methods']),set(METHODS));self.assertEqual(e.seeds,[0])
            self.assertIn('instance_repair_count',stages._gen_signatures(e)['quickdraw'])
            import pandas as pd
            from ocsd.semantic_report import report
            scene_dir=Path(plan[0]['scenes'][0]);scene_dir.mkdir(parents=True)
            Image.fromarray(self.scene.sketch).save(scene_dir/'sketch.png')
            rows=[]
            for method in METHODS:
                target=Path(p.outputs)/'quickdraw'/method/f'{scene_dir.name}_s0.png';target.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(self.image).save(target)
                target.with_suffix('.json').write_text(json.dumps(dict(compute={},scene_spec=dict(caption='test'))))
                rows.append(dict(method=method,sid=scene_dir.name,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',opr=1.,oce_c=0,miou=.8,ra=1.,clip=30.,time_s=1.,joint_layout_success=1.))
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):
                result=report(p,e)
            self.assertEqual(result['n_images'],4);self.assertEqual(result['research_claim'],'unproven')
            self.assertTrue(pd.read_csv(Path(p.results)/'tables/screen_paired.csv').empty)


if __name__=='__main__':unittest.main(verbosity=2)
