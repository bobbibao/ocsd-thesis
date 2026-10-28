"""Sparse M2 invariants, real tiny sampling/cache, and diagnostics; no quality claims."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import test_screening
from ocsd import screening as S, stages
from ocsd.config import Paths, ExperimentConfig
from ocsd.method import object_branch, compose_foreground
from ocsd.runner import Runner


class Sparse(unittest.TestCase):
    def setUp(self):
        test_screening.Screening.setUp(self)

    def test_subset_keeps_original_seeds_cache_and_indices(self):
        import open_clip
        from tiny import make_engine
        with tempfile.TemporaryDirectory() as d:
            eng = make_engine(d, str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'), 32)
            cfg = self.cfg.replace(steps=2, obj_steps=2, K=2, m2_per_seed=True, placement_search=False)
            full = object_branch(eng, None, self.scene, cfg, seed=7)
            start = dict(eng.compute)
            subset = object_branch(eng, None, self.scene, cfg, seed=7, object_indices=[1])
            self.assertTrue(np.array_equal(full[1].img, subset[0].img))
            self.assertEqual(eng.compute_delta(start)['unet_calls'], 2)
            runner = Runner(eng, None, str(Path(d)/'out'), variants={}, baselines={})
            key, sparse, elapsed = runner._objects('quickdraw', self.scene, cfg, 7, [1])
            self.assertIsNone(sparse[0]); self.assertTrue(np.array_equal(sparse[1].img, full[1].img))
            with patch('ocsd.runner.object_branch', side_effect=AssertionError('cache miss')):
                key2, loaded, elapsed2 = runner._objects('quickdraw', self.scene, cfg, 7, [1])
            self.assertEqual((key,elapsed),(key2,elapsed2))
            self.assertTrue(np.array_equal(loaded[1].img, sparse[1].img))
            # Packed obj_00 is original object 1; ID-Sim must crop object 1, not object 0.
            from ocsd.metrics import _id_sim
            image = self.image.copy()
            image[self.scene.objects[0].mask] = [200,0,0]
            image[self.scene.objects[1].mask] = [0,200,0]
            logpath = Path(runner.img_path('quickdraw','grounded_repair_sparse','s',7)).with_suffix('.json')
            logpath.parent.mkdir(parents=True,exist_ok=True)
            logpath.write_text(json.dumps({'m2_key':key}))
            calls=[]
            def emb(images):
                calls.append(images)
                return np.ones((len(images),2)) / np.sqrt(2)
            score=_id_sim(runner.out_root,'quickdraw','grounded_repair_sparse',self.scene,image,
                          SimpleNamespace(dino=SimpleNamespace(emb=emb)),7)
            self.assertAlmostEqual(score,1.)
            self.assertEqual(len(calls[1]),1)
            self.assertEqual(calls[1][0][...,0].max(),0)
            self.assertEqual(calls[1][0][...,1].max(),200)
            other, _, _ = runner._objects('quickdraw', self.scene, cfg, 7, [0])
            self.assertNotEqual(other,key)
            fg = compose_foreground(self.scene, loaded, cfg)
            self.assertEqual(len(fg[1]),2); self.assertFalse(fg[1][0].any()); self.assertTrue(fg[1][1].any())
            # Corrupt packed cache: it must regenerate, not return an apparently valid subset.
            Path(runner.obj_dir('quickdraw',key,self.scene.sid),'obj_00.png').write_bytes(b'broken')
            with patch('ocsd.runner.object_branch', wraps=object_branch) as generate:
                runner._objects('quickdraw',self.scene,cfg,7,[1]); self.assertEqual(generate.call_count,1)
            with self.assertRaises(ValueError): object_branch(eng,None,self.scene,cfg,object_indices=[1,1])

    def test_missing_ids_and_rejected_candidate_are_recorded(self):
        vis = SimpleNamespace(gdino=Mock(side_effect=[self.dets[:1],self.dets[:1]]),
                              clip=SimpleNamespace(score=Mock(return_value=[30.])))
        get=Mock(return_value=[None,object()]); emit=Mock(); log={}; candidate=self.image+50
        with patch.object(S,'_gligen',return_value=self.image), \
             patch.object(S,'compose_foreground',return_value=('fg',)), \
             patch.object(S,'repair_scene',return_value=candidate):
            out=S.grounded_repair(None,vis,self.scene,0,self.cfg,'sd15',None,log,get,sparse=True,diagnose=emit)
        get.assert_called_once_with([1]); self.assertIs(out,self.image)
        self.assertFalse(log['verify'][-1]['accepted'])
        self.assertEqual(log['verify'][-1]['missing_ids'],[1])
        self.assertEqual([c.args[0] for c in emit.call_args_list],['attempt_1_before','attempt_1_candidate'])
        self.assertTrue(np.array_equal(emit.call_args_list[-1].args[1],candidate))

    def test_real_sparse_runner_diagnostics_and_resume(self):
        import open_clip
        from tiny import make_engine
        with tempfile.TemporaryDirectory() as d:
            eng=make_engine(d,str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'),32)
            cfg=self.cfg.replace(steps=2,obj_steps=2,K=1,R=1,use_energy=False,use_region_attn=False)
            vis=SimpleNamespace(gdino=Mock(return_value=self.dets[:1]),clip=SimpleNamespace(score=Mock(return_value=[30.])))
            runner=Runner(eng,vis,str(Path(d)/'out'),variants={},baselines={'grounded_repair_sparse':('grounded_repair_sparse',cfg)})
            # Real M2 and repair; skip pretrained segmentation using only the object-branch test double.
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(S,'_gligen',return_value=self.image), \
                 patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw: object_branch(eng,None,*a,**kw)):
                items=[('quickdraw','unused',[('grounded_repair_sparse',0)])]
                a=runner.run_items(items,verbose=False); b=runner.run_items(items,verbose=False)
            self.assertEqual((a['done'],a['failed'],b['skipped']),(1,0,1))
            p=Path(runner.img_path('quickdraw','grounded_repair_sparse','s',0)).with_suffix('.json')
            meta=json.loads(p.read_text()); self.assertEqual(meta['m2_object_ids'],[1])
            self.assertEqual(meta['m2_candidate_count'],1)
            self.assertEqual(len(meta['diagnostics']),4)
            self.assertEqual(meta['repair_regions'][0]['missing_ids'],[1])
            # Removing evidence invalidates resume rather than silently declaring the run complete.
            (Path(runner.out_root)/next(iter(meta['diagnostics']))).unlink()
            self.assertFalse(runner.valid_image('quickdraw','grounded_repair_sparse',self.scene,0))

    def test_48_image_plan_and_cost_report(self):
        with tempfile.TemporaryDirectory() as d:
            paths=Paths(d); Path(paths.results).mkdir(parents=True)
            scenes=[f'/bench/qd_{n}_{c}_000' for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            e=ExperimentConfig(tier='repair_screen')
            with patch.object(stages,'list_scenes',return_value=scenes):
                plan=stages.experiment_plan(paths,e)
            self.assertEqual(len(plan[0]['scenes'])*len(plan[0]['methods']),48)
            self.assertIn('grounded_repair_sparse',stages._gen_signatures(e)['quickdraw'])
            rows=[dict(method=m,sid=Path(s).name,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',
                       opr=.5,miou=.4,oce_c=1,ra=.5,clip=30,time_s=2,m2_s=1 if m.endswith('sparse') else 4)
                  for m in S.REPAIR_METHODS for s in scenes]
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):
                result=S.screen_report(paths,e)
            self.assertEqual(result['decision'],'review_sparse_quality_and_cost')
            self.assertEqual(result['research_claim'],'unproven')
            paired=pd.read_csv(Path(paths.results)/'tables/screen_paired.csv')
            self.assertEqual(set(paired.ref),{'grounded_repair_sparse'})
            self.assertIn('grounded_repair',set(paired.method))

if __name__=='__main__': unittest.main(verbosity=2)
