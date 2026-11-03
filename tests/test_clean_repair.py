"""Test clean-canvas causality and real tiny diffusion ablation without claiming image quality."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from PIL import Image
import test_count_repair
from ocsd.clean_repair import clean_canvas, METHODS, FLAGS
from ocsd.runner import Runner
from ocsd.method import object_branch
from ocsd import stages
from ocsd.config import Paths,ExperimentConfig

class Clean(unittest.TestCase):
    def setUp(self):test_count_repair.Count.setUp(self)

    def test_canvas_removes_old_content_and_preserves_context(self):
        a=np.full((32,32,3),80,np.uint8);mask=np.zeros((32,32),bool);mask[6:24,6:24]=True
        b=a.copy();a[mask]=20;b[mask]=220
        fg=np.full_like(a,150);fm=np.zeros_like(mask);fm[10:14,10:14]=True
        ca,bg=clean_canvas(a,mask,fg,fm);cb,_=clean_canvas(b,mask,fg,fm)
        self.assertTrue(np.array_equal(ca,cb)) # changed old region has no causal path into seed
        self.assertTrue(np.array_equal(ca[~mask],a[~mask]));self.assertTrue((ca[fm]==150).all())
        for bad in [np.zeros_like(mask),np.ones_like(mask)]:
            with self.assertRaises(ValueError):clean_canvas(a,bad,fg,fm)
        with self.assertRaises(ValueError):clean_canvas(a,mask,fg,np.zeros_like(mask))

    def test_four_real_arms_masks_work_noise_and_diagnostics(self):
        import open_clip
        from tiny import make_engine
        from ocsd import semantic
        good=[dict(cls=o.cls,box=o.box,score=.9) for o in self.scene.objects]
        with tempfile.TemporaryDirectory() as tmp:
            eng=make_engine(tmp,str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'),32)
            vis=SimpleNamespace(gdino=Mock(side_effect=[self.dets,good]*4),clip=SimpleNamespace(score=Mock(return_value=[30.,30.])))
            runner=Runner(eng,vis,str(Path(tmp)/'out'),variants={},baselines={m:(m,self.cfg) for m in METHODS})
            def initial(*args):
                eng.compute['gligen_unet_sample_calls']=eng.compute.get('gligen_unet_sample_calls',0)+4
                return self.image.copy()
            items=[('quickdraw','unused',[(m,0) for m in METHODS])]
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(semantic,'_gligen',side_effect=initial) as gen,patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw:object_branch(eng,None,*a,**kw)):
                done=runner.run_items(items,verbose=False);resume=runner.run_items(items,verbose=False)
            self.assertEqual((done['done'],done['failed'],resume['skipped']),(4,0,4));self.assertEqual(gen.call_count,1)
            logs={};masks={};decoded={}
            for method in METHODS:
                path=Path(runner.img_path('quickdraw',method,'s',0));log=json.loads(path.with_suffix('.json').read_text());logs[method]=log
                self.assertTrue(log['verify'][-1]['accepted'])
                diag=path.parent/'_diagnostics/s_s0';mask=np.array(Image.open(diag/'local_edit_mask.png'))[...,0]>0;masks[method]=mask
                candidate=np.array(Image.open(diag/'local_candidate.png'));decoded[method]=np.array(Image.open(diag/'local_decoded_candidate.png'))
                self.assertTrue(np.array_equal(candidate[~mask],self.image[~mask]));self.assertFalse(mask[20:30,20:30].any())
                self.assertTrue((diag/'proposal.png').is_file());self.assertTrue((diag/'placed_proposal_mask.png').is_file())
                flags=FLAGS.get(method,(False,False));self.assertEqual((log['repair_canvas']['clean_init'],log['repair_canvas']['clean_blend']),flags)
                self.assertEqual(log['repair_canvas']['extra_vae_encodes'],int(flags[0]))
                if any(flags):
                    seed=np.array(Image.open(diag/'local_clean_seed.png'));self.assertTrue(np.array_equal(seed[~mask],self.image[~mask]))
            self.assertEqual(len({d['m2_key'] for d in logs.values()}),1)
            self.assertEqual(len({d['standalone_sampling_work'] for d in logs.values()}),1)
            self.assertTrue(all(np.array_equal(v,next(iter(masks.values()))) for v in masks.values()))
            # Blend-only changes no diffusion trajectory; clean init is shared by init/both.
            self.assertTrue(np.array_equal(decoded[METHODS[0]],decoded[METHODS[2]]))
            self.assertTrue(np.array_equal(decoded[METHODS[1]],decoded[METHODS[3]]))

    def test_plan_and_report(self):
        import pandas as pd
        from ocsd.semantic_report import report
        with tempfile.TemporaryDirectory() as tmp:
            p=Paths(tmp);e=ExperimentConfig(tier='clean_screen')
            scenes=[str(Path(tmp)/f'qd_{n}_{c}_000') for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            with patch.object(stages,'list_scenes',return_value=scenes):plan=stages.experiment_plan(p,e)
            self.assertEqual(set(plan[0]['methods']),set(METHODS));self.assertEqual(len(plan[0]['scenes']),1)
            self.assertIn(METHODS[-1],stages._gen_signatures(e)['quickdraw'])
            directory=Path(plan[0]['scenes'][0]);directory.mkdir(parents=True);Image.fromarray(self.scene.sketch).save(directory/'sketch.png')
            rows=[]
            for m in METHODS:
                path=Path(p.outputs)/'quickdraw'/m/f'{directory.name}_s0.png';path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(self.image).save(path);path.with_suffix('.json').write_text(json.dumps({'compute':{},'scene_spec':{'caption':'test'}}))
                rows.append(dict(method=m,sid=directory.name,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',opr=1.,oce_c=0,miou=.8,ra=1.,clip=30.,time_s=1.,joint_layout_success=1.))
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):result=report(p,e)
            self.assertEqual(result['n_images'],4);self.assertEqual(result['research_claim'],'unproven')
            self.assertTrue(pd.read_csv(Path(p.results)/'tables/screen_paired.csv').empty)
            self.assertFalse(any(c.endswith(('_lo','_hi')) for c in pd.read_csv(Path(p.results)/'tables/screen_means.csv').columns))

if __name__=='__main__':unittest.main(verbosity=2)
