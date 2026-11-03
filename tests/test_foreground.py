"""Foreground extraction integration: geometry/provenance, not a semantic quality certificate."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from PIL import Image
import test_count_repair
from ocsd.foreground import place_foreground,model_path,METHODS
from ocsd.sketch import CropInfo
from ocsd.runner import Runner
from ocsd.method import object_branch
from ocsd import stages
from ocsd.config import Paths,ExperimentConfig

class Foreground(unittest.TestCase):
    def setUp(self):test_count_repair.Count.setUp(self)

    def test_premultiplied_resize_no_background_bleed_or_probability_truncation(self):
        rgb=np.full((8,8,3),255,np.uint8);rgb[0,0]=0
        probability=np.zeros((8,8),np.float32);probability[0,0]=1
        canvas,alpha=place_foreground(rgb,probability,np.ones((8,8),bool),CropInfo(0,0,4,8),(1,0,0),8,np.ones((8,8),bool))
        self.assertAlmostEqual(float(alpha[0,0]),.25);self.assertTrue((canvas[0,0]==0).all())
        allowed=np.zeros((8,8),bool);allowed[:2,:2]=True
        _,alpha=place_foreground(rgb,np.full((8,8),.6,np.float32),np.ones((8,8),bool),CropInfo(0,0,4,8),(1,0,0),8,allowed)
        self.assertAlmostEqual(float(alpha[0,0]),.6);self.assertFalse(alpha[~allowed].any())
        with self.assertRaises(ValueError):place_foreground(rgb,np.full((8,8),np.nan),allowed,CropInfo(0,0,4,8),(1,0,0),8,allowed)

    def test_bad_model_never_silently_used(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'foreground/birefnet-general-lite.onnx';p.parent.mkdir();p.write_bytes(b'corrupted')
            with patch('urllib.request.urlopen') as net:
                with self.assertRaisesRegex(ValueError,'checksum'):model_path(d)
                net.assert_not_called()

    def test_four_arms_preserve_geometry_and_diffusion_budget(self):
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
            def probability(image,*args):return np.full(image.shape[:2],.75,np.float32)
            with patch('ocsd.runner.load_scene',return_value=self.scene),patch.object(semantic,'_gligen',side_effect=initial) as gen,patch('ocsd.runner.object_branch',side_effect=lambda eng,vis,*a,**kw:object_branch(eng,None,*a,**kw)),patch('ocsd.foreground.predict_foreground',side_effect=probability) as pred:
                done=runner.run_items(items,verbose=False);resume=runner.run_items(items,verbose=False)
            self.assertEqual((done['done'],done['failed'],resume['skipped']),(4,0,4));self.assertEqual(gen.call_count,1);self.assertEqual(pred.call_count,2)
            logs=[];masks=[];seeds={}
            for method in METHODS:
                path=Path(runner.img_path('quickdraw',method,'s',0));d=json.loads(path.with_suffix('.json').read_text());logs.append(d)
                self.assertTrue(d['verify'][-1]['accepted'])
                q=path.parent/'_diagnostics/s_s0';mask=np.array(Image.open(q/'local_edit_mask.png'))[...,0]>0;masks.append(mask)
                candidate=np.array(Image.open(q/'local_candidate.png'));self.assertTrue(np.array_equal(candidate[~mask],self.image[~mask]))
                self.assertFalse(mask[20:30,20:30].any())
                seeds[method]=np.array(Image.open(q/'local_clean_seed.png'))
                self.assertEqual(d['repair_canvas']['context_anchor'],method!=METHODS[0])
                self.assertEqual(d['repair_canvas']['extra_vae_encodes'],int(method==METHODS[0]))
                if 'matte_' in method:
                    self.assertTrue((q/'refined_probability.png').exists());self.assertTrue((q/'local_foreground_probability.png').exists())
            self.assertEqual(len({x['m2_key'] for x in logs}),1)
            self.assertEqual(len({x['standalone_sampling_work'] for x in logs}),1)
            self.assertTrue(all(np.array_equal(m,masks[0]) for m in masks))
            self.assertTrue(np.array_equal(seeds[METHODS[0]],seeds[METHODS[1]]))
            self.assertEqual(logs[2]['foreground_placement'],logs[3]['foreground_placement'])

    def test_four_image_plan_and_report(self):
        import pandas as pd
        from ocsd.semantic_report import report
        with tempfile.TemporaryDirectory() as tmp:
            p=Paths(tmp);e=ExperimentConfig(tier='foreground_screen')
            scenes=[str(Path(tmp)/f'qd_{n}_{c}_000') for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            with patch.object(stages,'list_scenes',return_value=scenes):plan=stages.experiment_plan(p,e)
            self.assertEqual(set(plan[0]['methods']),set(METHODS));self.assertEqual(len(plan[0]['scenes']),1)
            self.assertIn(METHODS[-1],stages._gen_signatures(e)['quickdraw'])
            directory=Path(plan[0]['scenes'][0]);directory.mkdir(parents=True);Image.fromarray(self.scene.sketch).save(directory/'sketch.png')
            rows=[]
            for method in METHODS:
                path=Path(p.outputs)/'quickdraw'/method/f'{directory.name}_s0.png';path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(self.image).save(path);path.with_suffix('.json').write_text(json.dumps({'compute':{},'scene_spec':{'caption':'test'}}))
                rows.append(dict(method=method,sid=directory.name,seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',opr=1.,oce_c=0,miou=.8,ra=1.,clip=30.,time_s=1.,joint_layout_success=1.))
            with patch.object(stages,'experiment_plan',return_value=plan),patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):result=report(p,e)
            self.assertEqual(result['n_images'],4);self.assertEqual(result['research_claim'],'unproven')
            self.assertTrue(pd.read_csv(Path(p.results)/'tables/screen_paired.csv').empty)

if __name__=='__main__':unittest.main(verbosity=2)
