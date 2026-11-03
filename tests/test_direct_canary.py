"""Canary orchestration tests use synthetic pixels, never claim pretrained quality."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from ocsd import direct_canary as canary
from ocsd.direct_instances import scene_condition, sampling_work
from ocsd.provenance import digest


class Canary(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.protocol,self.spec=canary.read_protocol()
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close);self.addCleanup(self.tmp.cleanup)
        for target,value in [('is_available',True),('get_device_properties',SimpleNamespace(name='test GPU',total_memory=24*1024**3)),
                             ('reset_peak_memory_stats',None),('synchronize',None),('max_memory_allocated',1024)]:
            self.stack.enter_context(patch.object(torch.cuda,target,return_value=value))
        self.calls=[]
        def generate(condition,seed,negative,**kwargs):
            self.calls.append((seed,kwargs['scribble']))
            pixels=np.full((512,512,3),30+seed*60+int(kwargs['scribble'])*20,np.uint8)
            return pixels,dict(unet_sample_calls=216)
        self.model=SimpleNamespace(provenance=self.protocol['model_provenance'],generate=generate)
        self.load=self.stack.enter_context(patch('ocsd.direct_instances.load_components',return_value=self.model))

    def run_attempt(self,name):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return canary.run(self.root/'cache',self.root,self.root/name)

    def test_frozen_inputs_budget_and_two_separate_bicycles(self):
        scene=canary.render_scene(self.spec);condition=scene_condition(scene)
        self.assertEqual([o.cls for o in scene.objects],['bicycle','bicycle','bench'])
        self.assertEqual(len(canary.jobs(self.protocol)),4)
        self.assertEqual(4*sampling_work(scene.n),864)
        self.assertEqual(condition,scene_condition(canary.render_scene(self.spec)))
        for obj,item in zip(scene.objects,condition['objects']):
            xy=np.rint(np.array(item['scribble']).reshape(20,2)*512).astype(int)
            self.assertTrue((obj.sketch[xy[:,1],xy[:,0]]==0).all())
        bad=dict(self.protocol,seeds=[0,1,2]);file=self.root/'bad.json';file.write_text(json.dumps(bad))
        with patch.object(canary,'PROTOCOL',file),self.assertRaisesRegex(ValueError,'limited'):
            canary.read_protocol()

    def test_complete_resume_review_originals_and_archive_checksums(self):
        result=self.run_attempt('attempt_a');self.assertEqual(result['status'],'complete')
        self.assertEqual(self.calls,[(0,False),(0,True),(1,False),(1,True)])
        directory=Path(result['directory']);key=json.loads((directory/'review_key.json').read_text())
        for seed in (0,1):
            sheet=np.array(Image.open(directory/f'SO_SANH_seed{seed}.png'))
            for i,label in enumerate('AB',1):
                original=np.array(Image.open(directory/f'{key[str(seed)][label]}_s{seed}.png'))
                self.assertTrue(np.array_equal(original,sheet[150:662,i*512:(i+1)*512]))
        with ZipFile(result['archive']) as z:
            import hashlib
            hashes=json.loads(z.read('checksums.json'))
            self.assertTrue(all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in hashes.items()))
            self.assertIn('run/REVIEW_TEMPLATE.txt',z.namelist())
            self.assertFalse(any(n.endswith('.html') for n in z.namelist()))
        again=self.run_attempt('attempt_b');self.assertEqual(again['directory'],result['directory'])
        self.assertEqual(again['status'],'complete');self.assertEqual(len(self.calls),4);self.assertEqual(self.load.call_count,1)

    def test_failure_keeps_completed_images_and_resumes_only_missing(self):
        original=self.model.generate
        def fail(condition,seed,negative,**kwargs):
            if len(self.calls)==1:raise RuntimeError('test OOM; no retry')
            return original(condition,seed,negative,**kwargs)
        self.model.generate=fail
        failed=self.run_attempt('attempt_a');self.assertEqual(failed['status'],'failed')
        directory=Path(failed['directory']);before=(directory/'direct_box_s0.png').read_bytes()
        self.assertEqual(len(self.calls),1)
        self.model.generate=original
        complete=self.run_attempt('attempt_b');self.assertEqual(complete['status'],'complete')
        self.assertEqual(len(self.calls),4);self.assertEqual(before,(directory/'direct_box_s0.png').read_bytes())
        self.assertEqual(len(list((directory/'attempts').glob('*.json'))),2)

    def test_changed_completed_image_fails_without_regeneration(self):
        result=self.run_attempt('attempt_a');directory=Path(result['directory'])
        (directory/'direct_box_s0.png').write_bytes(b'changed')
        result=self.run_attempt('attempt_b');self.assertEqual(result['status'],'failed')
        self.assertEqual(len(self.calls),4);self.assertEqual(self.load.call_count,1)

    def test_interrupted_run_can_be_packaged_without_model_loading(self):
        result=self.run_attempt('attempt_a');count=self.load.call_count
        with contextlib.redirect_stdout(io.StringIO()):packed=canary.finalize(self.root/'attempt_a',interrupted=True)
        self.assertEqual(packed['status'],'incomplete');self.assertEqual(self.load.call_count,count)
        state=json.loads((Path(packed['directory'])/'run.json').read_text())
        self.assertFalse(state['quality_validated'])
        self.assertEqual(len(state['completed']),4)


if __name__=='__main__':unittest.main()
