"""Development-study orchestration tests: synthetic pixels, no pretrained-quality claim."""
import contextlib
import hashlib
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
from ocsd import direct_dev as dev
from ocsd.direct_canary import render_scene
from ocsd.direct_instances import scene_condition, sampling_work


class Dev(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.protocol, self.scenes = dev.read_protocol()
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close); self.addCleanup(self.tmp.cleanup)
        for target, value in [('is_available', True), ('get_device_properties', SimpleNamespace(name='test GPU', total_memory=24*1024**3)),
                              ('reset_peak_memory_stats', None), ('synchronize', None), ('max_memory_allocated', 1024)]:
            self.stack.enter_context(patch.object(torch.cuda, target, return_value=value))
        self.stack.enter_context(patch.object(torch.backends.cudnn, 'version', return_value=1))
        self.calls = []
        def generate(condition, seed, negative, **kw):
            self.calls.append((condition['sid'], seed, kw['scribble'], kw['fusion']))
            value = (len(self.calls)*7) % 250
            return np.full((512, 512, 3), value, np.uint8), dict(unet_sample_calls=216)
        self.model = SimpleNamespace(provenance=self.protocol['model_provenance'], generate=generate)
        self.load = self.stack.enter_context(patch('ocsd.direct_instances.load_components', return_value=self.model))

    def run_attempt(self, name):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return dev.run(self.root/'cache', self.root, self.root/name)

    def test_locked_scenes_mix_and_budget(self):
        self.assertEqual(len(dev.jobs(self.protocol, self.scenes)), 48)
        self.assertEqual(sum(6*sampling_work(len(s['objects'])) for s in self.scenes), self.protocol['max_unet_sample_calls'])
        groups = [s['group'] for s in self.scenes]
        self.assertEqual((groups.count('same_class_colours'), groups.count('overlap'), groups.count('wide_box')), (4, 2, 2))
        canary = json.loads((dev.ROOT/'assets/direct_canary_v1/scene.json').read_text())
        canary_keys = {o['source']['key_id'] for o in canary['objects']}
        for spec in self.scenes:
            scene = render_scene(spec); boxes = [o.box for o in scene.objects]
            self.assertTrue(all(o['source']['line_index'] >= 4000 and o['source']['recognized'] for o in spec['objects']))
            self.assertFalse({o['source']['key_id'] for o in spec['objects']} & canary_keys)
            if spec['group'] == 'same_class_colours':
                self.assertEqual(scene.objects[0].cls, scene.objects[1].cls)
            if spec['group'] == 'wide_box':
                self.assertGreater((boxes[0][2]-boxes[0][0])/(boxes[0][3]-boxes[0][1]), 2)
            if spec['group'] == 'overlap':
                b, c = boxes[0], boxes[1]
                self.assertTrue(min(b[2], c[2]) > max(b[0], c[0]) and min(b[3], c[3]) > max(b[1], c[1]))
            scene_condition(scene)

    def test_complete_paired_noise_blind_sheets_and_resume(self):
        result = self.run_attempt('a'); self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.calls), 48)
        first = self.calls[:3]
        self.assertEqual([(c[1], c[2], c[3]) for c in first],
                         [(0, False, 'official_mean'), (0, True, 'official_mean'), (0, True, 'region')])
        directory = Path(result['directory']); key = json.loads((directory/'review_key.json').read_text())
        self.assertEqual(len(key), 16); self.assertEqual(len(list((directory/'review').glob('*.png'))), 16)
        self.assertGreater(len({tuple(v.values()) for v in key.values()}), 1)
        sid = self.scenes[0]['sid']
        sheet = np.array(Image.open(directory/'review'/f'01_{sid}_seed0.png'))
        for i, label in enumerate('ABC', 1):
            original = np.array(Image.open(directory/sid/f'{key[sid+"_s0"][label]}_s0.png'))
            self.assertTrue(np.array_equal(original, sheet[160:672, i*512:(i+1)*512]))
        with ZipFile(result['archive']) as z:
            hashes = json.loads(z.read('checksums.json'))
            self.assertTrue(all(hashlib.sha256(z.read(k)).hexdigest() == v for k, v in hashes.items()))
        with ZipFile(result['review_archive']) as z:
            names = z.namelist()
            self.assertIn('review/REVIEW_TEMPLATE.txt', names); self.assertEqual(sum(n.endswith('.png') for n in names), 16)
            self.assertFalse(any('key' in n for n in names))
        again = self.run_attempt('b'); self.assertEqual(again['status'], 'complete')
        self.assertEqual(len(self.calls), 48); self.assertEqual(self.load.call_count, 1)

    def test_failure_resumes_only_missing_and_records_history(self):
        original = self.model.generate
        def fail(condition, seed, negative, **kw):
            if len(self.calls) == 5: raise RuntimeError('test OOM; no retry')
            return original(condition, seed, negative, **kw)
        self.model.generate = fail
        failed = self.run_attempt('a'); self.assertEqual(failed['status'], 'failed'); self.assertEqual(len(self.calls), 5)
        self.model.generate = original
        done = self.run_attempt('b'); self.assertEqual(done['status'], 'complete'); self.assertEqual(len(self.calls), 48)
        self.assertEqual(len(list((Path(done['directory'])/'attempts').glob('*.json'))), 2)

    def test_interrupted_run_packages_without_model(self):
        self.run_attempt('a'); count = self.load.call_count
        with contextlib.redirect_stdout(io.StringIO()): packed = dev.finalize(self.root/'a', interrupted=True)
        self.assertEqual(packed['status'], 'incomplete'); self.assertEqual(self.load.call_count, count)


    def test_v2_silhouette_study_is_separate_and_passes_masks(self):
        seen = []
        original = self.model.generate
        def record(condition, seed, negative, **kw):
            seen.append((kw.get('silhouette', False), kw['scribble'], 'silhouette_rle' in condition['objects'][0]))
            return original(condition, seed, negative, **kw)
        self.model.generate = record
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = dev.run(self.root/'cache', self.root, self.root/'v2', version='v2')
        self.assertEqual(result['status'], 'complete'); self.assertEqual(len(seen), 48)
        self.assertEqual(seen[:3], [(False, False, True), (True, False, True), (True, True, True)])
        directory = Path(result['directory']); self.assertTrue(directory.name.startswith('dev2_'))
        key = json.loads((directory/'review_key.json').read_text())
        self.assertEqual(set(key[self.scenes[0]['sid']+'_s0'].values()), set(dev.ARMS_V2))
        condition = json.loads((directory/self.scenes[0]['sid']/'condition.json').read_text())
        self.assertIn('no oracle masks', condition['input_track']); self.assertEqual(len(condition['objects'][0]['polygon']), 512)
        v1, _ = dev.read_protocol(); self.assertNotIn('_version', v1)


if __name__ == '__main__': unittest.main()
