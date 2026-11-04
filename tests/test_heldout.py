"""Held-out orchestration tests: synthetic pixels and metrics, no pretrained-quality claim."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pandas as pd
import torch
from ocsd import heldout as H
from ocsd.config import ExperimentConfig, Paths, TIERS
from ocsd.data import load_scene
from ocsd.direct_instances import scene_condition, sampling_work


class Heldout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close); self.addCleanup(self.tmp.cleanup)
        for target, value in [('is_available', True), ('get_device_properties', SimpleNamespace(name='test GPU', total_memory=24*1024**3)),
                              ('reset_peak_memory_stats', None), ('synchronize', None), ('max_memory_allocated', 1024)]:
            self.stack.enter_context(patch.object(torch.cuda, target, return_value=value))
        self.stack.enter_context(patch.object(torch.backends.cudnn, 'version', return_value=1))
        self.protocol, self.scenes = H.read_protocol()
        self.calls = []; n = {s.sid: s.n for s in self.scenes}
        def generate(condition, seed, negative, **kw):
            self.calls.append((condition['sid'], seed, kw['scribble']))
            return np.full((512, 512, 3), (len(self.calls)*5) % 250, np.uint8), dict(
                unet_sample_calls=sampling_work(n[condition['sid']]), wall_s=1.0, external_method='InstanceDiffusion test')
        self.model = SimpleNamespace(provenance=self.protocol['model_provenance'], generate=generate)
        self.load = self.stack.enter_context(patch('ocsd.direct_instances.load_components', return_value=self.model))

    def direct(self, name):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return H.run_direct(self.root/'cache', self.root/'exp'/'direct', self.root/'exp'/'direct'/name)

    def test_locked_scenes_bins_and_budget(self):
        self.assertEqual(len(self.scenes), 18)
        bins = sorted(s.meta['count_bin'] for s in self.scenes)
        self.assertEqual({b: bins.count(b) for b in set(bins)}, {'3': 6, '5': 6, '8+': 6})
        self.assertTrue(all(s.sid.startswith('ho_qd_') for s in self.scenes))
        for s in self.scenes: scene_condition(s)
        self.assertEqual(sum(4*sampling_work(s.n) for s in self.scenes), self.protocol['direct_max_unet_sample_calls'])
        E = ExperimentConfig(tier='heldout_screen'); self.assertEqual(E.seeds, [0, 1])
        self.assertEqual(TIERS['heldout_screen']['cfg'], TIERS['foreground_screen']['cfg'])

    def test_tampered_scene_is_refused(self):
        rel = next(iter(H.manifest()))
        with patch.object(H, 'manifest', return_value={**H.manifest(), rel: '0'*64}):
            with self.assertRaisesRegex(ValueError, 'locked manifest'): H.verify_assets()

    def test_direct_import_plan_baselines_and_report(self):
        result = self.direct('a'); self.assertEqual(result['status'], 'complete'); self.assertEqual(len(self.calls), 72)
        self.assertEqual(self.direct('b')['status'], 'complete'); self.assertEqual(len(self.calls), 72)
        P = Paths(str(self.root/'exp'), run_id='r1'); E = ExperimentConfig(tier='heldout_screen')
        H.install_scenes(P)
        from ocsd import stages
        job, = stages.experiment_plan(P, E)
        self.assertEqual(len(job['scenes']), 18); self.assertEqual(job['methods'], list(H.METHODS))
        baselines = stages.tier_baselines(E, P=P)
        self.assertEqual(baselines['controlnet'][1].cn_scale, 1.0)
        self.assertEqual(baselines['instance_repair_count_matte_soft'][1].steps, 30)
        with contextlib.redirect_stdout(io.StringIO()): H.import_direct(P)
        out = Path(P.outputs)/'quickdraw'
        self.assertEqual(len(list((out/'direct_box').glob('*.png'))), 36)
        # synthetic old-pipeline outputs and per-image metrics
        rng = np.random.default_rng(0); res = Path(P.results)/'quickdraw'; res.mkdir(parents=True)
        for m in H.METHODS:
            rows = []
            for sid in H.scene_ids():
                for s in H.SEEDS:
                    if m not in H.EXTERNAL:
                        p = out/m/f'{sid}_s{s}.png'; p.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.zeros((512, 512, 3), np.uint8)).save(p)
                        p.with_suffix('.json').write_text(json.dumps(dict(compute=dict(unet_sample_calls=60), scene_time=5.0)))
                    rows.append(dict(method=m, sid=sid, seed=s, count_bin='3', opr=rng.random(), count_acc=rng.random(),
                                     oce_c=rng.random(), miou=rng.random(), ra=rng.random(), joint_layout_success=1.0,
                                     clip=30.0, obj_clip=20.0, time_s=1.0))
            pd.DataFrame(rows).to_csv(res/f'per_image_{m}.csv', index=False)
        with contextlib.redirect_stdout(io.StringIO()): summary = stages.report(P, E)
        with ZipFile(summary['review_archive']) as z:
            names = z.namelist()
            self.assertEqual(sum(n.endswith('.png') for n in names), 36); self.assertFalse(any('key' in n for n in names))
        key = json.loads((Path(P.results)/'review_key.json').read_text()); sid = H.scene_ids()[0]
        sheet = np.array(Image.open(Path(P.results)/'review'/f'01_{sid}_seed0.png'))
        for i, label in enumerate('ABCD', 1):
            original = np.array(Image.open(out/key[f'{sid}_s0'][label]/f'{sid}_s0.png').convert('RGB'))
            self.assertTrue(np.array_equal(original, sheet[160:672, i*512:(i+1)*512]))
        paired = pd.read_csv(Path(P.results)/'tables'/'heldout_paired.csv')
        self.assertTrue((paired[paired.a == 'direct_scribble'].role == 'post-hoc exploratory').all())

    def test_incomplete_direct_run_blocks_import(self):
        original = self.model.generate
        def fail(condition, seed, negative, **kw):
            if len(self.calls) == 3: raise RuntimeError('test OOM')
            return original(condition, seed, negative, **kw)
        self.model.generate = fail
        self.assertEqual(self.direct('a')['status'], 'failed')
        P = Paths(str(self.root/'exp'), run_id='r1')
        with self.assertRaisesRegex(RuntimeError, 'No complete direct'): H.import_direct(P)


if __name__ == '__main__': unittest.main()
