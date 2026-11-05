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
        P = Paths(str(self.root/'exp'), run_id='r1'); E = ExperimentConfig(tier=H.VERSIONS[H.VERSION]['tier'])
        H.install_scenes(P)
        from ocsd import stages
        job, = stages.experiment_plan(P, E)
        self.assertEqual(len(job['scenes']), 18); self.assertEqual(job['methods'], list(H.METHODS))
        baselines = stages.tier_baselines(E, P=P)
        self.assertEqual(baselines['controlnet'][1].cn_scale, 1.0)
        self.assertEqual(baselines['instance_repair_count_matte_soft'][1].steps, 30)
        self.assertEqual(len(list(self.root.rglob('*_base_s*.png'))), 36 if 'ocsd_v3' in H.METHODS else 0)
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
        for i, label in enumerate('ABCDEF'[:len(H.METHODS)], 1):
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



class HeldoutV2(Heldout):
    """Same orchestration on the v2 set: silhouette arm, sketch-fidelity primary tests."""
    def setUp(self):
        H.use('v2'); self.addCleanup(H.use, 'v1')
        super().setUp()
        original = self.model.generate
        def generate(condition, seed, negative, **kw):
            self.calls_sil = getattr(self, 'calls_sil', []) + [(kw.get('silhouette', False), 'silhouette_rle' in condition['objects'][0])]
            return original(condition, seed, negative, **kw)
        self.model.generate = generate

    def test_locked_scenes_bins_and_budget(self):
        self.assertEqual(len(self.scenes), 18)
        self.assertTrue(all(s.sid.startswith('h2_qd_') for s in self.scenes))
        bins = sorted(s.meta['count_bin'] for s in self.scenes)
        self.assertEqual({b: bins.count(b) for b in set(bins)}, {'3': 6, '5': 6, '8+': 6})
        self.assertEqual(sum(4*sampling_work(s.n) for s in self.scenes), self.protocol['direct_max_unet_sample_calls'])
        v1 = {str(o['qd_key']) for p in (H.ROOT/'assets/heldout_v1/quickdraw').glob('*/scene.json')
              for o in json.loads(p.read_text())['objects']}
        self.assertFalse({str(o['qd_key']) for s in self.scenes for o in s.meta['objects']} & v1)
        for s in self.scenes:
            c = scene_condition(s, silhouettes=True); self.assertEqual(len(c['objects']), s.n)

    def test_direct_import_plan_baselines_and_report(self):
        def segment(img, box):
            m = np.zeros(img.shape[:2], bool); x0, y0, x1, y1 = box; m[max(0, y0):y1, max(0, x0):x1] = True; return m
        with patch('ocsd.vision.Vision', return_value=SimpleNamespace(sam=segment)):
            super().test_direct_import_plan_baselines_and_report()
        self.assertEqual(sorted(set(self.calls_sil)), [(False, True), (True, True)])
        P = Paths(str(self.root/'exp'), run_id='r1')
        summary = json.loads((Path(P.results)/'summary.json').read_text())
        self.assertEqual(summary['heldout_set'], 'v2'); self.assertEqual([r['metric'] for r in summary['primary']], ['facing_acc', 'sketch_chamfer'])
        self.assertTrue(all(0 <= r['p_holm'] <= 1 and r['p_holm'] >= r['p'] for r in summary['primary']))
        fid = pd.read_csv(Path(P.results)/'quickdraw'/'sketch_fidelity.csv'); self.assertEqual(len(fid), 144)
        self.assertTrue((Path(P.results)/'ocsd_heldout2_results.zip').exists())
        with ZipFile(Path(P.results)/'ocsd_heldout2_results.zip') as z:
            self.assertTrue(any(n.startswith('direct/') and n.endswith('run.json') for n in z.namelist()))


class HeldoutV3(Heldout):
    """Final set: OCSD v3 (layout-only silhouettes + refiner) vs original OCSD and baselines, paper tier."""
    def setUp(self):
        H.use('v3'); self.addCleanup(H.use, 'v1')
        super().setUp()
        original = self.model.generate; self.kw = []
        def generate(condition, seed, negative, **kw):
            self.kw.append(kw); return original(condition, seed, negative, **kw)
        self.model.generate = generate
        self.refined = []
        def refine(pipe, pixels, caption, seed, strength, steps):
            self.refined.append((caption, seed, strength, steps)); return (255-pixels).astype(np.uint8), dict(refiner_steps=steps)
        rev = self.protocol['refiner_revision']
        self.stack.enter_context(patch('ocsd.direct_instances.load_refiner', return_value=(object(), rev)))
        self.stack.enter_context(patch('ocsd.direct_instances.refine', side_effect=refine))

    def test_locked_scenes_bins_and_budget(self):
        self.assertEqual(len(self.scenes), 18); self.assertTrue(all(s.sid.startswith('h3_qd_') for s in self.scenes))
        bins = sorted(s.meta['count_bin'] for s in self.scenes)
        self.assertEqual({b: bins.count(b) for b in set(bins)}, {'3': 6, '5': 6, '8+': 6})
        self.assertEqual(sum(4*sampling_work(s.n) for s in self.scenes), self.protocol['direct_max_unet_sample_calls'])
        used = set()
        for name in ('heldout_v1', 'heldout_v2'):
            used |= {str(o['qd_key']) for p in (H.ROOT/'assets'/name/'quickdraw').glob('*/scene.json')
                     for o in json.loads(p.read_text())['objects']}
        self.assertFalse({str(o['qd_key']) for s in self.scenes for o in s.meta['objects']} & used)
        from ocsd import stages, ocsd_v3
        E = ExperimentConfig(tier='heldout_final'); v, _ = stages.tier_variants(E)
        c = v['ocsd']
        self.assertEqual((c.alpha, c.lora_scale, c.use_energy, c.eta, c.use_region_attn, c.use_identity, c.S1, c.S2, c.steps),
                         (0.1, 0.5, True, 20.0, True, True, 100, 100, 30))
        b = stages.tier_baselines(E)
        self.assertEqual((b['gligen'][1].gligen_beta, b['controlnet'][1].cn_scale, b['collage'][1].collage_strength), (1.0, 1.0, 0.2))
        arm = self.protocol['direct_arms'][1]
        self.assertEqual((arm['silhouette_layout_only'], arm['refine']), (ocsd_v3.FROZEN.layout_only, ocsd_v3.FROZEN.refine))
        self.assertEqual(self.protocol['refiner_revision'], ocsd_v3.FROZEN.refiner_revision)

    def test_direct_import_plan_baselines_and_report(self):
        def segment(img, box):
            m = np.zeros(img.shape[:2], bool); x0, y0, x1, y1 = box; m[max(0, y0):y1, max(0, x0):x1] = True; return m
        from ocsd import stages
        with patch('ocsd.vision.Vision', return_value=SimpleNamespace(sam=segment)), \
             patch.object(stages, 'tier_baselines', side_effect=lambda E, P=None: {
                 'controlnet': ('controlnet', SimpleNamespace(cn_scale=1.0)),
                 'instance_repair_count_matte_soft': (None, SimpleNamespace(steps=30))}):
            super().test_direct_import_plan_baselines_and_report()
        self.assertEqual(len(self.refined), 36); self.assertEqual(self.refined[0][2:], (.35, 20))
        self.assertEqual(sum(bool(k.get('silhouette_layout_only')) for k in self.kw), 36)
        P = Paths(str(self.root/'exp'), run_id='r1')
        summary = json.loads((Path(P.results)/'summary.json').read_text())
        self.assertEqual(summary['tier'], 'heldout_final')
        self.assertEqual([(r['a'], r['b'], r['metric']) for r in summary['primary']],
                         [('ocsd_v3', 'ocsd', 'opr'), ('ocsd_v3', 'ocsd', 'miou'), ('ocsd_v3', 'ocsd', 'sketch_chamfer')])
        self.assertEqual(len(summary['key_secondary']), 6)
        self.assertTrue(all(r['p_holm'] >= r['p'] for r in summary['primary'] + summary['key_secondary']))
        self.assertTrue((Path(P.results)/'ocsd_heldout3_results.zip').exists())
        with Image.open(next((Path(P.results)/'review').glob('*.png'))) as sheet: self.assertEqual(sheet.size, (3584, 690))


class PaperV3(unittest.TestCase):
    """Paper-benchmark confirmation: scenes imported from the paper benchmark, checked against paper_v2."""
    def setUp(self):
        import shutil
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); root = Path(self.tmp.name)
        H.use('v4'); self.addCleanup(H.use, 'v1')
        assets = root/'assets'; assets.mkdir()
        for name in ('expected_scenes.json', 'protocol.json'): shutil.copyfile(H.ASSETS/name, assets/name)
        patcher = patch.multiple(H, ASSETS=assets, PROTOCOL=assets/'protocol.json'); patcher.start(); self.addCleanup(patcher.stop)
        self.src = root/'drive'/'quickdraw'; self.expected = H.expected_scenes()
        for sid, e in self.expected.items():
            d = self.src/sid; d.mkdir(parents=True)
            objects = [dict(cls='dog', phrase='dog', box=[10*i, 10, 10*i+40, 60]) for i in range(e['n_obj'])]
            (d/'scene.json').write_text(json.dumps(dict(sid=sid, split='quickdraw', n_obj=e['n_obj'], count_bin=e['count_bin'],
                complexity=e['complexity'], bg='in a park', caption='dogs in a park', objects=objects, relations=[])))
            Image.new('L', (512, 512), 255).save(d/'sketch.png')
            np.savez_compressed(d/'masks.npz', masks=np.zeros((e['n_obj'], 512, 512), bool))
            for i in range(e['n_obj']): Image.new('L', (512, 512), 255).save(d/f'obj_{i:02d}.png')
        (self.src/'qd_3_simple_000').mkdir()   # tuning-split scene in the same folder: never copied

    def test_import_checks_scenes_and_matches_budget(self):
        files = H.import_paper_scenes(self.src)
        self.assertEqual(len(H.scene_ids()), 72); self.assertEqual(H.verify_assets(), files)
        self.assertNotIn('qd_3_simple_000', H.scene_ids())
        bins = [e['count_bin'] for e in self.expected.values()]
        self.assertEqual({b: bins.count(b) for b in set(bins)}, {'1': 18, '3': 18, '5': 18, '8+': 18})
        protocol = json.loads(H.PROTOCOL.read_text())
        self.assertEqual(sum(4*sampling_work(e['n_obj']) for e in self.expected.values()), protocol['direct_max_unet_sample_calls'])
        self.assertEqual(len(load_scene(str(H.ASSETS/'quickdraw'/'qd_8p_complex_003')).objects), 9)
        manifest = (H.ASSETS/'manifest.json').read_text()
        H.import_paper_scenes(self.src); self.assertEqual((H.ASSETS/'manifest.json').read_text(), manifest)  # resumable
        P = Paths(str(Path(self.tmp.name)/'exp')); P.makedirs(); H.install_scenes(P)
        self.assertEqual(len(H.plan(P, ExperimentConfig(tier='heldout_final'))[0]['scenes']), 72)

    def test_rejects_a_different_scene(self):
        meta = json.loads((self.src/'qd_8p_complex_003'/'scene.json').read_text()); meta['n_obj'] = 10
        (self.src/'qd_8p_complex_003'/'scene.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'differs from the paper_v2'): H.import_paper_scenes(self.src)

    def test_frozen_ocsd_v3_and_same_analysis_as_heldout_v3(self):
        from ocsd import ocsd_v3
        p = json.loads(H.PROTOCOL.read_text()); v3 = json.loads((H.ROOT/'assets/heldout_v3/protocol.json').read_text())
        self.assertEqual(p['direct_arms'], v3['direct_arms'])
        for k in ('refiner_revision', 'refiner_strength', 'refiner_steps', 'steps', 'mis', 'alpha', 'model_provenance', 'primary_tests'):
            self.assertEqual(p[k], v3[k])
        self.assertEqual(p['refiner_revision'], ocsd_v3.FROZEN.refiner_revision)
        self.assertEqual(json.loads((H.ROOT/'assets/paper_v3/tuned_paper_v2.json').read_text()),
                         json.loads((H.ROOT/'assets/heldout_v3/tuned_paper_v2.json').read_text()))
        self.assertEqual(H.VERSIONS['v4']['primary'], H.VERSIONS['v3']['primary'])


class Fidelity(unittest.TestCase):
    def test_mirror_and_missing_object(self):
        from ocsd.sketch_fidelity import instance_fidelity
        ink = np.zeros((100, 100), bool); ink[20:80, 20] = True; ink[20, 20:80] = True; ink[20:40, 79] = True  # asymmetric
        mask = np.zeros((100, 100), bool); mask[20:81, 20:80] = True
        r = instance_fidelity(mask, ink, (10, 10, 90, 90))
        self.assertTrue(r['found']); self.assertTrue(r['facing_eligible'])
        mirrored = instance_fidelity(mask[:, ::-1].copy(), ink, (10, 10, 90, 90))
        self.assertLessEqual(r['chamfer'], mirrored['chamfer_mirror']+1e-9)
        empty = instance_fidelity(np.zeros((100, 100), bool), ink, (10, 10, 90, 90))
        self.assertFalse(empty['found']); self.assertFalse(empty['facing_correct']); self.assertEqual(empty['chamfer'], 1.0)

    def test_facing_detects_mirrored_shape(self):
        from ocsd.sketch_fidelity import instance_fidelity
        from ocsd.direct_instances import sketch_silhouette
        ink = np.full((120, 120), 255, np.uint8)
        ink[60, 20:100] = 0; ink[30:60, 90:100] = 0; ink[30, 70:100] = 0; ink[30:60, 70] = 0  # body + head on the right
        mask, _ = sketch_silhouette(ink); ink = ink < 128
        good = instance_fidelity(mask, ink, (10, 20, 110, 70)); bad = instance_fidelity(mask[:, ::-1].copy(), ink, (10, 20, 110, 70))
        if good['facing_eligible']:
            self.assertTrue(good['facing_correct']); self.assertFalse(bad['facing_correct'])


if __name__ == '__main__': unittest.main()
