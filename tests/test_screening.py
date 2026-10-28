"""Offline behavioral checks of the experimental sampler and its bounded screen. No quality evidence."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ocsd import screening as S, stages
from ocsd.config import Paths, ExperimentConfig, OCSDConfig
from ocsd.method import protect_regions
from ocsd.sketch import Scene, SceneObject


class Screening(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 3), np.uint8)
        a = np.zeros((32, 32), bool); a[2:12, 2:12] = True
        b = np.zeros_like(a); b[20:30, 20:30] = True
        self.scene = Scene('s', np.zeros((32, 32), np.uint8), [
            SceneObject('cat', 'cat', (2,2,12,12), a, a.astype(np.uint8)),
            SceneObject('dog', 'dog', (20,20,30,30), b, b.astype(np.uint8))], 'park', 'a cat and a dog')
        self.dets = [dict(cls=o.cls, box=o.box, score=.9) for o in self.scene.objects]
        self.cfg = OCSDConfig(height=32, width=32, R=2, use_identity=False)

    def run_candidate(self, detections, candidate=None, clip=(30.,30.), use_m2=True):
        vis = SimpleNamespace(gdino=Mock(side_effect=detections),
                              clip=SimpleNamespace(score=Mock(side_effect=[[x] for x in clip])))
        get = Mock(return_value=['objects'])
        log = {}
        with patch.object(S, '_gligen', return_value=self.image), \
             patch.object(S, 'compose_foreground', return_value=('fg',)), \
             patch.object(S, 'repair_scene', return_value=candidate) as repair:
            result = S.grounded_repair(None, vis, self.scene, 0, self.cfg, 'sd15', None, log, get, use_m2)
        return result, get, repair, log

    def test_passing_gligen_skips_m2_and_repair(self):
        result, get, repair, log = self.run_candidate([self.dets])
        self.assertIs(result, self.image); get.assert_not_called(); repair.assert_not_called()
        self.assertEqual(log['tries'], 1)

    def test_missing_object_repair_is_lazy_and_accepted(self):
        candidate = self.image+50
        result, get, repair, log = self.run_candidate([self.dets[:1], self.dets], candidate)
        self.assertIs(result, candidate); get.assert_called_once()
        self.assertTrue(repair.call_args.kwargs['protect_matched'])
        self.assertTrue(log['verify'][-1]['accepted'])

    def test_failed_or_quality_regressing_edit_keeps_initial(self):
        for ds, clip in [(self.dets[:1], (30.,30.)), (self.dets, (30.,27.))]:
            result, get, repair, log = self.run_candidate([self.dets[:1], ds], self.image+50, clip)
            self.assertIs(result, self.image)
            self.assertFalse(log['verify'][-1]['accepted'])
            self.assertEqual(repair.call_count, 1)  # stop spending after rejection

    def test_no_m2_ablation_and_extra_only_do_not_prepare_objects(self):
        _, get, _, _ = self.run_candidate([self.dets[:1], self.dets], self.image+50, use_m2=False)
        get.assert_not_called()
        _, get, _, _ = self.run_candidate([self.dets+[dict(cls='cat',box=(14,2,18,8),score=.9)], self.dets],
                                         self.image+50)
        get.assert_not_called()

    def test_losing_previously_found_object_is_rejected(self):
        from ocsd.matching import consistency
        a = consistency(self.scene, self.dets[:1])
        b = consistency(self.scene, self.dets[1:])
        self.assertEqual(S.accept_repair(a,b,30,30), (False, 'lost_object'))

    def test_protected_mask_excludes_matched_shape_and_detection(self):
        mask = protect_regions(np.ones((32,32), bool), [o.mask for o in self.scene.objects],
                               dict(matched=[(0,1.),(-1,0.)]), self.dets)
        self.assertFalse(mask[2:12,2:12].any())
        self.assertTrue(mask[20:30,20:30].all())

    def test_plan_is_fixed_and_has_strong_controls(self):
        with tempfile.TemporaryDirectory() as d:
            paths = Paths(d)
            scenes = [f'/bench/qd_{n}_{c}_000' for n in ('1','3','5','8p') for c in ('simple','medium','complex')]
            with patch.object(stages, 'list_scenes', return_value=scenes):
                plan = stages.experiment_plan(paths, ExperimentConfig(tier='screen'))
            with patch.object(stages, 'list_scenes', return_value=scenes[:-1]):
                with self.assertRaisesRegex(RuntimeError,'12 benchmark'):
                    stages.experiment_plan(paths, ExperimentConfig(tier='screen'))
            self.assertEqual(sum(len(j['scenes'])*len(j['seeds'])*len(j['methods']) for j in plan),84)
            self.assertEqual(set(plan[0]['methods']), set(S.METHODS))
            cfg = stages.tier_baselines(ExperimentConfig(tier='screen'))['grounded_repair'][1]
            self.assertFalse(cfg.use_identity); self.assertEqual(cfg.R,1)
            signatures = stages._gen_signatures(ExperimentConfig(tier='screen'))
            self.assertIn('grounded_repair', signatures['quickdraw'])
            with self.assertRaises(ValueError): ExperimentConfig(tier='screen',backbone='sd21')
            split = Path(paths.benchmarks)/'tuning_split.json'; split.parent.mkdir(parents=True)
            split.write_text(json.dumps(dict(quickdraw=[])))
            with patch.object(stages, 'list_scenes', return_value=scenes):
                with self.assertRaisesRegex(RuntimeError,'held-out'):
                    stages.experiment_plan(paths, ExperimentConfig(tier='screen'))

    def test_real_tiny_repair_preserves_protected_pixels(self):
        import open_clip
        from tiny import make_engine
        from ocsd.method import Prepared, repair_scene, repair_regions, scene_cond
        from ocsd.matching import consistency
        with tempfile.TemporaryDirectory() as d:
            eng = make_engine(d, str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'), 32)
            cfg = self.cfg.replace(steps=2, use_energy=False, use_region_attn=False, use_blend=False)
            ctx = scene_cond(eng, self.scene, cfg, None)
            check = consistency(self.scene, self.dets[:1], size=32)
            mask, _ = repair_regions(self.scene, ctx, check, self.dets[:1], 32)
            mask = protect_regions(mask, ctx.region_masks, check, self.dets[:1])
            for use_m2 in (False, True):
                fg = (self.image+180, [o.mask for o in self.scene.objects],
                      np.any([o.mask for o in self.scene.objects],0)) if use_m2 else None
                out = repair_scene(eng, self.scene, Prepared(None,fg,None), cfg.replace(use_blend=use_m2),
                                   self.image, check, self.dets[:1], 7, cfg.lambda0, protect_matched=True)
                self.assertEqual(out.shape, self.image.shape)
                self.assertTrue(np.array_equal(out[~mask], self.image[~mask]))
                self.assertTrue((out[mask] != self.image[mask]).any())
            # Exercise the real Runner dispatch, artifact signature, and second-pass resume too.
            from ocsd.runner import Runner
            vis = SimpleNamespace(gdino=Mock(return_value=self.dets))
            runner = Runner(eng, vis, str(Path(d)/'out'), variants={},
                            baselines={'grounded_repair': ('grounded_repair', cfg)})
            items = [('quickdraw', '/unused', [('grounded_repair', 0)])]
            with patch('ocsd.runner.load_scene', return_value=self.scene), \
                 patch.object(S, '_gligen', return_value=self.image):
                first = runner.run_items(items, verbose=False)
                second = runner.run_items(items, verbose=False)
            self.assertEqual((first['done'], first['failed'], second['skipped']), (1, 0, 1))
            meta = json.loads(Path(runner.img_path('quickdraw','grounded_repair','s',0)).with_suffix('.json').read_text())
            self.assertTrue(meta['experimental']); self.assertIn('compute', meta)

    def test_report_rejects_missing_rows_and_does_not_escalate(self):
        with tempfile.TemporaryDirectory() as d:
            paths = Paths(d); os.makedirs(paths.results)
            plan = [dict(methods=list(S.METHODS),scenes=[f'/bench/s{i}' for i in range(6)],seeds=[0])]
            rows = [dict(method=m,sid=f's{i}',seed=0,split='quickdraw',n_obj=3,count_bin='3',complexity='simple',
                         opr=.5,oce_c=1,miou=.4,ra=.5,clip=30,time_s=2,m2_s=1)
                    for m in S.METHODS for i in range(6)]
            with patch.object(stages,'experiment_plan',return_value=plan), \
                 patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows)):
                result = S.screen_report(paths,ExperimentConfig(tier='screen'))
                self.assertEqual(result['decision'],'do_not_scale_up')
                self.assertEqual(result['research_claim'],'unproven')
            with patch.object(stages,'experiment_plan',return_value=plan), \
                 patch('ocsd.report.load_per_image',return_value=pd.DataFrame(rows[:-1])):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'): S.screen_report(paths,ExperimentConfig(tier='screen'))

if __name__ == '__main__': unittest.main(verbosity=2)
