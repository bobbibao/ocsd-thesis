"""Regression tests for experiment identity, cache invalidation and unbiased evaluation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from ocsd.config import OCSDConfig, Paths, ExperimentConfig
from ocsd import provenance as P, metrics, report, stages
from ocsd.methods import m2_dir_key, m3_key
from ocsd.runner import Runner
from ocsd.sketch import Scene, SceneObject


class Regression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'HF_HOME': str(self.root / 'hf')})
        self.env.start()
        self.cfg = OCSDConfig()
        m = np.zeros((16, 16), bool); m[4:12, 4:12] = True
        sk = np.where(m, 0, 255).astype(np.uint8)
        self.scene = Scene('s', sk, [SceneObject('cat', 'red cat', (4,4,12,12), m, sk)], 'in a park', 'a cat')
        eng = SimpleNamespace(dtype='float32', scheduler=SimpleNamespace(config={}), compute={})
        self.runner = Runner(eng, None, str(self.root / 'out'), variants={'ocsd': self.cfg})

    def tearDown(self):
        P.MODEL_REVISIONS.clear()
        self.env.stop()
        self.tmp.cleanup()

    def test_m2_dependency_fields(self):
        key = m2_dir_key(self.cfg, 0)
        for kw in [dict(obj_steps=7), dict(obj_cn_scale=.7), dict(obj_prompt_tmpl='{phrase}'),
                   dict(select_weights=(2,1,1)), dict(height=256), dict(guidance=2), dict(negative_prompt='x')]:
            self.assertNotEqual(key, m2_dir_key(self.cfg.replace(**kw), 0))
        self.assertEqual(key, m2_dir_key(self.cfg.replace(alpha=.7, eta=4), 0))
        self.assertNotEqual(m3_key(self.cfg), m3_key(self.cfg.replace(lr_emb1=.001)))
        self.assertNotEqual(m3_key(self.cfg), m3_key(self.cfg.replace(lambda_att=.2)))

    def test_image_resume_requires_pixels_config_and_scene(self):
        r, sc = self.runner, self.scene
        img = np.zeros((16,16,3), np.uint8)
        r._write('quickdraw', 'ocsd', sc, 0, img, {}, self.cfg)
        self.assertTrue(r.valid_image('quickdraw','ocsd',sc,0))
        sc.caption = 'a different caption'
        self.assertFalse(r.valid_image('quickdraw','ocsd',sc,0))
        sc.caption = 'a cat'
        r.variants['ocsd'] = self.cfg.replace(guidance=5)
        self.assertFalse(r.valid_image('quickdraw','ocsd',sc,0))
        r.variants['ocsd'] = self.cfg
        Path(r.img_path('quickdraw','ocsd','s',0)).write_bytes(b'corrupt')
        self.assertFalse(r.valid_image('quickdraw','ocsd',sc,0))
        r._write('quickdraw','ocsd',sc,0,img+50,{},self.cfg)
        self.assertTrue(list((self.root/'out/_stale').rglob('*.png')))
        self.assertTrue(r.valid_image('quickdraw','ocsd',sc,0))

    def test_run_resume_and_data_drift(self):
        paths = Paths(str(self.root)); paths.makedirs()
        e = ExperimentConfig(tier='pilot')
        p, run = P.start_run(paths,e,model_revisions={'fake/model':'abc'})
        run.seal_data()
        p2, _ = P.start_run(paths,e,model_revisions={'fake/model':'abc'})
        self.assertEqual(p.run_id,p2.run_id)
        self.assertNotEqual(p.results, paths.results)
        (Path(p.benchmarks)/'tuning_split.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'different'):
            P.start_run(paths,e,run_id=p.run_id,model_revisions={'fake/model':'abc'})
        p3,_ = P.start_run(paths,e,model_revisions={'fake/model':'abc'})
        self.assertNotEqual(p.run_id,p3.run_id)

    def test_manifest_failure_propagates(self):
        p,run = P.start_run(Paths(str(self.root)),ExperimentConfig(tier='pilot'),model_revisions={})
        with self.assertRaisesRegex(ValueError,'intentional'):
            with run.stage('test'):
                raise ValueError('intentional')
        self.assertEqual(json.load(open(Path(p.results)/'run_manifest.json'))['status'],'failed')

    def test_colab_autosave_and_run_controls_do_not_break_resume(self):
        nb = self.root/'notebook.ipynb'
        value = dict(cells=[dict(cell_type='code', source=['RUN_ID = ""\n', 'MAX_MINUTES = 600\n', 'print("run")'],
                                outputs=[], execution_count=None)])
        nb.write_text(json.dumps(value))
        e = ExperimentConfig(tier='pilot')
        with patch.object(P, 'ROOT', self.root), patch.object(P, 'source_identity', return_value={'code': 'same'}):
            p, run = P.start_run(Paths(str(self.root)), e, notebook='notebook.ipynb', model_revisions={})
            run.seal_data()
            first_raw = run.record['notebook_file_sha256']
            value['cells'][0].update(outputs=[dict(output_type='stream', text='saved output')], execution_count=1,
                source=[f'RUN_ID = "{p.run_id}"\n', 'MAX_MINUTES = 10\n', 'print("run")'])
            nb.write_text(json.dumps(value))
            again, resumed = P.start_run(Paths(str(self.root)), e, run_id=p.run_id,
                                         notebook='notebook.ipynb', model_revisions={})
            self.assertEqual(p.run_id, again.run_id)
            self.assertNotEqual(first_raw, resumed.record['notebook_file_sha256'])
            value['cells'][0]['source'][-1] = 'print("changed algorithm")'
            nb.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'different'):
                P.start_run(Paths(str(self.root)), e, run_id=p.run_id, notebook='notebook.ipynb', model_revisions={})

    def test_model_lock_requires_known_model(self):
        P.MODEL_REVISIONS.update({'a':'sha'})
        self.assertEqual(P.model_kwargs('a'),{'revision':'sha'})
        with self.assertRaises(RuntimeError): P.model_kwargs('b')

    def test_manifest_finished_after_stage_has_valid_checksums(self):
        from ocsd import runlog
        p, run = P.start_run(Paths(str(self.root)), ExperimentConfig(tier='pilot'), model_revisions={})
        logs = Path(p.results)/'logs'; logs.mkdir()
        jobs = {'job': dict(total=1, todo=0)}
        plan = [dict(name='job', split='quickdraw', methods=['ocsd'], seeds=[0], scenes=['/bench/s'])]
        with patch.dict(runlog._STATE, {'log_dir': str(logs)}), \
             patch.object(stages, 'log_progress', return_value=jobs), \
             patch.object(stages, 'experiment_plan', return_value=plan):
            run.plan(None)
            self.assertEqual(run.record['planned_jobs'][0]['scenes'], ['s'])
            with run.stage('D_report'):
                (Path(p.results)/'summary.md').write_text('test result')
            run.finish(None)
            runlog.summary()
        manifest = json.loads((Path(p.results)/'run_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['stages']['D_report']['status'], 'complete')
        self.assertIn(f'results/runs/{p.run_id}/logs/status.json', manifest['artifacts'])
        for path, expected in manifest['artifacts'].items():
            self.assertEqual(P.file_hash(self.root/path), expected)

    def test_sync_selects_run_and_keeps_manifest(self):
        from ocsd import sync
        legacy = self.root/'results'; legacy.mkdir()
        self.assertEqual(sync.results_for_run(str(self.root)), str(legacy))
        p, _ = P.start_run(Paths(str(self.root)), ExperimentConfig(tier='pilot'), model_revisions={})
        self.assertEqual(sync.results_for_run(str(self.root)), p.results)
        self.assertEqual(sync.results_for_run(str(self.root), p.run_id), p.results)
        self.assertIn('run_manifest.json', sync.INCLUDE)
        with self.assertRaises(FileNotFoundError): sync.results_for_run(str(self.root), 'missing')
        with self.assertRaises(ValueError): sync.results_for_run(str(self.root), '../outside')

    def test_new_run_tuning_split_keeps_current_config(self):
        p, _ = P.start_run(Paths(str(self.root)), ExperimentConfig(tier='paper'), model_revisions={})
        config = Path(p.results)/'experiment_config.json'; config.write_text('{}')
        with patch.object(stages, 'list_scenes', return_value=['/bench/pilot_scene']):
            split = stages.freeze_tuning_split(p)
        self.assertEqual(split['quickdraw'], ['pilot_scene'])
        self.assertTrue(config.exists())
        self.assertFalse((Path(p.results)/'pilot').exists())

    def test_fid_content_and_reference_invalidation(self):
        out=self.root/'quality'; res=self.root/'results'
        image=out/'quickdraw/controlnet/s_s0.png'; image.parent.mkdir(parents=True)
        ref=self.root/'ref.png'
        cv2.imwrite(str(image),np.zeros((16,16,3),np.uint8)); cv2.imwrite(str(ref),np.ones((16,16,3),np.uint8))
        calls=[]
        def fid(gen, refs, device, **kwargs):
            calls.append((gen,refs)); return dict(fid=1,kid=1,n_gen=len(gen))
        args=(str(out),'unused','quickdraw',['controlnet'],[0],[str(ref)],str(res))
        with patch.object(metrics,'fid_kid',fid):
            for _ in range(2): metrics.image_level_quality(*args,with_lpips=False)
            self.assertEqual(len(calls),1)
            cv2.imwrite(str(image),np.full((16,16,3),25,np.uint8))
            metrics.image_level_quality(*args,with_lpips=False)
            cv2.imwrite(str(ref),np.full((16,16,3),75,np.uint8))
            metrics.image_level_quality(*args,with_lpips=False)
            self.assertEqual(len(calls),3)
            with self.assertRaisesRegex(RuntimeError,'Incomplete FID'):
                metrics.image_level_quality(*args,with_lpips=False,pairs={'controlnet':{('s',0),('s',1)}})

    def test_evaluation_identity_tracks_threshold_and_input(self):
        scene=self.root/'scene'; scene.mkdir()
        image=scene/'sketch.png'; image.write_bytes(b'a')
        a=metrics.evaluation_identity(str(image),str(scene),(.3,.1))
        b=metrics.evaluation_identity(str(image),str(scene),(.4,.1))
        self.assertNotEqual(a,b)
        image.write_bytes(b'b')
        self.assertNotEqual(a,metrics.evaluation_identity(str(image),str(scene),(.3,.1)))

    def test_paired_effect_ci_and_insufficient_pairs(self):
        rows=[dict(method=m,sid=str(i),opr=i/10+offset) for m,offset in [('a',.1),('b',0)] for i in range(8)]
        df=pd.DataFrame(rows)
        t=report.paired_tests(df,ref='a',metrics=['opr']).iloc[0]
        self.assertAlmostEqual(t['diff'],.1); self.assertAlmostEqual(t.diff_lo,.1)
        df.loc[(df.method=='b') & (df.sid.astype(int)>3),'opr']=np.nan
        t=report.paired_tests(df,ref='a',metrics=['opr']).iloc[0]
        self.assertEqual(t.status,'insufficient_pairs'); self.assertTrue(np.isnan(t.p))

    def test_quickdraw_incremental_matches_fresh(self):
        from ocsd.data import build_quickdraw_scenes
        raw = self.root / 'raw'; raw.mkdir()
        # One class present in every background, non-degenerate strokes at several complexity levels.
        rows = [dict(key_id=str(i), recognized=True, drawing=[[[0,255,255,0,0],[0,0,255,255,0]]] * (1+i%3)) for i in range(12)]
        (raw/'bird.ndjson').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        # Supply all classes so every documented background has an available pool.
        from ocsd.data import QD_CLASSES
        for cls in QD_CLASSES:
            (raw/(cls+'.ndjson')).write_text((raw/'bird.ndjson').read_text())
        incremental, fresh = self.root/'inc', self.root/'fresh'
        with patch.dict('ocsd.data.COUNTS', {'1': (1, 1), '3': (3, 3)}, clear=True):
            build_quickdraw_scenes(str(raw),str(incremental),per_cell=1,size=128)
            build_quickdraw_scenes(str(raw),str(incremental),per_cell=2,size=128)
            build_quickdraw_scenes(str(raw),str(fresh),per_cell=2,size=128)
        for p in fresh.rglob('scene.json'):
            self.assertEqual(p.read_bytes(), (incremental/p.relative_to(fresh)).read_bytes())
        for p in fresh.rglob('sketch.png'):
            self.assertEqual(p.read_bytes(), (incremental/p.relative_to(fresh)).read_bytes())
        states = []
        def retry_layout(n, aspects, overlap, rng, size):
            states.append(rng.getstate())
            return None if len(states) == 1 else [(20, 20, 100, 100)]
        with patch.dict('ocsd.data.COUNTS', {'1': (1, 1)}, clear=True), \
             patch('ocsd.data._sample_layout', side_effect=retry_layout):
            made = build_quickdraw_scenes(str(raw), str(self.root/'retry'), per_cell=1, size=128)
        self.assertEqual(len(made), 3)
        self.assertNotEqual(states[0], states[1], 'Failed layout must advance the scene RNG')

    def test_tuning_rejects_missing_evaluation(self):
        from ocsd import runner as runner_module
        fake = SimpleNamespace(pending_items=lambda *a: [], run_items=lambda *a, **kw: None)
        rows = pd.DataFrame([dict(method='x', sid='s', seed=0, opr=1., miou=1., ra=1.,
                                  oce_c=0., clip=30., id_sim=1., split='quickdraw')])
        paths = Paths(str(self.root)); paths.makedirs()
        with patch.object(stages, '_tuning_jobs', return_value=[('quickdraw', ['/tmp/s'])]), \
             patch.object(stages, 'load_scene', return_value=self.scene), \
             patch.object(runner_module, 'Runner', return_value=fake), \
             patch.object(metrics, 'evaluate_images', return_value=rows):
            with self.assertRaisesRegex(RuntimeError, 'evaluation incomplete'):
                stages._run_grid(paths, ExperimentConfig(tier='pilot'), None, None,
                                 {'x': self.cfg}, 'x', 'missing', None)

    def test_notebook_fail_closed_and_no_reset(self):
        import ast
        nb=json.load(open(Path(__file__).resolve().parents[1]/'notebooks/RUN_ALL.ipynb'))
        code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
        self.assertIn('P, RUN = start_run',code)
        self.assertIn('if PUSH_RESULTS:',code)
        self.assertIn('with RUN.stage',code)
        self.assertNotIn('reset -q --hard',code)
        for cell in nb['cells']:
            if cell['cell_type']=='code': compile(''.join(cell['source']),'<notebook>','exec')
        report_cell = next(''.join(c['source']) for c in nb['cells'] if 'RUN.finish(E)' in ''.join(c['source']))
        # Finalise outside the with-block, after runlog has written its final status.json.
        tree = ast.parse(report_cell)
        self.assertTrue(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                            and ast.unparse(n.value.func) == 'RUN.finish' for n in tree.body))

if __name__=='__main__': unittest.main(verbosity=2)
