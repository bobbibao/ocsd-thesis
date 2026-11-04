"""CPU bootstrap failures remain reviewable; Run all cannot fall through to GPU work."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tools import preflight_direct as preflight

ROOT=Path(__file__).resolve().parents[1]


class Preflight(unittest.TestCase):
    def test_tee_exposes_stream_properties_and_copies_all_text(self):
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as terminal:
            log=io.StringIO();tee=preflight.Tee(terminal,log)
            self.assertEqual(tee.isatty(),terminal.isatty())
            self.assertEqual(tee.fileno(),terminal.fileno())
            self.assertEqual(tee.encoding,'utf-8')
            self.assertEqual(tee.errors,terminal.errors)
            self.assertTrue(tee.writable())
            self.assertEqual(tee.write('Kiểm tra\n'),9)
            tee.writelines(['CLIP ', 'ready\n']);tee.flush()
            terminal.seek(0)
            self.assertEqual(terminal.read(),'Kiểm tra\nCLIP ready\n')
            self.assertEqual(log.getvalue(),'Kiểm tra\nCLIP ready\n')

    def test_clip_loading_report_inside_recorded_preflight(self):
        # Reproduce the uploaded run's real Transformers loading-report path.
        # A local tiny full-CLIP checkpoint provides unused vision/projection keys;
        # CLIPTextModel reports these and queries sys.stdout.isatty(). No HF access.
        import torch
        from transformers import CLIPConfig,CLIPModel,CLIPTextModel
        config=CLIPConfig(text_config=dict(vocab_size=32,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=8,
            bos_token_id=1,eos_token_id=2,pad_token_id=0),
            vision_config=dict(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
                               num_attention_heads=2,image_size=16,patch_size=8),projection_dim=8)
        full=CLIPModel(config).eval()
        with tempfile.TemporaryDirectory() as td:
            checkpoint=Path(td)/'tiny_clip';full.save_pretrained(checkpoint)
            def load(cache):
                text,info=CLIPTextModel.from_pretrained(checkpoint,local_files_only=True,output_loading_info=True)
                self.assertTrue(info['unexpected_keys'])
                self.assertFalse(info['missing_keys']);self.assertFalse(info['mismatched_keys'])
                for k,v in full.text_model.state_dict().items():self.assertTrue(torch.equal(v,text.state_dict()[k]))
                return SimpleNamespace(provenance={'test_checkpoint':'local synthetic CLIP'})
            with patch.object(preflight,'load_components',side_effect=load),patch.object(preflight,'probe',return_value={'test_only':True}):
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    result,archive=preflight.run(Path(td)/'cache',Path(td)/'out')
            self.assertEqual(result['status'],'pass',result.get('error'))
            with ZipFile(archive) as z:
                self.assertEqual(json.loads(z.read('preflight.json'))['status'],'pass')
                self.assertNotIn(b'AttributeError',z.read('preflight.log'))

    def test_failed_download_and_failed_install_keep_logs(self):
        for install_code in (0,1):
            with self.subTest(install_code=install_code),tempfile.TemporaryDirectory() as td:
                setup=Path(td)/'pip.log';setup.write_text('test installation output')
                with patch.object(preflight,'load_components',side_effect=RuntimeError('download unavailable')) as load:
                    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                        result,archive=preflight.run(Path(td)/'cache',Path(td)/'out',setup,install_code)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(load.call_count,0 if install_code else 1)
                with ZipFile(archive) as z:
                    self.assertEqual(set(z.namelist()),{'preflight.json','preflight.log','setup.log'})
                    self.assertEqual(json.loads(z.read('preflight.json'))['status'],'failed')
                    self.assertIn(b'Traceback',z.read('preflight.log'))
                    self.assertEqual(z.read('setup.log'),b'test installation output')
                latest=json.loads((Path(td)/'out/latest_preflight.json').read_text())
                self.assertEqual(latest['archive'],str(archive));self.assertEqual(latest['status'],'failed')

    def test_success_archive_preserves_probe_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            model=SimpleNamespace(provenance={'checkpoint_sha256':'test'})
            probe={'quality_validated':False,'device':'cpu'}
            with patch.object(preflight,'load_components',return_value=model),patch.object(preflight,'probe',return_value=probe):
                with contextlib.redirect_stdout(io.StringIO()): result,archive=preflight.run(td,td)
            with ZipFile(archive) as z: saved=json.loads(z.read('preflight.json'))
            self.assertEqual(result['status'],'pass');self.assertEqual(saved['probe'],probe)
            self.assertEqual(saved['model_provenance'],model.provenance)
            self.assertIn('ocsd/direct_instances.py',saved['code_sha256'])

    def test_run_all_dev_default_and_all_legacy_stages_guarded(self):
        nb=json.loads((ROOT/'notebooks/RUN_ALL.ipynb').read_text())
        self.assertEqual(nb['metadata']['accelerator'],'GPU')
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        config={};exec(code[0],config)
        self.assertEqual(config['RUN_MODE'],'heldout3')
        self.assertEqual(config['RUN_ID'],'')
        # No imports, GPU access, downloads, model generation or uploads are reachable
        # in later cells when the earlier preflight succeeds OR fails.
        index=next(i for i,c in enumerate(code) if c.startswith('#@title 3. Install libraries'))
        later=code[index:]
        self.assertGreaterEqual(len(later),9)
        for source in later:
            tree=ast.parse(source)
            self.assertEqual(len(tree.body),1)
            self.assertIsInstance(tree.body[0],ast.If)
            self.assertEqual(ast.unparse(tree.body[0].test),"RUN_MODE in ('experiment', 'heldout', 'heldout2', 'heldout3')")
            for mode in ('direct_preflight','direct_canary','direct_dev','silhouette_dev','naturalness_dev'):
                scope={'RUN_MODE':mode};exec(source,scope)
                self.assertEqual(set(scope),{'RUN_MODE','__builtins__'})
        # The branch is an indentation guard only: historical scientific code is retained.
        self.assertIn('stages.generate(P, E', '\n'.join(later))
        self.assertIn('sync.try_push_results', '\n'.join(later))


if __name__=='__main__':unittest.main()
