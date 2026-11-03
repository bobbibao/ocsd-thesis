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

    def test_run_all_cpu_default_and_all_legacy_stages_guarded(self):
        nb=json.loads((ROOT/'notebooks/RUN_ALL.ipynb').read_text())
        self.assertNotIn('accelerator',nb['metadata'])
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        config={};exec(code[0],config)
        self.assertEqual(config['RUN_MODE'],'direct_preflight')
        self.assertEqual(config['RUN_ID'],'')
        # No imports, GPU access, downloads, model generation or uploads are reachable
        # in later cells when the earlier preflight succeeds OR fails.
        later=code[3:]
        self.assertGreaterEqual(len(later),9)
        for source in later:
            tree=ast.parse(source)
            self.assertEqual(len(tree.body),1)
            self.assertIsInstance(tree.body[0],ast.If)
            self.assertEqual(ast.unparse(tree.body[0].test),"RUN_MODE == 'experiment'")
            scope={'RUN_MODE':'direct_preflight'};exec(source,scope)
            self.assertEqual(set(scope),{'RUN_MODE','__builtins__'})
        # The branch is an indentation guard only: historical scientific code is retained.
        self.assertIn('stages.generate(P, E', '\n'.join(later))
        self.assertIn('sync.try_push_results', '\n'.join(later))


if __name__=='__main__':unittest.main()
