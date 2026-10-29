"""Run the actual CPU export notebook with a small part budget and fake Colab downloads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import nbformat

class ReviewExport(unittest.TestCase):
    def test_parts_preserve_all_files_and_reject_corruption(self):
        notebook = Path(__file__).resolve().parents[1]/'notebooks/03_export_screen_review.ipynb'
        nb = nbformat.read(notebook, as_version=4)
        nbformat.validate(nb)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'root'; rid='test_run'
            results=root/'results/runs'/rid; results.mkdir(parents=True)
            (root/'results/latest_run.json').write_text(json.dumps({'run_id':rid}))
            selected={}
            for i in range(3):
                relative=f'outputs/runs/{rid}/quickdraw/grounded_repair_sparse/s{i}.png'
                p=root/relative; p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(bytes([i])*2**20)
                selected[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
            relative=f'results/runs/{rid}/summary.json'
            (root/relative).write_text('{}');selected[relative]=hashlib.sha256(b'{}').hexdigest()
            manifest={'status':'complete','contract':{'experiment':{'tier':'repair_screen'}},'artifacts':selected}
            (results/'run_manifest.json').write_text(json.dumps(manifest))
            code=nb.cells[1].source.replace("Path('/content/drive/MyDrive/KLTN_OCSD')",f'Path({str(root)!r})')
            code=code.replace("f'/content/ocsd_review_",f"f'{temp}/ocsd_review_")
            code=code.replace('MAX_PART_BYTES = 24 * 2**20','MAX_PART_BYTES = 2 * 2**20')
            downloaded=[];colab=types.ModuleType('google.colab')
            colab.drive=types.SimpleNamespace(mount=lambda _:None)
            colab.files=types.SimpleNamespace(download=downloaded.append)
            with patch.dict(sys.modules,{'google.colab':colab}):
                exec(compile(code,str(notebook),'exec'),{})
                self.assertEqual(len(downloaded),4)
                self.assertIn('_metrics_',downloaded[0])
                found={}
                for archive in downloaded:
                    self.assertLess(Path(archive).stat().st_size,32*2**20)
                    with zipfile.ZipFile(archive) as z:
                        self.assertIn('run_manifest.json',z.namelist())
                        for rel in z.namelist():
                            if rel!='run_manifest.json':
                                self.assertNotIn(rel,found)
                                found[rel]=hashlib.sha256(z.read(rel)).hexdigest()
                self.assertEqual(found,selected)
                (root/relative).write_text('corrupted')
                downloaded.clear()
                with self.assertRaisesRegex(AssertionError,'Changed:'):
                    exec(compile(code,str(notebook),'exec'),{})
                self.assertFalse(downloaded)

if __name__=='__main__':unittest.main(verbosity=2)
