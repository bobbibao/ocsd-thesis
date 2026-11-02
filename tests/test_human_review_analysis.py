"""Do not silently turn comments into scores or count duplicate exports as new votes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.analyze_human_review import analyze, CRITERIA

class HumanReviewAnalysis(unittest.TestCase):
    def test_roundtrip_missing_scores_and_reject_misattribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';root.mkdir();out=Path(tmp)/'out';artifacts={}
            def put(rel,content):
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
                artifacts[rel]=hashlib.sha256(content).hexdigest()
            key={'s_s0':{'A':'method_a','B':'method_b'}}
            put('results/runs/test/review_key.json',json.dumps(key).encode())
            for method in key['s_s0'].values():put(f'outputs/runs/test/quickdraw/{method}/s_s0.png',b'identical image')
            (root/'run_manifest.json').write_text(json.dumps({'run_id':'test','artifacts':artifacts}))
            d=dict(schema='ocsd-human-review-v1',run_id='test',ratings={'s_s0:AB':{'count':'2','comment':'Missing rabbit; leave shape unscored'}},rows=[])
            for label in 'AB':
                d['rows'].append(dict(item='s_s0',label=label,group='s_s0:AB',image_sha256=hashlib.sha256(b'identical image').hexdigest(),
                                     **{c:'2' if c=='count' else '' for c in CRITERIA},comment=d['ratings']['s_s0:AB']['comment']))
            path=Path(tmp)/'review.json'
            def run(obj):
                path.write_text(json.dumps(obj));return analyze(root,path,out)
            summary=run(d)
            self.assertEqual(summary['unique_numeric_answers'],1)
            self.assertEqual(summary['expanded_method_rows'],2)
            self.assertEqual(summary['unique_image_groups'],1)
            variants=[]
            bad=copy.deepcopy(d);bad['run_id']='wrong';variants.append(bad)
            bad=copy.deepcopy(d);bad['rows'][0]['image_sha256']='wrong';variants.append(bad)
            bad=copy.deepcopy(d);bad['rows'].append(bad['rows'][0]);variants.append(bad)
            bad=copy.deepcopy(d);bad['rows'][0]['count']='1';variants.append(bad)
            bad=copy.deepcopy(d);bad['ratings']['s_s0:AB']['shape']='5';variants.append(bad)
            bad=copy.deepcopy(d);bad['rows'].pop();variants.append(bad)
            for bad in variants:
                with self.assertRaises(ValueError):run(bad)
            (root/'outputs/runs/test/quickdraw/method_a/s_s0.png').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Changed artifact'):run(d)

if __name__=='__main__':unittest.main()
