"""Review packet integrity: real anonymization, duplicate mapping and fail-closed provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.build_human_review import build

class HumanReview(unittest.TestCase):
    def test_verified_anonymous_packet_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';root.mkdir();out=Path(tmp)/'out';artifacts={}
            def put(rel,data):
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                artifacts[rel]=hashlib.sha256(data).hexdigest()
            rp='results/runs/test/'
            key={'scene_s0':dict(zip('ABCD',['secret_method_1','secret_method_2','secret_method_3','secret_method_4']))}
            put(rp+'review_key.json',json.dumps(key).encode())
            put(rp+'review_blind/inputs/scene.png',b'sketch')
            spec={'caption':'caption </script><script>bad()</script>','objects':[]}
            for i,method in enumerate(key['scene_s0'].values()):
                stem='outputs/runs/test/quickdraw/'+method+'/scene_s0'
                put(stem+'.png',b'same' if i<2 else str(i).encode())
                put(stem+'.json',json.dumps({'scene_spec':spec}).encode())
            (root/'run_manifest.json').write_text(json.dumps({'run_id':'test','artifacts':artifacts}))
            result=build(root,out)
            self.assertEqual(result['original_rows'],4);self.assertEqual(result['unique_rating_cards'],3)
            html=(out/'REVIEW.html').read_text()
            self.assertNotIn('secret_method_',html)
            self.assertNotIn('caption </script>',html)
            raw=html.split('<script id="payload" type="application/json">')[1].split('</script>')[0]
            payload=json.loads(raw)
            self.assertEqual(payload['scenes'][0]['groups'][0]['labels'],['A','B'])
            self.assertEqual(payload['scenes'][0]['caption'],spec['caption'])
            (root/rp/'review_blind/inputs/scene.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed manifest'): build(root,out)

if __name__=='__main__':unittest.main()
