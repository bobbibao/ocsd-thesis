"""Real tiny-model tests of M3 scheduler targets, reset and adapter rank changes."""
import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import open_clip
from tiny import make_engine
from ocsd.config import OCSDConfig
from ocsd.method import ObjectResult, learn_identity
from ocsd.sketch import Scene, SceneObject, CropInfo

class Training(unittest.TestCase):
    def test_v_prediction_repeatability_and_rank(self):
        with tempfile.TemporaryDirectory() as d:
            bpe=str(Path(open_clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz')
            e=make_engine(d,bpe,64)
            e.scheduler.register_to_config(prediction_type='v_prediction')
            mask=np.zeros((64,64),bool);mask[12:52,12:52]=True
            sketch=np.where(mask,0,255).astype(np.uint8)
            image=np.full((64,64,3),180,np.uint8)
            sc=Scene('s',sketch,[SceneObject('cat','cat',(12,12,52,52),mask,sketch)],'in a park','a cat')
            obj=ObjectResult(image,mask,CropInfo(0,0,64,64),0.,1,sketch)
            c=OCSDConfig(height=64,width=64,S1=1,S2=1,lora_rank=2,use_att_sep=False)
            with patch.object(e.scheduler,'get_velocity',wraps=e.scheduler.get_velocity) as velocity:
                learn_identity(e,sc,[obj],c,seed=17)
                self.assertEqual(velocity.call_count,2)
            learned=[p.detach().clone() for p in e.lora_params()]
            embedding=e.text_encoder.get_input_embeddings().weight[e.id_token_ids[0]].detach().clone()
            learn_identity(e,sc,[obj],c,seed=17)
            self.assertTrue(torch.allclose(embedding,e.text_encoder.get_input_embeddings().weight[e.id_token_ids[0]]))
            self.assertTrue(all(torch.allclose(a,b) for a,b in zip(learned,e.lora_params())))
            learn_identity(e,sc,[obj],c.replace(lora_rank=4),seed=17)
            self.assertEqual(e.unet.peft_config['ocsd'].r,4)
            self.assertTrue(all(torch.isfinite(p).all() for p in e.lora_params()))
            e.reset_identity()
            self.assertFalse(e.text_encoder.get_input_embeddings().weight.requires_grad)

if __name__=='__main__': unittest.main(verbosity=2)
