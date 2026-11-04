"""Input-contract and official-core parity; no generated quality claims."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ocsd.direct_instances import (scene_condition, sketch_points, conditioning_batch,
                                   cached_conditioning, sampling_work, sample_latents, SOURCE_COMMIT, load_global_text_weights,
                                   fusion_weights, mis_sampler_class, sketch_silhouette, mask_polygon,
                                   rle_encode, rle_decode)


def scene():
    objects=[]
    for box in ((3,2,19,10),(23,9,36,18)):
        ink=np.full((20,40),255,np.uint8);x0,y0,x1,y1=box
        ink[y0:y1,x0]=0;ink[y1-1,x0:x1]=0
        objects.append(SimpleNamespace(cls='bicycle',phrase='yellow bicycle',box=box,sketch=ink))
    return SimpleNamespace(n=2,objects=objects,sketch=np.minimum(objects[0].sketch,objects[1].sketch),sid='test',caption='two yellow bicycles')


class DirectInputs(unittest.TestCase):
    def test_text_only_pooler_matches_official_clip_feature(self):
        from transformers import CLIPConfig, CLIPModel, CLIPTextModel
        config=CLIPConfig(text_config=dict(vocab_size=100,hidden_size=32,intermediate_size=48,num_hidden_layers=1,
            num_attention_heads=4,max_position_embeddings=8,bos_token_id=1,eos_token_id=2,pad_token_id=0),
            vision_config=dict(hidden_size=32,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,
                               image_size=32,patch_size=16),projection_dim=16)
        full=CLIPModel(config).eval();text=CLIPTextModel(config.text_config).eval()
        state={'transformer.text_model.'+k:v for k,v in full.text_model.state_dict().items()}
        state['transformer.text_model.embeddings.position_ids']=torch.arange(8)[None]
        self.assertEqual(load_global_text_weights(text,state),['embeddings.position_ids'])
        inputs=dict(input_ids=torch.tensor([[1,7,2,0],[1,8,9,2]]),attention_mask=torch.tensor([[1,1,1,0],[1,1,1,1]]))
        # Transformers 5 mutates CLIPOutput's text pooler to its projected feature.
        # Capture the text branch before that mutation: upstream training used v4's
        # unchanged pre-projection pooler. Different projection width catches a mix-up.
        before=[]
        hook=full.text_model.register_forward_hook(lambda m,a,out: before.append(out.pooler_output.clone()))
        try:
            with torch.no_grad():
                output=full(**inputs,pixel_values=torch.ones(2,3,32,32))
                self.assertTrue(torch.equal(before[0],text(**inputs).pooler_output))
                self.assertEqual(text(**inputs).pooler_output.shape,(2,32))
                self.assertEqual(output.text_model_output.pooler_output.shape,(2,16))
        finally:hook.remove()
        # The public CLIP source contains a full text+vision checkpoint. Exercise
        # the actual HF loader's prefix conversion when loading only its text tower.
        with tempfile.TemporaryDirectory() as td:
            full.save_pretrained(td)
            loaded,info=CLIPTextModel.from_pretrained(td,output_loading_info=True)
            self.assertFalse(info['missing_keys']);self.assertFalse(info['mismatched_keys'])
            with torch.no_grad():self.assertTrue(torch.equal(loaded(**inputs).pooler_output,before[0]))
        wrong=dict(state);wrong.pop('transformer.text_model.embeddings.token_embedding.weight')
        with self.assertRaisesRegex(RuntimeError,'Missing key'):load_global_text_weights(text,wrong)
        wrong=dict(state);wrong['transformer.text_model.embeddings.position_ids']=torch.zeros(1,8,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'Noncanonical'):load_global_text_weights(text,wrong)

    def test_distinct_instances_real_ink_and_axis_normalization(self):
        s=scene();c=scene_condition(s)
        self.assertEqual(len(c['objects']),2)
        self.assertEqual(c['objects'][0]['box'],[3/40,2/20,19/40,10/20])
        for obj,item in zip(s.objects,c['objects']):
            points=np.array(item['scribble']).reshape(20,2)
            pixels=np.rint(points*[40,20]).astype(int)
            self.assertTrue((obj.sketch[pixels[:,1],pixels[:,0]]==0).all())
            self.assertEqual(item['scribble'],sketch_points(obj.sketch))
        b=conditioning_batch(c,torch.ones(2,32))
        self.assertEqual(int(b['masks'].sum()),2)
        self.assertFalse(torch.equal(b['scribbles'][0,0],b['scribbles'][0,1]))
        self.assertFalse(b['segs'].any());self.assertFalse(b['polygons'].any())
        self.assertFalse(b['masks'][0,2:].any())
        self.assertEqual(sketch_points(np.array([[0]],np.uint8)),[0.]*40)

    def test_invalid_inputs_fail_instead_of_silent_box_fallback(self):
        s=scene();s.objects[0].sketch[:]=255
        with self.assertRaisesRegex(ValueError,'Empty'):scene_condition(s)
        s=scene();s.objects[0].box=(-1,2,9,10)
        with self.assertRaisesRegex(ValueError,'outside'):scene_condition(s)
        s=scene();s.n=31
        with self.assertRaisesRegex(ValueError,'1..30'):scene_condition(s)
        self.assertEqual(sampling_work(3),216)
        self.assertEqual(sampling_work(5),292)
        self.assertEqual(sampling_work(5,mis=0),102)
        with self.assertRaises(ValueError):sampling_work(3,steps=30)

    def test_cache_invalidates_mutation_and_restores_after_exception(self):
        net=torch.nn.Linear(2,2).eval();x=torch.ones(1,2);original=net.forward
        with torch.no_grad(),cached_conditioning(net) as stats:
            a=net(x);self.assertIs(a,net(x));x.add_(1);b=net(x)
            self.assertFalse(torch.equal(a,b));self.assertEqual(stats['evaluations'],2)
        self.assertEqual(net.forward,original)
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            with cached_conditioning(net):net(x)
        self.assertEqual(net.forward,original)


class Silhouettes(unittest.TestCase):
    def outline(self, closed):
        ink=np.full((200,200),255,np.uint8)
        ink[40:160,50]=0;ink[40:160,150]=0;ink[159,50:151]=0  # U: left, right, bottom
        if closed: ink[40,50:151]=0
        return ink

    def test_open_and_closed_outlines_fill_inside_but_not_outside(self):
        for closed in (False,True):
            mask,info=sketch_silhouette(self.outline(closed))
            self.assertTrue(mask[100,100]);self.assertFalse(mask[20,100]);self.assertFalse(mask[100,10])
            self.assertTrue(mask[self.outline(closed)<128].all())
            self.assertGreater(info['mask_px'],5*info['ink_px'])
        self.assertEqual(sketch_silhouette(self.outline(False))[1]['bridges'],1)

    def test_lines_stay_thin_and_parser_is_deterministic(self):
        ink=np.full((200,200),255,np.uint8);ink[100,20:180]=0
        mask,info=sketch_silhouette(ink)
        self.assertLess(info['mask_px'],4*info['ink_px'])
        again,_=sketch_silhouette(ink);self.assertTrue(np.array_equal(mask,again))
        with self.assertRaises(ValueError):sketch_silhouette(np.full((20,20),255,np.uint8))

    def test_polygon_matches_upstream_layout_and_rle_roundtrip(self):
        mask,_=sketch_silhouette(self.outline(True))
        poly=mask_polygon(mask);self.assertEqual(len(poly),512)
        self.assertTrue(all(0<=v<=1 for v in poly));self.assertEqual(poly,mask_polygon(mask))
        pts=np.array(poly).reshape(-1,2);d=np.hypot(pts[:,0]*200,pts[:,1]*200)
        self.assertTrue(np.all(np.diff(d)>=-1e-9))  # sorted by distance to the origin, as upstream
        self.assertTrue(np.array_equal(rle_decode(rle_encode(mask),mask.shape),mask))

    def test_condition_and_batch_carry_silhouettes_only_when_requested(self):
        plain=scene_condition(scene());self.assertNotIn('silhouette_rle',plain['objects'][0])
        batch=conditioning_batch(plain,torch.randn(2,32))
        self.assertEqual(float(batch['segs'].abs().sum()),0);self.assertEqual(float(batch['polygons'].abs().sum()),0)
        cond=scene_condition(scene(),silhouettes=True);self.assertEqual(cond['schema'],'ocsd-direct-instances-v2')
        self.assertIn('no oracle masks',cond['input_track'])
        batch=conditioning_batch(cond,torch.randn(2,32))
        self.assertEqual(tuple(batch['segs'].shape),(1,30,512,512))
        self.assertGreater(float(batch['segs'][0,0].sum()),0);self.assertGreater(float(batch['segs'][0,1].sum()),0)
        self.assertEqual(float(batch['segs'][0,2:].sum()),0);self.assertGreater(float(batch['polygons'][0,:2].abs().sum()),0)
        # Instance 0 lives in the left part of the 40-px-wide canvas, instance 1 in the right part.
        cols0=torch.nonzero(batch['segs'][0,0].sum(0))[:,0];cols1=torch.nonzero(batch['segs'][0,1].sum(0))[:,0]
        self.assertLess(int(cols0.max()),int(cols1.min())+64)
        self.assertLess(float(cols0.float().mean()),float(cols1.float().mean()))


@unittest.skipUnless(os.environ.get('OCSD_INSTDIFF_SOURCE'), 'Set OCSD_INSTDIFF_SOURCE to the pinned official checkout for parity tests')
class OfficialCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess
        source=os.environ['OCSD_INSTDIFF_SOURCE']
        assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==SOURCE_COMMIT
        sys.path.insert(0,source)
        from ldm.modules.diffusionmodules.openaimodel import UNetModel
        from ldm.models.diffusion.ldm import LatentDiffusion
        torch.set_num_threads(2);torch.manual_seed(34)
        cls.model=UNetModel(image_size=8,in_channels=4,out_channels=4,model_channels=32,
            attention_resolutions=[1,2],num_res_blocks=1,channel_mult=[1,2],num_heads=4,context_dim=32,
            fuser_type='gatedSA',sd_v1_5=True,efficient_attention=True,
            grounding_tokenizer=dict(target='ldm.modules.diffusionmodules.text_grounding_net.UniFusion',params=dict(
                in_dim=32,out_dim=32,mid_dim=48,train_add_masks=False,test_drop_masks=True,test_drop_scribbles=False)))
        cls.model.eval().requires_grad_(False)
        with torch.no_grad():
            cls.model.out[-1].weight.normal_(0,.02)
            for name,param in cls.model.named_parameters():
                if 'alpha_attn' in name or 'alpha_dense' in name:param.fill_(.4)
                if name.endswith('.proj_out.weight'):param.normal_(0,.02)
        cls.diffusion=LatentDiffusion(timesteps=1000,linear_start=.00085,linear_end=.012)

    def test_full_unifusion_including_unused_mask_backbone(self):
        from ldm.modules.diffusionmodules import text_grounding_net as grounding
        factory=grounding.convnext_tiny
        def no_download(**kwargs):return factory(**dict(kwargs,pretrained=False))
        with patch.object(grounding,'convnext_tiny',side_effect=no_download):
            net=grounding.UniFusion(in_dim=32,out_dim=32,mid_dim=48,train_add_masks=True,
                                   test_drop_masks=True,test_drop_scribbles=False).eval()
        batch=conditioning_batch(scene_condition(scene()),torch.randn(2,32))
        args=tuple(batch[k] for k in ('boxes','masks','positive_embeddings','scribbles','polygons','segs','points'))
        with torch.no_grad(),cached_conditioning(net) as stats:
            first=net(*args)[0];second=net(*args)[0]
        with torch.no_grad():uncached=net(*args)[0]
        self.assertEqual(first.shape,(1,184,32))
        self.assertTrue(torch.isfinite(first).all());self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(first,uncached))
        self.assertEqual(stats,dict(calls=2,evaluations=1,hits=1))
        # Silhouette masks reach the output only when the mask branch is enabled (upstream test_mask flag).
        sil=conditioning_batch(scene_condition(scene(),silhouettes=True),batch['positive_embeddings'][0,:2].clone())
        sargs=tuple(sil[k] for k in ('boxes','masks','positive_embeddings','scribbles','polygons','segs','points'))
        with torch.no_grad():
            dropped=net(*sargs)[0];net.test_drop_masks=False;used=net(*sargs)[0];net.test_drop_masks=True
        self.assertTrue(torch.isfinite(used).all());self.assertEqual(used.shape,dropped.shape)
        self.assertGreater(float((used-dropped).abs().max()),1e-6)

    def test_cached_and_uncached_official_trajectories_and_reuse(self):
        torch.manual_seed(73);model=self.model
        batch=conditioning_batch(scene_condition(scene()),torch.randn(2,32))
        noise=torch.randn(1,4,8,8);contexts=[torch.randn(1,5,32) for _ in range(3)];negative=torch.randn(1,5,32)
        original={k:v.clone() for k,v in model.input_blocks[0][0].state_dict().items()}
        base={k:v*.9 for k,v in original.items()}
        uncached,a=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2,cache=False)
        cached,b=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2)
        self.assertTrue(torch.equal(uncached,cached))
        self.assertEqual(a['unet_sample_calls'],34);self.assertEqual(b['unet_sample_calls'],34)
        self.assertEqual(a['conditioning_cache']['evaluations'],34)
        self.assertEqual(b['conditioning_cache']['evaluations'],4)
        self.assertEqual(b['base_conv_restorations'],1)
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.input_blocks[0][0].state_dict().items()))
        again,_=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2)
        self.assertTrue(torch.equal(cached,again))
        # A real control-path check: changing scribbles must affect the latent.
        changed={k:v.clone() for k,v in batch.items()};changed['scribbles'][:,:2]=.8
        other,_=sample_latents(model,self.diffusion,changed,contexts,negative,noise,base,steps=10,mis=.2)
        self.assertGreater(float((other-cached).abs().max()),1e-7)
        plain,c=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=0)
        self.assertTrue(torch.isfinite(plain).all());self.assertEqual(c['unet_sample_calls'],22)

    def test_official_mean_fusion_is_bit_identical_to_upstream_sampler(self):
        from ldm.models.diffusion.plms_instance import PLMSSamplerInst
        torch.manual_seed(91);model=self.model
        batch=conditioning_batch(scene_condition(scene()),torch.randn(2,32))
        noise=torch.randn(1,4,8,8);contexts=[torch.randn(1,5,32) for _ in range(3)];negative=torch.randn(1,5,32)
        base={k:v.clone()*.9 for k,v in model.input_blocks[0][0].state_dict().items()}
        ours,log=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2)
        with patch('ocsd.direct_instances.mis_sampler_class',side_effect=lambda f,d: PLMSSamplerInst):
            upstream,_=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2)
        self.assertTrue(torch.equal(ours,upstream))
        self.assertEqual(log['mis_fusion'],'official_mean');self.assertEqual(log['mis_merge']['merge_step'],2)
        self.assertEqual(len(log['mis_merge']['background_gap_to_global']),2)
        region,rlog=sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2,fusion='region')
        self.assertEqual(rlog['unet_sample_calls'],log['unet_sample_calls'])
        self.assertTrue(torch.isfinite(region).all());self.assertGreater(float((region-ours).abs().max()),1e-7)
        with self.assertRaisesRegex(ValueError,'Unknown MIS fusion'):
            sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2,fusion='paste')

    def test_region_fusion_keeps_global_background_and_halves_inside_boxes(self):
        boxes=torch.zeros(1,30,4);masks=torch.zeros(1,30)
        boxes[0,0]=torch.tensor([0,0,.5,.5]);boxes[0,1]=torch.tensor([.25,.25,.75,.75]);masks[0,:2]=1
        mean=fusion_weights(boxes,masks,8,'official_mean');self.assertTrue(torch.equal(mean,torch.ones(3,1,1,8,8)))
        hard=fusion_weights(boxes,masks,8,'region',feather=0)
        self.assertEqual(hard.shape,(3,1,1,8,8));self.assertTrue(torch.equal(hard[0],torch.ones(1,1,8,8)))
        self.assertEqual(float(hard[1,0,0,0,0]),1.);self.assertEqual(float(hard[1,0,0,7,7]),0.)
        stack=torch.stack([torch.full((1,1,8,8),v) for v in (0.,3.,6.)])
        fused=(stack*hard).sum(0)/hard.sum(0)
        self.assertEqual(float(fused[0,0,7,7]),0.)   # background: global only
        self.assertEqual(float(fused[0,0,0,0]),1.5)  # one box: (global+instance)/2
        self.assertEqual(float(fused[0,0,3,3]),3.)   # overlap: (0+3+6)/3
        soft=fusion_weights(boxes,masks,8,'region',feather=2)
        self.assertTrue(((soft>=0)&(soft<=1)).all());self.assertTrue(torch.equal(soft[0],hard[0]))

    def test_failed_sampling_restores_convolution_null_cache_and_scales(self):
        from ldm.modules.attention import GatedSelfAttentionDense
        model=self.model;null=model.grounding_tokenizer_input;forward=model.position_net.forward
        restore=model.restore_first_conv_from_SD
        fusers=[m for m in model.modules() if isinstance(m,GatedSelfAttentionDense)]
        scales=[m.scale for m in fusers]
        original={k:v.clone() for k,v in model.input_blocks[0][0].state_dict().items()}
        batch=conditioning_batch(scene_condition(scene()),torch.randn(2,32))
        def fail(**kwargs):
            model.restore_first_conv_from_SD()
            for m in fusers:m.scale=.123
            raise RuntimeError('sampling interrupted')
        with patch('ldm.models.diffusion.plms_instance.PLMSSamplerInst.sample',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'sampling interrupted'):
                sample_latents(model,self.diffusion,batch,[torch.randn(1,5,32)]*3,torch.randn(1,5,32),
                               torch.randn(1,4,8,8),{k:v*.9 for k,v in original.items()},steps=10,mis=.2)
        self.assertIs(model.grounding_tokenizer_input,null)
        self.assertEqual(model.restore_first_conv_from_SD,restore)
        self.assertEqual(model.position_net.forward,forward)
        self.assertEqual([m.scale for m in fusers],scales)
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.input_blocks[0][0].state_dict().items()))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model._forward_hooks)

    def test_timeout_and_nonfinite_stop_before_extra_unet_work(self):
        model=self.model;batch=conditioning_batch(scene_condition(scene()),torch.randn(2,32))
        noise=torch.randn(1,4,8,8);contexts=[torch.randn(1,5,32)]*3;negative=torch.randn(1,5,32)
        base={k:v.clone() for k,v in model.input_blocks[0][0].state_dict().items()}
        with self.assertRaises(TimeoutError) as caught:
            sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2,deadline=0)
        self.assertEqual(caught.exception.ocsd_compute['unet_calls'],0)
        with patch.object(model,'forward',return_value=torch.full_like(noise,float('nan'))):
            with self.assertRaisesRegex(RuntimeError,'Nonfinite UNet') as caught:
                sample_latents(model,self.diffusion,batch,contexts,negative,noise,base,steps=10,mis=.2)
        self.assertEqual(caught.exception.ocsd_compute['unet_calls'],1)
        self.assertFalse(model._forward_hooks);self.assertFalse(model._forward_pre_hooks)


if __name__=='__main__':unittest.main()
