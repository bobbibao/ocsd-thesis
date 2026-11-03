"""Direct instance conditioning using the pinned official InstanceDiffusion core.

This is an external baseline integration, not an OCSD architecture claim. No
diffusers fork is installed. Raw sketches are assumed already instance-labeled.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

SOURCE_URL = 'https://github.com/frank-xwang/InstanceDiffusion.git'
SOURCE_COMMIT = '722900ef03e3579fb693e65a1550333a4a162581'
MODEL_REPO = 'xudongw/InstanceDiffusion'
MODEL_FILE = 'instancediffusion_sd15.pth'
CLIP_REPO = 'openai/clip-vit-large-patch14'
MAX_OBJECTS = 30


def checkout_source(cache):
    path = Path(cache) / ('InstanceDiffusion-' + SOURCE_COMMIT[:12])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', '--no-checkout', SOURCE_URL, str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', SOURCE_COMMIT], check=True)
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
    if commit != SOURCE_COMMIT or dirty.strip():
        raise ValueError('InstanceDiffusion source must match the clean pinned commit')
    if 'ldm' in sys.modules:
        module_paths = list(getattr(sys.modules['ldm'], '__path__', ()))
        if not module_paths or any(not Path(p).resolve().is_relative_to(path.resolve()) for p in module_paths):
            raise RuntimeError('Another ldm package is already loaded; use a fresh process')
    sys.path.insert(0, str(path.resolve()))
    return path


def sketch_points(sketch, count=20):
    """Deterministic farthest-point coverage on real ink, then radial ordering.

    Points are x/y in [0,1), never interpolated through background. This is a
    lossy 20-point representation of an instance raster, not its full contour.
    """
    a = np.asarray(sketch)
    if a.ndim != 2 or a.dtype != np.uint8 or count != 20:
        raise ValueError('Expected uint8 grayscale instance sketch and 20 points')
    yy, xx = np.where(a < 128)
    if len(xx) == 0:
        raise ValueError('Empty instance sketch; refusing box-only substitution')
    xy = np.column_stack((xx, yy)).astype(np.float64)
    xy /= np.array([a.shape[1], a.shape[0]])
    start = int(np.argmin((xy**2).sum(1)))
    selected = [start]; nearest = ((xy - xy[start])**2).sum(1)
    for _ in range(min(count, len(xy))-1):
        index = int(np.argmax(nearest)); selected.append(index)
        nearest = np.minimum(nearest, ((xy - xy[index])**2).sum(1))
    selected = [selected[i % len(selected)] for i in range(count)]
    points = xy[selected]
    order = np.lexsort((points[:, 1], points[:, 0], (points**2).sum(1)))
    return points[order].reshape(-1).tolist()


def scene_condition(scene):
    height, width = scene.sketch.shape[:2]
    if not 1 <= scene.n <= MAX_OBJECTS or scene.n != len(scene.objects):
        raise ValueError('InstanceDiffusion supports 1..30 separate instances; no truncation')
    objects = []
    for index, obj in enumerate(scene.objects):
        box = np.asarray(obj.box, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid instance box')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Instance box outside the canvas or empty')
        if np.asarray(obj.sketch).shape != (height, width) or not obj.phrase.strip():
            raise ValueError('Misaligned sketch or missing instance phrase')
        objects.append(dict(index=index, cls=obj.cls, phrase=obj.phrase,
                            box=(box / [width, height, width, height]).tolist(),
                            scribble=sketch_points(obj.sketch)))
    return dict(schema='ocsd-direct-instances-v1', sid=scene.sid, caption=scene.caption,
                width=width, height=height, objects=objects,
                input_track='supplied instance phrases/boxes and 20 raster-ink points; no instance masks')


def conditioning_batch(condition, features, device='cpu', dtype=torch.float32):
    objects = condition['objects']; n = len(objects)
    if (not 1 <= n <= MAX_OBJECTS or features.ndim != 2 or features.shape[0] != n
            or not torch.isfinite(features).all()):
        raise ValueError('One text feature per separate instance is required')
    def zeros(*shape): return torch.zeros(*shape, device=device, dtype=dtype)
    batch = dict(boxes=zeros(1, MAX_OBJECTS, 4), masks=zeros(1, MAX_OBJECTS),
                 positive_embeddings=zeros(1, MAX_OBJECTS, features.shape[-1]),
                 scribbles=zeros(1, MAX_OBJECTS, 40), polygons=zeros(1, MAX_OBJECTS, 512),
                 segs=zeros(1, MAX_OBJECTS, 512, 512), points=zeros(1, MAX_OBJECTS, 2))
    batch['boxes'][0, :n] = torch.as_tensor([o['box'] for o in objects], device=device, dtype=dtype)
    batch['scribbles'][0, :n] = torch.as_tensor([o['scribble'] for o in objects], device=device, dtype=dtype)
    batch['positive_embeddings'][0, :n] = features.to(device=device, dtype=dtype)
    batch['masks'][0, :n] = 1
    batch['points'][0, :n] = (batch['boxes'][0, :n, :2] + batch['boxes'][0, :n, 2:]) / 2
    return batch


class NullCondition:
    """Reuse an immutable null batch instead of reallocating masks at every CFG step."""
    set = True
    def __init__(self, batch): self.batch = {k: torch.zeros_like(v) for k, v in batch.items()}
    def get_null_input(self, **kwargs):
        if kwargs: raise ValueError('This adapter samples one image at a time')
        return self.batch


@contextmanager
def cached_conditioning(position_net, enabled=True):
    """Request-local cache; never reused across weights, modalities or scenes."""
    stats = dict(calls=0, evaluations=0, hits=0)
    original = position_net.forward
    cache = {}
    def forward(*args):
        stats['calls'] += 1
        if not enabled:
            stats['evaluations'] += 1
            return original(*args)
        if position_net.training or torch.is_grad_enabled():
            raise RuntimeError('Conditioning cache is inference-only')
        # Keep tensors alive so an allocator cannot reuse an old identity. An
        # in-place edit bumps _version and must cause a new evaluation.
        key = tuple((id(a), a._version) if torch.is_tensor(a) else a for a in args)
        if key not in cache:
            cache[key] = (args, original(*args)); stats['evaluations'] += 1
        else:
            stats['hits'] += 1
        return cache[key][1]
    position_net.forward = forward
    try:
        yield stats
    finally:
        position_net.forward = original
        cache.clear()


def sampling_work(n_objects, steps=50, mis=.36):
    # Official PLMS uses an extra improved-Euler evaluation for the first step
    # of each trajectory. With 1000 train timesteps, require an exact divisor.
    if not 1 <= n_objects <= MAX_OBJECTS or steps < 2 or 1000 % steps or not 0 <= mis < 1:
        raise ValueError('Invalid sampling schedule')
    split = int(steps * mis)
    return 2 * (steps + (n_objects * split + n_objects + 1 if split else 1))


@torch.no_grad()
def sample_latents(model, diffusion, batch, contexts, negative, noise, base_conv,
                   steps=50, alpha=.8, mis=.36, guidance=7.5, cache=True):
    """Official PLMS/MIS with fixed input RNG and reversible per-image state.

    This preserves the official late SD input-convolution restoration and
    restores learned weights before the next image. It does not rerank outputs.
    """
    from ldm.models.diffusion.plms import PLMSSampler
    from ldm.models.diffusion.plms_instance import PLMSSamplerInst
    from ldm.modules.attention import GatedSelfAttentionDense
    n = int(batch['masks'].sum().item())
    expected = sampling_work(n, steps, mis)
    if len(contexts) != n+1 or not 0 < alpha <= 1 or guidance <= 1 or noise.shape[0] != 1:
        raise ValueError('Expected global and per-instance contexts, one CFG sample')
    if int(steps*mis) == 0: mis = 0
    model.eval()
    original_null = model.grounding_tokenizer_input
    null = NullCondition(batch)
    conv = model.input_blocks[0][0]
    learned_conv = {k: v.detach().clone() for k, v in conv.state_dict().items()}
    original_restore = model.restore_first_conv_from_SD
    restorations = 0
    def restore():
        nonlocal restorations
        if restorations == 0:
            conv.load_state_dict(base_conv, strict=True); restorations += 1
    fusers = [m for m in model.modules() if isinstance(m, GatedSelfAttentionDense)]
    scales = [m.scale for m in fusers]
    def scale(_, value):
        for m in fusers: m.scale = value
    def schedule(length):
        return [1.] * int(alpha*length) + [0.] * (length-int(alpha*length))
    calls = dict(unet_calls=0, unet_sample_calls=0)
    def count(_, args):
        calls['unet_calls'] += 1; calls['unet_sample_calls'] += args[0]['x'].shape[0]
    global_input = dict(x=noise.clone(), context=contexts[0], grounding_input=batch, timesteps=None)
    inputs = [global_input]
    if mis:
        for i in range(n):
            single = {k: torch.zeros_like(v) for k, v in batch.items()}
            for k in single: single[k][:, 0] = batch[k][:, i]
            inputs.append(dict(x=noise.clone(), context=contexts[i+1], grounding_input=single, timesteps=None))
    sampler = (PLMSSamplerInst(diffusion, model, alpha_generator_func=schedule, set_alpha_scale=scale, mis=mis)
               if mis else PLMSSampler(diffusion, model, alpha_generator_func=schedule, set_alpha_scale=scale))
    hook = model.register_forward_pre_hook(count)
    try:
        model.grounding_tokenizer_input = null
        model.restore_first_conv_from_SD = restore
        with cached_conditioning(model.position_net, cache) as cache_stats:
            latent = sampler.sample(S=steps, shape=tuple(noise.shape), input=inputs if mis else global_input,
                                    uc=negative, guidance_scale=guidance)
        if calls['unet_sample_calls'] != expected:
            raise RuntimeError('Official sampler work differs from the planned budget')
        if not torch.isfinite(latent).all(): raise RuntimeError('Nonfinite direct-generation latent')
        return latent, dict(**calls, conditioning_cache=cache_stats, base_conv_restorations=restorations)
    finally:
        hook.remove(); conv.load_state_dict(learned_conv, strict=True)
        model.restore_first_conv_from_SD = original_restore
        model.grounding_tokenizer_input = original_null
        for m, value in zip(fusers, scales): m.scale = value


def load_global_text_weights(encoder, state):
    """Map the upstream transformers-4 wrapper to the pinned transformers-5 text model.

    Only module prefixes and the historical non-learned position buffer differ;
    every learned parameter must still match strictly.
    """
    text_state, ignored = {}, []
    for key, value in state.items():
        if not key.startswith('transformer.text_model.'):
            raise ValueError('Unexpected text-encoder state key: ' + key)
        key = key.removeprefix('transformer.text_model.')
        if key == 'embeddings.position_ids':
            expected = torch.arange(encoder.config.max_position_embeddings)[None]
            if not torch.equal(value.cpu(), expected):
                raise ValueError('Noncanonical CLIP position IDs')
            ignored.append(key)
        else:
            text_state[key] = value
    encoder.load_state_dict(text_state, strict=True, assign=True)
    return ignored


def load_components(cache, lock=None):
    """Load only fixed public model sources; strict weights, no upstream training stack.

    First resolve HF revisions, then download at those exact revisions. The
    official PTH contains OmegaConf metadata and is a trusted-source pickle;
    arbitrary user checkpoint URLs/paths are intentionally not accepted here.
    """
    import yaml
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from .provenance import file_hash
    source = checkout_source(cache)
    api = HfApi(token=False)
    lock = dict(lock or {repo: api.model_info(repo).sha for repo in (MODEL_REPO, CLIP_REPO)})
    if set(lock) != {MODEL_REPO, CLIP_REPO} or any(not isinstance(v, str) or not re.fullmatch(r'[a-f0-9]{40}', v) for v in lock.values()):
        raise ValueError('Expected exact public checkpoint and CLIP revisions')
    weights = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=lock[MODEL_REPO], cache_dir=str(Path(cache)/'hf'), token=False)
    print('Loading official InstanceDiffusion checkpoint on CPU...', flush=True)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
    config = yaml.safe_load((source/'configs/test_box.yaml').read_text())
    from ldm.modules.diffusionmodules.openaimodel import UNetModel
    from ldm.modules.diffusionmodules import text_grounding_net
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.models.diffusion.ldm import LatentDiffusion
    # All ConvNeXt weights are supplied by the strict full checkpoint load.
    # Avoid downloading a second initializer that would immediately be replaced.
    original_factory = text_grounding_net.convnext_tiny
    def uninitialized_convnext(**kwargs):
        kwargs['pretrained'] = False
        return original_factory(**kwargs)
    text_grounding_net.convnext_tiny = uninitialized_convnext
    try:
        model = UNetModel(**config['model']['params'])
    finally:
        text_grounding_net.convnext_tiny = original_factory
    weight_key = 'ema' if 'ema' in checkpoint else 'model'
    model.load_state_dict(checkpoint[weight_key], strict=True, assign=True)
    vae = AutoencoderKL(**config['autoencoder']['params'])
    vae.load_state_dict(checkpoint['autoencoder'], strict=True, assign=True)
    diffusion = LatentDiffusion(**config['diffusion']['params'])
    diffusion.load_state_dict(checkpoint['diffusion'], strict=True, assign=True)
    kwargs = dict(revision=lock[CLIP_REPO], cache_dir=str(Path(cache)/'hf'), token=False)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_REPO, **kwargs)
    clip_config = CLIPTextConfig.from_pretrained(CLIP_REPO, **kwargs)
    global_encoder = CLIPTextModel(clip_config)
    ignored = load_global_text_weights(global_encoder, checkpoint['text_encoder'])
    # Upstream used transformers-4 CLIPModel.text_model_output before projection.
    # In v5 that field becomes projected; use the text-only model's pooler directly.
    # The unused CLIP vision forward is not needed for a text-only feature.
    phrase_encoder, phrase_load = CLIPTextModel.from_pretrained(CLIP_REPO, output_loading_info=True, **kwargs)
    if any(phrase_load.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('CLIP phrase encoder did not load all learned weights: ' + str(phrase_load))
    base_path = source/'pretrained/SD_v1_5_input_conv_weight_bias.pth'
    base_conv = torch.load(base_path, map_location='cpu', weights_only=True)
    provenance = dict(source_commit=SOURCE_COMMIT, model_revisions=lock,
                      checkpoint_sha256=file_hash(weights), checkpoint_state=weight_key,
                      phrase_text_weights_complete=True,
                      base_conv_sha256=file_hash(base_path), ignored_nonlearned_text_buffers=ignored,
                      config_sha256=file_hash(source/'configs/test_box.yaml'))
    for module in (model, vae, diffusion, global_encoder, phrase_encoder): module.float().eval().requires_grad_(False)
    return DirectModel(model, vae, diffusion, global_encoder, phrase_encoder, tokenizer, base_conv, provenance)


class DirectModel:
    def __init__(self, model, vae, diffusion, global_encoder, phrase_encoder, tokenizer, base_conv, provenance):
        self.model, self.vae, self.diffusion = model, vae, diffusion
        self.global_encoder, self.phrase_encoder, self.tokenizer = global_encoder, phrase_encoder, tokenizer
        self.base_conv, self.provenance = base_conv, provenance

    @torch.no_grad()
    def encode(self, condition, negative):
        phrases = [o['phrase'] for o in condition['objects']]
        tokens = self.tokenizer(phrases, padding=True, truncation=True, max_length=77, return_tensors='pt')
        features = self.phrase_encoder(**tokens).pooler_output
        strings = [condition['caption'], *phrases, negative]
        tokens = self.tokenizer(strings, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        embeddings = self.global_encoder(input_ids=tokens['input_ids']).last_hidden_state
        return features, [x[None] for x in embeddings[:-1]], embeddings[-1:]

    @torch.no_grad()
    def generate(self, condition, seed, negative, scribble=True, device='cuda', steps=50, mis=.36, alpha=.8):
        start = time.perf_counter()
        features, contexts, uncond = self.encode(condition, negative)
        batch = conditioning_batch(condition, features, device=device)
        self.model.to(device); self.vae.to(device); self.diffusion.to(device)
        self.model.position_net.test_drop_scribbles = not scribble
        self.model.position_net.test_drop_masks = True  # never consume benchmark masks
        contexts = [c.to(device) for c in contexts]; uncond = uncond.to(device)
        noise = torch.randn((1,4,64,64), generator=torch.Generator('cpu').manual_seed(seed)).to(device)
        autocast = torch.autocast(device_type='cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with autocast:
            latent, log = sample_latents(self.model, self.diffusion, batch, contexts, uncond, noise, self.base_conv,
                                        steps=steps, mis=mis, alpha=alpha)
            image = self.vae.decode(latent)
        if not torch.isfinite(image).all(): raise RuntimeError('Nonfinite direct-generation image')
        pixels = ((image[0].clamp(-1,1)*.5+.5)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        log.update(provenance=self.provenance, steps=steps, mis=mis, alpha=alpha, seed=seed,
                   conditioning='boxes+centers+20_ink_points' if scribble else 'boxes+centers',
                   wall_s=time.perf_counter()-start, image_size=512, refiner=False,
                   external_method='InstanceDiffusion pinned official core with OCSD input adapter')
        return pixels, log
