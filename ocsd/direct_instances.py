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


SILHOUETTE_CLOSE = 0.03     # closing radius as a fraction of the ink-box diagonal; fixed before output
SILHOUETTE_BRIDGE = 1.0     # join mutual-nearest stroke ends up to one ink-box diagonal apart
POLYGON_POINTS = 256


def _disk(radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return xx*xx + yy*yy <= radius*radius


def _stroke_ends(ink):
    """End pixels of the 1-px skeleton (exactly one 8-neighbour)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    skel = skeletonize(ink)
    neighbours = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - 1
    return np.argwhere(skel & (neighbours == 1))


def sketch_silhouette(sketch):
    """OCSD sketch parser: instance strokes (raster) -> filled silhouette mask. No oracle masks.

    QuickDraw strokes are open line drawings (a car body drawn as an arc, a tree as a U).
    1. Bridge open outlines: join each pair of mutually nearest stroke ends (skeleton end
       pixels) with a straight line if they are at most 1.0 x the ink-box diagonal apart.
    2. Close small gaps with a disk of radius 0.03 x diagonal and fill enclosed holes.
    The ink itself is always kept. Deterministic; returns (bool mask, info).
    """
    from scipy import ndimage
    from PIL import Image, ImageDraw
    a = np.asarray(sketch)
    if a.ndim != 2: raise ValueError('Expected a 2-D instance sketch')
    ink = a < 128
    yy, xx = np.where(ink)
    if not len(xx): raise ValueError('Empty instance sketch')
    diag = float(np.hypot(xx.max()-xx.min()+1, yy.max()-yy.min()+1))
    ends = _stroke_ends(ink)
    bridges = []
    if len(ends) >= 2:
        d = np.sqrt(((ends[:, None, :]-ends[None, :, :])**2).sum(-1)); np.fill_diagonal(d, np.inf)
        nearest = d.argmin(1)
        for i, j in enumerate(nearest):
            if i < j and nearest[j] == i and d[i, j] <= SILHOUETTE_BRIDGE*diag:
                bridges.append((ends[i].tolist(), ends[j].tolist()))
    canvas = Image.fromarray(ink.astype(np.uint8)*255); draw = ImageDraw.Draw(canvas)
    for (y0, x0), (y1, x1) in bridges: draw.line([(x0, y0), (x1, y1)], fill=255, width=3)
    joined = np.asarray(canvas) > 127
    r = max(2, int(round(SILHOUETTE_CLOSE*diag)))
    closed = ndimage.binary_closing(np.pad(joined, r+1), structure=_disk(r))[r+1:-r-1, r+1:-r-1]
    mask = ndimage.binary_fill_holes(closed) | ink
    return mask, dict(close_radius_px=r, stroke_ends=int(len(ends)), bridges=len(bridges),
                      ink_px=int(ink.sum()), mask_px=int(mask.sum()))


def mask_polygon(mask, k=POLYGON_POINTS):
    """Upstream `sample_sparse_points_from_mask` (half boundary, half interior, sorted by
    distance to the origin), made deterministic, normalised to [0, 1] like `convert_points`."""
    import random
    from skimage import measure
    m = np.asarray(mask).astype(np.int32)
    height, width = m.shape
    contours = measure.find_contours(np.pad(m, 1), 0)
    boundary = []
    for c in contours:
        if not np.array_equal(c[0], c[-1]): c = np.vstack((c, c[0]))
        c = measure.approximate_polygon(c, 0)
        if len(c) < 3: continue
        boundary += [max(0, v) for v in np.flip(c, axis=1).ravel().tolist()]
    if not boundary: raise ValueError('Silhouette has no contour')
    boundary = np.array(boundary).reshape(-1, 2)
    half = k // 2
    boundary = [tuple(boundary[int(i)]) for i in np.linspace(0, len(boundary)-1, half)]
    fg = np.argwhere(m == 1); need = k - len(boundary)
    if len(fg) < need:
        rng = random.Random(0); inside = [tuple(rng.choice(list(fg))[::-1]) for _ in range(need)]
    else:
        center = fg.mean(0); fg = sorted(fg, key=lambda x: np.linalg.norm(x-center)); per = len(fg)//need
        inside = [tuple(fg[i*per:(i+1)*per][len(fg[i*per:(i+1)*per])//2][::-1]) for i in range(need)]
    points = sorted(inside + boundary, key=lambda x: np.linalg.norm(np.array(x, dtype=float)))
    flat = []
    for x, y in points: flat += [min(float(x)/width, 1.), min(float(y)/height, 1.)]
    return flat


def rle_encode(mask):
    flat = np.asarray(mask, dtype=bool).ravel()
    change = np.flatnonzero(np.diff(np.concatenate(([0], flat.view(np.uint8), [0]))))
    return change.reshape(-1, 2).tolist()


def rle_decode(runs, shape):
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    for start, stop in runs: flat[start:stop] = True
    return flat.reshape(shape)


def scene_condition(scene, silhouettes=False):
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
        item = dict(index=index, cls=obj.cls, phrase=obj.phrase,
                    box=(box / [width, height, width, height]).tolist(),
                    scribble=sketch_points(obj.sketch))
        if silhouettes:
            mask, info = sketch_silhouette(obj.sketch)
            item.update(silhouette_rle=rle_encode(mask), silhouette=info, polygon=mask_polygon(mask))
        objects.append(item)
    track = ('supplied instance phrases/boxes, 20 raster-ink points and silhouettes parsed from the instance '
             'sketch (OCSD sketch parser); no oracle masks') if silhouettes else \
            'supplied instance phrases/boxes and 20 raster-ink points; no instance masks'
    return dict(schema='ocsd-direct-instances-v2' if silhouettes else 'ocsd-direct-instances-v1',
                sid=scene.sid, caption=scene.caption, width=width, height=height, objects=objects, input_track=track)


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
    if all('silhouette_rle' in o for o in objects):
        from PIL import Image
        shape = (condition['height'], condition['width'])
        for i, o in enumerate(objects):
            seg = Image.fromarray(rle_decode(o['silhouette_rle'], shape).astype(np.uint8)*255)
            if seg.size != (512, 512): seg = seg.resize((512, 512), Image.Resampling.NEAREST)
            batch['segs'][0, i] = torch.as_tensor(np.asarray(seg) > 127, device=device, dtype=dtype)
            batch['polygons'][0, i] = torch.as_tensor(o['polygon'], device=device, dtype=dtype)
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
    own = 'forward' in vars(position_net)
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
        key = tuple((id(a), a._version) if torch.is_tensor(a) else a for a in args) + tuple(
            getattr(position_net, f, None) for f in ('test_drop_masks', 'test_drop_scribbles', 'test_drop_boxes'))
        if key not in cache:
            cache[key] = (args, original(*args)); stats['evaluations'] += 1
        else:
            stats['hits'] += 1
        return cache[key][1]
    position_net.forward = forward
    try:
        yield stats
    finally:
        if own: position_net.forward = original
        else: del position_net.forward
        cache.clear()


def sampling_work(n_objects, steps=50, mis=.36):
    # Official PLMS uses an extra improved-Euler evaluation for the first step
    # of each trajectory. With 1000 train timesteps, require an exact divisor.
    if not 1 <= n_objects <= MAX_OBJECTS or steps < 2 or 1000 % steps or not 0 <= mis < 1:
        raise ValueError('Invalid sampling schedule')
    split = int(steps * mis)
    return 2 * (steps + (n_objects * split + n_objects + 1 if split else 1))


FUSIONS = ('official_mean', 'region')
REGION_FEATHER = 2  # latent pixels (16 image pixels); fixed before any GPU output


def fusion_weights(boxes, masks, size, fusion='official_mean', feather=REGION_FEATHER):
    """Per-trajectory weights for the MIS merge, shape (n+1, 1, 1, size, size).

    official_mean: every trajectory has weight 1 everywhere, i.e. the upstream
    torch.mean over the global and single-instance latents.
    region: the global trajectory keeps weight 1 everywhere; instance i only
    contributes inside its own box (soft edge of `feather` latent pixels). So the
    background comes from the full-caption trajectory instead of an average of
    n+1 independently composed backgrounds, and an instance's own trajectory
    gets 1/2 instead of 1/(n+1) of its box. This is our variant, not upstream.
    """
    if fusion not in FUSIONS:
        raise ValueError('Unknown MIS fusion: ' + str(fusion))
    n = int(masks.sum().item())
    ones = torch.ones(n+1, 1, 1, size, size, dtype=torch.float32)
    if fusion == 'official_mean':
        return ones
    grid = (torch.arange(size, dtype=torch.float32)+.5)/size
    for i in range(n):
        x0, y0, x1, y1 = [float(v) for v in boxes[0, i]]
        def ramp(lo, hi):
            if feather <= 0:
                return ((grid >= lo) & (grid < hi)).float()
            width = feather/size
            return ((grid-lo)/width+.5).clamp(0, 1) * ((hi-grid)/width+.5).clamp(0, 1)
        ones[i+1, 0, 0] = ramp(y0, y1)[:, None] * ramp(x0, x1)[None, :]
    return ones


def mis_sampler_class(fusion='official_mean', diagnostics=None, on_merge=None):
    """Official PLMSSamplerInst with only the latent merge made pluggable.

    With official_mean the loop is the upstream code path and must reproduce
    upstream latents exactly (tested against the pinned source).
    """
    from ldm.models.diffusion.plms_instance import PLMSSamplerInst

    class FusionPLMSSamplerInst(PLMSSamplerInst):
        @torch.no_grad()
        def plms_sampling(self, shape, input_all, uc=None, guidance_scale=1, mask=None, x0=None):
            b = shape[0]; latent_size = shape[2]
            img = input_all[0]['x']
            if img is None:
                img = torch.randn(shape, device=self.device)
                for item in input_all: item['x'] = img
            time_range = np.flip(self.ddim_timesteps); total_steps = self.ddim_timesteps.shape[0]
            if self.alpha_generator_func is not None:
                alphas = self.alpha_generator_func(len(time_range))
            mis_step = int(total_steps * self.mis)
            old_eps_dic = {}
            for input_idx, item in enumerate(input_all):
                old_eps_dic[str(input_idx)] = []
                for i, step in enumerate(time_range[:mis_step]):
                    if self.alpha_generator_func is not None:
                        self.set_alpha_scale(self.model, alphas[i])
                        if alphas[i] == 0: self.model.restore_first_conv_from_SD()
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=self.device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)
                    img, _, e_t = self.p_sample_plms(item, ts, index=index, uc=uc, guidance_scale=guidance_scale,
                                                     old_eps=old_eps_dic[str(input_idx)], t_next=ts_next)
                    item['x'] = img
                    old_eps_dic[str(input_idx)].append(e_t)
                    if len(old_eps_dic[str(input_idx)]) >= 4: old_eps_dic[str(input_idx)].pop(0)
            item = input_all[0]; old_eps = old_eps_dic['0']
            stack = torch.stack([x['x'] for x in input_all[:]])
            if fusion == 'official_mean':
                fused = torch.mean(stack, dim=0)
            else:
                g = input_all[0]['grounding_input']
                w = fusion_weights(g['boxes'].float().cpu(), g['masks'].float().cpu(), latent_size, fusion)
                w = w.to(device=stack.device, dtype=torch.float32)
                fused = ((stack.float()*w).sum(0)/w.sum(0)).to(stack.dtype)
            if diagnostics is not None:
                # Free diagnostics: how much the n+1 trajectories disagree on the
                # background (outside every box) at the merge step.
                g = input_all[0]['grounding_input']
                inside = fusion_weights(g['boxes'].float().cpu(), g['masks'].float().cpu(), latent_size,
                                        'region', feather=0)[1:].amax(0)[0, 0].to(stack.device) > 0
                outside = ~inside; s32 = stack.float()
                gap = [float((s32[k]-s32[0]).abs()[..., outside].mean()) if outside.any() else None
                       for k in range(1, len(s32))]
                diagnostics.update(fusion=fusion, merge_step=mis_step,
                                   trajectory_std=[float(x.std()) for x in s32], fused_std=float(fused.float().std()),
                                   background_fraction=float(outside.float().mean()),
                                   background_gap_to_global=gap)
            item['x'] = fused
            if on_merge is not None: on_merge()
            for i, step in enumerate(time_range):
                if i < mis_step: continue
                if self.alpha_generator_func is not None:
                    self.set_alpha_scale(self.model, alphas[i])
                    if alphas[i] == 0: self.model.restore_first_conv_from_SD()
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=self.device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)
                img, pred_x0, e_t = self.p_sample_plms(item, ts, index=index, uc=uc, guidance_scale=guidance_scale,
                                                       old_eps=old_eps, t_next=ts_next)
                item['x'] = img
                old_eps.append(e_t)
                if len(old_eps) >= 4: old_eps.pop(0)
            return img

    return FusionPLMSSamplerInst


@torch.no_grad()
def sample_latents(model, diffusion, batch, contexts, negative, noise, base_conv,
                   steps=50, alpha=.8, mis=.36, guidance=7.5, cache=True, deadline=None,
                   fusion='official_mean', release_silhouette_after_merge=False):
    """Official PLMS/MIS with fixed input RNG and reversible per-image state.

    This preserves the official late SD input-convolution restoration and
    restores learned weights before the next image. It does not rerank outputs.
    """
    from ldm.models.diffusion.plms import PLMSSampler
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
    drop_masks = getattr(model.position_net, 'test_drop_masks', None)
    def scale(_, value):
        for m in fusers: m.scale = value
    def schedule(length):
        return [1.] * int(alpha*length) + [0.] * (length-int(alpha*length))
    calls = dict(unet_calls=0, unet_sample_calls=0)
    def count(_, args):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Direct-generation time budget exhausted')
        calls['unet_calls'] += 1; calls['unet_sample_calls'] += args[0]['x'].shape[0]
    def finite(_, args, output):
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite UNet output; stopping before further diffusion steps')
    global_input = dict(x=noise.clone(), context=contexts[0], grounding_input=batch, timesteps=None)
    inputs = [global_input]
    if mis:
        for i in range(n):
            single = {k: torch.zeros_like(v) for k, v in batch.items()}
            for k in single: single[k][:, 0] = batch[k][:, i]
            inputs.append(dict(x=noise.clone(), context=contexts[i+1], grounding_input=single, timesteps=None))
    if fusion not in FUSIONS: raise ValueError('Unknown MIS fusion: ' + str(fusion))
    merge = {}
    if mis:
        # Upstream loop with a pluggable merge; official_mean is bit-identical to PLMSSamplerInst (tested).
        def release():
            # OCSD v3: silhouette masks shape the layout phase (the n+1 MIS trajectories); afterwards the joint
            # trajectory keeps boxes + phrases only, so texture and lighting are not held to a flat mask.
            if release_silhouette_after_merge: model.position_net.test_drop_masks = True
        sampler = mis_sampler_class(fusion, merge, release)(diffusion, model, alpha_generator_func=schedule,
                                                             set_alpha_scale=scale, mis=mis)
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=schedule, set_alpha_scale=scale)
    hook = model.register_forward_pre_hook(count)
    finite_hook = model.register_forward_hook(finite)
    try:
        model.grounding_tokenizer_input = null
        model.restore_first_conv_from_SD = restore
        with cached_conditioning(model.position_net, cache) as cache_stats:
            latent = sampler.sample(S=steps, shape=tuple(noise.shape), input=inputs if mis else global_input,
                                    uc=negative, guidance_scale=guidance)
        if calls['unet_sample_calls'] != expected:
            raise RuntimeError('Official sampler work differs from the planned budget')
        if not torch.isfinite(latent).all(): raise RuntimeError('Nonfinite direct-generation latent')
        extra = dict(mis_fusion=fusion if mis else 'none')
        if merge: extra['mis_merge'] = merge
        return latent, dict(**calls, conditioning_cache=cache_stats, base_conv_restorations=restorations, **extra)
    except Exception as exc:
        exc.ocsd_compute = dict(**calls, base_conv_restorations=restorations,
                                note='Forward attempts before failure; an OOM call may be incomplete')
        raise
    finally:
        hook.remove(); finite_hook.remove(); conv.load_state_dict(learned_conv, strict=True)
        model.restore_first_conv_from_SD = original_restore
        model.grounding_tokenizer_input = original_null
        for m, value in zip(fusers, scales): m.scale = value
        if drop_masks is not None: model.position_net.test_drop_masks = drop_masks


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


REFINER_REPO = 'stabilityai/stable-diffusion-xl-refiner-1.0'
REFINER_STRENGTH, REFINER_STEPS = .35, 20  # upstream inference.py defaults (--cascade_strength 0.35, 20 steps)


def load_refiner(cache, revision=None, device='cuda'):
    """Upstream InstanceDiffusion cascade: SDXL-Refiner img2img on the 512 px output (fp16 weights).

    Same model, dtype, variant and settings as upstream inference.py; we add a pinned revision and
    a seeded generator. Returns (pipeline, resolved revision)."""
    from huggingface_hub import HfApi
    from diffusers import StableDiffusionXLImg2ImgPipeline
    revision = revision or HfApi(token=False).model_info(REFINER_REPO).sha
    if not re.fullmatch(r'[a-f0-9]{40}', revision): raise ValueError('Expected an exact refiner revision')
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_REPO, revision=revision, torch_dtype=torch.float16, variant='fp16', use_safetensors=True,
        cache_dir=str(Path(cache)/'hf'), token=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device), revision


@torch.no_grad()
def refine(pipe, pixels, caption, seed, strength=REFINER_STRENGTH, steps=REFINER_STEPS, device='cuda'):
    """Prompt = global caption, no negative prompt (as upstream). Returns uint8 pixels and a log."""
    from PIL import Image
    start = time.perf_counter()
    generator = torch.Generator('cpu').manual_seed(int(seed))
    out = pipe(caption, image=Image.fromarray(pixels), strength=strength, num_inference_steps=steps,
               generator=generator, output_type='np').images[0]
    if not np.isfinite(out).all(): raise RuntimeError('Nonfinite refiner image')
    refined = (np.clip(out, 0, 1)*255).round().astype(np.uint8)
    if refined.shape != pixels.shape: raise RuntimeError('Refiner changed the image size')
    return refined, dict(refiner=REFINER_REPO, refiner_strength=strength, refiner_steps=steps,
                         refiner_denoise_steps=int(steps*strength), refiner_wall_s=time.perf_counter()-start)


def load_components(cache, lock=None, expected_checkpoint_sha256=None):
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
    checkpoint_hash = file_hash(weights)
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError('Checkpoint differs from the accepted CPU preflight')
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
                      checkpoint_sha256=checkpoint_hash, checkpoint_state=weight_key,
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
    def generate(self, condition, seed, negative, scribble=True, device='cuda', steps=50, mis=.36, alpha=.8, deadline=None,
                 fusion='official_mean', silhouette=False, silhouette_layout_only=False):
        start = time.perf_counter()
        features, contexts, uncond = self.encode(condition, negative)
        batch = conditioning_batch(condition, features, device=device)
        if silhouette and not all('silhouette_rle' in o for o in condition['objects']):
            raise ValueError('Silhouette arm needs parsed sketch silhouettes for every instance')
        if not silhouette: batch['segs'].zero_(); batch['polygons'].zero_()
        self.model.to(device); self.vae.to(device); self.diffusion.to(device)
        self.model.position_net.test_drop_scribbles = not scribble
        # Masks are only ever silhouettes parsed from the input sketch, never benchmark/oracle masks
        # (upstream configs/test_mask.yaml sets the same flag).
        self.model.position_net.test_drop_masks = not silhouette
        contexts = [c.to(device) for c in contexts]; uncond = uncond.to(device)
        noise = torch.randn((1,4,64,64), generator=torch.Generator('cpu').manual_seed(seed)).to(device)
        autocast = torch.autocast(device_type='cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with autocast:
            latent, log = sample_latents(self.model, self.diffusion, batch, contexts, uncond, noise, self.base_conv,
                                        steps=steps, mis=mis, alpha=alpha, deadline=deadline, fusion=fusion,
                                        release_silhouette_after_merge=silhouette and silhouette_layout_only)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Direct-generation time budget exhausted before VAE decode')
        # Fixed for both arms: decode in FP32 to avoid VAE FP16 overflow.
        image = self.vae.decode(latent.float())
        if not torch.isfinite(image).all(): raise RuntimeError('Nonfinite direct-generation image')
        pixels = ((image[0].clamp(-1,1)*.5+.5)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        log.update(provenance=self.provenance, steps=steps, mis=mis, alpha=alpha, seed=seed,
                   conditioning=('boxes+centers' + ('+20_ink_points' if scribble else '')
                                 + ('+sketch_silhouette_mask+256pt_polygon' if silhouette else '')
                                 + ('(layout_steps_only)' if silhouette and silhouette_layout_only else '')),
                   unet_precision='autocast_float16' if str(device).startswith('cuda') else 'float32', vae_precision='float32',
                   wall_s=time.perf_counter()-start, image_size=512, refiner=False,
                   external_method='InstanceDiffusion pinned official core with OCSD input adapter')
        return pixels, log
