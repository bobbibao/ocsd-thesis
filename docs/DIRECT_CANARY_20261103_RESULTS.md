# Direct canary results (run `direct_9b63ee5bdadc694320c9`, 2026-11-03)

Development canary for the external InstanceDiffusion baseline (Wang et al., CVPR 2024,
https://arxiv.org/abs/2402.03290; official core pinned at `722900ef03e3`). One scene, two seeds,
two arms, four images. **This is not a quality PASS, not a method comparison and not a paper result.**
ControlNet and OCSD soft-matte were not in this run.

## 1. Technical audit (done before looking at images)

| Check | Result |
|---|---|
| ZIP checksums | 25/25 files match `checksums.json` |
| Status | `complete`, 4/4 jobs (`direct_box`, `direct_scribble` × seeds 0, 1), one attempt, no failed or interrupted work |
| Contract | protocol, source hashes and model provenance equal the locked CPU-PASS identities: source `722900ef03e3`, HF InstanceDiffusion `fc74d76c`, CLIP `32bd6428`, checkpoint SHA256 `b4de3f78…de1d1`, EMA state, complete phrase-text weights |
| Code | OCSD commit recorded in the run: `fb4311d0…` (historical SHA before the author rewrite, kept as evidence) |
| Environment | NVIDIA L4 (22.0 GiB), CUDA 13.0, Python 3.13.16, torch 2.11.0+cu130, transformers 5.17.0 |
| Work | 216 UNet sample calls per image, 864 total = plan; conditioner cache 5 evaluations / 216 calls per image |
| Peak CUDA memory | 5.95 GiB (6 393 032 704 bytes) for every image |
| Wall time | 18.7 to 22.4 s per image; 199.3 s for the whole attempt including checkpoint load (excludes pip install and idle time) |
| Numerics | all outputs finite; UNet autocast FP16, VAE FP32 |

Notes:

* The 20 GiB VRAM gate is conservative: peak use was 5.95 GiB. The binding resource is host RAM
  for loading the 13.6 GB checkpoint on CPU, which is why the gate was not relaxed to T4 here.
* `ComplexHalf support is experimental` comes from the official `Fourier_filter` (FreeU-style) in
  `openaimodel.py`, which runs `torch.fft` on FP16 latents under autocast. Upstream's own
  `inference.py` also runs under FP16 autocast, so this is upstream behaviour, not an adapter bug.
* The CLIP "UNEXPECTED keys" report lists only vision-tower and projection weights, which the text
  encoder does not use. This is expected.

**Bug found:** the Vietnamese text in `SO_SANH_seed*.png` rendered as boxes. Colab has no
`DejaVuSans.ttf` on Pillow's search path, so the code fell back to Pillow's built-in font, which has
no Vietnamese glyphs (reproduced locally). Fixed in `ocsd/direct_canary.py`: the review font is now
chosen from several system fonts after a glyph check, with an accent-free fallback instead of boxes.
The two sheets were rebuilt from the original PNGs; the image panels are pixel-identical to the
originals (checked). The images themselves were not affected.

## 2. Analyst observations (not the user's review, not scores)

Read with the overlay of the input boxes and the 20 ink points per instance. Labels below name the
arms, so a blind reviewer should look at the sheets first.

| Image | Count 2 bicycles + 1 bench | Colour per bicycle | Left/right, bench above | Visible problems |
|---|---|---|---|---|
| box, seed 0 | yes | yes | yes | bench mounted on a wall, not standing in a park; yellow parts on the bench |
| scribble, seed 0 | yes | yes | yes | blue bicycle sits on a blue/green rectangle that follows its box; bench frame yellow |
| box, seed 1 | yes | yes | yes | blue bicycle melted/lying, shape broken; otherwise the most coherent scene (wooden bench, street with trees) |
| scribble, seed 1 | yes | yes | yes | image splits into two panels with a hard horizontal seam just under the bench box; bench yellow |

What this does and does not show:

* Count, colour binding and box layout were right in 4/4. This is the kind of case (two same-class
  bicycles with different colours) where earlier OCSD routes, on a different scene, needed
  count-aware repair and still produced a distorted yellow bicycle. One scene and two seeds cannot
  establish a rate, and the scenes differ, so this is not a comparison.
* Three failure modes, each seen in at least two images:
  1. **Attribute leakage to a third object**: "yellow" leaks to the "wooden bench" in 3/4.
  2. **Box-shaped scene incoherence**: region panels/seams aligned with instance boxes in 2/4 (both
     scribble images) and a floating wall-mounted bench in a third.
  3. **Shape failure on the flat blue-bicycle box** (aspect about 2.2:1) in 1/4.
* No evidence yet that scribble beats box or the reverse; two seeds per arm is far too few.

## 3. Mechanism behind failure mode 2 (from the pinned source, not from new runs)

Official `PLMSSamplerInst.plms_sampling` runs the full-caption trajectory **and one trajectory per
instance**, each conditioned only on that instance's phrase as its caption, for the first
`int(50 × 0.36) = 18` steps. It then replaces the latent with the **plain mean of all n+1 latents**
(`torch.mean(torch.stack(...))`; the crop-and-paste alternative is disabled upstream). Consequences:

* Outside the boxes, the background is the average of n+1 independently composed backgrounds, at
  the steps where global layout is decided. That is a plausible cause of panels and seams.
* Inside its own box an instance trajectory gets weight 1/(n+1), i.e. 1/4 here, while the
  full-caption trajectory (where "yellow" and "bench" share one caption) and two unrelated
  trajectories make up the rest. The upstream README says a higher `mis` reduces leakage; the merge
  weight may be part of why leakage survives.

These are hypotheses. They are consistent with the four images but not tested by them.

## 4. Change made: region-aware MIS merge (our variant, off by default)

`ocsd/direct_instances.py` now runs the upstream loop with only the merge step pluggable:

* `official_mean` (default): bit-identical to upstream `PLMSSamplerInst` (tested against the pinned
  source on the tiny official core).
* `region`: the full-caption trajectory keeps weight 1 everywhere; instance *i* adds weight 1 only
  inside its own box, with a fixed 2-latent-pixel soft edge. Background = full-caption trajectory;
  inside one box = (global + instance)/2; overlaps average all covering trajectories. Same UNet work.

Every image now also logs free merge diagnostics: per-trajectory latent std, fused std, background
fraction and the mean background gap between each instance trajectory and the global one.

Status: **implemented and CPU-tested only; no GPU output exists**. The feather width was fixed before
any output and will not be tuned on the canary scene. It is credited as an OCSD modification of the
InstanceDiffusion sampler, not as part of InstanceDiffusion.

## 5. Next step

See [DIRECT_DEV_PROTOCOL.md](DIRECT_DEV_PROTOCOL.md). Nothing new should be run on Colab
until that protocol is locked; re-running `direct_canary` now would regenerate the same four images
under a new contract (source hashes changed) and waste compute.

## Files

* [run.json](direct_canary_20261103/run.json) and [checksums.json](direct_canary_20261103/checksums.json): copied unchanged from the ZIP.
* [Overlay of boxes and ink points](direct_canary_20261103/overlay_boxes_points.jpg) (analyst view, names the arms).
* Rebuilt review sheets with readable text: [seed 0](direct_canary_20261103/SO_SANH_seed0.png), [seed 1](direct_canary_20261103/SO_SANH_seed1.png) (blind A/B; key unchanged in the run's `review_key.json`).
