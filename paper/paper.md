---
title: "Object-Consistent Sketch-and-Text Guided Scene Image Generation with Diffusion Models"
---

::: {custom-style="Author"}
Le Hoang Bao, Thai Ba Cuong, Nguyen Thanh Chuyen
:::

::: {custom-style="Affiliation"}
Faculty of Information Technology, Industrial University of Ho Chi Minh City, Vietnam
:::

::: {custom-style="Note"}
**Manuscript status (29 September 2026).** All baseline numbers are final. Rows marked † (OCSD, OCSD-lite, the ablation study and the α study) come from the first paper-tier run, before energy guidance M5(b) was re-tuned on the tuning split; they are preliminary and will be replaced by the re-run. The analysis below is written to be honest about what these preliminary numbers do and do not show.
:::

::: {custom-style="AbstractTitle"}
Abstract
:::

::: {custom-style="Abstract"}
Sketch-and-text conditioned diffusion models such as ControlNet produce realistic images from a single object sketch, but they lose objects, miscount them and misplace them as soon as a freehand scene sketch contains several objects. We study this *object-consistency* problem and make three contributions. First, we propose OCSD (Object-Consistent Sketch-guided Diffusion), a framework that decomposes the sketch into objects, generates and selects each object independently, learns a per-object identity token with a masked diffusion loss, composes the scene with blended latent inference, and adds an object-aware conditioning module that restricts cross-attention to each object's sketch region, steers attention with an energy function, re-injects the scene sketch through a low-weight ControlNet and verifies the result with an open-set detector. Second, we build an evaluation protocol that isolates object consistency: QuickDraw-Scenes, a controlled benchmark of real freehand object sketches arranged into 1 to 10-object scenes at three abstraction levels, and COCO-Sketch, a real-image benchmark; object preservation, class-wise count error, layout IoU and relation accuracy are scored with a detector (OWLv2) that the method never sees, alongside a detector ceiling, a best-of-3 control for regeneration, bootstrap confidence intervals and Holm-corrected Wilcoxon tests. Third, we report a controlled comparison of eight methods on a shared Stable Diffusion 1.5 backbone. OCSD improves significantly over a faithful re-implementation of the two-branch method of Zhang et al. on every consistency metric (object preservation 62.5% vs. 44.3%, layout mIoU 0.459 vs. 0.251, relation accuracy 34.5% vs. 10.1%; Holm-adjusted p < 0.05), and its training-free variant has the lowest class-wise count error of all methods. However, the box-conditioned GLIGEN remains the strongest method on most metrics, and our preliminary ablation shows why: energy guidance on identity tokens costs 18.5 points of object preservation, and removing it raises OCSD to 78.1% on the ablation subset. We release code, benchmarks and all per-scene results.
:::

**Keywords:** diffusion models, sketch-to-image, scene generation, object consistency, controllable generation, evaluation protocol.

# 1. Introduction

Text-to-image diffusion models [@rombach2022ldm; @saharia2022imagen; @ramesh2022dalle2] have made photorealistic synthesis accessible, and spatial adapters such as ControlNet [@zhang2023controlnet] and T2I-Adapter [@mou2024t2i] let a user add a sketch to control shape and layout. For a single object this works well. For a *scene* sketch, which a non-expert draws as several rough objects on one canvas, it frequently does not. The control network treats the whole sketch as one geometric map: it knows *where* there are strokes, but not *which strokes belong to which object*, and binding each region to a phrase of the prompt is left to cross-attention. The result is a familiar set of failures: objects disappear (especially small or abstract ones), duplicates merge or multiply, attributes leak between objects, and objects drift away from where they were drawn. Recent analysis shows that scene complexity, not data imbalance, is the main driver of these multi-object failures, and that counting and spatial relations are the least robust abilities [@jeong2026multiobj].

We call the requirement that the generated image contain *the right objects, in the right number, at the right place and shape, in the right relations* **object consistency**, and we make it the explicit target of both the method and the evaluation. The closest prior work, Zhang et al. [@zhang2025sketchscene], splits generation into an object-level stage (generate each object from its own sketch, learn an identity embedding with a masked loss) and a scene-level stage (blend object and background latents early, then denoise freely with identity tokens). This preserves object appearance, but once blending ends nothing keeps an object in its region, and the prompt describes only the background. The authors themselves note that objects are still lost as the object count grows.

This paper makes three contributions:

1. **OCSD, an object-aware extension of the two-branch approach.** Object phrases from the user's prompt drive object generation, the best of *K* candidates is selected by semantic, shape and detection scores, identity tokens are trained with an additional attention-separation term, and a new object-aware conditioning module (M5) acts during the free-denoising phase through four mechanisms: region-restricted cross-attention, attention-energy guidance, a low-weight scene ControlNet and post-hoc verification with regeneration (Section 3).
2. **An evaluation protocol for object consistency.** Two automatically built benchmarks, QuickDraw-Scenes (controlled object count and sketch abstraction, real freehand strokes) and COCO-Sketch (real photographs), with metrics scored by an evaluation detector that differs from the one the method uses internally, a detector-ceiling row, a best-of-3 control that gives baselines the same regeneration budget, a held-out tuning split, and paired statistical tests (Section 4).
3. **A controlled, honest comparison.** Eight methods run on the same backbone, sampler, seeds and scenes. We report where OCSD helps, where it does not, and which component is responsible (Section 5).

# 2. Related Work

**Sketch-to-image generation.** Early work used conditional GANs: pix2pix [@isola2017pix2pix] for pixel-aligned edges and SketchyGAN [@chen2018sketchygan] on the Sketchy database [@sangkloy2016sketchy]. With diffusion models, Voynov et al. [@voynov2023sketch] guide sampling with a latent edge predictor, and Koley et al. [@koley2023picture; @koley2024sketch] argue that abstract sketches should be interpreted semantically rather than traced pixel by pixel. ControlNet [@zhang2023controlnet], T2I-Adapter [@mou2024t2i] and Uni-ControlNet [@zhao2023unicontrolnet] add spatial conditions to a frozen text-to-image model and are the de facto baselines for sketch control.

**Scene-level sketches.** Sketch2Photo [@chen2009sketch2photo] composed retrieved photographs; SketchyCOCO [@gao2020sketchycoco] separated foreground and background with EdgeGAN but covers only 14 foreground classes; FS-COCO [@chowdhury2022fscoco] and SketchyScene [@zou2018sketchyscene] provide freehand scene sketches. Unsupervised and text-guided scene sketch-to-photo methods [@wang2022unsupscene; @maungmaung2023textscene] normalise sketches and photos to an edge domain, Cheng et al. [@cheng2024multiobj] address multi-object interference with per-object control, FineControlNet [@choi2023finecontrolnet] injects region-wise text alongside spatial control, and SketchingReality [@bourouis2026sketchingreality] trains a sketch-semantics modulation network with attention supervision. Zhang et al. [@zhang2025sketchscene] is our starting point and is described in Section 1.

**Layout and multi-object control.** For text prompts, Attend-and-Excite [@chefer2023attend] maximises the attention of neglected tokens, Structured Diffusion [@feng2023structured] and SynGen [@rassin2023linguistic] bind attributes through syntax, and Composable Diffusion [@liu2022composable] adds score functions. With boxes, GLIGEN [@li2023gligen] trains gated self-attention layers for grounding, while BoxDiff [@xie2023boxdiff], Layout Guidance [@chen2024trainingfreelayout] and Attention Refocusing [@phung2024refocus] steer attention maps with energy functions at inference; DenseDiffusion [@kim2023dense] modulates attention scores with region masks and MultiDiffusion [@bartal2023multidiffusion] fuses region-wise diffusion paths. Object-level semantic alignment has also been shown to improve multi-object fidelity [@liu2026objalign]. These methods use boxes or masks, not the shape information of a freehand sketch, and have no notion of object identity.

**Identity and consistency.** Textual Inversion [@gal2023textual], DreamBooth [@ruiz2023dreambooth], LoRA [@hu2022lora] and IP-Adapter [@ye2023ipadapter] personalise a model to a concept; Break-A-Scene [@avrahami2023breakascene] extracts several concepts from one image with a masked loss and a cross-attention loss; MasaCtrl, ConsiStory and StoryDiffusion [@cao2023masactrl; @tewel2024consistory; @zhou2024storydiffusion] share self-attention to keep subjects consistent across images. OCSD borrows the masked loss and attention supervision for per-scene identity tokens, and combines them with sketch-derived regions.

# 3. Method

## 3.1 Problem formulation

The input is a scene sketch $\mathcal{S}$ of size $H\times W$ ($512\times512$) containing $N$ foreground objects and a prompt $\mathcal{P}$ that contains object phrases $\{p_1,\ldots,p_N\}$ (class and attributes, e.g. "two white sheep") and a background phrase $p_{bg}$. Each object carries *object information* $o_i=(c_i,\mathbf{m}_i,\mathbf{b}_i,a_i)$: class, region mask, bounding box and attributes, and the objects are linked by spatial relations $\mathcal{R}=\{(i,j,r)\}$ with $r\in\{$left, right, above, below, in front, behind$\}$. An output image $\mathbf{I}$ is *object-consistent* if (R1) every object appears with the right class and attributes, (R2) the per-class count matches the sketch, (R3) each object occupies its sketched region and follows its shape, and (R4) every relation in $\mathcal{R}$ holds. We ask: (Q1) how fast does a sketch-conditioned baseline degrade as object count and sketch abstraction grow; (Q2) does splitting generation into object and scene levels with identity tokens help; and (Q3) does object-aware conditioning reduce the remaining loss, count and relation errors.

## 3.2 Overview

Figure 1 shows the pipeline. M1 parses the sketch and prompt into objects. The object-level branch generates each object (M2) and learns its identity (M3); the scene-level branch composes the scene with blended and customised inference (M4) under object-aware conditioning (M5). All modules share Stable Diffusion 1.5 [@rombach2022ldm] with ControlNet Scribble v1.1, so every baseline in Section 5 can run on the same backbone.

![**Figure 1.** Overview of OCSD. The object-level branch (M2, M3) generates each object from its own sketch and learns an identity token; the scene-level branch (M4, M5) composes the scene with blended inference and then denoises under object-aware conditioning.](figs/fig1_arch.png){width=6.5in}

## 3.3 Object decomposition (M1)

The sketch is binarised, padded to $512\times512$, thinned and re-thickened to about 3 px to match ControlNet Scribble's training distribution. Strokes are grouped into objects from drawing layers when the user draws one object per layer (the interactive mode), from benchmark annotations, or, as a fallback, by connected components after dilation. The mask $\mathbf{m}_i$ is the convex hull of the object's strokes after morphological closing, dilated by 9 px, and $\mathbf{b}_i$ is its bounding box. Relations follow geometric rules on box centres with a margin $\delta=0.05W$; "in front of" is decided by the lower box edge under a ground-plane assumption. Object phrases are matched to sketched objects by class and left-to-right order, and the global prompt is rebuilt as $\mathcal{P}_g=$ "a photo of $\langle o_1\rangle$ $c_1$ and … $\langle o_N\rangle$ $c_N$, $p_{bg}$".

## 3.4 Object-level generation with candidate selection (M2)

Each object sketch is cropped to its box, centred and resized to $512\times512$, and $K$ candidates are generated with ControlNet Scribble and the prompt "a photo of a $p_i$, simple white background". Using the user's full phrase, not only the class name as in [@zhang2025sketchscene], attaches attributes to the right object from the start. Grounded-SAM [@ren2024groundedsam; @liu2024groundingdino; @kirillov2023sam] segments each candidate, and the candidate with the highest normalised score

$$k^*=\arg\max_k\Big[\lambda_1\,\mathrm{CLIP}\big(\mathbf{x}_i^{(k)}\odot\hat{\mathbf{m}}_i^{(k)},p_i\big)+\lambda_2\,\mathrm{IoU}\big(\hat{\mathbf{m}}_i^{(k)},\mathbf{m}_i^{crop}\big)+\lambda_3\,\mathrm{conf}_{det}\Big]$$

is kept ($\lambda_1=\lambda_2=\lambda_3=1$), rewarding correct semantics, faithful shape and a confident detection. M2 and M3 run once per scene with a fixed preparation seed and are reused across generation seeds and ablation variants, so that differences between variants come only from the changed component.

## 3.5 Identity learning (M3)

For each object a new token $\langle o_i\rangle$ is added to the text encoder, initialised from the embedding of $c_i$, and trained on its object image with the masked diffusion loss

$$\mathcal{L}_{id}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}_{\boldsymbol{\epsilon},t}\Big[\big\|\big(\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\theta,\Delta}(\mathbf{z}_{i,t},t,\text{“a photo of }\langle o_i\rangle\,c_i\text{”})\big)\odot\tilde{\mathbf{m}}_i\big\|_2^2\Big].$$

Training has two stages: embeddings only (learning rate $5\times10^{-3}$), then embeddings ($5\times10^{-5}$) jointly with rank-16 LoRA [@hu2022lora] on the attention projections ($10^{-4}$). To keep tokens of same-class objects apart, we add an attention-separation term computed on random composites of two or three objects (up to six composites per scene, sampled with probability 0.35):

$$\mathcal{L}_{M3}=\mathcal{L}_{id}+\lambda_{att}\,\frac{1}{N}\sum_{i}\big\|\bar{A}_i-\tilde{\mathbf{m}}_i\big\|_2^2,\qquad \lambda_{att}=0.01,$$

where $\bar{A}_i$ is the normalised $16\times16$ cross-attention map of $\langle o_i\rangle$, in the spirit of Break-A-Scene [@avrahami2023breakascene].

## 3.6 Scene composition (M4)

Object images are cut out with their masks, scaled and shifted to maximise IoU with the sketch masks, and pasted far-to-near, giving a foreground image with latent $\mathbf{z}_{fg,0}$ and a union mask $\tilde{\mathbf{m}}$. A background latent $\mathbf{z}_{bg,t}$ is denoised from noise with the prompt "a photo of $p_{bg}$"; its negative prompt lists the foreground classes so that the background does not spawn extra copies of them (applied to both OCSD and our Zhang et al. re-implementation). Denoising then has two phases:

$$\mathbf{z}_t\leftarrow\mathbf{z}_{bg,t}\odot(1-\tilde{\mathbf{m}})+\mathbf{z}_{fg,t}\odot\tilde{\mathbf{m}}\quad (t>\alpha T),\qquad \mathbf{z}_{t-1}=\mathrm{Denoise}_{\theta,\,s\Delta}(\mathbf{z}_t,\mathcal{P}_g,t)\quad(t\le\alpha T).$$

Blending locks the layout while it is formed; customised inference then harmonises lighting and boundaries while the identity tokens keep each object's appearance. At inference the LoRA update is scaled by a strength $s\in(0,1]$. Because $\mathbf{z}_t$ is overwritten at every blending step, only the background branch needs to be denoised during blending; the scene U-Net runs from the last blending step on, which gives identical results with about half the U-Net calls.

## 3.7 Object-aware conditioning (M5)

M5 replaces the plain denoising step of the customised phase with four mechanisms, each aimed at one consistency requirement.

**(a) Region-restricted cross-attention (R1, R3).** Let $\mathcal{T}_i$ index the prompt tokens of object $i$ (its identity token, attributes and class) and $\mathcal{T}_{bg}$ those of the background phrase. A bias is added before the softmax,

$$A=\mathrm{softmax}\Big(\tfrac{QK^\top}{\sqrt d}+B\Big),\quad B_{n,j}=\begin{cases}-\lambda_t & j\in\mathcal{T}_i,\ n\notin\tilde{\mathbf{m}}_i\\ -\lambda_t & j\in\mathcal{T}_{bg},\ n\in\tilde{\mathbf{m}}\\ 0&\text{otherwise,}\end{cases}\qquad \lambda_t=\lambda_0\big(t/\alpha T\big)^{\gamma},$$

so each object's tokens act only inside its own sketch region and the background tokens only outside the foreground; the restriction relaxes towards the end so boundaries can blend. This resembles DenseDiffusion [@kim2023dense], but uses sketch-shaped masks and identity tokens.

**(b) Attention-energy guidance (R1, R2).** During the first $\tau$ customised steps, the latent is updated by $\mathbf{z}_t\leftarrow\mathbf{z}_t-\eta_t\nabla_{\mathbf{z}_t}E$ with

$$E(\mathbf{z}_t)=\sum_{i=1}^{N}\Big[\big(1-\max_{n\in\tilde{\mathbf{m}}_i}\bar{A}_i[n]\big)+\beta\,\frac{\sum_{n\notin\tilde{\mathbf{m}}_i}\bar{A}_i[n]}{\sum_n\bar{A}_i[n]}\Big].$$

The first term penalises an object that no location attends to strongly (a missing object), as in Attend-and-Excite [@chefer2023attend]; the second penalises attention outside the region (drift or duplication), as in BoxDiff and Layout Guidance [@xie2023boxdiff; @chen2024trainingfreelayout]. Attention maps follow Attend-and-Excite (drop the start token, scale by 100, softmax over text tokens, $3\times3$ Gaussian smoothing), and $\eta_t$ decays linearly from $\eta$ to $\eta/2$. In the preliminary run the energy acts on the identity tokens $\langle o_i\rangle$; Section 5.4 shows that this is harmful and motivates steering the object-phrase tokens instead.

**(c) Low-weight scene ControlNet (R3, R4).** The full scene sketch is fed to ControlNet Scribble with conditioning scale $\omega=0.4$ during customised inference, reminding the model of contours and relative placement after hard blending stops, without imposing strokes on the background.

**(d) Post-hoc verification (R1, R2, R4).** Grounding DINO (threshold 0.35) detects every sketched class; detections are matched to sketch boxes by same-class Hungarian matching on IoU. An image passes when every object is matched, every class count is correct and every relation holds. Otherwise OCSD regenerates with a new seed and $\lambda_0\times1.5$, at most $R=2$ times, and returns the image with the highest $\mathrm{OPR}+0.5\,\mathrm{mIoU}+0.25\,\mathrm{RA}-0.25\,\mathrm{OCE}_c/N$. The verifier deliberately differs from the evaluation detector (Section 4.3).

Table 1 lists the hyperparameters. The default column is the full configuration in the code; the experiment column is the reduced setting that fits a Colab Pro budget, with $\alpha$ and $s$ chosen on a held-out tuning split (Section 4.4).

::: {custom-style="TableCaption"}
**Table 1.** OCSD hyperparameters: code default and the setting used in all experiments.
:::

| Symbol | Meaning | Default | Experiments |
|:--|:--|:--:|:--:|
| $T$, $w$ | DDIM steps, CFG scale | 50, 7.5 | 30, 7.5 |
| $K$ | candidates per object (M2), denoising steps | 4, 30 | 2, 20 |
| $S_1$, $S_2$ | embedding / joint steps (M3), LoRA rank | 200 / 200, 16 | 100 / 100, 16 |
| $\alpha$, $s$ | blending switch point, LoRA strength (M4) | 0.5, 1.0 | 0.1, 0.5 (tuned) |
| $\lambda_0$, $\gamma$ | region-attention strength and decay (M5a) | 8, 1 | 8, 1 |
| $\beta$, $\eta$, $\tau$ | energy guidance (M5b) | 1, 20, 10 steps | 1, 20, 10 steps |
| $\omega$ | scene ControlNet scale (M5c) | 0.4 | 0.4 |
| $R$ | maximum regenerations (M5d) | 2 | 2 |

# 4. Evaluation Protocol

## 4.1 Benchmarks

Public scene-sketch datasets [@gao2020sketchycoco; @chowdhury2022fscoco] do not let us vary object count and sketch abstraction independently, and do not provide per-object sketches with phrases. We therefore build two complementary benchmarks, both generated automatically by the released code.

**QuickDraw-Scenes.** Real freehand object sketches from Quick, Draw! [@jongejan2016quickdraw] in 28 classes shared with COCO are arranged on a $512\times512$ canvas with a perspective layout in 12 settings (e.g. "in a green meadow", "on a city street"), so class, mask, box and relations of every object are known exactly. Scenes form a $4\times3$ grid: 1, 3, 5 or 8 to 10 objects, times three abstraction levels. Sketch detail is measured as points plus ten times strokes and split into per-class terciles: *simple* uses the most detailed tercile, recognised sketches and no overlap; *medium* the middle tercile with box IoU up to 0.15; *complex* the most abstract tercile, IoU up to 0.30, Gaussian stroke jitter ($\sigma=2.5$ px) and thinner strokes. To test counting, 40% of objects duplicate an existing class and phrase ("three white sheep"), and the caption states the counts so that text-only methods also know them.

**COCO-Sketch.** COCO val2017 [@lin2014coco] images with 1 to 8 salient objects (each at least 1.5% of the image and not cut by the border), balanced over 1, 2–3, 4–5 and 6–8 objects, cropped to $512\times512$. Sketches are PiDiNet [@su2021pidinet] scribbles, split per object by the ground-truth masks; captions are the human COCO captions. Real images allow FID/KID and a detector ceiling.

## 4.2 Protocol

The paper tier uses 72 QuickDraw-Scenes scenes (6 per cell). Methods that need per-scene identity learning (OCSD, Zhang et al.) run on a stratified 36-scene subset (3 per cell) with two seeds; training-free methods run on all 72 with seed 0 and additionally with seed 1 on the 36-scene subset, so the main comparison has every method on the same scenes and seeds. On COCO-Sketch, training-free methods run on 32 scenes and all methods on a common 16-scene subset (one seed). The ablation uses 18 identity-learning scenes with 3 and 5 objects, and the $\alpha$ study 12 of them (one seed). Every method uses SD 1.5, DDIM with 30 steps, CFG 7.5, the same negative prompt and the same seeds. Experiments ran on one NVIDIA A100 40 GB in Google Colab.

## 4.3 Metrics

Detection-based metrics use **OWLv2** [@minderer2023owlv2] (threshold 0.30), queried with the sketched class names; detections are matched to sketch boxes by same-class Hungarian matching with IoU at least 0.1. OWLv2 is deliberately different from Grounding DINO, which OCSD uses inside M2 and M5(d), so the method cannot be rewarded for optimising against its own judge.

- **Object preservation rate** $\mathrm{OPR}=N_{preserved}/N_{input}$ (R1).
- **Class-wise object count error** $\mathrm{OCE}_c=\sum_c\left|N_{gen,c}-N_{input,c}\right|$, which, unlike the total count error, cannot cancel a missing object of one class against an extra object of another; and **count accuracy**, the share of images with $\mathrm{OCE}_c=0$ (R2).
- **Layout mIoU** between matched detections and sketch boxes, unmatched objects counting as zero (R3), and **relation accuracy (RA)**, the share of sketch relations preserved (R4).
- **CLIP score** [@hessel2021clipscore] with ViT-L/14 [@radford2021clip] for the whole image, and **object CLIP**, computed on crops at the *sketch* boxes against "a photo of a $p_i$", which checks attribute binding independently of the detector.
- **ID-Sim**, the DINOv2 [@oquab2024dinov2] cosine similarity between each object region in the scene and its M2 object image (two-branch methods only), and **FID/KID** [@heusel2017fid; @binkowski2018kid] against 2,000 COCO val2017 crops.

**Detector ceiling and fair controls.** On COCO-Sketch we score the real photographs themselves: they certainly contain all objects, so their OPR bounds what OWLv2 lets any method reach. Because OCSD may generate up to three times, we add **ControlNet best-of-3**, which regenerates the baseline with the *same* verifier and selection score, so any gain of M5(d) over it is not just extra sampling.

**Statistics.** Metrics are averaged over seeds per scene and then over scenes, reported as mean ± half-width of a 95% bootstrap interval (2,000 scene resamples) [@efron1993bootstrap]. OCSD is compared with each method by a paired Wilcoxon signed-rank test on per-scene values [@wilcoxon1945] with Holm correction [@holm1979].

## 4.4 Held-out tuning

A pilot run with the default $\alpha=0.5$, $s=1.0$ showed that OPR rose as $\alpha$ fell and that full OCSD trailed OCSD-lite, suggesting that full-strength identity LoRA pulls the scene away from the sketch layout. Before the paper run we therefore tuned $\alpha\in\{0,0.1,0.2,0.3,0.5\}\times s\in\{0.5,1.0\}$ with two seeds on the 16 pilot scenes (12 QuickDraw-Scenes, 4 COCO-Sketch), which are **excluded from every evaluation set**. The rule selects the maximum OPR + mIoU + RA subject to global CLIP within 1 point of the default. It chose $\alpha=0.1$, $s=0.5$ (score 1.562 vs. 1.294 for the default; OPR 66.4% vs. 57.0%, mIoU 0.484 vs. 0.371, CLIP 26.36 vs. 26.37). LoRA strength 0.5 beat 1.0 for four of five $\alpha$ values. Our Zhang et al. re-implementation keeps its published setting ($\alpha=0.5$, $s=1.0$).

# 5. Results

## 5.1 Compared methods

(1) **SD + ControlNet** Scribble on the whole sketch (scale 1.0); (2) **T2I-Adapter** Sketch [@mou2024t2i]; (3) **GLIGEN** [@li2023gligen] with per-object boxes and phrases (no sketch shape); (4) **ControlNet + region attention**, DenseDiffusion-style [@kim2023dense] masking with the same masks and phrases as OCSD, replacing FineControlNet [@choi2023finecontrolnet], which has no public SD 1.5 code; (5) **ControlNet + attention energy**, BoxDiff/Attend-and-Excite-style [@xie2023boxdiff; @chefer2023attend]; (6) **ControlNet best-of-3**; (7) **Zhang et al.** [@zhang2025sketchscene], re-implemented in the same code base: one candidate with a class-name prompt, no attention separation, $\alpha=0.5$, no M5; (8) **OCSD-lite**, OCSD without identity learning (training-free); (9) **OCSD**.

## 5.2 Main comparison

::: {custom-style="TableCaption"}
**Table 2.** QuickDraw-Scenes, 36 identity-learning scenes × 2 seeds (mean ± 95% CI half-width). † preliminary. Bold: best.
:::

| Method | OPR (%) ↑ | OCE~c~ ↓ | Count acc. (%) ↑ | mIoU ↑ | RA (%) ↑ | Obj. CLIP ↑ | ID-Sim ↑ | FID ↓ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SD + ControlNet | 67.4 ± 8.9 | 2.46 ± 0.81 | 25.0 ± 13.2 | 0.510 ± 0.080 | 36.6 ± 10.4 | 23.52 | – | 225.0 |
| T2I-Adapter | 60.5 ± 9.7 | 2.89 ± 0.92 | 22.2 ± 11.8 | 0.470 ± 0.080 | 28.3 ± 10.6 | 22.70 | – | 236.8 |
| GLIGEN | **72.9 ± 8.2** | 2.47 ± 1.03 | **41.7 ± 14.6** | **0.565 ± 0.073** | **43.2 ± 11.1** | 24.04 | – | **219.4** |
| CN + region attention | 68.0 ± 9.2 | 2.86 ± 1.03 | 29.2 ± 13.2 | 0.508 ± 0.082 | 38.0 ± 12.1 | 23.56 | – | 224.7 |
| CN + attention energy | 70.5 ± 8.2 | 3.06 ± 1.40 | 30.6 ± 13.2 | 0.496 ± 0.076 | 42.3 ± 11.5 | **24.22** | – | 226.3 |
| CN best-of-3 | 70.1 ± 8.7 | 2.47 ± 0.87 | 29.2 ± 14.6 | 0.530 ± 0.080 | 41.1 ± 10.8 | 23.62 | – | 225.3 |
| Zhang et al. (re-impl.) | 44.3 ± 10.9 | 3.42 ± 0.85 | 12.5 ± 8.3 | 0.251 ± 0.078 | 10.1 ± 4.7 | 20.08 | 0.276 | 248.0 |
| OCSD-lite † | 66.3 ± 9.1 | **2.25 ± 0.64** | 29.2 ± 13.2 | 0.488 ± 0.078 | 38.8 ± 11.1 | 23.30 | **0.396** | 248.9 |
| OCSD † | 62.5 ± 9.4 | 2.40 ± 0.71 | 29.2 ± 13.2 | 0.459 ± 0.081 | 34.5 ± 11.9 | 23.03 | 0.395 | 278.9 |

**Two-branch generation needs object-aware conditioning.** Our re-implementation of Zhang et al. is the weakest method on QuickDraw-Scenes: with blending switched off halfway ($\alpha=0.5$) and nothing constraining the free phase, objects drift and vanish (OPR 44.3%, RA 10.1%). OCSD, built on the same two-branch idea, is significantly better on every consistency metric: OPR +18.2 points, $\mathrm{OCE}_c$ −1.01, mIoU +0.208, RA +24.4 points and object CLIP +2.95 (Wilcoxon, Holm-adjusted $p$ = 0.022, 0.005, 0.0005, 0.004 and $10^{-8}$). It also keeps object identity better (ID-Sim 0.395 vs. 0.276). This answers Q2 with a qualification: the two-branch decomposition preserves identity, but it only preserves *layout and count* once the scene phase is constrained.

**A strong box-conditioned baseline remains ahead.** GLIGEN leads on OPR, count accuracy, mIoU, RA and FID. Per-object boxes tied to phrases are evidently a stronger grounding signal than feeding the whole sketch to ControlNet, even though GLIGEN ignores stroke shape. Preliminary OCSD does not beat it, nor the plain ControlNet baseline, on OPR or mIoU; none of these differences is significant after Holm correction ($p_{Holm}=1.0$ for every training-free baseline), so on 36 scenes every method except Zhang et al. is statistically indistinguishable from OCSD. The regeneration control is informative: best-of-3 lifts the baseline from 67.4% to 70.1% OPR, which is the gain available from verification alone.

**Counting.** OCSD-lite has the lowest class-wise count error of all methods (2.25), and both OCSD variants have lower $\mathrm{OCE}_c$ than GLIGEN, ControlNet and every attention-control baseline. Attention-energy guidance applied to a ControlNet baseline has the *highest* count error (3.06): exciting object tokens raises presence but also creates duplicates.

**Image quality.** Two-branch methods pay in FID/KID (OCSD 278.9 vs. 225.0 FID). Visible pasting boundaries and the long blending phase selected by tuning ($\alpha=0.1$) both contribute; Section 5.5 shows FID falling monotonically as $\alpha$ grows.

::: {custom-style="TableCaption"}
**Table 3.** Training-free methods on all 72 QuickDraw-Scenes scenes (seed 0). † preliminary.
:::

| Method | OPR (%) ↑ | OCE~c~ ↓ | Count acc. (%) ↑ | mIoU ↑ | RA (%) ↑ | CLIP ↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| SD + ControlNet | 65.2 ± 6.5 | 2.49 ± 0.57 | 26.4 ± 9.7 | 0.493 ± 0.058 | 32.0 ± 7.5 | 27.98 |
| T2I-Adapter | 61.7 ± 7.3 | 2.92 ± 0.65 | 26.4 ± 9.0 | 0.485 ± 0.064 | 29.5 ± 8.6 | 27.30 |
| GLIGEN | **71.4 ± 6.3** | 2.54 ± 0.72 | **38.9 ± 10.4** | **0.548 ± 0.054** | **41.0 ± 8.3** | 27.98 |
| CN + region attention | 66.5 ± 6.8 | 2.83 ± 0.70 | 28.5 ± 10.4 | 0.492 ± 0.059 | 32.8 ± 8.1 | 27.61 |
| CN + attention energy | 70.0 ± 5.7 | 2.78 ± 0.84 | 31.9 ± 10.1 | 0.490 ± 0.056 | 38.0 ± 8.0 | **28.59** |
| CN best-of-3 | 67.8 ± 6.2 | 2.35 ± 0.56 | 29.9 ± 9.7 | 0.507 ± 0.057 | 34.6 ± 7.4 | 28.23 |
| OCSD-lite † | 69.3 ± 6.4 | **2.18 ± 0.52** | 35.4 ± 10.1 | 0.509 ± 0.056 | 39.8 ± 8.7 | 27.62 |

On the full 72 scenes (Table 3), the training-free OCSD-lite is second only to GLIGEN on count accuracy and RA, within 0.7 points of the best ControlNet variant on OPR, beats the ControlNet baseline on every consistency metric (OPR +4.1, RA +7.8 points) and again has the lowest count error.

## 5.3 Degradation with object count and abstraction (E1, E2)

![**Figure 2.** QuickDraw-Scenes, 36-scene subset: OPR (left) and class-wise count error (right) against the number of sketched objects. OCSD rows are preliminary.](figs/fig2_e1.png){width=6.5in}

Figure 2 answers Q1. The ControlNet baseline falls from 94.4% OPR on single objects to 40.1% at 8–10 objects, and its count error grows from 0.28 to 5.33; Zhang et al. collapses fastest (77.8% to 19.0%). The two-branch methods pay a cost on single objects (OCSD 83.3%: generating the object separately and re-composing it loses some objects the baseline would have drawn directly), but degrade more slowly. At 8+ objects, OCSD-lite has the best OPR of all methods (53.1% vs. 47.8% for GLIGEN and 40.1% for the baseline) and the lowest count error (4.56 vs. 6.44 for GLIGEN), and at 3 objects both OCSD variants have the lowest count error (1.00–1.06). This is the regime the method was designed for: independent generation (M2) and sketch-region constraints (M4, M5a) keep many objects from competing in one attention map. Across abstraction levels, all methods lose most on *complex* sketches; the baseline drops from 74.4% (medium) to 62.3% (complex), OCSD-lite from 69.5% to 61.0%, while GLIGEN, which ignores strokes, stays highest on complex sketches (72.2%). This is consistent with abstraction hurting sketch-conditioned methods at the object-generation stage, where ControlNet has to recognise each object from its strokes.

## 5.4 Ablation study (E4)

![**Figure 3.** Preliminary ablation on 18 QuickDraw-Scenes scenes with 3 and 5 objects (one seed): OPR with 95% bootstrap intervals. The dashed line is full OCSD.](figs/fig3_ablation.png){width=5.2in}

::: {custom-style="TableCaption"}
**Table 4.** Preliminary ablation (†), 18 scenes. Rows change one component of full OCSD.
:::

| Configuration | OPR (%) ↑ | OCE~c~ ↓ | mIoU ↑ | RA (%) ↑ | CLIP ↑ | ID-Sim ↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Full OCSD | 59.6 ± 13.0 | 1.92 | 0.420 | 40.9 | 27.80 | 0.349 |
| w/o blended inference ($\alpha=1$) | 57.4 ± 10.2 | 2.17 | 0.278 | 29.1 | 28.00 | 0.294 |
| blending over the whole trajectory ($\alpha=0$) | 60.4 ± 12.0 | 2.22 | 0.437 | 38.8 | 29.29 | 0.426 |
| w/o identity learning (OCSD-lite) | 64.4 ± 11.3 | 1.89 | 0.460 | 43.5 | 29.55 | 0.374 |
| w/o $\mathcal{L}_{att}$ in M3 | 57.0 ± 13.0 | 1.94 | 0.423 | 36.9 | 29.59 | 0.400 |
| M2 with $K=1$ | 54.4 ± 16.5 | 2.00 | 0.398 | 41.0 | 27.48 | 0.348 |
| w/o region attention M5(a) | 63.3 ± 13.3 | 1.83 | 0.444 | 44.7 | 29.07 | 0.385 |
| w/o energy guidance M5(b) | **78.1 ± 8.7** | 1.22 | **0.561** | **61.0** | **31.09** | **0.483** |
| w/o scene ControlNet M5(c) | 53.0 ± 14.6 | 2.11 | 0.385 | 35.0 | 28.43 | 0.360 |
| w/o verification M5(d) | 52.6 ± 16.1 | 2.17 | 0.384 | 36.2 | 28.30 | 0.348 |
| w/o all of M5 | 68.1 ± 13.9 | **1.17** | 0.492 | 50.5 | 30.64 | 0.463 |
| background prompt only | 66.3 ± 11.7 | 2.00 | 0.464 | 44.3 | 29.49 | 0.408 |
| global prompt only | 62.6 ± 13.1 | 2.56 | 0.354 | 42.8 | 28.34 | 0.342 |
| Zhang et al. (re-impl.) | 40.2 ± 9.2 | 3.00 | 0.204 | 13.2 | 24.75 | 0.248 |

The ablation (Table 4, Figure 3) separates helpful from harmful components; with 18 scenes and one seed, individual differences are not significant after Holm correction, so we read directions and effect sizes.

- **Helpful:** the scene ControlNet M5(c) (removing it costs 6.6 OPR points and 5.9 RA points), verification M5(d) (−7.0 OPR, beyond what the best-of-3 control recovers for the baseline), candidate selection in M2 (−5.2 OPR with $K=1$), attention separation (−2.6 OPR, −4.0 RA) and blended inference (without it, mIoU drops from 0.420 to 0.278, the largest layout effect; unadjusted $p=0.008$).
- **Harmful:** attention-energy guidance on identity tokens. Removing M5(b) raises OPR by 18.5 points (59.6% to 78.1%; unadjusted $p=0.029$), mIoU by 0.141, RA by 20.1 points, and lowers the count error from 1.92 to 1.22. Because M5(b) is applied during the customised phase, where the identity tokens are also shaped by LoRA, maximising their attention appears to overshoot: the latent is pushed towards strong but spatially wrong activations. The same mechanism on a plain ControlNet also raised count error (Table 2). Removing all of M5 is better than full OCSD for the same reason, but worse than removing only M5(b) (68.1% vs. 78.1%), which confirms that M5(a), (c) and (d) together contribute about 10 OPR points once M5(b) is gone.
- **Identity learning** currently costs consistency (OCSD-lite 64.4% vs. 59.6%) while it should preserve appearance; ID-Sim is in fact *higher* without it on this subset (0.374 vs. 0.349), because the LoRA-conditioned free phase is exactly where energy guidance acts. Whether identity learning pays off once M5(b) is fixed is the main open question of the re-run.

These findings motivated a second tuning phase on the held-out pilot scenes that chooses between no energy guidance, identity-token guidance ($\eta\in\{10,20\}$) and guidance on the object-phrase tokens ($\eta\in\{10,20\}$), with the same selection rule; only the OCSD-family rows are regenerated.

## 5.5 Effect of α

![**Figure 4.** Preliminary α study on 12 scenes (one seed, verification off): OPR, layout mIoU and FID against α. α = 0 blends during the whole trajectory; α = 1 never blends.](figs/fig4_alpha.png){width=6.5in}

With verification disabled (Figure 4), layout mIoU is flat for $\alpha\le0.4$ (0.38–0.42) and collapses beyond it (0.164 at $\alpha=1$, significantly below full OCSD, $p_{Holm}=0.012$); OPR peaks at $\alpha=0.4$ (67.8%) and falls to 44.4% without blending. FID improves steadily as the blending phase shortens (406.8 at $\alpha=0$ to 328.3 at $\alpha=0.8$). The range reported by Zhang et al. ($\alpha\in[0.4,0.6]$) thus sits on the edge where layout starts to break for multi-object hand-drawn scenes. The tuning split preferred $\alpha=0.1$ with regeneration enabled; the two results are compatible (both favour $\alpha\le0.4$ for layout) but the study grid did not include 0.1, and with 12 scenes the OPR peak at 0.4 is within noise. The trade-off between layout and realism is real, and $\alpha$ is the knob that sets it.

## 5.6 COCO-Sketch

::: {custom-style="TableCaption"}
**Table 5.** COCO-Sketch, 16 scenes common to all methods (one seed). The first row scores the real photographs (detector ceiling). † preliminary.
:::

| Method | OPR (%) ↑ | OCE~c~ ↓ | Count acc. (%) ↑ | mIoU ↑ | RA (%) ↑ | CLIP ↑ | FID ↓ | KID×10³ ↓ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Real images (ceiling) | 57.7 | 2.28 | – | 0.485 | 33.8 | 24.89 | – | – |
| SD + ControlNet | 35.4 ± 17.7 | 2.62 | 12.5 | 0.306 | 10.2 | 24.66 | 262.2 | **6.17** |
| T2I-Adapter | 35.4 ± 18.5 | 2.62 | 18.8 | 0.297 | 12.0 | 24.69 | 252.0 | 7.58 |
| GLIGEN | **51.4 ± 20.3** | **1.88** | **31.2** | **0.441** | **23.3** | 24.62 | **249.0** | 6.26 |
| CN + region attention | 43.8 ± 19.8 | 2.12 | 25.0 | 0.371 | 13.5 | 19.77 | 275.5 | 10.33 |
| CN + attention energy | 40.6 ± 16.9 | 2.50 | 12.5 | 0.297 | 8.5 | 19.48 | 274.8 | 15.64 |
| CN best-of-3 | 33.9 ± 18.0 | 2.62 | 12.5 | 0.293 | 9.7 | **25.45** | 262.8 | 7.45 |
| Zhang et al. (re-impl.) | 49.5 ± 18.5 | 2.50 | 18.8 | 0.369 | 14.9 | 18.86 | 296.0 | 20.85 |
| OCSD-lite † | 47.8 ± 17.8 | 2.06 | 18.8 | 0.364 | 21.9 | 19.99 | 286.6 | 26.78 |
| OCSD † | **51.4 ± 18.8** | 2.00 | **31.2** | 0.392 | 20.8 | 20.08 | 326.6 | 28.21 |

Real COCO scenes are much harder: even the photographs reach only 57.7% OPR under OWLv2, because objects are small and occluded, so every generated result should be read against this ceiling. OCSD ties GLIGEN on OPR (51.4%, 89% of the ceiling) and count accuracy (31.2%) and is 16 points above the ControlNet baseline, but trails it on mIoU, RA and image quality. All methods that restrict attention by region or rewrite latents (region attention, attention energy, Zhang et al., OCSD) lose 4–6 points of global CLIP on COCO but not on QuickDraw-Scenes; PiDiNet scribbles also contain background strokes, and constraining background tokens away from object regions is a plausible cause that we will verify on the images. With 16 scenes the intervals are about ±18 OPR points, so these are trends, not conclusions.

## 5.7 Cost

On the A100, one generation with the ControlNet baseline takes 2.2 s, and best-of-3 5.5 s (2.4 generations on average). OCSD takes 5.7 s per image including verification (2.2 generations on average), plus a per-scene preparation of 12.4 s for M2 and 37.0 s for M3 that is reused across seeds and prompts; peak memory is 12.5 GB versus 12.0 GB. OCSD-lite avoids M3 entirely. When a user changes only the background or $\alpha$, the prepared objects are reused, which the demo application exploits by caching M1–M3.

# 6. Discussion and Limitations

**What the preliminary evidence supports.** (i) Object consistency degrades sharply with object count for sketch-conditioned diffusion (Q1). (ii) The two-branch approach is only competitive once the scene phase is constrained; with object-aware conditioning, OCSD improves over the two-branch baseline significantly on all consistency metrics (Q2). (iii) Region constraints, the scene ControlNet, candidate selection and verification each help, and the training-free OCSD-lite gives the best counting behaviour and the best object preservation in dense scenes (Q3), but attention-energy guidance on identity tokens is harmful and currently masks these gains.

**What it does not support (yet).** OCSD does not beat GLIGEN overall. We do not claim state of the art; the claim that removing M5(b) would lift OCSD above GLIGEN is not established, because the ablation subset (18 scenes, one seed) differs from the main table, and the re-tuned run is required.

**Limitations.** Per-scene identity learning adds about 50 s of preparation. Results depend on auxiliary models: Grounded-SAM masks in M2, Grounding DINO in M5(d) and OWLv2 in the evaluation; the COCO ceiling shows that the evaluation detector alone misses 42% of real objects. QuickDraw-Scenes uses real strokes but synthetic layouts, and COCO-Sketch uses PiDiNet scribbles rather than human scene sketches; FS-COCO [@chowdhury2022fscoco] is the natural next test set. The experiments use a reduced configuration (30 steps, $K=2$, 100 + 100 training steps) and small samples (36 and 16 scenes), so the confidence intervals are wide and most pairwise differences are not significant. Very abstract object sketches make ControlNet produce wrong-class objects already in M2, a limitation shared by ControlNet-based methods [@koley2024sketch; @bourouis2026sketchingreality]. Finally, a user study, planned with an anonymised automatically generated questionnaire following [@zhang2025sketchscene], has not been run.

# 7. Conclusion

We framed sketch-and-text scene generation around object consistency and contributed a method, OCSD, and an evaluation protocol that measures it with an independent detector, a detector ceiling, a fair regeneration control, a held-out tuning split and paired tests. On a common SD 1.5 backbone, object-aware conditioning turns the two-branch approach from the weakest into a competitive method, significantly better than its re-implemented predecessor, with the lowest count errors and the best preservation in dense scenes for its training-free variant, while a box-conditioned model remains the strongest baseline. The ablation identifies attention-energy guidance on identity tokens as the component holding OCSD back; the re-tuned run, a larger evaluation, FS-COCO and a user study are the next steps. Code, benchmarks, per-scene results and the tuning record are available at github.com/bobbibao/ocsd-thesis.

# Acknowledgements

The authors thank the Faculty of Information Technology, Industrial University of Ho Chi Minh City, for supporting this work.

# References
