# OCSD improvement plan: what is left for a 10/10 thesis and a publishable paper

Status as of 2026-10-02 (v3). Written from this repository (main at ee72f6f), the final paper-tier results
(`/mnt/project-files/results/paper/`, run 2026-10-01 on A100), the thesis `KLTN_Sketch_Text_Diffusion_FINAL_v2.docx`
(132 pages), the defense slides (34), the English paper (15 pages) and the to-do notes of the thesis and paper threads.

Priority legend: **P0** = do before the defense or a submission; **P1** = strongly expected by a committee or
reviewers; **P2** = polish. Owner is a suggestion (B = Bảo, C = Cường, Claude = a Claude thread).
Section 1 lists what is done; sections 2–6 list only what is still open.

---

## 0. Where the result stands

The final run supports the claim on the mean, but not yet with statistical significance.

QuickDraw-Scenes, 36 identity-learning scenes x 2 seeds (`tables/E3_quickdraw`):

| Method | OPR ↑ | OCE-class ↓ | Count acc. ↑ | mIoU ↑ | RA ↑ |
|---|---|---|---|---|---|
| **OCSD** | **75.4** | **1.92** | 36.1 | 0.547 | **52.3** |
| OCSD-lite (no identity learning) | 73.0 | 2.14 | 33.3 | 0.550 | 47.9 |
| GLIGEN (boxes + text) | 72.9 | 2.47 | **41.7** | **0.565** | 43.2 |
| ControlNet (base) | 67.4 | 2.46 | 25.0 | 0.510 | 36.6 |
| Zhang et al. 2025 (re-impl.) | 44.3 | 3.42 | 12.5 | 0.251 | 10.1 |

- OCSD vs every single-branch method: **no difference survives Holm correction** (`tables/stats_quickdraw.csv`;
  vs GLIGEN OPR p = 0.30 raw; vs ControlNet p = 0.07 raw). Only vs T2I-Adapter (RA, object CLIP) and vs Zhang
  (all metrics, p_holm < 0.001) is it significant.
- The gain is concentrated on crowded scenes: at 8+ objects OCSD reaches 64.4% OPR vs GLIGEN 47.8% and
  ControlNet 40.1% (`E1_quickdraw`). This is the strongest story the data offers.
- COCO-Sketch (16 scenes): OCSD 53.0% OPR vs GLIGEN 51.4%, ControlNet 35.4%; real photos reach 57.7% under the same
  detector, so OCSD is at about 92% of the detector ceiling. No COCO difference is significant either.
- Ablation (18 scenes): removing M5(b) energy guidance (81.1% OPR) or M5(a) region attention (78.5%) alone scores
  **higher** than full OCSD (74.1%); removing all of M5 gives 70.7%. M5(a) and M5(b) overlap and should not both be on.
- Alpha sweep (12 test scenes): best OPR at α = 0.6 (77.8%), best OCE at α = 0.4, while tuning picked α = 0.1.
  The sweep runs on test scenes, so it must not be used to pick α (see 2.3).

The thesis, paper and slides already state all of this honestly. The plan below is about turning "best on average"
into "significantly better" and closing the remaining gaps.

---

## 1. Done since v2

- Final paper-tier run with tuned energy guidance (phrase tokens, η = 20; old identity-token setting ranked last).
- All tables from `results/paper/`: E1 (by object count), E2 (by complexity), E3, E3all, E4, alpha, runtime,
  detector ceiling, Wilcoxon–Holm statistics. "Run 1" labels removed everywhere.
- Qualitative QuickDraw and COCO grids, benchmark examples, OPR-by-count and alpha charts back in the thesis, paper
  and slides.
- Report fixed to read only the paper plan's scenes; FID now computed on the same images for every method (ee72f6f).
- Thesis FINAL_v2 (132 pages, no placeholders), slides (34), paper (15 pages, 57 references) all on final numbers.
- Runtime reported: OCSD 4.9 s per image plus 12.4 s (M2) and 35.6 s (M3) per scene, 7.7 GB VRAM; OCSD-lite skips M3.

### 1b. Implemented in code on 2026-10-04, waiting for a GPU run

One Run all of the `paper` tier does all of this: stage T runs only the new phase 3, then stage B regenerates only the
(split, method) folders whose settings changed and adds the new jobs. Rough cost on A100 (from the measured times):
~1 h tuning phase 3, ~0.6 h for the power job, ~0.6 h for the two new ablation rows, plus what phase 3 makes stale:
~3 h when it changes α or M5 (every OCSD-family row), ~0.3 h otherwise (only the COCO rows, for the caption).
Run 2.1 (evaluation only) before pulling this code if the FID/KID numbers of the current images are still needed.

- **2.3** Tuning phase 3 (`config.TUNE_M5`, `TUNE_M5_ALPHA`; `stages._tune_m5_variants`): {M5(a)+M5(b), M5(a) only,
  M5(b) only} x α ∈ {0.1, 0.3, 0.4, 0.5, 0.6} on the 16 tuning scenes, same rule, phase-2 setting as reference.
  Table `results/tuning/tuning_m5_table.md`. Ablation row "+ both M5(a) and M5(b)" added; a row that tuning makes
  identical to OCSD is skipped.
- **2.4** Job `E3_quickdraw_power` (OCSD, OCSD-lite, GLIGEN, ControlNet on all 72 scenes x 2 seeds). Tests fixed in
  advance in `docs/PREREGISTRATION.md`; tables `E3power_quickdraw`, `E3power8_quickdraw`, `prereg_quickdraw`.
  Commit that file before the run so its date precedes the images.
- **2.6** Caption appended to P_g as free tokens when it adds information (`OCSDConfig.use_caption`, COCO only; also for
  the cn_region / cn_energy baselines, not Zhang). Check E3 COCO global CLIP after the run.
- **2.7** Ablation row "+ M3 with the full 200 + 200 steps" and the OCSD vs OCSD-lite test on 72 scenes (secondary
  pre-registered family).
- **3.1** Code defaults frozen to the tuned values (α 0.1, LoRA 0.5, phrase tokens, η 20, caption on), listed in the
  README; Zhang et al. pinned to its own setup; tier/tuned values no longer overwrite a variant's own settings.
- Report: E4 and E3all rows now average the same seeds for every method.

---

## 2. Experiments (open)

| # | Pri | Item | How | Owner |
|---|---|---|---|---|
| 2.1 | P0 | **Final FID/KID** | Run all once more (evaluation only, no generation). Then replace the provisional FID/KID column in thesis Table E3 COCO, §4.6.1 "Chất lượng ảnh", limitations and Ch.5; the ‡ values in paper Tables 2 and 5; slide 26 (†). Remove the "giá trị tạm thời" notes. Keep KID as the main quality number; FID on 16–72 images is relative only | Bobby, then Claude |
| 2.2 | P0 | **Run the user study** | Kit is ready (thesis Appendix D, Drive `results/user_study`). Google Form: 10 scenes x 4 methods (OCSD, GLIGEN, ControlNet, Zhang), randomised, 3 questions (right objects, matches sketch layout, overall quality). 15–20+ respondents; report mean rank + Friedman / Wilcoxon. Fill thesis §4.x and add one paper paragraph | B, C |
| 2.3 | P1 | **Lighter M5 + α, tuned jointly on the tuning split** (code done, §1b) | New tuning phase over {M5(a) only, M5(b) only, both} x α ∈ {0.1, 0.3, 0.4, 0.5, 0.6} on the 16 pilot scenes (same selection rule), then regenerate only OCSD-family rows. Never pick α from the test-set sweep. Expected +4–7 OPR (unconfirmed) | Claude (code thread), Bobby runs |
| 2.4 | P1 | **Statistical power** (code done, §1b) | Cheapest route to a significant headline: OCSD (or OCSD-lite) + GLIGEN + ControlNet on all 72 QuickDraw scenes with 2 seeds (E3all already has OCSD-lite on 72 at seed 0). Pre-state the 8+ object subgroup as a test before running so the crowded-scene claim is not post hoc | Bobby (GPU), Claude |
| 2.5 | P1 | **Validate the Zhang et al. re-implementation** | It scores below plain ControlNet on QuickDraw (44.3 vs 67.4) but near GLIGEN on COCO. Reviewers will question it. Reproduce 2–3 of their figures with their settings (SD 2.1, α 0.4–0.6, LoRA 1.0) for the appendix and list every deviation | Claude (code thread) |
| 2.6 | P1 | **Explain the COCO CLIP drop** (fix in code, §1b) | On COCO, global CLIP falls to about 19–20 for region/energy/Zhang/OCSD vs 24.7 for ControlNet, but not on QuickDraw. Likely the prompt built from object phrases. Check images; fix (keep the caption as global prompt) or document as a trade-off | Claude (code thread) |
| 2.7 | P1 | **Identity learning (M3) value** (runs added, §1b) | OCSD beats OCSD-lite by only 2.4 OPR (not significant) at a cost of ~36 s per scene. Either show M3 helps at 200+200 steps or on identity-heavy scenes (ID-Sim 0.467 vs 0.433 is the supporting number), or present OCSD-lite as the fast variant | Claude |
| 2.8 | P2 | **Stronger baselines** | One recent layout method with SD 1.5 code (BoxDiff, Attend-and-Excite on GLIGEN, or MIGC) | Claude (code thread) |
| 2.9 | P2 | **Real free-hand scene sketches** | Small FS-COCO evaluation (30–50 scenes, training-free methods at least) and 5–10 sketches drawn by the students in the qualitative figure. Replace the approximate FS-COCO pilot numbers in thesis Table 4.7 with exact ones if available | Claude, B, C |
| 2.10 | P2 | **Failure-case gallery** | 6–8 failures with a one-line cause each (tiny objects, heavy overlap, rare classes) | Claude |

## 3. Code and reproducibility (open)

| # | Pri | Item | How |
|---|---|---|---|
| 3.1 | P1 | Freeze the final configuration (done, §1b) | `config.py` defaults are still alpha 0.5, lora_scale 1.0, energy_tokens "id"; the run reads the tuned values from `results/tuning/tuned.json`. Set defaults to alpha 0.1, lora_scale 0.5, energy_tokens "phrase", eta 20 (or the 2.3 result) and list them in the README |
| 3.2 | P1 | Version the results | Commit `results/paper/` tables, tuning and figures (not per-image images) under `results/paper/` in this repo, so every thesis number traces to a commit |
| 3.3 | P1 | Pin dependencies and tag | Pin the unpinned packages in `requirements-colab.txt` to the final run's versions (`pip freeze` in Colab); tag the commit used for the paper `v1.0-paper` |
| 3.4 | P1 | Add a LICENSE | None exists. MIT or Apache-2.0 for code; note SD 1.5 is CreativeML OpenRAIL-M and QuickDraw is CC BY 4.0 |
| 3.5 | P1 | CI for the CPU tests | GitHub Actions running `tests/test_coco_builder.py`, `tests/test_tune.py`, `tests/test_vision_tiny.py` |
| 3.6 | P2 | English README | Needed when the paper links the repo; keep a short Vietnamese section |
| 3.7 | P2 | Release the benchmark | QuickDraw-Scenes (scene JSON, sketches, prompts, masks, splits incl. tuning split) as a GitHub release or Zenodo DOI with a datasheet |
| 3.8 | P2 | Demo backup | Screenshot of the Gradio app (`02_app_gradio.ipynb`) for thesis §4.10.2 and a 1-minute screen recording for the defense |

## 4. Thesis report (.docx, Vietnamese)

| # | Pri | Item | How |
|---|---|---|---|
| 4.1 | P0 | FID/KID update | See 2.1 |
| 4.2 | P0 | Confirm cover details with the advisor | Major "Khoa học máy tính", date "tháng 12 năm 2026", method name OCSD; faculty name, signature pages, working log with real dates |
| 4.3 | P0 | User study section | See 2.2; Chapter 4 currently says it is designed but not run |
| 4.4 | P1 | Template compliance pass in Word | Margins, fonts, captions "Hình c.n:" / "Bảng c.n:", refresh TOC and lists (Ctrl+A, F9), compare against "Mau bao cao KLTN.docx" |
| 4.5 | P1 | File size | The docx is 16 MB because of the qualitative figures; recompress them if the submission portal has a limit |
| 4.6 | P1 | Similarity check | Run the university's checker early; rewrite in the students' own voice anything they cannot explain at the defense |
| 4.7 | P1 | Reference hygiene | Each reference cited in the text; published versions instead of arXiv where they exist |
| 4.8 | P2 | Length | 132 pages vs ~100 requested; Chapter 2 can lose theory the method does not use |
| 4.9 | P2 | Gradio screenshot | See 3.8 |

Rebuild: edit `content/*.md` in `thesis/src/thesis_build_src.tar.gz`, run `python3 figs3.py && python3 assemble.py`.

## 5. Research paper (English)

| # | Pri | Item | How |
|---|---|---|---|
| 5.1 | P0 | Replace the ‡ FID/KID values | See 2.1 |
| 5.2 | P1 | Pick a venue and template | Realistic: SoICT, KSE, RIVF, MAPR, or an arXiv preprint first; convert to the venue's LaTeX (IEEE / LNCS) and page limit |
| 5.3 | P1 | Headline claim | "Object-aware two-branch generation preserves more objects than single-branch sketch control, most clearly for 8+ objects", with the 8+ subgroup test from 2.4 and the user study from 2.2 |
| 5.4 | P1 | Contributions that already stand | Two benchmarks with OPR / per-class OCE by an independent detector plus a detector ceiling; evidence that object count, not sketch abstraction, drives failures; a fair best-of-3 control for detector-based regeneration; the finding that attention-energy guidance on learned identity tokens hurts and that M5(a)/M5(b) are redundant |
| 5.5 | P2 | Reproducibility and ethics statements | Code link with tag (3.3), dataset licenses, generated-image disclosure |

Rebuild: edit `paper/src/paper.md`; run `figs_paper.py` then `build_paper.py` inside the unpacked thesis sources.

## 6. Defense (Vietnamese slides)

| # | Pri | Item | How |
|---|---|---|---|
| 6.1 | P0 | Slide 26 FID/KID | See 2.1 |
| 6.2 | P0 | Rehearse to the time limit | 34 slides is long for 15–20 min; move background slides 8–14 to backup if the limit is 15 min; split speaking between both students |
| 6.3 | P1 | Q&A preparation | Why SD 1.5; why OCSD's lead over GLIGEN is not significant and what would make it so; GLIGEN gets boxes, OCSD derives them from the sketch; why OWLv2 for scoring; FID at this sample size; cost per scene; what is new vs Zhang et al.; how the tuning split prevents leakage; why removing M5(a) or M5(b) helps |
| 6.4 | P2 | Backup slides | Full ablation, runtime, failure cases, per-count tables |

Rebuild: `slides/src/build.js` (see `slides/src/README.md`).

## 7. Suggested order

1. Evaluation-only Run all (2.1) and update FID/KID in thesis, paper and slides.
2. User study (2.2) in parallel; it needs no GPU.
3. Advisor confirmation of cover details (4.2), template pass, similarity check, rehearsal.
4. If GPU time allows before the defense: 2.3 (lighter M5 + α), then 2.4 (more scenes).
5. Before submitting the paper: 2.5–2.7, freeze and tag the code (3.1–3.5).

## 8. Where everything is

All paths are in the project's shared folder (`/mnt/project-files/`).

| Deliverable | Where | Status |
|---|---|---|
| Thesis (VN) | `thesis/KLTN_Sketch_Text_Diffusion_FINAL_v2.docx` (+ `.pdf`); sources `thesis/src/thesis_build_src.tar.gz`; notes `thesis/THESIS_IMPROVEMENTS.md` | 132 pages, final numbers; FID/KID provisional |
| Defense slides (VN) | `slides/OCSD_BaoVe_KLTN.pptx`; sources `slides/src/` | 34 slides, final numbers; slide 26 FID/KID provisional |
| Research paper (EN) | `paper/OCSD_paper.docx` (+ `.pdf`); sources `paper/src/`; also branch `claude/project-thread-qysrwj` | 15 pages; ‡ FID/KID provisional |
| Results | `results/paper/` (tables, figures, tuning, per_image, summary.md) | Final except FID/KID; replaces `results/paper_run1/` |
| Code | github.com/bobbibao/ocsd-thesis, main | Run at 69750d5; report fixes in ee72f6f |
