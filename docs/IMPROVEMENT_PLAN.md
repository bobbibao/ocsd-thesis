# OCSD improvement plan: what is left for a 10/10 thesis and a publishable paper

Status as of 2026-09-29 (v1). Written from a review of this repository (commit 69750d5), the partial paper-tier
results (`results/paper_run1`: baseline rows final, OCSD-family rows not final), and the thesis draft v4
(`KLTN_Sketch_Text_Diffusion_v4.docx`, 123 pages). The thesis, defense slides and English paper were being
finished in parallel; section 7 is updated once they land.

Priority legend: **P0** = blocks the defense or the paper (do first); **P1** = strongly expected by a committee or
reviewers; **P2** = polish, do if time allows. "Owner" is a suggestion (B = Bảo, C = Cường, Claude = a Claude thread).

---

## 0. The one thing that decides everything

The partial numbers do not yet support the main claim. On QuickDraw-Scenes (36 scenes, 2 seeds):

| Method | OPR | OCE-class | Count acc. | mIoU | RA |
|---|---|---|---|---|---|
| GLIGEN (boxes + text) | **72.9** | 2.47 | **41.7** | **0.565** | **43.2** |
| ControlNet (base) | 67.4 | 2.46 | 25.0 | 0.510 | 36.6 |
| OCSD-lite (not final) | 66.3 | **2.25** | 29.2 | 0.488 | 38.8 |
| OCSD (not final) | 62.5 | 2.40 | 29.2 | 0.459 | 34.5 |

The ablation (E4) shows why: removing M5(b) energy guidance on identity tokens raises OPR from 59.6 to **78.1**,
mIoU 0.420 to 0.561, RA 40.9 to 61.0. That configuration would beat GLIGEN on every consistency metric.
Commit 69750d5 re-tunes M5(b) (off / id tokens / phrase tokens x eta) and regenerates only OCSD-family rows.

**Action (P0, Bobby):** run `RUN_ALL.ipynb` (tier `paper`) once more, then `00_push_logs.ipynb`. Then decide the
storyline from the new E3 table:

- **Case A: tuned OCSD beats GLIGEN and ControlNet on OPR / OCE-class / count accuracy** with Holm-corrected p < 0.05
  on at least OPR and OCE-class. Keep the current storyline ("OCSD improves object consistency").
- **Case B: OCSD beats ControlNet but not GLIGEN.** Reframe honestly: OCSD needs only a sketch (no boxes typed by the
  user), so the fair comparison is against sketch-only methods; GLIGEN is an upper reference that receives
  ground-truth boxes. Say so in the table caption, and add the item in 2.1 (GLIGEN-backed OCSD) as the fix.
- **Case C: OCSD still trails ControlNet.** Lead the contribution with the benchmark, metrics and analysis
  (QuickDraw-Scenes + OWLv2 protocol, the finding that attention energy on learned tokens hurts, the alpha study),
  and present OCSD-lite / the best ablation config as the method. A committee accepts a well-analysed negative result;
  it does not accept a table that contradicts the text.

In every case: **the thesis text, abstract, conclusion and slides must be rewritten from the final numbers**, not
from the expected ones. Search the docx for "cải thiện", "vượt trội", "tốt hơn" and check each claim against a table.

---

## 1. Experiments and results (P0 / P1)

| # | Pri | Item | How | Owner |
|---|---|---|---|---|
| 1.1 | P0 | Finish the paper-tier re-run and copy the full result set | Run all (paper), push logs; copy `results/` (tables md/csv/tex, figures, summary.md, tuning) into `/mnt/project-files/results/paper/` **and** into this repo under `results/paper/` (tables + figures only, no images) so the numbers in the thesis are versioned | Bobby, Claude |
| 1.2 | P0 | Report significance, not only means | `tables/stats_*.csv` already has Wilcoxon + Holm. Put the p-values (or stars) in the main E3 table and state in the text which differences are significant. With n = 36 and CI of about ±9 OPR points, most gaps under 8 points will not be significant; say so | Claude (thesis/paper threads) |
| 1.3 | P0 | Drop FID from headline claims | FID over 36–72 images is not meaningful (all values 220–400, OCSD 279–327). Keep KID (unbiased on small sets) with its std, move FID to the appendix with a sentence explaining sample size. Alternatively compute FID on COCO with ≥ 1,000 images for ControlNet vs OCSD-lite only (no training needed) | Claude |
| 1.4 | P0 | Fill E1 (object count) and E2 (sketch complexity) tables and the OPR-vs-count curve | Produced by `report.py` after the re-run (`E1_*`, `E2_*`, `opr_curves_quickdraw.png`). These are the direct evidence for the research aim ("keeps objects as scenes get harder") | Claude |
| 1.5 | P0 | Run the user study | The anonymised pack is in Drive `results/user_study`. Make a Google Form (10 scenes x 4 methods, randomised order, 3 questions: object correctness, layout match, overall quality), 20+ participants (classmates are fine), report mean rank + Friedman test. Thesis currently has `[số người tham gia]`, `[số bộ ảnh]` | B, C |
| 1.6 | P1 | Explain the CLIP drop on COCO | On COCO-Sketch, OCSD CLIP = 20.1 vs ControlNet 24.7; ControlNet+region (19.8) and Zhang (18.9) drop too. Likely cause: region-restricted cross-attention starves the global prompt. Check one image per method; if confirmed, either keep the global prompt in unmasked heads / early steps, or document it as a trade-off in Limitations | Claude (code thread) |
| 1.7 | P1 | Validate the Zhang et al. (2025) re-implementation | It scores below plain ControlNet (OPR 44 vs 67 on QuickDraw). Reviewers will say the baseline is broken. Reproduce 2–3 figures from their paper with their settings (SD 2.1, alpha 0.4–0.6, LoRA 1.0) and show them side by side in the appendix; state all deviations | Claude (code thread) |
| 1.8 | P1 | More statistical power | If Colab units allow: 3rd seed on the 36 trained scenes (about +4 A100 h), or tier `full` for OCSD-lite + ControlNet + GLIGEN only (training-free, cheap). Report per-scene paired differences as a histogram | Bobby |
| 1.9 | P1 | Real scene sketches | QuickDraw-Scenes is synthetic composition and COCO-Sketch uses PiDiNet edges. Add a small FS-COCO evaluation (public, free-hand scene sketches by people): 30–50 scenes, training-free methods at least. Also include 5–10 sketches drawn by the students in the qualitative figure | Claude (code thread) |
| 1.10 | P1 | Runtime and memory table | `runtime_*` is produced; put it in Chapter 4 and the paper. OCSD needs identity learning per scene; state the minutes per scene honestly and compare with OCSD-lite | Claude |
| 1.11 | P1 | Detector-ceiling row on COCO | Make sure the "real image" row (`detector_ceiling_coco`) appears next to E3_coco so readers see OWLv2's own OPR on real photos | Claude |
| 1.12 | P2 | Failure-case gallery | 6–8 failure cases with a one-line cause each (tiny objects, heavy overlap, rare classes). Committees reward this | Claude |
| 1.13 | P2 | Alpha study with the tuned setting | Current alpha table is not final and non-monotonic (alpha 0.4 best there, tuning picked 0.1). Re-read it after the re-run and make the text match the curve | Claude |

## 2. Method and code (P1 / P2)

| # | Pri | Item | How |
|---|---|---|---|
| 2.1 | P1 | GLIGEN-backed scene branch (only if Case B/C) | OCSD already derives boxes from the sketch (M1). Feed those boxes + phrases to GLIGEN as the scene model in M4 (and/or combine GLIGEN with ControlNet Scribble). This keeps "sketch only, no user boxes" and turns GLIGEN's strength into part of OCSD. Report it as a new row, tuned on the tuning split only |
| 2.2 | P1 | Freeze the final configuration | After the re-run, write the chosen values (`alpha`, `lora_scale`, `energy_tokens`, `eta`) as the defaults in `config.py` and in a table in the README, so the code, thesis Table 3.x and paper agree |
| 2.3 | P1 | Reproducibility | Pin the unpinned packages in `requirements-colab.txt` (`torch-fidelity`, `timm`, `einops`, `scikit-image`, `opencv-python-headless`, `pandas`, `scipy`) to the versions of the final run (`pip freeze` in Colab); record torch/CUDA/GPU in `summary.json`; tag the commit used for the paper (`git tag v1.0-paper`) |
| 2.4 | P1 | Add a LICENSE | There is none. MIT or Apache-2.0 for the code; note that SD 1.5 weights are under CreativeML OpenRAIL-M and QuickDraw is CC BY 4.0 |
| 2.5 | P1 | CI for the CPU tests | GitHub Actions workflow running `tests/test_coco_builder.py`, `tests/test_tune.py`, `tests/test_vision_tiny.py` (and `smoke_test.py` if the tiny bench can be built in CI). Shows the code is maintained |
| 2.6 | P2 | English README and code comments | Needed if the repo is linked from an English paper. Keep a short Vietnamese section for the committee |
| 2.7 | P2 | Release the benchmark | Publish QuickDraw-Scenes (scene JSON, sketches, prompts, masks, splits incl. `tuning_split.json`) as a GitHub release or Zenodo DOI with a datasheet; it is a citable contribution by itself |
| 2.8 | P2 | Demo | `02_app_gradio.ipynb` works on Colab; record a 1-minute screen video for the defense in case the live demo fails |
| 2.9 | P2 | Clean-up | Remove `__pycache__` from the shared folder copy; add `results/` and `*.png` rules to `.gitignore` except the versioned tables |

## 3. Thesis report (.docx, Vietnamese)

| # | Pri | Item | How |
|---|---|---|---|
| 3.1 | P0 | Replace every placeholder | v4 still has **226 yellow `[…]` cells** and 15 bracketed notes (hardware `[tên GPU…]`, `[gói sử dụng…]`, `[theo phiên Colab]`, E1–E2 / E3 / E4 / alpha / qualitative commentary, user-study counts, the Chapter 5 summary). Regenerate from the final tables with the build in `thesis/src` and remove all highlighting. Final check: zero `w:highlight` in the docx |
| 3.2 | P0 | Claims match numbers | See section 0. Abstract (VN + EN), Ch.1 contributions, Ch.4 discussion, Ch.5 conclusion and the research-question answers must quote the final numbers and significance |
| 3.3 | P0 | Confirm facts only the students know | Major ("Khoa học máy tính"?), defense date ("tháng 12 năm 2026"?), method name "OCSD", faculty/university names, the other-thesis-style signature pages, working log (Nhật kí làm việc) with real dates |
| 3.4 | P1 | Template compliance | Check against "Mau bao cao KLTN.docx": margins, fonts, spacing (header and footer are already correct in v4), caption format "Hình c.n:" / "Bảng c.n:", TOC / list of figures / tables / abbreviations refreshed (open in Word, Ctrl+A, F9) |
| 3.5 | P1 | Figures from the real run | Replace illustrative figures in Ch.4 with `qualitative_*.png`, `benchmark_examples.png`, `opr_curves_quickdraw.png`, `alpha_quickdraw.png`; each figure referenced and discussed in the text |
| 3.6 | P1 | Limitations section with substance | Per-scene identity learning cost, dependence on Grounding DINO / SAM quality, small n, synthetic sketches, CLIP drop on COCO, SD 1.5 backbone |
| 3.7 | P1 | Reference hygiene | 82 references: check every one is cited in the text, formats follow the template (tiếng Việt / tiếng Anh groups, year, venue), arXiv-only papers replaced by published versions where they exist (e.g. Zhang et al. is Computers & Graphics 129, 104226, 2025) |
| 3.8 | P1 | Plagiarism / AI-text check | Run the university's similarity checker early; rewrite in the students' own voice any passage they cannot explain at the defense |
| 3.9 | P2 | Length and balance | ~100 pages requested, draft is 123. Chapter 2 can lose theory the method does not use; Chapter 4 should grow with results |
| 3.10 | P2 | Export a PDF | Submit a PDF built from Word (fonts embedded) alongside the docx |

## 4. Research paper (English)

| # | Pri | Item | How |
|---|---|---|---|
| 4.1 | P0 | Same numbers as the thesis | Paper tables generated from the same `results/paper/tables/*.tex` |
| 4.2 | P1 | Pick a venue and format | Realistic: a Vietnamese/regional conference (e.g. SoICT, KSE, RIVF, MAPR) or a journal special issue; or an arXiv preprint first. Convert to that venue's LaTeX template (IEEE / Springer LNCS) and page limit |
| 4.3 | P1 | Sharpen the contribution list | 3 bullets max, each backed by a table: (i) the method (or best configuration), (ii) QuickDraw-Scenes + detector-independent consistency protocol, (iii) the analysis (energy guidance on learned tokens hurts; alpha trade-off) |
| 4.4 | P1 | Related work coverage | Layout-to-image (GLIGEN, BoxDiff, LayoutGuidance, MIGC, InstanceDiffusion), sketch-to-image (ControlNet, T2I-Adapter, FineControlNet, Zhang 2025), counting (Make It Count, CountGen), multi-concept (Break-A-Scene, Custom Diffusion). Say why the heavier ones were not compared (SD version, no code) |
| 4.5 | P2 | Reproducibility statement + code link, ethics note (generated images, dataset licenses) | |

## 5. Defense slides and presentation (Vietnamese)

| # | Pri | Item | How |
|---|---|---|---|
| 5.1 | P0 | Numbers and figures from the final run | Same tables as the thesis; one headline chart (OPR by object count, all methods) |
| 5.2 | P0 | Rehearse to the time limit | Usually 15–20 min; ~1 slide per minute; split speaking between both students |
| 5.3 | P1 | Q&A preparation | Prepare answers for: why SD 1.5 not SDXL; why OCSD vs GLIGEN; why OWLv2 for evaluation (not the detector used inside); is FID meaningful at this sample size; cost per scene; what is new vs Zhang et al. 2025; how the tuning split avoids test leakage; what failed and why (M5(b) finding) |
| 5.4 | P2 | Backup slides | Ablation details, runtime, failure cases, per-class OCE |
| 5.5 | P2 | Demo video | See 2.8 |

## 6. Suggested order of work

1. Re-run paper tier (1.1), copy results into repo and shared folder.
2. Decide Case A/B/C (section 0); if B/C and time allows, implement 2.1 and run it on the test split.
3. Run the user study (1.5) in parallel; it does not need GPU.
4. Regenerate thesis, paper and slides from the final tables (3.1, 3.2, 4.1, 5.1), then the P1 items.
5. Freeze: pin deps, license, tag `v1.0-paper` (2.2–2.4).
6. Template/format pass, similarity check, rehearsal.

## 7. Status of the deliverables (to be updated)

| Deliverable | Where | Status |
|---|---|---|
| Thesis docx | `/mnt/project-files/thesis/KLTN_Sketch_Text_Diffusion_v4.docx` | 123 pages; baselines filled; OCSD-family rows, E1/E2, user study still placeholders |
| Defense slides | pending | being built |
| Research paper docx | pending | being written |
| Results | `/mnt/project-files/results/paper_run1` | baselines final; OCSD-family rows waiting for the re-run |
