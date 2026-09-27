# OCSD improvement plan: what is left for a 10/10 thesis and a publishable paper

Status as of 2026-09-29 (v2). Written from a review of this repository (commit 69750d5), the partial paper-tier
results (`results/paper_run1`: baseline rows final, OCSD-family rows from run 1 and preliminary), the final thesis
draft (`KLTN_Sketch_Text_Diffusion_FINAL.docx`, 128 pages, no placeholders), the defense slides (25 slides), the
English paper (14 pages) and the thesis thread's own to-do list (`thesis/THESIS_IMPROVEMENTS.md`, merged in here).
All files are in the project's shared folder; section 7 lists them.

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

The final thesis draft already reports run 1 honestly (OCSD beats Zhang et al. by 18 OPR points, significant after
Holm; does not beat single-branch baselines on QuickDraw; ties GLIGEN on 8+ objects and on COCO; M5(b) hurts), and
marks OCSD-family rows "lần chạy 1". After the re-run: **replace those rows and rewrite Ch.4–5, both abstracts, the
paper's † rows and sections 5.2/5.4/6, and the slides from the new numbers.** Check every "cải thiện", "vượt trội",
"tốt hơn" against a table.

A related decision (from the thesis review): **identity learning M3 does not pay off at 100+100 steps**, since
OCSD-lite scores at or above full OCSD. Either show M3 helps at 200+200 steps (tier `full`), or make OCSD-lite
(training-free, about 18 s per scene) the main method and present M3 as optional. The second is the stronger,
cheaper paper story.

---

## 1. Experiments and results (P0 / P1)

| # | Pri | Item | How | Owner |
|---|---|---|---|---|
| 1.1 | P0 | Finish the paper-tier re-run and copy the full result set | Run all (paper), push logs; copy `results/` (tables md/csv/tex, figures, summary.md, tuning) into `/mnt/project-files/results/paper/` **and** into this repo under `results/paper/` (tables + figures only, no images) so the numbers in the thesis are versioned | Bobby, Claude |
| 1.2 | P0 | Report significance, not only means | `tables/stats_*.csv` already has Wilcoxon + Holm. Put the p-values (or stars) in the main E3 table and state in the text which differences are significant. With n = 36 and CI of about ±9 OPR points, most gaps under 8 points will not be significant; say so | Claude (thesis/paper threads) |
| 1.3 | P0 | Drop FID from headline claims | FID over 36–72 images is not meaningful (all values 220–400, OCSD 279–327). Keep KID (unbiased on small sets) with its std, move FID to the appendix with a sentence explaining sample size. Alternatively compute FID on COCO with ≥ 1,000 images for ControlNet vs OCSD-lite only (no training needed) | Claude |
| 1.4 | P0 | Fill E1 (object count) and E2 (sketch complexity) tables and the OPR-vs-count curve | Produced by `report.py` after the re-run (`E1_*`, `E2_*`, `opr_curves_quickdraw.png`). These are the direct evidence for the research aim ("keeps objects as scenes get harder") | Claude |
| 1.5 | P0 | Run the user study | Do it after the re-run so it uses the final OCSD images. The anonymised pack is in Drive `results/user_study` (thesis Appendix D). Make a Google Form (10 scenes x 4 methods, randomised order, 3 questions: object correctness, layout match, overall quality), 20+ participants (classmates are fine), report mean rank + Friedman test. Chapter 4 currently says it is designed but not yet run | B, C |
| 1.5b | P0 | Put the qualitative figures back | The final thesis dropped the benchmark examples, failure examples, qualitative comparison, identity-across-backgrounds figure and app screenshot because they were not reachable. Copy them from Drive `results/figures/` (`qualitative_*.png`, `benchmark_examples.png`) into thesis Ch.4, the paper and the slides; add a Gradio screenshot. Committees and reviewers expect them | Bobby, Claude |
| 1.6 | P1 | Explain the CLIP drop on COCO | On COCO-Sketch, OCSD CLIP = 20.1 vs ControlNet 24.7; ControlNet+region (19.8) and Zhang (18.9) drop too. Likely cause: region-restricted cross-attention starves the global prompt. Check one image per method; if confirmed, either keep the global prompt in unmasked heads / early steps, or document it as a trade-off in Limitations | Claude (code thread) |
| 1.7 | P1 | Validate the Zhang et al. (2025) re-implementation | It scores below plain ControlNet (OPR 44 vs 67 on QuickDraw). Reviewers will say the baseline is broken. Reproduce 2–3 figures from their paper with their settings (SD 2.1, alpha 0.4–0.6, LoRA 1.0) and show them side by side in the appendix; state all deviations | Claude (code thread) |
| 1.8 | P1 | More statistical power | If Colab units allow: 3rd seed on the 36 trained scenes (about +4 A100 h), or tier `full` for OCSD-lite + ControlNet + GLIGEN only (training-free, cheap). Report per-scene paired differences as a histogram | Bobby |
| 1.9 | P1 | Real scene sketches | QuickDraw-Scenes is synthetic composition and COCO-Sketch uses PiDiNet edges. Add a small FS-COCO evaluation (public, free-hand scene sketches by people): 30–50 scenes, training-free methods at least. Also include 5–10 sketches drawn by the students in the qualitative figure | Claude (code thread) |
| 1.10 | P1 | Runtime and memory table | `runtime_*` is produced; put it in Chapter 4 and the paper. OCSD needs identity learning per scene; state the minutes per scene honestly and compare with OCSD-lite | Claude |
| 1.11 | P1 | OPR relative to the detector ceiling on COCO | Real COCO photos reach only 57.7% OPR under OWLv2 (already in the thesis). Also report each method's OPR as a fraction of that ceiling | Claude |
| 1.11b | P1 | Stronger baselines for the paper | Add one recent layout method with SD 1.5 code, e.g. BoxDiff or Attend-and-Excite on top of GLIGEN, or MIGC if time allows | Claude (code thread) |
| 1.11c | P2 | Exact FS-COCO pilot numbers | Thesis Table 4.7 uses values read off a chart and says "approximate"; replace with exact values if the students still have them | B, C |
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
| 3.1 | P0 | Update the run-1 rows after the re-run | Placeholders are all filled in the FINAL draft. After the re-run, replace the OCSD / OCSD-lite / E4 / alpha rows marked "lần chạy 1", set the final M5(b) value in Table 3.x (default vs experiment columns), and rebuild: edit `content/*.md` in `thesis/src/thesis_build_src.tar.gz`, run `python3 figs3.py && python3 assemble.py`. The docx still has 3 highlighted runs; check and clear them |
| 3.2 | P0 | Claims match numbers | See section 0. Abstract (VN + EN), Ch.1 contributions, Ch.4 discussion, Ch.5 conclusion and the research-question answers must quote the final numbers and significance |
| 3.3 | P0 | Confirm facts only the students know | With the advisor: major ("Khoa học máy tính"?), defense date ("tháng 12 năm 2026"?), method name "OCSD"; also faculty/university names, signature pages, working log (Nhật kí làm việc) with real dates |
| 3.4 | P1 | Template compliance | Check against "Mau bao cao KLTN.docx": margins, fonts, spacing (header and footer are already correct in v4), caption format "Hình c.n:" / "Bảng c.n:", TOC / list of figures / tables / abbreviations refreshed (open in Word, Ctrl+A, F9) |
| 3.5 | P1 | Figures from the real run | Replace illustrative figures in Ch.4 with `qualitative_*.png`, `benchmark_examples.png`, `opr_curves_quickdraw.png`, `alpha_quickdraw.png`; each figure referenced and discussed in the text |
| 3.6 | P1 | Limitations section with substance | Per-scene identity learning cost, dependence on Grounding DINO / SAM quality, small n, synthetic sketches, CLIP drop on COCO, SD 1.5 backbone |
| 3.7 | P1 | Reference hygiene | 82 references: check every one is cited in the text, formats follow the template (tiếng Việt / tiếng Anh groups, year, venue), arXiv-only papers replaced by published versions where they exist (e.g. Zhang et al. is Computers & Graphics 129, 104226, 2025) |
| 3.8 | P1 | Plagiarism / AI-text check | Run the university's similarity checker early; rewrite in the students' own voice any passage they cannot explain at the defense |
| 3.9 | P2 | Length and balance | ~100 pages requested, draft is 128. Chapter 2 can lose theory the method does not use; Chapter 4 should grow with results |
| 3.10 | P2 | Export a PDF | Submit a PDF built from Word (fonts embedded) alongside the docx |

## 4. Research paper (English)

| # | Pri | Item | How |
|---|---|---|---|
| 4.1 | P0 | Replace the † rows after the re-run | Tables 2–5, Figs 2–4 and the abstract numbers are marked † (preliminary); replace them from the same `results/paper/tables/*` as the thesis and drop the status note. Rebuild: edit `paper/src/paper.md`, run `figs_paper.py` then `build_paper.py` (they expect to sit inside the unpacked thesis sources) |
| 4.2 | P1 | Pick a venue and format | Realistic: a Vietnamese/regional conference (e.g. SoICT, KSE, RIVF, MAPR) or a journal special issue; or an arXiv preprint first. Convert to that venue's LaTeX template (IEEE / Springer LNCS) and page limit |
| 4.3 | P1 | Sharpen the contribution list | Target claim after the re-run: "object-aware two-branch generation (without energy guidance) preserves more objects than single-branch sketch control, especially for 8+ objects and on real images". Contributions that already stand: the two benchmarks with OPR / per-class OCE by an independent detector plus a detector ceiling; evidence that object count, not sketch abstraction, drives failures; the fair best-of-3 control for detector-based regeneration; the negative result on attention-energy guidance |
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

## 7. Status of the deliverables (2026-09-29)

All paths are in the project's shared folder (`/mnt/project-files/`).

| Deliverable | Where | Status |
|---|---|---|
| Thesis docx (VN) | `thesis/KLTN_Sketch_Text_Diffusion_FINAL.docx` (+ `.pdf`), sources `thesis/src/thesis_build_src.tar.gz` | 128 pages, no placeholders. OCSD-family rows are from run 1 and marked preliminary; 5 figures dropped (1.5b) |
| Defense slides (VN) | `slides/OCSD_BaoVe_KLTN.pptx`, sources `slides/src/` | 25 slides; needs the re-run numbers, qualitative figures, rehearsal (section 5) |
| Research paper (EN) | `paper/OCSD_paper.docx` (+ `.pdf`), sources `paper/src/`; also on branch `claude/project-thread-qysrwj` | 14 pages, 5 tables, 4 figures, 57 references; † rows preliminary (4.1) |
| Results | `results/paper_run1` | Baselines final; OCSD-family rows wait for the M5(b) re-run (1.1) |
| Thesis to-do list | `thesis/THESIS_IMPROVEMENTS.md` | Merged into this plan |
