# Research paper (English)

- `OCSD_paper.docx` / `OCSD_paper.pdf`: "Keeping Every Object in Shape: Sketch-Silhouette Conditioning for
  Object-Consistent Scene Generation, and a Controlled Study of What Does Not Help". 23 pages, 12 tables, 7 figures,
  60 references. Updated 2026-11-05.
- Proposed model: OCSD v3 = OCSD instance parser + sketch-silhouette parser + InstanceDiffusion (Wang et al., CVPR 2024,
  external, credited) with silhouettes released after the layout steps + SDXL refiner (Section 3.9).
- Section 5 = the October controlled study of the original OCSD (results/paper_v2/, unchanged).
  Section 6 = locked held-out studies v1, v2, v3 (results/heldout_v1, heldout_v2, heldout_v3).
- Claims kept to the evidence: v3 vs original OCSD mIoU +0.195 (Holm p 0.023), OPR +0.128 trend (Holm p 0.070),
  naturalness 1.42 vs 0.36; parser vs boxes on the same generator better chamfer/facing; level with Collage; no SOTA;
  8+ object counting unsolved; 18-scene held-out sets, full-benchmark confirmation pending.

## Rebuild
Scripts expect to sit in a `paper/` folder inside the unpacked thesis sources (thesis/src/thesis_build_src.tar.gz),
because they reuse `figs.py` and `refs.py`. Needs pypandoc_binary, python-docx, matplotlib, pymupdf (pandas, scipy for
extra_stats.py), and LibreOffice Writer **plus libreoffice-math** for the PDF (without it, equations vanish).
`python3 figs_paper.py; python3 build_paper.py`. Tables come from the result files (`tables_paper.py`); result folders
can be overridden with OCSD_RESULTS, OCSD_HELDOUT1/2/3. Prose numbers in `paper.md` need a manual pass after a new run.

## Still open
1. Run OCSD v3 on the full benchmark (72 QuickDraw-Scenes + 32 COCO-Sketch) and add it to Section 5's tables.
2. Counting in 8+ object scenes; a multi-rater review; FS-COCO human scene sketches; the venue's LaTeX template.
