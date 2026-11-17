# Research paper (English)

- `OCSD_paper.docx` / `OCSD_paper.pdf`: "Keeping Every Object in Shape: Two Improved Models for Object-Consistent
  Sketch-and-Text Scene Generation, and a Controlled Study of What Does Not Help". 27 pages, 17 tables, 7 figures,
  60 references. Updated 2026-11-17.
- Base method: the original OCSD on SD 1.5 (Sections 3.3-3.7), studied in Section 5 (results/paper_v2/, unchanged).
- Improved model 1, OCSD-XL (Sections 3.10, 7): unchanged M1-M5 (M3 off) on SDXL + xinsir scribble ControlNet, two
  frozen settings (natural: omega 0.4 + energy 0.02; faithful: omega 0.8, no energy). Results results/xl_confirm_v1
  (held-out v3, Table 15) and results/xl_paper_v1 (72 scenes, Tables 16-17). Faithful vs original OCSD on 72 scenes:
  OPR 0.825 vs 0.711 (Holm p 0.0003), mIoU 0.623 vs 0.529 (0.0005), not significant on held-out v3. Natural:
  naturalness 1.08 vs 0.22 (0.0001), OPR 0.819 (0.0003); also holds on held-out v3.
- Improved model 2, OCSD v3 (Sections 3.9, 6): instance parser + sketch-silhouette parser + InstanceDiffusion (external,
  credited) + SDXL refiner. Held-out v1-v3 (Tables 10-12) and 72-scene confirmation (results/paper_v3, Tables 13-14).
- No SOTA claim: OCSD v3 / GLIGEN / InstanceDiffusion more natural than OCSD-XL; OCSD v3 better mIoU; 8+ object counting
  unsolved; one blind rater (the developer). 18-scene and 72-scene results never pooled.

## Rebuild
Scripts expect to sit in a `paper/` folder inside the unpacked thesis sources (thesis/src/thesis_build_src.tar.gz),
because they reuse `figs.py` and `refs.py`. Needs pypandoc_binary, python-docx, matplotlib, pymupdf (pandas, scipy for
extra_stats.py), and LibreOffice Writer **plus libreoffice-math** for the PDF (without it, equations vanish).
`python3 figs_paper.py; python3 build_paper.py`. Tables come from the result files (`tables_paper.py`); result folders
can be overridden with OCSD_RESULTS, OCSD_HELDOUT1/2/3. Prose numbers in `paper.md` need a manual pass after a new run.

The 72-scene tables read OCSD_PAPER3 (default /mnt/project-files/results/paper_v3/paper3_20261105); the OCSD-XL tables read
OCSD_XL_CONFIRM and OCSD_XL_PAPER (default /mnt/project-files/results/xl_confirm_v1 and xl_paper_v1, copied from
branch claude/project-thread-ezrky1).

## Still open
1. Run OCSD v3 and OCSD-XL on COCO-Sketch; one OCSD-XL setting that is both natural and faithful.
2. Counting in 8+ object scenes; a multi-rater review; FS-COCO human scene sketches; the venue's LaTeX template.
