# Research paper (English)

- `OCSD_paper.docx` / `OCSD_paper.pdf`: manuscript, 14 pages, 5 tables, 4 figures, 57 references.
- Numbers: final paper-tier run (2026-10-01, A100) from ../results/paper/; updated 2026-10-02. Only the FID/KID
  columns (marked ‡) are provisional until the evaluation-only pass scores every method on the same images.
- Rebuild: edit `src/paper.md`; `python3 src/figs_paper.py; python3 src/build_paper.py`. Both scripts expect to sit
  in a `paper/` folder inside the unpacked thesis sources (thesis/src/thesis_build_src.tar.gz) because they reuse
  `figs.py` (diagram helpers) and `refs.py` (reference list). Needs pypandoc_binary, python-docx, matplotlib.

## Still open
1. After the evaluation-only pass: replace the ‡ FID/KID values in Tables 2 and 5 and drop the status note.
2. Optional, for a stronger paper: re-select alpha and a lighter M5 (only one of M5(a)/M5(b)) jointly on the tuning
   split; a user study; FS-COCO; more scenes so the gains over GLIGEN can reach significance; the venue's LaTeX template.
