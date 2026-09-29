# Research paper (English)

- `OCSD_paper.docx` / `OCSD_paper.pdf`: manuscript, 14 pages, 5 tables, 4 figures, 57 references.
- Numbers: baselines final (run 1, 2026-09-29, A100); OCSD / OCSD-lite / ablation / alpha rows marked † are
  preliminary (before the M5(b) energy-guidance re-tuning). Source: Drive MyDrive/KLTN_OCSD/results summary.md,
  stats_quickdraw.csv, runtime_*.md (same as ../results/paper_run1).
- Rebuild: edit `src/paper.md`; `python3 src/figs_paper.py; python3 src/build_paper.py`. Both scripts expect to sit
  in a `paper/` folder inside the unpacked thesis sources (thesis/src/thesis_build_src.tar.gz) because they reuse
  `figs.py` (diagram helpers) and `refs.py` (reference list). Needs pypandoc_binary, python-docx, matplotlib.

## To finish after the re-run
1. Replace every † row (Tables 2-5, Figs 2-4, abstract numbers) with the re-run values; drop the status note.
2. Rewrite 5.2/5.4/6 around the re-tuned M5(b) result; add GLIGEN on the ablation subset if OCSD w/o M5(b) is compared to it.
3. Add a qualitative figure (qualitative_quickdraw.png on Drive) and the benchmark example figure.
4. Optional for a venue: user study, FS-COCO, full-tier config; convert to the venue's LaTeX template.
