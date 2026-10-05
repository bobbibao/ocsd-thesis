# Research paper (English)

- `OCSD_paper.docx` / `OCSD_paper.pdf`: "Keeping Every Object: A Controlled Study of Sketch-and-Text Guided Scene
  Generation with Diffusion Models". 17 pages, 9 tables, 5 figures, 58 references.
- Numbers: improved run of 2026-10-04 (../results/paper_v2/, code e590c1a), all final incl. FID/KID. Updated 2026-10-05.
- Framing: (1) evaluation protocol, (2) object-first composition (M2 + Collage) is what helps, (3) negative results
  (OCSD/OCSD-v2 do not beat tuned GLIGEN/Collage; M5(a)/(b) and identity learning do not help). No claim that OCSD wins.
- The previous version (OCSD best on OPR, results/paper/) is in git history.

## Swapping in another run
All result tables are generated from the run's CSVs (`src/tables_paper.py`) and figures 3-5 read the same CSVs
(`src/figs_paper.py`). To use another run: `OCSD_RESULTS=/mnt/project-files/results/<run> python3 src/figs_paper.py;
OCSD_RESULTS=... python3 src/build_paper.py`. The prose numbers in `src/paper.md` (abstract, Sections 5-7) still need
a manual pass; the extra object-first tests in Section 5.3 come from `src/extra_stats.py` (output in `extra_stats.txt`).

Build: both scripts expect to sit in a `paper/` folder inside the unpacked thesis sources
(thesis/src/thesis_build_src.tar.gz) because they reuse `figs.py` and `refs.py`. Needs pypandoc_binary, python-docx,
matplotlib; extra_stats.py needs pandas and scipy; LibreOffice Writer for the PDF.

## Still open
1. Pre-register (with an external timestamp: OSF, AsPredicted or a dated e-mail to the advisor) and run the
   combined method: M2 objects as the initial image + GLIGEN grounding (beta 1) + v2 region repair, against GLIGEN
   best-of-N and Collage best-of-3 at matched compute. If it wins, it becomes the paper's method; if not, keep this framing.
2. A larger tuning split (16 scenes cannot rank close configurations, Section 5.6) and more evaluation scenes.
3. FS-COCO (human scene sketches), a user study, and the venue's LaTeX template.
