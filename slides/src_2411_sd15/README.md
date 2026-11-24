# Defense deck, single OCSD (SD 1.5, M1-M5)

`../OCSD_BaoVe_KLTN_24-11_SD15.pptx` (47 slides) = `OCSD_BaoVe_KLTN_24-11.pptx` with only one OCSD:
- SDXL variant removed: slide 35 (SDXL method slide) deleted, SDXL mentions on slides 10, 11 removed;
  InstanceDiffusion removed from the list of compared methods (slide 23).
- Results slides (now 39-45) use the thesis run `results/paper_v2` (QuickDraw-Scenes, E3 36 scenes x 2 seeds,
  E1 by object count, E4 ablation, runtime table): ControlNet, Zhang et al., GLIGEN, Collage, OCSD.
- Qualitative slide: 4 columns (sketch, baseline, GLIGEN, OCSD SD 1.5); limitations show an SD 1.5 OCSD failure.

Rebuild: `cd charts && python ../charts_sd15.py && cd .. && python single_ocsd.py ../OCSD_BaoVe_KLTN_24-11.pptx ../OCSD_BaoVe_KLTN_24-11_SD15.pptx`
