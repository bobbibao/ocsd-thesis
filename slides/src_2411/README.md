# Defense deck, 24-11 update

`../OCSD_BaoVe_KLTN_24-11.pptx` (48 slides) = `OCSD_BaoVe_KLTN_17-11_XL.pptx` plus:
- "OCSD-XL" written as "OCSD" everywhere (text and charts; the SD 1.5 version stays "OCSD gốc").
- Slides 14-20: diffusion model explained in depth (idea/ink analogy, forward process, training,
  reverse process, sampling, 2D toy example, bridge to Stable Diffusion); slide 21 = slide 13 again
  with the Stable Diffusion half highlighted.

Rebuild (from the project files `slides/` folder):
```
python diff_figs.py <unzipped deck>/ppt/media figs     # noising/denoising figures, computed (DDPM, T=1000)
cd charts && python ../charts_2411.py                   # needs results/xl_paper_v1/tables/by_count_bin.csv
python build_2411.py OCSD_BaoVe_KLTN_17-11_XL.pptx figs charts raw.pptx
python omml.py raw.pptx OCSD_BaoVe_KLTN_24-11.pptx       # native PowerPoint equations
```
Reverse-process figures use the ideal (posterior-mean) denoiser over 12 deck images instead of a
trained U-Net, so they are real DDPM trajectories that end on one of those images (stated on the slides).
