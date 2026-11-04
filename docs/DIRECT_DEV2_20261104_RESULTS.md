# Sketch-silhouette dev study results (run `dev2_76dcb0d55956ca5e9135`, 2026-11-04)

Protocol: [DIRECT_DEV2_PROTOCOL.md](DIRECT_DEV2_PROTOCOL.md) (main `29e4459`). 8 dev scenes x 2 seeds x 3 arms.
**Development data, not a paper result.**

## Audit

137/137 ZIP files match `checksums.json`; status `complete`, 48/48 images, one attempt, no failures; code `29e4459a`;
NVIDIA L4; 1046 s for the attempt including model load; about 19 s per image for every arm (the silhouette input adds
no measurable time); peak CUDA 5.95 GiB for every arm. Conditioning recorded per image:
`boxes+centers` / `+sketch_silhouette_mask+256pt_polygon` / `+20_ink_points+sketch_silhouette_mask+256pt_polygon`.

## Review (analyst review by Claude, at the user's request; not blind)

The user asked Claude to review and escalate only hard cases. Claude designed the method, saw the images unblinded
([scenes 1-4](direct_dev2/dev2_unblinded_scenes1-4.jpg), [5-8](direct_dev2/dev2_unblinded_scenes5-8.jpg); columns:
sketch, box, silhouette, silhouette+scribble), and used the held-out v1 criteria (shape includes pose/facing).

| pair | box | silhouette | silhouette+scribble | most natural with right structure |
|---|---|---|---|---|
| dev1 s0 | colour fail (both cars red) | colour fail (right car red) | colour fail, cartoon (nat 0) | none |
| dev1 s1 | ok (rear views) | ok, box-shaped panels (nat 1) | ok, cartoon + panels (nat 0) | box |
| dev2 s0 | ok | ok, tree follows the U sketch | colour fail (black cat is black-and-white), flat | silhouette |
| dev2 s1 | ok | ok | ok, flat | silhouette |
| dev3 s0 | **shape fail**: umbrellas tilted on the ground | ok: upright with J handles | ok, palm in a pot | silhouette |
| dev3 s1 | **shape fail**: umbrellas not upright | ok: upright, J handles | ok | silhouette |
| dev4 s0 | ok | ok | ok, flat | tie |
| dev4 s1 | ok | ok | ok, studio background | tie |
| dev5 s0 | **shape fail**: dog faces away from the sketch's direction | ok: dog faces right as drawn | ok, studio | silhouette |
| dev5 s1 | **shape fail**: dog lying, head down; extra person | ok: dog standing, faces right | ok, studio | silhouette |
| dev6 s0 | ok | ok, post-and-rail fence as drawn (nat 1) | ok (nat 1) | box |
| dev6 s1 | ok | ok, fence fragment on a wooden wall (nat 1) | ok (nat 1) | box |
| dev7 s0 | ok | ok: steam engine with chimney as drawn | ok, flat | silhouette |
| dev7 s1 | ok | ok | ok | tie |
| dev8 s0 | ok (kayak-like) | ok, canoe follows crescent | ok | tie |
| dev8 s1 | ok (kayak) | ok, canoe follows crescent | ok | silhouette |

Structure failures: **box 5, silhouette 1, silhouette+scribble 2**. Picks: silhouette 9, box 3, silhouette+scribble 0,
tie 4. Silhouette+scribble follows outlines most tightly but often looks flat or cartoon-like.

Automatic sanity check of the new sketch-fidelity metric (GrabCut instead of SAM, because the sandbox cannot download
SAM; dev data only, used to fix the metric thresholds): mean sketch chamfer box 0.081, silhouette 0.058,
silhouette+scribble 0.037; facing accuracy on asymmetric drawings 0.50 / 0.85 / 0.95. The dog facing flips seen by eye
(dev5) are detected by the metric (box 0/2, silhouette 2/2).

## Locked rule applied

Rule 2: kept silhouette arm = fewer structure failures -> `direct_silhouette` (1 vs 2). Rule 3: 4 fewer structure
failures than `direct_box` (>= 3) -> **goes to a new locked held-out comparison**:
[HELDOUT2_PROTOCOL.md](HELDOUT2_PROTOCOL.md).

## Limits

8 dev scenes, author-reviewer, not blind: direction only. The parser stays thin on some open drawings (mountain).
