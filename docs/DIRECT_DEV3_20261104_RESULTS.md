# Dev v3 results (2026-11-04): OCSD v3 settings frozen

Run `dev3_c4c37905e295bb1fb60f` (code 9b2e6a6, L4). The audit is clean:

* 64 of 64 images were made, using 6912 of 6912 planned UNet calls (only the two unrefined arms run the diffusion model).
* Wall time was 1021 s. That is about 18.9 s per InstanceDiffusion image plus 1.2 s per refiner pass. Peak memory was 6.6 GiB.
* Refiner revision: `5d4cfe854c9a9a87939ff3653551c2b3c99a4356`.
* SAM revision: `87aecf0df4ce6b30cd7de76e87673c49644bdf67`.

Protocol: [DIRECT_DEV3_PROTOCOL.md](DIRECT_DEV3_PROTOCOL.md).

The review was done by the analyst (Claude), since Bobby delegated review. It was written blind, before `review_key.json` was opened: [analyst_review_blind.txt](direct_dev3_20261104/analyst_review_blind.txt), sha256 `648a3058…7798`. Unblinded numbers: [analysis.json](direct_dev3_20261104/analysis.json).

| arm | structure failures (of 16) | mean naturalness (0-2) | sketch chamfer (lower better) | facing acc. (14 asymmetric instances) | admissible |
|---|---|---|---|---|---|
| sil_full (= held-out v2 winner) | 5 | 1.56 | **0.0353** | **1.00** | yes (reference) |
| sil_layout | **2** | **1.75** | 0.0449 | 0.79 | **no**: facing below 0.90 |
| sil_full_refine | 5 | 1.56 | 0.0383 | 0.93 | yes |
| sil_layout_refine | **2** | 1.69 | 0.0451 | 0.93 | yes |

Blind "most natural with right structure" picks:

| pick | sheets |
|---|---|
| tie | 5 |
| sil_layout_refine | 3 |
| sil_full | 3 |
| sil_full_refine | 2 |
| none | 2 |
| sil_layout | 1 |

**Decision (locked rule):** keep **`sil_layout_refine`**. Among the admissible arms it has the highest naturalness, 0.125 above `sil_full`, which is outside the 0.10 tie band. It also has fewer structure failures (2 vs 5).

This configuration is frozen as **OCSD v3** in `ocsd/ocsd_v3.py` (`FROZEN`) and in `assets/heldout_v3/protocol.json`.

**Honest notes.**

* Releasing the silhouettes trades some sketch fidelity for naturalness. Chamfer is 0.045 vs 0.035, just inside the 0.0453 limit.
* The release alone loses facing direction (0.79). The refiner restores facing to 0.93, which is the reason both stages are kept together.
* Most structure failures are colour failures shared by all four arms ("white car" drawn red, "white cup" drawn black on one seed). The silhouette settings do not fix these. They come from the generator's attribute binding.
* The remaining naturalness problems also show up in every arm: panel/frame seams on two seeds (cars seed 1, train seed 1).
* These are 8 development scenes, not evidence for the paper. The held-out v3 comparison against the original OCSD is the test ([HELDOUT3_PROTOCOL.md](HELDOUT3_PROTOCOL.md)).
