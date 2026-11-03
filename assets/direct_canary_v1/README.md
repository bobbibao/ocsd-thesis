# Fixed development canary inputs

These are real human QuickDraw strokes, not model outputs. Source: [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`scene.json` preserves the source URL, original `key_id`, zero-based line index and strokes. Selection was fixed before any generated output: the first two recognized bicycle records and first recognized bench record at index >= 3000 in the public simplified NDJSON files. These are outside the first 3000 records used by the repository's usual QuickDraw downloader. This does not prove that the generative model never saw these drawings; no training-data independence claim is made.

OCSD transforms the strokes to a 512×512 canvas using the recorded placement boxes, preserving aspect ratio, rounding coordinates and drawing 3-pixel black lines. Conditioning uses tight ink bounding boxes, per-instance text and 20 sampled ink points; no oracle segmentation masks. The raw scene is new, but the subjects were chosen in response to earlier development failures. It is **not a held-out paper benchmark**.

`protocol.json` fixes the two seeds, two arms, 50 PLMS steps, alpha, MIS, negative prompt, 864 total UNet sample calls, 600-second child-process cap, and model revisions/checksums from the successful CPU preflight. There is no parameter sweep, candidate selection or automatic retry after an error.
