"""Recap slide (18) had its own copy of the old toy Diffusion figure: swap in the farm figure (same 1900x615 size).
Usage: fix_recap.py IN OUT FIG13_PNG"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[17]
x = [x for x in s.shapes if x.name == 'Picture 21'][0]
rid = x._element.blipFill.blip.rEmbed
part = s.part.related_part(rid)
new = s.part.get_or_add_image_part(open(sys.argv[3], 'rb'))[1]
x._element.blipFill.blip.rEmbed = new
p.save(sys.argv[2]); print('ok')

# Slide 15 box "Nhiễu ban đầu khác → ảnh khác": real SD 1.5 text-only outputs of notebook 09 (seeds 0, 3, 4)
# and a fresh Gaussian-noise tile per seed, instead of the old toy images. Pass SEED_DIR as argv[4].
if len(sys.argv) > 4:
    import io, numpy as np
    from PIL import Image
    s = p.slides[14]; sh = {x.name: x for x in s.shapes}
    for seed, nz, out in ((0, 'Picture 78', 'Picture 80'), (3, 'Picture 81', 'Picture 83'), (4, 'Picture 84', 'Picture 86')):
        im = Image.open(f'{sys.argv[4]}/text_only_s{seed}.png').convert('RGB').resize((256, 256))
        b = io.BytesIO(); im.save(b, 'JPEG', quality=90); b.seek(0)
        sh[out]._element.blipFill.blip.rEmbed = s.part.get_or_add_image_part(b)[1]
        n = np.clip(np.random.default_rng(seed).normal(128, 60, (64, 64, 3)), 0, 255).astype('uint8')
        b = io.BytesIO(); Image.fromarray(n).resize((192, 192), Image.NEAREST).save(b, 'PNG'); b.seek(0)
        sh[nz]._element.blipFill.blip.rEmbed = s.part.get_or_add_image_part(b)[1]
    p.save(sys.argv[2]); print('slide 15 seeds ok')
