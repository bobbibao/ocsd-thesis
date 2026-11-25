"""Figures for the new example scene qd_3_simple_003 (2 gray birds + 1 tree by a lake), final OCSD, notebook 08
(Bobby 2026-11-26). Usage: figs_bird.py UNZIPPED_DIR (contains trace/ and bgswap/)  -> img/"""
import glob, json, shutil, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
R = sys.argv[1].rstrip('/') + '/'; T, B = R + 'trace/', R + 'bgswap/'
L = lambda f, n=512: np.array(Image.open(f).convert('RGB').resize((n, n), Image.LANCZOS))
S = lambda a, f: Image.fromarray(a).save('img/' + f)
tr = json.load(open(T + 'trace.json'))
sk = L(T + '00_input/sketch.png'); strokes = sk.astype(int).sum(2) < 380
mask = L(T + 'M4/blend_mask.png')[..., 0] > 127
S(sk, 'sketch.png'); S(L(T + 'M1/parsed_masks_boxes.png'), 'parsed.png')
for o, c in ((1, 'tree'), (2, 'bird'), (3, 'bird')):
    S(L(T + f'M2/obj{o}_{c}_sketch_crop.png'), f'crop{o}.png')
    S(L(T + f'M2/obj{o}_{c}_cand1_SELECTED.jpg'), f'cand{o}_1.png'); S(L(T + f'M2/obj{o}_{c}_cand2.jpg'), f'cand{o}_2.png')
# M3: follow bird 2 (<o1>)
S(L(T + 'M2/obj2_bird_selected_cutout.png'), 'm3_ref.png')
S(L(T + 'M3/obj2_bird_prompt_class_word.jpg'), 'm3_class.png'); S(L(T + 'M3/obj2_bird_prompt_learned_token.jpg'), 'm3_token.png')
final = L(T + 'final.png'); S(final, 'final.png')
# M4: M2 objects placed on the blue sketch strokes; real U-Net inputs with an orange ring around the M2 regions
a = L(T + 'M4/composed_foreground.jpg'); a[nd.binary_dilation(strokes, iterations=1)] = (46, 90, 156); S(a, 'placed.png')
edge = nd.binary_dilation(mask, iterations=7) & ~nd.binary_dilation(mask, iterations=2)
outer = nd.binary_dilation(mask, iterations=9) & ~nd.binary_dilation(mask, iterations=7)
for st in (1, 12, 18, 24):
    a = L(glob.glob(T + f'M5_denoise/step{st - 1:02d}_t*_unet_input.jpg')[0]); a[edge] = (255, 120, 0); a[outer] = (255, 255, 255); S(a, f'st{st:02d}.png')
S(L(glob.glob(T + 'M5_denoise/step29_t*_predicted_x0.jpg')[0]), 'm4_out.png')
# M5 (a)/(b): attention at step 11 (file step10) with the M1 box in white
for i, (o, c) in enumerate(((1, 'tree'), (2, 'bird'), (3, 'bird'))):
    im = Image.fromarray(L(T + f'M5_attention_scene/step10_obj{o}_{c}.jpg')); d = ImageDraw.Draw(im)
    d.rectangle(tr['m1']['objects'][i]['box'], outline=(255, 255, 255), width=5); im.save(f'img/attn_{i}.png')
# M5 (c): sketch strokes over the final image; (d): the check image with detector boxes
a = final.copy(); th = nd.binary_dilation(strokes, iterations=2)
a[nd.binary_dilation(th, iterations=2) & ~th] = (30, 30, 30); a[th] = (255, 255, 255); S(a, 'sketch_on_final.png')
S(L(T + 'M5d/check1.jpg'), 'check1.png')
for f in ('1_in_a_city_park.png', '2_on_a_snowy_field.png', '3_on_a_sandy_beach.png', 'bgswap.json'): shutil.copy(B + f, 'img/' + f)
shutil.copy(T + 'trace.json', 'img/trace.json')
print('ok', float(mask.mean()), float(np.abs(L(T + 'final.png').astype(int) - L(glob.glob(T + 'M5_denoise/step29_t*_predicted_x0.jpg')[0]).astype(int)).mean()))
