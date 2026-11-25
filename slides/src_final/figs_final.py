"""Figures for the FINAL OCSD (harmonised M4) slides, from notebook 07 ZIPs (Bobby, 2026-11-25).
Usage: python figs_final.py BGSWAP_DIR SLIDEIMG_DIR   -> img/"""
import json, sys, shutil, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
B, Q = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
L = lambda f, n=512: np.array(Image.open(f).convert('RGB').resize((n, n), Image.LANCZOS))
S = lambda a, f: Image.fromarray(a).save('img/' + f)
meta = json.load(open(B + 'bgswap.json'))
obj = L(B + 'objects_m2.png'); sk = L(B + 'sketch.png')
mask = nd.binary_fill_holes(nd.binary_closing(obj.astype(int).sum(2) < 735, iterations=3))
mask = nd.binary_opening(mask, iterations=2)
strokes = sk.astype(int).sum(2) < 380
# input: M2 objects on blue sketch strokes
a = obj.copy(); a[nd.binary_dilation(strokes, iterations=1)] = (46, 90, 156); S(a, 'placed.png')
# real U-Net inputs with orange ring around the M2 regions
edge = nd.binary_dilation(mask, iterations=7) & ~nd.binary_dilation(mask, iterations=2)
outer = nd.binary_dilation(mask, iterations=9) & ~nd.binary_dilation(mask, iterations=7)
for st in (1, 12, 18, 24):
    a = L(B + f'steps/step{st:02d}_unet_input.jpg'); a[edge] = (255, 120, 0); a[outer] = (255, 255, 255); S(a, f'st{st:02d}.png')
S(L(B + 'steps/step30_x0.jpg'), 'm4_out.png')
S(L(B + '0_near_a_train_station.png'), 'final.png')
# M5 (a)/(b): attention at step 12 with the object's sketch box in white
boxes = meta['steps']['region_masks_boxes']
for i in range(3):
    f = [q for q in meta['steps']['attention']['12']][i].split('/')[-1]
    im = Image.fromarray(L(B + 'attention/' + f)); d = ImageDraw.Draw(im)
    d.rectangle(boxes[i], outline=(255, 255, 255), width=5); im.save(f'img/attn_{i}.png')
# M5 (c): sketch strokes (white, dark outline) over the final image
a = L(B + '0_near_a_train_station.png'); th = nd.binary_dilation(strokes, iterations=2)
a[nd.binary_dilation(th, iterations=2) & ~th] = (30, 30, 30); a[th] = (255, 255, 255); S(a, 'sketch_on_final.png')
for f in ('sketch.png', '1_on_a_city_street.png', '2_on_a_snowy_field.png', '3_on_a_sandy_beach.png', 'bgswap.json'):
    shutil.copy(B + f, 'img/' + f)
shutil.copy(B + '0_near_a_train_station.png', 'img/0_near_a_train_station.png')
# qualitative + old-vs-new + limitation images
for sid in ('qd_1_medium_003', 'qd_3_medium_003', 'qd_8p_complex_003', 'qd_1_medium_004', 'qd_3_simple_005',
            'qd_5_simple_003', 'qd_8p_simple_003', 'qd_8p_simple_005', 'qd_5_medium_003'):
    for arm in ('ocsd', 'ocsd_final', 'gligen', 'collage', 'controlnet'):
        Image.open(Q + f'{arm}/{sid}_s0.jpg').convert('RGB').save(f'img/q_{arm}_{sid}.jpg', quality=92)
    Image.open(Q + f'scenes/{sid}/sketch.png').convert('RGB').resize((384, 384)).save(f'img/q_sketch_{sid}.png')
    shutil.copy(Q + f'scenes/{sid}/scene.json', f'img/q_scene_{sid}.json')
print('ok', mask.mean())
