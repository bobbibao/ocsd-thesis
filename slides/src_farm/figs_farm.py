"""Figures for the deck-wide example scene qd_5_medium_005 (three small cows, a tall tree and a red truck on a farm),
final OCSD, notebook 10 ZIP (Bobby 2026-11-26). Usage (run here): figs_farm.py UNZIPPED_DIR OLD_FIG13.png -> img/"""
import glob, json, shutil, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
R = sys.argv[1].rstrip('/') + '/'; T, B = R + 'trace/', R + 'bgswap/'
L = lambda f, n=512: np.array(Image.open(f).convert('RGB').resize((n, n), Image.LANCZOS))
S = lambda a, f: Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save('img/' + f)
tr = json.load(open(T + 'trace.json')); objs = tr['m1']['objects']
sk = L(T + '00_input/sketch.png'); strokes = sk.astype(int).sum(2) < 380
mask = L(T + 'M4/blend_mask.png')[..., 0] > 127
final = L(T + 'final.png'); S(final, 'final.png'); S(sk, 'sketch.png'); S(L(T + 'M1/parsed_masks_boxes.png'), 'parsed.png')
# ---- M2 (columns: cow 1, tree, truck)
for o, c in ((1, 'cow'), (2, 'tree'), (3, 'truck')):
    S(L(T + f'M2/obj{o}_{c}_sketch_crop.png'), f'crop{o}.png')
    sel = glob.glob(T + f'M2/obj{o}_{c}_cand*_SELECTED.jpg')[0]; oth = [f for f in glob.glob(T + f'M2/obj{o}_{c}_cand*.jpg') if f != sel][0]
    S(L(sel), f'cand{o}_sel.png'); S(L(oth), f'cand{o}_oth.png')
# ---- M3: follow the truck (<o2>)
ref = Image.open(T + 'M2/obj3_truck_selected_cutout.png').convert('RGBA'); bg = Image.new('RGBA', ref.size, 'white'); bg.alpha_composite(ref)
S(np.array(bg.convert('RGB').resize((512, 512))), 'm3_ref.png')
S(L(T + 'M3/obj3_truck_prompt_class_word.jpg'), 'm3_class.png'); S(L(T + 'M3/obj3_truck_prompt_learned_token.jpg'), 'm3_token.png')
# ---- M4: M2 objects on the blue sketch strokes; real U-Net inputs with an orange ring around the M2 regions
a = L(T + 'M4/composed_foreground.jpg'); a[nd.binary_dilation(strokes, iterations=1)] = (46, 90, 156); S(a, 'placed.png')
edge = nd.binary_dilation(mask, iterations=7) & ~nd.binary_dilation(mask, iterations=2)
outer = nd.binary_dilation(mask, iterations=9) & ~nd.binary_dilation(mask, iterations=7)
for st in (1, 12, 18, 24):
    a = L(glob.glob(T + f'M5_denoise/step{st - 1:02d}_t*_unet_input.jpg')[0]); a[edge] = (255, 120, 0); a[outer] = (255, 255, 255); S(a, f'st{st:02d}.png')
S(L(glob.glob(T + 'M5_denoise/step29_t*_predicted_x0.jpg')[0]), 'm4_out.png')
# ---- M5 (a)/(b): attention at step 21 (file step20) with the M1 box in white: cow 1, tree, truck
for i, (o, c) in enumerate(((1, 'cow'), (2, 'tree'), (3, 'truck'))):
    im = Image.fromarray(L(T + f'M5_attention_scene/step20_obj{o}_{c}.jpg')); d = ImageDraw.Draw(im)
    d.rectangle(objs[o - 1]['box'], outline=(255, 255, 255), width=5); im.save(f'img/attn_{i}.png')
a = final.copy(); th = nd.binary_dilation(strokes, iterations=2)
a[nd.binary_dilation(th, iterations=2) & ~th] = (30, 30, 30); a[th] = (255, 255, 255); S(a, 'sketch_on_final.png')
S(L(T + 'M5d/check1.jpg'), 'check1.png')
for f in ('1_in_a_city_park.png', '2_on_a_snowy_field.png', '3_on_a_sandy_beach.png', 'bgswap.json'): shutil.copy(B + f, 'img/' + f)
shutil.copy(T + 'trace.json', 'img/trace.json')

# ---- background: forward diffusion in pixel space on the final image (DDPM linear schedule, T = 1000)
rng = np.random.default_rng(7)
betas = np.linspace(1e-4, 0.02, 1000); ab = np.cumprod(1 - betas)
x0 = final.astype(float) / 127.5 - 1
def noisy(t, eps): return (np.sqrt(ab[t - 1]) * x0 + np.sqrt(1 - ab[t - 1]) * eps + 1) * 127.5
def noise_img(eps): return (np.clip(eps, -2.5, 2.5) / 2.5 + 1) * 127.5
E = [rng.standard_normal(x0.shape) for _ in range(4)]
S(final, 'fw_t0.png')
for k, t in enumerate((150, 400, 1000)):
    S(noisy(t, E[k]), f'fw_t{t}.png'); S(noise_img(E[k]), f'fw_eps{k}.png')
S(noisy(400, E[3]), 'tr_noisy.png'); S(noise_img(E[3]), 'tr_eps.png')
# figure on the Diffusion/SD slide: same layout as the old figure, panels replaced
fig = Image.open(sys.argv[2]).convert('RGB'); n = 320
for x, t in ((10, 0), (400, 150), (790, 350), (1180, 600), (1570, 999)):
    a = final if t == 0 else noisy(t, rng.standard_normal(x0.shape))
    fig.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((n, n), Image.LANCZOS), (x, 143))
fig.save('img/fig13.png')
# reverse: the REAL OCSD trajectory (30 DDIM steps, decoded); 'U-Net noise' recomputed from x_t and predicted x0
sab = np.cumprod(1 - np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2)
def step(k):
    xi = glob.glob(T + f'M5_denoise/step{k:02d}_t*_unet_input.jpg')[0]; t = int(xi.split('_t')[1][:3])
    return t, L(xi), L(xi.replace('unet_input', 'predicted_x0'))
for k, kk in enumerate((0, 19, 25)):
    t, xt, xp = step(kk); S(xt, f'rv_x{k}.png')
    e = ((xt / 127.5 - 1) - np.sqrt(sab[t]) * (xp / 127.5 - 1)) / np.sqrt(1 - sab[t])
    e = (e - e.mean()) / (e.std() + 1e-6); S(noise_img(e), f'rv_eps{k}.png'); print('rev', kk, t)
for k, kk in enumerate((0, 5, 11, 17, 23)):
    t, xt, _ = step(kk); S(xt, f'rs_{k}.png'); print('small', kk, t)
print('ok')
