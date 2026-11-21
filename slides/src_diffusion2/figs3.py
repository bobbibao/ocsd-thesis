"""Extra visual figures (no formulas): training examples, one reverse step, different seeds -> different images."""
import numpy as np, sys, os
from PIL import Image
MEDIA, OUT = sys.argv[1:3]
T = 1000; beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
ab = lambda t: 1.0 if t == 0 else abar[t - 1]
def load(f, s=128): return np.asarray(Image.open(os.path.join(MEDIA, f)).convert('RGB').resize((s, s), Image.LANCZOS), np.float32) / 127.5 - 1
def save(x, name, s=360):
    Image.fromarray((np.clip((x + 1) / 2, 0, 1) * 255).astype(np.uint8)).resize((s, s), Image.LANCZOS).save(os.path.join(OUT, name))
SET = ['image95.jpg', 'image75.jpeg', 'image12.jpeg', 'image19.jpeg', 'image100.jpg', 'image85.jpeg',
       'image88.jpeg', 'image83.jpeg', 'image104.jpg', 'image97.jpg', 'image74.jpeg', 'image80.jpeg']
D = np.stack([load(f) for f in SET]).reshape(len(SET), -1)
# training examples: other images of the set at different noise levels
rng = np.random.default_rng(11)
for k, (i, t) in enumerate([(1, 100), (2, 250), (4, 400)]):
    x0 = D[i].reshape(128, 128, 3); e = rng.standard_normal(x0.shape).astype(np.float32)
    save(np.sqrt(ab(t)) * x0 + np.sqrt(1 - ab(t)) * e, f'c_tr{k}_noisy.png'); save(e * 0.5, f'c_tr{k}_noise.png'); save(x0, f'c_tr{k}_clean.png')
def x0_hat(x, t):
    a_ = ab(t); d2 = ((x[None] - np.sqrt(a_) * D) ** 2).sum(1)
    lw = -d2 / (2 * (1 - a_)); w = np.exp(lw - lw.max()); w /= w.sum(); return w @ D
def run(seed, keep=()):
    r = np.random.default_rng(seed); x = r.standard_normal(D.shape[1]); x_init = x.copy(); snaps = {}
    for t in range(T, 0, -1):
        xh = x0_hat(x, t)
        if t in keep: snaps[t] = (x.copy(), xh.copy())
        e = (x - np.sqrt(ab(t)) * xh) / np.sqrt(1 - ab(t))
        mean = (x - beta[t - 1] / np.sqrt(1 - ab(t)) * e) / np.sqrt(alpha[t - 1])
        sig = np.sqrt(beta[t - 1] * (1 - ab(t - 1)) / (1 - ab(t)))
        x = mean + (sig * r.standard_normal(x.shape) if t > 1 else 0)
    return x_init, x, snaps
# one reverse step on the cow trajectory (seed 5 = the slide's cow)
_, _, sn = run(5, keep=(500, 300))
save(sn[500][0].reshape(128, 128, 3), 'c_step_xa.png'); save(sn[500][1].reshape(128, 128, 3), 'c_step_guess.png')
save(sn[300][0].reshape(128, 128, 3), 'c_step_xb.png')
# different starting noise -> different images
got, seed, k = set(), 6, 0
while k < 3:
    xi, xf, _ = run(seed); idx = int(((D - xf) ** 2).sum(1).argmin())
    if idx not in got and idx != 0:
        got.add(idx); save(xi.reshape(128, 128, 3), f'c_seed{k}_noise.png'); save(xf.reshape(128, 128, 3), f'c_seed{k}_img.png'); k += 1
        print('seed', seed, SET[idx])
    seed += 1
