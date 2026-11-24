"""Figures for the 2-slide diffusion explainer. DDPM, linear beta 1e-4..0.02, T=1000, computed."""
import numpy as np, sys, os
from PIL import Image
MEDIA, OUT = sys.argv[1:3]
T = 1000; beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
ab = lambda t: 1.0 if t == 0 else abar[t - 1]
def load(f, s=128): return np.asarray(Image.open(os.path.join(MEDIA, f)).convert('RGB').resize((s, s), Image.LANCZOS), np.float32) / 127.5 - 1
def save(x, name, s=360):
    Image.fromarray((np.clip((x + 1) / 2, 0, 1) * 255).astype(np.uint8)).resize((s, s), Image.LANCZOS).save(os.path.join(OUT, name))
for t in (100, 300, 500, 800): print(t, round(float(np.sqrt(ab(t))), 2), round(float(np.sqrt(1 - ab(t))), 2))
rng = np.random.default_rng(7); x0 = load('image95.jpg'); eps = rng.standard_normal(x0.shape).astype(np.float32)
for t in (0, 100, 300, 500, 800, 1000): save(np.sqrt(ab(t)) * x0 + np.sqrt(1 - ab(t)) * eps, f'a_fwd_{t}.png')
save(x0, 'a_x0.png'); save(eps * 0.5, 'a_eps.png'); save(np.sqrt(ab(300)) * x0 + np.sqrt(1 - ab(300)) * eps, 'a_x300.png')
SET = ['image95.jpg', 'image75.jpeg', 'image12.jpeg', 'image19.jpeg', 'image100.jpg', 'image85.jpeg',
       'image88.jpeg', 'image83.jpeg', 'image104.jpg', 'image97.jpg', 'image74.jpeg', 'image80.jpeg']
D = np.stack([load(f) for f in SET]).reshape(len(SET), -1)
def x0_hat(x, t):
    a_ = ab(t); d2 = ((x[None] - np.sqrt(a_) * D) ** 2).sum(1)
    lw = -d2 / (2 * (1 - a_)); w = np.exp(lw - lw.max()); w /= w.sum(); return w @ D
KEEP = (1000, 800, 600, 400, 200, 0)
for seed in range(400):
    r = np.random.default_rng(seed); x = r.standard_normal(D.shape[1]); snaps = {}
    for t in range(T, 0, -1):
        xh = x0_hat(x, t)
        if t in KEEP: snaps[t] = (x.copy(), xh.copy())
        e = (x - np.sqrt(ab(t)) * xh) / np.sqrt(1 - ab(t))
        mean = (x - beta[t - 1] / np.sqrt(1 - ab(t)) * e) / np.sqrt(alpha[t - 1])
        sig = np.sqrt(beta[t - 1] * (1 - ab(t - 1)) / (1 - ab(t)))
        x = mean + (sig * r.standard_normal(x.shape) if t > 1 else 0)
    snaps[0] = (x, x)
    if int(((D - x) ** 2).sum(1).argmin()) == 0: break
print('seed', seed)
for t in KEEP:
    save(snaps[t][0].reshape(128, 128, 3), f'b_x_{t}.png'); save(snaps[t][1].reshape(128, 128, 3), f'b_x0hat_{t}.png')
