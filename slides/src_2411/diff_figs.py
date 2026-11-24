"""Figures for the diffusion explainer slides (defense deck, 24-11).
All noising/denoising here is computed, not drawn: DDPM linear schedule, T=1000.
Reverse trajectories use the ideal (posterior-mean) denoiser over a small image set,
so they are real DDPM samples, just with an oracle instead of a trained U-Net."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
MEDIA = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'
INK, MUT, ORG, BLU = '#1F2430', '#5B6270', '#D9692B', '#2E5A9C'

T = 1000
beta = np.linspace(1e-4, 0.02, T)          # beta_1..beta_T
alpha = 1 - beta
abar = np.cumprod(alpha)                     # abar[t-1] = abar_t
def ab(t): return 1.0 if t == 0 else abar[t - 1]

def load(f, s=128):
    im = Image.open(os.path.join(MEDIA, f)).convert('RGB').resize((s, s), Image.LANCZOS)
    return np.asarray(im, np.float32) / 127.5 - 1
def show(x): return (np.clip((x + 1) / 2, 0, 1) * 255).astype(np.uint8)
def save(x, name, s=300):
    Image.fromarray(show(x)).resize((s, s), Image.NEAREST if x.shape[0] < s else Image.LANCZOS).save(os.path.join(OUT, name))

rng = np.random.default_rng(7)
x0 = load('image95.jpg')
eps = rng.standard_normal(x0.shape).astype(np.float32)

# ---- forward strip (same eps for all t: closed form x_t = sqrt(abar) x0 + sqrt(1-abar) eps)
TS = [0, 100, 200, 300, 500, 1000]
for t in TS:
    save(np.sqrt(ab(t)) * x0 + np.sqrt(1 - ab(t)) * eps, f'fwd_{t}.png')
print('sqrt abar at', {t: round(float(np.sqrt(ab(t))), 3) for t in TS})
print('abar 500', ab(500), 'sqrt', np.sqrt(ab(500)), np.sqrt(1 - ab(500)))

# ---- training-flow images
t_tr = 300
save(x0, 'tr_x0.png'); save(eps * 0.5, 'tr_eps.png')
save(np.sqrt(ab(t_tr)) * x0 + np.sqrt(1 - ab(t_tr)) * eps, 'tr_xt.png')
eh = 0.92 * eps + 0.39 * rng.standard_normal(eps.shape).astype(np.float32)
save(eh * 0.5, 'tr_epshat.png')

# ---- signal / noise curve
ts = np.arange(0, T + 1)
a = np.array([ab(t) for t in ts])
fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=250)
ax.plot(ts, np.sqrt(a), color=BLU, lw=3, label='phần ảnh gốc còn lại  √ᾱₜ')
ax.plot(ts, np.sqrt(1 - a), color=ORG, lw=3, label='phần nhiễu  √(1−ᾱₜ)')
ax.axvline(500, color='#C9CDD4', ls=':', lw=1.5)
ax.annotate('t = 500: ảnh còn 28%,\nnhiễu chiếm 96%', xy=(500, np.sqrt(ab(500))), xytext=(560, 0.5), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle='->', color=MUT))
ax.set_xlabel('bước t', fontsize=10, color=INK); ax.set_xlim(0, 1000); ax.set_ylim(0, 1.04)
ax.tick_params(labelsize=8.5, colors=MUT); [ax.spines[s].set_visible(False) for s in ['top', 'right']]
ax.legend(fontsize=8.5, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.72))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'curve.png')); plt.close()

# ---- ideal-denoiser DDPM sampling on a small image set
SET = ['image95.jpg', 'image75.jpeg', 'image12.jpeg', 'image19.jpeg', 'image100.jpg', 'image85.jpeg',
       'image88.jpeg', 'image83.jpeg', 'image104.jpg', 'image97.jpg', 'image74.jpeg', 'image80.jpeg']
S2 = 128
D = np.stack([load(f, S2) for f in SET]).reshape(len(SET), -1)
def x0_hat(x, t):
    a_ = ab(t)
    d2 = ((x[None] - np.sqrt(a_) * D) ** 2).sum(1)
    lw = -d2 / (2 * (1 - a_)); w = np.exp(lw - lw.max()); w /= w.sum()
    return w @ D
def sample(seed, keep):
    r = np.random.default_rng(seed); x = r.standard_normal(D.shape[1]); snaps = {T: x.copy()}
    for t in range(T, 0, -1):
        xh = x0_hat(x, t)
        e = (x - np.sqrt(ab(t)) * xh) / np.sqrt(1 - ab(t))          # eps_theta(x_t, t)
        mean = (x - beta[t - 1] / np.sqrt(1 - ab(t)) * e) / np.sqrt(alpha[t - 1])
        sig = np.sqrt(beta[t - 1] * (1 - ab(t - 1)) / (1 - ab(t)))
        x = mean + (sig * r.standard_normal(x.shape) if t > 1 else 0)
        if t - 1 in keep: snaps[t - 1] = x.copy()
    return snaps
RT = [1000, 500, 300, 200, 100, 0]
got, seed, k = set(), 0, 0
while k < 3:
    sn = sample(seed, set(RT)); idx = int(((D - sn[0]) ** 2).sum(1).argmin())
    if idx not in got:
        got.add(idx)
        for t in RT: save(sn[t].reshape(S2, S2, 3), f'rev_{k}_{t}.png')
        print('row', k, 'seed', seed, SET[idx]); k += 1
    seed += 1
print('reverse done')

# ---- 2D toy: points on a heart
def heart(n, r):
    u = r.uniform(0, 2 * np.pi, n)
    x = 16 * np.sin(u) ** 3; y = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
    P = np.stack([x, y], 1) / 17.0
    return P + 0.02 * r.standard_normal(P.shape)
r = np.random.default_rng(0)
DATA = heart(600, r)
def x0h2(X, t):
    a_ = ab(t)
    d2 = ((X[:, None, :] - np.sqrt(a_) * DATA[None]) ** 2).sum(2)
    lw = -d2 / (2 * (1 - a_)); lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    return w @ DATA
show_t = [0, 100, 300, 1000]
fig, axs = plt.subplots(2, 4, figsize=(10.4, 5.0), dpi=220)
E = r.standard_normal((1500, 2)); X0 = heart(1500, r)
for k, t in enumerate(show_t):
    P = np.sqrt(ab(t)) * X0 + np.sqrt(1 - ab(t)) * E
    axs[0, k].scatter(P[:, 0], P[:, 1], s=2.2, color=BLU, alpha=.7); axs[0, k].set_title(f't = {t}', fontsize=11, color=INK)
X = r.standard_normal((1500, 2)); snaps = {1000: X.copy()}
for t in range(T, 0, -1):
    xh = x0h2(X, t); e = (X - np.sqrt(ab(t)) * xh) / np.sqrt(1 - ab(t))
    mean = (X - beta[t - 1] / np.sqrt(1 - ab(t)) * e) / np.sqrt(alpha[t - 1])
    sig = np.sqrt(beta[t - 1] * (1 - ab(t - 1)) / (1 - ab(t)))
    X = mean + (sig * r.standard_normal(X.shape) if t > 1 else 0)
    if t - 1 in (300, 100, 0): snaps[t - 1] = X.copy()
for k, t in enumerate([1000, 300, 100, 0]):
    P = snaps[t]; axs[1, k].scatter(P[:, 0], P[:, 1], s=2.2, color=ORG, alpha=.7); axs[1, k].set_title(f't = {t}', fontsize=11, color=INK)
for a_ in axs.flat:
    a_.set_xlim(-3.2, 3.2); a_.set_ylim(-3.2, 3.2); a_.set_aspect('equal'); a_.set_xticks([]); a_.set_yticks([])
    for s in a_.spines.values(): s.set_color('#D5D8DE')
fig.tight_layout(h_pad=1.6); fig.savefig(os.path.join(OUT, 'toy2d.png')); plt.close()

# ---- ink drop in water (analogy), 2D heat equation solved by Gaussian blur
from scipy.ndimage import gaussian_filter
H = 200; yy, xx = np.mgrid[0:H, 0:H]
c0 = np.exp(-(((xx - 100) ** 2 + (yy - 70) ** 2) / (2 * 6 ** 2)))
mask = ((xx - 100) / 80) ** 2 + ((yy - 100) / 90) ** 2 <= 1
for k, s in enumerate([0, 12, 28, 80]):
    c = gaussian_filter(c0, s, mode='constant') if s else c0
    c = c / c[mask].max() if s < 80 else np.full_like(c, 0.25)
    if s == 80: c = 0.18 + 0 * c
    col = np.ones((H, H, 3))
    water = np.array([0.90, 0.95, 0.99]); ink = np.array([0.10, 0.17, 0.45])
    v = np.clip(c, 0, 1)[..., None] if s < 80 else np.full((H, H, 1), 0.22)
    img = water * (1 - v) + ink * v
    col[mask] = img[mask]
    edge = mask & ~(((xx - 100) / 78) ** 2 + ((yy - 100) / 88) ** 2 <= 1)
    col[edge] = [0.55, 0.6, 0.68]
    Image.fromarray((col * 255).astype(np.uint8)).resize((300, 300), Image.LANCZOS).save(os.path.join(OUT, f'ink_{k}.png'))
print('ok')
