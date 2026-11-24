"""Picture equations (exact): image + added noise = noisier image; noisy image - removed part = clearer image.
The '+' / '-' tiles are the exact differences between consecutive images, so each equation holds pixel by pixel."""
import numpy as np, sys, os
from PIL import Image
MEDIA, OUT = sys.argv[1:3]
T = 1000; beta = np.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = np.cumprod(alpha)
ab = lambda t: 1.0 if t == 0 else abar[t - 1]
def load(f, s=128): return np.asarray(Image.open(os.path.join(MEDIA, f)).convert('RGB').resize((s, s), Image.LANCZOS), np.float32) / 127.5 - 1
def save(x, name, s=360):
    Image.fromarray((np.clip((x + 1) / 2, 0, 1) * 255).astype(np.uint8)).resize((s, s), Image.LANCZOS).save(os.path.join(OUT, name))
# forward chain on the cow, one fixed noise sample per step (Markov chain, exact)
r = np.random.default_rng(3); x = load('image95.jpg'); xs = {0: x}; prev = 0
for t in (150, 400, 1000):
    a = ab(t) / ab(prev); x = np.sqrt(a) * x + np.sqrt(1 - a) * r.standard_normal(x.shape).astype(np.float32); xs[t] = x; prev = t
ts = [0, 150, 400, 1000]
for i, t in enumerate(ts):
    save(xs[t], f'd_fwd_{t}.png')
    if i: save(xs[t] - xs[ts[i - 1]], f'd_add_{t}.png')
# reverse (cow trajectory, seed 5, ideal denoiser over 12 images)
SET = ['image95.jpg', 'image75.jpeg', 'image12.jpeg', 'image19.jpeg', 'image100.jpg', 'image85.jpeg',
       'image88.jpeg', 'image83.jpeg', 'image104.jpg', 'image97.jpg', 'image74.jpeg', 'image80.jpeg']
D = np.stack([load(f) for f in SET]).reshape(len(SET), -1)
def x0_hat(x, t):
    a_ = ab(t); d2 = ((x[None] - np.sqrt(a_) * D) ** 2).sum(1)
    lw = -d2 / (2 * (1 - a_)); w = np.exp(lw - lw.max()); w /= w.sum(); return w @ D
R = [1000, 350, 150, 0]
rr = np.random.default_rng(5); x = rr.standard_normal(D.shape[1]); snap = {1000: x.copy()}
for t in range(T, 0, -1):
    xh = x0_hat(x, t); e = (x - np.sqrt(ab(t)) * xh) / np.sqrt(1 - ab(t))
    mean = (x - beta[t - 1] / np.sqrt(1 - ab(t)) * e) / np.sqrt(alpha[t - 1])
    sig = np.sqrt(beta[t - 1] * (1 - ab(t - 1)) / (1 - ab(t)))
    x = mean + (sig * rr.standard_normal(x.shape) if t > 1 else 0)
    if t - 1 in R: snap[t - 1] = x.copy()
assert int(((D - snap[0]) ** 2).sum(1).argmin()) == 0
for i, t in enumerate(R):
    save(snap[t].reshape(128, 128, 3), f'd_rev_{t}.png')
    if i: save((snap[R[i - 1]] - snap[t]).reshape(128, 128, 3), f'd_sub_{t}.png')
print('ok')
# training example, exact: noisy - (scaled noise) = (slightly faded) original, t = 100
rng = np.random.default_rng(11)
x0 = D[1].reshape(128, 128, 3); e = rng.standard_normal(x0.shape).astype(np.float32); t = 100
save(np.sqrt(ab(t)) * x0 + np.sqrt(1 - ab(t)) * e, 'd_tr_noisy.png'); save(np.sqrt(1 - ab(t)) * e, 'd_tr_noise.png'); save(np.sqrt(ab(t)) * x0, 'd_tr_clean.png')
