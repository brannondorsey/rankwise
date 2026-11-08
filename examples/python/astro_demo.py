"""The visual demo: 40 synthetic subs of an emission nebula, stacked three ways.

    python examples/python/astro_demo.py [outdir]

The target is synthetic, so the true image is known: Hα filaments, an OIII core, dust lanes
and 5,200 Moffat stars; 40 subs of 30 s at 1800×1200 RGB with shot noise and 8 e⁻ read
noise, 160 cosmic rays and 400 hot pixels each, five satellite trails, an airplane and a
meteor. rankwise stacks the subs straight from one (40, H, W, 3) array: a median and an
astropy-identical sigma-clipped mean (3σ, ≤5 iterations), with the rejection map.

Writes hero.jpg, compare.jpg and rejection.jpg (and stats.json) to outdir.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

import rankwise as rw

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)
H, W, N = 1200, 1800, 40
rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def fbm(beta, seed):
    r = np.random.default_rng(seed)
    f = np.fft.rfft2(r.standard_normal((H, W)))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1
    out = np.fft.irfft2(f * k ** (-beta / 2), s=(H, W)).astype(np.float32)
    return (out - out.mean()) / out.std()


# ---- the sky: nebula with filaments, an OIII core and dust lanes
cx, cy = 0.54 * W, 0.50 * H
u = ((xx - cx) * np.cos(0.5) + (yy - cy) * np.sin(0.5)) / (0.50 * W)
v = (-(xx - cx) * np.sin(0.5) + (yy - cy) * np.cos(0.5)) / (0.34 * H)
envelope = np.exp(-(u ** 2 + v ** 2) ** 1.2 * 2.2)
turb, ridges = fbm(3.4, 1), (1 - np.abs(fbm(2.8, 2))) ** 4
ha = envelope * np.clip(0.35 + 0.65 * turb, 0, None) * (0.15 + 2.6 * ridges)
ha += 0.5 * np.exp(-(u ** 2 + v ** 2) * 9) * np.clip(1 + 0.3 * fbm(3.0, 3), 0, None)
oiii = 0.55 * np.exp(-(((xx - 0.57 * W) / (0.10 * W)) ** 2 + ((yy - 0.47 * H) / (0.12 * H)) ** 2)) * np.clip(1 + 0.7 * fbm(3.2, 4), 0, None)
dust = np.clip(fbm(3.6, 5) - 0.35, 0, None)
tau = 2.6 * gaussian_filter(dust, 2) * envelope ** 0.3
neb = np.stack([1.00 * ha + 0.05 * oiii, 0.16 * ha + 0.80 * oiii, 0.24 * ha + 0.85 * oiii], -1) * np.exp(-tau)[..., None]
neb *= 210.0

# ---- stars: power-law fluxes, colour by temperature, Moffat core + halo
ns = 5200
sx, sy = rng.uniform(0, W, ns), rng.uniform(0, H, ns)
flux = np.minimum((rng.pareto(1.35, ns) + 1) * 900, 4e6)
temp = rng.choice([3500, 4500, 5500, 6500, 8000, 11000, 20000], ns, p=[.18, .25, .22, .14, .10, .07, .04])
tint = {3500: (1, .62, .38), 4500: (1, .78, .58), 5500: (1, .92, .82), 6500: (1, 1, 1),
        8000: (.85, .9, 1), 11000: (.72, .82, 1), 20000: (.62, .74, 1)}
r = np.arange(-20, 21, dtype=np.float32)
R2 = r[:, None] ** 2 + r[None, :] ** 2
al = 2.3 / (2 * np.sqrt(2 ** (1 / 2.8) - 1))
psf = (1 + R2 / al ** 2) ** -2.8
psf /= psf.sum()
halo = np.exp(-R2 / (2 * 7.0 ** 2))
halo /= halo.sum()
stars = np.zeros((H, W, 3), np.float32)
for c in range(3):
    pts, big = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    w = flux * np.array([tint[t][c] for t in temp])
    np.add.at(pts, (sy.astype(int), sx.astype(int)), w)
    bright = flux > 2e5
    np.add.at(big, (sy[bright].astype(int), sx[bright].astype(int)), 0.08 * w[bright])
    stars[..., c] = fftconvolve(pts, psf, mode="same") + fftconvolve(big, halo, mode="same")
sky = np.stack([420 + 60 * xx / W, 380 + 45 * xx / W, 330 + 30 * xx / W], -1)
truth = np.minimum((sky + neb + stars).astype(np.float32), 60000)

# ---- 40 subs with real-world defects
st = np.empty((N, H, W, 3), np.float32)
for i in range(N):
    f = truth * rng.uniform(0.96, 1.04)
    f = f + rng.standard_normal(f.shape, dtype=np.float32) * np.sqrt(f + 64.0)
    for _ in range(160):
        y0, x0 = rng.integers(2, H - 4), rng.integers(2, W - 4)
        f[y0:y0 + rng.integers(1, 4), x0:x0 + rng.integers(1, 3), :] += rng.uniform(3e3, 3e4)
    hp = (rng.integers(0, H, 400), rng.integers(0, W, 400))
    f[hp[0], hp[1], rng.integers(0, 3, 400)] += rng.uniform(2e3, 1e4, 400)
    st[i] = f
for i, (x0, y0, ang, amp) in zip([3, 11, 19, 27, 33], [(200, 50, 0.32, 9000), (1500, 0, 2.2, 6000),
                                                       (0, 900, -0.45, 12000), (900, 1200, 1.95, 7000), (100, 300, 0.05, 5000)]):
    d = np.abs((xx - x0) * np.sin(ang) - (yy - y0) * np.cos(ang))
    st[i] += (amp * np.exp(-0.5 * (d / 1.6) ** 2))[..., None]
i, ang, x0, y0 = 23, 0.78, 300, 1150  # airplane: dashed wing lights + strobe
along = (xx - x0) * np.cos(-ang) + (yy - y0) * np.sin(-ang)
d = (xx - x0) * np.sin(-ang) - (yy - y0) * np.cos(-ang)
blink = (np.floor(along / 38) % 2 == 0).astype(np.float32)
for off, col in [(-5, (1, .15, .1)), (5, (.15, 1, .3))]:
    st[i] += (9000 * np.exp(-0.5 * ((d - off) / 1.4) ** 2) * blink)[..., None] * np.array(col, np.float32)
st[i] += (25000 * (np.abs(along % 95) < 3) * np.exp(-0.5 * (d / 2.0) ** 2))[..., None]
i, x0, y0, ang = 31, 1250, 150, 2.5  # meteor
along = (xx - x0) * np.cos(ang) + (yy - y0) * np.sin(ang)
d = (xx - x0) * np.sin(ang) - (yy - y0) * np.cos(ang)
st[i] += (np.clip(along / 520, 0, 1) * (along < 520) * np.exp(-0.5 * (d / 2.2) ** 2) * 30000)[..., None] * np.array([.6, 1, .7], np.float32)

# ---- stack: plain mean (NumPy), median and sigma-clipped mean (rankwise, zero-copy)
t = time.perf_counter()
mean = st.mean(0)
t_mean = time.perf_counter() - t
t = time.perf_counter()
med = rw.median(st, axis=0)
t_med = time.perf_counter() - t
rejmap = np.empty(st.shape, bool)
t = time.perf_counter()
sig, lo, hi = rw.sigma_clip_mean(st, axis=0, sigma=(3, 3), maxiters=5, mask=rejmap, return_counts=True)
t_sig = time.perf_counter() - t
removed = np.where(rejmap, st - sig[None], 0).clip(0).sum(0).max(-1)  # signal removed per pixel


# ---- autostretch (Siril-style midtone transfer, unlinked channels), from the clipped stack
def mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


P = []
for c in range(3):
    ch = sig[..., c]
    m0 = np.median(ch)
    mad = np.median(np.abs(ch - m0)) * 1.4826
    lo_, hi_ = m0 - 2.2 * mad, np.percentile(ch, 99.97)
    P.append((lo_, hi_, mtf(0.17, (m0 - lo_) / (hi_ - lo_))))


def stretch(a, sat=1.6):
    out = np.empty(a.shape, np.float32)
    for c, (lo_, hi_, m) in enumerate(P):
        out[..., c] = mtf(m, np.clip((a[..., c] - lo_) / (hi_ - lo_), 0, 1))
    lum = out.mean(-1, keepdims=True)
    return (np.clip(lum + (out - lum) * sat, 0, 1) * 255 + 0.5).astype(np.uint8)


def noise(a):
    """Robust background noise (green, e⁻ per sub-equivalent) against the known truth."""
    skym = truth[..., 1] < np.percentile(truth[..., 1], 25)
    k = np.median(a[..., 1][skym] / truth[..., 1][skym])
    res = (a[..., 1] - k * truth[..., 1])[skym]
    return float(np.median(np.abs(res - np.median(res))) * 1.4826)


imgs = {"single": st[19], "mean": mean, "median": med, "sig": sig}
nz = {k: noise(v) for k, v in imgs.items()}
rgb = {k: stretch(v) for k, v in imgs.items()}

# hero
Image.fromarray(rgb["sig"]).save(OUT / "hero.jpg", quality=90, optimize=True, progressive=True)

# 2×2 comparison on a crop where trails, the airplane and the meteor cross the nebula
font_b = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_r = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
y0, x0, S, g = 150, 790, 560, 12
labels = [("single", "1 sub, 30 s", f"noise {nz['single']:.1f} e⁻ · satellite in frame"),
          ("mean", "Mean of 40", f"noise {nz['mean']:.1f} e⁻ · every trail survives"),
          ("median", "Median of 40", f"noise {nz['median']:.1f} e⁻ · clean, {nz['median'] / nz['sig'] * 100 - 100:.0f}% noisier"),
          ("sig", "Sigma-clipped mean of 40", f"noise {nz['sig']:.1f} e⁻ · clean and smooth")]
canvas = Image.new("RGB", (2 * S + 3 * g, 2 * S + 3 * g), (14, 14, 16))
dr = ImageDraw.Draw(canvas)
fb, fr = ImageFont.truetype(font_b, 26), ImageFont.truetype(font_r, 19)
for j, (k, title, sub) in enumerate(labels):
    px, py = g + (j % 2) * (S + g), g + (j // 2) * (S + g)
    canvas.paste(Image.fromarray(rgb[k][y0:y0 + S, x0:x0 + S]), (px, py))
    dr.rectangle([px, py, px + S, py + 74], fill=(0, 0, 0))
    dr.text((px + 16, py + 10), title, font=fb, fill=(255, 255, 255))
    dr.text((px + 16, py + 44), sub, font=fr, fill=(200, 200, 205))
    if k == "sig":
        dr.rectangle([px - 3, py - 3, px + S + 2, py + S + 2], outline=(57, 135, 229), width=3)
canvas.save(OUT / "compare.jpg", quality=90, optimize=True, progressive=True)

# rejection map: signal removed per pixel, log scale, over a dimmed stack
base = rgb["sig"].mean(-1) * 0.30
lv = np.clip(np.log10(np.maximum(removed, 1)) / np.log10(40000), 0, 1)
lv = np.where(removed < 400, 0, lv)
heat = np.stack([lv * 255, lv * 160, lv * 40], -1)
im = Image.fromarray(np.clip(base[..., None] + heat, 0, 255).astype(np.uint8))
dr = ImageDraw.Draw(im)
dr.rectangle([24, 1080, 600, 1176], fill=(0, 0, 0))
dr.text((40, 1090), "Signal removed by rejection, per pixel", font=ImageFont.truetype(font_b, 24), fill=(255, 255, 255))
for j, (val, lab) in enumerate([(0, "< 400 e⁻"), (2000, "2k"), (10000, "10k"), (40000, "40k+")]):
    l_ = 0 if val == 0 else np.log10(val) / np.log10(40000)
    c = tuple(int(x) for x in np.array([255, 160, 40]) * l_)
    dr.rectangle([40 + j * 135, 1132, 72 + j * 135, 1160], fill=c, outline=(110, 110, 110))
    dr.text((80 + j * 135, 1134), lab, font=ImageFont.truetype(font_r, 20), fill=(230, 230, 230))
im.save(OUT / "rejection.jpg", quality=90, optimize=True, progressive=True)

stats = dict(noise_e=nz, frames=N, shape=[H, W, 3], seconds=dict(numpy_mean=t_mean, rankwise_median=t_med,
             rankwise_sigma_clip=t_sig), rejected_fraction=float(rejmap.mean()),
             mean_rejected_per_pixel=float((lo + hi).mean()))
(OUT / "astro_demo.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
