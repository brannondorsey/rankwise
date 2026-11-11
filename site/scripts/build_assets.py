"""Build the site's data assets from the repository's datasets and benchmark results.

    python site/scripts/build_assets.py [astro] [landsat] [gefs] [bench]   (default: all)

Inputs (see site/README.md):
* astro:   the synthetic subs of examples/python/astro_demo.py (regenerated, seed 42)
* landsat: bench/data/landsat_046027_site512.npz   (site/scripts/fetch_landsat.py)
* gefs:    bench/data/gefs_t2m_conus.npz           (site/scripts/fetch_gefs.py, same cache as bench/gefs.py)
* bench:   bench/results/*.json

Every order statistic here is computed by rankwise. Outputs go to site/public/data.
"""

import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import rankwise as rw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench" / "data"
RES = ROOT / "bench" / "results"
OUT = ROOT / "site" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
WHAT = set(sys.argv[1:]) or {"astro", "landsat", "gefs", "bench"}


def atlas(tiles, cols):
    n, h, w = tiles.shape[:3]
    rows = -(-n // cols)
    a = np.zeros((rows * h, cols * w) + tiles.shape[3:], tiles.dtype)
    for i, t in enumerate(tiles):
        a[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    return a


def kb(p):
    return f"{p.name}: {p.stat().st_size / 1024:.0f} KB"


# ------------------------------------------------------------------------------------ astro
if "astro" in WHAT:
    tmp = Path(tempfile.mkdtemp())
    argv = sys.argv
    sys.argv = ["astro_demo.py", str(tmp)]
    g = runpy.run_path(str(ROOT / "examples" / "python" / "astro_demo.py"), run_name="astro")
    sys.argv = argv
    st, sig, P, stretch = g["st"], g["sig"], g["P"], g["stretch"]
    N, H, W = st.shape[:3]

    # Crop at native resolution where five trails, the airplane and the meteor cross the nebula.
    # Linear electrons are stored as 8-bit codes: linear up to X1 (7 e- per code around the sky),
    # logarithmic above it up to MAX. The page inverts this before stacking.
    X1, C1, MAX = 1200.0, 170.0, 60000.0
    Y0, X0, S = 140, 820, 384
    crop = st[:, Y0:Y0 + S, X0:X0 + S]
    x = np.clip(crop, 0, MAX)
    lin = np.where(x < X1, x / X1 * C1, C1 + (255 - C1) * np.log(np.maximum(x, X1) / X1) / np.log(MAX / X1))
    codes = np.clip(np.round(lin), 0, 255).astype(np.uint8)
    p = OUT / "astro_subs.jpg"
    Image.fromarray(atlas(codes, 8)).save(p, quality=92, subsampling=0, optimize=True)
    print(kb(p))

    # Hero: every sub at 1/4 size (450x300), display-stretched, plus the rankwise stack.
    def down(a, f=4):
        return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f, -1).mean((1, 3))

    hero = np.stack([stretch(down(s)) for s in st])
    p = OUT / "hero_subs.jpg"
    Image.fromarray(atlas(hero, 8)).save(p, quality=78, optimize=True, progressive=True)
    print(kb(p))
    p = OUT / "hero_stack.jpg"
    Image.fromarray(stretch(down(sig, 2))).save(p, quality=88, optimize=True, progressive=True)
    print(kb(p))

    # default probe: the crop pixel where rejection removed the most signal from 2-4 frames
    rej = g["rejmap"][:, Y0:Y0 + S, X0:X0 + S, 1].sum(0)
    removed = np.where((rej >= 2) & (rej <= 4), g["removed"][Y0:Y0 + S, X0:X0 + S], 0)
    py, px = np.unravel_index(int(np.argmax(removed)), removed.shape)

    # lanes demo: 16 neighbouring pixels x 9 frames, chosen so trails cross several lanes
    fr = [0, 3, 6, 11, 14, 19, 23, 27, 33]
    med = np.median(crop[..., 1], axis=0)
    hot = (crop[fr, ..., 1] - med[None]) > 1500  # (9, S, S)
    best, at = -1, (0, 0)
    for y in range(4, S - 4, 2):
        for x in range(0, S - 16, 2):
            h = hot[:, y, x:x + 16]
            per_lane = h.sum(0)
            if per_lane.max() > 3:
                continue
            score = int(h.sum()) + 3 * len(set(np.nonzero(h)[0]))
            if score > best:
                best, at = score, (y, x)
    ly, lx = at
    lanes = codes[fr, ly, lx:lx + 16, 1].astype(int).tolist()

    meta = dict(frames=int(N), tile=S, cols=8, crop=[Y0, X0], enc=[X1, C1, MAX], probe=[int(px), int(py)],
                lanes=dict(frames=fr, y=int(ly), x=int(lx), codes=lanes),
                stretch=[[float(lo), float(hi), float(m)] for lo, hi, m in P], sat=1.6,
                hero=dict(w=W // 4, h=H // 4, cols=8),
                noise=g["nz"], seconds=g["t_sig"])
    (OUT / "astro.json").write_text(json.dumps(meta, indent=1))

# ---------------------------------------------------------------------------------- landsat
if "landsat" in WHAT:
    z = np.load(DATA / "landsat_046027_site512.npz")
    rgb, bad, dates = z["rgb"], z["bad"], z["dates"]  # (90, 3, 512, 512) uint16, (90, 512, 512)
    n = rgb.shape[0]
    T = 384

    def display(r):
        v = np.clip((r.astype(np.float32) * 2.75e-5 - 0.2) / 0.28, 0, 1) ** (1 / 1.9)
        return (v * 255 + 0.5).astype(np.uint8)

    tiles, masks = [], []
    for i in range(n):
        tiles.append(np.asarray(Image.fromarray(display(rgb[i].transpose(1, 2, 0))).resize((T, T), Image.LANCZOS)))
        m = np.asarray(Image.fromarray((bad[i] * 255).astype(np.uint8)).resize((T, T), Image.BILINEAR)) > 0
        masks.append(np.where(m, 255, 0).astype(np.uint8))
    p = OUT / "landsat_scenes.webp"
    Image.fromarray(atlas(np.stack(tiles), 10)).save(p, quality=70, method=6)
    print(kb(p))
    p = OUT / "landsat_masks.png"
    Image.fromarray(atlas(np.stack(masks), 10)).convert("1").save(p, optimize=True)
    print(kb(p))
    meta = dict(scenes=int(n), tile=T, cols=10, dates=[str(d) for d in dates],
                cloud=[round(float(b.mean()), 3) for b in bad], cloud_all=float(bad.mean()))
    (OUT / "landsat.json").write_text(json.dumps(meta))

# ------------------------------------------------------------------------------------- gefs
if "gefs" in WHAT:
    z = np.load(DATA / "gefs_t2m_conus.npz")
    t2m, lat, lon, lead_h = z["t2m"], z["lat"], z["lon"], z["lead_h"]
    q = rw.quantile(t2m, [0.1, 0.5, 0.9], axis=0)  # (3, 81, 121, 257), identical to np.quantile
    keep = np.arange(0, len(lead_h), 2)  # every 6 h
    TMIN, TMAX = -20.0, 40.0
    enc = np.clip(np.round((q[:, keep] - TMIN) / (TMAX - TMIN) * 255), 0, 255).astype(np.uint8)
    frames = enc.transpose(1, 2, 3, 0)  # (leads, lat, lon, 3): R=p10 G=p50 B=p90
    p = OUT / "gefs.jpg"  # 4:4:4 so p10/p50/p90 never bleed into each other
    Image.fromarray(frames.reshape(-1, frames.shape[2], 3)).save(p, quality=92, subsampling=0, optimize=True)
    print(kb(p))
    city = (39.74, -104.99)
    iy, ix = int(np.abs(lat - city[0]).argmin()), int(np.abs(lon - city[1]).argmin())
    fan = rw.quantile(t2m[:, :, iy, ix], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
    spread = q[2] - q[0]
    meta = dict(init="2026-11-01T00Z", lat=[float(lat[0]), float(lat[-1])], lon=[float(lon[0]), float(lon[-1])],
                shape=[int(len(keep)), int(len(lat)), int(len(lon))], lead_h=[float(lead_h[k]) for k in keep],
                tmin=TMIN, tmax=TMAX, spread_max=float(np.nanmax(spread)),
                denver=dict(lead_h=[float(x) for x in lead_h], members=np.round(t2m[:, :, iy, ix], 1).tolist(),
                            fan=np.round(fan, 2).tolist()))
    (OUT / "gefs.json").write_text(json.dumps(meta))

# ------------------------------------------------------------------------------------ bench
if "bench" in WHAT:
    def load(name):
        return json.loads((RES / name).read_text())

    astro, ens, gefs, land = load("astro_31x2048.json"), load("ensemble_31x721x1440.json"), load("gefs_conus.json"), load("landsat_90x2048.json")
    sweep, ccd, siril, e2e = load("sweep_1024.json"), load("ccd_real.json"), load("siril.json"), load("e2e_astro.json")
    pick = lambda rows, **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]  # noqa: E731
    out = dict(
        machine=astro["machine"]["cpu"],
        median=[dict(tool=r["tool"], s=r["seconds"]) for r in pick(astro["rows"], op="median")],
        sigclip=[dict(tool=r["tool"].replace(" (1 pass)", " · 1 pass"), s=r["seconds"]) for r in pick(astro["rows"], op="sigclip") if "f32" not in r["tool"]],
        gefs=[dict(tool=r["tool"], s=r["seconds"]) for r in gefs["rows"]],
        landsat=[dict(tool=r["tool"], s=r["seconds"]) for r in land["rows"]],
        ensemble=[dict(tool=r["tool"], kind=r["kind"], s=r["seconds"]) for r in ens["rows"]],
        siril=[dict(tool="Siril 1.2.1 · own I/O", op=r["op"], s=r["seconds"]) for r in siril["rows"]],
        sweep=sweep["rows"], e2e=e2e["rows"],
        ccd={k: [dict(ref=r["reference"], pixels=r["pixels"], same=r["masks_identical_pixels"]) for r in v["rows"]
                 if r.get("masks_identical_pixels") is not None] for k, v in ccd["kinds"].items()},
    )
    (ROOT / "site" / "src" / "bench.json").write_text(json.dumps(out, indent=1))
    print("bench.json")
