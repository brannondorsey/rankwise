"""Validation on real CCD frames: rankwise against astropy, ccdproc and Siril.

    taskset -c 1 python bench/ccd_real.py [workdir]

Data: the raw calibration frames of Matt Craig's CCD reduction guide, "Sample raw images from
thermo-electrically cooled CCD" (Zenodo 3245296, CC BY 4.0): 10 bias, 10 dark (90 s) and 10
twilight flat frames of 4109×4096 uint16 from an Apogee Aspen CG16M. Downloaded on first run
into bench/data/ccd (480 MB).

For each frame type:
* astropy: `sigma_clip(..., axis=0)` with the median center and `std` or `mad_std` spread,
  against `rankwise.sigma_clip_mean(..., mask=...)`: the rejection masks must be identical.
* ccdproc: `Combiner.sigma_clipping` (one pass, median center, mad_std spread, as the guide
  combines calibration frames) against rankwise's `clip_combine(method="mad", maxiters=1)`.
* Siril 1.2.1 (if `siril-cli` is installed): the sigma, median-sigma, winsorized and
  percentile stacks with -rejmaps against rankwise's `preset="siril"`.
Flats are scaled by the inverse of their median first, as the guide does. Writes
results/ccd_real.json.
"""

import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import common  # noqa: F401  (one-thread env first)
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip

import rankwise as rw
from common import RESULTS, machine, save, versions

ROOT = RESULTS.parents[1]
DATA = ROOT / "bench" / "data" / "ccd"
URL = "https://zenodo.org/api/records/3245296/files/example-thermo-electric.tar.bz2/content"
SRC = DATA / "example-thermo-electric"
KINDS = {"bias": "Bias-*.fit", "dark": "Dark-*.fit", "flat": "AutoFlat-*.fit"}


def fetch():
    if SRC.exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    tb = DATA / "example-thermo-electric.tar.bz2"
    if not tb.exists():
        urllib.request.urlretrieve(URL, tb)
    with tarfile.open(tb) as t:
        t.extractall(DATA, filter="data")


def load(kind):
    paths = sorted(SRC.glob(KINDS[kind]))
    x = np.stack([fits.getdata(p).astype(np.float32) for p in paths])  # uint16 + BZERO -> exact in float32
    if kind == "flat":
        x *= (1 / np.median(x, axis=(1, 2)))[:, None, None].astype(np.float32)
    return x


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def check_astropy(x, sigma, stdfunc):
    mask = np.empty(x.shape, bool)
    out, t_rw = timed(lambda: rw.sigma_clip_mean(x, axis=0, sigma=sigma, maxiters=5, stdfunc=stdfunc, mask=mask))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc, t_ap = timed(lambda: sigma_clip(x, sigma=sigma, maxiters=5, cenfunc="median", stdfunc=stdfunc, axis=0))
    same = (mask == sc.mask).all(axis=0)
    kept = np.where(sc.mask, 0.0, x.astype(np.float64))
    ref = (kept.sum(0) / (~sc.mask).sum(0)).astype(np.float32)
    ulp = np.abs(out.astype(np.float64) - ref) / np.spacing(np.abs(ref))
    return dict(reference="astropy sigma_clip", sigma=sigma, stdfunc=stdfunc, maxiters=5, pixels=int(same.size),
                masks_identical_pixels=int(same.sum()), rejected_values=int(sc.mask.sum()),
                max_mean_diff_ulp=float(np.nanmax(ulp)), rankwise_seconds=t_rw, reference_seconds=t_ap)


def check_ccdproc(x, sigma):
    import astropy.units as u
    from ccdproc import CCDData, Combiner

    x64 = x.astype(np.float64)
    mask = np.empty(x.shape, bool)
    out, t_rw = timed(lambda: rw.clip_combine(x64, "mad", sigma=sigma, maxiters=1, mask=mask))

    def ref():
        c = Combiner([CCDData(f, unit=u.adu) for f in x64])
        c.sigma_clipping(low_thresh=sigma, high_thresh=sigma, func=np.ma.median, dev_func=mad_std)
        return c.data_arr.mask.copy(), c.average_combine().data

    (cmask, cmean), t_cc = timed(ref)
    same = (mask == cmask).all(axis=0)
    rel = np.abs(out - cmean) / np.abs(cmean)
    return dict(reference="ccdproc Combiner.sigma_clipping + average_combine", sigma=sigma, stdfunc="mad_std",
                maxiters=1, pixels=int(same.size), masks_identical_pixels=int(same.sum()),
                rejected_values=int(cmask.sum()), max_rel_mean_diff=float(np.nanmax(rel)),
                rankwise_seconds=t_rw, reference_seconds=t_cc)


SIRIL = {"sigma": ("s", (3.0, 3.0)), "median_sigma": ("m", (3.0, 3.0)), "winsorized": ("w", (3.0, 3.0)),
         "percentile": ("p", (0.2, 0.1))}


def check_siril(x, kind, work):
    x = x / np.float32(65535) if kind != "flat" else x / np.float32(4)  # Siril's float convention, within 0..1
    lights = work / kind / "lights"
    lights.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(x):
        fits.writeto(lights / f"light_{k + 1:05d}.fit", f, overwrite=True)
    lines = ["requires 1.2.0", "setcpu 1", "cd lights", "convert light -out=../seq", "cd ../seq"]
    for name, (letter, (lo, hi)) in SIRIL.items():
        lines.append(f"stack light_ rej {letter} {lo} {hi} -nonorm -rejmaps -out={name}")
    script = work / kind / "run.ssf"
    script.write_text("\n".join(lines) + "\n")
    subprocess.run(["siril-cli", "-d", str(work / kind), "-s", str(script)], check=True, capture_output=True)
    n = x.shape[0]
    rows = []
    for name, (_, sig) in SIRIL.items():
        rd = lambda s: np.asarray(fits.getdata(work / kind / "seq" / f"{name}_{s}_rejmap.fit"), np.float64)  # noqa: E731
        sl, sh = np.round(rd("low") * n).astype(int), np.round(rd("high") * n).astype(int)
        val = np.asarray(fits.getdata(work / kind / "seq" / f"{name}.fit"), np.float64)
        out, lo, hi = rw.clip_combine(x, name, preset="siril", sigma=sig, return_counts=True)
        same = (lo == sl) & (hi == sh)
        rel = np.abs(out - val) / np.maximum(np.abs(val), 1e-30)
        rows.append(dict(reference="siril-cli 1.2.1", method=name, sigma=sig, pixels=int(same.size),
                         counts_identical_pixels=int(same.sum()), rejected_by_siril=int((sl + sh).sum()),
                         rejected_by_rankwise=int((lo + hi).sum()),
                         max_rel_value_diff_where_counts_agree=float(rel[same].max())))
    return rows


def main():
    fetch()
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rw-ccd-"))
    report = dict(machine=machine(), source="Zenodo 3245296 (M. Craig, CC BY 4.0), Apogee Aspen CG16M, 4109x4096",
                  versions=versions("rankwise", "numpy", "astropy", "ccdproc"), kinds={})
    for kind in KINDS:
        x = load(kind)
        rows = [check_astropy(x, 3.0, "std"), check_astropy(x, 5.0, "mad_std"), check_ccdproc(x, 5.0)]
        if shutil.which("siril-cli"):
            rows += check_siril(x, kind, work)
        report["kinds"][kind] = dict(frames=int(x.shape[0]), shape=list(x.shape[1:]), rows=rows)
        for r in rows:
            print(kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        del x
    save("ccd_real", report)


if __name__ == "__main__":
    main()
