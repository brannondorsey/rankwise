"""Black-box check of rankwise's Siril presets against siril-cli.

For several stack depths, writes synthetic float32 FITS frames, stacks them in Siril with
each rejection type and -rejmaps, and compares Siril's per-pixel low/high reject counts and
stacked values with rankwise. Writes presets.json next to this file.

    python bench/siril/compare_presets.py [workdir]
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

import rankwise as rw

HERE = Path(__file__).resolve().parent
METHODS = {  # siril letter, rankwise method, kwargs
    "sigma": ("s", "sigma", dict(sigma=(3.0, 3.0))),
    "median_sigma": ("m", "median_sigma", dict(sigma=(3.0, 3.0))),
    "winsorized": ("w", "winsorized", dict(sigma=(3.0, 3.0))),
    "percentile": ("p", "percentile", dict(sigma=(0.2, 0.1))),
}


def make(n, h, w, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(1000, 30, (n, h, w))
    hits = rng.random(x.shape) < 0.05
    x[hits] += rng.uniform(200, 5000, hits.sum())
    return (x / 65535).astype(np.float32)  # Siril's float convention, 0..1


def run_siril(work: Path, x):
    lights = work / "lights"
    lights.mkdir(parents=True)
    for k, f in enumerate(x):
        fits.writeto(lights / f"light_{k + 1:05d}.fit", f, overwrite=True)
    lines = ["requires 1.2.0", "setcpu 1", "cd lights", "convert light -out=../seq", "cd ../seq"]
    for name, (letter, _, kw) in METHODS.items():
        lo, hi = kw["sigma"]
        lines.append(f"stack light_ rej {letter} {lo} {hi} -nonorm -rejmaps -out={name}")
    (work / "run.ssf").write_text("\n".join(lines) + "\n")
    subprocess.run(["siril-cli", "-d", str(work), "-s", str(work / "run.ssf")], check=True, capture_output=True)


def main():
    if shutil.which("siril-cli") is None:
        sys.exit("siril-cli not found")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rw-siril-"))
    report = {"siril": subprocess.run(["siril-cli", "--version"], capture_output=True, text=True).stdout.strip(),
              "rankwise": rw.__version__, "results": []}
    for n in (5, 8, 15, 31):
        x = make(n, 48, 64, seed=n)
        work = root / f"n{n}"
        run_siril(work, x)
        for name, (_, method, kw) in METHODS.items():
            rd = lambda s: np.asarray(fits.getdata(work / "seq" / f"{name}_{s}_rejmap.fit"), np.float64)
            sl = np.round(rd("low") * n).astype(int)
            sh = np.round(rd("high") * n).astype(int)
            val = np.asarray(fits.getdata(work / "seq" / f"{name}.fit"), np.float64)
            out, lo, hi = rw.clip_combine(x, method, preset="siril", return_counts=True, **kw)
            same = (lo == sl) & (hi == sh)
            rel = np.abs(out - val) / np.abs(val)
            row = dict(n=n, method=name, pixels=int(same.size), identical_counts=int(same.sum()),
                       max_rel_value_diff_where_counts_agree=float(rel[same].max()))
            report["results"].append(row)
            print(f"n={n:3d} {name:13s} counts identical on {same.sum()}/{same.size} pixels, "
                  f"value rel diff <= {row['max_rel_value_diff_where_counts_agree']:.1e}")
    (HERE / "presets.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
