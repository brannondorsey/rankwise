"""Time Siril's median and sigma-clipping stacks on the astro benchmark data, one thread.

    taskset -c 1 python bench/siril/time_siril.py [workdir]

Siril reads its own sequence from disk (page cache after a warm-up run), so its times
include file reading; the in-RAM tools in bench/astro.py do not. Writes results/siril.json.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from astropy.io import fits  # noqa: E402

from common import astro_stack, machine, save  # noqa: E402


def siril(work, lines):
    script = work / "s.ssf"
    script.write_text("\n".join(["requires 1.2.0", "setcpu 1"] + lines) + "\n")
    t = time.perf_counter()
    subprocess.run(["siril-cli", "-d", str(work), "-s", str(script)], check=True, capture_output=True)
    return time.perf_counter() - t


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rw-siril-time-"))
    (work / "lights").mkdir(parents=True, exist_ok=True)
    st = astro_stack(31, 2048, 2048, seed=1) / 65535.0
    for k, f in enumerate(st):
        fits.writeto(work / "lights" / f"light_{k + 1:05d}.fit", f.astype("float32"), overwrite=True)
    siril(work, ["cd lights", "convert light -out=../seq"])
    base = min(siril(work, ["cd seq"]) for _ in range(3))  # startup cost
    rows = []
    for op, cmd in [("median", "stack light_ med -nonorm -out=med"), ("sigclip", "stack light_ rej s 3 3 -nonorm -out=sig")]:
        siril(work, ["cd seq", cmd])  # warm-up: page cache
        t = min(siril(work, ["cd seq", cmd]) for _ in range(3)) - base
        rows.append(dict(tool="Siril 1.2.1", op=op, seconds=t, ns_per_px=t / 2048 ** 2 * 1e9,
                         note="includes reading 31 FITS files from the page cache"))
        print(rows[-1], flush=True)
    save("siril", dict(machine=machine(), startup_seconds_subtracted=base, rows=rows))


if __name__ == "__main__":
    main()
