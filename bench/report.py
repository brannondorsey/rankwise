"""Render docs/benchmarks.md from bench/results/*.json.

    python bench/report.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt_s(x):
    return f"{x:.3g}"


def ratio_col(rows, key="seconds"):
    best = min(r[key] for r in rows if r["tool"].startswith("rankwise") and ("AVX-512" in r["tool"] or "NEON" in r["tool"]))
    return best


def picture(name, alt):
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="img/{name}_dark.png">'
            f'<img alt="{alt}" src="img/{name}_light.png"></picture>')


def main():
    out = ["# Benchmarks", "",
           "Every number here comes from the scripts in [`bench/`](../bench) and the JSON they wrote to "
           "[`bench/results/`](../bench/results). Regenerate this page with `python bench/report.py` and the charts "
           "with `python bench/figures.py`.", ""]
    astro = load("astro_31x2048")
    m = astro["machine"] if astro else {}
    out += ["## Method", "",
            f"* **Machine:** {m.get('cpu', '?')}, {m.get('logical_cpus', '?')} vCPUs (cloud VM, KVM), ISAs available: "
            f"{', '.join(m.get('isas', []))}.",
            "* **One core for every tool.** Scripts run under `taskset -c 1`, and thread pools are pinned with "
            "`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS` and friends set to 1 before any import. "
            "rankwise runs with `threads=1`.",
            "* **Same input for everyone.** Each tool gets the same array, already in RAM. Siril reads its own FITS "
            "files (from the page cache) and is marked as such.",
            "* **Timing:** best of 3 runs after a warm-up (best of 2, or 1 for tools that take tens of seconds).",
            "* **Every result is checked**, not just timed: medians against `np.median`, quantiles against "
            "`np.quantile`/`np.nanquantile`, sigma clipping against astropy's mask.",
            "* **Like for like.** NaN-aware tools are compared with rankwise's NaN-aware functions; ccdproc's "
            "`Combiner.sigma_clipping` does a single pass and is labeled so.",
            "* The machine is a shared cloud VM; repeated runs vary by about ±15%.", ""]
    if astro:
        out += ["## Astro stack: 31 × 2048 × 2048 float32", "", picture("bench_astro", "Astro stacking benchmark"), "",
                "| Tool | Operation | Seconds | ns / pixel | vs rankwise AVX-512 | Check |", "|---|---|---|---|---|---|"]
        for op in ("sigclip", "median"):
            rows = sorted([r for r in astro["rows"] if r["op"] == op], key=lambda r: r["seconds"])
            best = ratio_col(rows)
            for r in rows:
                if op == "median":
                    chk = "identical to np.median" if r.get("identical_to_np_median") else "differs"
                else:
                    k = r.get("kept_counts_identical_to_astropy")
                    rel = r.get("max_rel_diff_vs_astropy_mean")
                    chk = (f"astropy kept counts identical on {k * 100:.5g}% of pixels" if k is not None
                           else f"mean differs from astropy's by up to {rel * 100:.2g}%")
                opname = "sigma-clipped mean" if op == "sigclip" else "median"
                out.append(f"| {r['tool']} | {opname} | {fmt_s(r['seconds'])} | {r['ns_per_px']:.1f} | "
                           f"{r['seconds'] / best:.1f}× | {chk} |")
        siril = load("siril")
        if siril:
            for r in siril["rows"]:
                out.append(f"| Siril 1.2.1 (reads FITS) | {'sigma-clipped mean' if r['op'] == 'sigclip' else 'median'} | "
                           f"{fmt_s(r['seconds'])} | {r['ns_per_px']:.1f} | — | own I/O |")
        out += ["", "reducers and imcombiners clip differently from astropy (they do not re-apply the final bounds "
                "to values rejected earlier), so their means differ from astropy's on some pixels. ccdproc runs one "
                "pass. The f32-bounds row is rankwise's opt-in float32 center/spread/bounds mode.", ""]
    sw = load("sweep_1024")
    if sw:
        out += ["## Stack depth sweep: 1024 × 1024 float32", "", picture("bench_sweep", "Depth sweep"), "",
                "| Frames | rankwise median | rankwise σ-clip | reducers median | imcombiners median | reducers σ-clip | "
                "imcombiners σ-clip | NumPy median | median speedup | σ-clip speedup |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sw["rows"]:
            ms = min(r["reducers_median"], r["imcombiners_median"]) / r["rankwise_median"]
            ss = min(r["reducers_sigclip"], r["imcombiners_sigclip"]) / r["rankwise_sigclip"]
            out.append(f"| {r['n']} | {r['rankwise_median']:.1f} | {r['rankwise_sigclip']:.1f} | {r['reducers_median']:.0f} | "
                       f"{r['imcombiners_median']:.0f} | {r['reducers_sigclip']:.0f} | {r['imcombiners_sigclip']:.0f} | "
                       f"{r['numpy_median']:.0f} | {ms:.1f}× | {ss:.1f}× |")
        out += ["", "ns per output pixel, one core, AVX-512. Speedup is against the faster of reducers and imcombiners.", ""]
    ens = load("ensemble_31x721x1440")
    if ens:
        out += ["## Weather ensemble: p10 / p50 / p90 of 31 members, 721 × 1440", "",
                picture("bench_ensemble", "Ensemble quantile benchmark"), "",
                "| Tool | Function | Seconds per lead time | ns / cell | Identical to NumPy |", "|---|---|---|---|---|"]
        for kind in ("quantile", "nanquantile"):
            for r in sorted([r for r in ens["rows"] if r["kind"] == kind], key=lambda r: r["seconds"]):
                out.append(f"| {r['tool']} | {kind} | {fmt_s(r['seconds'])} | {r['ns_per_cell']:.1f} | "
                           f"{'yes' if r['identical_to_numpy'] else 'no'} |")
        c = ens.get("comparators", {})
        out += ["", f"Synthetic GEFS-shaped data (heavy-tailed members in kelvin); the nanquantile rows add 2% NaN. "
                f"The p10/p50/p90 network keeps {c.get('p10_p50_p90')} of {c.get('full_sort')} comparators. numbagg and "
                "fastnanquantile interpolate differently from NumPy, so their results differ in the last bits.", ""]
    gefs = load("gefs_conus")
    if gefs:
        out += ["## Real data: NOAA GEFS 2 m temperature", "",
                f"{gefs['source']}; array {' × '.join(map(str, gefs['shape']))} (members × leads × lat × lon).", "",
                "| Tool | Seconds | ns / cell | Identical to np.quantile |", "|---|---|---|---|"]
        for r in sorted(gefs["rows"], key=lambda r: r["seconds"]):
            out.append(f"| {r['tool']} | {fmt_s(r['seconds'])} | {r['ns_per_cell']:.1f} | {'yes' if r['identical_to_numpy'] else 'no'} |")
        out.append("")
    ls = load("landsat_90x2048")
    if ls:
        out += ["## Satellite composite: 90 uint16 scenes, 2048 × 2048, clouds as nodata", "",
                f"{ls['cloud_fraction'] * 100:.0f}% of values are cloud (fill value 0). rankwise reads the uint16 data "
                "with `nodata=0`; the baselines get float32 with NaN, converted once and not timed.", "",
                "| Tool | Seconds | ns / pixel | Identical to np.nanmedian |", "|---|---|---|---|"]
        for r in sorted(ls["rows"], key=lambda r: r["seconds"]):
            out.append(f"| {r['tool']} | {fmt_s(r['seconds'])} | {r['ns_per_px']:.1f} | {'yes' if r['identical'] else 'no'} |")
        out.append("")
    st = load("stride")
    if st:
        out += ["## Power-of-two strides", "",
                "A contiguous stack of 2^k-byte frames puts every frame's current cache line in the same set. "
                "31 frames, rankwise AVX-512, ns per pixel:", "",
                "| Frame | Stride | Median | Sigma clip |", "|---|---|---|---|"]
        for r in st["rows"]:
            pad = "exactly 2^k" if r["pad_bytes"] == 0 and r["frame"] != "721x1440" else f"+{r['pad_bytes']} B"
            if r["frame"] == "721x1440" and r["pad_bytes"] == 0:
                pad = "not 2^k"
            out.append(f"| {r['frame']} | {pad} | {r['median_ns_px']:.1f} | {r['sigclip_ns_px']:.1f} |")
        out.append("")
    (ROOT / "docs" / "benchmarks.md").write_text("\n".join(out) + "\n")
    print("wrote docs/benchmarks.md")


if __name__ == "__main__":
    main()
