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
            "* **Timing:** best of N runs after a warm-up: 7 for rankwise's sub-second runs, which this VM's noise "
            "affects most, 3 for most tools, 2 or 1 for tools that take tens of seconds.",
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
        src = ls.get("source")
        out += ["## Real data: Landsat composite, 90 uint16 scenes, 2048 × 2048, clouds as nodata", "",
                '<img alt="Landsat 8/9 over Seattle and Tacoma: one cloudy scene, the median with clouds, and the '
                'median with clouds as nodata" src="img/landsat.jpg">', "",
                (f"{src}. " if src else "") +
                f"{ls['cloud_fraction'] * 100:.0f}% of the red-band values are flagged by QA_PIXEL as cloud, cirrus or "
                "shadow and set to the fill value 0. rankwise reads the uint16 data with `nodata=0`; the baselines "
                "get float32 with NaN, converted once and not timed.", "",
                "| Tool | Seconds | ns / pixel | Identical to np.nanmedian |", "|---|---|---|---|"]
        for r in sorted(ls["rows"], key=lambda r: r["seconds"]):
            out.append(f"| {r['tool']} | {fmt_s(r['seconds'])} | {r['ns_per_px']:.1f} | {'yes' if r['identical'] else 'no'} |")
        out.append("")
    e2e = load("e2e_astro")
    if e2e:
        out += ["## End to end: read 31 FITS files, then stack", "",
                "The same 31 float32 FITS files of 2048 × 2048 (the astro set), from the page cache. The Python "
                f"pipelines read them with {e2e['reader']} into one array, then reduce it; Siril reads them itself. "
                "Seconds:", "",
                "| Tool | Operation | Read | Reduce | Total | vs rankwise |", "|---|---|---|---|---|---|"]
        for op in ("median", "sigclip"):
            rows = sorted([r for r in e2e["rows"] if r["op"] == op], key=lambda r: r["total_seconds"])
            best = min(r["total_seconds"] for r in rows if r["tool"].startswith("rankwise"))
            for r in rows:
                rd_ = r["read_seconds"]
                red = "—" if rd_ is None else fmt_s(r["total_seconds"] - rd_)
                out.append(f"| {r['tool']} | {'sigma-clipped mean' if op == 'sigclip' else 'median'} | "
                           f"{'own I/O' if rd_ is None else fmt_s(rd_)} | {red} | {fmt_s(r['total_seconds'])} | "
                           f"{r['total_seconds'] / best:.1f}× |")
        out += ["", "Reading costs more than a rankwise median here, so the end-to-end gain is smaller than the "
                "kernel's; with sigma clipping the kernel still dominates.", ""]
    st = load("stride")
    if st:
        out += ["## Power-of-two strides", "",
                "A contiguous stack of 2^k-byte frames puts every frame's current cache line in the same set; NumPy's "
                "2 MiB pages keep that conflict in L2. rankwise reads 8–47 such frames skewed and 48 or more through "
                "its tile ([design](design.md#memory-layout-deep-stacks-and-the-power-of-two-stride)). "
                "rankwise AVX-512, ns per pixel, best of four interleaved rounds.", "",
                "31 frames:", "",
                "| Frame | Stride | Read path | Median | Sigma clip |", "|---|---|---|---|---|"]
        for r in st["rows"]:
            pad = "exactly 2^k" if r["pad_bytes"] == 0 and r["frame"] != "721x1440" else f"+{r['pad_bytes']} B"
            if r["frame"] == "721x1440" and r["pad_bytes"] == 0:
                pad = "not 2^k"
            out.append(f"| {r['frame']} | {pad} | {r.get('path', '?')} | {r['median_ns_px']:.1f} | "
                       f"{r['sigclip_ns_px']:.1f} |")
        if st.get("depth"):
            out += ["", "1024 × 1024 frames by depth, exactly 2^k against +64 B:", "",
                    "| Frames | 2^k path | 2^k median | padded median | ratio | 2^k σ-clip | padded σ-clip | ratio |",
                    "|---|---|---|---|---|---|---|---|"]
            for r in st["depth"]:
                out.append(f"| {r['n']} | {r['pow2_path']} | {r['pow2_median']:.1f} | {r['padded_median']:.1f} | "
                           f"{r['pow2_median'] / r['padded_median']:.2f}× | {r['pow2_sigclip']:.1f} | "
                           f"{r['padded_sigclip']:.1f} | {r['pow2_sigclip'] / r['padded_sigclip']:.2f}× |")
        out.append("")
    (ROOT / "docs" / "benchmarks.md").write_text("\n".join(out) + "\n")
    print("wrote docs/benchmarks.md")


if __name__ == "__main__":
    main()
