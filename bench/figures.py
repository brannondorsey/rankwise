"""Render the benchmark charts from bench/results/*.json into docs/img (light and dark).

    python bench/figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e6e5e0",
                  rw=["#2a78d6", "#6da7ec", "#b7d3f6"], other="#c3c2bb", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8f8e88", grid="#2e2d2a",
                 rw=["#5598e7", "#3987e5", "#256abf"], other="#55544f", s2="#d95926"),
}
FONT = "Inter" if any("Inter" == f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 10.5, "svg.fonttype": "none"})


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def panel_title(fig, ax, title, t):
    tr = blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.text(0.015, 1.0, title, transform=tr, fontsize=11.5, color=t["ink"], fontweight="semibold", va="bottom")


def headline(fig, title, sub, t, top=0.985):
    h = fig.get_size_inches()[1]
    fig.text(0.015, top, title, ha="left", va="top", fontsize=15, fontweight="bold", color=t["ink"])
    fig.text(0.015, top - 0.34 / h, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def rw_color(tool, t):
    if "AVX-512" in tool or "NEON" in tool:
        return t["rw"][0]
    if "AVX2" in tool:
        return t["rw"][1]
    return t["rw"][2]


def bars(ax, rows, t, ref, unit="s"):
    rows = sorted(rows, key=lambda r: r["seconds"])
    names = [r["tool"].replace("rankwise · scalar", "rankwise · portable") for r in rows]
    vals = [r["seconds"] for r in rows]
    cols = [rw_color(r["tool"], t) if r["tool"].startswith("rankwise") else t["other"] for r in rows]
    y = range(len(rows))[::-1]
    ax.barh(list(y), vals, color=cols, height=0.62)
    vmax = max(vals)
    for yi, r, v in zip(y, rows, vals):
        is_rw = r["tool"].startswith("rankwise")
        label = f"{v:.3g} s"
        if not is_rw:
            label += f"  ·  {v / ref:.0f}× slower" if v / ref >= 10 else f"  ·  {v / ref:.1f}× slower"
        ax.text(v + vmax * 0.012, yi, label, va="center", fontsize=9.5,
                color=t["ink"] if is_rw else t["ink2"], fontweight="semibold" if is_rw else "normal")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    for lab, r in zip(ax.get_yticklabels(), rows):
        is_rw = r["tool"].startswith("rankwise")
        lab.set_color(t["ink"] if is_rw else t["ink2"])
        lab.set_fontweight("semibold" if is_rw else "normal")
    ax.set_xlim(0, vmax * 1.42)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(False)


def astro(theme):
    t = THEMES[theme]
    d = json.loads((RES / "astro_31x2048.json").read_text())
    siril = json.loads((RES / "siril.json").read_text()) if (RES / "siril.json").exists() else None
    rows = [r for r in d["rows"] if "f32 bounds" not in r["tool"]]
    groups = []
    for op, title in (("sigclip", "Sigma-clipped mean · 3σ low and high, up to 5 iterations"), ("median", "Median")):
        rs = [dict(tool=r["tool"].replace(" (1 pass)", " (one pass)"), seconds=r["seconds"]) for r in rows if r["op"] == op]
        if siril:
            rs += [dict(tool="Siril 1.2.1 *", seconds=r["seconds"]) for r in siril["rows"] if r["op"] == op]
        groups.append((title, rs))
    h = [len(g[1]) for g in groups]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 0.36 * sum(h) + 2.6), gridspec_kw=dict(height_ratios=h, hspace=0.32),
                             facecolor=t["surface"])
    speed = {}
    for ax, (title, rs), op in zip(axes, groups, ("sigclip", "median")):
        style(ax, t)
        ref = min(r["seconds"] for r in rs if "AVX-512" in r["tool"] or "NEON" in r["tool"])
        bars(ax, rs, t, ref)
        panel_title(fig, ax, title, t)
        speed[op] = min(r["seconds"] for r in rs if not r["tool"].startswith("rankwise")) / ref
    m = d["machine"]
    headline(fig, f"rankwise sigma-clips {speed['sigclip']:.0f}× and medians {speed['median']:.0f}× faster than the fastest alternative",
             f"Seconds to stack {d['n']} frames of {d['side']}×{d['side']} float32, lower is better · {m['cpu']}, one core, "
             "best of 3 after a warm-up", t)
    note = "rankwise results are identical to np.median and to astropy's sigma_clip decisions on every pixel."
    if siril:
        note += "  * Siril reads its 31 FITS files from the page cache; the others start from arrays in RAM."
    fig.text(0.015, 0.008, note, fontsize=8.5, color=t["muted"], wrap=True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.07)
    fig.savefig(IMG / f"bench_astro_{theme}.png", dpi=200, facecolor=t["surface"])
    plt.close(fig)


def sweep(theme):
    t = THEMES[theme]
    d = json.loads((RES / "sweep_1024.json").read_text())
    ns = [r["n"] for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.4), facecolor=t["surface"])
    style(ax, t)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    series = []
    for key, label, col in (("sigclip", "Sigma-clipped mean", t["rw"][0]), ("median", "Median", t["s2"])):
        sp = [min(r[f"reducers_{key}"], r[f"imcombiners_{key}"]) / r[f"rankwise_{key}"] for r in d["rows"]]
        series.append((label, col, sp))
        ax.plot(ns, sp, color=col, linewidth=2, marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.text(ns[-1] * 1.12, sp[-1], f"{label}  {sp[-1]:.1f}×" if sp[-1] < 10 else f"{label}  {sp[-1]:.0f}×",
                va="center", color=t["ink"], fontsize=10)
        i = max(range(len(sp)), key=lambda k: sp[k])
        ax.annotate(f"{sp[i]:.0f}×", (ns[i], sp[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5,
                    color=t["ink2"])
    ax.axhline(5, color=t["muted"], linewidth=1)
    ax.text(ns[0] * 0.92, 5, "5× target", va="bottom", ha="left", fontsize=9, color=t["muted"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 3.4)
    ax.set_ylim(0, max(max(s[2]) for s in series) * 1.18)
    ax.set_xlabel("Frames in the stack", color=t["ink2"])
    ax.set_ylabel("Speedup (×)", color=t["ink2"])
    lo = min(min(s[2]) for s in series)
    headline(fig, f"At least {lo:.1f}× faster at every stack depth from 8 to 511 frames",
             "1024×1024 float32 frames, one core, AVX-512 · speedup = time of the faster of reducers and imcombiners "
             "÷ rankwise's time", t, top=0.975)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.13)
    fig.savefig(IMG / f"bench_sweep_{theme}.png", dpi=200, facecolor=t["surface"])
    plt.close(fig)


def ensemble(theme):
    t = THEMES[theme]
    d = json.loads((RES / "ensemble_31x721x1440.json").read_text())
    groups = []
    for kind, title in (("quantile", "quantile · fields without NaN (GEFS)"), ("nanquantile", "nanquantile · 2% NaN, against NaN-aware tools")):
        groups.append((title, [dict(tool=r["tool"], seconds=r["seconds"]) for r in d["rows"] if r["kind"] == kind]))
    h = [len(g[1]) for g in groups]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 0.36 * sum(h) + 2.5), gridspec_kw=dict(height_ratios=h, hspace=0.34),
                             facecolor=t["surface"])
    speed = []
    for ax, (title, rs) in zip(axes, groups):
        style(ax, t)
        ref = min(r["seconds"] for r in rs if "AVX-512" in r["tool"] or "NEON" in r["tool"])
        bars(ax, rs, t, ref)
        panel_title(fig, ax, title, t)
        speed.append(min(r["seconds"] for r in rs if not r["tool"].startswith("rankwise")) / ref)
    headline(fig, f"Ensemble quantiles {speed[0]:.0f}× faster than NumPy, {speed[1]:.0f}× faster than fastnanquantile",
             "Seconds per lead time: 31 members × 721 × 1440 float32 (0.25° global grid), lower is better · one core", t)
    fig.text(0.015, 0.01, "rankwise is bit-identical to np.quantile / np.nanquantile, dtype included; numbagg and "
             "fastnanquantile differ from NumPy in the last bits.", fontsize=8.5, color=t["muted"])
    fig.subplots_adjust(left=0.27, right=0.98, top=0.8, bottom=0.08)
    fig.savefig(IMG / f"bench_ensemble_{theme}.png", dpi=200, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    for th in THEMES:
        for f in (astro, sweep, ensemble):
            try:
                f(th)
            except FileNotFoundError as e:
                print("skip", f.__name__, e)
    print("charts in", IMG)
