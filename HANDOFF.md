# rankwise: handoff after v0.1 implementation

**Owner:** Brannon Dorsey · **Written:** 2026-11-09 · **State:** v0.1 feature-complete on `main`;
CI covers Linux x86-64/aarch64, macOS arm64 and Windows. Not published; names not registered.

Read this first, then the scope doc (*rankwise — Scope & Case Studies*, ask Brannon) and the
original handoff (Brannon has it; its §4 facts are all folded into [docs/conformance.md](docs/conformance.md)).

---

## 0. TL;DR

1. **Done.** Every v0.1 scope item: `select_ranks`, median, 13 NumPy quantile methods,
   min/max/ptp, arg-variants, trimmed mean/IQR/MAD, five rejection algorithms with
   astropy/Siril/PixInsight presets, NaN and `nodata`, f32/f64/u16 (u8 widened), any axis or
   frame lists, n ≤ 512, AVX-512/AVX2/NEON/portable chosen at runtime. Rust crate, C ABI +
   header, abi3 Python wheel. 1,039 conformance tests pass on every ISA.
2. **Speed (one core, AVX-512, vs the fastest alternative):** sigma clip 14×, median 18×
   (31 × 2048²); ≥ 5.3× at every depth 8–511; real GEFS quantiles 72×; uint16 nodata median
   15×. Targets were ≥ 5×.
3. **Before 0.1.0 can ship** (§4): real-data validation against Siril 1.4 / DSS / PixInsight,
   desktop CPUs (Zen 4/5, AVX2-only Intel, Apple M), real Landsat, an end-to-end run with file
   reads, the PCL license check, a legal review (TI US9665542), and Brannon registering the
   names. The power-of-two stride gap at 24–47 frames (1.46× vs the 1.2× target) is open.

## 1. Working agreements for this repo

* For v0.1 Brannon said: **commit and push straight to `main`, no PRs, don't register the
  crate/PyPI names.** Ask before changing either.
* Never write "identical" in docs without a conformance run showing it.
* Licensing: no Siril (GPLv3) code; check the PCL license before porting from PixInsight;
  implement from docs plus black-box comparison; keep [docs/design.md](docs/design.md)'s
  provenance table current.
* Results must stay **bit-identical across ISAs and thread counts** (the Rust oracle test
  enforces it): no FMA, sums through the 4-accumulator `acc4` keyed by `k % 4`.
* Out-of-scope work goes in a GitHub issue labelled `v0.2` (none filed yet; list in §5).
* Brannon wants short updates, and screenshots for anything visual.

## 2. Repo map

| Path | What |
|---|---|
| `rankwise/` | Core crate. `build.rs` emits register-resident networks (n = 2..32) to `$OUT_DIR/unrolled.rs`. |
| `rankwise/src/simd/` | `Vector` trait; `avx512.rs`, `avx2.rs`, `neon.rs`, `scalar.rs` (portable, AVX2 lane counts); `resolve_isa`, `set_isa_override`. |
| `rankwise/src/netgen.rs`, `net.rs` | Batcher networks, rank pruning (backwards liveness), bitonic merge; `NetPlan` = Trivial / Unrolled / Hybrid / Interp. |
| `rankwise/src/plan.rs` | Validation, ISA resolution, staging decision (`DEEP_STAGE = 96`, `CONFLICT_STAGE = 48`), threading, `describe()`. |
| `rankwise/src/engine.rs` | Per-block pipeline: load/stage, normalize, NaN/nodata masks, sort/select, epilogues. |
| `rankwise/src/clip.rs` | Rejection: per-lane rank bounds, sigma/MAD/median-sigma/winsorized/percentile, astropy final mask. |
| `rankwise/src/quantile.rs`, `spec.rs` | NumPy/SciPy quantile arithmetic; every op, option and preset type. |
| `rankwise/src/arch.rs` | `#[target_feature]` entry points per ISA × dtype (`entries!` macro). |
| `rankwise/tests/` | `networks.rs` (0-1 principle, n ≤ 20, exhaustive), `oracle.rs` (every ISA vs scalar oracle, bit for bit). |
| `rankwise-capi/` | `rw_*` C ABI, `catch_unwind` everywhere, `include/rankwise.h` (cbindgen 0.29.4; CI diffs it). |
| `rankwise-py/` | PyO3 0.30 abi3-py39 extension (`src/lib.rs`) + NumPy-style API (`python/rankwise/__init__.py`). |
| `conformance/` | pytest suite vs pinned NumPy 2.5.3, SciPy 1.18.1, astropy 8.0.2; Siril preset checks; C ABI vs Python. |
| `bench/` | One-core benchmarks → `bench/results/*.json` → `figures.py` (charts) and `report.py` (`docs/benchmarks.md`). |
| `examples/` | `c/stack.c`, `python/astro_demo.py` (README hero, comparison and rejection images). |
| `docs/` | `design.md` (architecture + provenance), `conformance.md`, `presets.md`, generated `benchmarks.md`, `diagrams.py` → SVGs. |

## 3. How to build, test and benchmark

```bash
cargo run --release -q -p rankwise --example describe     # ISAs this CPU has, networks chosen
cargo test --release -p rankwise --features ndarray       # 0-1 principle + oracle on every ISA
RANKWISE_ISA=scalar cargo test --release -p rankwise --test oracle
cargo test --release -p rankwise-capi
cargo fmt --all --check && cargo clippy --workspace --all-targets --features rankwise/ndarray -- -D warnings

pip install -r conformance/requirements.txt
maturin build --release -m rankwise-py/Cargo.toml -o dist && pip install --find-links dist rankwise
python -m pytest conformance -q                           # 1,039 tests, ~1 min per ISA set

pip install -r bench/requirements.txt                     # read its header: two packages need --no-deps
bash bench/run_all.sh                                     # pins taskset -c 1, then charts + report
```

NEON on an x86 box: `apt-get install qemu-user gcc-aarch64-linux-gnu libc6-dev-arm64-cross`,
`rustup target add aarch64-unknown-linux-gnu`, set
`CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc` and
`CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"`, then
`cargo test --release -p rankwise --target aarch64-unknown-linux-gnu`. CI also runs native
aarch64 Linux and macOS arm64. MSRV is 1.89 (checked in CI).

## 4. Open items, in priority order

Each has a recommendation; go ahead with it unless Brannon says otherwise.

| # | Item | State | Recommendation |
|---|---|---|---|
| 1 | **M7 real-data validation** | Siril comparisons used synthetic stacks and Siril 1.2.1 (Ubuntu apt). No DSS, no PixInsight. | Get a real registered light set from Brannon; compare against Siril 1.4 (build from source or AppImage), DSS, PixInsight and astropy. Add the script under `bench/` and a page section. |
| 2 | **Desktop CPUs** | All numbers are from one 2.1 GHz Xeon cloud VM (±15% noise). | Run `bench/run_all.sh` on Zen 4/5, an AVX2-only Intel and an Apple M chip; commit each JSON with its machine block; extend `report.py` to show a per-CPU table. |
| 3 | **Power-of-two strides, 24–47 frames** | 31 × 1024² median: 18 ns/px on 2^k strides vs 12 padded (1.46×; target 1.2×). Sigma clip 1.15×. Staging only pays from 48 frames; software prefetch and a grouped-comparator interpreter were tried and were slower. | Try staging only the frames that share an L2 set (partial tile), or walking two blocks a page apart in alternation. Measure with `bench/stride.py` and `staging="always"/"never"`. If nothing lands, keep the documented padding advice. |
| 4 | **Real Landsat demo** | `bench/landsat.py` uses synthetic uint16 scenes with cloud nodata: the USGS bucket is requester-pays. | Needs AWS credentials (ask Brannon); the script's shape and nodata handling already match C2 L2 SR. |
| 5 | **End-to-end with file reads** | Only Siril's rows include I/O. | Add a run that reads the 31 FITS files (astropy.io.fits or fitsio, memory-mapped) and then calls rankwise, next to Siril's row. |
| 6 | **Siril presets** | sigma, median-sigma, percentile: identical at n = 5, 8, 15, 31. Winsorized: exact at 31, 95–98% below ~8 frames. k-MAD (`rej a`): not reproduced; `preset="siril", method="mad"` raises. Details in [docs/presets.md](docs/presets.md). | Retest against Siril 1.4 before more black-box work; its docs may have changed. Do not read Siril source. |
| 7 | **PixInsight winsorized preset** | Implemented from PCL's public docs (Sn·1.1926 start, cutoff 5); never run against PixInsight. | Check the PCL license, then validate on Brannon's PixInsight install or drop the preset from 0.1.0. |
| 8 | **Legal** | Design log notes TI US9665542 (different method). | Brannon arranges review before 1.0. |
| 9 | **Release** | `wheels.yml` builds manylinux x86-64/aarch64, macOS arm64/x86-64 and Windows wheels but does not publish. Names unregistered. | When Brannon registers names: add trusted publishing to `wheels.yml`, `cargo publish` the crate (capi stays unpublished or ships as release artifacts), tag `v0.1.0`, move the CHANGELOG entry out of "unreleased". |
| 10 | Known differences | Signed-zero sign bit (x86 vs NEON min/max); `median(out=float64)` on f32 averages in f32; `trimmed_mean` more accurate than SciPy's f32; `nanarg*` all-NaN returns −1 (+ `strict=True`). All in [docs/conformance.md](docs/conformance.md#known-differences). | Leave unless a user needs it; −0 canonicalization would cost a blend per output. |

## 5. v0.2 backlog (from the scope doc; file as `v0.2` issues)

* Rolling windows and grouped reductions.
* Weighted median and quantiles.
* More than 512 frames (`MAX_FRAMES` in `lib.rs`; the hybrid network and tile sizes assume it).
* Linear-fit clipping and generalized ESD. These break v0.1's "kept is exactly
  `v_lo <= x <= v_hi`" rescan trick, so frame identity has to ride through the network.
* R and Julia bindings (the C ABI is the base), xarray/dask helpers, and a Siril script.

## 6. Things that cost time and aren't obvious

* **NumPy's quantile arithmetic depends on how `q` and `out=` are passed.** The original
  handoff said `out=` means "float64 math, rounded once"; that holds only for the `t < 0.5`
  branch. NumPy's `_lerp` does the `t >= 0.5` subtraction in `out.dtype`.
* **`np.nanmedian` on ndim ≥ 2** goes through `np.ma.median` and computes `(x + x) / 2` for odd
  counts (overflows to inf above FLT_MAX/2). **`scipy.stats.iqr`** uses `scipy.stats.quantile`,
  not NumPy's, and its normal constant is 1.3489795003921636.
* **Power-of-two penalty depends on page size.** L2 is physically indexed; NumPy madvises big
  arrays onto 2 MiB transparent huge pages, which keeps the conflict. A test that allocates
  with 4 KiB pages will not reproduce it.
* **Register budget.** Unrolled networks stop at 32 because AVX-512 has 32 zmm registers;
  `Vector::UNROLL` per ISA (AVX-512 32, AVX2 16, NEON 24, portable 0) picks unrolled vs
  interpreted. `MAXL = 32` lanes is the widest block (u16 on AVX-512).
* **The cloud VM is noisy.** rankwise rows are best of 7; competitors best of 2–3. Pin with
  `taskset` and set every `*_NUM_THREADS=1` before imports (`bench/common.py` does this).
* **Tooling quirks.** `siril-cli -d DIR -s /abs/path/script.ssf` (relative script paths fail).
  `imcombiners`/`ccdproc` depend on `pims`, which fails to build on Python 3.13: install them
  with `--no-deps`. The capi crate must not build an rlib named `rankwise` (it collides with
  the core crate), so it is cdylib + staticlib with `doctest = false`. Windows skips the
  Unix-only C example test.
* **GEFS data** comes from `https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr`
  (needs `aiohttp`); `bench/gefs.py` caches a 310 MB `.npz` in `bench/data/` (gitignored) and
  `--figures-only` redraws the charts from the cache.
* **Charts** use `bench/figures.py`'s `THEMES` (light/dark), Inter, and `headline()`; README
  images are `<picture>` pairs. Regenerate both themes when changing any chart.

## 7. Suggested first session

1. `cargo test` + conformance on your machine to confirm a clean baseline.
2. Item 3 (stride gap) if you have only a VM; items 1–2 if Brannon can provide data and hardware.
3. File the §5 backlog as `v0.2` issues once Brannon confirms he wants issues created.
