# rankwise marketing site

Static site (Vite + three.js, no framework). Every demo runs on the repository's real data:

| Demo | Data |
|---|---|
| Hero deck, live stacking (01), lanes (02) | the 40 synthetic subs of `examples/python/astro_demo.py` |
| Pruned networks (03) | `src/netgen.js`, a port of `rankwise/src/netgen.rs` |
| Weather (04) | real NOAA GEFS run of `bench/gefs.py`; p10/p50/p90 from `rw.quantile` |
| Composites (05) | the 90 Landsat scenes of `bench/landsat.py`, at 120 m |
| Benchmarks, conformance (06–07) | `bench/results/*.json` |

The stacking and composite demos compute mean / median / sigma clipping per pixel in a
fragment shader (rank bounds over sorted codes, as rankwise does per SIMD lane).

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # → dist/, deployable anywhere (relative base path)
```

## Regenerating `public/data` and `src/bench.json`

Needs the rankwise wheel (`maturin build --release -m rankwise-py/Cargo.toml`) plus
`scipy pillow xarray zarr fsspec aiohttp pystac-client planetary-computer rasterio pyproj`.

```bash
python site/scripts/fetch_gefs.py        # → bench/data/gefs_t2m_conus.npz (same cache as bench/gefs.py)
python site/scripts/fetch_landsat.py     # → bench/data/landsat_046027_site512.npz
python site/scripts/build_assets.py      # astro, landsat, gefs, bench → site/public/data, site/src/bench.json
```

GEFS data © NOAA via dynamical.org (CC BY 4.0). Landsat imagery courtesy of the U.S. Geological Survey.
