"""Download the GEFS run bench/gefs.py uses into bench/data (same cache file)."""
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
f = ROOT / "bench" / "data" / "gefs_t2m_conus.npz"
if not f.exists():
    t = time.perf_counter()
    ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr", chunks=None)
    run = ds["temperature_2m"].sel(init_time="2026-11-01T00", latitude=slice(52.0, 22.0), longitude=slice(-128.0, -64.0)).isel(lead_time=slice(0, 81))
    run = run.transpose("ensemble_member", "lead_time", "latitude", "longitude")
    t2m = np.ascontiguousarray(run.values.astype(np.float32))
    lat, lon = run.latitude.values, run.longitude.values
    lead_h = (run.lead_time.values / np.timedelta64(1, "h")).astype(np.float32)
    np.savez(f, t2m=t2m, lat=lat, lon=lon, lead_h=lead_h)
    print(f"downloaded {t2m.shape} in {time.perf_counter() - t:.0f} s")
print("ok")
