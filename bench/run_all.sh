#!/usr/bin/env bash
# Run every benchmark on one pinned core, then render charts and the report.
#   bash bench/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PIN="taskset -c ${BENCH_CPU:-1}"
command -v taskset >/dev/null || PIN=""
$PIN python3 bench/astro.py 31 2048
$PIN python3 bench/ensemble.py
$PIN python3 bench/stride.py
# the "before" curve in bench_stride_*.png came from the previous build: bench/stride.py --depth-only --out stride_v0.1
$PIN python3 bench/sweep.py
$PIN python3 bench/landsat.py
if command -v siril-cli >/dev/null; then $PIN python3 bench/siril/time_siril.py; fi
$PIN python3 bench/e2e.py  # copies Siril's row from results/siril.json
$PIN python3 bench/gefs.py
python3 bench/figures.py
python3 bench/report.py
