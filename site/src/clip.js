// astropy-style sigma clipping of one slice (median center, population std, inclusive bounds),
// the same decisions rankwise makes per lane. Used for the probes; the images run on the GPU.

export function median(a, n) {
  const s = Array.from(a.slice(0, n)).sort((x, y) => x - y);
  return n % 2 ? s[n >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

export function sigmaClip(vals, n, sigma = 3, maxiters = 5) {
  const kept = new Uint8Array(n).fill(1);
  let lo = -Infinity, hi = Infinity, center = 0, std = 0, count = n;
  const buf = new Float64Array(n);
  for (let it = 0; it < maxiters; it++) {
    let m = 0;
    for (let i = 0; i < n; i++) if (kept[i]) buf[m++] = vals[i];
    center = median(buf, m);
    let mean = 0;
    for (let i = 0; i < m; i++) mean += buf[i];
    mean /= m;
    let v = 0;
    for (let i = 0; i < m; i++) v += (buf[i] - mean) ** 2;
    std = Math.sqrt(v / m);
    lo = center - sigma * std;
    hi = center + sigma * std;
    let c = 0;
    for (let i = 0; i < n; i++) {
      kept[i] = vals[i] >= lo && vals[i] <= hi ? 1 : 0;
      c += kept[i];
    }
    if (c === count) break;
    count = c;
  }
  let s = 0, c = 0;
  for (let i = 0; i < n; i++) if (kept[i]) { s += vals[i]; c++; }
  return { kept, lo, hi, center, std, mean: c ? s / c : NaN, rejected: n - c };
}
