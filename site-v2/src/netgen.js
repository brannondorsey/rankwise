// Port of rankwise/src/netgen.rs: Batcher's odd-even merge network and rank pruning.

export function batcher(n) {
  let p2 = 1;
  while (p2 < n) p2 *= 2;
  const out = [];
  for (let p = 1; p < p2; p *= 2) {
    for (let k = p; k >= 1; k = Math.floor(k / 2)) {
      for (let j = k % p; j + k < p2; j += 2 * k) {
        for (let i = 0; i < k; i++) {
          const a = i + j, b = i + j + k;
          if (b >= p2) break;
          if (Math.floor(a / (2 * p)) === Math.floor(b / (2 * p)) && b < n) out.push([a, b]);
        }
      }
    }
  }
  return out;
}

/** Indices (into net) of the comparators that can influence the wires in `needed`. */
export function pruneMask(net, n, needed) {
  const live = new Array(n).fill(false);
  for (const k of needed) if (k < n) live[k] = true;
  const keep = new Array(net.length).fill(false);
  for (let c = net.length - 1; c >= 0; c--) {
    const [a, b] = net[c];
    if (live[a] || live[b]) {
      live[a] = live[b] = true;
      keep[c] = true;
    }
  }
  return keep;
}

export function medianRanks(n) {
  return n % 2 ? [n >> 1] : [n / 2 - 1, n / 2];
}

/** Ranks np.quantile(method="linear") reads for q over n values. */
export function quantileRanks(n, qs) {
  const s = new Set();
  for (const q of qs) {
    const h = q * (n - 1), lo = Math.floor(h);
    s.add(lo);
    if (h > lo && lo + 1 < n) s.add(lo + 1);
  }
  return [...s].sort((a, b) => a - b);
}

/** Columns for drawing: ASAP layers by wire availability, then sub-columns so spans never overlap. */
export function layout(net) {
  const free = new Map();
  const layerOf = [];
  let layers = 0;
  for (const [a, b] of net) {
    let l = 0;
    for (let w = a; w <= b; w++) l = Math.max(l, free.get(w) ?? 0);
    layerOf.push(l);
    for (let w = a; w <= b; w++) free.set(w, l + 1);
    layers = Math.max(layers, l + 1);
  }
  // within each layer, pack spans greedily into sub-columns
  const cols = [];
  const pos = new Array(net.length);
  let x = 0;
  for (let l = 0; l < layers; l++) {
    const idx = net.map((_, i) => i).filter((i) => layerOf[i] === l);
    const sub = [];
    for (const i of idx) {
      const [a, b] = net[i];
      let s = 0;
      while (sub[s] && sub[s].some(([c, d]) => !(b < c || a > d))) s++;
      (sub[s] ||= []).push([a, b]);
      pos[i] = x + s;
    }
    cols.push(sub.length);
    x += sub.length;
  }
  return { pos, width: x, layerOf, layers };
}
