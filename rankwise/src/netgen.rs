// Comparator-network generators. This file is shared verbatim by `build.rs` (which emits
// straight-line, register-resident networks for small n) and by the crate itself, so it must
// stay free of crate-internal dependencies.
//
// Design log: Batcher's odd-even merge sort (K. E. Batcher, "Sorting networks and their
// applications", AFIPS 1968) and the bitonic half-cleaner (same paper). Rank pruning is a
// backwards liveness pass over the comparator list. None of this is derived from Siril,
// PixInsight or any other stacker's code.

/// A compare-exchange: after it runs, wire `.0` holds the min and wire `.1` the max.
pub type Comparator = (u16, u16);

/// Batcher odd-even merge sort for `n` wires, built for the next power of two and then
/// restricted to wires `< n` (a padded wire would hold +inf and never swap).
///
/// Returns the comparators and, for each comparator, the merge level `p` (1, 2, 4, ...) that
/// produced it. Every comparator with `p < c` stays inside an aligned block of `2p <= c` wires,
/// so the prefix with `p < c` sorts each aligned block of `c` wires independently.
pub fn batcher_levels(n: usize) -> (Vec<Comparator>, Vec<u32>) {
    let mut p2 = 1usize;
    while p2 < n {
        p2 *= 2;
    }
    let mut out = Vec::new();
    let mut lvl = Vec::new();
    let mut p = 1usize;
    while p < p2 {
        let mut k = p;
        while k >= 1 {
            let mut j = k % p;
            while j + k < p2 {
                for i in 0..k {
                    let a = i + j;
                    let b = i + j + k;
                    if b >= p2 {
                        break;
                    }
                    if a / (2 * p) == b / (2 * p) && b < n {
                        out.push((a as u16, b as u16));
                        lvl.push(p as u32);
                    }
                }
                j += 2 * k;
            }
            k /= 2;
        }
        p *= 2;
    }
    (out, lvl)
}

/// Full Batcher sorting network for `n` wires.
pub fn batcher(n: usize) -> Vec<Comparator> {
    batcher_levels(n).0
}

/// Drop every comparator that cannot influence the wires in `needed`.
///
/// Walks the network backwards keeping a live set: a comparator survives if either of its
/// wires is live after it, and then both of its wires become live before it. Removing a dead
/// comparator never changes a live output, whatever ran before it.
pub fn prune(net: &[Comparator], n: usize, needed: &[usize]) -> Vec<Comparator> {
    let mut live = vec![false; n];
    for &k in needed {
        if k < n {
            live[k] = true;
        }
    }
    let mut keep = Vec::with_capacity(net.len());
    for &(a, b) in net.iter().rev() {
        if live[a as usize] || live[b as usize] {
            live[a as usize] = true;
            live[b as usize] = true;
            keep.push((a, b));
        }
    }
    keep.reverse();
    keep
}

/// The ranks a plain median of `n` values needs.
pub fn median_ranks(n: usize) -> Vec<usize> {
    if n == 0 {
        vec![]
    } else if n % 2 == 1 {
        vec![n / 2]
    } else {
        vec![n / 2 - 1, n / 2]
    }
}

/// Bitonic half-cleaner cascade that sorts any bitonic (valley-shaped) sequence of `n` wires,
/// padded virtually with +inf up to the next power of two. Used to sort |x - median| over an
/// already sorted stack, whose deviations fall and then rise.
pub fn bitonic_merge(n: usize) -> Vec<Comparator> {
    let mut p2 = 1usize;
    while p2 < n {
        p2 *= 2;
    }
    let mut out = Vec::new();
    let mut k = p2 / 2;
    while k >= 1 {
        for i in 0..p2 {
            if i & k == 0 && i + k < n {
                out.push((i as u16, (i + k) as u16));
            }
        }
        k /= 2;
    }
    out
}
