//! Network correctness by the 0-1 principle: a comparator network sorts (or selects rank r
//! for) every input iff it does so for every 0/1 input.

use rankwise::netgen::{batcher, bitonic_merge, median_ranks, prune};

fn run01(net: &[(u16, u16)], bits: u32, n: usize) -> u32 {
    let mut v = bits;
    for &(a, b) in net {
        let (x, y) = ((v >> a) & 1, (v >> b) & 1);
        // min to a, max to b
        let (lo, hi) = (x & y, x | y);
        v = (v & !(1 << a) & !(1 << b)) | (lo << a) | (hi << b);
    }
    v & ((1u64 << n) - 1) as u32
}

fn sorted01(bits: u32, n: usize) -> u32 {
    let ones = bits.count_ones();
    // zeros first, then ones
    (((1u64 << ones) - 1) << (n as u32 - ones)) as u32
}

#[test]
fn batcher_sorts_every_01_input_up_to_20() {
    for n in 1..=20usize {
        let net = batcher(n);
        for bits in 0u32..(1u32 << n) {
            assert_eq!(run01(&net, bits, n), sorted01(bits, n), "n={n} bits={bits:b}");
        }
    }
}

#[test]
fn pruned_networks_select_their_ranks_up_to_20() {
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for n in 1..=20usize {
        let full = batcher(n);
        let mut rank_sets = vec![median_ranks(n), vec![0], vec![n - 1]];
        for _ in 0..3 {
            let k = 1 + (rnd() as usize % 3);
            rank_sets.push((0..k).map(|_| rnd() as usize % n).collect());
        }
        for ranks in rank_sets {
            let net = prune(&full, n, &ranks);
            assert!(net.len() <= full.len());
            for bits in 0u32..(1u32 << n) {
                let got = run01(&net, bits, n);
                let want = sorted01(bits, n);
                for &r in &ranks {
                    assert_eq!((got >> r) & 1, (want >> r) & 1, "n={n} ranks={ranks:?} r={r}");
                }
            }
        }
    }
}

#[test]
fn bitonic_merger_sorts_valleys() {
    for n in 1..=64usize {
        let net = bitonic_merge(n);
        // 0-1 valleys: ones, zeros, ones
        for a in 0..=n {
            for b in a..=n {
                let bits: u64 = (0..n).filter(|&i| i < a || i >= b).fold(0, |v, i| v | (1 << i));
                let mut v: Vec<u8> = (0..n).map(|i| ((bits >> i) & 1) as u8).collect();
                for &(x, y) in &net {
                    let (p, q) = (v[x as usize], v[y as usize]);
                    v[x as usize] = p.min(q);
                    v[y as usize] = p.max(q);
                }
                assert!(v.windows(2).all(|w| w[0] <= w[1]), "n={n} a={a} b={b}");
            }
        }
    }
}

#[test]
fn comparator_counts_match_the_prototype() {
    // Median-only vs full-sort sizes quoted in the scope doc.
    for (n, med, full) in [(15, 49, 59), (31, 152, 186), (63, 439, 537), (127, 1206, 1464), (255, 3189, 3831)] {
        assert_eq!(batcher(n).len(), full, "n={n}");
        assert_eq!(prune(&batcher(n), n, &median_ranks(n)).len(), med, "n={n}");
    }
}
