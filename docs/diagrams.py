"""Draw the README's explanatory SVGs: the lane layout and a rank-pruned sorting network.

    python docs/diagrams.py

Each SVG carries light and dark colors and switches with prefers-color-scheme.
"""

from pathlib import Path

import numpy as np

import rankwise as rw

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = """<style>
  .bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #8a8984; }
  .wire { stroke: #c9c8c2; } .cmp { stroke: #2a78d6; fill: #2a78d6; } .dead { stroke: #dddcd7; fill: #dddcd7; }
  .hl { stroke: #eb6834; } .hlfill { fill: #eb6834; } .frame { stroke: #0b0b0b; fill: none; }
  .cell0 { fill: #cde2fb; } .cell1 { fill: #9ec5f4; } .cell2 { fill: #6da7ec; } .cell3 { fill: #3987e5; }
  .cell4 { fill: #256abf; } .cell5 { fill: #184f95; } .cell6 { fill: #0d366b; }
  text { font-family: ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
  .mono { font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .muted { fill: #8f8e88; }
    .wire { stroke: #4a4945; } .cmp { stroke: #3987e5; fill: #3987e5; } .dead { stroke: #3a3936; fill: #3a3936; }
    .hl { stroke: #d95926; } .hlfill { fill: #d95926; } .frame { stroke: #f0efec; }
  }
</style>"""


def batcher(n):
    p2 = 1
    while p2 < n:
        p2 *= 2
    out, p = [], 1
    while p < p2:
        k = p
        while k >= 1:
            j = k % p
            while j + k < p2:
                for i in range(k):
                    a, b = i + j, i + j + k
                    if b >= p2:
                        break
                    if a // (2 * p) == b // (2 * p) and b < n:
                        out.append((a, b))
                j += 2 * k
            k //= 2
        p *= 2
    return out


def prune(net, n, needed):
    live = [False] * n
    for k in needed:
        live[k] = True
    keep = set()
    for i in range(len(net) - 1, -1, -1):
        a, b = net[i]
        if live[a] or live[b]:
            live[a] = live[b] = True
            keep.add(i)
    return keep


def network_svg(n=9):
    net = batcher(n)
    keep = prune(net, n, [n // 2])
    assert len(net) == rw.network_size(n) and len(keep) == rw.network_size(n, [n // 2])
    # greedy columns so comparators in one column never overlap vertically
    occ, cols = [-1] * n, []
    for a, b in net:
        c = max(occ[a:b + 1]) + 1
        cols.append(c)
        for w in range(a, b + 1):
            occ[w] = c
    ncol = max(cols) + 1
    x0, dx, y0, dy = 150, 34, 92, 34
    width = x0 + ncol * dx + 170
    height = y0 + (n - 1) * dy + 92
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
         f'aria-label="A {n}-wire Batcher sorting network with the comparators a median needs highlighted">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}" rx="10"/>',
         f'<text class="ink" x="24" y="34" font-size="17" font-weight="600">A median needs {len(keep)} of {len(net)} comparators (152 of 186 at 31 frames)</text>',
         f'<text class="ink2" x="24" y="58" font-size="13">Batcher network for {n} frames. Grey comparators cannot reach the middle rank, so rankwise drops them.</text>']
    xe = x0 + ncol * dx
    for w in range(n):
        y = y0 + w * dy
        cls = "hl" if w == n // 2 else "wire"
        sw = 2.5 if w == n // 2 else 1.5
        s.append(f'<line class="{cls}" x1="{x0 - 30}" y1="{y}" x2="{xe + 10}" y2="{y}" stroke-width="{sw}"/>')
        s.append(f'<text class="ink2 mono" x="{x0 - 40}" y="{y + 4}" font-size="12" text-anchor="end">frame {w}</text>')
        if w == n // 2:
            s.append(f'<text class="ink" x="{xe + 20}" y="{y + 5}" font-size="13" font-weight="600">median (rank {w})</text>')
    for i, ((a, b), c) in enumerate(zip(net, cols)):
        x = x0 + c * dx + dx / 2
        cls = "cmp" if i in keep else "dead"
        ya, yb = y0 + a * dy, y0 + b * dy
        s.append(f'<line class="{cls}" x1="{x}" y1="{ya}" x2="{x}" y2="{yb}" stroke-width="2.5"/>')
        s.append(f'<circle class="{cls}" cx="{x}" cy="{ya}" r="4.5"/><circle class="{cls}" cx="{x}" cy="{yb}" r="4.5"/>')
    yl = y0 + (n - 1) * dy + 44
    s.append(f'<line class="cmp" x1="24" y1="{yl}" x2="24" y2="{yl + 14}" stroke-width="2.5"/>'
             f'<circle class="cmp" cx="24" cy="{yl}" r="4"/><circle class="cmp" cx="24" cy="{yl + 14}" r="4"/>'
             f'<text class="ink2" x="38" y="{yl + 12}" font-size="12">one comparator = one vminps + one vmaxps on two whole registers (16 pixels)</text>')
    s.append("</svg>")
    (OUT / "network.svg").write_text("\n".join(s))


def lanes_svg(n=9, lanes=16):
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (n, lanes))
    data[2, 5] += 6  # a cosmic ray in frame 2, pixel 5
    data[6, 11] += 5
    srt = np.sort(data, axis=0)
    lo, hi = np.percentile(data, 2), np.percentile(data, 98)

    def cls(v):
        return f"cell{int(np.clip((v - lo) / (hi - lo) * 6.99, 0, 6))}"

    cs, gap = 22, 3
    gw = lanes * (cs + gap)
    x1, x2, y0 = 120, 120 + gw + 120, 96
    width = x2 + gw + 40
    height = y0 + n * (cs + gap) + 70
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
         f'aria-label="Registers hold one frame each for 16 pixels; after the network each register holds one rank">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}" rx="10"/>',
         '<text class="ink" x="24" y="34" font-size="17" font-weight="600">Every SIMD lane sorts its own pixel</text>',
         '<text class="ink2" x="24" y="58" font-size="13">Register k holds frame k for 16 pixels (AVX-512, float32). '
         'The network min/maxes whole registers, so all 16 stacks sort at once, with no branches.</text>',
         f'<text class="ink2" x="{x1}" y="{y0 - 12}" font-size="12">loaded: register k = frame k</text>',
         f'<text class="ink2" x="{x2}" y="{y0 - 12}" font-size="12">after the network: register k = rank k</text>']
    for gx, mat in ((x1, data), (x2, srt)):
        for k in range(n):
            y = y0 + k * (cs + gap)
            if gx == x1:
                s.append(f'<text class="ink2 mono" x="{x1 - 12}" y="{y + 15}" font-size="11" text-anchor="end">zmm{k}</text>')
            for j in range(lanes):
                s.append(f'<rect class="{cls(mat[k, j])}" x="{gx + j * (cs + gap)}" y="{y}" width="{cs}" height="{cs}" rx="3"/>')
        # one lane = one pixel
        xl = gx + 5 * (cs + gap) - 2
        s.append(f'<rect class="frame" x="{xl}" y="{y0 - 3}" width="{cs + 4}" height="{n * (cs + gap) + 2}" rx="4" stroke-width="1.5"/>')
    ym = y0 + (n // 2) * (cs + gap)
    s.append(f'<rect class="hl" fill="none" x="{x2 - 3}" y="{ym - 3}" width="{gw + 3}" height="{cs + 6}" rx="5" stroke-width="2"/>')
    s.append(f'<text class="ink" x="{x2 + gw + 8 - 4}" y="{ym - 8}" font-size="12" font-weight="600" text-anchor="end"></text>')
    ya = y0 + n * (cs + gap) / 2
    s.append(f'<line class="wire" x1="{x1 + gw + 18}" y1="{ya}" x2="{x2 - 22}" y2="{ya}" stroke-width="2"/>'
             f'<path class="ink2" d="M {x2 - 22} {ya - 6} L {x2 - 12} {ya} L {x2 - 22} {ya + 6} Z"/>'
             f'<text class="ink2" x="{(x1 + gw + x2) / 2}" y="{ya - 10}" font-size="12" text-anchor="middle">network</text>')
    yb = y0 + n * (cs + gap) + 30
    s.append(f'<text class="ink2" x="{x1}" y="{yb}" font-size="12">outlined: one lane = one pixel\'s 9 frames (with a cosmic ray)</text>')
    s.append(f'<text class="ink2" x="{x2}" y="{yb}" font-size="12">the median is register 4, already in place for all 16 pixels</text>')
    s.append("</svg>")
    (OUT / "lanes.svg").write_text("\n".join(s))


if __name__ == "__main__":
    network_svg()
    lanes_svg()
    print("wrote", OUT / "network.svg", OUT / "lanes.svg")
