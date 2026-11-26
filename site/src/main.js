import "./style.css";
import { $, $$, reduced, whenNear } from "./util.js";
import { initHero } from "./hero.js";
import { initStack } from "./stack.js";
import { initLanes } from "./lanes.js";
import { initNetwork } from "./network.js";
import { initWeather } from "./weather.js";
import { initLandsat } from "./landsat.js";
import { initBench } from "./bench.js";
import { initCorrect } from "./correct.js";
import { initCode } from "./code.js";

// nav: solid after the hero starts moving; highlight the section in view
const nav = $("#nav");
const onScroll = () => nav.classList.toggle("solid", scrollY > 40);
addEventListener("scroll", onScroll, { passive: true });
onScroll();
const links = $$(".links a");
const io = new IntersectionObserver((es) => {
  for (const e of es) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
}, { rootMargin: "-45% 0px -50% 0px" });
["lanes", "bench", "code", "examples"].forEach((id) => io.observe(document.getElementById(id)));

// parallax cards: as each section's successor slides up over it, the section drifts down at a
// fraction of the scroll speed, shrinks a touch and dims
const cards = $$("main > section");
cards.forEach((c, i) => (c.style.zIndex = String(i + 1)));
if (!reduced) {
  let queued = false;
  const k0 = new Float32Array(cards.length);
  const parallax = () => {
    queued = false;
    const vh = innerHeight;
    for (let i = 0; i < cards.length - 1; i++) {
      const k = Math.min(1, Math.max(0, (vh - cards[i + 1].getBoundingClientRect().top) / vh));
      if (k === k0[i]) continue;
      k0[i] = k;
      const c = cards[i];
      c.style.transform = k ? `translate3d(0, ${(k * vh * 0.32).toFixed(1)}px, 0) scale(${(1 - 0.035 * k).toFixed(4)})` : "";
      c.style.setProperty("--dim", (0.6 * k).toFixed(3));
    }
  };
  const kick = () => { if (!queued) { queued = true; requestAnimationFrame(parallax); } };
  addEventListener("scroll", kick, { passive: true });
  addEventListener("resize", kick);
  new ResizeObserver(kick).observe($("main")); // demos that load late change section heights
  parallax();
}

// reveal on scroll
const rv = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), rv.unobserve(e.target))), { rootMargin: "0px 0px -8% 0px" });
$$(".sec-head, .ex-head, .demo, .explain3, .race-card, .sweep-card, .tests-card, .code-card, .api").forEach((x) => { x.classList.add("rv"); rv.observe(x); });

const safe = (f) => Promise.resolve().then(f).catch((e) => console.error(e));
safe(initHero);
whenNear($("#lanes"), () => safe(initLanes));
whenNear($("#prune"), () => safe(initNetwork));
whenNear($("#bench"), () => safe(initBench));
whenNear($("#correct"), () => safe(initCorrect));
safe(initCode);
whenNear($("#stack"), () => safe(initStack));
whenNear($("#cases"), () => safe(initWeather));
whenNear($("#ls-view"), () => safe(initLandsat));
