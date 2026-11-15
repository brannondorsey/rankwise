import "./style.css";
import { $, $$, whenNear } from "./util.js";
import { initLanes } from "./lanes.js";
import { initStack } from "./stack.js";
import { initWeather } from "./weather.js";
import { initLandsat } from "./landsat.js";
import { initBench } from "./bench.js";
import { initCorrect } from "./correct.js";
import { initCode } from "./code.js";

// nav: solid once the page scrolls; highlight the section in view
const nav = $("#nav");
const onScroll = () => nav.classList.toggle("solid", scrollY > 40);
addEventListener("scroll", onScroll, { passive: true });
onScroll();
const links = $$(".links a");
const io = new IntersectionObserver((es) => {
  for (const e of es) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
}, { rootMargin: "-45% 0px -50% 0px" });
["why", "examples", "bench", "correct", "code"].forEach((id) => io.observe(document.getElementById(id)));

// reveal on scroll
const rv = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), rv.unobserve(e.target))), { rootMargin: "0px 0px -8% 0px" });
$$(".sec > .wrap > h2, .sec-head, .narrow p, .ex-head, .sec .demo, .race-card, .sweep-card, .tests-card, .ccd-card, .code-card, .api").forEach((x) => { x.classList.add("rv"); rv.observe(x); });

const safe = (f) => Promise.resolve().then(f).catch((e) => console.error(e));
safe(initLanes);
safe(initStack);
whenNear($("#cases"), () => safe(initWeather));
whenNear($("#ls-view"), () => safe(initLandsat));
whenNear($("#bench"), () => safe(initBench));
whenNear($("#correct"), () => safe(initCorrect));
safe(initCode);
