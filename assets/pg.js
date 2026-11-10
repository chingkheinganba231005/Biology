/* Cells to Code · playground helpers. Each playground loads this file, then its own script. */
(function () {
  "use strict";
  try { const t = parent.document.documentElement.dataset.theme; if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  const $ = (id) => document.getElementById(id);
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? "" : v);
    }
    kids.flat(Infinity).forEach((c) => { if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c))); });
    return e;
  }
  const NS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...kids) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
    kids.flat().forEach((c) => { if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c))); });
    return e;
  }
  // tabs: buttons with data-t="paneId" inside .tabs, panes with class "pane"
  function tabs() {
    document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
      document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === b.dataset.t));
      window.dispatchEvent(new Event("tabchange"));
    }));
  }
  // a labelled slider: returns {el, input, value()}
  function slider(label, min, max, step, value, fmt, oninput) {
    const val = h("b");
    const inp = h("input", { type: "range", min, max, step, value, "aria-label": label });
    const upd = () => { val.textContent = fmt ? fmt(+inp.value) : inp.value; };
    inp.addEventListener("input", () => { upd(); oninput && oninput(+inp.value); });
    upd();
    return { el: h("label", null, label + ": ", val, inp), input: inp, get value() { return +inp.value; }, set value(v) { inp.value = v; upd(); } };
  }
  // seeded random numbers (mulberry32) so results are reproducible
  function rng(seed) {
    let a = seed >>> 0;
    const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    r.normal = () => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return r;
  }
  // line chart into an <svg viewBox="0 0 W H">: series = [{xs, ys, color, label, dash, width}]
  function plot(svg, series, o = {}) {
    const W = o.W || 700, H = o.H || 280, L = o.L ?? 48, R = o.R ?? 12, T = o.T ?? 14, B = o.B ?? 34;
    const xs = series.flatMap((d) => d.xs), ys = series.flatMap((d) => d.ys).filter(Number.isFinite);
    const xmin = o.xmin ?? Math.min(...xs), xmax = o.xmax ?? Math.max(...xs);
    let ymin = o.ymin ?? Math.min(...ys), ymax = o.ymax ?? Math.max(...ys);
    if (ymax === ymin) { ymax += 1; ymin -= 1; }
    const logx = !!o.logx, fx = (x) => (logx ? Math.log10(x) : x);
    const X = (x) => L + (W - L - R) * (fx(x) - fx(xmin)) / (fx(xmax) - fx(xmin) || 1);
    const Y = (y) => T + (H - T - B) * (ymax - y) / (ymax - ymin);
    const parts = [];
    const nice = (lo, hi, n) => { const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)); const st = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((k) => span / k <= n) || mag * 10; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10)); return out; };
    nice(ymin, ymax, o.yticks || 5).forEach((v) => { parts.push(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "ax" })); parts.push(s("text", { x: L - 6, y: Y(v) + 4, class: "tx", "text-anchor": "end" }, o.yfmt ? o.yfmt(v) : +v.toPrecision(4))); });
    const xt = logx ? (() => { const out = []; for (let e = Math.ceil(Math.log10(xmin)); e <= Math.log10(xmax) + 1e-9; e++) out.push(10 ** e); return out; })() : nice(xmin, xmax, o.xticks || 6);
    xt.forEach((v) => parts.push(s("text", { x: X(v), y: H - B + 16, class: "tx", "text-anchor": "middle" }, o.xfmt ? o.xfmt(v) : +v.toPrecision(4))));
    if (o.xlabel) parts.push(s("text", { x: (L + W - R) / 2, y: H - 3, class: "tx", "text-anchor": "middle" }, o.xlabel));
    if (o.ylabel) parts.push(s("text", { x: 12, y: (T + H - B) / 2, class: "tx", "text-anchor": "middle", transform: `rotate(-90 12 ${(T + H - B) / 2})` }, o.ylabel));
    (o.bands || []).forEach((b) => parts.push(s("rect", { x: X(b.x0), y: T, width: Math.max(0, X(b.x1) - X(b.x0)), height: H - T - B, fill: b.color, opacity: b.opacity ?? 0.12 })));
    series.forEach((d) => {
      const pts = d.xs.map((x, i) => [X(x), Y(Math.max(ymin, Math.min(ymax, d.ys[i])))]).filter((p) => Number.isFinite(p[1]));
      if (d.dots) pts.forEach((p) => parts.push(s("circle", { cx: p[0], cy: p[1], r: d.r || 3, fill: d.color })));
      else parts.push(s("path", { d: pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(""), fill: "none", stroke: d.color, "stroke-width": d.width || 2.2, "stroke-dasharray": d.dash || null }));
    });
    let ly = T + 4;
    series.filter((d) => d.label).forEach((d) => { parts.push(s("rect", { x: W - R - 150, y: ly, width: 12, height: 3, fill: d.color })); parts.push(s("text", { x: W - R - 133, y: ly + 5, class: "tx" }, d.label)); ly += 15; });
    svg.replaceChildren(...parts);
    return { X, Y };
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  window.PG = { $, h, s, tabs, slider, rng, plot, css };
})();
