/* Interactive figures of the project website. Data: window.SITE (data/site-data.js), written by
   cwm/export_site_data.py from results/ and the filled paper. No library: SVG for charts, canvas for maps. */
(function () {
  "use strict";
  var S = window.SITE;
  if (!S) { return; }
  var NS = "http://www.w3.org/2000/svg";
  var SYS = { calendar: "Google Calendar", gmail: "Gmail", gong: "Gong", jira: "Jira", salesforce: "Salesforce",
              slack: "Slack", zendesk: "Zendesk" };
  var SYS_SHORT = { calendar: "Calendar", gmail: "Gmail", gong: "Gong", jira: "Jira", salesforce: "Salesforce",
                    slack: "Slack", zendesk: "Zendesk" };
  var LANES = ["gmail", "slack", "salesforce", "zendesk", "gong", "calendar", "jira"];
  function quarterName(q) { return q.slice(0, 4) + " Q" + q.slice(-1); }
  var REDUCE = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function $(id) { return document.getElementById(id); }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) { if (attrs[k] !== undefined && attrs[k] !== null) { e.setAttribute(k, attrs[k]); } }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function text(parent, x, y, s, attrs) {
    var t = svg("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function clear(e) { while (e.firstChild) { e.removeChild(e.firstChild); } }
  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  /* "0.951 ± 0.003" -> {v, sd}; "+12 [−14, +29]" -> {v, lo, hi}; "−5%" -> {v}; "--" -> null */
  function num(s) {
    if (s === undefined || s === null) { return null; }
    var t = String(s).replace(/−/g, "-").replace(/ /g, "").replace(/%/g, "").replace(/\*/g, "");
    var m = t.match(/^\s*([+-]?\d*\.?\d+)\s*(?:±\s*(\d*\.?\d+))?\s*(?:\[\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\])?/);
    if (!m) { return null; }
    var r = { v: parseFloat(m[1]) };
    if (m[2]) { r.sd = parseFloat(m[2]); }
    if (m[3]) { r.lo = parseFloat(m[3]); r.hi = parseFloat(m[4]); }
    return r;
  }
  function rows(label, part) {
    var t = S.tables[label];
    if (!t) { return []; }
    return t.parts[part || 0].filter(function (r) { return r !== "rule"; });
  }
  function bodyRows(label, part) {
    var all = S.tables[label].parts[part || 0], out = [], seen = false;
    all.forEach(function (r) { if (r === "rule") { seen = true; } else if (seen) { out.push(r); } });
    return out;
  }
  function cell(r, i) { return r[i] ? r[i].text : ""; }
  function pct(v, d) { return (100 * v).toFixed(d === undefined ? 0 : d) + "%"; }

  /* ------------------------------------------------------------------ links */
  if (/github\.io$/.test(location.hostname)) {
    document.querySelectorAll('[data-link="paper"]').forEach(function (a) { a.setAttribute("href", "paper.pdf"); });
  }

  /* ------------------------------------------------------------------ canvas helper */
  function fitCanvas(c, aspect) {
    var w = c.clientWidth || c.parentNode.clientWidth || 600, h = Math.round(w * aspect);
    var r = window.devicePixelRatio || 1;
    c.width = Math.round(w * r); c.height = Math.round(h * r);
    c.style.height = h + "px";
    var g = c.getContext("2d");
    g.setTransform(r, 0, 0, r, 0, 0);
    return { g: g, w: w, h: h };
  }

  /* ------------------------------------------------------------------ hero figure */
  var H = S.hero, M = S.map;
  var heroStep = 4, heroTimer = null;
  var STEPS = [
    "Seven systems wrote " + H.records.length + " records about Customer " + H.customer + " in " + quarterName(H.quarter) +
      ". Each line is one record, at the time that the system wrote it.",
    "Each system links its records to the customer. Two records of the same customer and quarter are two views of " +
      "one situation, from one system or from two. We call them good views.",
    "The world model learns to predict one view from the other. It never reads the links. The links only choose " +
      "the pairs.",
    "The world model places each record by the situation that it describes. The records of this customer and quarter " +
      "land close together, whichever system wrote them.",
    "For comparison, bad views pair a record with a damaged copy of itself. Then the same records spread out. Only " +
      "good views gather the records of a customer, whatever system wrote them."
  ];
  var heroPairs = (function () {
    var recs = H.records.map(function (r, i) { return { i: i, lane: LANES.indexOf(M.systems[r[0]]), t: r[1] }; });
    var out = [], used = {};
    var seed = 7;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    for (var k = 0; k < 600 && out.length < 18; k++) {
      var a = recs[Math.floor(rnd() * recs.length)], b = recs[Math.floor(rnd() * recs.length)];
      if (a.i === b.i || used[a.i] || used[b.i] || Math.abs(a.t - b.t) > 0.35) { continue; }
      if (a.lane === b.lane && Math.abs(a.t - b.t) < 0.05) { continue; }   // a pair in one lane must be visible
      used[a.i] = used[b.i] = 1;
      out.push([a, b]);
    }
    return out;
  })();

  function drawLanes() {
    var root = $("lanes");
    if (!root) { return; }
    clear(root);
    var W = Math.round(root.getBoundingClientRect().width) || 700, narrow = W < 560;
    var Hh = narrow ? 300 : 330, L = narrow ? 100 : 152, R = 14, T = 14, B = 34, lh = (Hh - T - B) / LANES.length;
    root.setAttribute("viewBox", "0 0 " + W + " " + Hh);
    var ink2 = css("--ink-2"), ink3 = css("--ink-3"), rule = css("--rule"), sig = css("--signal");
    var counts = {};
    H.records.forEach(function (r) { var k = M.systems[r[0]]; counts[k] = (counts[k] || 0) + 1; });
    LANES.forEach(function (k, j) {
      var y = T + j * lh + lh / 2;
      text(root, 8, y + 4, (narrow ? SYS_SHORT : SYS)[k], { class: "label", style: "font-size:13px;fill:" + css("--ink") });
      text(root, L - 10, y + 4, String(counts[k] || 0), { "text-anchor": "end", style: "font-size:12px;fill:" + ink3 });
      svg("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: rule, "stroke-width": 1 }, root);
    });
    // months
    var start = new Date(H.start * 1000), end = new Date(H.end * 1000);
    var d = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 1));
    var names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    function xt(f) { return L + f * (W - L - R); }
    while (d < end) {
      var f = (d / 1000 - H.start) / (H.end - H.start);
      svg("line", { x1: xt(f), x2: xt(f), y1: T, y2: Hh - B + 4, stroke: rule, "stroke-dasharray": "2 3" }, root);
      text(root, xt(f) + 4, Hh - B + 18, names[d.getUTCMonth()] + " " + d.getUTCFullYear(), { style: "font-size:12px;fill:" + ink3 });
      d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
    }
    var colour = heroStep >= 2 ? sig : ink2;
    var g = svg("g", { "stroke-width": 1.4, "stroke-linecap": "round" }, root);
    H.records.forEach(function (r) {
      var j = LANES.indexOf(M.systems[r[0]]), y = T + j * lh + lh / 2, x = xt(r[1]);
      svg("line", { x1: x, x2: x, y1: y - 8, y2: y + 8, stroke: colour, "stroke-opacity": heroStep >= 2 ? 0.75 : 0.6 }, g);
    });
    if (heroStep >= 2 && heroStep <= 4) {
      var arcs = svg("g", { fill: "none", stroke: sig, "stroke-width": 1.6 }, root);
      heroPairs.forEach(function (p, n) {
        var a = p[0], b = p[1];
        var ya = T + a.lane * lh + lh / 2, yb = T + b.lane * lh + lh / 2, xa = xt(a.t), xb = xt(b.t);
        var d = a.lane === b.lane
          ? "M" + xa + "," + ya + " Q" + (xa + xb) / 2 + "," + (ya - lh * 0.55) + " " + xb + "," + yb     // same system: an arch
          : "M" + xa + "," + ya + " C" + (xa + 30) + "," + ya + " " + (xb - 30) + "," + yb + " " + xb + "," + yb;
        var path = svg("path", { d: d, "stroke-opacity": 0.55 }, arcs);
        if (heroStep === 3 && !REDUCE) {
          path.setAttribute("stroke-dasharray", "4 6");
          path.style.animation = "flow 1.4s linear infinite";
          path.style.animationDelay = (n * 0.08) + "s";
        }
      });
    }
  }
  (function () {
    var st = document.createElement("style");
    st.textContent = "@keyframes flow { to { stroke-dashoffset: -20; } }";
    document.head.appendChild(st);
  })();

  var heroMapState = { from: null, to: null, t0: 0, anim: null, space: "good" };
  function heroMapCoords(space) { return M.coords[space] || M.coords.good; }
  function drawHeroMap(progress) {
    var c = $("hero-map");
    if (!c) { return; }
    var f = fitCanvas(c, c.parentNode.clientWidth < 500 ? 0.75 : 330 / 420), g = f.g;
    g.clearRect(0, 0, f.w, f.h);
    var A = heroMapState.from || heroMapCoords(heroMapState.space), B = heroMapState.to || A, p = progress === undefined ? 1 : progress;
    var pad = 18, sx = (f.w - 2 * pad) / 1000, sy = (f.h - 2 * pad - 14) / 1000;
    var mute = css("--mute"), sig = css("--signal");
    var hl = {};
    H.records.forEach(function (r) { if (r[2] >= 0) { hl[r[2]] = 1; } });
    function pos(i) {
      var x = A[2 * i] + (B[2 * i] - A[2 * i]) * p, y = A[2 * i + 1] + (B[2 * i + 1] - A[2 * i + 1]) * p;
      return [pad + x * sx, pad + 14 + y * sy];
    }
    g.fillStyle = mute;
    g.globalAlpha = heroStep >= 4 ? 0.9 : 0.55;
    for (var i = 0; i < M.sample.system.length; i++) {
      if (hl[i] && heroStep >= 4) { continue; }
      var q = pos(i); g.fillRect(q[0] - 1, q[1] - 1, 2, 2);
    }
    g.globalAlpha = 1;
    if (heroStep >= 4) {
      g.fillStyle = sig;
      g.strokeStyle = css("--paper");
      g.lineWidth = 1;
      Object.keys(hl).forEach(function (k) {
        var q = pos(+k); g.beginPath(); g.arc(q[0], q[1], 3.4, 0, 6.2832); g.fill(); g.stroke();
      });
    }
    $("hero-map-label").textContent = (heroStep === 5 ? "Bad views" : "World model") +
      (heroStep >= 4 ? ". Blue: Customer " + H.customer + " in " + quarterName(H.quarter) : "");
  }
  function animateHeroMap(toSpace) {
    var from = heroMapCoords(heroMapState.space), to = heroMapCoords(toSpace);
    heroMapState.space = toSpace;
    if (REDUCE || from === to) { heroMapState.from = heroMapState.to = null; drawHeroMap(); return; }
    heroMapState.from = from; heroMapState.to = to;
    var t0 = performance.now();
    cancelAnimationFrame(heroMapState.anim);
    (function frame(now) {
      var p = Math.min(1, (now - t0) / 900);
      drawHeroMap(ease(p));
      if (p < 1) { heroMapState.anim = requestAnimationFrame(frame); }
      else { heroMapState.from = heroMapState.to = null; }
    })(t0);
  }
  function setStep(n) {
    heroStep = Math.max(1, Math.min(5, n));
    $("step-count").textContent = "Step " + heroStep + " of 5";
    $("step-text").textContent = STEPS[heroStep - 1];
    drawLanes();
    animateHeroMap(heroStep === 5 ? "bad" : "good");
    $("step-prev").disabled = heroStep === 1;
    $("step-next").disabled = heroStep === 5;
  }
  function stopPlay() { clearInterval(heroTimer); heroTimer = null; $("step-play").textContent = "Play"; }
  if ($("lanes")) {
    $("step-prev").addEventListener("click", function () { stopPlay(); setStep(heroStep - 1); });
    $("step-next").addEventListener("click", function () { stopPlay(); setStep(heroStep + 1); });
    $("step-play").addEventListener("click", function () {
      if (heroTimer) { stopPlay(); return; }
      setStep(1);
      $("step-play").textContent = "Pause";
      heroTimer = setInterval(function () {
        if (heroStep >= 5) { stopPlay(); return; }
        setStep(heroStep + 1);
      }, 3400);
    });
    setStep(4);
  }

  /* ------------------------------------------------------------------ figure 2: the map */
  var mapState = { space: "good", mode: "customer", pick: H.customer, from: null, anim: null, pts: null, hover: -1 };
  function mapSetup() {
    var segS = $("map-space");
    if (!segS) { return; }
    M.spaces.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = s[1]; b.dataset.space = s[0];
      b.setAttribute("aria-pressed", s[0] === mapState.space ? "true" : "false");
      b.addEventListener("click", function () { mapSpace(s[0]); });
      segS.appendChild(b);
    });
    $("map-mode").querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        mapState.mode = b.dataset.mode;
        $("map-mode").querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        mapState.pick = mapState.mode === "customer" ? H.customer : M.systems.indexOf("gmail");
        fillPick(); drawMap();
      });
    });
    $("map-pick").addEventListener("change", function () { mapState.pick = +this.value; drawMap(); });
    fillPick();
    var c = $("map");
    c.addEventListener("pointermove", function (e) { if (e.pointerType !== "touch") { mapHover(e, 10); } });
    c.addEventListener("click", function (e) { mapHover(e, 22); });
    c.addEventListener("pointerleave", function (e) {
      if (e.pointerType !== "touch") { mapState.hover = -1; $("map-tip").classList.remove("on"); drawMap(); }
    });
    drawMap();
  }
  function fillPick() {
    var sel = $("map-pick");
    clear(sel);
    if (mapState.mode === "customer") {
      var n = {};
      M.sample.customer.forEach(function (x) { n[x] = (n[x] || 0) + 1; });
      Object.keys(n).map(Number).sort(function (a, b) { return a - b; }).forEach(function (cu) {
        var o = document.createElement("option");
        o.value = cu; o.textContent = "Customer " + cu + " (" + n[cu] + " records)";
        if (cu === mapState.pick) { o.selected = true; }
        sel.appendChild(o);
      });
    } else {
      M.systems.forEach(function (s, i) {
        var o = document.createElement("option");
        var cnt = M.sample.system.filter(function (x) { return x === i; }).length;
        o.value = i; o.textContent = SYS[s] + " (" + cnt.toLocaleString("en-US") + " records)";
        if (i === mapState.pick) { o.selected = true; }
        sel.appendChild(o);
      });
    }
  }
  function mapSpace(k) {
    if (!M.coords[k] || k === mapState.space) { return; }
    $("map-space").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.space === k ? "true" : "false"); });
    var from = M.coords[mapState.space];
    mapState.space = k;
    if (REDUCE) { drawMap(); return; }
    var t0 = performance.now();
    cancelAnimationFrame(mapState.anim);
    (function frame(now) {
      var p = Math.min(1, (now - t0) / 900);
      drawMap(from, ease(p));
      if (p < 1) { mapState.anim = requestAnimationFrame(frame); }
    })(t0);
  }
  function drawMap(from, p) {
    var c = $("map");
    if (!c) { return; }
    var f = fitCanvas(c, c.clientWidth < 600 ? 0.9 : 0.5), g = f.g;
    g.clearRect(0, 0, f.w, f.h);
    var B = M.coords[mapState.space], A = from || B, q = p === undefined ? 1 : p;
    var pad = 16, sx = (f.w - 2 * pad) / 1000, sy = (f.h - 2 * pad) / 1000;
    var n = M.sample.system.length, pts = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      pts[2 * i] = pad + (A[2 * i] + (B[2 * i] - A[2 * i]) * q) * sx;
      pts[2 * i + 1] = pad + (A[2 * i + 1] + (B[2 * i + 1] - A[2 * i + 1]) * q) * sy;
    }
    mapState.pts = pts; mapState.w = f.w;
    var key = mapState.mode === "customer" ? M.sample.customer : M.sample.system;
    var hl = mapState.mode === "customer" ? css("--signal") : css("--ink");
    g.fillStyle = css("--mute");
    for (i = 0; i < n; i++) { if (key[i] !== mapState.pick) { g.beginPath(); g.arc(pts[2 * i], pts[2 * i + 1], 2, 0, 6.2832); g.fill(); } }
    g.fillStyle = hl; g.strokeStyle = css("--paper"); g.lineWidth = 1;
    var count = 0;
    for (i = 0; i < n; i++) {
      if (key[i] === mapState.pick) { count++; g.beginPath(); g.arc(pts[2 * i], pts[2 * i + 1], 3.6, 0, 6.2832); g.fill(); g.stroke(); }
    }
    if (mapState.hover >= 0) {
      g.strokeStyle = css("--ink"); g.lineWidth = 2;
      g.beginPath(); g.arc(pts[2 * mapState.hover], pts[2 * mapState.hover + 1], 6, 0, 6.2832); g.stroke();
    }
    var name = mapState.mode === "customer" ? "Customer " + mapState.pick : SYS[M.systems[mapState.pick]];
    var leg = $("map-legend");
    leg.innerHTML = "";
    var a = document.createElement("span");
    a.innerHTML = '<i class="key dot" style="background:' + hl + '"></i>' + name + ": " + count.toLocaleString("en-US") + " records";
    var b = document.createElement("span");
    b.innerHTML = '<i class="key dot" style="background:' + css("--mute") + '"></i>The other ' + (n - count).toLocaleString("en-US") + " records";
    leg.appendChild(a); leg.appendChild(b);
  }
  function placeTip(tip, x, y, w) {
    tip.style.left = "0px"; tip.style.top = "0px";
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.max(tw / 2 + 4, Math.min(w - tw / 2 - 4, x)) + "px";
    tip.style.top = y + "px";
    tip.classList.toggle("below", y - th - 12 < 0);
    tip.classList.add("on");
  }
  function mapHover(e, radius) {
    var c = $("map"), r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, pts = mapState.pts;
    if (!pts) { return; }
    var best = -1, bd = radius * radius;
    for (var i = 0; i < pts.length / 2; i++) {
      var dx = pts[2 * i] - x, dy = pts[2 * i + 1] - y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    var tip = $("map-tip");
    if (best !== mapState.hover) { mapState.hover = best; drawMap(); }
    if (best < 0) { tip.classList.remove("on"); return; }
    tip.textContent = SYS[M.systems[M.sample.system[best]]] + " record, Customer " + M.sample.customer[best] + ", " +
      quarterName(M.quarters[M.sample.quarter[best]]);
    placeTip(tip, pts[2 * best], pts[2 * best + 1], c.clientWidth);
  }

  /* ------------------------------------------------------------------ generic horizontal charts */
  /* Each chart draws at the width of its box, so its text keeps the size of the page text. Label and value columns
     take the width of their longest text. When the plot would be too narrow, each label goes above its row. */
  function scaleX(lo, hi, x0, x1) { return function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); }; }
  function boxWidth(root, fallback) { var w = Math.round(root.getBoundingClientRect().width); return w > 0 ? w : fallback; }
  function textWidth(root, strings, attrs) {
    var t = svg("text", Object.assign({ x: 0, y: -50 }, attrs || {}), root), max = 0;
    strings.forEach(function (s) { if (s) { t.textContent = s; max = Math.max(max, t.getComputedTextLength()); } });
    root.removeChild(t);
    return max;
  }
  function axis(root, x, ticks, y0, y1, fmt) {
    var g = svg("g", { class: "grid" }, root);
    ticks.forEach(function (t) {
      svg("line", { x1: x(t), x2: x(t), y1: y0, y2: y1 }, g);
      text(root, x(t), y1 + 16, fmt(t), { "text-anchor": "middle" });
    });
  }
  /* reference lines; their labels go above the plot, on a second line where two labels would touch */
  function layoutRefs(root, refs, x, xmin, xmax) {
    var ends = [-1e9, -1e9], lines = 0;
    refs.slice().sort(function (a, b) { return a.v - b.v; }).forEach(function (rf) {
      var w = textWidth(root, [rf.label], { style: "font-size:12px" });
      rf.cx = Math.max(xmin + w / 2, Math.min(xmax - w / 2, x(rf.v)));
      rf.line = rf.cx - w / 2 > ends[0] + 8 ? 0 : 1;
      ends[rf.line] = rf.cx + w / 2;
      lines = Math.max(lines, rf.line + 1);
    });
    return lines;
  }
  function drawRefs(root, refs, x, top, bottom) {
    refs.forEach(function (rf) {
      svg("line", { x1: x(rf.v), x2: x(rf.v), y1: top, y2: bottom, stroke: css("--ink-3"),
                    "stroke-dasharray": rf.solid ? null : "4 3", "stroke-width": rf.solid ? 1.5 : 1 }, root);
      text(root, rf.cx, top - 7 - rf.line * 15, rf.label, { "text-anchor": "middle", class: "halo", style: "font-size:12px" });
    });
  }
  function barChart(id, data, opt) {
    var root = $(id);
    if (!root) { return; }
    clear(root);
    var W = boxWidth(root, 520);
    var lw = textWidth(root, data.map(function (d) { return d.label; }), { class: "label" });
    var R = textWidth(root, ["100%"], { class: "value" }) + 16, L = lw ? lw + 20 : (opt.left || 190);
    var stacked = W - L - R < 200;
    if (stacked) { L = textWidth(root, ["0%"], {}) / 2 + 4; }
    var rowH = stacked ? 46 : 30, x = scaleX(0, 1, L, W - R);
    var refs = (opt.refs || []).slice(), lines = Math.max(layoutRefs(root, refs, x, L, W - R), opt.headLines || 0);
    var top = 6 + lines * 15, Hh = top + data.length * rowH + 30;
    root.setAttribute("viewBox", "0 0 " + W + " " + Hh);
    axis(root, x, [0, 0.25, 0.5, 0.75, 1], top, Hh - 26, function (t) { return pct(t); });
    drawRefs(root, refs, x, top, Hh - 26);
    data.forEach(function (d, i) {
      var y0 = top + i * rowH, yb = stacked ? y0 + 22 : y0 + 7, bh = 16;
      if (stacked) { text(root, L, y0 + 15, d.label, { class: "label halo" }); }
      else { text(root, L - 10, y0 + rowH / 2 + 4, d.label, { "text-anchor": "end", class: "label" }); }
      svg("rect", { x: L, y: yb, width: Math.max(1, x(d.v) - L), height: bh, rx: 3, fill: d.color }, root);
      text(root, x(d.v) + 6, yb + bh / 2 + 4.5, pct(d.v, opt.digits !== undefined ? opt.digits : (d.v < 0.1 ? 1 : 0)), { class: "value" });
    });
  }
  function repColour(name) {
    var n = name.toLowerCase();
    if (n.indexOf("head") >= 0 || n.indexOf("pretrained") >= 0) { return css("--pre"); }
    if (n.indexOf("good") >= 0) { return css("--signal"); }
    if (n.indexOf("bad") >= 0) { return css("--bad"); }
    if (n.indexOf("next-token") >= 0) { return css("--ntp"); }
    return css("--mute-2");
  }
  function shortRep(name) {
    return name.replace("TF-IDF + LSA", "TF-IDF").replace("Pretrained encoder, frozen", "Pretrained, frozen")
      .replace(/^\s*\+\s*good-view head/, "Pretrained + good-view head").replace("Next-token", "Next-token prediction");
  }
  function drawBars() {
    var e1 = bodyRows("tab:e1").map(function (r) {
      var nm = shortRep(cell(r, 0));
      return { label: nm, v: num(cell(r, 1)).v, color: repColour(nm) };
    });
    barChart("bars-e1", e1, { headLines: 1, digits: 1 });   // the table has three decimals: one decimal of a percent avoids rounding twice
    var ho = bodyRows("tab:heldout").filter(function (r) { return cell(r, 1) !== ""; }).map(function (r) {
      var nm = shortRep(cell(r, 0));
      return { label: nm, v: num(cell(r, 3)).v, color: repColour(nm) };
    });
    var largest = bodyRows("tab:heldout").filter(function (r) { return /largest/i.test(cell(r, 0)); })[0];
    barChart("bars-ho", ho, { refs: largest ? [{ v: num(cell(largest, 3)).v, label: "largest customer" }] : [] });
  }

  /* dots with an optional interval: rows [{label, marks:[{v, lo, hi, color, ring}], strong, value, rule}] */
  function dotChart(id, data, opt) {
    var root = $(id);
    if (!root) { return; }
    clear(root);
    var W = boxWidth(root, opt.width || 1040);
    var lw = Math.max(textWidth(root, data.filter(function (d) { return !d.strong; }).map(function (d) { return d.label; }), { class: "label" }),
                      textWidth(root, data.filter(function (d) { return d.strong; }).map(function (d) { return d.label; }),
                                { class: "label", style: "font-weight:700" }));
    var vw = textWidth(root, data.map(function (d) { return d.value || ""; }).concat([opt.valueTitle || ""]), { class: "value" });
    var R = vw ? vw + 24 : 18, L = lw ? lw + 24 : (opt.left || 300);
    var stacked = W - L - R < (opt.minPlot || 240);
    if (stacked) { L = Math.max(12, textWidth(root, [opt.fmt(opt.ticks[0])], {}) / 2 + 4); }
    var rowH = (opt.rowH || 30) + (stacked ? 20 : 0), x = scaleX(opt.lo, opt.hi, L, W - R);
    var refs = (opt.refs || []).slice();
    if (opt.zero !== undefined && opt.zeroLabel) { refs.push({ v: opt.zero, label: opt.zeroLabel, solid: true }); }
    var lines = layoutRefs(root, refs, x, L, W - R);
    var top = (opt.top || 8) + Math.max(lines * 15, opt.valueTitle ? 14 : 0), Hh = top + data.length * rowH + 30;
    root.setAttribute("viewBox", "0 0 " + W + " " + Hh);
    axis(root, x, opt.ticks, top, Hh - 26, opt.fmt);
    if (opt.valueTitle) { text(root, W - R + 12, top - 7, opt.valueTitle, { style: "font-size:12px" }); }
    if (opt.zero !== undefined && !opt.zeroLabel) {
      svg("line", { x1: x(opt.zero), x2: x(opt.zero), y1: top, y2: Hh - 26, class: "zero", "stroke-width": 1.5 }, root);
    }
    drawRefs(root, refs, x, top, Hh - 26);
    data.forEach(function (d, i) {
      var y0 = top + i * rowH, y = stacked ? y0 + 32 : y0 + rowH / 2;
      if (d.rule) { svg("line", { x1: 8, x2: W - 8, y1: y0, y2: y0, stroke: css("--rule") }, root); }
      var st = d.strong ? "font-weight:700" : "";
      if (stacked) { text(root, Math.max(L, 12), y0 + 15, d.label, { class: "label halo", style: st }); }
      else { text(root, L - 12, y + 4, d.label, { "text-anchor": "end", class: "label", style: st }); }
      if (d.marks.length === 2 && opt.link) {
        svg("line", { x1: x(d.marks[0].v), x2: x(d.marks[1].v), y1: y, y2: y, stroke: css("--mute-2"), "stroke-width": 2 }, root);
      }
      d.marks.forEach(function (m) {
        if (m.lo !== undefined) {
          svg("line", { x1: x(m.lo), x2: x(m.hi), y1: y, y2: y, stroke: m.color, "stroke-width": 2, "stroke-opacity": 0.6 }, root);
        }
        svg("circle", { cx: x(m.v), cy: y, r: m.r || 5.5, fill: m.ring ? css("--paper") : m.color, stroke: m.ring ? m.color : css("--paper"),
                        "stroke-width": m.ring ? 2 : 1.2 }, root);
      });
      if (d.value) { text(root, W - R + 12, y + 4, d.value, { class: "value" }); }
    });
  }

  /* ------------------------------------------------------------------ figure 4 and 5: access */
  var leakReader = null;
  function drawLeak() {
    var body = bodyRows("tab:acl"), groups = [], cur = null;
    body.forEach(function (r) {
      if (cell(r, 0)) { cur = { reader: cell(r, 0), rows: [] }; groups.push(cur); }
      cur.rows.push(r);
    });
    var seg = $("leak-reader");
    if (!seg) { return; }
    if (!seg.childNodes.length) {
      groups.forEach(function (gp, k) {
        var b = document.createElement("button");
        b.type = "button"; b.textContent = gp.reader; b.setAttribute("aria-pressed", k === 0 ? "true" : "false");
        b.addEventListener("click", function () {
          leakReader = gp.reader;
          seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
          drawLeak();
        });
        seg.appendChild(b);
      });
      leakReader = groups[0].reader;
    }
    var gp = groups.filter(function (x) { return x.reader === leakReader; })[0];
    var data = [];
    [["Codename search", 2, 3], ["Secret sentence", 4, 5]].forEach(function (m, k) {
      gp.rows.forEach(function (r, j) {
        var w = num(cell(r, m[1])).v, wo = num(cell(r, m[2])).v;
        data.push({ label: m[0] + ": " + cell(r, 1), rule: k === 1 && j === 0, strong: /built in/.test(cell(r, 1)),
                    marks: [{ v: wo, color: css("--mute-2") }, { v: w, color: css("--bad") }],
                    value: Math.abs(w - wo) < 0.005 ? "0.00" : (w > wo ? "+" : "−") + Math.abs(w - wo).toFixed(2) });
      });
    });
    dotChart("leak", data, { lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return t.toFixed(2); },
                             link: true, valueTitle: "leak" });
  }
  function drawRescue() {
    var data = bodyRows("tab:aclrescue").map(function (r) {
      var own = num(cell(r, 2)), bh = num(cell(r, 4)), all = num(cell(r, 6));
      var marks = [];
      if (all) { marks.push({ v: all.v, color: css("--ink"), ring: true, r: 6 }); }
      if (bh) { marks.push({ v: bh.v, color: css("--pre") }); }
      if (own) { marks.push({ v: own.v, color: css("--signal") }); }
      return { label: cell(r, 0) + " (" + cell(r, 1) + " readable)", marks: marks };
    });
    dotChart("rescue", data, { lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return pct(t); } });
  }

  /* ------------------------------------------------------------------ figures 6 and 7: memory */
  var anchorNames = "kept";
  function drawAnchor() {
    var body = bodyRows("tab:anchor"), col = anchorNames === "kept" ? 4 : 2;
    var closed = null, oracle = null, data = [];
    body.forEach(function (r) {
      var v = num(cell(r, col));
      if (!v) { return; }
      var name = cell(r, 0), rd = cell(r, 1);
      if (/closed book/i.test(name)) { closed = v.v; }
      if (/own link/i.test(name) && rd === "detail") { oracle = v.v; }
      var strong = anchorNames === "kept" && /^Placement, state channel/.test(name);   // SI Table 3 of the paper bolds names kept only
      data.push({ label: name + (rd && rd !== "–" ? ", reads " + rd : ""), strong: strong,
                  marks: [{ v: v.v, lo: v.v - (v.sd || 0), hi: v.v + (v.sd || 0), color: strong ? css("--signal") : css("--ink-2") }],
                  value: pct(v.v, 1) });
    });
    var refs = [];
    if (closed !== null) { refs.push({ v: closed, label: "no store" }); }
    if (oracle !== null) { refs.push({ v: oracle, label: "oracle, reads detail" }); }
    dotChart("anchor", data, { lo: 0.45, hi: 0.75, ticks: [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                               fmt: function (t) { return pct(t); }, refs: refs, rowH: 28 });
  }
  function drawMem() {
    var data = bodyRows("tab:memory2").map(function (r) {
      var cb = num(cell(r, 2)), tk = num(cell(r, 3));
      return { label: cell(r, 0).replace("*", ""), marks: [
        { v: cb.v, color: css("--mute-2") },
        { v: tk.v, lo: tk.v - (tk.sd || 0), hi: tk.v + (tk.sd || 0), color: css("--signal") }], value: cell(r, 4) + " pts" };
    });
    dotChart("mem", data, { lo: 0.55, hi: 0.8, ticks: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8], fmt: function (t) { return pct(t); },
                            link: true, valueTitle: "gain" });
  }

  /* ------------------------------------------------------------------ figure 8: time */
  function drawTime() {
    var data = bodyRows("tab:time").filter(function (r) { return num(cell(r, 4)); }).map(function (r) {
      var tuned = num(cell(r, 4)).v, shrink = num(cell(r, 5)).v;
      return { label: cell(r, 0), marks: [{ v: shrink, color: css("--ink-3"), ring: true }, { v: tuned, color: css("--ink") }],
               value: (tuned > 0 ? "+" : "−") + Math.abs(tuned) + "%" };
    });
    dotChart("timechart", data, { lo: -40, hi: 10, ticks: [-40, -30, -20, -10, 0, 10],
                                  fmt: function (t) { return (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t) + "%"; },
                                  zero: 0, zeroLabel: "no change", link: true });
  }

  /* ------------------------------------------------------------------ figures 9 and 10: decisions */
  var D = S.decisions, sim = { i: null, space: "good_month", outcome: "urgent_next" };
  function simSetup() {
    var sel = $("sim-month");
    if (!sel) { return; }
    var opts = D.troubled.slice().sort(function (a, b) {
      var A = D.items[a], B = D.items[b];
      return A.customer - B.customer || (A.month < B.month ? -1 : 1);
    });
    opts.forEach(function (i) {
      var it = D.items[i], o = document.createElement("option");
      o.value = i;
      o.textContent = "Customer " + it.customer + ", " + it.month + (it.intervened ? ", intervened" : ", did not intervene");
      sel.appendChild(o);
    });
    sim.i = opts[0];
    sel.addEventListener("change", function () { sim.i = +this.value; drawSim(); });
    var seg = $("sim-space");
    D.spaces.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = s[1]; b.dataset.space = s[0];
      b.setAttribute("aria-pressed", s[0] === sim.space ? "true" : "false");
      b.addEventListener("click", function () {
        sim.space = s[0];
        seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        drawSim();
      });
      seg.appendChild(b);
    });
    $("sim-outcome").querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        sim.outcome = b.dataset.outcome;
        $("sim-outcome").querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        drawSim();
      });
    });
    drawSim();
  }
  function drawSim() {
    var it = D.items[sim.i];
    var nb = D.neighbours.filter(function (n) { return n.space === sim.space && n.i === sim.i; })[0];
    var oname = sim.outcome === "urgent_next" ? "an urgent or high-priority ticket" : "a deal that advanced";
    var head = $("sim-head");
    head.innerHTML = "";
    var p1 = document.createElement("p");
    p1.innerHTML = "<b>Customer " + it.customer + ", " + it.month + ".</b> An urgent or high-priority ticket arrived during an open deal. " +
      (it.intervened ? "The company intervened with a sales call or an engineering escalation." : "The company did not call or escalate.");
    var p2 = document.createElement("p");
    p2.textContent = "The next month had " + oname + ": " + (it[sim.outcome] ? "yes." : "no.");
    head.appendChild(p1); head.appendChild(p2);
    var box = $("sim-rows");
    box.innerHTML = "";
    [["Similar months in which the company intervened", nb.on], ["Similar months in which it did not intervene", nb.off]].forEach(function (row) {
      var list = row[1], div = document.createElement("div");
      div.className = "sim-row";
      var h = document.createElement("h4");
      h.textContent = row[0] + " (" + list.length + ")";
      var chips = document.createElement("div");
      chips.className = "chips";
      var yes = 0, tr = 0;
      list.forEach(function (j) {
        var o = D.items[j], c = document.createElement("span");
        c.className = "chip" + (o[sim.outcome] ? " yes" : "") + (o.trouble ? " trouble" : "");
        c.title = "Customer " + o.customer + ", " + o.month + ": " + (o[sim.outcome] ? oname + " followed" : "no " + oname.replace(/^an? /, "")) +
          (o.trouble ? "; in trouble that month" : "");
        chips.appendChild(c);
        yes += o[sim.outcome]; tr += o.trouble;
      });
      var rates = document.createElement("div");
      rates.className = "rates";
      rates.innerHTML = list.length ? "Outcome: <b>" + pct(yes / list.length) + "</b><br>In trouble: <b>" + pct(tr / list.length) + "</b>"
                                    : "No similar month";
      div.appendChild(h); div.appendChild(chips); div.appendChild(rates);
      box.appendChild(div);
    });
  }
  function drawSimAll() {
    var body = bodyRows("tab:simulate");
    [["simall-urgent", 6], ["simall-deal", 3]].forEach(function (m) {
      var data = body.map(function (r) {
        var v = num(cell(r, m[1]));
        var nm = cell(r, 0), strong = /^Similar states, good/.test(nm);
        var mk = { v: v.v, color: /Observed/.test(nm) ? css("--ink-3") : (/^Similar/.test(nm) ? css("--signal") : css("--ink-2")) };
        if (v.lo !== undefined) { mk.lo = v.lo; mk.hi = v.hi; }
        if (/Observed/.test(nm)) { mk.ring = true; }
        return { label: nm.replace(/\s*\(k\s*=\s*\d+\)/, ""), marks: [mk], strong: strong, value: (v.v > 0 ? "+" : v.v < 0 ? "−" : "") + Math.abs(v.v) };
      });
      dotChart(m[0], data, { lo: -40, hi: 60, ticks: [-40, -20, 0, 20, 40, 60],
                             fmt: function (t) { return (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t); },
                             width: 560, zero: 0, rowH: 28, minPlot: 180 });
    });
  }

  /* ------------------------------------------------------------------ the paper's tables */
  function renderTable(label) {
    var t = S.tables[label];
    if (!t) { return null; }
    /* the caption stays outside the scrolling box, so it wraps at the width of the page */
    var fig = document.createElement("figure");
    fig.className = "table-block";
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<b>" + (t.prefix || "Table") + " " + (t.number || "") + ".</b> ";
    cap.appendChild(document.createTextNode(t.caption));
    fig.appendChild(cap);
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", (t.prefix || "Table") + " " + (t.number || ""));
    t.parts.forEach(function (part) {
      var tb = document.createElement("table");
      tb.className = "paper";
      var thead = document.createElement("thead"), tbody = document.createElement("tbody"), inBody = false, ruleNext = false;
      part.forEach(function (r) {
        if (r === "rule") { if (inBody) { ruleNext = true; } inBody = true; return; }
        var tr = document.createElement("tr");
        if (ruleNext) { tr.className = "rule"; ruleNext = false; }
        r.forEach(function (c) {
          var td = document.createElement(inBody ? "td" : "th");
          td.textContent = c.text;
          if (c.b && inBody) { td.className = "b"; }
          if (c.span > 1) { td.colSpan = c.span; td.style.textAlign = "center"; }
          tr.appendChild(td);
        });
        (inBody ? tbody : thead).appendChild(tr);
      });
      tb.appendChild(thead); tb.appendChild(tbody);
      wrap.appendChild(tb);
    });
    fig.appendChild(wrap);
    return fig;
  }
  document.querySelectorAll("details.numbers").forEach(function (d) {
    d.addEventListener("toggle", function () {
      if (!d.open || d.dataset.done) { return; }
      d.dataset.done = "1";
      d.dataset.tables.split(" ").forEach(function (lab) { var e = renderTable(lab); if (e) { d.appendChild(e); } });
    });
  });

  /* ------------------------------------------------------------------ BibTeX and navigation */
  var copy = $("copy-bib");
  if (copy) {
    copy.addEventListener("click", function () {
      var txt = $("bibtex").textContent;
      function done(ok) { copy.textContent = ok ? "Copied" : "Select the text to copy"; setTimeout(function () { copy.textContent = "Copy"; }, 2200); }
      try { navigator.clipboard.writeText(txt).then(function () { done(true); }, function () { done(false); }); }
      catch (e) { done(false); }
    });
  }
  if ("IntersectionObserver" in window) {
    var links = {};
    document.querySelectorAll(".nav a").forEach(function (a) {
      (a.getAttribute("data-spy") || a.getAttribute("href").slice(1)).split(" ").forEach(function (id) { links[id] = a; });
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) { return; }
        Object.keys(links).forEach(function (k) { links[k].removeAttribute("aria-current"); });
        if (links[en.target.id]) { links[en.target.id].setAttribute("aria-current", "true"); }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("main section[id]").forEach(function (s) { io.observe(s); });
  }

  /* ------------------------------------------------------------------ draw everything */
  function drawAll() {
    if ($("lanes")) { drawLanes(); drawHeroMap(); }
    drawMap(); drawBars(); drawLeak(); drawRescue(); drawAnchor(); drawMem(); drawTime(); drawSimAll();
  }
  mapSetup();
  simSetup();
  var seg = $("anchor-names");
  if (seg) {
    seg.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        anchorNames = b.dataset.names;
        seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        drawAnchor();
      });
    });
  }
  drawAll();
  if (document.fonts && document.fonts.ready) { document.fonts.ready.then(drawAll); }
  var rt = null;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) { mq.addEventListener("change", drawAll); }
  }
  new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
