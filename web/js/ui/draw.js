// Canvas drawing for the lanes: piano-roll notes, drum rows, waveforms, the time ruler and the playhead.
// Canvases are viewport-sized (never as wide as the song) and drawn in CSS pixels on a devicePixelRatio-scaled context.

import { drumName, pitchName } from "./gm.js";

/** Track colours ([r, g, b]); drums get their own. */
export const PALETTE = [
  [122, 162, 255],
  [255, 145, 92],
  [245, 198, 81],
  [110, 231, 183],
  [192, 132, 252],
  [56, 189, 248],
  [244, 114, 182],
  [163, 230, 53],
  [251, 113, 133],
  [94, 234, 212],
];
export const DRUM_COLOR = [255, 92, 138];
export const ORIGINAL_COLOR = [154, 164, 186];
export const RESIDUAL_COLOR = [120, 128, 146];

/** CSS rgba() of an [r, g, b] colour. */
export const rgba = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;
const lighten = ([r, g, b], f) => [r + (255 - r) * f, g + (255 - g) * f, b + (255 - b) * f].map(Math.round);

/**
 * Size a canvas backing store to its CSS box at the device pixel ratio; returns a context in CSS pixels.
 * @param {HTMLCanvasElement} canvas
 * @param {number} cssW
 * @param {number} cssH
 */
export function fitCanvas(canvas, cssW, cssH) {
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

/** Grid step (seconds) giving at least `minPx` between lines. */
export function gridStep(pps, minPx = 80) {
  for (const s of [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]) if (s * pps >= minPx) return s;
  return 600;
}

/** "1:05" / "1:05.3" */
export function fmtTime(t, decimals = 0) {
  const s = Math.max(0, t);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  const sec = decimals ? r.toFixed(decimals).padStart(3 + decimals, "0") : String(Math.floor(r)).padStart(2, "0");
  return `${m}:${sec}`;
}

/**
 * Vertical layout of a note lane: which pitch (or drum key) sits on which row.
 * Pitched tracks span their own pitch range (+ margin); drum lanes get one labelled row per key present.
 * @param {{isDrum: boolean, notes: number[][]}} track
 * @returns {{rows: number[], rowOf: Map<number, number>, rowH: number, height: number, isDrum: boolean}}
 */
export function laneLayout(track) {
  const pitches = [...new Set(track.notes.map((n) => n[2]))].sort((a, b) => b - a);
  if (track.isDrum) {
    // Order top to bottom: cymbals / hats, then snares / toms, kick at the bottom (like a drum chart).
    const rank = (k) =>
      k === 35 || k === 36 ? 0 : [38, 40, 37, 39].includes(k) ? 2 : [41, 43, 45, 47, 48, 50].includes(k) ? 1 : 3;
    const rows = pitches.sort((a, b) => rank(b) - rank(a) || b - a);
    const rowH = 15;
    return {
      rows,
      rowOf: new Map(rows.map((p, i) => [p, i])),
      rowH,
      height: Math.max(48, rows.length * rowH + 6),
      isDrum: true,
    };
  }
  if (!pitches.length) return { rows: [60], rowOf: new Map([[60, 0]]), rowH: 8, height: 64, isDrum: false };
  const hi = pitches[0] + 2;
  const lo = pitches.at(-1) - 2;
  const rows = [];
  for (let p = hi; p >= lo; p--) rows.push(p);
  const rowH = Math.max(3, Math.min(9, Math.round(170 / rows.length)));
  return {
    rows,
    rowOf: new Map(rows.map((p, i) => [p, i])),
    rowH,
    height: Math.max(72, Math.min(210, rows.length * rowH)),
    isDrum: false,
  };
}

/** Index of the first note that may be visible at t0 given the longest note (notes sorted by onset). */
function firstVisible(notes, t0, maxDur) {
  let lo = 0;
  let hi = notes.length;
  const t = t0 - maxDur;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (notes[m][0] < t) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function drawTimeGrid(g, view, w, h) {
  const step = gridStep(view.pps);
  g.lineWidth = 1;
  for (let t = Math.ceil(view.t0 / step) * step; t <= view.t1; t += step) {
    const x = Math.round(view.x(t)) + 0.5;
    const major = Math.abs(t / (step * 4) - Math.round(t / (step * 4))) < 1e-6;
    g.strokeStyle = major ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.035)";
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, h);
    g.stroke();
  }
}

/** Playhead line at time t. */
export function drawPlayhead(g, view, h, t) {
  const x = Math.round(view.x(t)) + 0.5;
  if (x < -2 || x > view.width + 2) return;
  g.fillStyle = "rgba(255,255,255,0.10)";
  g.fillRect(x - 3, 0, 6, h);
  g.strokeStyle = "#ffffff";
  g.lineWidth = 1.25;
  g.beginPath();
  g.moveTo(x, 0);
  g.lineTo(x, h);
  g.stroke();
}

/**
 * Draw one note lane.
 * @param {CanvasRenderingContext2D} g
 * @param {object} o
 * @param {import("./view.js").View} o.view
 * @param {number} o.h lane height (CSS px)
 * @param {{notes: number[][], maxDur: number}} o.track notes sorted by onset
 * @param {ReturnType<typeof laneLayout>} o.layout
 * @param {number[]} o.color
 * @param {number} o.time playhead time
 * @param {boolean} o.highlight highlight notes sounding at `time`
 */
export function drawNoteLane(g, { view, h, track, layout, color, time, highlight }) {
  const w = view.width;
  g.clearRect(0, 0, w, h);
  const { rows, rowH, isDrum } = layout;
  const top = Math.max(0, (h - rows.length * rowH) / 2);
  // Row backgrounds: black keys darker, C rows marked; drum rows alternate.
  rows.forEach((p, i) => {
    const y = top + i * rowH;
    if (isDrum) {
      g.fillStyle = i % 2 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.0)";
      g.fillRect(0, y, w, rowH);
    } else if ([1, 3, 6, 8, 10].includes(p % 12)) {
      g.fillStyle = "rgba(0,0,0,0.22)";
      g.fillRect(0, y, w, rowH);
    }
    if (!isDrum && p % 12 === 0) {
      g.fillStyle = "rgba(255,255,255,0.06)";
      g.fillRect(0, y + rowH - 1, w, 1);
    }
  });
  drawTimeGrid(g, view, w, h);

  const notes = track.notes;
  const t0 = view.t0;
  const t1 = view.t1;
  const pad = rowH >= 6 ? 1 : 0.5;
  const active = [];
  for (let i = firstVisible(notes, t0, track.maxDur); i < notes.length; i++) {
    const n = notes[i];
    if (n[0] > t1) break;
    if (n[1] < t0) continue;
    const r = layout.rowOf.get(n[2]);
    if (r == null) continue;
    const y = top + r * rowH + pad;
    const x = view.x(n[0]);
    const nw = isDrum ? Math.max(4, Math.min(10, (n[1] - n[0]) * view.pps)) : Math.max(2, (n[1] - n[0]) * view.pps);
    const nh = rowH - 2 * pad;
    const on = highlight && time >= n[0] && time < Math.max(n[1], n[0] + 0.12);
    if (on) {
      active.push([x, y, nw, nh]);
      continue;
    }
    const a = 0.32 + 0.6 * ((n[3] ?? 100) / 127);
    g.fillStyle = rgba(color, a);
    roundRect(g, x, y, nw, nh, Math.min(2, nh / 2));
  }
  if (active.length) {
    g.save();
    g.shadowColor = rgba(color, 0.9);
    g.shadowBlur = 10;
    g.fillStyle = rgba(lighten(color, 0.55));
    for (const [x, y, nw, nh] of active) roundRect(g, x, y, nw, nh, Math.min(2, nh / 2));
    g.restore();
  }
  // Row labels at the left edge.
  g.font = `600 ${isDrum ? 10 : 9}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  g.textBaseline = "middle";
  rows.forEach((p, i) => {
    if (!isDrum && (p % 12 !== 0 || rowH * 12 < 26)) return;
    const label = isDrum ? drumName(p) : pitchName(p);
    const y = top + i * rowH + rowH / 2;
    const tw = g.measureText(label).width;
    g.fillStyle = "rgba(13,16,22,0.78)";
    g.fillRect(0, y - 6.5, tw + 10, 13);
    g.fillStyle = isDrum ? "rgba(230,233,240,0.78)" : "rgba(230,233,240,0.45)";
    g.fillText(label, 5, y + 0.5);
  });
  drawPlayhead(g, view, h, time);
}

function roundRect(g, x, y, w, h, r) {
  if (w < 3 || h < 3 || r < 0.5) {
    g.fillRect(x, y, w, h);
    return;
  }
  g.beginPath();
  g.roundRect(x, y, w, h, r);
  g.fill();
}

/**
 * Min/max peaks of audio at a fixed bucket size, for waveform drawing at any zoom.
 * @param {Float32Array[]} channels
 * @param {number} rate
 * @param {number} [bucketS]
 * @returns {{min: Float32Array, max: Float32Array, bucketS: number}}
 */
export function computePeaks(channels, rate, bucketS = 0.005) {
  const n = channels[0].length;
  const size = Math.max(1, Math.round(rate * bucketS));
  const nb = Math.ceil(n / size);
  const min = new Float32Array(nb);
  const max = new Float32Array(nb);
  const inv = 1 / channels.length;
  for (let b = 0; b < nb; b++) {
    let lo = 0;
    let hi = 0;
    const end = Math.min(n, (b + 1) * size);
    for (let i = b * size; i < end; i++) {
      let v = 0;
      for (const ch of channels) v += ch[i];
      v *= inv;
      if (v < lo) lo = v;
      else if (v > hi) hi = v;
    }
    min[b] = lo;
    max[b] = hi;
  }
  return { min, max, bucketS: size / rate };
}

/**
 * Draw a waveform lane from peaks (or a placeholder message if there are none yet).
 * @param {CanvasRenderingContext2D} g
 * @param {{view: import("./view.js").View, h: number, peaks: ReturnType<typeof computePeaks> | null, color: number[],
 *   time: number, message?: string}} o
 */
export function drawWaveLane(g, { view, h, peaks, color, time, message }) {
  const w = view.width;
  g.clearRect(0, 0, w, h);
  drawTimeGrid(g, view, w, h);
  const mid = h / 2;
  g.fillStyle = "rgba(255,255,255,0.05)";
  g.fillRect(0, Math.round(mid), w, 1);
  if (peaks) {
    const { min, max, bucketS } = peaks;
    const px = Math.max(1, Math.floor(w));
    const amp = (h / 2) * 0.92;
    const xPlay = view.x(time);
    for (const played of [true, false]) {
      g.fillStyle = rgba(played ? lighten(color, 0.3) : color, played ? 0.95 : 0.55);
      g.beginPath();
      for (let x = 0; x < px; x++) {
        if (played !== x < xPlay) continue;
        const a = Math.floor(view.time(x) / bucketS);
        const b = Math.max(a + 1, Math.floor(view.time(x + 1) / bucketS));
        if (a >= min.length) break;
        if (b <= 0) continue;
        let lo = 0;
        let hi = 0;
        for (let i = Math.max(0, a); i < Math.min(min.length, b); i++) {
          if (min[i] < lo) lo = min[i];
          if (max[i] > hi) hi = max[i];
        }
        const y0 = mid - hi * amp;
        const y1 = mid - lo * amp;
        g.rect(x, y0, 1, Math.max(1, y1 - y0));
      }
      g.fill();
    }
  } else if (message) {
    g.fillStyle = "rgba(230,233,240,0.38)";
    g.font = "500 12px system-ui, -apple-system, sans-serif";
    g.textBaseline = "middle";
    g.fillText(message, 14, mid);
  }
  drawPlayhead(g, view, h, time);
}

/**
 * Time ruler with labels.
 * @param {CanvasRenderingContext2D} g
 * @param {import("./view.js").View} view
 * @param {number} h
 * @param {number} time
 */
export function drawRuler(g, view, h, time) {
  const w = view.width;
  g.clearRect(0, 0, w, h);
  const step = gridStep(view.pps);
  g.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  g.textBaseline = "middle";
  for (let t = Math.ceil(view.t0 / step) * step; t <= view.t1 + 1e-9; t += step) {
    const x = Math.round(view.x(t)) + 0.5;
    g.fillStyle = "rgba(255,255,255,0.18)";
    g.fillRect(x - 0.5, h - 7, 1, 7);
    g.fillStyle = "rgba(230,233,240,0.55)";
    g.fillText(fmtTime(t, step < 1 ? (step < 0.1 ? 2 : 1) : 0), x + 4, h / 2 - 1);
  }
  const x = view.x(time);
  if (x >= 0 && x <= w) {
    g.fillStyle = "#fff";
    g.beginPath();
    g.moveTo(x - 5, 0);
    g.lineTo(x + 5, 0);
    g.lineTo(x, 7);
    g.closePath();
    g.fill();
  }
}
