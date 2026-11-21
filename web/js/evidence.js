// The evidence repairer around repairer.onnx: per-note evidence vectors (769 numbers), the windowed edits and their application.
// Port of dettato/evidence/features.py, dettato/evidence/repair.py and the numpy packing of dettato/evidence/model.py:candidates.

import { roundF16 } from "./f16.js";
import { HOP, N_KEYS, PITCH_LO, notes as transcribe, roundHalfEven } from "./transcriber.js";

export const C_H = 368;
export const N_MAP = 33;
export const N_MF = N_MAP - 2; // 31 map numbers computed here, the two log spreads come from assemble
export const N_EV = N_MAP + 2 * C_H; // 769
export const L_WIN = 8.0;
export const MAX_WIN = 4096;
export const ONSET_TH = 0.25;
export const DEDUP_S = 0.03;
const Q = 32.0;
const N_SLOT = 2;

const fr = Math.fround;

// ---------------------------------------------------------------------------------------------------------- small helpers

/** First index i with a[i] >= v (np.searchsorted, side left). */
function searchLeft(a, v) {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

/** numpy np.round(x, 4): multiply, rint (half even), divide. */
function npRound4(x) {
  return roundHalfEven(x * 1e4) / 1e4;
}

/**
 * Python's round(x, 4): correctly rounded on the exact binary value, exact ties to even.
 * @param {number} x
 * @returns {number}
 */
export function pyRound4(x) {
  const away = Number(x.toFixed(4)); // nearest, exact ties away from zero
  const s = Math.abs(x).toFixed(30); // exact for any |x| this code meets (a 4-decimal tie has 5 decimals)
  const dot = s.indexOf(".");
  if (!/^50*$/.test(s.slice(dot + 5))) return away;
  if ((s.charCodeAt(dot + 4) - 48) % 2 === 1) return away;
  return Math.sign(x) * Number(s.slice(0, dot + 5));
}

/** Indices 0..n-1 sorted by cmp (stable, like np.lexsort / kind="stable"). */
function argsort(n, cmp) {
  return Int32Array.from({ length: n }, (_, i) => i).sort(cmp);
}

// ---------------------------------------------------------------------------------------------------------- map evidence

/**
 * _decode_maps: transcriber notes and their (key, onset frame, end frame, voice) indices.
 * @param {Record<string, Float32Array>} maps the six heads, each [T][88]
 * @param {object} th transcriber thresholds (meta.json transcriber.thresholds, optional hop)
 * @returns {{res: ReturnType<typeof transcribe>, L: Int32Array, A: Int32Array, B: Int32Array, S: Int8Array}}
 */
export function decodeMaps(maps, th) {
  const hop = th.hop ?? HOP;
  const res = transcribe(maps, th);
  const T = maps.onset.length / N_KEYS;
  const n = res.onset.length;
  const L = Int32Array.from(res.pitch, (p) => p - PITCH_LO);
  const A = new Int32Array(n);
  const B = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    A[i] = clip(roundHalfEven(res.onset[i] / hop), 0, T - 1);
    B[i] = clip(roundHalfEven(res.offset[i] / hop), A[i] + 1, T);
  }
  return { res, L, A, B, S: Int8Array.from(res.slot) };
}

/** The 24 numbers _map_feats reads from a note's own voice (own) and the other voice (other). */
function ownFeats(own, other, T, l, a, b) {
  const [on, frm, off] = own;
  const at = (m, t) => m[clip(t, 0, T - 1) * N_KEYS + l];
  const span = Math.max(b - a, 1);
  const f = [];
  for (let d = -2; d <= 2; d++) f.push(at(on, a + d));
  for (let j = 0; j < 8; j++) f.push(at(frm, a + Math.floor((j * span) / 8)));
  for (const k of [1, 3, 6, 12]) f.push(at(frm, b - 1 + k));
  let pre = -Infinity;
  for (let k = 1; k <= 5; k++) pre = Math.max(pre, a - k >= 0 ? at(frm, a - k) : 0);
  f.push(pre);
  f.push(...offsetPeak(off, T, l, a, b, span), at(off, b - 1));
  const [oOn, oFr] = other;
  f.push(Math.max(at(oOn, a - 1), at(oOn, a), at(oOn, a + 1)));
  f.push(at(oFr, a));
  let s = 0;
  for (let j = 0; j < 4; j++) s = fr(s + at(oFr, a + Math.floor((j * span) / 4)));
  f.push(fr(s / 4));
  return f;
}

/** Highest offset probability over [a+1, min(b+12, T)) and its relative position (capped at 2). */
function offsetPeak(off, T, l, a, b, span) {
  let best = -Infinity;
  let jb = -1;
  for (let t = a + 1, j = 0; t < Math.min(b + 12, T); t++, j++) {
    const v = off[t * N_KEYS + l];
    if (v > best) [best, jb] = [v, j];
  }
  return jb < 0 ? [0, 0] : [best, fr(Math.min((jb + 1) / span, 2.0))];
}

/** First voice onset (max over a-1..a+1) on the keys l-1, l+1, l-12, l+12, l+19, l+24 (0 off the keyboard). */
function neighbourFeats(on1, T, l, a) {
  const f = [];
  for (const dl of [-1, 1, -12, 12, 19, 24]) {
    const l2 = l + dl;
    let v = -Infinity;
    for (let d = -1; d <= 1; d++) v = Math.max(v, on1[clip(a + d, 0, T - 1) * N_KEYS + clip(l2, 0, N_KEYS - 1)]);
    f.push(l2 >= 0 && l2 < N_KEYS ? v : 0);
  }
  return f;
}

/**
 * map_evidence: every map number of every note.
 * @param {Record<string, Float32Array>} maps the six heads, each [T][88]
 * @param {Int32Array} L key index
 * @param {Int32Array} A onset frame
 * @param {Int32Array} B end frame
 * @param {Int8Array} S voice (1 / 2)
 * @returns {Float32Array} [N][31]
 */
export function mapEvidence(maps, L, A, B, S) {
  const T = maps.onset.length / N_KEYS;
  const s1 = [maps.onset, maps.frame, maps.offset];
  const s2 = [maps.onset2, maps.frame2, maps.offset2];
  const out = new Float32Array(L.length * N_MF);
  for (let i = 0; i < L.length; i++) {
    const [own, other] = S[i] === 2 ? [s2, s1] : [s1, s2];
    const f = ownFeats(own, other, T, L[i], A[i], B[i]);
    f.push(S[i] === 2 ? 1 : 0, ...neighbourFeats(maps.onset, T, L[i], A[i]));
    out.set(f, i * N_MF);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------- hidden state

/** Group note indices by key: Map(lane -> note indices). */
function byLane(L) {
  const g = new Map();
  for (let i = 0; i < L.length; i++) {
    if (!g.has(L[i])) g.set(L[i], []);
    g.get(L[i]).push(i);
  }
  return g;
}

/**
 * Float32 prefix sums at the requested frames of one lane, as torch's CPU cumsum (double accumulator, float32 store):
 * Map(frame x -> Float32Array(C_H) = sum of frames [0, x)).
 */
function lanePrefix(store, lane, frames) {
  const want = [...new Set(frames)].sort((a, b) => a - b);
  const acc = new Float64Array(C_H);
  const v = new Float32Array(C_H);
  const out = new Map();
  let t = 0;
  for (const x of want) {
    for (; t < x; t++) {
      store.vec("hidden", t, lane, v);
      for (let c = 0; c < C_H; c++) acc[c] += v[c];
    }
    out.set(x, Float32Array.from(acc));
  }
  return out;
}

/** mean over [a, b) of one lane from the reference's float32 prefix sums. */
function meanFromPrefix(cs, a, b, out, off) {
  const ca = cs.get(a);
  const cb = cs.get(b);
  const d = fr(Math.max(b - a, 1));
  for (let c = 0; c < C_H; c++) out[off + c] = fr(fr(cb[c] - ca[c]) / d);
}

/** mean over [a, b) of one lane, summed directly in float64 (only frames a..b-1 are read). */
function meanDirect(store, lane, a, b, out, off) {
  const acc = new Float64Array(C_H);
  const v = new Float32Array(C_H);
  for (let t = a; t < b; t++) {
    store.vec("hidden", t, lane, v);
    for (let c = 0; c < C_H; c++) acc[c] += v[c];
  }
  const d = Math.max(b - a, 1);
  for (let c = 0; c < C_H; c++) out[off + c] = acc[c] / d;
}

/**
 * gather_hidden: per note the hidden state at (A, L) and its mean over [A, B) on lane L.
 * Default: the mean is summed over [A, B) only, in float64 (works with a bounded store that released earlier frames; differs from
 * the reference by the reference's own float32 cumsum rounding, ~4e-5 absolute on the band clip). With `{exact: true}` it
 * reproduces the reference bit for bit (torch's float32 running sum from frame 0), but reads every frame 0..max(B) of each used
 * lane, so the store must still hold them.
 * @param {{vec: (tap: string, t: number, lane: number, out?: Float32Array) => Float32Array}} store
 * @param {Int32Array} L
 * @param {Int32Array} A
 * @param {Int32Array} B
 * @param {{exact?: boolean}} [opts]
 * @returns {{at: Float32Array, mean: Float32Array}} each [N][368]
 */
export function gatherHidden(store, L, A, B, opts = {}) {
  const exact = opts.exact ?? false;
  const N = L.length;
  const at = new Float32Array(N * C_H);
  const mean = new Float32Array(N * C_H);
  const v = new Float32Array(C_H);
  for (const [lane, sel] of byLane(L)) {
    const cs = exact
      ? lanePrefix(
          store,
          lane,
          sel.flatMap((i) => [A[i], B[i]]),
        )
      : null;
    for (const i of sel) {
      at.set(store.vec("hidden", A[i], lane, v), i * C_H);
      if (exact) meanFromPrefix(cs, A[i], B[i], mean, i * C_H);
      else meanDirect(store, lane, A[i], B[i], mean, i * C_H);
    }
  }
  return { at, mean };
}

// ---------------------------------------------------------------------------------------------------------- assembly

/** numpy's float32 pairwise summation (add.reduce over a contiguous axis), a[off .. off+n). */
function pairwiseF32(a, off, n) {
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r = fr(r + a[off + i]);
    return r;
  }
  if (n <= 128) {
    const r = Array.from({ length: 8 }, (_, j) => a[off + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = fr(r[j] + a[off + i + j]);
    let res = fr(fr(fr(r[0] + r[1]) + fr(r[2] + r[3])) + fr(fr(r[4] + r[5]) + fr(r[6] + r[7])));
    for (; i < n; i++) res = fr(res + a[off + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return fr(pairwiseF32(a, off, n2) + pairwiseF32(a, off + n2, n - n2));
}

const SD_EPS = fr(1e-6);
const Z_MAX = fr(3.96);

/** _std_q of one row: standardise over its C_H dims, quantise to int8 steps of 1/Q -> (q / Q values, log sd). */
function stdQ(x, off, out, outOff) {
  const mu = fr(pairwiseF32(x, off, C_H) / C_H);
  const d = new Float32Array(C_H);
  for (let c = 0; c < C_H; c++) {
    const e = fr(x[off + c] - mu);
    d[c] = fr(e * e);
  }
  const sd = fr(fr(Math.sqrt(fr(pairwiseF32(d, 0, C_H) / C_H))) + SD_EPS);
  for (let c = 0; c < C_H; c++) {
    const z = clip(fr(fr(x[off + c] - mu) / sd), -Z_MAX, Z_MAX);
    out[outOff + c] = roundHalfEven(fr(z * Q)) / Q;
  }
  return fr(Math.log(sd));
}

/**
 * _assemble: map numbers + two log spreads (float16-rounded) and the two quantised hidden vectors.
 * @param {Float32Array} mf [N][31]
 * @param {Float32Array} at [N][368]
 * @param {Float32Array} mean [N][368]
 * @returns {Float32Array} [N][769]
 */
export function assemble(mf, at, mean) {
  const N = at.length / C_H;
  const ev = new Float32Array(N * N_EV);
  for (let i = 0; i < N; i++) {
    const o = i * N_EV;
    for (let j = 0; j < N_MF; j++) ev[o + j] = roundF16(mf[i * N_MF + j]);
    ev[o + N_MF] = roundF16(stdQ(at, i * C_H, ev, o + N_MAP));
    ev[o + N_MF + 1] = roundF16(stdQ(mean, i * C_H, ev, o + N_MAP + C_H));
  }
  return ev;
}

// ---------------------------------------------------------------------------------------------------------- repairer input

/** conf_vel: onset confidence [0.25, 1] -> velocity 1..127. */
function confVel(conf) {
  const x = (conf - ONSET_TH) / (1.0 - ONSET_TH);
  return clip(roundHalfEven(1 + 126 * x), 1, 127);
}

/**
 * @typedef {{n: number, inst: Uint8Array, pitch: Uint8Array, vel: Uint8Array, on: Float32Array, off: Float32Array}} NoteTable
 *   the repairer's note array (NOTE_DT), sorted by (on, inst, pitch)
 */

/**
 * input_notes: transcriber notes -> the repairer's note table and the sort order applied (rows of ev must follow it).
 * @param {{onset: Float64Array, offset: Float64Array, pitch: ArrayLike<number>, conf: Float32Array}} res
 * @returns {{k: NoteTable, order: Int32Array}}
 */
export function inputNotes(res) {
  const n = res.onset.length;
  const on = new Float32Array(n);
  const off = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = npRound4(res.onset[i]);
    on[i] = t;
    off[i] = Math.max(npRound4(res.offset[i]), t + 1e-3);
  }
  const pitch = Uint8Array.from(res.pitch);
  const order = argsort(n, (a, b) => on[a] - on[b] || pitch[a] - pitch[b]); // inst is always 0
  const take = (src, Ctor) => Ctor.from(order, (i) => src[i]);
  const k = {
    n,
    inst: new Uint8Array(n),
    pitch: take(pitch, Uint8Array),
    vel: Uint8Array.from(order, (i) => confVel(res.conf[i])),
    on: take(on, Float32Array),
    off: take(off, Float32Array),
  };
  return { k, order };
}

/**
 * Notes [lo, hi) of a note table (e.g. one 8 s block).
 * @param {NoteTable} k
 * @param {number} lo
 * @param {number} hi
 * @returns {NoteTable}
 */
export function sliceNotes(k, lo, hi) {
  const f = (a) => a.slice(lo, hi);
  return { n: hi - lo, inst: f(k.inst), pitch: f(k.pitch), vel: f(k.vel), on: f(k.on), off: f(k.off) };
}

/**
 * Rows of a row-major [N][width] matrix in the given order (e.g. ev[order] for the note table's order).
 * @param {Float32Array} m
 * @param {ArrayLike<number>} order
 * @param {number} width
 * @returns {Float32Array} [order.length][width]
 */
export function takeRows(m, order, width) {
  const out = new Float32Array(order.length * width);
  order.forEach((i, r) => out.set(m.subarray(i * width, (i + 1) * width), r * width));
  return out;
}

// ---------------------------------------------------------------------------------------------------------- repairer windows

/** _window_batch as ONNX feeds: notes [lo, hi) of k with onsets relative to t0. */
function windowFeeds(ort, k, ev, lo, hi, t0) {
  const n = hi - lo;
  const i64 = (src) =>
    new ort.Tensor(
      "int64",
      BigInt64Array.from(src.subarray(lo, hi), (v) => BigInt(v)),
      [1, n],
    );
  const on = Float32Array.from(k.on.subarray(lo, hi), (v) => v - t0);
  const dur = Float32Array.from({ length: n }, (_, i) => Math.max(fr(k.off[lo + i] - k.on[lo + i]), 0));
  return {
    inst: i64(k.inst),
    pitch: i64(k.pitch),
    vel: i64(k.vel),
    on: new ort.Tensor("float32", on, [1, n]),
    dur: new ort.Tensor("float32", dur, [1, n]),
    mask: new ort.Tensor("bool", new Uint8Array(n).fill(1), [1, n]),
    ev: new ort.Tensor("float32", ev.slice(lo * N_EV, hi * N_EV), [1, n, N_EV]),
  };
}

/** Repairer.candidates' numpy packing (CAND_DT): n edited notes (kind 0), then 2 n insertion proposals (kind 1). */
function packCandidates(o, k, lo, n) {
  const m = n + n * N_SLOT;
  const c = {
    pitch: new Int16Array(m),
    on: new Float32Array(m),
    off: new Float32Array(m),
    score: new Float32Array(m),
    kind: new Int8Array(m),
  };
  const [pk, onK, durK, pe, onI, durI, pitchI] = ["pk", "on_k", "dur_k", "pe", "on_i", "dur_i", "pitch_i"].map(
    (x) => o[x].data,
  );
  for (let i = 0; i < n; i++) {
    c.pitch[i] = k.pitch[lo + i];
    c.on[i] = onK[i];
    c.off[i] = onK[i] + durK[i];
    c.score[i] = pk[i];
  }
  for (let j = 0; j < n * N_SLOT; j++) {
    c.pitch[n + j] = Number(pitchI[j]);
    c.on[n + j] = onI[j];
    c.off[n + j] = onI[j] + durI[j];
    c.score[n + j] = pe[j];
    c.kind[n + j] = 1;
  }
  return c;
}

/**
 * One repairer window: notes [lo, hi) of k (onsets relative to t0) -> its candidate table, as Repairer.candidates(...)[0].
 * @param {{ort: object, session: (name: string) => Promise<object>}} rt
 * @param {NoteTable} k
 * @param {Float32Array} ev [k.n][769], rows in k order
 * @param {number} lo
 * @param {number} hi
 * @param {number} t0 window start (s)
 * @returns {Promise<{pitch: Int16Array, on: Float32Array, off: Float32Array, score: Float32Array, kind: Int8Array}>}
 */
export async function candidates(rt, k, ev, lo, hi, t0) {
  const sess = await rt.session("repairer");
  const out = await sess.run(windowFeeds(rt.ort, k, ev, lo, hi, t0));
  return packCandidates(out, k, lo, hi - lo);
}

/** Copy one window's candidates into the per-note edits and the insertion list (EvidenceRepairer.edits' loop body). */
function collectWindow(c, e, lo, m, t0, lWin) {
  for (let i = 0; i < m; i++) {
    const ok = c.on[i] >= 0 && c.on[i] < lWin; // a note whose corrected onset leaves the window is dropped
    e.pk[lo + i] = ok ? c.score[i] : -1.0;
    e.onR[lo + i] = c.on[i] + t0;
    e.offR[lo + i] = c.off[i] + t0;
  }
  for (let i = m; i < c.on.length; i++) {
    if (c.kind[i] !== 0 && c.on[i] >= 0 && c.on[i] < lWin)
      e.ins.push([c.on[i] + t0, c.off[i] + t0, c.pitch[i], c.score[i]]);
  }
}

/**
 * @typedef {{pk: Float64Array, onR: Float64Array, offR: Float64Array, ins: number[][]}} Edits
 *   per input note P(keep) (-1 = corrected onset left the window), repaired onset / end (NaN = window passed through
 *   unchanged), and insertion proposals [[onsetS, endS, pitch, score], ...]
 */

/**
 * EvidenceRepairer.edits: run the repairer over 8 s windows of k (t0 = 0, 8, 16, ...; empty windows and windows with more
 * than MAX_WIN notes are skipped). k may hold any subset of the song's notes, e.g. one window's.
 * @param {{ort: object, session: (name: string) => Promise<object>}} rt
 * @param {NoteTable} k
 * @param {Float32Array} ev [k.n][769], rows in k order
 * @param {{lWin?: number, maxWin?: number}} [opts]
 * @returns {Promise<Edits>}
 */
export async function edits(rt, k, ev, opts = {}) {
  const lWin = opts.lWin ?? L_WIN;
  const maxWin = opts.maxWin ?? MAX_WIN;
  const n = k.n;
  const e = {
    pk: new Float64Array(n).fill(NaN),
    onR: new Float64Array(n).fill(NaN),
    offR: new Float64Array(n).fill(NaN),
    ins: [],
  };
  if (!n) return e;
  let end = -Infinity;
  for (let i = 0; i < n; i++) end = Math.max(end, k.off[i]);
  const nWin = Math.ceil((end + lWin) / lWin);
  for (let w = 0; w < nWin; w++) {
    const t0 = w * lWin;
    const lo = searchLeft(k.on, t0);
    const hi = searchLeft(k.on, t0 + lWin);
    if (hi <= lo || hi - lo > maxWin) continue;
    collectWindow(await candidates(rt, k, ev, lo, hi, t0), e, lo, hi - lo, t0, lWin);
  }
  return e;
}

/** Alias used by docs/ARCHITECTURE.md. */
export const repairWindows = edits;

// ---------------------------------------------------------------------------------------------------------- applying edits

/** @typedef {{on: Float64Array, off: Float64Array, pitch: Int32Array, score: Float32Array, kind: Int8Array}} Rows */

/** Rows from plain JS arrays [[on, off, pitch, score, kind], ...]. */
function toRows(list) {
  return {
    on: Float64Array.from(list, (r) => r[0]),
    off: Float64Array.from(list, (r) => r[1]),
    pitch: Int32Array.from(list, (r) => r[2]),
    score: Float32Array.from(list, (r) => r[3]),
    kind: Int8Array.from(list, (r) => r[4]),
  };
}

/**
 * The candidate rows of EvidenceRepairer.apply before dedup: unchanged / kept notes in k order, then the insertions (score >=
 * insertTh) in edits order.
 * @param {NoteTable} k
 * @param {Edits} e
 * @param {number} keepTh
 * @param {number} insertTh
 * @returns {Rows}
 */
export function candidateRows(k, e, keepTh, insertTh) {
  const rows = [];
  for (let j = 0; j < k.n; j++) {
    if (Number.isNaN(e.pk[j])) rows.push([k.on[j], k.off[j], k.pitch[j], 1.0, 0]);
    else if (e.pk[j] >= keepTh) rows.push([e.onR[j], e.offR[j], k.pitch[j], e.pk[j], 0]);
  }
  for (const [a, b, p, s] of e.ins) if (s >= insertTh) rows.push([a, b, p, s, 1]);
  return toRows(rows);
}

/**
 * Join row tables (e.g. one per 8 s block) in order.
 * @param {Rows[]} list
 * @returns {Rows}
 */
export function concatRows(list) {
  const cat = (key, Ctor) => {
    const out = new Ctor(list.reduce((s, r) => s + r[key].length, 0));
    let o = 0;
    for (const r of list) {
      out.set(r[key], o);
      o += r[key].length;
    }
    return out;
  };
  return {
    on: cat("on", Float64Array),
    off: cat("off", Float64Array),
    pitch: cat("pitch", Int32Array),
    score: cat("score", Float32Array),
    kind: cat("kind", Int8Array),
  };
}

/** The rows idx of a row table. */
function selectRows(rows, idx) {
  const f = (a) => a.constructor.from(idx, (i) => a[i]);
  return { on: f(rows.on), off: f(rows.off), pitch: f(rows.pitch), score: f(rows.score), kind: f(rows.kind) };
}

/**
 * Split a row table into its kept / unchanged notes (kind 0) and its insertions (kind 1), each in order. Concatenating all
 * blocks' kept tables, then all their insertion tables, gives the reference's global row order for safeDedupIndex.
 * @param {Rows} rows
 * @returns {{kept: Rows, inserted: Rows}}
 */
export function splitRowsByKind(rows) {
  const idx = [[], []];
  rows.kind.forEach((v, i) => idx[v === 0 ? 0 : 1].push(i));
  return { kept: selectRows(rows, idx[0]), inserted: selectRows(rows, idx[1]) };
}

/** _dedup_idx over the rows idx: same pitch within tol -> keep the higher score. Returns kept entries of idx, sorted. */
function dedupIdx(rows, idx, tol) {
  if (idx.length < 2) return Int32Array.from(idx);
  const { pitch, on, score } = rows;
  const o = argsort(idx.length, (a, b) => pitch[idx[a]] - pitch[idx[b]] || on[idx[a]] - on[idx[b]]);
  const keep = new Uint8Array(o.length).fill(1);
  for (let j = 0; j + 1 < o.length; j++) {
    const a = idx[o[j]];
    const b = idx[o[j + 1]];
    if (pitch[a] !== pitch[b] || !(on[b] - on[a] < tol) || !keep[j]) continue;
    if (score[a] >= score[b]) keep[j + 1] = 0;
    else keep[j] = 0;
  }
  return Int32Array.from(
    o.filter((_, j) => keep[j]),
    (j) => idx[j],
  ).sort();
}

/** Sorted onsets of the kept rows (kind 0), per pitch. */
function keptOnsets(rows) {
  const kp = new Map();
  for (let q = 0; q < rows.kind.length; q++) {
    if (rows.kind[q] !== 0) continue;
    if (!kp.has(rows.pitch[q])) kp.set(rows.pitch[q], []);
    kp.get(rows.pitch[q]).push(rows.on[q]);
  }
  for (const v of kp.values()) v.sort((a, b) => a - b);
  return kp;
}

/**
 * _safe_dedup_idx: a kept note is never removed; an insertion is dropped when a kept note of its pitch starts within tol, the
 * remaining insertions are deduplicated among themselves.
 * @param {Rows} rows
 * @param {number} [tol]
 * @returns {Int32Array} kept row indices, sorted
 */
export function safeDedupIndex(rows, tol = DEDUP_S) {
  const n = rows.kind.length;
  const kept = [];
  const ins = [];
  for (let q = 0; q < n; q++) (rows.kind[q] === 0 ? kept : ins).push(q);
  if (!ins.length) return Int32Array.from({ length: n }, (_, i) => i);
  const kp = keptOnsets(rows);
  const insOk = ins.filter((q) => {
    const v = kp.get(rows.pitch[q]);
    if (!v) return true;
    const i = searchLeft(v, rows.on[q]);
    return ![i - 1, i].some((r) => r >= 0 && r < v.length && Math.abs(v[r] - rows.on[q]) < tol);
  });
  return Int32Array.from([...kept, ...dedupIdx(rows, insOk, tol)]).sort();
}

/**
 * The final step of EvidenceRepairer.apply: the selected rows, times rounded to 0.1 ms, ends at least 10 ms after onsets.
 * @param {Rows} rows
 * @param {ArrayLike<number>} idx row indices to keep
 * @returns {{onset: Float64Array, offset: Float64Array, pitch: Int32Array, inserted: Uint8Array, score: Float32Array}}
 */
export function finishRows(rows, idx) {
  return {
    onset: Float64Array.from(idx, (i) => pyRound4(rows.on[i])),
    offset: Float64Array.from(idx, (i) => pyRound4(Math.max(rows.off[i], rows.on[i] + 0.01))),
    pitch: Int32Array.from(idx, (i) => rows.pitch[i]),
    inserted: Uint8Array.from(idx, (i) => (rows.kind[i] === 1 ? 1 : 0)),
    score: Float32Array.from(idx, (i) => rows.score[i]),
  };
}

/**
 * EvidenceRepairer.apply: input notes + edits -> the repaired notes.
 * @param {NoteTable} k
 * @param {Edits} e
 * @param {number} [keepTh]
 * @param {number} [insertTh]
 * @returns {{onset: Float64Array, offset: Float64Array, pitch: Int32Array, inserted: Uint8Array, score: Float32Array}}
 */
export function apply(k, e, keepTh = 0.4, insertTh = 0.6) {
  const rows = candidateRows(k, e, keepTh, insertTh);
  return finishRows(rows, rows.on.length ? safeDedupIndex(rows) : []);
}

// ---------------------------------------------------------------------------------------------------------- convenience

/**
 * EvidenceRepairer.from_run: maps + hidden-state store of one transcriber pass -> repaired notes.
 * @param {{ort: object, session: (name: string) => Promise<object>}} rt
 * @param {Record<string, Float32Array>} maps the six heads, each [T][88]
 * @param {{vec: Function}} store tap store holding "hidden"
 * @param {object} th transcriber thresholds (meta.json transcriber.thresholds)
 * @param {{keep_th?: number, insert_th?: number, L_WIN?: number, exact?: boolean}} [cfg] meta.json repairer; exact (default
 *   true) is gatherHidden's option: false when the store no longer holds the early frames
 * @returns {Promise<{rep: ReturnType<typeof apply>, res: ReturnType<typeof transcribe>, info: {transcribed: number,
 *   kept: number, inserted: number, dropped: number}}>}
 */
export async function repair(rt, maps, store, th, cfg = {}) {
  const { res, L, A, B, S } = decodeMaps(maps, th);
  const { at, mean } = gatherHidden(store, L, A, B, { exact: cfg.exact ?? true });
  const ev = assemble(mapEvidence(maps, L, A, B, S), at, mean);
  const { k, order } = inputNotes(res);
  const e = await edits(rt, k, takeRows(ev, order, N_EV), { lWin: cfg.L_WIN });
  const rep = apply(k, e, cfg.keep_th ?? 0.4, cfg.insert_th ?? 0.6);
  const inserted = rep.inserted.reduce((s, v) => s + v, 0);
  const kept = rep.inserted.length - inserted;
  return { rep, res, info: { transcribed: k.n, kept, inserted, dropped: k.n - kept } };
}
