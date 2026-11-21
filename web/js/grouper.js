// The grouper: notes -> instrument-named tracks. A port of dettato/grouper/core.py (slots, the note gather, the 24 s windows,
// the prediction average) and dettato/grouper/decode.py ("cluster, then name"). Float32 steps of the reference (NumPy
// float32 arrays with Python scalars, torch float32 tensors) are rounded with Math.fround at every operation; float64 steps
// follow NumPy's summation order (pairwise for contiguous rows, sequential down columns / np.add.at).
import { roundF16 } from "./f16.js";
import {
  argsortStable,
  erf,
  fclusterDistance,
  linearSumAssignment,
  linkageAverage,
  npSum,
  roundHalfEven,
  rowNorm,
  rowNorm32,
} from "./cluster.js";

const f32 = Math.fround;

export const SR = 16000;
export const PITCH_LO = 21;
export const N_KEYS = 88;
export const PAD0 = 0.5;
export const PAD1 = 1.0;
export const SPAN_CAP = 50;
export const STEM_SPAN = 10;
export const SLOT_TOL = 0.1;
export const DEDUP = 0.03;
export const FEAT_DIM = 368;
export const META_DIM = 12;
export const HOP = 0.02;
export const AGGLO_CAP = 600;

// ============================================================================= note features from the transcriber

/**
 * core.assign_slots: input notes + the transcriber's own notes -> [N][4] (onset, end >= onset + 20 ms, pitch, voice slot of
 * the nearest same-pitch own note within SLOT_TOL else 0), sorted by onset, then pitch.
 * @param {ArrayLike<ArrayLike<number>>} notes [[onset, offset, pitch]]
 * @param {ArrayLike<ArrayLike<number>>} own [[onset, offset, pitch, slot 0 / 1]]
 * @returns {Float64Array} [N][4]
 */
export function assignSlots(notes, own) {
  const N = notes.length;
  const byPitch = new Map();
  for (let j = 0; j < own.length; j++) {
    const p = own[j][2];
    if (!byPitch.has(p)) byPitch.set(p, []);
    byPitch.get(p).push(j);
  }
  const slot = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const b = byPitch.get(notes[i][2]);
    if (!b) continue;
    let best = Infinity;
    let bj = -1;
    for (const j of b) {
      const d = Math.abs(notes[i][0] - own[j][0]);
      if (d < best) {
        best = d;
        bj = j;
      }
    }
    slot[i] = best <= SLOT_TOL ? own[bj][3] : 0;
  }
  const order = Array.from({ length: N }, (_, i) => i);
  order.sort((a, b) => notes[a][0] - notes[b][0] || notes[a][2] - notes[b][2] || a - b);
  const out = new Float64Array(N * 4);
  order.forEach((o, r) => {
    const on = notes[o][0];
    out.set([on, Math.max(notes[o][1], on + 0.02), notes[o][2], slot[o]], r * 4);
  });
  return out;
}

/** Per-lane running sums of a [T][88] map, accumulated in float64 and stored in float32 (torch's CPU cumsum), T + 1 rows. */
function cumsumF32(m, T) {
  const cs = new Float32Array((T + 1) * N_KEYS);
  const acc = new Float64Array(N_KEYS);
  for (let t = 0; t < T; t++) {
    for (let l = 0; l < N_KEYS; l++) {
      acc[l] += m[t * N_KEYS + l];
      cs[(t + 1) * N_KEYS + l] = acc[l];
    }
  }
  return cs;
}

function noteFrames(nt, N, T, hop) {
  const lane = new Int32Array(N);
  const t0 = new Int32Array(N);
  const te = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    lane[i] = Math.min(Math.max(roundHalfEven(nt[i * 4 + 2]) - PITCH_LO, 0), N_KEYS - 1);
    t0[i] = Math.min(Math.max(roundHalfEven(nt[i * 4] / hop), 0), T - 1);
    te[i] = Math.min(Math.max(Math.ceil(nt[i * 4 + 1] / hop), t0[i] + 1), T);
  }
  return { lane, t0, te };
}

function gatherMeta(maps, nt, fr, T) {
  const N = fr.lane.length;
  const meta = new Float32Array(N * META_DIM);
  const csF = cumsumF32(maps.frame, T);
  const csF2 = cumsumF32(maps.frame2, T);
  const at = (m, t, l) => m[Math.min(Math.max(t, 0), T - 1) * N_KEYS + l];
  const mx3 = (m, t, l) => Math.max(at(m, t - 1, l), at(m, t, l), at(m, t + 1, l));
  const spanMean = (cs, a, b, l) => f32(f32(cs[b * N_KEYS + l] - cs[a * N_KEYS + l]) / Math.max(b - a, 1));
  for (let i = 0; i < N; i++) {
    const l = fr.lane[i];
    const a = fr.t0[i];
    const b = fr.te[i];
    meta.set(
      [
        nt[i * 4],
        nt[i * 4 + 1],
        nt[i * 4 + 2],
        nt[i * 4 + 3],
        mx3(maps.onset, a, l),
        mx3(maps.onset2, a, l),
        at(maps.frame, a, l),
        at(maps.frame2, a, l),
        spanMean(csF, a, b, l),
        spanMean(csF2, a, b, l),
        mx3(maps.offset, b - 1, l),
        mx3(maps.offset2, b - 1, l),
      ],
      i * META_DIM,
    );
  }
  return meta;
}

/** float32 mean of tap rows t0 .. min(te, t0 + cap) - 1 (core.gather's span()). */
function spanTap(store, tap, a, b, cap, l, row, out) {
  out.fill(0);
  const lim = Math.min(b, a + cap);
  let cnt = 0;
  for (let t = a; t < lim; t++) {
    const v = store.vec(tap, t, l, row);
    for (let c = 0; c < FEAT_DIM; c++) out[c] = f32(out[c] + v[c]);
    cnt++;
  }
  const d = Math.max(cnt, 1);
  for (let c = 0; c < FEAT_DIM; c++) out[c] = f32(out[c] / d);
}

const SC_MIN = f32(1e-6);

/** One tap vector -> int8 row + float32 scale (max |x| / 127, rows quantised half-to-even). */
function quantise(x, feat, off) {
  let mx = 0;
  for (let c = 0; c < FEAT_DIM; c++) mx = Math.max(mx, Math.abs(x[c]));
  const sc = f32(Math.max(mx, SC_MIN) / 127);
  for (let c = 0; c < FEAT_DIM; c++) {
    feat[off + c] = Math.min(Math.max(roundHalfEven(f32(x[c] / sc)), -127), 127);
  }
  return sc;
}

/**
 * core.gather: the sub-run maps and taps at the notes.
 * @param {{onset: Float32Array, frame: Float32Array, offset: Float32Array, onset2: Float32Array, frame2: Float32Array,
 *   offset2: Float32Array}} maps each [T][88]
 * @param {{vec: (tap: string, t: number, lane: number, out?: Float32Array) => Float32Array}} store tap store ("subHidden",
 *   "subStem"; float16-rounded values)
 * @param {Float64Array} nt [N][4] from assignSlots
 * @param {number} [hop] frame hop in seconds
 * @returns {{meta: Float32Array, feat: Int8Array, fsc: Float32Array}} meta [N][12], feat [N][3][368], fsc [N][3] (float16 values)
 */
export function gather(maps, store, nt, hop = HOP) {
  const T = maps.onset.length / N_KEYS;
  const N = nt.length / 4;
  const fr = noteFrames(nt, N, T, hop);
  const meta = gatherMeta(maps, nt, fr, T);
  const feat = new Int8Array(N * 3 * FEAT_DIM);
  const fsc = new Float32Array(N * 3);
  const row = new Float32Array(FEAT_DIM);
  const x = new Float32Array(FEAT_DIM);
  for (let i = 0; i < N; i++) {
    const l = fr.lane[i];
    const a = fr.t0[i];
    const b = fr.te[i];
    x.set(store.vec("subHidden", a, l, row).subarray(0, FEAT_DIM));
    fsc[i * 3] = roundF16(quantise(x, feat, i * 3 * FEAT_DIM));
    spanTap(store, "subHidden", a, b, SPAN_CAP, l, row, x);
    fsc[i * 3 + 1] = roundF16(quantise(x, feat, (i * 3 + 1) * FEAT_DIM));
    spanTap(store, "subStem", a, b, STEM_SPAN, l, row, x);
    fsc[i * 3 + 2] = roundF16(quantise(x, feat, (i * 3 + 2) * FEAT_DIM));
  }
  return { meta, feat, fsc };
}

// ============================================================================= windows

/** First index with on[i] >= v (np.searchsorted, side "left"; compared in float64). */
function searchLeft(on, v) {
  let lo = 0;
  let hi = on.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (on[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * core.song_windows: windows covering every note (onsets sorted); a window with more than nmax notes is followed by one
 * that starts at its first uncovered note.
 * @param {ArrayLike<number>} on sorted onsets
 * @param {number} W window length (s)
 * @param {number} hop window hop (s)
 * @param {number} nmax most notes per window
 * @returns {{t0: number, sel: Int32Array}[]}
 */
export function songWindows(on, W, hop, nmax) {
  const out = [];
  const n = on.length;
  if (n === 0) return out;
  const end = on[n - 1];
  let t0 = 0;
  for (;;) {
    const a = searchLeft(on, t0);
    const b = searchLeft(on, t0 + W);
    const m = Math.max(Math.min(b, a + nmax) - a, 0);
    if (m >= 1) out.push({ t0, sel: Int32Array.from({ length: m }, (_, k) => a + k) });
    if (b >= n && a + nmax >= b) break;
    if (b - a > nmax) t0 = Math.max(t0 + 0.25 * hop, on[a + Math.floor(nmax / 2)]);
    else t0 += hop;
    if (t0 > end) break;
  }
  return out;
}

/**
 * The song's model inputs (core._Song): 16-bit audio, note scalars, taps.
 * @param {Float32Array} y 16 kHz audio (retuned)
 * @param {Float32Array} meta [N][12]
 * @param {Int8Array} feat [N][3][368]
 * @param {Float32Array} fsc [N][3]
 * @returns {{audio: Int16Array, meta: Float32Array, feat: Int8Array, fsc: Float32Array, n: number}}
 */
export function makeSong(y, meta, feat, fsc) {
  const audio = new Int16Array(y.length);
  for (let i = 0; i < y.length; i++) audio[i] = Math.min(Math.max(roundHalfEven(y[i] * 32768), -32768), 32767);
  return { audio, meta, feat, fsc, n: meta.length / META_DIM };
}

function audioSlice(audio, tA, L) {
  const x = new Float32Array(L);
  const s0 = roundHalfEven(tA * SR);
  const lo = Math.max(s0, 0);
  const hi = Math.min(s0 + L, audio.length);
  for (let i = lo; i < hi; i++) x[i - s0] = audio[i] / 32768;
  return x;
}

/**
 * core._Song.window (with Grouper.predict's fallback for windows of fewer than 4 notes: the notes repeated to 4).
 * @param {{audio: Int16Array, meta: Float32Array, feat: Int8Array, fsc: Float32Array}} song from makeSong
 * @param {number} t0 window start (s)
 * @param {number} W window length (s)
 * @param {ArrayLike<number>} sel note indices
 * @returns {{audio: Float32Array, notes: Float32Array, feat: Float32Array, m: number}} notes [m][12], feat [m][3][368]
 *   (float32 int8 * scale, before the float16 cast of collate)
 */
export function windowInputs(song, t0, W, sel) {
  const idx = sel.length >= 4 ? Array.from(sel) : Array.from({ length: 4 }, (_, k) => sel[k % sel.length]);
  const m = idx.length;
  const notes = new Float32Array(m * META_DIM);
  const feat = new Float32Array(m * 3 * FEAT_DIM);
  const t0f = f32(t0);
  const lim = f32(W + PAD1);
  idx.forEach((s, r) => {
    notes.set(song.meta.subarray(s * META_DIM, (s + 1) * META_DIM), r * META_DIM);
    notes[r * META_DIM] = f32(notes[r * META_DIM] - t0f);
    notes[r * META_DIM + 1] = Math.min(f32(notes[r * META_DIM + 1] - t0f), lim);
    for (let k = 0; k < 3; k++) {
      const sc = song.fsc[s * 3 + k];
      const src = (s * 3 + k) * FEAT_DIM;
      const dst = (r * 3 + k) * FEAT_DIM;
      for (let c = 0; c < FEAT_DIM; c++) feat[dst + c] = f32(song.feat[src + c] * sc);
    }
  });
  const audio = audioSlice(song.audio, t0 - PAD0, roundHalfEven((PAD0 + W + PAD1) * SR));
  return { audio, notes, feat, m };
}

// ============================================================================= the network over the windows

/**
 * One window through grouper.onnx (batch 1, exactly the window's notes).
 * @param {{ort: any, session: (name: string) => Promise<any>}} rt runtime handle
 * @param {{X: Float32Array, P: Float32Array, T: number}} front the window's front end ([9][T][288], [425][T])
 * @param {{notes: Float32Array, feat: Float32Array, m: number}} win from windowInputs
 * @returns {Promise<{probs: Float32Array, emb: Float32Array, h: Float32Array}>} [m][73], [m][32], [m][176]
 */
export async function runWindow(rt, front, win) {
  const { ort } = rt;
  const sess = await rt.session("grouper");
  const { m } = win;
  const feat16 = Float32Array.from(win.feat, roundF16); // collate stores feat as float16
  const feeds = {
    X: new ort.Tensor("float32", front.X, [1, 9, front.T, front.X.length / (9 * front.T)]),
    P: new ort.Tensor("float32", front.P, [1, front.P.length / front.T, front.T]),
    notes: new ort.Tensor("float32", win.notes, [1, m, META_DIM]),
    mask: new ort.Tensor("bool", new Uint8Array(m).fill(1), [1, m]),
    feat: new ort.Tensor("float32", feat16, [1, m, 3, FEAT_DIM]),
  };
  const o = await sess.run(feeds);
  return { probs: o.probs.data, emb: o.emb.data, h: o.h.data };
}

const W_LO32 = f32(0.05);

/** The window weight of an onset in float32 (Grouper.predict: on_all is float32). */
function weight32(on, t0f, W) {
  const c = f32(Math.abs(f32(f32(f32(on - t0f) / f32(W)) - 0.5)));
  return Math.min(Math.max(f32(1 - f32(c * 2)), W_LO32), 1);
}

/** The same in float64 (decode.group_chain: on is float64). */
function weight64(on, t0, W) {
  return Math.min(Math.max(1 - Math.abs((on - t0) / W - 0.5) * 2, 0.05), 1);
}

function accumulate(st, t0, sel, o) {
  const { K, E, D, W, song } = st;
  const t0f = f32(t0);
  for (let r = 0; r < sel.length; r++) {
    const s = sel[r];
    const w = weight32(song.meta[s * META_DIM], t0f, W);
    for (let k = 0; k < K; k++) st.acc[s * K + k] += f32(o.probs[r * K + k] * w);
    st.wsum[s] += w;
    for (let k = 0; k < E; k++) st.eacc[s * E + k] += f32(o.emb[r * E + k] * w);
    if (w > st.best[s]) {
      for (let k = 0; k < D; k++) st.h[s * D + k] = roundF16(o.h[r * D + k]);
      st.best[s] = w;
    }
  }
  st.wins.push({
    t0,
    sel: Int32Array.from(sel),
    emb: Float32Array.from(o.emb.subarray(0, sel.length * E), roundF16),
    probs: Float32Array.from(o.probs.subarray(0, sel.length * K), roundF16),
  });
}

function finishPredict(st) {
  const { n, K, E, D } = st;
  const probs = new Float32Array(n * K);
  const emb = new Float32Array(n * E);
  for (let i = 0; i < n; i++) {
    const ws = Math.max(st.wsum[i], 1e-9);
    for (let k = 0; k < K; k++) probs[i * K + k] = st.acc[i * K + k] / ws;
    const nrm = Math.max(rowNorm(st.eacc, i * E, E), 1e-9);
    for (let k = 0; k < E; k++) emb[i * E + k] = st.eacc[i * E + k] / nrm;
  }
  return { probs, emb, h: st.h ?? new Float32Array(n * D), wins: st.wins, W: st.W, n, K, E, D };
}

/**
 * Grouper.predict: per-note instrument probabilities, track embeddings and note states averaged over the windows that hold
 * the note (weighted towards the window centre; h from the most central window).
 * @param {{ort: any, session: (name: string) => Promise<any>}} rt runtime handle
 * @param {{audio: Int16Array, meta: Float32Array, feat: Int8Array, fsc: Float32Array, n: number}} song from makeSong
 * @param {(audio16k: Float32Array) => ({X: Float32Array, P: Float32Array, T: number} | Promise<{X: Float32Array,
 *   P: Float32Array, T: number}>)} frontFn the grouper front end (dsp.grouperFront)
 * @param {{window_s: number, window_hop: number, max_notes: number}} cfg meta.json's grouper section
 * @param {(done: number, total: number) => void} [onProgress]
 * @returns {Promise<{probs: Float32Array, emb: Float32Array, h: Float32Array, wins: {t0: number, sel: Int32Array,
 *   emb: Float32Array, probs: Float32Array}[], W: number, n: number, K: number, E: number, D: number}>}
 */
export async function predict(rt, song, frontFn, cfg, onProgress) {
  const W = cfg.window_s;
  const n = song.n;
  const on = song.meta.filter((_, i) => i % META_DIM === 0);
  const jobs = songWindows(on, W, W * cfg.window_hop, cfg.max_notes);
  const st = { n, W, song, K: 73, E: 32, D: 176, acc: null, eacc: null, h: null, wins: [] };
  st.wsum = new Float64Array(n);
  st.best = new Float64Array(n).fill(-1);
  for (let j = 0; j < jobs.length; j++) {
    const { t0, sel } = jobs[j];
    const win = windowInputs(song, t0, W, sel);
    const o = await runWindow(rt, await frontFn(win.audio), win);
    if (!st.acc) {
      st.K = o.probs.length / win.m;
      st.E = o.emb.length / win.m;
      st.D = o.h.length / win.m;
      st.acc = new Float64Array(n * st.K);
      st.eacc = new Float64Array(n * st.E);
      st.h = new Float32Array(n * st.D);
    }
    accumulate(st, t0, sel, o);
    if (onProgress) onProgress(j + 1, jobs.length);
  }
  if (!st.acc) {
    st.acc = new Float64Array(n * st.K);
    st.eacc = new Float64Array(n * st.E);
  }
  return finishPredict(st);
}

// ============================================================================= decode: tracklets

/** Gram row in float64 (the reference's BLAS products, summed in order). */
function dot(a, ao, b, bo, n) {
  let s = 0;
  for (let k = 0; k < n; k++) s += a[ao + k] * b[bo + k];
  return s;
}

/** Unit rows in float32 (NumPy float32 norm + divide). */
function unitRows32(p, m, K) {
  const out = new Float32Array(m * K);
  const lo = f32(1e-9);
  for (let i = 0; i < m; i++) {
    const nrm = Math.max(rowNorm32(p, i * K, K), lo);
    for (let k = 0; k < K; k++) out[i * K + k] = f32(p[i * K + k] / nrm);
  }
  return out;
}

/** Unit rows in float64 (rows of a float64 matrix over max(norm, 1e-9)). */
function unitRows(a, m, K) {
  const out = new Float64Array(m * K);
  for (let i = 0; i < m; i++) {
    const nrm = Math.max(rowNorm(a, i * K, K), 1e-9);
    for (let k = 0; k < K; k++) out[i * K + k] = a[i * K + k] / nrm;
  }
  return out;
}

function windowDistances(e, p, m, E, K, lam) {
  const Dm = new Float32Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) Dm[i * m + j] = f32(1 - f32(dot(e, i * E, e, j * E, E)));
  if (lam > 0) {
    const pn = unitRows32(p, m, K);
    const l32 = f32(lam);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const g = f32(1 - f32(dot(pn, i * K, pn, j * K, K)));
        Dm[i * m + j] = f32(Dm[i * m + j] + f32(l32 * g));
      }
    }
  }
  const y = new Float64Array((m * (m - 1)) / 2);
  let k = 0;
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) y[k++] = Math.max(f32(f32(Dm[i * m + j] + Dm[j * m + i]) / 2), 0);
  }
  return y;
}

/**
 * decode._cluster: one window's average-linkage clusters on (1 - emb cosine) + lam (1 - instrument cosine), cut at tau.
 * @param {Float32Array} e [m][E] (float16 values)
 * @param {Float32Array} p [m][K] (float16 values)
 * @param {number} m
 * @param {number} tau
 * @param {number} lam
 * @returns {Int32Array} 0-based labels in fcluster's numbering
 */
export function clusterWindow(e, p, m, tau, lam) {
  if (m === 1) return new Int32Array(1);
  const y = windowDistances(e, p, m, e.length / m, p.length / m, lam);
  return fclusterDistance(linkageAverage(y, m), m, tau).map((v) => v - 1);
}

function sortedUnique(vals) {
  return Int32Array.from(new Set(vals)).sort();
}

function chainLabels(lab, nl, prev) {
  const globOf = new Int32Array(nl).fill(-1);
  const seen = [];
  prev.forEach((g) => g >= 0 && seen.push(g));
  if (seen.length === 0) return globOf;
  const gids = sortedUnique(seen);
  const col = new Map(Array.from(gids, (g, c) => [g, c]));
  const nc = gids.length;
  const ov = new Float64Array(nl * nc);
  prev.forEach((g, r) => {
    if (g >= 0) ov[lab[r] * nc + col.get(g)] += 1;
  });
  const { rows, cols } = linearSumAssignment(
    ov.map((v) => -v),
    nl,
    nc,
  );
  for (let q = 0; q < rows.length; q++) if (ov[rows[q] * nc + cols[q]] > 0) globOf[rows[q]] = gids[cols[q]];
  return globOf;
}

/**
 * decode.group_chain: per-window clusters chained across windows by shared notes (Hungarian matching on the overlap).
 * @param {{wins: {t0: number, sel: Int32Array, emb: Float32Array, probs: Float32Array}[], W: number}} P predictions
 * @param {ArrayLike<number>} on onsets (float64)
 * @param {number} tau
 * @param {number} lam
 * @returns {Int32Array} tracklet id per note (-1 = none)
 */
export function groupChain(P, on, tau, lam) {
  const n = on.length;
  const track = new Int32Array(n).fill(-1);
  const bestW = new Float64Array(n).fill(-1);
  const owner = new Int32Array(n).fill(-1);
  let nGlob = 0;
  const wins = P.wins.map((w, i) => [w, i]).sort((a, b) => a[0].t0 - b[0].t0 || a[1] - b[1]);
  for (const [{ t0, sel, emb, probs }] of wins) {
    if (sel.length === 0) continue;
    const lab = clusterWindow(emb, probs, sel.length, tau, lam);
    const nl = Math.max(...lab) + 1;
    const globOf = chainLabels(
      lab,
      nl,
      Array.from(sel, (s) => owner[s]),
    );
    for (let li = 0; li < nl; li++) if (globOf[li] < 0) globOf[li] = nGlob++;
    sel.forEach((s, r) => {
      const g = globOf[lab[r]];
      owner[s] = g;
      const w = weight64(on[s], t0, P.W);
      if (w > bestW[s]) {
        track[s] = g;
        bestW[s] = w;
      }
    });
  }
  return track;
}

/** np.unique(v, return_inverse=True) */
function uniqueInverse(v) {
  const tids = sortedUnique(v);
  const pos = new Map(Array.from(tids, (t, i) => [t, i]));
  return { tids, inv: Int32Array.from(v, (x) => pos.get(x)) };
}

/** Row sums by group (np.add.at: sequential), float64. */
function groupSums(src, rows, inv, M, D) {
  const out = new Float64Array(M * D);
  rows.forEach((r, q) => {
    const g = inv[q];
    for (let k = 0; k < D; k++) out[g * D + k] += src[r * D + k];
  });
  return out;
}

function bincount(inv, M) {
  const c = new Float64Array(M);
  for (const g of inv) c[g]++;
  return c;
}

// ----------------------------------------------------------------------------- _agglo

function rowMinScan(Dm, M, r, rowMin, rowArg) {
  let mv = Infinity;
  let ma = 0;
  for (let c = 0; c < M; c++) {
    if (Dm[r * M + c] < mv) {
      mv = Dm[r * M + c];
      ma = c;
    }
  }
  rowMin[r] = mv;
  rowArg[r] = ma;
}

function aggloDistances(spaces, M) {
  const Dm = new Float64Array(M * M);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      let d = 0;
      for (const { m, w, D } of spaces) d += w * (1 - dot(m, i * D, m, j * D, D));
      Dm[i * M + j] = i === j ? Infinity : d;
    }
  }
  return Dm;
}

function mergeMeans(spaces, size, i, j) {
  const si = size[i];
  const sj = size[j];
  for (const { m, D } of spaces) {
    for (let k = 0; k < D; k++) m[i * D + k] = (si * m[i * D + k] + sj * m[j * D + k]) / (si + sj);
  }
  size[i] = si + sj;
}

/**
 * decode._agglo: size-weighted average linkage of M groups until the closest pair is farther than tau2.
 * @param {{m: Float64Array, w: number}[]} spacesIn per space the group means [M][D] and its weight
 * @param {Float64Array} sizeIn group sizes
 * @param {number} tau2
 * @param {number} [cap] beyond this many groups only the largest merge freely
 * @returns {Int32Array} a label per group
 */
export function agglo(spacesIn, sizeIn, tau2, cap = AGGLO_CAP) {
  const M = sizeIn.length;
  const spaces = spacesIn.filter((s) => s.w > 0).map((s) => ({ m: Float64Array.from(s.m), w: s.w, D: s.m.length / M }));
  const size = Float64Array.from(sizeIn);
  const lab = Int32Array.from({ length: M }, (_, i) => i);
  if (M < 2) return lab;
  const Dm = aggloDistances(spaces, M);
  const alive = new Uint8Array(M).fill(1);
  if (M > cap) {
    const order = argsortStable(M, (i) => -size[i]); // np.argsort(-size) is not stable: see the report
    for (let q = cap; q < M; q++) {
      const s = order[q];
      for (let c = 0; c < M; c++) Dm[s * M + c] = Dm[c * M + s] = Infinity;
    }
  }
  const rowMin = new Float64Array(M);
  const rowArg = new Int32Array(M);
  for (let r = 0; r < M; r++) rowMinScan(Dm, M, r, rowMin, rowArg);
  for (;;) {
    let i = 0;
    for (let r = 1; r < M; r++) if (rowMin[r] < rowMin[i]) i = r;
    const j = rowArg[i];
    const dij = Dm[i * M + j];
    if (!Number.isFinite(dij) || dij > tau2) break;
    mergeMeans(spaces, size, i, j);
    alive[j] = 0;
    for (let q = 0; q < M; q++) if (lab[q] === j) lab[q] = i;
    for (let r = 0; r < M; r++) {
      let d = Infinity;
      if (alive[r] && r !== i) {
        d = 0;
        for (const { m, w, D } of spaces) d += w * (1 - dot(m, r * D, m, i * D, D));
      }
      Dm[j * M + r] = Dm[r * M + j] = Infinity;
      Dm[i * M + r] = Dm[r * M + i] = d;
    }
    for (let r = 0; r < M; r++) {
      if (r === i || r === j || rowArg[r] === i || rowArg[r] === j) rowMinScan(Dm, M, r, rowMin, rowArg);
      else {
        const d = Dm[r * M + i];
        if (d < rowMin[r] || (d === rowMin[r] && i < rowArg[r])) {
          rowMin[r] = d;
          rowArg[r] = i;
        }
      }
    }
  }
  return lab;
}

/**
 * decode.group_tracklet: tracklets, then the whole song's tracklets merged.
 * @param {object} P predictions (probs, emb, wins, W)
 * @param {ArrayLike<number>} on onsets
 * @param {number} tau1
 * @param {number} tau2
 * @param {number} lam
 * @returns {Int32Array} track id per note (-1 = none)
 */
export function groupTracklet(P, on, tau1, tau2, lam) {
  const tr = groupChain(P, on, tau1, lam);
  const idx = [];
  tr.forEach((t, i) => t >= 0 && idx.push(i));
  if (idx.length === 0) return tr;
  const { tids, inv } = uniqueInverse(idx.map((i) => tr[i]));
  const M = tids.length;
  if (M === 1) return tr;
  const n = on.length;
  const E = P.emb.length / n;
  const K = P.probs.length / n;
  const pm = unitRows(Float64Array.from(P.probs), n, K);
  const size = bincount(inv, M);
  const me = groupSums(P.emb, idx, inv, M, E);
  const mp = groupSums(pm, idx, inv, M, K);
  for (let g = 0; g < M; g++) {
    for (let k = 0; k < E; k++) me[g * E + k] /= size[g];
    for (let k = 0; k < K; k++) mp[g * K + k] /= size[g];
  }
  const lab = agglo(
    [
      { m: me, w: 1.0 },
      { m: mp, w: lam },
    ],
    size,
    tau2,
  );
  const out = new Int32Array(tr.length).fill(-1);
  idx.forEach((i, q) => (out[i] = lab[inv[q]]));
  return out;
}

// ============================================================================= decode: tracks and names

/**
 * decode.track_stats: pitch mean / std / min / max, log size, polyphony share.
 * @param {ArrayLike<number>} on
 * @param {ArrayLike<number>} off
 * @param {ArrayLike<number>} pitch
 * @returns {Float64Array} 6 values
 */
export function trackStats(on, off, pitch) {
  const n = pitch.length;
  const p = Float64Array.from(pitch, (v) => v / 64 - 1);
  const mean = npSum(p) / n;
  const sq = Float64Array.from(p, (v) => (v - mean) * (v - mean));
  let poly = 0;
  if (n <= 3000) {
    let cnt = 0;
    for (let i = 0; i < n; i++) {
      let c = 0;
      for (let j = 0; j < n; j++) if (Math.min(off[i], off[j]) > Math.max(on[i], on[j]) + 0.03) c++;
      if (c > 1.5) cnt++;
    }
    poly = cnt / n;
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of p) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return Float64Array.of(mean, Math.sqrt(npSum(sq) / n), lo, hi, Math.log1p(n) / 5, poly);
}

/** log(softmax) of a float64 vector with NumPy's order. */
function logSoftmax(x) {
  let mx = -Infinity;
  for (const v of x) mx = Math.max(mx, v);
  const lt = Float64Array.from(x, (v) => v - mx);
  const ls = Math.log(npSum(lt.map(Math.exp)));
  return lt.map((v) => v - ls);
}

const SQRT2 = Math.sqrt(2);

/**
 * The track head (float64, exact-erf GELU between layers).
 * @param {{W: Float64Array, b: Float64Array, rows: number, cols: number}[]} th
 * @param {Float64Array} x0
 * @returns {Float64Array}
 */
export function trackHead(th, x0) {
  let x = x0;
  th.forEach(({ W, b, rows, cols }, li) => {
    const y = new Float64Array(rows);
    for (let r = 0; r < rows; r++) y[r] = dot(W, r * cols, x, 0, cols) + b[r];
    if (li < th.length - 1) for (let r = 0; r < rows; r++) y[r] = 0.5 * y[r] * (1 + erf(y[r] / SQRT2));
    x = y;
  });
  return x;
}

function columnSum(src, ids, D) {
  const out = new Float64Array(D);
  for (const i of ids) for (let k = 0; k < D; k++) out[k] += src[i * D + k];
  return out;
}

/**
 * decode.track_logits: alpha log p(track head) + (1 - alpha) log (summed note probabilities).
 * @param {ArrayLike<number>} ids note ids of the track
 * @param {object} P predictions (probs, h)
 * @param {ArrayLike<number>[]} notes [on, off, pitch]
 * @param {object[] | null} th track head layers
 * @param {number} alpha
 * @returns {Float64Array} K log-probabilities
 */
export function trackLogits(ids, P, notes, th, alpha) {
  const n = notes[0].length;
  const K = P.probs.length / n;
  const pn = columnSum(P.probs, ids, K);
  const tot = npSum(pn);
  const ln = pn.map((v) => Math.log(Math.max(v / tot, 1e-9)));
  if (!th || alpha <= 0 || !P.h) return ln;
  const [on, off, pitch] = notes;
  const D = P.h.length / n;
  const hm = columnSum(P.h, ids, D).map((v) => v / ids.length);
  const st = trackStats(
    Array.from(ids, (i) => on[i]),
    Array.from(ids, (i) => Math.max(off[i], on[i] + 0.02)),
    Array.from(ids, (i) => pitch[i]),
  );
  const x = new Float64Array(D + st.length);
  x.set(hm);
  x.set(st, D);
  const lt = logSoftmax(trackHead(th, x));
  return lt.map((v, k) => alpha * v + (1 - alpha) * ln[k]);
}

function argmaxRow(a, off, n) {
  let b = 0;
  for (let k = 1; k < n; k++) if (a[off + k] > a[off + b]) b = k;
  return b;
}

function refineOnce(inv, size, emb, pnU, probsIdx, T, E, K, lam, mn) {
  const keepc = [];
  size.forEach((s, t) => s >= mn && keepc.push(t));
  if (keepc.length < 2) return null;
  const ids = Int32Array.from({ length: inv.length }, (_, i) => i);
  const cenAll = groupSums(emb, ids, inv, T, E);
  const cen = unitRows(
    Float64Array.from(keepc.flatMap((t) => Array.from(cenAll.subarray(t * E, (t + 1) * E)))),
    keepc.length,
    E,
  );
  let pm = null;
  if (lam > 0) {
    const pmAll = groupSums(probsIdx, ids, inv, T, K);
    pm = unitRows(
      Float64Array.from(keepc.flatMap((t) => Array.from(pmAll.subarray(t * K, (t + 1) * K)))),
      keepc.length,
      K,
    );
  }
  const nk = keepc.length;
  const sc = new Float64Array(nk);
  const out = new Int32Array(inv.length);
  for (let q = 0; q < inv.length; q++) {
    for (let c = 0; c < nk; c++) {
      sc[c] = dot(emb, q * E, cen, c * E, E);
      if (pm) sc[c] = sc[c] + lam * dot(pnU, q * K, pm, c * K, K);
    }
    out[q] = keepc[argmaxRow(sc, 0, nk)];
  }
  return out;
}

function foldSmall(inv, size, probsIdx, T, K, mn) {
  const ids = Int32Array.from({ length: inv.length }, (_, i) => i);
  const mass = groupSums(probsIdx, ids, inv, T, K);
  let big = [];
  size.forEach((s, t) => s >= mn && big.push(t));
  if (big.length === 0) big = [argmaxRow(size, 0, T)];
  const isbig = new Uint8Array(T);
  big.forEach((t) => (isbig[t] = 1));
  const bign = unitRows(
    Float64Array.from(big.flatMap((t) => Array.from(mass.subarray(t * K, (t + 1) * K)))),
    big.length,
    K,
  );
  const srcV = unitRows(mass, T, K);
  const sc = new Float64Array(big.length);
  for (let t = 0; t < T; t++) {
    if (isbig[t] || size[t] <= 0) continue;
    for (let c = 0; c < big.length; c++) sc[c] = dot(bign, c * K, srcV, t * K, K);
    const to = big[argmaxRow(sc, 0, big.length)];
    for (let q = 0; q < inv.length; q++) if (inv[q] === t) inv[q] = to;
  }
}

function dissolveOne(inv, e, n, E, frac, tauA) {
  const { tids, inv: lab } = uniqueInverse(inv);
  const M = tids.length;
  if (M < 2) return false;
  const size = bincount(lab, M);
  const cen = groupSums(
    e,
    Int32Array.from({ length: n }, (_, i) => i),
    lab,
    M,
    E,
  );
  for (let t = 0; t < M; t++) for (let k = 0; k < E; k++) cen[t * E + k] /= size[t];
  for (const t of argsortStable(M, (i) => size[i])) {
    const s = size[t] / n;
    if (s >= frac) break;
    const big = [];
    size.forEach((z, b) => z > size[t] && big.push(b));
    if (big.length === 0) continue;
    const thr = tauA * (1 - s / frac);
    if (!big.some((b) => 1 - dot(cen, b * E, cen, t * E, E) - thr < 0)) continue;
    const cu = unitRows(
      Float64Array.from(big.flatMap((b) => Array.from(cen.subarray(b * E, (b + 1) * E)))),
      big.length,
      E,
    );
    const sc = new Float64Array(big.length);
    for (let q = 0; q < n; q++) {
      if (lab[q] !== t) continue;
      for (let c = 0; c < big.length; c++) sc[c] = dot(e, q * E, cu, c * E, E);
      inv[q] = tids[big[argmaxRow(sc, 0, big.length)]];
    }
    return true;
  }
  return false;
}

/**
 * decode.absorb_small: a track with less than `frac` of the notes survives only if it is clearly its own instrument.
 * @param {Int32Array} invIn track id per note
 * @param {Float64Array} emb [n][E]
 * @param {number} frac
 * @param {number} tauA
 * @returns {Int32Array} new track ids
 */
export function absorbSmall(invIn, emb, frac, tauA) {
  const inv = Int32Array.from(invIn);
  const n = inv.length;
  if (n === 0 || frac <= 0 || tauA <= 0) return inv;
  const E = emb.length / n;
  const e = unitRows(emb, n, E);
  while (dissolveOne(inv, e, n, E, frac, tauA));
  return inv;
}

function nameTrack(ids, P, notes, th, prm, credit) {
  const z = trackLogits(ids, P, notes, th, prm.alpha);
  const K = z.length;
  if (!prm.ec) return argmaxRow(z, 0, K);
  let mx = -Infinity;
  for (const v of z) mx = Math.max(mx, v);
  const q = z.map((v) => Math.exp(v - mx));
  const qs = npSum(q);
  const qn = q.map((v) => v / qs);
  const ex = new Float64Array(K);
  for (let a = 0; a < K; a++) ex[a] = dot(credit, a * K, qn, 0, K);
  return argmaxRow(ex, 0, K);
}

const DECODE_DEFAULTS = {
  tau: 0.1,
  tau2: 0.4,
  lam: 0.5,
  min_notes: 30,
  refine: 2,
  alpha: 0.5,
  ec: true,
  merge: true,
  absorb: 0.3,
  absorb_frac: 0.25,
};

function initialTracks(P, on, prm) {
  const track = groupTracklet(P, on, prm.tau, prm.tau2, prm.lam);
  const idx = [];
  track.forEach((t, i) => t >= 0 && idx.push(i));
  return { idx: Int32Array.from(idx), ...uniqueInverse(idx.map((i) => track[i])) };
}

function selectRows(src, idx, D) {
  const out = new Float64Array(idx.length * D);
  idx.forEach((i, q) => {
    for (let k = 0; k < D; k++) out[q * D + k] = src[i * D + k];
  });
  return out;
}

function nameTracks(inv, idx, P, notes, th, prm, credit) {
  const out = new Map();
  for (const t of sortedUnique(inv)) {
    const ids = [];
    inv.forEach((v, q) => v === t && ids.push(idx[q]));
    const k = nameTrack(ids, P, notes, th, prm, credit);
    const key = prm.merge ? k : `${k}:${t}`;
    if (prm.merge && out.has(key)) out.get(key)[1].push(...ids);
    else out.set(key, [k, ids]);
  }
  return [...out.values()].map(([k, ids]) => [k, Int32Array.from(ids).sort()]);
}

/**
 * decode.decode: per-note predictions -> [(instrument index, sorted note ids)].
 * @param {{probs: Float32Array, emb: Float32Array, h: Float32Array | null, wins: object[], W: number}} P from predict
 * @param {ArrayLike<number>[]} notes [on, off, pitch] (float64)
 * @param {{W: Float64Array, b: Float64Array, rows: number, cols: number}[] | null} th track head (loadTrackHead)
 * @param {object} params meta.json grouper.decode (tau, tau2, lam, min_notes, refine, alpha, ec, merge, absorb, absorb_frac)
 * @param {Float64Array | number[][]} credit [K][K] expected-credit matrix (meta.json instruments.credit)
 * @returns {[number, Int32Array][]}
 */
export function decode(P, notes, th, params, credit) {
  const prm = { ...DECODE_DEFAULTS, ...params };
  const cr = Float64Array.from(Array.isArray(credit[0]) ? credit.flat() : credit);
  const [on] = notes;
  const n = on.length;
  if (n === 0) return [];
  const { idx, tids, inv: inv0 } = initialTracks(P, on, prm);
  if (idx.length === 0) return [];
  const T = tids.length;
  const K = P.probs.length / n;
  const E = P.emb.length / n;
  const emb = selectRows(P.emb, idx, E);
  const probsIdx = selectRows(P.probs, idx, K);
  const pnU = unitRows(probsIdx, idx.length, K);
  let inv = inv0;
  let size = bincount(inv, T);
  for (let r = 0; r < prm.refine; r++) {
    const next = refineOnce(inv, size, emb, pnU, probsIdx, T, E, K, prm.lam, prm.min_notes);
    if (!next) break;
    inv = next;
    size = bincount(inv, T);
  }
  foldSmall(inv, size, probsIdx, T, K, prm.min_notes);
  if (prm.absorb > 0 && prm.absorb_frac > 0) inv = absorbSmall(inv, emb, prm.absorb_frac, prm.absorb);
  return nameTracks(inv, idx, P, notes, th, prm, cr);
}

// ============================================================================= tracks

function dedupTrack(ids, on, pitch) {
  const o = Array.from(ids).sort((a, b) => pitch[a] - pitch[b] || on[a] - on[b] || 0);
  const keep = [];
  let last = -1;
  for (const i of o) {
    if (last >= 0 && pitch[i] === pitch[last] && on[i] - on[last] <= DEDUP) continue;
    keep.push(i);
    last = i;
  }
  return { keep, merged: o.length - keep.length };
}

function velocity(c) {
  const v = Math.min(Math.max(f32(20 + f32(107 * c)), 1), 127);
  return Math.trunc(v);
}

/**
 * Grouper._tracks: decoded groups -> tracks, the largest first (same-pitch notes within 30 ms in one track are one note).
 * @param {ArrayLike<number>} on onsets (s, original axis)
 * @param {ArrayLike<number>} off offsets
 * @param {ArrayLike<number>} pitch integer MIDI pitches
 * @param {ArrayLike<number>} conf float32 note confidences
 * @param {[number, ArrayLike<number>][]} groups from decode
 * @param {{names: string[], programs: number[]}} instruments meta.json instruments
 * @returns {{name: string, instrument: string, program: number, isDrum: boolean, notes: number[][], mergedDuplicates: number}[]}
 */
export function tracks(on, off, pitch, conf, groups, instruments) {
  const order = groups.map((g, i) => [g, i]).sort((a, b) => b[0][1].length - a[0][1].length || a[1] - b[1]);
  return order.map(([[k, ids]]) => {
    const { keep, merged } = dedupTrack(ids, on, pitch);
    const rows = keep
      .map((i, q) => [i, q])
      .sort((a, b) => on[a[0]] - on[b[0]] || a[1] - b[1])
      .map(([i]) => [on[i], Math.max(off[i], on[i] + 0.02), pitch[i], velocity(conf[i])]);
    const name = instruments.names[k];
    return {
      name,
      instrument: name,
      program: instruments.programs[k],
      isDrum: false,
      notes: rows,
      mergedDuplicates: merged,
    };
  });
}

// ============================================================================= all of it

/**
 * The track head from web/models/track_head.bin (float64 LE [W1, b1, W2, b2, W3, b3]) and meta.grouper.track_head_shapes.
 * @param {ArrayBuffer} buf
 * @param {[[number, number], [number]][]} shapes
 * @returns {{W: Float64Array, b: Float64Array, rows: number, cols: number}[]}
 */
export function loadTrackHead(buf, shapes) {
  const all = new Float64Array(buf);
  let o = 0;
  return shapes.map(([[rows, cols]]) => {
    const W = all.slice(o, o + rows * cols);
    o += rows * cols;
    const b = all.slice(o, o + rows);
    o += rows;
    return { W, b, rows, cols };
  });
}

function noteColumns(meta, n) {
  const col = (c) => Float64Array.from({ length: n }, (_, i) => meta[i * META_DIM + c]);
  const conf = Float32Array.from({ length: n }, (_, i) =>
    meta[i * META_DIM + 3] > 0.5 ? meta[i * META_DIM + 5] : meta[i * META_DIM + 4],
  );
  return { on: col(0), off: col(1), pitch: col(2).map(roundHalfEven), conf };
}

const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

/**
 * Grouper.group with a precomputed transcriber run: notes already on the retuned axis, tracks on the original axis (x r).
 * @param {{ort: any, session: (name: string) => Promise<any>}} rt runtime handle
 * @param {Float32Array} y16k the retuned 16 kHz audio
 * @param {ArrayLike<ArrayLike<number>>} repairedNotes [[onset, offset, pitch]] (retuned axis)
 * @param {ArrayLike<ArrayLike<number>>} ownNotes the transcriber's notes on the sub maps [[onset, offset, pitch, slot 0 / 1]]
 * @param {object} smaps the sub-run maps {onset, ..., offset2} [T][88]
 * @param {{vec: Function}} store tap store (subHidden / subStem)
 * @param {{r: number, delta?: number, strength?: number, shift?: number}} tuning dsp.retuneRatio result
 * @param {Function} frontFn grouper front end (audio16k -> {X, P, T})
 * @param {{window_s: number, window_hop: number, max_notes: number, decode: object, th: object[], instruments: object,
 *   hop?: number}} cfg meta.grouper + th (loadTrackHead) + instruments (meta.instruments)
 * @param {(done: number, total: number) => void} [onProgress]
 * @returns {Promise<{tracks: object[], info: object}>}
 */
export async function group(rt, y16k, repairedNotes, ownNotes, smaps, store, tuning, frontFn, cfg, onProgress) {
  const r = tuning.r;
  const nt = assignSlots(repairedNotes, ownNotes);
  const { meta, feat, fsc } = gather(smaps, store, nt, cfg.hop ?? HOP);
  const song = makeSong(y16k, meta, feat, fsc);
  const { on, off, pitch, conf } = noteColumns(meta, song.n);
  let groups = [];
  if (song.n) {
    const P = await predict(rt, song, frontFn, cfg, onProgress);
    groups = decode(P, [on, off, pitch], cfg.th, cfg.decode, cfg.instruments.credit);
  }
  const out = tracks(
    on.map((v) => v * r),
    off.map((v) => v * r),
    pitch,
    conf,
    groups,
    cfg.instruments,
  );
  const info = {
    tuningCents: round(tuning.delta ?? 0, 2),
    tuningStrength: round(tuning.strength ?? 0, 3),
    retuneCents: round(tuning.shift ?? 0, 2),
    retuneRatio: r,
    durationS: round((y16k.length * r) / SR, 2),
    notesIn: song.n,
    notesOut: out.reduce((s, t) => s + t.notes.length, 0),
  };
  return { tracks: out, info };
}
