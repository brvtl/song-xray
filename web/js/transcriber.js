// Transcriber helpers: the windowed-pass layout and the map -> notes decode (peak picking, no learned parameters).
// Port of dettato/transcriber/core.py (Transcriber._layout, frame_offset, notes) and dettato/transcriber/decode.py.

export const SR = 16000;
export const HOP = 0.02;
export const WIN_S = 8.0;
export const N_KEYS = 88;
export const PITCH_LO = 21;

const fr = Math.fround;

/**
 * Python's round(x) / numpy's rint: round half to even.
 * @param {number} x
 * @returns {number}
 */
export function roundHalfEven(x) {
  const r = Math.round(x); // half towards +inf
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * Transcriber._layout: window geometry of the overlapping 8 s pass.
 * @param {number} n samples at 16 kHz
 * @param {number} hopFrac window hop as a fraction of the window (0.25 -> every 2 s)
 * @param {number} [hop] output frame hop in seconds
 * @returns {{W: number, Fw: number, T: number, starts: number[], pad: number, gPad: number, step: number}}
 */
export function layout(n, hopFrac, hop = HOP) {
  const W = roundHalfEven(WIN_S * SR);
  const Fw = roundHalfEven(WIN_S / hop);
  const T = Math.ceil(n / SR / hop);
  const step = Math.max(1, roundHalfEven(W * hopFrac));
  const pad = Math.floor(W / 2);
  const starts = [];
  for (let s = 0; s < n + pad; s += step) starts.push(s);
  const gPad = roundHalfEven(pad / SR / hop);
  return { W, Fw, T, starts, pad, gPad, step };
}

/**
 * Song frame of window frame 0 for a window starting at padded sample s0 (Transcriber.frame_offset).
 * @param {number} s0
 * @param {number} gPad
 * @param {number} [hop]
 * @returns {number}
 */
export function frameOffset(s0, gPad, hop = HOP) {
  return roundHalfEven(s0 / SR / hop) - gPad;
}

/**
 * The overlap-add taper: np.hanning(Fw + 2)[1:-1] as float32.
 * @param {number} Fw
 * @returns {Float32Array}
 */
export function hannTaper(Fw) {
  const M = Fw + 2;
  const w = new Float32Array(Fw);
  for (let i = 1; i <= Fw; i++) w[i - 1] = 0.5 + 0.5 * Math.cos((Math.PI * (1 - M + 2 * i)) / (M - 1));
  return w;
}

/** (T, 88) -> mask of local maxima >= th (plateaus: first frame), as decode._peaks. */
function peaks(p, T, th) {
  const thf = fr(th);
  const out = new Uint8Array(T * N_KEYS);
  for (let t = 0; t < T; t++) {
    for (let l = 0; l < N_KEYS; l++) {
      const v = p[t * N_KEYS + l];
      const left = t > 0 ? p[(t - 1) * N_KEYS + l] : -1;
      const right = t + 1 < T ? p[(t + 1) * N_KEYS + l] : -1;
      out[t * N_KEYS + l] = v >= thf && v > left && v >= right ? 1 : 0;
    }
  }
  return out;
}

const DEN_MIN = fr(1e-6);

/** Sub-frame position of a peak from its two neighbours, clipped to +-0.5 frame (float32 arithmetic, as decode._refine). */
function refine(p, T, t, l) {
  const C = p[t * N_KEYS + l];
  const A = t > 0 ? p[(t - 1) * N_KEYS + l] : C;
  const B = t + 1 < T ? p[(t + 1) * N_KEYS + l] : C;
  const den = fr(2 * fr(C - Math.min(A, B)));
  if (den <= DEN_MIN) return 0;
  return Math.min(0.5, Math.max(-0.5, fr(fr(B - A) / den)));
}

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

/** Column l of a mask -> frame indices where it is set. */
function nonzeroCol(mask, T, l) {
  const out = [];
  for (let t = 0; t < T; t++) if (mask[t * N_KEYS + l]) out.push(t);
  return out;
}

/** Onset non-maximum suppression: peaks within nms frames of a higher one are dropped (stable order on ties). */
function suppress(ons, on, l, nms) {
  if (nms <= 0 || ons.length < 2) return ons;
  const keep = new Uint8Array(ons.length).fill(1);
  const order = ons.map((_, i) => i).sort((a, b) => on[ons[b] * N_KEYS + l] - on[ons[a] * N_KEYS + l]);
  for (const oi of order) {
    if (!keep[oi]) continue;
    for (let j = 0; j < ons.length; j++) if (j !== oi && Math.abs(ons[j] - ons[oi]) <= nms) keep[j] = 0;
  }
  return ons.filter((_, i) => keep[i]);
}

/** Frames where the frame probability stays below fr_th for fr_low_frames frames. */
function lowRuns(frm, T, frTh, frLow) {
  const thf = fr(frTh);
  const run = new Uint8Array(T * N_KEYS);
  for (let i = 0; i < run.length; i++) run[i] = frm[i] < thf ? 1 : 0;
  const low = run.slice();
  for (let k = 1; k < frLow; k++) {
    for (let t = 0; t < T - k; t++) {
      for (let l = 0; l < N_KEYS; l++) run[t * N_KEYS + l] &= low[(t + k) * N_KEYS + l];
    }
  }
  return run;
}

/** The end (in frames, fractional) of the note starting at frame t0 on key l (decode's three end rules). */
function noteEnd(ctx, l, t0, nxt, offs, lows) {
  const { on, off, T, offMin } = ctx;
  let j = searchLeft(lows, t0 + 1);
  const eFr = j < lows.length ? lows[j] : T;
  j = searchLeft(offs, t0 + offMin);
  const eOff = j < offs.length ? offs[j] : T;
  if (eOff < T && eOff <= Math.min(eFr, nxt)) return eOff + refine(off, T, eOff, l);
  if (eFr <= nxt) return eFr - 0.5;
  return nxt + (nxt < T ? refine(on, T, nxt, l) : 0);
}

/** Lexicographic comparison of [on, off, pitch] rows (Python tuple order). */
function cmpRows(a, b) {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

/**
 * Maps of one voice -> notes [[onsetS, offsetS, midiPitch], ...] sorted (decode.decode).
 * @param {Float32Array} on onset probabilities [T][88]
 * @param {Float32Array} frm frame probabilities [T][88]
 * @param {Float32Array} off offset probabilities [T][88]
 * @param {number} T frames
 * @param {number} hop seconds per frame
 * @param {{on_th?: number, off_th?: number, fr_th?: number, fr_low_frames?: number, on_nms?: number, min_dur?: number,
 *   off_min_frames?: number}} [opts]
 * @returns {number[][]}
 */
export function decode(on, frm, off, T, hop, opts = {}) {
  const o = {
    on_th: 0.25,
    off_th: 0.3,
    fr_th: 0.4,
    fr_low_frames: 2,
    on_nms: 3,
    min_dur: 0.02,
    off_min_frames: 2,
    ...opts,
  };
  const pkOn = peaks(on, T, o.on_th);
  const pkOff = peaks(off, T, o.off_th);
  const run = lowRuns(frm, T, o.fr_th, o.fr_low_frames);
  const ctx = { on, off, T, offMin: o.off_min_frames };
  const notes = [];
  for (let l = 0; l < N_KEYS; l++) {
    let ons = nonzeroCol(pkOn, T, l);
    if (!ons.length) continue;
    ons = suppress(ons, on, l, o.on_nms);
    const offs = nonzeroCol(pkOff, T, l);
    const lows = nonzeroCol(run, T, l);
    for (let i = 0; i < ons.length; i++) {
      const t0 = ons[i];
      const u0 = t0 + refine(on, T, t0, l);
      const u1 = noteEnd(ctx, l, t0, i + 1 < ons.length ? ons[i + 1] : T, offs, lows);
      notes.push([u0 * hop, Math.max(u1 * hop, u0 * hop + o.min_dur), PITCH_LO + l]);
    }
  }
  return notes.sort(cmpRows);
}

/**
 * Transcriber.notes: decode both voices -> note table sorted by (onset, offset, pitch).
 * conf is the voice's onset probability at the onset frame; slot is 1 or 2.
 * @param {{onset: Float32Array, frame: Float32Array, offset: Float32Array, onset2: Float32Array, frame2: Float32Array,
 *   offset2: Float32Array}} maps each [T][88]
 * @param {{on_th: number, off_th: number, fr_th: number, fr_low_frames: number, on_nms?: number, v2_th?: number|null,
 *   hop?: number}} th thresholds (web/models/meta.json transcriber.thresholds)
 * @returns {{onset: Float64Array, offset: Float64Array, pitch: Int32Array, conf: Float32Array, slot: Int8Array}}
 */
export function notes(maps, th) {
  const hop = th.hop ?? HOP;
  const T = maps.onset.length / N_KEYS;
  const args = { off_th: th.off_th, fr_th: th.fr_th, fr_low_frames: th.fr_low_frames, on_nms: th.on_nms ?? 0 };
  const voices = [
    [1, maps.onset, maps.frame, maps.offset, th.on_th],
    [2, maps.onset2, maps.frame2, maps.offset2, th.v2_th],
  ];
  const rows = [];
  for (const [slot, on, frm, off, onTh] of voices) {
    if (onTh == null) continue;
    for (const [u0, u1, p] of decode(on, frm, off, T, hop, { ...args, on_th: onTh })) {
      const t = Math.min(Math.max(roundHalfEven(u0 / hop), 0), T - 1);
      rows.push([u0, u1, p, on[t * N_KEYS + p - PITCH_LO], slot]);
    }
  }
  rows.sort(cmpRows);
  return {
    onset: Float64Array.from(rows, (r) => r[0]),
    offset: Float64Array.from(rows, (r) => r[1]),
    pitch: Int32Array.from(rows, (r) => r[2]),
    conf: Float32Array.from(rows, (r) => r[3]),
    slot: Int8Array.from(rows, (r) => r[4]),
  };
}
