// Signal processing front ends of Dettato, ported from the reference Python (vendor/dettato/src/dettato):
//   hcqt          transcriber/frontend.py   HCQT.forward (batch 1)
//   grouperFront  grouper/front.py          Front.forward (+ pcen, standardise)
//   drumFeatures  drums/features.py         features()
//   tuning, retuneRatio, retuneResample     grouper/retune.py
//   resample      soxr.resample(..., "HQ")  as used by audio.py and the modules above (see resample.js)
// The reference computes in float32 (torch / numpy); here sums run in float64 and values are rounded to float32
// (Math.fround) where the reference stores float32, so results agree to float32 rounding.
// DOM-free: runs in browser workers and under Node.

import { getFFT, RealFFT } from "./fft.js";
import { resample } from "./resample.js";
import { roundF16 } from "./f16.js";

export { FFT } from "./fft.js";
export { resample, resampledLength } from "./resample.js";

const fr = Math.fround;

// ---------------------------------------------------------------------------------------------------------------- windows

/**
 * torch.hann_window(n) (periodic), computed the way ATen does in float32.
 * @param {number} n
 * @returns {Float32Array}
 */
export function hannPeriodic(n) {
  const w = new Float32Array(n);
  const a = fr((2 * Math.PI) / n);
  for (let i = 0; i < n; i++) w[i] = fr(fr(-0.5 * fr(Math.cos(fr(i * a)))) + 0.5);
  return w;
}

/**
 * np.hanning(n) (symmetric), float64 values rounded to float32 (the reference casts it with .astype(np.float32)).
 * @param {number} n
 * @returns {Float32Array}
 */
export function hanning(n) {
  const w = new Float32Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.5 + 0.5 * Math.cos((Math.PI * (1 - n + 2 * i)) / (n - 1));
  return w;
}

const WINDOWS = new Map();

/** Cached torch.hann_window(n). */
function hannCached(n) {
  let w = WINDOWS.get(n);
  if (!w) {
    w = hannPeriodic(n);
    WINDOWS.set(n, w);
  }
  return w;
}

// ---------------------------------------------------------------------------------------------------------------- STFT

/** Copy frame t of the zero-padded (center=True, constant) signal, multiplied by the window, into dst. */
function windowedFrame(x, t, hop, win, dst) {
  const n = win.length;
  const s = t * hop - (n >> 1);
  if (s >= 0 && s + n <= x.length) {
    for (let j = 0; j < n; j++) dst[j] = x[s + j] * win[j];
    return;
  }
  for (let j = 0; j < n; j++) {
    const i = s + j;
    dst[j] = i >= 0 && i < x.length ? x[i] * win[j] : 0;
  }
}

/**
 * |STFT|² with torch.stft semantics: center=True, pad_mode="constant" (n_fft // 2 zeros each side), win_length = n_fft.
 * Two real frames share one complex FFT.
 * @param {Float32Array} x signal
 * @param {number} nFft FFT size (any length)
 * @param {number} hop hop in samples
 * @param {Float32Array} win window of length nFft
 * @param {number} nFrames frames to compute (frame t is centred on sample t * hop)
 * @returns {Float32Array} [nFrames][floor(nFft / 2) + 1]
 */
export function stftPower(x, nFft, hop, win, nFrames) {
  const nb = (nFft >> 1) + 1;
  const out = new Float32Array(nFrames * nb);
  const fft = getFFT(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let t = 0; t < nFrames; t += 2) {
    windowedFrame(x, t, hop, win, re);
    if (t + 1 < nFrames) windowedFrame(x, t + 1, hop, win, im);
    else im.fill(0);
    fft.forward(re, im);
    splitPower(re, im, nFft, out, t * nb, t + 1 < nFrames ? (t + 1) * nb : -1);
  }
  return out;
}

/** Separate the spectra of the two real frames packed in (re, im) and write their power spectra. */
function splitPower(re, im, n, out, oa, ob) {
  const nb = (n >> 1) + 1;
  for (let k = 0; k < nb; k++) {
    const nk = k === 0 ? 0 : n - k;
    const ar = 0.5 * (re[k] + re[nk]);
    const ai = 0.5 * (im[k] - im[nk]);
    out[oa + k] = ar * ar + ai * ai;
    if (ob >= 0) {
      const br = 0.5 * (im[k] + im[nk]);
      const bi = 0.5 * (re[nk] - re[k]);
      out[ob + k] = br * br + bi * bi;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------- filterbanks

/**
 * A sparse filterbank: row r has weights w[off[r] .. off[r + 1]) applied to bins start[r] ...
 * @typedef {{ rows: number, cols: number, start: Int32Array, off: Int32Array, w: Float32Array }} Sparse
 */

/** Dense (rows x cols, row-major Float32Array) -> sparse rows (each row's nonzero span). */
function sparsify(dense, rows, cols) {
  const start = new Int32Array(rows);
  const off = new Int32Array(rows + 1);
  const parts = [];
  let total = 0;
  for (let r = 0; r < rows; r++) {
    let lo = 0;
    let hi = cols;
    while (lo < cols && dense[r * cols + lo] === 0) lo++;
    while (hi > lo && dense[r * cols + hi - 1] === 0) hi--;
    start[r] = lo;
    off[r] = total;
    parts.push(dense.subarray(r * cols + lo, r * cols + hi));
    total += hi - lo;
  }
  off[rows] = total;
  const w = new Float32Array(total);
  for (let r = 0; r < rows; r++) w.set(parts[r], off[r]);
  return { rows, cols, start, off, w };
}

/** numpy.linspace(start, stop, num) (endpoint included), float64, same arithmetic. */
function linspace(start, stop, num) {
  const y = new Float64Array(num);
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) y[i] = i * step + start;
  if (num > 1) y[num - 1] = stop;
  return y;
}

/** Triangle max(0, min((f - lo) / dl, (hi - f) / dh)) over the frequency grid into row r of fb (float64 scratch). */
function triangle(freqs, lo, hi, dl, dh, v) {
  for (let k = 0; k < freqs.length; k++) v[k] = Math.max(0, Math.min((freqs[k] - lo) / dl, (hi - freqs[k]) / dh));
  return v;
}

/** Sum of a Float64Array. */
function sum64(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s;
}

/**
 * transcriber/frontend.py logfreq_filterbank (16 kHz): triangles on a log grid, bps per semitone from fmin, each summing to 1.
 * @returns {Sparse}
 */
export function hcqtFilterbank(nBins, nFft, fmin = 27.5, bps = 3, minWidthBins = 2.0) {
  const SR = 16000;
  const cols = (nFft >> 1) + 1;
  const freqs = linspace(0, SR / 2, cols);
  const dense = new Float32Array(nBins * cols);
  const v = new Float64Array(cols);
  for (let i = 0; i < nBins; i++) {
    const ctr = fmin * 2.0 ** (i / (12.0 * bps));
    let lo = ctr * 2 ** (-1 / (12 * bps));
    let hi = ctr * 2 ** (1 / (12 * bps));
    const width = Math.max(hi - lo, (minWidthBins * SR) / nFft);
    lo = ctr - width / 2;
    hi = ctr + width / 2;
    triangle(freqs, lo, hi, Math.max(ctr - lo, 1e-9), Math.max(hi - ctr, 1e-9), v);
    const s = sum64(v);
    if (s > 0) for (let k = 0; k < cols; k++) dense[i * cols + k] = v[k] / s;
  }
  return sparsify(dense, nBins, cols);
}

/**
 * drums/features.py mel_filterbank: Slaney-normalised triangular mel filters (float32, normalised in float32).
 * @returns {Sparse}
 */
export function melFilterbank(sr, nFft, nMels, fmin, fmax) {
  const cols = (nFft >> 1) + 1;
  const freqs = linspace(0, sr / 2, cols);
  const hzToMel = (f) => 2595.0 * Math.log10(1.0 + f / 700.0);
  const melToHz = (m) => 700.0 * (10.0 ** (m / 2595.0) - 1.0);
  const mels = linspace(hzToMel(fmin), hzToMel(fmax), nMels + 2);
  const pts = Float64Array.from(mels, melToHz);
  const dense = new Float32Array(nMels * cols);
  const v = new Float64Array(cols);
  for (let i = 0; i < nMels; i++) {
    const lo = pts[i];
    const ctr = pts[i + 1];
    const hi = pts[i + 2];
    triangle(freqs, lo, hi, Math.max(ctr - lo, 1e-9), Math.max(hi - ctr, 1e-9), v);
    const norm = fr(2.0 / (pts[i + 2] - pts[i]));
    for (let k = 0; k < cols; k++) dense[i * cols + k] = fr(fr(v[k]) * norm);
  }
  return sparsify(dense, nMels, cols);
}

/**
 * drums/features.py logfreq_filterbank: triangles at bpo per octave from fmin, each summing to 1; a band narrower than about
 * two FFT bins reads the spectrum by linear interpolation at its centre.
 * @returns {Sparse}
 */
export function drumLogfreqFilterbank(sr = 16000, nFft = 6144, fmin = 36.708, bpo = 24, n = 120) {
  const cols = (nFft >> 1) + 1;
  const freqs = Float64Array.from({ length: cols }, (_, k) => (k * sr) / nFft);
  const c = Float64Array.from({ length: n + 2 }, (_, i) => fmin * 2.0 ** ((i - 1) / bpo));
  const dense = new Float32Array(n * cols);
  const v = new Float64Array(cols);
  for (let k = 0; k < n; k++) {
    const lo = c[k];
    const ce = c[k + 1];
    const hi = c[k + 2];
    triangle(freqs, lo, hi, ce - lo, hi - ce, v);
    if (sum64(v) < 1.0) {
      const j = (ce * nFft) / sr;
      const j0 = Math.floor(j);
      const a = j - j0;
      v.fill(0);
      v[j0] = 1 - a;
      v[j0 + 1] = a;
    }
    const s = sum64(v);
    for (let i = 0; i < cols; i++) dense[k * cols + i] = v[i] / s;
  }
  return sparsify(dense, n, cols);
}

/**
 * out[t * outStride + outOff + r] = f(fround(fb[r] · p[t])) for every frame t and row r.
 * @param {Sparse} fb
 * @param {Float32Array} p [T][fb.cols] power spectra
 * @param {number} T frames
 * @param {(v: number) => number} f elementwise map of the float32 product
 * @param {Float32Array} out
 * @param {number} rowStride stride between rows in out
 * @param {number} tStride stride between frames in out
 * @param {number} base offset of (t = 0, r = 0) in out
 */
function applyFilterbank(fb, p, T, f, out, rowStride, tStride, base) {
  const { rows, cols, start, off, w } = fb;
  for (let t = 0; t < T; t++) {
    const pt = t * cols;
    for (let r = 0; r < rows; r++) {
      const s = pt + start[r];
      let acc = 0;
      for (let i = off[r], j = s, e = off[r + 1]; i < e; i++, j++) acc += w[i] * p[j];
      out[base + r * rowStride + t * tStride] = f(fr(acc));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------- HCQT

const HOP = 160;
const N_BINS = 288;
const BPS = 3;
const HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7];
const SHIFTS = HARMONICS.map((h) => Math.round(12 * BPS * Math.log2(h)));
const SHIFT_LO = Math.min(0, ...SHIFTS);
const SHIFT_HI = Math.max(0, ...SHIFTS);
const LOG_EPS = 1e-5;
const LOG_EPS32 = fr(LOG_EPS);
const LOG_FLOOR = fr(Math.log(LOG_EPS));
/** Channels of the HCQT: one per harmonic plus the 512-point channel. */
export const HCQT_CHANNELS = HARMONICS.length + 1;
/** Rows of the extended log-frequency axis (grouper P). */
export const HCQT_NE = N_BINS + SHIFT_HI - SHIFT_LO;

let HCQT_FB = null;

/** The HCQT filterbanks (extended 2048-point axis and the 288-bin 512-point one), built once. */
function hcqtBanks() {
  if (!HCQT_FB) {
    HCQT_FB = {
      main: hcqtFilterbank(HCQT_NE, 2048, 27.5 * 2 ** (SHIFT_LO / (12 * BPS)), BPS),
      x512: hcqtFilterbank(N_BINS, 512, 27.5, BPS),
    };
  }
  return HCQT_FB;
}

/** log(v + 1e-5) in float32 arithmetic. */
function logPow(v) {
  return fr(Math.log(fr(v + LOG_EPS32)));
}

/** Identity on the float32 matmul result. */
function identity(v) {
  return v;
}

/** Stack the harmonic-shifted channels of lf [T][NE] plus the 512-point channel into X [9][T][288]. */
function stackChannels(audio, lf, T) {
  const X = new Float32Array(HCQT_CHANNELS * T * N_BINS);
  for (let c = 0; c < SHIFTS.length; c++) {
    const a = SHIFTS[c] - SHIFT_LO;
    const n = Math.max(0, Math.min(N_BINS, HCQT_NE - a));
    for (let t = 0; t < T; t++) {
      const dst = (c * T + t) * N_BINS;
      X.set(lf.subarray(t * HCQT_NE + a, t * HCQT_NE + a + n), dst);
      if (n < N_BINS) X.fill(LOG_FLOOR, dst + n, dst + N_BINS);
    }
  }
  const p = stftPower(audio, 512, HOP, hannCached(512), T);
  applyFilterbank(hcqtBanks().x512, p, T, logPow, X, 1, N_BINS, SHIFTS.length * T * N_BINS);
  return X;
}

/**
 * The 2048-point power through the extended log-frequency filterbank.
 * @returns {{ p: Float32Array, T: number }} p [T][2048 / 2 + 1]
 */
function mainPower(audio) {
  const T = Math.floor(audio.length / HOP);
  return { p: stftPower(audio, 2048, HOP, hannCached(2048), T), T };
}

/**
 * transcriber/frontend.py HCQT.forward for one window of 16 kHz audio.
 * @param {Float32Array} audio16k
 * @returns {{ data: Float32Array, T: number }} data [9][T][288] log power, T = floor(len / 160)
 */
export function hcqt(audio16k) {
  const { p, T } = mainPower(audio16k);
  const lf = new Float32Array(T * HCQT_NE);
  applyFilterbank(hcqtBanks().main, p, T, logPow, lf, 1, HCQT_NE, 0);
  return { data: stackChannels(audio16k, lf, T), T };
}

// ---------------------------------------------------------------------------------------------------------------- grouper front

const PCEN = { s: 0.025, alpha: 0.98, delta: 2.0, r: 0.5, eps: 1e-6, K: 200 };

/**
 * The FFT-convolution plan of pcen for T frames: length n = 2^ceil(log2(T + K)), the kernel s (1 - s)^k (k < K, float32 as
 * torch builds it) and its float32 spectrum, plus float32 scratch.
 */
function pcenPlan(T) {
  const { s, K } = PCEN;
  const n = 1 << Math.ceil(Math.log2(T + K));
  const rf = new RealFFT(n);
  const h = n / 2;
  const k = new Float32Array(n);
  const base = fr(1 - s);
  for (let j = 0; j < K; j++) k[j] = fr(fr(s) * fr(base ** j));
  const kr = new Float32Array(h + 1);
  const ki = new Float32Array(h + 1);
  const zr = new Float32Array(h);
  const zi = new Float32Array(h);
  rf.forward(k, kr, ki, zr, zi);
  return { n, rf, kr, ki, zr, zi, x: new Float32Array(n), re: new Float32Array(h + 1), im: new Float32Array(h + 1) };
}

/**
 * M = irfft(rfft(E, n) * rfft(k, n), n)[:T] for one frequency row (column f of E [T][F]), in float32 storage like the
 * reference's torch.fft call. Exact arithmetic would give the causal FIR sum_k k[j] E[t - j]; the reference's float32
 * round-off (~1e-7 of the row's level, then clamped at 0) dominates M wherever the row is quiet, and the grouper was trained
 * on that, so it is reproduced in kind rather than computed exactly.
 */
function pcenSmooth(plan, E, f, F, T, M) {
  const { n, rf, kr, ki, zr, zi, x, re, im } = plan;
  for (let t = 0; t < T; t++) x[t] = E[t * F + f];
  x.fill(0, T);
  rf.forward(x, re, im, zr, zi);
  for (let k = 0; k <= n / 2; k++) {
    const a = re[k];
    const b = im[k];
    re[k] = a * kr[k] - b * ki[k];
    im[k] = a * ki[k] + b * kr[k];
  }
  rf.inverse(re, im, x, zr, zi);
  for (let t = 0; t < T; t++) M[t * F + f] = x[t] > 0 ? x[t] : 0;
}

/**
 * grouper/front.py pcen on one channel [T][F] in place: E = exp(X), M = E smoothed by the causal kernel s (1 - s)^k
 * (k < K, through a float32 FFT convolution as the reference), out = log((E / (eps + M)^alpha + delta)^r - delta^r + 1e-4).
 */
function pcenChannel(X, base, T, F, plan) {
  const { alpha, delta, r, eps } = PCEN;
  if (r !== 0.5) throw new Error("pcen: r = 0.5 assumed");
  const dr = fr(delta ** r);
  const E = new Float32Array(T * F);
  for (let i = 0; i < T * F; i++) E[i] = Math.exp(X[base + i]);
  const M = new Float32Array(T * F);
  for (let f = 0; f < F; f++) pcenSmooth(plan, E, f, F, T, M);
  for (let i = 0; i < T * F; i++) {
    const v = Math.sqrt(E[i] * Math.exp(-alpha * Math.log(eps + M[i])) + delta) - dr; // r = 0.5
    X[base + i] = Math.log(v + 1e-4);
  }
}

/** grouper/front.py standardise on one channel in place: centre and scale over the active region (top 12 log units). */
function standardiseChannel(X, base, n) {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (X[base + i] > mx) mx = X[base + i];
  const th = fr(mx - 12.0);
  let cnt = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = X[base + i];
    if (v > th) {
      cnt++;
      sum += v;
    }
  }
  cnt = Math.max(cnt, 1);
  const mu = fr(sum / cnt);
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const v = X[base + i];
    if (v > th) ss += (v - mu) * (v - mu);
  }
  const sd = Math.max(fr(Math.sqrt(ss / cnt)), 0.5);
  for (let i = 0; i < n; i++) X[base + i] = fr((X[base + i] - mu) / sd);
}

/**
 * grouper/front.py Front.forward for one window of 16 kHz audio (25.5 s in use).
 * @param {Float32Array} audio16k
 * @returns {{ X: Float32Array, P: Float32Array, T: number, NE: number }}
 *   X [9][T][288] standardised PCEN of the HCQT channels; P [NE][T] linear power on the extended log-frequency axis
 */
export function grouperFront(audio16k) {
  const { p, T } = mainPower(audio16k);
  const NE = HCQT_NE;
  const P = new Float32Array(NE * T);
  applyFilterbank(hcqtBanks().main, p, T, identity, P, T, 1, 0);
  const lf = new Float32Array(T * NE);
  for (let r = 0; r < NE; r++) for (let t = 0; t < T; t++) lf[t * NE + r] = logPow(P[r * T + t]);
  const X = stackChannels(audio16k, lf, T);
  const n = T * N_BINS;
  const plan = pcenPlan(T);
  for (let c = 0; c < HCQT_CHANNELS; c++) {
    pcenChannel(X, c * n, T, N_BINS, plan);
    standardiseChannel(X, c * n, n);
  }
  return { X, P, T, NE };
}

// ---------------------------------------------------------------------------------------------------------------- tuning

const RETUNE = { SR: 16000, MIN_CENTS: 10.0, MIN_STRENGTH: 0.12 };

/** Harmonic salience [nCents][nfr] (float32 arithmetic as numpy) of the square-root magnitude spectra S [nfr][nb]. */
function salience(S, nfr, nb, f0, nFft, H) {
  const nc = f0.length;
  const sal = new Float32Array(nc * nfr);
  for (let h = 1; h <= H; h++) {
    const g = fr(0.85 ** (h - 1));
    for (let j = 0; j < nc; j++) {
      const fb = (h * f0[j] * nFft) / RETUNE.SR;
      if (!(fb < nb - 2)) continue;
      const i0 = Math.floor(fb);
      const w = fr(fb - i0);
      const w1 = fr(1 - w);
      for (let t = 0; t < nfr; t++) {
        const v = fr(fr(w1 * S[t * nb + i0]) + fr(w * S[t * nb + i0 + 1]));
        sal[j * nfr + t] = fr(sal[j * nfr + t] + fr(g * v));
      }
    }
  }
  return sal;
}

/** Square-root magnitude spectra of the hop-spaced, np.hanning-windowed frames of y: [nfr][nFft / 2 + 1] float32. */
function sqrtMagFrames(y, nFft, hop, nfr) {
  const nb = (nFft >> 1) + 1;
  const win = hanning(nFft);
  const S = new Float32Array(nfr * nb);
  const fft = getFFT(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let t = 0; t < nfr; t++) {
    for (let j = 0; j < nFft; j++) re[j] = fr(y[t * hop + j] * win[j]);
    im.fill(0);
    fft.forward(re, im);
    for (let k = 0; k < nb; k++) S[t * nb + k] = fr(Math.sqrt(fr(Math.hypot(re[k], im[k]))));
  }
  return S;
}

/** Parabolic peak picking + circular accumulation over the salience map; returns the weighted circular mean. */
function circularPeaks(sal, nc, nfr, cents, stepC, rel) {
  const mx = new Float32Array(nfr).fill(-Infinity);
  for (let j = 0; j < nc; j++) for (let t = 0; t < nfr; t++) mx[t] = Math.max(mx[t], sal[j * nfr + t]);
  let zr = 0;
  let zi = 0;
  let wsum = 0;
  for (let j = 0; j + 2 < nc; j++) {
    for (let t = 0; t < nfr; t++) {
      const A = sal[j * nfr + t];
      const B = sal[(j + 1) * nfr + t];
      const C = sal[(j + 2) * nfr + t];
      if (!(B > A && B >= C && B > fr(rel * mx[t]) && mx[t] > 0)) continue;
      const den = fr(fr(fr(A - fr(2 * B)) + C) - fr(1e-12));
      const d = fr(fr(0.5 * fr(A - C)) / den);
      const dc = fr(Math.min(1, Math.max(-1, d)) * stepC);
      const v = cents[j + 1] + dc + 50;
      const c = (((v % 100) + 100) % 100) - 50;
      const ang = (2 * Math.PI * c) / 100;
      zr += B * Math.cos(ang);
      zi += B * Math.sin(ang);
      wsum += B;
    }
  }
  return { zr, zi, wsum: fr(wsum) };
}

/**
 * grouper/retune.py tuning: offset of the recording from A440 in cents ([-50, 50)) and its strength ([0, 1]).
 * @param {Float32Array} y16k 16 kHz mono audio
 * @returns {{ delta: number, strength: number }}
 */
export function tuning(y16k, nFft = 8192, hop = 1024, H = 6, fmin = 65.0, fmax = 1000.0, stepC = 5.0, rel = 0.5) {
  if (y16k.length < nFft) return { delta: 0, strength: 0 };
  const nfr = 1 + Math.floor((y16k.length - nFft) / hop);
  const c0 = 1200 * Math.log2(fmin / 440);
  const nc = Math.ceil((1200 * Math.log2(fmax / 440) - c0) / stepC);
  const cents = Float64Array.from({ length: nc }, (_, i) => c0 + i * stepC);
  const f0 = Float64Array.from(cents, (c) => 440 * 2 ** (c / 1200));
  const nb = (nFft >> 1) + 1;
  const S = sqrtMagFrames(y16k, nFft, hop, nfr);
  const sal = salience(S, nfr, nb, f0, nFft, H);
  const { zr, zi, wsum } = circularPeaks(sal, nc, nfr, cents, stepC, rel);
  if (!(wsum > 0)) return { delta: 0, strength: 0 };
  const re = zr / wsum;
  const im = zi / wsum;
  return { delta: (Math.atan2(im, re) * 100) / (2 * Math.PI), strength: Math.hypot(re, im) };
}

/**
 * grouper/retune.py ratio: the tuning estimate and the correction applied (r = 1 when the recording is left as it is).
 * @param {Float32Array} y16k
 * @returns {{ delta: number, strength: number, shift: number, r: number }}
 */
export function retuneRatio(y16k) {
  const { delta, strength } = tuning(y16k);
  const shift = Math.abs(delta) >= RETUNE.MIN_CENTS && strength >= RETUNE.MIN_STRENGTH ? -delta : 0.0;
  return { delta, strength, shift, r: 2.0 ** (shift / 1200.0) };
}

/**
 * grouper/retune.py resample: play 16 kHz audio r times faster (pitch x r), 16 kHz -> 16 kHz / r samples.
 * @param {Float32Array} y
 * @param {number} r
 * @returns {Float32Array}
 */
export function retuneResample(y, r) {
  return resample(y, RETUNE.SR, RETUNE.SR / r);
}

// ---------------------------------------------------------------------------------------------------------------- drums

const DRUM = { SR: 44100, N_LO: 128, N_HI: 32, N_CQ: 120, CQ_NFFT: 6144, FLOOR: fr(1e-5) };
/** Feature bands per drum frame (lo mel | hi mel | cq). */
export const DRUM_BANDS = DRUM.N_LO + DRUM.N_HI + DRUM.N_CQ;

let DRUM_FB = null;

/** The three drum filterbanks, built once. */
function drumBanks() {
  if (!DRUM_FB) {
    DRUM_FB = {
      lo: melFilterbank(16000, 512, DRUM.N_LO, 20.0, 7800.0),
      hi: melFilterbank(DRUM.SR, 1411, DRUM.N_HI, 7800.0, 20000.0),
      cq: drumLogfreqFilterbank(),
    };
  }
  return DRUM_FB;
}

/**
 * drums/features.py level_gain: the gain bringing x to an RMS of -18 dBFS (computed in float64).
 * @param {Float32Array} x
 * @returns {number}
 */
export function levelGain(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  const rms = Math.sqrt(s / x.length) + 1e-9;
  return 10 ** (-18 / 20) / rms;
}

/**
 * The level-normalised 44.1 kHz signal the drum features read (float32 product, as numpy).
 * @param {Float32Array} x44
 * @returns {Float32Array}
 */
export function drumLevel(x44) {
  const g = fr(levelGain(x44));
  const x = new Float32Array(x44.length);
  for (let i = 0; i < x.length; i++) x[i] = x44[i] * g;
  return x;
}

/** log(fb @ |STFT|² + 1e-5) of one stream into columns [col, col + fb.rows) of out [T][280]. */
function drumStream(x, nFft, hop, fb, T, out, col) {
  const p = stftPower(x, nFft, hop, hannCached(nFft), T);
  const floorLog = (v) => fr(Math.log(fr(v + DRUM.FLOOR)));
  applyFilterbank(fb, p, T, floorLog, out, 1, DRUM_BANDS, col);
}

/**
 * drums/features.py features(): 280 log-power bands per 10 ms frame [lo mel 128 | hi mel 32 | cq 120], float16-rounded.
 * @param {Float32Array} x44 mono audio at 44.1 kHz
 * @param {{ x16?: Float32Array }} [opts] x16: the level-normalised signal already resampled to 16 kHz (skips the resampler)
 * @returns {{ data: Float32Array, T: number }} data [T][280]
 */
export function drumFeatures(x44, opts = {}) {
  const x = drumLevel(x44);
  const x16 = opts.x16 || resample(x, DRUM.SR, 16000);
  const n16 = 1 + Math.floor(x16.length / 160);
  const n44 = 1 + Math.floor(x.length / (DRUM.SR / 100));
  const T = Math.min(Math.floor(x16.length / 160), n16, n44);
  const data = new Float32Array(T * DRUM_BANDS);
  const fb = drumBanks();
  drumStream(x16, 512, 160, fb.lo, T, data, 0);
  drumStream(x, 1411, DRUM.SR / 100, fb.hi, T, data, DRUM.N_LO);
  drumStream(x16, DRUM.CQ_NFFT, 160, fb.cq, T, data, DRUM.N_LO + DRUM.N_HI);
  for (let i = 0; i < data.length; i++) data[i] = roundF16(data[i]);
  return { data, T };
}
