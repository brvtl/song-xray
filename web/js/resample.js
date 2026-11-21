// Band-limited resampling that reproduces soxr.resample(x, srIn, srOut, quality="HQ") (libsoxr 0.1.3 via python-soxr 1.1.0).
//
// soxr HQ is a linear-phase, zero-delay resampler: output sample m is a weighted sum of input samples around time m / srOut,
// and its effective impulse response is (to ~1e-4 of its peak) a Kaiser-windowed sinc on the time axis of the lower of the
// two rates. The cutoff, window length and window shape below were fitted to soxr's measured impulse responses; soxr builds
// different filter chains for different ratios, hence the classes:
//   simple  small-integer ratios (2, 3, 3/2, 1/2, 6, ...): one DFT-convolution stage, an exact Kaiser design
//   arb     other rational ratios and all upsampling: a polyphase "arb" stage
//   down    non-rational downsampling (retune by r > 1): the arb stage behind a 2x pre-upsampler
// Output length follows soxr exactly: floor(n / (srIn / srOut) + 0.5). Signal outside [0, n) is zero.

/** Kaiser windowed-sinc parameters: fc (cutoff / Nyquist of the lower rate), th (window half-length in lower-rate samples), beta. */
const KERNELS = {
  simple: { fc: 0.956872, th: 97.25, beta: 13.04 },
  arb: { fc: 0.956838, th: 110.45, beta: 16.06 },
  down: { fc: 0.95686, th: 104.4, beta: 14.4 },
};

const MAX_EXACT_COEFS = 1 << 22; // rational ratios with more polyphase coefficients than this use interpolated phases
const INTERP_PHASES = 2048;

/** Modified Bessel function of the first kind, order 0 (power series; converges for the beta range used here). */
function besselI0(x) {
  const q = (x * x) / 4;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

/** @param {number} a @param {number} b @returns {number} */
function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** Strip the factors of two from an integer. */
function oddPart(v) {
  while (v % 2 === 0) v /= 2;
  return v;
}

/**
 * Which soxr filter chain a rate pair uses, and its exact polyphase grid when the ratio is a fraction of integers.
 * @param {number} srIn
 * @param {number} srOut
 * @returns {{ kind: "simple" | "arb" | "down", P: number, Q: number }} P / Q = srIn / srOut reduced (0 / 0 if not rational)
 */
function classify(srIn, srOut) {
  if (Number.isInteger(srIn) && Number.isInteger(srOut)) {
    const g = gcd(srIn, srOut);
    const P = srIn / g;
    const Q = srOut / g;
    const simple = oddPart(P) <= 3 && oddPart(Q) <= 3;
    return { kind: simple ? "simple" : "arb", P, Q };
  }
  return { kind: srOut > srIn ? "arb" : "down", P: 0, Q: 0 };
}

/** Kaiser windowed sinc at u (lower-rate samples), unit area. */
function kernelAt(u, k, i0beta) {
  const a = Math.abs(u);
  if (a >= k.th) return 0;
  const x = k.fc * u;
  const s = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
  const y = u / k.th;
  return k.fc * s * (besselI0(k.beta * Math.sqrt(1 - y * y)) / i0beta);
}

/**
 * Polyphase table: row p holds the taps for an output whose position is n0 + p / phases (n0 integer, input-sample units),
 * applied to inputs n0 - half + 1 .. n0 + half.
 */
function buildTable(srIn, srOut, k, phases, rows) {
  const lower = Math.min(srIn, srOut);
  const step = lower / srIn; // lower-rate samples per input sample
  const scale = srOut < srIn ? srOut / srIn : 1;
  const half = Math.ceil(k.th / step);
  const taps = 2 * half;
  const table = new Float64Array(rows * taps);
  const i0beta = besselI0(k.beta);
  for (let p = 0; p < rows; p++) {
    const phi = p / phases;
    for (let j = 0; j < taps; j++) {
      const d = phi + half - 1 - j;
      table[p * taps + j] = scale * kernelAt(d * step, k, i0beta);
    }
  }
  return { table, half, taps };
}

const PLANS = new Map();

/** Cached resampling plan for a rate pair. */
function plan(srIn, srOut) {
  const key = `${srIn}:${srOut}`;
  let p = PLANS.get(key);
  if (p) return p;
  const c = classify(srIn, srOut);
  const k = KERNELS[c.kind];
  const lower = Math.min(srIn, srOut);
  const taps = 2 * Math.ceil(k.th / (lower / srIn));
  const exact = c.Q > 0 && c.Q * taps <= MAX_EXACT_COEFS;
  const phases = exact ? c.Q : INTERP_PHASES;
  const t = buildTable(srIn, srOut, k, phases, exact ? phases : phases + 1);
  p = { ...t, exact, phases, P: c.P, Q: c.Q, kind: c.kind };
  PLANS.set(key, p);
  return p;
}

/** soxr's output length for n input samples. */
export function resampledLength(n, srIn, srOut) {
  return Math.floor(n / (srIn / srOut) + 0.5);
}

/** Dot product of `taps` coefficients from row offset `r` with xp starting at `s`. */
function dot(table, r, xp, s, taps) {
  let acc = 0;
  for (let j = 0; j < taps; j++) acc += table[r + j] * xp[s + j];
  return acc;
}

/**
 * Resample mono audio from srIn to srOut Hz, matching soxr HQ (see the file comment).
 * @param {Float32Array} x input samples
 * @param {number} srIn input rate (Hz, any positive number)
 * @param {number} srOut output rate (Hz, any positive number)
 * @returns {Float32Array} floor(x.length / (srIn / srOut) + 0.5) samples
 */
export function resample(x, srIn, srOut) {
  if (srIn === srOut) return Float32Array.from(x);
  const n = x.length;
  const olen = resampledLength(n, srIn, srOut);
  const out = new Float32Array(olen);
  if (!olen) return out;
  const p = plan(srIn, srOut);
  const { table, half, taps } = p;
  const xp = new Float32Array(n + 2 * half + 2);
  xp.set(x, half);
  // input index i sits at xp[i + half]; taps for n0 start at input n0 - half + 1, i.e. xp[n0 + 1]
  if (p.exact) {
    for (let m = 0; m < olen; m++) {
      const num = m * p.P;
      const n0 = Math.floor(num / p.Q);
      const ph = num - n0 * p.Q;
      out[m] = dot(table, ph * taps, xp, n0 + 1, taps);
    }
    return out;
  }
  const ratio = srIn / srOut;
  for (let m = 0; m < olen; m++) {
    const c = m * ratio;
    const n0 = Math.floor(c);
    const f = (c - n0) * p.phases;
    const ph = Math.floor(f);
    const w = f - ph;
    const s0 = dot(table, ph * taps, xp, n0 + 1, taps);
    const s1 = dot(table, (ph + 1) * taps, xp, n0 + 1, taps);
    out[m] = s0 + w * (s1 - s0);
  }
  return out;
}
