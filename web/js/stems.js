// Score-informed soft-mask "stems" (not part of Dettato: Dettato transcribes, it does not separate audio).
//
// The mix is analysed with an STFT (periodic Hann, n_fft 4096, hop 1024) and every time-frequency bin is split between the
// transcribed tracks and a residual by Wiener-like ratios of per-track energy templates:
//   - pitched tracks: harmonic combs (h = 1..20) around each sounding note's f0, Gaussian in frequency, with an attack
//     pre-roll and a release tail;
//   - the drum track: the percussive part of a median-filter HPSS (run on every frame), gated to +-60 ms around the hits;
//   - the residual: a floor proportional to |X|^2 in the harmonic part, plus all percussive energy outside the drum gates
//     (all of it when there is no drum track).
// The masks of a bin sum to exactly 1, and the analysis/synthesis windows overlap-add to a constant, so the sum of all
// stems and the residual reconstructs the input up to float rounding. Masks are computed once per frame from the
// channel-summed magnitude and applied to every channel. Frames are streamed through a small ring buffer (the HPSS time
// median needs a few frames of look-ahead); no spectrogram or mask is ever held for the whole song.
//
// DOM-free: runs in a worker (stems-worker.js) and under Node (tests/stems.test.mjs).

const TWO_PI = 2 * Math.PI;

/** Complex radix-2 FFT of a fixed size, in place on Float64Array re/im. */
class ComplexFFT {
  /** @param {number} n power of two */
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((TWO_PI * i) / n);
      this.sin[i] = -Math.sin((TWO_PI * i) / n);
    }
  }

  /** Forward transform (e^{-i...}) in place. */
  forward(re, im) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0, tw = 0; k < half; k++, tw += step) {
          const a = start + k;
          const b = a + half;
          const wr = cos[tw];
          const wi = sin[tw];
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

/** Median of the first `n` entries of `buf` (reorders them). Quickselect; n is small. */
function median(buf, n) {
  let lo = 0;
  let hi = n - 1;
  const k = n >> 1;
  while (hi > lo) {
    const pivot = buf[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (buf[i] < pivot) i++;
      while (buf[j] > pivot) j--;
      if (i <= j) {
        const t = buf[i];
        buf[i] = buf[j];
        buf[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return buf[k];
}

/**
 * Sparse harmonic comb of one note: bins and amplitude weights (max over overlapping partials).
 * @returns {{bins: Int32Array, w: Float32Array}}
 */
function noteComb(pitch, tuningCents, binHz, nBins, opts) {
  const f0 = 440 * 2 ** ((pitch - 69) / 12 + tuningCents / 1200);
  const map = new Map();
  const ratio = 2 ** (opts.combCents / 1200) - 1;
  for (let h = 1; h <= opts.harmonics; h++) {
    const c = (f0 * h) / binHz;
    if (c >= nBins - 1) break;
    const sigma = Math.max(opts.minSigmaBins, c * ratio);
    const wh = h ** -opts.harmonicDecay;
    const lo = Math.max(0, Math.ceil(c - 3 * sigma));
    const hi = Math.min(nBins - 1, Math.floor(c + 3 * sigma));
    for (let k = lo; k <= hi; k++) {
      const d = (k - c) / sigma;
      const v = wh * Math.exp(-0.5 * d * d);
      if (v > (map.get(k) ?? 0)) map.set(k, v);
    }
  }
  const bins = Int32Array.from(map.keys());
  const w = Float32Array.from(map.values());
  return { bins, w };
}

/** Per-frame envelope of a note at time `t` (seconds): pre-roll ramp, sustain, exponential release. */
function noteEnvelope(t, on, off, opts) {
  if (t < on - opts.preRollS) return 0;
  if (t < on) return 1 - (on - t) / opts.preRollS;
  if (t <= off) return 1;
  const dt = t - off;
  return dt > opts.releaseMaxS ? 0 : Math.exp(-dt / opts.releaseTauS);
}

/** Tracks notes of one pitched track as they enter/leave the analysis window while frames advance. */
class ActiveNotes {
  constructor(notes, opts) {
    this.notes = [...notes].sort((a, b) => a[0] - b[0]);
    this.opts = opts;
    this.next = 0;
    this.active = [];
  }

  /** Advance to frame time t; returns the notes whose envelope can be non-zero. */
  at(t) {
    const { notes, opts } = this;
    while (this.next < notes.length && notes[this.next][0] - opts.preRollS <= t) this.active.push(notes[this.next++]);
    this.active = this.active.filter((n) => n[1] + opts.releaseMaxS >= t);
    return this.active;
  }
}

const DEFAULTS = {
  nfft: 4096,
  hop: 1024,
  harmonics: 20,
  combCents: 35,
  minSigmaBins: 1,
  harmonicDecay: 0.8,
  preRollS: 0.03,
  releaseTauS: 0.08,
  releaseMaxS: 0.3,
  residualFloor: 0.05,
  drumGateS: 0.06,
  hpssTimeFrames: 17,
  hpssFreqBins: 17,
  progressEvery: 64,
};

// Stem output is (tracks + 1) x channels x samples x 4 bytes, held twice briefly (worker output + AudioBuffers).
const STEREO_STEMS_MAX_BYTES = 700e6;
const MONO_STEMS_MAX_BYTES = 900e6;

/**
 * How many channels to separate within the memory budget: 2 (stereo source that fits), 1 (mono / downmix), or 0 (too
 * long even in mono: skip stems).
 * @param {{channels: number, tracks: number, samples: number}} size source channel count, track count, samples per channel
 * @returns {0 | 1 | 2}
 */
export function stemChannels({ channels, tracks, samples }) {
  const bytes = (tracks + 1) * samples * 4;
  if (channels >= 2 && 2 * bytes <= STEREO_STEMS_MAX_BYTES) return 2;
  return bytes <= MONO_STEMS_MAX_BYTES ? 1 : 0;
}

/**
 * Split a mix into one stem per transcribed track plus a residual, by score-informed soft masking.
 *
 * @param {object} args
 * @param {Float32Array[]} args.channels mix, one Float32Array per channel (all the same length)
 * @param {number} args.rate sample rate of `channels`
 * @param {{name: string, program: number, isDrum: boolean, notes: number[][]}[]} args.tracks transcription; notes are
 *   [onsetS, offsetS, midiPitch, velocity1to127]
 * @param {number} [args.tuningCents] detuning of the recording vs A440, applied to every pitched note's f0
 * @param {(done: number, total: number) => void} [args.onProgress] called with frames processed / total frames
 * @param {Partial<typeof DEFAULTS>} [args.options] analysis parameters (see DEFAULTS)
 * @returns {{stems: Float32Array[][], residual: Float32Array[], info: {frames: number, ms: number}}} `stems[k][c]` is
 *   track k, channel c; `residual[c]` is what no track explains. sum_k stems[k][c] + residual[c] == channels[c].
 */
export function separateStems({ channels, rate, tracks, tuningCents = 0, onProgress, options = {} }) {
  const t0 = Date.now();
  const opts = { ...DEFAULTS, ...options };
  const N = opts.nfft;
  const H = opts.hop;
  const nBins = N / 2 + 1;
  const C = channels.length;
  const n = channels[0].length;
  const binHz = rate / N;

  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((TWO_PI * i) / N);
  // sum_m w^2[i + mH] is periodic in H; every output sample is covered by N/H full frames (see `pad`).
  const olaNorm = new Float64Array(H);
  for (let i = 0; i < N; i++) olaNorm[i % H] += win[i] * win[i];

  const pad = N - H;
  const T = Math.ceil((n + pad) / H);

  const pitched = [];
  let drum = null;
  tracks.forEach((tr, k) => {
    if (tr.isDrum) {
      if (!drum) drum = { k, hits: tr.notes.map((x) => x[0]).sort((a, b) => a - b) };
    } else pitched.push({ k, active: new ActiveNotes(tr.notes, opts), combs: new Map() });
  });
  const K = tracks.length;

  // Output buffers.
  const stems = tracks.map(() => channels.map(() => new Float32Array(n)));
  const residual = channels.map(() => new Float32Array(n));
  const outputs = [];
  for (let k = 0; k <= K; k++) for (let c = 0; c < C; c++) outputs.push(k < K ? stems[k][c] : residual[c]);

  // Frame ring: half-spectra per channel + summed magnitude, `delay` frames of look-ahead for the HPSS time median.
  const delay = opts.hpssTimeFrames >> 1;
  const ringLen = 2 * delay + 1;
  const ring = Array.from({ length: ringLen }, () => ({
    re: channels.map(() => new Float64Array(nBins)),
    im: channels.map(() => new Float64Array(nBins)),
    mag: new Float32Array(nBins),
  }));

  const fft = new ComplexFFT(N);
  const zr = new Float64Array(N);
  const zi = new Float64Array(N);
  const masks = Array.from({ length: K + 1 }, () => new Float32Array(nBins));
  const tmpl = new Float32Array(nBins);
  const sumA = new Float32Array(nBins);
  const percFrac = new Float32Array(nBins);
  const medBuf = new Float32Array(Math.max(opts.hpssTimeFrames, opts.hpssFreqBins));
  const harmMed = new Float32Array(nBins);
  let hitPtr = 0;

  const frameStart = (t) => t * H - pad;
  const frameTime = (t) => (frameStart(t) + N / 2) / rate;

  /** STFT of frame t into its ring slot (two real channels packed into one complex FFT). */
  function analyse(t) {
    const slot = ring[t % ringLen];
    if (t >= T) return;
    slot.mag.fill(0);
    const s = frameStart(t);
    for (let c = 0; c < C; c += 2) {
      const a = channels[c];
      const b = c + 1 < C ? channels[c + 1] : null;
      for (let i = 0; i < N; i++) {
        const j = s + i;
        const inside = j >= 0 && j < n;
        zr[i] = inside ? a[j] * win[i] : 0;
        zi[i] = inside && b ? b[j] * win[i] : 0;
      }
      fft.forward(zr, zi);
      for (let k = 0; k < nBins; k++) {
        const m = (N - k) % N;
        // A = (Z[k] + conj Z[N-k]) / 2, B = (Z[k] - conj Z[N-k]) / 2i
        const ar = 0.5 * (zr[k] + zr[m]);
        const ai = 0.5 * (zi[k] - zi[m]);
        slot.re[c][k] = ar;
        slot.im[c][k] = ai;
        slot.mag[k] += ar * ar + ai * ai;
        if (b) {
          const br = 0.5 * (zi[k] + zi[m]);
          const bi = -0.5 * (zr[k] - zr[m]);
          slot.re[c + 1][k] = br;
          slot.im[c + 1][k] = bi;
          slot.mag[k] += br * br + bi * bi;
        }
      }
    }
    for (let k = 0; k < nBins; k++) slot.mag[k] = Math.sqrt(slot.mag[k]);
  }

  /** Is frame time tc within the gate of a drum hit? (hits sorted; frames visited in increasing time) */
  function drumGate(tc) {
    const reach = opts.drumGateS + (0.5 * H) / rate;
    const hits = drum.hits;
    while (hitPtr < hits.length && hits[hitPtr] < tc - reach) hitPtr++;
    return hitPtr < hits.length && hits[hitPtr] <= tc + reach;
  }

  /** Soft percussive fraction P^2 / (H^2 + P^2) of each bin of frame t (median-filter HPSS). */
  function hpss(t) {
    const centre = ring[t % ringLen].mag;
    const half = opts.hpssTimeFrames >> 1;
    const cols = [];
    for (let u = Math.max(0, t - half); u <= Math.min(T - 1, t + half); u++) cols.push(ring[u % ringLen].mag);
    const m = cols.length;
    for (let k = 0; k < nBins; k++) {
      for (let j = 0; j < m; j++) medBuf[j] = cols[j][k];
      harmMed[k] = median(medBuf, m);
    }
    const fh = opts.hpssFreqBins >> 1;
    for (let k = 0; k < nBins; k++) {
      let m = 0;
      for (let q = Math.max(0, k - fh); q <= Math.min(nBins - 1, k + fh); q++) medBuf[m++] = centre[q];
      const p = median(medBuf, m);
      const h = harmMed[k];
      const p2 = p * p;
      const den = p2 + h * h;
      percFrac[k] = den > 0 ? p2 / den : 0;
    }
  }

  /** Masks of frame t: masks[k] for each track, masks[K] for the residual; they sum to 1 per bin. */
  function buildMasks(t) {
    const tc = frameTime(t);
    const floor2 = opts.residualFloor * opts.residualFloor;
    sumA.fill(floor2);
    for (const p of pitched) {
      tmpl.fill(0);
      for (const note of p.active.at(tc)) {
        const env = noteEnvelope(tc, note[0], note[1], opts);
        if (env <= 0) continue;
        let comb = p.combs.get(note[2]);
        if (!comb) p.combs.set(note[2], (comb = noteComb(note[2], tuningCents, binHz, nBins, opts)));
        const g = env * (0.4 + (0.6 * (note[3] ?? 100)) / 127);
        const { bins, w } = comb;
        for (let i = 0; i < bins.length; i++) {
          const v = g * w[i];
          if (v > tmpl[bins[i]]) tmpl[bins[i]] = v;
        }
      }
      const mk = masks[p.k];
      for (let k = 0; k < nBins; k++) {
        const a = tmpl[k] * tmpl[k];
        mk[k] = a;
        sumA[k] += a;
      }
    }
    const gated = drum ? drumGate(tc) : false;
    hpss(t);
    // Harmonic part (1 - perc) is shared among pitched tracks and the floor; the percussive part goes to the drums
    // inside a hit gate, else (or without a drum track) to the residual.
    for (const p of pitched) {
      const mk = masks[p.k];
      for (let k = 0; k < nBins; k++) mk[k] = (1 - percFrac[k]) * (mk[k] / sumA[k]);
    }
    if (drum) {
      const md = masks[drum.k];
      if (gated) md.set(percFrac);
      else md.fill(0);
    }
    const res = masks[K];
    for (let k = 0; k < nBins; k++) {
      let s = 0;
      for (let j = 0; j < K; j++) s += masks[j][k];
      res[k] = 1 - s;
    }
  }

  /** Output o (= track * C + channel) of frame `slot`: its mask times that channel's spectrum. */
  function maskedSpectrum(slot, o, dstRe, dstIm) {
    const mk = masks[Math.floor(o / C)];
    const xr = slot.re[o % C];
    const xi = slot.im[o % C];
    for (let k = 0; k < nBins; k++) {
      dstRe[k] = mk[k] * xr[k];
      dstIm[k] = mk[k] * xi[k];
    }
  }

  /** Inverse STFT of every (mask x channel) pair of frame t, two real outputs per complex IFFT, overlap-added. */
  function synthesise(t) {
    const slot = ring[t % ringLen];
    const s = frameStart(t);
    const total = outputs.length;
    for (let o = 0; o < total; o += 2) {
      maskedSpectrum(slot, o, spectrum.are, spectrum.aim);
      if (o + 1 < total) maskedSpectrum(slot, o + 1, spectrum.bre, spectrum.bim);
      else {
        spectrum.bre.fill(0);
        spectrum.bim.fill(0);
      }
      // Z = A + iB over the full (Hermitian) spectrum; conj for the inverse via the forward FFT.
      const { are, aim, bre, bim } = spectrum;
      for (let k = 0; k < nBins; k++) {
        zr[k] = are[k] - bim[k];
        zi[k] = -(aim[k] + bre[k]);
        if (k > 0 && k < N / 2) {
          zr[N - k] = are[k] + bim[k];
          zi[N - k] = -(-aim[k] + bre[k]);
        }
      }
      fft.forward(zr, zi);
      const outA = outputs[o];
      const outB = o + 1 < total ? outputs[o + 1] : null;
      const lo = Math.max(0, -s);
      const hi = Math.min(N, n - s);
      for (let i = lo; i < hi; i++) {
        const g = win[i] / (N * olaNorm[(s + i) % H]);
        outA[s + i] += g * zr[i];
        if (outB) outB[s + i] -= g * zi[i];
      }
    }
  }

  const spectrum = {
    are: new Float64Array(nBins),
    aim: new Float64Array(nBins),
    bre: new Float64Array(nBins),
    bim: new Float64Array(nBins),
  };

  for (let t = 0; t < delay; t++) analyse(t);
  for (let t = 0; t < T; t++) {
    analyse(t + delay);
    buildMasks(t);
    synthesise(t);
    if (onProgress && (t % opts.progressEvery === 0 || t === T - 1)) onProgress(t + 1, T);
  }
  return { stems, residual, info: { frames: T, ms: Date.now() - t0 } };
}
