// Complex FFT of any length (forward, e^{-2πi jk/n}, unnormalised, like numpy.fft / torch.fft), in place on Float64Arrays.
//   n = 2^k          iterative radix-2
//   n = q * 2^k      q interleaved radix-2 transforms and one O(q n) combine (q odd, small: 6144 = 3 * 2048)
//   anything else    Bluestein (chirp-z) through a power-of-two convolution (1411 = 17 * 83)
// Plans (twiddles, chirps, scratch) are built once per instance; `getFFT(n)` caches one instance per size.

const MAX_SMALL_FACTOR = 15;

/** @param {number} n @returns {boolean} */
function isPow2(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

/** Iterative radix-2 FFT of a power-of-two length. */
class Radix2 {
  /** @param {number} n */
  constructor(n) {
    this.n = n;
    const half = n >> 1;
    this.cos = new Float64Array(Math.max(half, 1));
    this.sin = new Float64Array(Math.max(half, 1));
    for (let k = 0; k < half; k++) {
      const a = (-2 * Math.PI * k) / n;
      this.cos[k] = Math.cos(a);
      this.sin[k] = Math.sin(a);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.round(Math.log2(n));
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** @param {Float64Array} re @param {Float64Array} im */
  forward(re, im) {
    const n = this.n;
    const rev = this.rev;
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
    const cos = this.cos;
    const sin = this.sin;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = start + j;
          const b = a + half;
          const wr = cos[k];
          const wi = sin[k];
          const br = re[b] * wr - im[b] * wi;
          const bi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - br;
          im[b] = im[a] - bi;
          re[a] += br;
          im[a] += bi;
        }
      }
    }
  }
}

/** n = q * m with m a power of two and q a small odd factor: q radix-2 FFTs on the decimated sequences, then combine. */
class SmallFactor {
  /** @param {number} n @param {number} q */
  constructor(n, q) {
    this.n = n;
    this.q = q;
    this.m = n / q;
    this.sub = new Radix2(this.m);
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const a = (-2 * Math.PI * k) / n;
      this.cos[k] = Math.cos(a);
      this.sin[k] = Math.sin(a);
    }
    this.bre = new Float64Array(n);
    this.bim = new Float64Array(n);
  }

  /** @param {Float64Array} re @param {Float64Array} im */
  forward(re, im) {
    const { n, q, m, bre, bim, cos, sin } = this;
    for (let r = 0; r < q; r++) {
      const off = r * m;
      for (let j = 0; j < m; j++) {
        bre[off + j] = re[j * q + r];
        bim[off + j] = im[j * q + r];
      }
      this.sub.forward(bre.subarray(off, off + m), bim.subarray(off, off + m));
    }
    for (let k = 0; k < n; k++) {
      const km = k % m;
      let sr = bre[km];
      let si = bim[km];
      for (let r = 1, tw = k; r < q; r++, tw += k) {
        const t = tw % n;
        const xr = bre[r * m + km];
        const xi = bim[r * m + km];
        sr += xr * cos[t] - xi * sin[t];
        si += xr * sin[t] + xi * cos[t];
      }
      re[k] = sr;
      im[k] = si;
    }
  }
}

/** Arbitrary n through Bluestein's chirp-z: X[k] = w[k] * sum_j (x[j] w[j]) conj(w[k-j]), w[k] = e^{-iπ k²/n}. */
class Bluestein {
  /** @param {number} n */
  constructor(n) {
    this.n = n;
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;
    this.m = m;
    this.sub = new Radix2(m);
    this.wr = new Float64Array(n);
    this.wi = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const k2 = (k * k) % (2 * n); // exact: k² < 2^53 for any practical n
      const a = (-Math.PI * k2) / n;
      this.wr[k] = Math.cos(a);
      this.wi[k] = Math.sin(a);
    }
    this.Br = new Float64Array(m);
    this.Bi = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      this.Br[k] = this.wr[k];
      this.Bi[k] = -this.wi[k];
      if (k > 0) {
        this.Br[m - k] = this.wr[k];
        this.Bi[m - k] = -this.wi[k];
      }
    }
    this.sub.forward(this.Br, this.Bi);
    this.ar = new Float64Array(m);
    this.ai = new Float64Array(m);
  }

  /** @param {Float64Array} re @param {Float64Array} im */
  forward(re, im) {
    const { n, m, wr, wi, ar, ai, Br, Bi } = this;
    for (let k = 0; k < n; k++) {
      ar[k] = re[k] * wr[k] - im[k] * wi[k];
      ai[k] = re[k] * wi[k] + im[k] * wr[k];
    }
    ar.fill(0, n);
    ai.fill(0, n);
    this.sub.forward(ar, ai);
    // pointwise product, conjugated so the inverse transform is a forward one
    for (let k = 0; k < m; k++) {
      const xr = ar[k] * Br[k] - ai[k] * Bi[k];
      const xi = ar[k] * Bi[k] + ai[k] * Br[k];
      ar[k] = xr;
      ai[k] = -xi;
    }
    this.sub.forward(ar, ai);
    const s = 1 / m;
    for (let k = 0; k < n; k++) {
      const cr = ar[k] * s;
      const ci = -ai[k] * s;
      re[k] = cr * wr[k] - ci * wi[k];
      im[k] = cr * wi[k] + ci * wr[k];
    }
  }
}

/** Odd part q of n when n = q * 2^k with q <= MAX_SMALL_FACTOR and 2^k >= 2, else 0. */
function smallFactor(n) {
  let q = n;
  while (q % 2 === 0) q /= 2;
  return q > 1 && q <= MAX_SMALL_FACTOR && n / q >= 2 ? q : 0;
}

/** In-place complex FFT of any length n >= 1 (forward transform, no scaling). */
export class FFT {
  /** @param {number} n transform length (any positive integer) */
  constructor(n) {
    if (!Number.isInteger(n) || n < 1) throw new Error(`FFT: bad length ${n}`);
    this.n = n;
    const q = smallFactor(n);
    if (isPow2(n)) this.impl = new Radix2(n);
    else if (q) this.impl = new SmallFactor(n, q);
    else this.impl = new Bluestein(n);
  }

  /**
   * Forward DFT in place: X[k] = sum_j x[j] e^{-2πi jk/n}.
   * @param {Float64Array} re real parts (length n), overwritten
   * @param {Float64Array} im imaginary parts (length n), overwritten
   */
  forward(re, im) {
    if (this.n === 1) return;
    this.impl.forward(re, im);
  }
}

/**
 * Real FFT of even length n through one complex FFT of length n / 2 (spectrum bins 0 .. n / 2). Works on any typed arrays:
 * given Float32Arrays every intermediate is stored in float32, which reproduces the rounding noise of a float32 FFT.
 */
export class RealFFT {
  /** @param {number} n even transform length with n / 2 a valid FFT length */
  constructor(n) {
    if (n % 2) throw new Error(`RealFFT: odd length ${n}`);
    this.n = n;
    this.h = n / 2;
    this.sub = new FFT(this.h);
    this.wr = new Float64Array(this.h + 1);
    this.wi = new Float64Array(this.h + 1);
    for (let k = 0; k <= this.h; k++) {
      this.wr[k] = Math.cos((-2 * Math.PI * k) / n);
      this.wi[k] = Math.sin((-2 * Math.PI * k) / n);
    }
  }

  /**
   * Forward transform of real x (length n) into (re, im) bins 0 .. n / 2 (length n / 2 + 1 each).
   * zr, zi: scratch of length n / 2 (same element type as re / im).
   */
  forward(x, re, im, zr, zi) {
    const { h, wr, wi } = this;
    for (let j = 0; j < h; j++) {
      zr[j] = x[2 * j];
      zi[j] = x[2 * j + 1];
    }
    this.sub.forward(zr, zi);
    for (let k = 0; k <= h; k++) {
      const a = k % h;
      const b = (h - k) % h;
      const er = 0.5 * (zr[a] + zr[b]);
      const ei = 0.5 * (zi[a] - zi[b]);
      const or = 0.5 * (zi[a] + zi[b]);
      const oi = -0.5 * (zr[a] - zr[b]);
      re[k] = er + wr[k] * or - wi[k] * oi;
      im[k] = ei + wr[k] * oi + wi[k] * or;
    }
  }

  /**
   * Inverse of `forward` (numpy irfft: scaled by 1 / n) from bins (re, im) 0 .. n / 2 into real x (length n).
   * zr, zi: scratch of length n / 2.
   */
  inverse(re, im, x, zr, zi) {
    const { h, n, wr, wi } = this;
    for (let k = 0; k < h; k++) {
      const b = h - k;
      const er = 0.5 * (re[k] + re[b]);
      const ei = 0.5 * (im[k] - im[b]);
      const dr = 0.5 * (re[k] - re[b]);
      const di = 0.5 * (im[k] + im[b]);
      // O = D * conj(W^k); Z = E + i O, conjugated so the inverse runs as a forward transform
      const or = dr * wr[k] + di * wi[k];
      const oi = di * wr[k] - dr * wi[k];
      zr[k] = er - oi;
      zi[k] = -(ei + or);
    }
    this.sub.forward(zr, zi);
    const s = 2 / n;
    for (let j = 0; j < h; j++) {
      x[2 * j] = zr[j] * s;
      x[2 * j + 1] = -zi[j] * s;
    }
  }
}

const CACHE = new Map();

/**
 * A cached FFT instance for length n (plans are reused across calls; not safe for concurrent use, which JS never does).
 * @param {number} n
 * @returns {FFT}
 */
export function getFFT(n) {
  let f = CACHE.get(n);
  if (!f) {
    f = new FFT(n);
    CACHE.set(n, f);
  }
  return f;
}
