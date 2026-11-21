// Hann-weighted overlap-add of the transcriber's window outputs, with bounded memory.
//
// The reference (dettato/transcriber/core.py, _Average) keeps three whole-song float16 buffers of T x 88 x 368 (about 10 MB per
// second of audio): "hidden" averaged over every window, and "subHidden" / "subStem" averaged over every other window. That does not
// fit a browser tab for a full song, so this store keeps a sliding region of frames in a ring buffer instead. The reference's
// arithmetic is kept: each window's contribution is rounded to float16 and added to the float16 sum, and a frame is divided by its
// float16 weight sum once no later window can touch it. Frames the pipeline has finished with are released.
//
// The six probability maps (T x 88 float32 each, ~250 KB per second of audio for both averages) are kept for the whole song.

import { F16_TABLE, toF16, roundF16 } from "./f16.js";

export const N_KEYS = 88;
export const C_TAP = 368;
const ROW = N_KEYS * C_TAP;
export const HEADS = ["onset", "frame", "offset", "onset2", "frame2", "offset2"];

/**
 * One Hann-weighted average of the six maps over the whole song. As in the reference (float32 torch tensors), the weighted sums
 * are accumulated in float32 and divided in place once a frame is final: 2.1 KB per frame per average (~250 KB per second of
 * audio for both averages).
 */
class MapAverage {
  constructor(T) {
    this.T = T;
    this.out = HEADS.map(() => new Float32Array(T * N_KEYS));
    this.wsum = new Float32Array(T);
    this.final = 0; // frames [0, final) are divided and valid in `out`
  }

  /** probs: Float32Array [6][Fw][88] of one window; frames a..b of the song are window frames a-g0..b-g0. */
  add(probs, Fw, g0, a, b, wt) {
    for (let h = 0; h < HEADS.length; h++) {
      const s = this.out[h];
      const base = h * Fw * N_KEYS;
      for (let t = a; t < b; t++) {
        const w = wt[t - g0];
        const src = base + (t - g0) * N_KEYS;
        const dst = t * N_KEYS;
        for (let l = 0; l < N_KEYS; l++) s[dst + l] += Math.fround(probs[src + l] * w);
      }
    }
    for (let t = a; t < b; t++) this.wsum[t] += wt[t - g0];
  }

  finalize(upto) {
    upto = Math.min(upto, this.T);
    for (let h = 0; h < HEADS.length; h++) {
      const o = this.out[h];
      for (let t = this.final; t < upto; t++) {
        const ws = Math.max(this.wsum[t], 1e-6);
        for (let l = 0; l < N_KEYS; l++) o[t * N_KEYS + l] /= ws;
      }
    }
    this.final = Math.max(this.final, upto);
  }

  /** {onset, frame, ...} views of the maps (frames beyond `final` still hold partial sums). */
  maps() {
    const m = {};
    HEADS.forEach((h, i) => (m[h] = this.out[i]));
    return m;
  }
}

/** A float16 ring buffer of [frames][88][368] holding a sliding range of song frames. */
class TapRing {
  constructor(cap) {
    this.cap = cap;
    this.bits = new Uint16Array(cap * ROW);
    this.lo = 0; // oldest frame held
    this.hi = 0; // one past the newest frame written
  }

  _slot(t) {
    return (t % this.cap) * ROW;
  }

  /** Make room for frames up to `b` (exclusive). Frames that fall out of the ring must already be released. */
  ensure(b) {
    if (b - this.lo > this.cap) {
      throw new Error(`tap ring overflow: need frames [${this.lo}, ${b}) but capacity is ${this.cap}`);
    }
    for (let t = this.hi; t < b; t++) this.bits.fill(0, this._slot(t), this._slot(t) + ROW);
    this.hi = Math.max(this.hi, b);
  }

  /** x: Float32Array [Fw][88][368] (one window's tap), weights wt per window frame. f16 += f16(x * w), as the reference. */
  add(x, g0, a, b, wt) {
    this.ensure(b);
    const bits = this.bits;
    for (let t = a; t < b; t++) {
      const w = wt[t - g0];
      const src = (t - g0) * ROW;
      const dst = this._slot(t);
      for (let i = 0; i < ROW; i++) {
        const v = Math.fround(x[src + i] * w);
        bits[dst + i] = toF16(F16_TABLE[bits[dst + i]] + F16_TABLE[toF16(v)]);
      }
    }
  }

  /** Divide frames [a, b) by their float16 weight sums (the reference's `buf /= ws.half()`). */
  divide(a, b, wsum) {
    const bits = this.bits;
    for (let t = a; t < b; t++) {
      const ws = roundF16(Math.max(Math.fround(wsum[t]), 1e-6));
      const dst = this._slot(t);
      for (let i = 0; i < ROW; i++) bits[dst + i] = toF16(F16_TABLE[bits[dst + i]] / ws);
    }
  }

  release(upto) {
    this.lo = Math.max(this.lo, Math.min(upto, this.hi));
  }

  vec(t, lane, out) {
    if (t < this.lo || t >= this.hi) throw new Error(`tap frame ${t} not held (holding [${this.lo}, ${this.hi}))`);
    const o = out || new Float32Array(C_TAP);
    const base = this._slot(t) + lane * C_TAP;
    for (let c = 0; c < C_TAP; c++) o[c] = F16_TABLE[this.bits[base + c]];
    return o;
  }
}

/**
 * The transcriber pass's two averages: "all" (every window) and "sub" (windows whose padded start is a multiple of `subStep`).
 * In fast mode every window is a sub window and "hidden" is the sub average.
 */
export class TapStore {
  /**
   * @param {number} T song frames
   * @param {number} cap ring capacity in frames
   * @param {boolean} fast one average only (the 4 s hop run)
   */
  constructor(T, cap, fast = false) {
    this.T = T;
    this.fast = fast;
    this.all = fast ? null : new MapAverage(T);
    this.sub = new MapAverage(T);
    this.rings = { subHidden: new TapRing(cap), subStem: new TapRing(cap) };
    this.rings.hidden = fast ? this.rings.subHidden : new TapRing(cap);
    this.final = 0;
  }

  /**
   * Add one window's outputs.
   * @param {{probs: Float32Array, hidden: Float32Array, stem: Float32Array}} out window outputs ([6][Fw][88], [Fw][88][368] x2)
   * @param {number} Fw frames per window
   * @param {number} g0 song frame of window frame 0
   * @param {Float32Array} wt Hann taper over the window's frames
   * @param {boolean} isSub whether this window belongs to the sub average
   */
  add(out, Fw, g0, wt, isSub) {
    const a = Math.max(0, g0);
    const b = Math.min(this.T, g0 + Fw);
    if (b <= a) return;
    if (!this.fast) {
      this.all.add(out.probs, Fw, g0, a, b, wt);
      this.rings.hidden.add(out.hidden, g0, a, b, wt);
    }
    if (isSub || this.fast) {
      this.sub.add(out.probs, Fw, g0, a, b, wt);
      this.rings.subHidden.add(out.hidden, g0, a, b, wt);
      this.rings.subStem.add(out.stem, g0, a, b, wt);
    }
  }

  /** Frames [final, upto) will receive no more windows: divide them by their weight sums. */
  finalize(upto) {
    upto = Math.min(upto, this.T);
    if (upto <= this.final) return;
    if (!this.fast) {
      this.all.finalize(upto);
      this._divide(this.rings.hidden, upto, this.all.wsum);
    }
    this.sub.finalize(upto);
    this._divide(this.rings.subHidden, upto, this.sub.wsum);
    this._divide(this.rings.subStem, upto, this.sub.wsum);
    this.final = upto;
  }

  _divide(ring, upto, wsum) {
    ring.ensure(upto);
    ring.divide(Math.max(this.final, ring.lo), upto, wsum);
  }

  /** Song maps averaged over every window (the repairer's evidence) and over the sub windows (the grouper's). */
  maps() {
    return this.fast ? this.sub.maps() : this.all.maps();
  }

  subMaps() {
    return this.sub.maps();
  }

  /** Release frames before `upto` (they will not be read again). */
  release(upto) {
    for (const r of new Set(Object.values(this.rings))) r.release(Math.min(upto, this.final));
  }

  /** The finished 368-d tap vector at song frame t and key lane (float16 values). */
  vec(tap, t, lane, out) {
    if (t >= this.final) throw new Error(`tap frame ${t} not final yet (final < ${this.final})`);
    return this.rings[tap].vec(t, lane, out);
  }
}

/** Bytes a store with this capacity allocates (to show / budget memory). */
export function tapStoreBytes(cap, fast) {
  return (fast ? 2 : 3) * cap * ROW * 2;
}
