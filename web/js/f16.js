// IEEE half precision helpers. The reference keeps the transcriber's tap buffers in float16 and accumulates them there, so the
// browser stores them as raw float16 bits (Uint16Array) and rounds the same way.

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/** number -> float16 bits, round to nearest even (as torch's .half()). */
export function toF16(v) {
  F32[0] = v;
  const x = U32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow -> signed zero
    mant |= 0x800000; // subnormal half: shift the implicit bit in
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    const rest = mant & ((1 << shift) - 1);
    let m = mant >>> shift;
    if (rest > half || (rest === half && m & 1)) m++;
    return sign | m;
  }
  const rest = mant & 0x1fff;
  let m = mant >>> 13;
  if (rest > 0x1000 || (rest === 0x1000 && m & 1)) {
    m++;
    if (m === 0x400) {
      m = 0;
      e++;
      if (e >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (e << 10) | m;
}

/** float16 bits -> number, via a 65536-entry table. */
export const F16_TABLE = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) t[h] = s * m * 2 ** -24;
    else if (e === 31) t[h] = m ? NaN : s * Infinity;
    else t[h] = s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function fromF16(bits) {
  return F16_TABLE[bits];
}

/** Round a number through float16 and back (what `.half().float()` does). */
export function roundF16(v) {
  return F16_TABLE[toF16(v)];
}
