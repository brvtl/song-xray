import test from "node:test";
import assert from "node:assert/strict";
import { TapStore, N_KEYS, C_TAP, HEADS } from "../web/js/tapstore.js";
import { toF16, F16_TABLE, roundF16 } from "../web/js/f16.js";

const ROW = N_KEYS * C_TAP;

/** Deterministic pseudo-random numbers. */
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function window(Fw, r) {
  return {
    probs: Float32Array.from({ length: 6 * Fw * N_KEYS }, r),
    hidden: Float32Array.from({ length: Fw * ROW }, () => r() * 8 - 4),
    stem: Float32Array.from({ length: Fw * ROW }, () => r() * 8 - 4),
  };
}

/** The reference's arithmetic on whole-song buffers (transcriber/core.py _Average, CPU path). */
function naive(T, wins, Fw, wt, pick) {
  const buf = new Uint16Array(T * ROW);
  const ws = new Float64Array(T);
  const maps = new Float64Array(T * N_KEYS);
  for (const { out, g0, isSub } of wins) {
    if (!pick(isSub)) continue;
    const a = Math.max(0, g0);
    const b = Math.min(T, g0 + Fw);
    for (let t = a; t < b; t++) {
      const w = wt[t - g0];
      for (let i = 0; i < ROW; i++) {
        const v = toF16(Math.fround(out.hidden[(t - g0) * ROW + i] * w));
        buf[t * ROW + i] = toF16(F16_TABLE[buf[t * ROW + i]] + F16_TABLE[v]);
      }
      for (let l = 0; l < N_KEYS; l++) maps[t * N_KEYS + l] += out.probs[(t - g0) * N_KEYS + l] * w;
      ws[t] += w;
    }
  }
  for (let t = 0; t < T; t++) {
    const d = roundF16(Math.max(Math.fround(ws[t]), 1e-6));
    for (let i = 0; i < ROW; i++) buf[t * ROW + i] = toF16(F16_TABLE[buf[t * ROW + i]] / d);
    for (let l = 0; l < N_KEYS; l++) maps[t * N_KEYS + l] /= Math.max(Math.fround(ws[t]), 1e-6);
  }
  return { buf, maps };
}

test("TapStore equals whole-song float16 accumulation across ring wrap-around and release", () => {
  const Fw = 8;
  const T = 40;
  const step = 2;
  const r = rng(7);
  const wt = Float32Array.from({ length: Fw }, (_, i) => Math.sin((Math.PI * (i + 1)) / (Fw + 1)) ** 2);
  const wins = [];
  for (let k = 0; k * step - Fw / 2 < T; k++)
    wins.push({ out: window(Fw, r), g0: k * step - Fw / 2, isSub: k % 2 === 0 });
  const cap = 14; // much smaller than T: the ring wraps several times
  const store = new TapStore(T, cap, false);
  const refAll = naive(T, wins, Fw, wt, () => true);
  const refSub = naive(T, wins, Fw, wt, (s) => s);
  const got = new Map();
  for (let k = 0; k < wins.length; k++) {
    const { out, g0, isSub } = wins[k];
    store.add(out, Fw, g0, wt, isSub);
    const F = k + 1 < wins.length ? Math.max(0, wins[k + 1].g0) : T;
    store.finalize(F);
    // read every final frame once, then release it (as the pipeline does per block)
    for (let t = store.rings.hidden.lo; t < store.final; t++) {
      for (const lane of [0, 41, 87]) {
        got.set(`h${t},${lane}`, store.vec("hidden", t, lane));
        got.set(`s${t},${lane}`, store.vec("subHidden", t, lane));
      }
    }
    store.release(store.final);
  }
  assert.equal(store.final, T);
  for (let t = 0; t < T; t++) {
    for (const lane of [0, 41, 87]) {
      const h = got.get(`h${t},${lane}`);
      const s = got.get(`s${t},${lane}`);
      for (let c = 0; c < C_TAP; c++) {
        assert.equal(h[c], F16_TABLE[refAll.buf[t * ROW + lane * C_TAP + c]], `hidden t=${t} lane=${lane} c=${c}`);
        assert.equal(s[c], F16_TABLE[refSub.buf[t * ROW + lane * C_TAP + c]], `subHidden t=${t} lane=${lane} c=${c}`);
      }
    }
  }
  const maps = store.maps();
  for (let i = 0; i < T * N_KEYS; i++) assert.ok(Math.abs(maps[HEADS[0]][i] - refAll.maps[i]) < 1e-6);
});

test("TapStore refuses to read frames that are not final or already released", () => {
  const store = new TapStore(20, 10, true);
  const Fw = 8;
  const wt = new Float32Array(Fw).fill(1);
  const r = rng(1);
  store.add(window(Fw, r), Fw, 0, wt, true);
  assert.throws(() => store.vec("hidden", 3, 0), /not final/);
  store.finalize(4);
  store.vec("hidden", 3, 0);
  store.release(4);
  assert.throws(() => store.vec("hidden", 3, 0), /not held/);
});
