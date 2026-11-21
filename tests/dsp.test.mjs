// web/js/dsp.js against the reference: fixtures/<clip>/*.npy from scripts/dump_fixtures.py, and fixtures/<clip>/dsp/*.npy
// from tests/py/dsp_fixtures.py (soxr resampling cases, C-order drum features, a detuned clip). Skips when they are missing.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { ROOT, npy, manifest, hasFixture, relDiff } from "./fixtures.mjs";
import {
  FFT,
  stftPower,
  hannPeriodic,
  hanning,
  hcqt,
  grouperFront,
  drumFeatures,
  resample,
  resampledLength,
  tuning,
  retuneRatio,
  retuneResample,
} from "../web/js/dsp.js";

const CLIPS = ["band", "duet", "piano"];
const DSP_MANIFEST = path.join(ROOT, "fixtures", "band", "dsp", "manifest.json");
const hasDsp = () => fs.existsSync(DSP_MANIFEST);
const dspManifest = () => JSON.parse(fs.readFileSync(DSP_MANIFEST, "utf8"));

/** max / mean |a - b| (and the same skipping `edge` samples at both ends). */
function absStats(a, b, edge = 0) {
  assert.equal(a.length, b.length);
  let max = 0;
  let sum = 0;
  let inner = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    assert.ok(!Number.isNaN(d), `NaN at ${i}`);
    if (d > max) max = d;
    if (i >= edge && i < a.length - edge && d > inner) inner = d;
    sum += d;
  }
  return { max, inner, mean: sum / a.length };
}

/** Naive DFT for the FFT check. */
function dft(re, im) {
  const n = re.length;
  const R = new Float64Array(n);
  const I = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * ((j * k) % n)) / n;
      sr += re[j] * Math.cos(a) - im[j] * Math.sin(a);
      si += re[j] * Math.sin(a) + im[j] * Math.cos(a);
    }
    R[k] = sr;
    I[k] = si;
  }
  return [R, I];
}

test("FFT matches a naive DFT for n in {8, 1411, 2048, 6144}", () => {
  for (const n of [8, 1411, 2048, 6144]) {
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 1.3) + (i % 5) - 2);
    const im = Float64Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    const [R, I] = dft(re, im);
    new FFT(n).forward(re, im);
    const err = Math.max(relDiff(re, R), relDiff(im, I));
    assert.ok(err < 1e-12, `n=${n}: rel err ${err}`);
  }
});

test("windows: periodic Hann and symmetric np.hanning", () => {
  const h = hannPeriodic(8);
  assert.ok(Math.abs(h[0]) < 1e-7 && Math.abs(h[4] - 1) < 1e-7 && Math.abs(h[2] - 0.5) < 1e-7);
  const s = hanning(5);
  assert.deepEqual(Array.from(s), [0, 0.5, 1, 0.5, 0]);
});

test("stftPower: centred, zero-padded frames (odd n_fft too)", () => {
  const n = 1411;
  const x = Float32Array.from({ length: 5000 }, (_, i) => Math.sin(i * 0.05) * (1 + (i % 7) / 10));
  const win = hannPeriodic(n);
  const T = 5;
  const p = stftPower(x, n, 441, win, T);
  const nb = (n >> 1) + 1;
  for (const t of [0, 3, 4]) {
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const i = t * 441 - (n >> 1) + j;
      re[j] = i >= 0 && i < x.length ? x[i] * win[j] : 0;
    }
    const [R, I] = dft(re, im);
    const ref = Float64Array.from({ length: nb }, (_, k) => R[k] ** 2 + I[k] ** 2);
    assert.ok(relDiff(p.subarray(t * nb, (t + 1) * nb), ref) < 1e-6, `frame ${t}`);
  }
});

for (const clip of CLIPS) {
  test(`hcqt matches win_hcqt (${clip})`, (t) => {
    if (!hasFixture(clip)) return t.skip("no fixtures");
    const ref = npy(clip, "win_hcqt");
    const { data, T } = hcqt(npy(clip, "win_audio").data);
    assert.deepEqual([9, T, 288], ref.shape);
    const s = absStats(data, ref.data);
    t.diagnostic(`log-domain |diff| max ${s.max.toExponential(2)} mean ${s.mean.toExponential(2)}`);
    assert.ok(s.max < 2e-3 && s.mean < 3e-5, JSON.stringify(s));
  });

  test(`grouperFront matches grp_w0_P / grp_w0_X (${clip})`, (t) => {
    if (!hasFixture(clip)) return t.skip("no fixtures");
    const P = npy(clip, "grp_w0_P");
    const X = npy(clip, "grp_w0_X");
    const r = grouperFront(npy(clip, "grp_w0_audio").data);
    assert.deepEqual([r.NE, r.T], P.shape);
    assert.deepEqual([9, r.T, 288], X.shape);
    const rp = relDiff(r.P, P.data);
    // X: the reference's PCEN smoother is a float32 FFT convolution whose round-off (clamped at 0) dominates quiet regions;
    // the port reproduces that noise in kind, not bit for bit. Another float32 FFT length in torch itself differs from the
    // fixture by mean |dX| 0.002 / 0.17 / 0.057 (band / duet / piano); hold the port to the same order.
    const sx = absStats(r.X, X.data);
    t.diagnostic(`P relDiff ${rp.toExponential(2)}; X |diff| mean ${sx.mean.toFixed(4)} max ${sx.max.toFixed(3)}`);
    assert.ok(rp < 1e-5, `P relDiff ${rp}`);
    const bound = { band: 0.01, duet: 0.25, piano: 0.1 }[clip];
    assert.ok(sx.mean < bound, `X mean |diff| ${sx.mean}`);
    if (clip === "band") {
      // band is mostly loud: its median is not noise-dominated, so it pins the pcen / standardise formulas (torch-vs-torch 2.3e-4)
      const d = Float64Array.from(r.X, (v, i) => Math.abs(v - X.data[i])).sort();
      assert.ok(d[d.length >> 1] < 2e-3, `X median |diff| ${d[d.length >> 1]}`);
    }
  });

  test(`retuneRatio matches manifest.tuning (${clip})`, (t) => {
    if (!hasFixture(clip)) return t.skip("no fixtures");
    const m = manifest(clip).tuning;
    const r = retuneRatio(npy(clip, "x16").data);
    t.diagnostic(`delta ${r.delta} vs ${m.delta}; strength ${r.strength} vs ${m.strength}`);
    assert.ok(Math.abs(r.delta - m.delta) < 1e-5);
    assert.ok(Math.abs(r.strength - m.strength) < 1e-5);
    assert.equal(r.shift === 0, m.shift === 0);
    assert.ok(Math.abs(r.r - m.r) < 1e-8);
  });

  test(`resample 44.1k -> 16k matches x16 (${clip})`, (t) => {
    if (!hasFixture(clip) || !manifest(clip).arrays.x44) return t.skip("no 44.1 kHz fixture");
    const y = resample(npy(clip, "x44").data, 44100, 16000);
    const s = absStats(y, npy(clip, "x16").data, 2);
    t.diagnostic(`max ${s.max.toExponential(2)} interior ${s.inner.toExponential(2)}`);
    assert.ok(s.max < 1e-3, JSON.stringify(s));
  });
}

test("resample: soxr output length rule", (t) => {
  if (!hasDsp()) return t.skip("run tests/py/dsp_fixtures.py");
  for (const [n, a, b, len] of dspManifest().lengths) assert.equal(resampledLength(n, a, b), len, `${n} ${a}->${b}`);
});

test("resample: soxr HQ cases (48k->16k, 48k->44.1k, 16k->44.1k, retune both ways)", (t) => {
  if (!hasDsp()) return t.skip("run tests/py/dsp_fixtures.py");
  for (const [clip, cases] of Object.entries(dspManifest().resample)) {
    for (const [name, [a, b]] of Object.entries(cases)) {
      const ref = npy(clip, `dsp/rs_${name}_out`).data;
      const y = resample(npy(clip, `dsp/rs_${name}_in`).data, a, b);
      const s = absStats(y, ref, 2);
      t.diagnostic(`${clip} ${name}: max ${s.max.toExponential(2)} interior ${s.inner.toExponential(2)}`);
      assert.ok(s.max < 1e-3, `${clip} ${name} ${JSON.stringify(s)}`);
    }
  }
});

test("tuning of a detuned clip (+35 cents): retune shift != 0, and retuneResample matches soxr", (t) => {
  if (!hasDsp()) return t.skip("run tests/py/dsp_fixtures.py");
  const m = dspManifest().detuned35;
  const det = npy("band", "dsp/detuned35").data;
  const r = retuneRatio(det);
  t.diagnostic(`delta ${r.delta} vs ${m.delta}; shift ${r.shift}; r ${r.r}`);
  assert.notEqual(r.shift, 0);
  assert.ok(Math.abs(r.delta - m.delta) < 1e-5 && Math.abs(r.strength - m.strength) < 1e-5);
  assert.ok(Math.abs(r.r - m.r) < 1e-8);
  assert.equal(tuning(det.subarray(0, 100)).strength, 0);
  const y = retuneResample(npy("band", "x16").data, 2 ** (35 / 1200));
  const s = absStats(y, det, 2);
  t.diagnostic(`retuneResample max ${s.max.toExponential(2)} interior ${s.inner.toExponential(2)}`);
  assert.ok(s.max < 1e-3, JSON.stringify(s));
});

test("drumFeatures matches drum_feats (band), float16 grid", (t) => {
  if (!hasDsp() || !fs.existsSync(path.join(ROOT, "fixtures", "band", "dsp", "drum_feats.npy")))
    return t.skip("no fixture");
  const ref = npy("band", "dsp/drum_feats");
  const x44 = npy("band", "x44").data;
  const check = (label, r) => {
    assert.deepEqual([r.T, 280], ref.shape);
    let eq = 0;
    let max = 0;
    for (let i = 0; i < r.data.length; i++) {
      const d = Math.abs(r.data[i] - ref.data[i]);
      if (d === 0) eq++;
      if (d > max) max = d;
    }
    t.diagnostic(`${label}: exact ${((100 * eq) / r.data.length).toFixed(3)} %, max |diff| ${max}`);
    assert.ok(max <= 2 * 2 ** -7 + 1e-9, `${label} max ${max}`); // <= 2 float16 ulps at |v| < 16
    assert.ok(eq / r.data.length > 0.98, `${label} exact fraction ${eq / r.data.length}`);
  };
  check("own resampler", drumFeatures(x44));
  check("soxr x16", drumFeatures(x44, { x16: npy("band", "dsp/drum_x16").data }));
});
