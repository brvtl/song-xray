import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { hasFixture, manifest, npy, parseNpy, nodeSessions, ROOT } from "./fixtures.mjs";
import {
  drumProbabilities,
  drumDecode,
  transcribeDrums,
  embedLong,
  peaks,
  pyRound,
  splitFeatures,
} from "../web/js/drums.js";

const meta = JSON.parse(fs.readFileSync(path.join(ROOT, "web/models/meta.json"), "utf8")).drums;
const ready = (clip) => hasFixture(clip) && fs.existsSync(path.join(ROOT, "fixtures", clip, "drum_feats.npy"));
const maxAbs = (a, b) => {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

/** drum_feats.npy is written Fortran-ordered ((T, 280) with T fastest): read it and return a C-order [T][280] array. */
function readFeats(clip) {
  const buf = fs.readFileSync(path.join(ROOT, "fixtures", clip, "drum_feats.npy"));
  const fortran = buf.toString("latin1", 0, 120).includes("'fortran_order': True");
  if (!fortran) return parseNpy(buf);
  const patched = Buffer.from(buf);
  patched.write("'fortran_order': False,", patched.indexOf("'fortran_order'"), "latin1"); // same length as "... True,"
  const raw = parseNpy(patched);
  const [T, F] = raw.shape;
  const data = new Float32Array(T * F);
  for (let f = 0; f < F; f++) for (let t = 0; t < T; t++) data[t * F + f] = raw.data[f * T + t];
  return { data, shape: [T, F] };
}

// The adapter's GRU/attention graphs differ from torch by up to ~1e-3 in logits on kit music even under native onnxruntime
// (checked with the Python onnxruntime on the same features), so the expert drum / timpani probabilities get a looser bound.
const ADAPTER_TOL = 3e-3;

let rtPromise;
const runtime = () => (rtPromise ??= nodeSessions());

for (const clip of ["band", "duet"]) {
  test(
    `drums ${clip}: probabilities, router and hits match the reference`,
    { skip: !ready(clip), timeout: 600000 },
    async () => {
      const rt = await runtime();
      const f = readFeats(clip);
      const m = manifest(clip);
      const P = await drumProbabilities(rt, f.data, f.shape[0]);
      for (const [mine, key, tol] of [
        ["kit", "drum_kit", 1e-4],
        ["kitVel", "drum_kit_vel", 1e-4],
        ["drum", "drum_drum", ADAPTER_TOL],
        ["timpani", "drum_timpani", ADAPTER_TOL],
        ["vel", "drum_vel", 1e-4],
      ]) {
        const d = maxAbs(P[mine], npy(clip, key).data);
        console.log(`  ${clip} ${key}: max abs diff ${d.toExponential(2)}`);
        assert.ok(d < tol, `${key} differs by ${d}`);
      }
      assert.ok(Math.abs(P.router - m.drum_router) < 1e-4, `router ${P.router} vs ${m.drum_router}`);
      const D = drumDecode(P, meta);
      assert.equal(D.model, m.drum_model === "drum-kit" ? "drum-kit" : "orchestral");

      const T = f.shape[0];
      const out = await transcribeDrums(rt, f.data, T, T * 441, meta);
      assert.equal(out.info.model, m.drum_model);
      const want = m.drum_hits;
      assert.equal(out.hits.length, want.length);
      let vdiff = 0;
      out.hits.forEach((h, i) => {
        assert.equal(h.onset, want[i].onset);
        assert.equal(h.offset, want[i].offset);
        assert.equal(h.pitch, want[i].pitch);
        assert.equal(h.isDrum, true);
        vdiff = Math.max(vdiff, Math.abs(h.velocity - want[i].velocity));
      });
      console.log(`  ${clip}: ${out.hits.length} hits, max velocity diff ${vdiff}`);
      assert.ok(vdiff < 1e-3);
      if (clip === "duet") assert.equal(out.hits.length, 0);
    },
  );
}

test(
  "drums: embed_long chunking matches the unchunked path on T=9100",
  { skip: !ready("band"), timeout: 600000 },
  async () => {
    const rt = await runtime();
    const f = readFeats("band");
    const T = 9100;
    const tiled = new Float32Array(T * 280);
    for (let t = 0; t < T; t++)
      tiled.set(f.data.subarray((t % f.shape[0]) * 280, ((t % f.shape[0]) + 1) * 280), t * 280);
    const { mel } = splitFeatures(tiled, T);
    const whole = await embedLong(rt, "drums/kit_embed", mel, T, 1e9);
    const chunked = await embedLong(rt, "drums/kit_embed", mel, T, 9000);
    const d = maxAbs(chunked, whole);
    console.log(`  embed_long vs unchunked max abs diff ${d.toExponential(2)}`);
    assert.ok(d < 1e-3, `diff ${d}`);
  },
);

test("drums: short input gives no hits", async () => {
  const out = await transcribeDrums(null, new Float32Array(0), 0, 4409, meta);
  assert.deepEqual(out, { hits: [], info: { router: null, model: null } });
});

test("drums: peaks follows infer.peaks (edge padding, >=, gap)", () => {
  // frame 0 counts (edge row is its own neighbour); frame 2 is a plateau peak only 2 after the hit at 0 (gap 3: rejected);
  // frame 3 is accepted; frame 5 is 2 after it; the last row counts
  const p = Float32Array.from([0.9, 0.2, 0.7, 0.7, 0.1, 0.6, 0.1, 0.8]);
  const { frames, lanes } = peaks(p, 1, 0.5, 3);
  assert.deepEqual(frames, [0, 3, 7]);
  assert.deepEqual(lanes, [0, 0, 0]);
});

test("drums: pyRound matches Python round (half-even on exact ties)", () => {
  assert.equal(pyRound(0.03125, 4), 0.0312);
  assert.equal(pyRound(0.09375, 4), 0.0938);
  assert.equal(pyRound(0.0625, 3), 0.062);
  assert.equal(pyRound(0.1, 4), 0.1);
  assert.equal(pyRound(2.675, 2), 2.67); // binary value is below the tie
  assert.equal(pyRound(0.846, 3), 0.846);
  assert.equal(pyRound(0, 4), 0);
});

test("drums: orchestral decode merges drum and timpani hits (repeat suppression, stable order, velocity clip)", () => {
  const T = 12;
  const drum = new Float32Array(T * 61);
  const timpani = new Float32Array(T * 13);
  const vel = new Float32Array(T * 61).fill(0.5);
  drum[2 * 61 + 14] = 0.9; // key 41 at 0.04 s
  timpani[3 * 13 + 0] = 0.8; // key 41 at 0.06 s: within 40 ms of the drum hit -> dropped
  timpani[8 * 13 + 0] = 0.8; // key 41 at 0.16 s: kept, velocity 0.8
  timpani[2 * 13 + 5] = 0.55; // key 46 at 0.04 s: same onset as the drum hit, listed after it
  timpani[10 * 13 + 1] = 0.51; // key 42 at 0.2 s
  const P = { router: 0.9, kit: new Float32Array(T * 61), kitVel: new Float32Array(T * 61), drum, timpani, vel };
  const D = drumDecode(P, meta);
  assert.equal(D.model, "orchestral");
  assert.deepEqual(Array.from(D.keys), [41, 46, 41, 42]);
  assert.deepEqual(Array.from(D.onsets), [0.04, 0.04, 0.16, 0.2]);
  assert.ok(Math.abs(D.vels[1] - 0.55) < 1e-6 && Math.abs(D.vels[0] - 0.5) < 1e-9);
});
