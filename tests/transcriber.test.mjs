import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { ROOT, hasFixture, manifest, npy } from "./fixtures.mjs";
import { layout, frameOffset, hannTaper, decode, notes, roundHalfEven } from "../web/js/transcriber.js";

const META = JSON.parse(fs.readFileSync(path.join(ROOT, "web", "models", "meta.json"), "utf8"));
const TH = { ...META.transcriber.thresholds, hop: META.transcriber.hop };
const HEADS = ["onset", "frame", "offset", "onset2", "frame2", "offset2"];
const CLIPS = ["band", "duet", "piano"];

/** maps.npy [6][T][88] -> {head: Float32Array [T][88]} */
function loadMaps(clip, key = "maps") {
  const { data, shape } = npy(clip, key);
  const sz = shape[1] * shape[2];
  return Object.fromEntries(HEADS.map((h, i) => [h, data.subarray(i * sz, (i + 1) * sz)]));
}

test("roundHalfEven matches Python round()", () => {
  const cases = [
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [-0.5, -0],
    [-1.5, -2],
    [-2.5, -2],
    [2.4999, 2],
    [3.7, 4],
  ];
  for (const [x, want] of cases) assert.equal(roundHalfEven(x) + 0, want + 0, `round(${x})`);
});

test("hannTaper is np.hanning(Fw + 2)[1:-1]", () => {
  const w = hannTaper(400);
  assert.equal(w.length, 400);
  assert.ok(w[0] > 0 && w[0] < 1e-3);
  assert.equal(w[0], w[399]);
  assert.ok(Math.abs(w[199] - w[200]) < 1e-7 && w[199] > 0.9999);
});

test("decode: a single isolated onset peak with an offset peak", () => {
  const T = 20;
  const on = new Float32Array(T * 88);
  const frm = new Float32Array(T * 88);
  const off = new Float32Array(T * 88);
  const l = 39;
  on[5 * 88 + l] = 0.9;
  for (let t = 5; t < 12; t++) frm[t * 88 + l] = 0.8;
  off[11 * 88 + l] = 0.7;
  const got = decode(on, frm, off, T, 0.02, TH);
  assert.equal(got.length, 1);
  assert.ok(Math.abs(got[0][0] - 0.1) < 1e-12);
  assert.ok(Math.abs(got[0][1] - 0.22) < 1e-12);
  assert.equal(got[0][2], 60);
});

for (const clip of CLIPS) {
  test(`layout matches Transcriber._layout (${clip})`, { skip: !hasFixture(clip) }, () => {
    const man = manifest(clip);
    const n = npy(clip, "y").data.length;
    const lay = layout(n, 0.25);
    assert.equal(lay.W, man.layout.W);
    assert.equal(lay.Fw, man.layout.Fw);
    assert.equal(lay.T, man.layout.T);
    assert.equal(lay.pad, man.layout.pad);
    assert.equal(lay.gPad, man.layout.g_pad);
    assert.deepEqual(lay.starts, man.layout.starts);
    assert.equal(frameOffset(lay.starts[0], lay.gPad), -lay.gPad);
  });

  test(`notes(maps) equals the reference tnotes exactly (${clip})`, { skip: !hasFixture(clip) }, () => {
    const got = notes(loadMaps(clip), TH);
    const want = {};
    for (const k of ["onset", "offset", "pitch", "conf", "slot"]) want[k] = npy(clip, `tnotes_${k}`).data;
    assert.equal(got.onset.length, want.onset.length, "note count");
    for (const k of ["onset", "offset", "pitch", "conf", "slot"]) {
      let bad = 0;
      for (let i = 0; i < got[k].length; i++) if (got[k][i] !== Number(want[k][i])) bad++;
      assert.equal(bad, 0, `${k}: ${bad} mismatches`);
    }
  });
}
