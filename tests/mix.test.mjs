import { test } from "node:test";
import assert from "node:assert/strict";
import { mixGains } from "../web/js/audio/mix.js";

const state = (over = {}) => ({
  original: {},
  residual: {},
  tracks: [{}, {}, {}],
  stemsReady: true,
  ...over,
});

test("stems ready, untouched: stems + residual play, never the original file", () => {
  const g = mixGains(state());
  assert.equal(g.original, 0);
  assert.equal(g.residual, 1);
  assert.deepEqual(g.stems, [1, 1, 1]);
  assert.deepEqual(g.synths, [0, 0, 0]);
  assert.equal(g.mode, "Original mix");
});

test("mute removes a stem; solo keeps only soloed rows (several allowed)", () => {
  let g = mixGains(state({ tracks: [{ muted: true }, {}, {}] }));
  assert.deepEqual(g.stems, [0, 1, 1]);
  assert.equal(g.mode, "Stems");
  g = mixGains(state({ tracks: [{ soloed: true }, {}, { soloed: true }] }));
  assert.deepEqual(g.stems, [1, 0, 1]);
  assert.equal(g.residual, 0);
  assert.equal(g.original, 0);
});

test("synth swap: stem off, synth on; muted synth track is silent", () => {
  let g = mixGains(state({ tracks: [{ synth: true }, {}, {}] }));
  assert.deepEqual(g.stems, [0, 1, 1]);
  assert.deepEqual(g.synths, [1, 0, 0]);
  assert.equal(g.mode, "Stems + synth");
  g = mixGains(state({ tracks: [{ synth: true, muted: true }, {}, {}] }));
  assert.deepEqual(g.synths, [0, 0, 0]);
  assert.deepEqual(g.stems, [0, 1, 1]);
});

test("soloing Original plays the file alone (A/B); muting it leaves only synths", () => {
  let g = mixGains(state({ original: { soloed: true } }));
  assert.equal(g.original, 1);
  assert.deepEqual(g.stems, [0, 0, 0]);
  assert.equal(g.residual, 0);
  g = mixGains(state({ original: { muted: true }, tracks: [{ synth: true }, {}, {}] }));
  assert.equal(g.original, 0);
  assert.deepEqual(g.stems, [0, 0, 0]);
  assert.deepEqual(g.synths, [1, 0, 0]);
});

test("volumes multiply: track volume x original row volume", () => {
  const g = mixGains(state({ original: { volume: 0.5 }, tracks: [{ volume: 0.5 }, {}, {}] }));
  assert.deepEqual(g.stems, [0.25, 0.5, 0.5]);
  assert.equal(g.residual, 0.5);
});

test("before stems: the file plays as Original; synth/solo on a track play its synth", () => {
  let g = mixGains(state({ stemsReady: false }));
  assert.equal(g.original, 1);
  assert.deepEqual(g.stems, [0, 0, 0]);
  assert.equal(g.mode, "Original mix");
  g = mixGains(state({ stemsReady: false, tracks: [{}, { soloed: true }, {}] }));
  assert.equal(g.original, 0);
  assert.deepEqual(g.synths, [0, 1, 0]);
  g = mixGains(state({ stemsReady: false, tracks: [{ synth: true }, {}, {}] }));
  assert.equal(g.original, 1);
  assert.deepEqual(g.synths, [1, 0, 0]);
  assert.equal(g.mode, "Original mix + synth");
});

test("original and stems are never audible together", () => {
  const flags = [false, true];
  for (const stemsReady of flags)
    for (const os of flags)
      for (const om of flags)
        for (const ts of flags)
          for (const tsy of flags) {
            const g = mixGains(
              state({ stemsReady, original: { soloed: os, muted: om }, tracks: [{ soloed: ts, synth: tsy }, {}, {}] }),
            );
            if (g.original > 0) assert.ok(g.stems.every((x) => x === 0) && g.residual === 0);
          }
});
