import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { ROOT, hasFixture, npy, nodeSessions } from "./fixtures.mjs";
import { F16_TABLE } from "../web/js/f16.js";
import {
  N_EV,
  C_H,
  decodeMaps,
  mapEvidence,
  gatherHidden,
  assemble,
  inputNotes,
  takeRows,
  candidates,
  edits,
  apply,
  candidateRows,
  concatRows,
  safeDedupIndex,
  finishRows,
  repair,
  pyRound4,
  splitRowsByKind,
  sliceNotes,
} from "../web/js/evidence.js";

const META = JSON.parse(fs.readFileSync(path.join(ROOT, "web", "models", "meta.json"), "utf8"));
const TH = { ...META.transcriber.thresholds, hop: META.transcriber.hop };
const CFG = META.repairer;
const HEADS = ["onset", "frame", "offset", "onset2", "frame2", "offset2"];
const CLIPS = ["band", "duet", "piano"];
const hasModel = fs.existsSync(path.join(ROOT, "web", "models", "repairer.onnx"));
const hasTaps = (clip) => hasFixture(clip) && fs.existsSync(path.join(ROOT, "fixtures", clip, "hidden.npy"));

function loadMaps(clip) {
  const { data, shape } = npy(clip, "maps");
  const sz = shape[1] * shape[2];
  return Object.fromEntries(HEADS.map((h, i) => [h, data.subarray(i * sz, (i + 1) * sz)]));
}

/** A TapStore stand-in over the reference's float16 hidden buffer [T][88][368]. */
function fixtureStore(clip) {
  const bits = npy(clip, "hidden").data;
  return {
    vec(tap, t, lane, out = new Float32Array(C_H)) {
      assert.equal(tap, "hidden");
      const o = (t * 88 + lane) * C_H;
      for (let c = 0; c < C_H; c++) out[c] = F16_TABLE[bits[o + c]];
      return out;
    },
  };
}

function tnotes(clip) {
  const r = {};
  for (const k of ["onset", "offset", "pitch", "conf", "slot"]) r[k] = npy(clip, `tnotes_${k}`).data;
  return r;
}

function countDiff(a, b, tol = 0) {
  assert.equal(a.length, b.length, "length");
  let bad = 0;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const bi = Number(b[i]);
    if (Number.isNaN(a[i]) && Number.isNaN(bi)) continue;
    const d = Math.abs(a[i] - bi);
    if (!(d <= tol)) bad++;
    if (d > max || Number.isNaN(d)) max = Number.isNaN(d) ? Infinity : d;
  }
  return { bad, max };
}

/** The reference's edits for a clip (rep_pk / rep_on_r / rep_off_r / rep_ins). */
function refEdits(clip) {
  const insF = npy(clip, "rep_ins").data;
  const ins = [];
  for (let i = 0; i < insF.length; i += 4) ins.push([insF[i], insF[i + 1], insF[i + 2], insF[i + 3]]);
  return { pk: npy(clip, "rep_pk").data, onR: npy(clip, "rep_on_r").data, offR: npy(clip, "rep_off_r").data, ins };
}

const hasExtra = (clip) => fs.existsSync(path.join(ROOT, "fixtures", clip, "evidence", "apply_all_onset.npy"));

let rtP = null;
const runtime = () => (rtP ??= nodeSessions());

for (const clip of CLIPS) {
  const skip = !hasFixture(clip);

  test(`mapEvidence equals ev_map (${clip})`, { skip }, () => {
    const maps = loadMaps(clip);
    const { L, A, B, S } = decodeMaps(maps, TH);
    const r = countDiff(mapEvidence(maps, L, A, B, S), npy(clip, "ev_map").data);
    assert.equal(r.bad, 0, `mismatches ${r.bad}, max ${r.max}`);
  });

  test(`assemble(ev_map, ev_at, ev_mean) equals ev (${clip})`, { skip }, () => {
    const ev = assemble(npy(clip, "ev_map").data, npy(clip, "ev_at").data, npy(clip, "ev_mean").data);
    const r = countDiff(ev, npy(clip, "ev").data);
    console.log(`# ${clip} assemble: ${r.bad}/${ev.length} differ, max ${r.max}`);
    assert.equal(r.bad, 0);
  });

  test(`inputNotes equals rep_in_* (${clip})`, { skip }, () => {
    const { k, order } = inputNotes(tnotes(clip));
    for (const [key, ref] of [
      ["pitch", "pitch"],
      ["vel", "vel"],
      ["on", "on"],
      ["off", "off"],
    ]) {
      const r = countDiff(k[key], npy(clip, `rep_in_${ref}`).data);
      assert.equal(r.bad, 0, `${key}: ${r.bad} mismatches`);
    }
    assert.equal(countDiff(order, npy(clip, "rep_in_order").data).bad, 0, "order");
  });

  test(`apply(reference edits) equals rep_* (${clip})`, { skip }, () => {
    const { k } = inputNotes(tnotes(clip));
    const rep = apply(k, refEdits(clip), CFG.keep_th, CFG.insert_th);
    for (const f of ["onset", "offset", "pitch", "inserted", "score"]) {
      const r = countDiff(rep[f], npy(clip, `rep_${f}`).data);
      assert.equal(r.bad, 0, `${f}: ${r.bad} mismatches, max ${r.max}`);
    }
  });

  // thresholds lowered so real insertion proposals go through the insert filter and both dedup stages
  // (reference arrays from tests/py/evidence_fixtures.py)
  for (const [tag, keepTh, insertTh] of [
    ["lo", 0.2, 0.05],
    ["all", 0.0, 0.0],
  ]) {
    test(
      `apply(reference edits) at keep ${keepTh} / insert ${insertTh} equals the reference (${clip})`,
      {
        skip: !hasExtra(clip),
      },
      () => {
        const { k } = inputNotes(tnotes(clip));
        const rep = apply(k, refEdits(clip), keepTh, insertTh);
        const ref = (f) => npy(clip, `evidence/apply_${tag}_${f}`).data;
        assert.equal(rep.onset.length, ref("onset").length, "note count");
        for (const f of ["onset", "offset", "pitch", "inserted", "score"]) {
          const r = countDiff(rep[f], ref(f));
          assert.equal(r.bad, 0, `${f}: ${r.bad} mismatches, max ${r.max}`);
        }
      },
    );
  }

  test(`repairer window 0 matches rep_w0_* (${clip})`, { skip: skip || !hasModel }, async () => {
    const { k, order } = inputNotes(tnotes(clip));
    const ev = takeRows(npy(clip, "ev").data, order, N_EV);
    let lo = 0; // notes starting before 0 s are in no window (searchsorted(on, 0.0))
    while (lo < k.n && k.on[lo] < 0) lo++;
    let hi = lo;
    while (hi < k.n && k.on[hi] < 8) hi++;
    const c = await candidates(await runtime(), k, ev, lo, hi, 0);
    const pitch = countDiff(c.pitch, npy(clip, "rep_w0_pitch").data);
    const score = countDiff(c.score, npy(clip, "rep_w0_score").data, 1e-4);
    const on = countDiff(c.on, npy(clip, "rep_w0_on").data, 1e-5);
    const off = countDiff(c.off, npy(clip, "rep_w0_off").data, 1e-4);
    console.log(
      `# ${clip} w0 (${c.on.length} rows): score max ${score.max.toExponential(2)}, on flips ${on.bad} (max ${on.max}),` +
        ` off flips ${off.bad} (max ${off.max.toExponential(2)}), pitch flips ${pitch.bad}`,
    );
    assert.equal(score.bad, 0);
    assert.ok(on.bad + off.bad + pitch.bad <= Math.ceil(0.02 * c.on.length), "too many argmax flips");
  });

  test(`edits + apply via ONNX match rep_* (${clip})`, { skip: skip || !hasModel }, async () => {
    const { k, order } = inputNotes(tnotes(clip));
    const e = await edits(await runtime(), k, takeRows(npy(clip, "ev").data, order, N_EV));
    const pk = countDiff(e.pk, npy(clip, "rep_pk").data, 1e-4);
    const on = countDiff(e.onR, npy(clip, "rep_on_r").data, 1e-4);
    const off = countDiff(e.offR, npy(clip, "rep_off_r").data, 1e-4);
    const insRef = npy(clip, "rep_ins").data;
    assert.equal(e.ins.length * 4, insRef.length, "insertion count");
    const ins = countDiff(e.ins.flat(), insRef, 1e-4);
    console.log(
      `# ${clip} edits: pk max ${pk.max.toExponential(2)} (${pk.bad} > 1e-4), onset flips ${on.bad}, end flips ${off.bad}` +
        ` (max ${off.max.toExponential(2)}), insertion values off ${ins.bad}/${insRef.length}`,
    );
    assert.equal(pk.bad, 0);
    assert.equal(ins.bad, 0, "insertion proposals");
    // ORT vs torch can flip the onset box-argmax on near ties: at most 3 % of notes, by one 5 ms bin
    assert.ok(on.bad <= Math.ceil(0.03 * k.n), `${on.bad} onset flips`);
    assert.ok(on.max <= 0.00501 && off.max <= 0.00501, "flips larger than one 5 ms bin");
    const rep = apply(k, e, CFG.keep_th, CFG.insert_th);
    assert.equal(rep.onset.length, npy(clip, "rep_onset").data.length, "repaired note count");
    const ro = countDiff(rep.onset, npy(clip, "rep_onset").data, 1e-4);
    const rf = countDiff(rep.offset, npy(clip, "rep_offset").data, 1e-4);
    assert.equal(countDiff(rep.pitch, npy(clip, "rep_pitch").data).bad, 0, "pitch");
    assert.equal(countDiff(rep.inserted, npy(clip, "rep_inserted").data).bad, 0, "inserted");
    assert.equal(countDiff(rep.score, npy(clip, "rep_score").data, 1e-4).bad, 0, "score");
    assert.ok(ro.bad <= on.bad && ro.max <= 0.00501, "repaired onsets differ beyond the onset flips");
    assert.ok(rf.bad <= off.bad && rf.max <= 0.00501, "repaired ends differ beyond the end flips");
  });
}

test("gatherHidden (exact) equals ev_at / ev_mean (band)", { skip: !hasTaps("band") }, () => {
  const maps = loadMaps("band");
  const { L, A, B } = decodeMaps(maps, TH);
  const store = fixtureStore("band");
  const { at, mean } = gatherHidden(store, L, A, B, { exact: true });
  assert.equal(countDiff(at, npy("band", "ev_at").data).bad, 0, "at");
  const m = countDiff(mean, npy("band", "ev_mean").data);
  assert.equal(m.bad, 0, `mean: ${m.bad} mismatches, max ${m.max}`);
  const d = gatherHidden(store, L, A, B, { exact: false });
  const md = countDiff(d.mean, npy("band", "ev_mean").data, 1e-4);
  console.log(`# band gatherHidden direct (float64) mean: max diff ${md.max.toExponential(2)}`);
  assert.equal(md.bad, 0);
});

test(
  "browser path: gatherHidden (float64 direct) -> ev and edits vs reference (band)",
  {
    skip: !hasTaps("band") || !hasModel,
  },
  async () => {
    const maps = loadMaps("band");
    const { res, L, A, B, S } = decodeMaps(maps, TH);
    const { at, mean } = gatherHidden(fixtureStore("band"), L, A, B);
    const ev = assemble(mapEvidence(maps, L, A, B, S), at, mean);
    const evd = countDiff(ev, npy("band", "ev").data);
    const { k, order } = inputNotes(res);
    const rep = apply(k, await edits(await runtime(), k, takeRows(ev, order, N_EV)), CFG.keep_th, CFG.insert_th);
    console.log(
      `# band direct-sum ev: ${evd.bad}/${ev.length} entries differ (max ${evd.max}); repaired ${rep.onset.length}`,
    );
    assert.ok(evd.max <= 1 / 32 + 1e-6, "only single int8 quantisation steps may differ");
    assert.equal(rep.onset.length, npy("band", "rep_onset").data.length);
    assert.equal(countDiff(rep.pitch, npy("band", "rep_pitch").data).bad, 0);
  },
);

test("repair() end to end from maps + taps equals rep_* (band)", { skip: !hasTaps("band") || !hasModel }, async () => {
  const out = await repair(await runtime(), loadMaps("band"), fixtureStore("band"), TH, CFG);
  const ref = npy("band", "rep_onset").data;
  assert.equal(out.rep.onset.length, ref.length);
  assert.equal(out.info.transcribed, 146);
  assert.equal(out.info.kept + out.info.dropped, out.info.transcribed);
  assert.equal(countDiff(out.rep.pitch, npy("band", "rep_pitch").data).bad, 0);
});

test("dedup: an insertion 10 ms from a kept note on the same pitch is dropped; the kept note stays", () => {
  const k = {
    n: 2,
    inst: new Uint8Array(2),
    pitch: Uint8Array.of(60, 64),
    vel: Uint8Array.of(80, 80),
    on: Float32Array.of(1.0, 2.0),
    off: Float32Array.of(1.5, 2.5),
  };
  const e = {
    pk: Float64Array.of(0.5, 0.9),
    onR: Float64Array.of(1.0, 2.0),
    offR: Float64Array.of(1.5, 2.5),
    ins: [
      [1.01, 1.4, 60, 0.99], // 10 ms after the kept note on 60, higher score: still dropped
      [3.0, 3.2, 62, 0.7], // two insertions 20 ms apart on 62: the higher score wins
      [3.02, 3.3, 62, 0.8],
      [4.0, 4.1, 65, 0.5], // below insert_th
    ],
  };
  const rows = candidateRows(k, e, 0.4, 0.6);
  assert.deepEqual(Array.from(rows.kind), [0, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(safeDedupIndex(rows)), [0, 1, 4]);
  const rep = apply(k, e, 0.4, 0.6);
  assert.deepEqual(Array.from(rep.pitch), [60, 64, 62]);
  assert.deepEqual(Array.from(rep.inserted), [0, 0, 1]);
  assert.deepEqual(Array.from(rep.onset), [1, 2, 3.02]);
  // the same through block-wise rows
  const both = concatRows([candidateRows(k, e, 0.4, 0.6), candidateRows(k, { ...e, ins: [] }, 0.4, 0.6)]);
  assert.equal(both.on.length, 7);
  const split = splitRowsByKind(both);
  assert.deepEqual(Array.from(split.kept.pitch), [60, 64, 60, 64]);
  assert.deepEqual(Array.from(split.inserted.on), [1.01, 3.0, 3.02]);
  assert.equal(sliceNotes(k, 1, 2).pitch[0], 64);
  assert.deepEqual(finishRows(rows, safeDedupIndex(rows)), rep);
});

test("pyRound4 rounds exact ties to even like Python round(x, 4)", () => {
  assert.equal(pyRound4(0.03125), 0.0312); // exact tie, 2 is even
  assert.equal(pyRound4(0.09375), 0.0938); // exact tie, 7 is odd -> up
  assert.equal(pyRound4(1.00005), 1.0001); // not an exact tie in binary (above)
  assert.equal(pyRound4(-0.03125), -0.0312);
  assert.equal(pyRound4(2.12344), 2.1234);
});

test(
  "edits on one window's notes alone equals that window of the full run (piano)",
  { skip: !hasFixture("piano") || !hasModel },
  async () => {
    const { k, order } = inputNotes(tnotes("piano"));
    const ev = takeRows(npy("piano", "ev").data, order, N_EV);
    const rt = await runtime();
    const full = await edits(rt, k, ev);
    let lo = 0;
    while (k.on[lo] < 8) lo++;
    let hi = lo;
    while (hi < k.n && k.on[hi] < 16) hi++;
    const sub = Object.fromEntries(Object.entries(k).map(([f, v]) => [f, f === "n" ? hi - lo : v.slice(lo, hi)]));
    const part = await edits(rt, sub, ev.slice(lo * N_EV, hi * N_EV));
    assert.deepEqual(Array.from(part.pk), Array.from(full.pk.subarray(lo, hi)));
    assert.deepEqual(Array.from(part.onR), Array.from(full.onR.subarray(lo, hi)));
    const inWin = full.ins.filter((r) => r[0] >= 8 && r[0] < 16);
    assert.deepEqual(part.ins, inWin);
  },
);
