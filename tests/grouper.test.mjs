// grouper.js against the reference: each stage gets the reference's own input (fixtures/<clip>/ from scripts/dump_fixtures.py,
// fixtures/<clip>/grouper/ from tests/py/grouper_fixtures.py) and is compared with the reference's output.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { ROOT, hasFixture, manifest, nodeSessions, npy, relDiff } from "./fixtures.mjs";
import { F16_TABLE } from "../web/js/f16.js";
import {
  assignSlots,
  decode,
  gather,
  group,
  loadTrackHead,
  makeSong,
  predict,
  runWindow,
  songWindows,
  tracks,
  windowInputs,
} from "../web/js/grouper.js";

const MODELS = path.join(ROOT, "web", "models");
const MODELS_OK = ["meta.json", "track_head.bin", "grouper.onnx"].every((f) => fs.existsSync(path.join(MODELS, f)));
const NO_MODELS = MODELS_OK ? false : "web/models not exported (run scripts/export_onnx.py)";
const CLIPS = MODELS_OK ? ["band", "duet", "piano"].filter(hasFixture) : [];
const META = MODELS_OK ? JSON.parse(fs.readFileSync(path.join(MODELS, "meta.json"), "utf8")) : null;
const CFG = META?.grouper;
const TH = MODELS_OK
  ? loadTrackHead(new Uint8Array(fs.readFileSync(path.join(MODELS, "track_head.bin"))).buffer, CFG.track_head_shapes)
  : null;
let rtPromise = null;
/** One shared onnxruntime-web session factory for the whole file (session creation is not part of the timings). */
const getRt = () => (rtPromise ??= nodeSessions());
const HEADS = ["onset", "frame", "offset", "onset2", "frame2", "offset2"];
const N_KEYS = 88;
const C = 368;

const exists = (clip, key) => fs.existsSync(path.join(ROOT, "fixtures", clip, `${key}`));
const hasExtra = (clip) => exists(clip, "grouper/w0_probs.npy");
const json = (clip, key) =>
  JSON.parse(fs.readFileSync(path.join(ROOT, "fixtures", clip, "grouper", `${key}.json`), "utf8"));
const rows = ({ data, shape }) =>
  Array.from({ length: shape[0] }, (_, i) => data.subarray(i * shape[1], (i + 1) * shape[1]));
const f16arr = (a) => Float32Array.from(a.data, (b) => F16_TABLE[b]);
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const groupsJson = (g) => g.map(([k, ids]) => [k, Array.from(ids)]);
const NO_EXTRA = "run tests/py/grouper_fixtures.py first";
/** false when every condition holds, else the first skip reason. */
const need = (...reasons) => reasons.find(Boolean) ?? false;

/** Song inputs (gathered by the reference) of a clip. */
function songOf(clip) {
  return makeSong(
    npy(clip, "y").data,
    npy(clip, "grp_meta").data,
    npy(clip, "grp_feat").data,
    f16arr(npy(clip, "grp_fsc")),
  );
}

/** Notes (on, off, pitch) as Grouper.group hands them to decode. */
function noteArrays(meta) {
  const n = meta.length / 12;
  const col = (c) => Float64Array.from({ length: n }, (_, i) => meta[i * 12 + c]);
  return [col(0), col(1), col(2).map((v) => Math.round(v))];
}

/** Grouper.predict output as saved by the reference: (dir "" = dump_fixtures arrays, else grouper/<prefix>_*). */
function savedPred(clip, prefix) {
  if (!prefix) {
    const man = manifest(clip);
    const wins = man.grp_wins.map(([t0, sel], i) => ({
      t0,
      sel: Int32Array.from(sel),
      emb: npy(clip, `grp_win${i}_emb`).data,
      probs: npy(clip, `grp_win${i}_probs`).data,
    }));
    return {
      probs: npy(clip, "grp_probs").data,
      emb: npy(clip, "grp_emb").data,
      h: npy(clip, "grp_h").data,
      wins,
      W: CFG.window_s,
    };
  }
  const g = `grouper/${prefix}`;
  const wins = json(clip, `${prefix}_wins`).map(([t0, sel], i) => ({
    t0,
    sel: Int32Array.from(sel),
    emb: npy(clip, `${g}_win${i}_emb`).data,
    probs: npy(clip, `${g}_win${i}_probs`).data,
  }));
  return {
    probs: npy(clip, `${g}_probs`).data,
    emb: npy(clip, `${g}_emb`).data,
    h: npy(clip, `${g}_h`).data,
    wins,
    W: CFG.window_s,
  };
}

/** A tap store over the reference's float16 sub-run tap buffers [T][88][368]. */
function fixtureStore(clip) {
  const bufs = { subHidden: npy(clip, "sub_hidden").data, subStem: npy(clip, "sub_stem").data };
  return {
    vec(tap, t, lane, out = new Float32Array(C)) {
      const b = bufs[tap];
      const o = (t * N_KEYS + lane) * C;
      for (let c = 0; c < C; c++) out[c] = F16_TABLE[b[o + c]];
      return out;
    },
  };
}

function subMaps(clip) {
  const { data, shape } = npy(clip, "smaps");
  const sz = shape[1] * shape[2];
  return Object.fromEntries(HEADS.map((h, k) => [h, data.subarray(k * sz, (k + 1) * sz)]));
}

/** The fixed front end of window 0 (all three clips have a single grouper window). */
function fixedFront(clip) {
  const X = npy(clip, "grp_w0_X");
  return () => ({ X: X.data, P: npy(clip, "grp_w0_P").data, T: X.shape[1] });
}

for (const clip of CLIPS) {
  test(`${clip}: assignSlots == assign_slots`, () => {
    const nt = assignSlots(rows(npy(clip, "grp_notes_in")), rows(npy(clip, "grp_own")));
    assert.deepEqual(Array.from(nt), Array.from(npy(clip, "grp_nt").data));
  });

  test(`${clip}: songWindows == song_windows`, () => {
    const meta = npy(clip, "grp_meta").data;
    const on = meta.filter((_, i) => i % 12 === 0);
    const jobs = songWindows(on, CFG.window_s, CFG.window_s * CFG.window_hop, CFG.max_notes);
    assert.deepEqual(
      jobs.map(({ t0, sel }) => [t0, Array.from(sel)]),
      manifest(clip).grp_jobs,
    );
  });

  test(`${clip}: windowInputs == _Song.window (audio, notes, feat)`, () => {
    const song = songOf(clip);
    const [t0, sel] = manifest(clip).grp_jobs[0];
    const w = windowInputs(song, t0, CFG.window_s, sel);
    assert.deepEqual(Array.from(w.audio), Array.from(npy(clip, "grp_w0_audio").data));
    assert.deepEqual(Array.from(w.notes), Array.from(npy(clip, "grp_w0_notes").data));
    assert.equal(maxAbs(w.feat, npy(clip, "grp_w0_feat").data), 0);
  });

  test(`${clip}: decode(reference predictions) == grp_groups exactly`, () => {
    const t = performance.now();
    const g = decode(
      savedPred(clip, ""),
      noteArrays(npy(clip, "grp_meta").data),
      TH,
      CFG.decode,
      META.instruments.credit,
    );
    const ms = performance.now() - t;
    assert.deepEqual(groupsJson(g), manifest(clip).grp_groups);
    console.log(`# ${clip} decode ${ms.toFixed(1)} ms`);
  });

  test(`${clip}: tracks == Grouper._tracks`, () => {
    const meta = npy(clip, "grp_meta").data;
    const [on, off, pitch] = noteArrays(meta);
    const n = on.length;
    const conf = Float32Array.from({ length: n }, (_, i) =>
      meta[i * 12 + 3] > 0.5 ? meta[i * 12 + 5] : meta[i * 12 + 4],
    );
    const man = manifest(clip);
    const r = man.tuning.r;
    const groups = man.grp_groups.map(([k, ids]) => [k, Int32Array.from(ids)]);
    const out = tracks(
      on.map((v) => v * r),
      off.map((v) => v * r),
      pitch,
      conf,
      groups,
      META.instruments,
    );
    assert.deepEqual(
      out.map((t) => ({ instrument: t.instrument, program: t.program, notes: t.notes })),
      man.tracks,
    );
  });

  test(`${clip}: grouper.onnx window 0 vs the PyTorch model`, { skip: hasExtra(clip) ? false : NO_EXTRA }, async () => {
    const rt = await getRt();
    const X = npy(clip, "grp_w0_X");
    const notes = npy(clip, "grp_w0_notes").data;
    const win = { notes, feat: npy(clip, "grp_w0_feat").data, m: notes.length / 12 };
    const front = { X: X.data, P: npy(clip, "grp_w0_P").data, T: X.shape[1] };
    await runWindow(rt, front, win); // warm-up: session creation and first-run allocations are not timed
    const t = performance.now();
    const o = await runWindow(rt, front, win);
    const ms = performance.now() - t;
    const d = {
      probs: maxAbs(o.probs, npy(clip, "grouper/w0_probs").data),
      emb: maxAbs(o.emb, npy(clip, "grouper/w0_emb").data),
      h: relDiff(o.h, npy(clip, "grouper/w0_h").data),
    };
    console.log(
      `# ${clip} window 0 (${win.m} notes) ${ms.toFixed(0)} ms, max |d| probs ${d.probs.toExponential(2)} emb ${d.emb.toExponential(2)}, rel h ${d.h.toExponential(2)}`,
    );
    assert.ok(d.probs < 1e-4 && d.emb < 1e-4 && d.h < 5e-4, JSON.stringify(d));
  });

  test(`${clip}: predict == Grouper.predict, and decode(predict) == grp_groups`, async () => {
    const rt = await getRt();
    const song = songOf(clip);
    const t = performance.now();
    const P = await predict(rt, song, fixedFront(clip), CFG);
    const ms = performance.now() - t;
    const ref = savedPred(clip, "");
    const d = { probs: maxAbs(P.probs, ref.probs), emb: maxAbs(P.emb, ref.emb), h: relDiff(P.h, ref.h) };
    console.log(
      `# ${clip} predict ${P.wins.length} window(s) ${ms.toFixed(0)} ms, max |d| probs ${d.probs.toExponential(2)} emb ${d.emb.toExponential(2)} rel h ${d.h.toExponential(2)}`,
    );
    assert.ok(d.probs < 1e-4 && d.emb < 1e-4 && d.h < 1e-3, JSON.stringify(d));
    assert.deepEqual(
      P.wins.map((w) => [w.t0, Array.from(w.sel)]),
      manifest(clip).grp_wins,
    );
    const g = decode(P, noteArrays(song.meta), TH, CFG.decode, META.instruments.credit);
    assert.deepEqual(groupsJson(g), manifest(clip).grp_groups);
  });
}

test(
  "band: gather == core.gather (fixture tap store)",
  {
    skip: need(
      NO_MODELS,
      !hasFixture("band") || !exists("band", "sub_hidden.npy") ? "no tap buffers (dump_fixtures.py --taps)" : false,
    ),
  },
  () => {
    const nt = npy("band", "grp_nt").data;
    const t = performance.now();
    const { meta, feat, fsc } = gather(subMaps("band"), fixtureStore("band"), nt);
    console.log(`# band gather ${(performance.now() - t).toFixed(0)} ms`);
    assert.deepEqual(Array.from(meta), Array.from(npy("band", "grp_meta").data));
    assert.deepEqual(Array.from(fsc), Array.from(f16arr(npy("band", "grp_fsc"))));
    assert.deepEqual(Array.from(feat), Array.from(npy("band", "grp_feat").data));
  },
);

test(
  "band: group() end to end == Grouper.group tracks",
  {
    skip: need(
      NO_MODELS,
      !hasFixture("band") || !exists("band", "sub_hidden.npy") ? "no tap buffers (dump_fixtures.py --taps)" : false,
    ),
  },
  async () => {
    const rt = await getRt();
    const man = manifest("band");
    const cfg = { ...CFG, th: TH, instruments: META.instruments };
    const { tracks: out, info } = await group(
      rt,
      npy("band", "y").data,
      rows(npy("band", "grp_notes_in")),
      rows(npy("band", "grp_own")),
      subMaps("band"),
      fixtureStore("band"),
      man.tuning,
      fixedFront("band"),
      cfg,
    );
    assert.deepEqual(
      out.map((t) => ({ instrument: t.instrument, program: t.program, notes: t.notes })),
      man.tracks,
    );
    assert.equal(info.notesIn, npy("band", "grp_meta").shape[0]);
  },
);

test(
  "songWindows == song_windows (synthetic onsets, small nmax)",
  { skip: exists("band", "grouper/cluster_refs.json") ? false : NO_EXTRA },
  () => {
    for (const c of json("band", "cluster_refs").windows) {
      const jobs = songWindows(Float32Array.from(c.on), 24.0, 12.0, c.nmax);
      assert.deepEqual(
        jobs.map(({ t0, sel }) => [t0, Array.from(sel)]),
        c.jobs,
      );
    }
  },
);

const MW = need(NO_MODELS, hasFixture("band") && exists("band", "grouper/mw_meta.npy") ? false : NO_EXTRA);

test("band multi-window song: predict == Grouper.predict, decode == reference groups", { skip: MW }, async () => {
  const rt = await getRt();
  const meta = npy("band", "grouper/mw_meta").data;
  const song = makeSong(
    new Float32Array(16000 * 90),
    meta,
    npy("band", "grouper/mw_feat").data,
    f16arr(npy("band", "grouper/mw_fsc")),
  );
  const t = performance.now();
  const P = await predict(rt, song, fixedFront("band"), CFG);
  const ms = performance.now() - t;
  const ref = savedPred("band", "mw");
  const d = { probs: maxAbs(P.probs, ref.probs), emb: maxAbs(P.emb, ref.emb), h: relDiff(P.h, ref.h) };
  console.log(
    `# band multi-window predict: ${P.wins.length} windows, ${(ms / P.wins.length).toFixed(0)} ms / window, max |d| ${JSON.stringify(d)}`,
  );
  assert.deepEqual(
    P.wins.map((w) => [w.t0, Array.from(w.sel)]),
    json("band", "mw_wins"),
  );
  assert.ok(d.probs < 1e-4 && d.emb < 1e-4 && d.h < 1e-3, JSON.stringify(d));
  const notes = noteArrays(meta);
  assert.deepEqual(groupsJson(decode(ref, notes, TH, CFG.decode, META.instruments.credit)), json("band", "mw_groups"));
  assert.deepEqual(groupsJson(decode(P, notes, TH, CFG.decode, META.instruments.credit)), json("band", "mw_groups"));
});

for (const name of ["synA", "synB", "synC"]) {
  test(
    `synthetic decode ${name} == reference groups`,
    { skip: need(NO_MODELS, exists("band", `grouper/${name}_on.npy`) ? false : NO_EXTRA) },
    () => {
      const g = `grouper/${name}`;
      const notes = [npy("band", `${g}_on`).data, npy("band", `${g}_off`).data, npy("band", `${g}_pitch`).data];
      const P = savedPred("band", name);
      const t = performance.now();
      const out = groupsJson(decode(P, notes, TH, CFG.decode, META.instruments.credit));
      console.log(
        `# ${name} decode (${notes[0].length} notes, ${P.wins.length} windows) ${(performance.now() - t).toFixed(0)} ms`,
      );
      // JS sorts stably; NumPy's argsort (AVX-512) is not. Where the reference itself changes with a stable argsort, only the
      // stable variant is a port check.
      assert.deepEqual(out, json("band", `${name}_groups_stable`));
      if (json("band", "syn_info")[name].stable_same) assert.deepEqual(out, json("band", `${name}_groups`));
    },
  );
}
