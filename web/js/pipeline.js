// The whole Dettato system in the tab: mono audio -> instrument-named tracks + a drum track.
//
// This is dettato/pipeline.py (Dettato.transcribe / Dettato.pitched) with one structural change, made for memory: the reference
// holds the transcriber's whole-song tap buffers (~10 MB per second of audio) and runs every later stage after the whole pass. Here
// the pass streams. After every window the frames no later window can touch are final, and each 8 s block (one repairer window)
// is processed as soon as the frames it reads are final: transcriber notes -> evidence -> repairer -> the grouper's per-note
// features. The tap buffers then only hold a sliding ~26 s region. Up to the cases below the arithmetic is the reference's:
//   * a note that starts in a block and lasts longer than ~8 s past the block's end is decoded against the maps final so far
//     (its end, and the evidence / features averaged over it, are cut there);
//   * nothing else differs: the repairer's windows are the reference's (aligned at multiples of 8 s), and the global de-duplication
//     of the repaired notes runs once at the end over all blocks, as in the reference.
// Fast mode (opts.fast) uses one window every 4 s instead of every 2 s: half the network time; the repairer then reads the 4 s-hop
// average (the reference reads the 2 s one), so results differ a little from the reference.

import { resample, retuneRatio, retuneResample, hcqt, grouperFront, drumFeatures } from "./dsp.js";
import { layout, frameOffset, hannTaper, notes as transcriberNotes } from "./transcriber.js";
import * as EV from "./evidence.js";
import * as GR from "./grouper.js";
import { transcribeDrums } from "./drums.js";
import { TapStore, tapStoreBytes, N_KEYS, HEADS } from "./tapstore.js";

export const SR = 16000;
const HOP = 0.02; // transcriber frame (s)
const BLOCK_S = 8.0; // repairer window
const BLOCK_F = Math.round(BLOCK_S / HOP); // 400 frames
const AFTER_F = 400; // frames a block waits for past its end before it is processed (8 s)
const MIN_END_S = 0.03; // every pitched note lasts at least 30 ms before grouping
const MIN_SOURCE_RATE = 40000;

/** Python round() / np.round for a scalar: round half to even. */
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The first `F` frames of each map (views, no copy). */
function prefixMaps(maps, F) {
  const out = {};
  for (const h of HEADS) out[h] = maps[h].subarray(0, F * N_KEYS);
  return out;
}

/** Progress reporter with an ETA from the observed rate of the current stage. */
function makeProgress(onProgress) {
  const t0 = performance.now();
  let stage = null;
  let stageStart = 0;
  return (s, label, done, total) => {
    const now = performance.now();
    if (s !== stage) {
      stage = s;
      stageStart = now;
    }
    const el = (now - stageStart) / 1000;
    const etaS = done > 0 && total > done ? (el / done) * (total - done) : undefined;
    onProgress?.({ stage: s, label, done, total, etaS, elapsedS: (now - t0) / 1000 });
  };
}

/** One row of a block's candidate notes after apply()'s final rounding -> the grouper input note [on, off, pitch]. */
function grouperNote(on, off, pitch) {
  return [on, Math.max(off, on + MIN_END_S), pitch];
}

function noteKey(on, off, pitch) {
  return `${on}|${off}|${pitch}`;
}

/**
 * The pitched part on 16 kHz mono x: tuning -> streaming transcriber pass with per-block evidence repair and grouper features ->
 * grouper. Returns tracks on x's time axis plus diagnostics.
 */
async function pitched(rt, x, meta, opts, progress) {
  progress("tuning", "Estimating tuning", 0, 1);
  const tun = retuneRatio(x);
  const y = tun.shift ? retuneResample(x, tun.r) : x;
  progress("tuning", `Tuning ${tun.delta.toFixed(1)} cents`, 1, 1);

  const fast = !!opts.fast;
  const lay = layout(y.length, fast ? 0.5 : 0.25);
  const { W, Fw, T, starts, pad, gPad } = lay;
  const subStep = Math.max(1, Math.round(W * 0.5));
  const cap = BLOCK_F + AFTER_F + Fw + 200;
  const store = new TapStore(T, cap, fast);
  const taper = hannTaper(Fw);
  const th = meta.transcriber.thresholds;
  const repCfg = meta.repairer;
  const xAll = new Float32Array(pad + y.length + pad + W);
  xAll.set(y, pad);

  const nBlocks = Math.max(1, Math.ceil(T / BLOCK_F));
  const kept = []; // per block: candidate rows of kind 0 (kept / unchanged notes), in k order
  const ins = []; // per block: insertion rows, in window order
  const feats = new Map(); // grouper note key -> [{meta, feat, fsc}] gathered while the taps were held
  const stats = { transcribed: 0, cutAtFrontier: 0 };
  const partial = [];
  const tnotes = []; // the transcriber's notes (input of the repairer), for the parity check
  let nextBlock = 0;

  const processBlock = async (j, F) => {
    const maps = prefixMaps(store.maps(), F);
    const smaps = prefixMaps(store.subMaps(), F);
    const dm = EV.decodeMaps(maps, th);
    const { k, order } = EV.inputNotes(dm.res);
    const t0 = j * BLOCK_S;
    // block 0 also takes notes with a (slightly) negative onset: the reference puts them in no repairer window, so they pass
    // through unchanged (edits() leaves them NaN, as in the reference)
    const lo = j === 0 ? 0 : lowerBound(k.on, t0);
    const hi = lowerBound(k.on, t0 + BLOCK_S);
    if (hi <= lo) return;
    const sel = Array.from(order.subarray(lo, hi));
    const L = Int32Array.from(sel, (i) => dm.L[i]);
    const A = Int32Array.from(sel, (i) => dm.A[i]);
    const B = Int32Array.from(sel, (i) => dm.B[i]);
    const S = Int32Array.from(sel, (i) => dm.S[i]);
    for (let i = 0; i < B.length; i++) if (B[i] >= F && F < T) stats.cutAtFrontier++;
    const mf = EV.mapEvidence(maps, L, A, B, S);
    const { at, mean } = EV.gatherHidden(store, L, A, B);
    const ev = EV.assemble(mf, at, mean);
    const kb = EV.sliceNotes(k, lo, hi);
    for (let i = 0; i < kb.on.length; i++) tnotes.push([kb.on[i] * tun.r, kb.off[i] * tun.r, kb.pitch[i]]);
    const e = await EV.edits(rt, kb, ev);
    const rows = EV.candidateRows(kb, e, repCfg.keep_th, repCfg.insert_th);
    stats.transcribed += hi - lo;
    const split = EV.splitRowsByKind(rows);
    kept.push(split.kept);
    ins.push(split.inserted);
    // the grouper's per-note features, while this block's taps are still held
    const fin = EV.finishRows(
      rows,
      Int32Array.from({ length: rows.on.length }, (_, i) => i),
    );
    const gnotes = [];
    for (let i = 0; i < fin.onset.length; i++) gnotes.push(grouperNote(fin.onset[i], fin.offset[i], fin.pitch[i]));
    const own = ownNotes(smaps, th);
    const nt = GR.assignSlots(gnotes, own);
    const g = GR.gather(smaps, store, nt, HOP);
    for (let i = 0; i < nt.length / 4; i++) {
      const key = noteKey(nt[4 * i], nt[4 * i + 1], nt[4 * i + 2]);
      if (!feats.has(key)) feats.set(key, []);
      feats.get(key).push({
        meta: g.meta.slice(12 * i, 12 * i + 12),
        feat: g.feat.slice(3 * 368 * i, 3 * 368 * (i + 1)),
        fsc: g.fsc.slice(3 * i, 3 * i + 3),
      });
    }
    for (let i = 0; i < fin.onset.length; i++)
      partial.push([fin.onset[i] * tun.r, fin.offset[i] * tun.r, fin.pitch[i]]);
    opts.onPartial?.({ notes: partial });
  };

  // ------------------------------------------------------------------ the streaming pass
  const sess = await rt.session("transcriber");
  const tw0 = performance.now();
  for (let w = 0; w < starts.length; w++) {
    const s0 = starts[w];
    const g0 = frameOffset(s0, gPad);
    const F = w + 1 < starts.length ? Math.min(T, Math.max(0, frameOffset(starts[w + 1], gPad))) : T;
    if (Math.min(T, g0 + Fw) > Math.max(0, g0)) {
      const hc = hcqt(xAll.subarray(s0, s0 + W));
      const out = await sess.run({ hcqt: new rt.ort.Tensor("float32", hc.data, [1, 9, hc.T, 288]) });
      store.add(
        { probs: out.probs.data, hidden: out.hidden.data, stem: out.stem.data },
        Fw,
        g0,
        taper,
        fast || s0 % subStep === 0,
      );
      for (const t of Object.values(out)) t.dispose?.();
    }
    store.finalize(F);
    while (nextBlock < nBlocks && (F >= Math.min(T, (nextBlock + 1) * BLOCK_F + AFTER_F) || F === T)) {
      await processBlock(nextBlock, F);
      store.release(Math.max(0, (nextBlock + 1) * BLOCK_F - 2));
      nextBlock++;
    }
    const perWin = (performance.now() - tw0) / 1000 / (w + 1);
    progress("transcriber", `Transcriber window ${w + 1} / ${starts.length}`, w + 1, starts.length, perWin);
  }

  // ------------------------------------------------------------------ the repaired notes (global de-duplication, as the reference)
  progress("repair", "Evidence repair: merging blocks", 0, 1);
  const rowsAll = EV.concatRows([...kept, ...ins]);
  const idx = EV.safeDedupIndex(rowsAll);
  const rep = EV.finishRows(rowsAll, idx);
  const nIns = rep.inserted.reduce((s, v) => s + (v ? 1 : 0), 0);
  const info = {
    transcribed: stats.transcribed,
    kept: rep.onset.length - nIns,
    inserted: nIns,
    dropped: stats.transcribed - (rep.onset.length - nIns),
    notesCutAtStreamFrontier: stats.cutAtFrontier,
  };
  progress("repair", "Evidence repair done", 1, 1);

  // ------------------------------------------------------------------ the grouper
  const gnotes = [];
  for (let i = 0; i < rep.onset.length; i++) gnotes.push(grouperNote(rep.onset[i], rep.offset[i], rep.pitch[i]));
  const own = ownNotes(store.subMaps(), th);
  const nt = GR.assignSlots(gnotes, own);
  const N = nt.length / 4;
  const metaArr = new Float32Array(N * 12);
  const featArr = new Int8Array(N * 3 * 368);
  const fscArr = new Float32Array(N * 3);
  const used = new Map();
  for (let i = 0; i < N; i++) {
    const key = noteKey(nt[4 * i], nt[4 * i + 1], nt[4 * i + 2]);
    const list = feats.get(key);
    const u = used.get(key) || 0;
    if (!list || !list.length) throw new Error(`no grouper features for note ${key}`);
    const f = list[Math.min(u, list.length - 1)];
    used.set(key, u + 1);
    metaArr.set(f.meta, 12 * i);
    metaArr[12 * i + 3] = nt[4 * i + 3]; // the slot from the whole-song assignment
    featArr.set(f.feat, 3 * 368 * i);
    fscArr.set(f.fsc, 3 * i);
  }
  const gcfg = { ...meta.grouper, instruments: meta.instruments };
  const song = GR.makeSong(y, metaArr, featArr, fscArr);
  const pred = await GR.predict(rt, song, grouperFront, gcfg, (d, t) =>
    progress("grouping", `Grouper window ${d} / ${t}`, d, t),
  );
  const on = new Float64Array(N);
  const off = new Float64Array(N);
  const pitch = new Int32Array(N);
  const conf = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    on[i] = metaArr[12 * i];
    off[i] = metaArr[12 * i + 1];
    pitch[i] = roundHalfEven(metaArr[12 * i + 2]);
    conf[i] = metaArr[12 * i + 3] > 0.5 ? metaArr[12 * i + 5] : metaArr[12 * i + 4];
  }
  const groups = N
    ? GR.decode(pred, [on, off, pitch], await trackHead(rt, meta), meta.grouper.decode, meta.instruments.credit)
    : [];
  const onR = on.map((v) => v * tun.r);
  const offR = off.map((v) => v * tun.r);
  const tracks = GR.tracks(onR, offR, pitch, conf, groups, meta.instruments);
  progress("grouping", "Grouping done", 1, 1);
  return {
    tracks,
    info: {
      notes: { ...info, grouped: tracks.reduce((s, t) => s + t.notes.length, 0) },
      tuningCents: Math.round(tun.delta * 100) / 100,
      tuning: tun,
      tapStoreMB: Math.round(tapStoreBytes(cap, fast) / 1e6),
      windows: starts.length,
      repaired: { ...rep, onset: rep.onset.map((v) => v * tun.r), offset: rep.offset.map((v) => v * tun.r) },
      transcriberNotes: tnotes,
    },
    debugMaps: opts.debug ? concatMaps(store.maps(), T) : null,
  };
}

function concatMaps(maps, T) {
  const out = new Float32Array(HEADS.length * T * N_KEYS);
  HEADS.forEach((h, i) => out.set(maps[h].subarray(0, T * N_KEYS), i * T * N_KEYS));
  return out;
}

/** The transcriber's own notes on the sub-average maps, as [on, off, pitch, slot 0/1] rows (assign_slots' input). */
function ownNotes(smaps, th) {
  const tn = transcriberNotes(smaps, th);
  const own = [];
  for (let i = 0; i < tn.onset.length; i++) own.push([tn.onset[i], tn.offset[i], tn.pitch[i], tn.slot[i] - 1]);
  return own;
}

let _trackHead = null;
async function trackHead(rt, meta) {
  if (!_trackHead) _trackHead = GR.loadTrackHead(await rt.asset("track_head.bin"), meta.grouper.track_head_shapes);
  return _trackHead;
}

function lowerBound(arr, v) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] < v) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/**
 * Transcribe mono audio.
 * @param {{ort: object, session: (name: string) => Promise<object>, asset: (name: string) => Promise<ArrayBuffer>}} rt runtime
 * @param {Float32Array} audio mono samples
 * @param {number} rate sample rate of `audio`
 * @param {object} opts { meta, sourceRate, fast, onProgress, onPartial, x16, x44 (optional pre-resampled inputs, for parity) }
 */
export async function transcribe(rt, audio, rate, opts) {
  const meta = opts.meta;
  const progress = makeProgress(opts.onProgress);
  const sourceRate = opts.skipDrums ? null : opts.sourceRate || rate;
  const timings = {};
  let t = performance.now();
  const lap = (k) => {
    const now = performance.now();
    timings[k] = Math.round(now - t) / 1000;
    t = now;
  };
  progress("resample", "Resampling to 16 kHz", 0, 1);
  const x = opts.x16 || (rate === SR ? audio : resample(audio, rate, SR));
  lap("resample16k");
  const p = await pitched(rt, x, meta, opts, progress);
  lap("pitched");
  const tracks = p.tracks.map((g) => ({
    name: g.instrument.replace(/_/g, " "),
    program: g.program,
    isDrum: false,
    notes: g.notes,
  }));
  const info = { ...p.info, sourceRate, fast: !!opts.fast };
  if (sourceRate >= MIN_SOURCE_RATE && !opts.skipDrums) {
    progress("drums", "Drums: features", 0, 1);
    const x44 = opts.x44 || (rate === 44100 ? audio : resample(audio, rate, 44100));
    const f = drumFeatures(x44);
    const d = await transcribeDrums(rt, f.data, f.T, x44.length, meta.drums, (fr) =>
      progress("drums", "Drums", Math.round(fr * 100), 100),
    );
    if (d.hits.length) {
      const notes = d.hits.map((h) => [
        h.onset,
        h.offset,
        h.pitch,
        Math.max(1, Math.min(127, roundHalfEven(h.velocity * 127))),
      ]);
      tracks.push({ name: "Drums", program: 0, isDrum: true, notes });
    }
    info.drums = { hits: d.hits.length, ...d.info };
    lap("drums");
  } else {
    info.drums = opts.skipDrums
      ? "skipped: the source's sample rate is unknown for this format (the drum model needs >= 40 kHz; use wav or mp3)"
      : `skipped: source recorded at ${sourceRate} Hz (the drum model needs >= ${MIN_SOURCE_RATE} Hz)`;
  }
  info.timings = timings;
  progress("done", "Done", 1, 1);
  const out = { seconds: audio.length / rate, sourceRate, tracks, info };
  if (opts.debug) out.debugMaps = p.debugMaps;
  return out;
}
