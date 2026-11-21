// Drum stage: the JavaScript port of vendor/dettato/src/dettato/drums/infer.py (+ model.py forward passes) on the split ONNX graphs.
// Reference: DrumTranscriber.probabilities / decode / transcribe, DrumNet.embed_long, OrchestralExpert.forward.
// All arrays are row-major typed arrays; [T][K] means a Float32Array of T*K values. No DOM, no onnxruntime import (rt provides it).

const N_MEL = 160;
const N_FEAT = 280;
const N_CQ = N_FEAT - N_MEL;
const N_KEYS = 61;
const N_TIMPANI = 13;
const SR = 44100;
const FPS = 50;
const N_EXPERTS = 3;
const LONG_KIT = 9000; // DrumNet.forward long_above (frames at 100 fps)
const CHUNK = 8000; // embed_long window (frames)
const MARGIN = 200; // embed_long context each side
const LONG_EXPERT = CHUNK + 2 * MARGIN; // OrchestralExpert.forward threshold (8400)
const ATTN_BUDGET = 1 << 25; // max H * rows * L score entries per attn.onnx call (128 MB of float32)
const ATTN_MAX_ROWS = 1024;
const ATTN_MIN_ROWS = 64;

/** @typedef {{ort: any, session: (name: string) => Promise<any>}} Runtime */
/**
 * @typedef {object} DrumProbs
 * @property {number} router mean over the experts of sigmoid(router logit)
 * @property {Float32Array} kit [T2][61]
 * @property {Float32Array} kitVel [T2][61]
 * @property {Float32Array} drum [n][61] orchestral drum probabilities (averaged experts)
 * @property {Float32Array} timpani [n][13]
 * @property {Float32Array} vel [n][61]
 */

const sigmoid32 = (x) => Math.fround(1 / (1 + Math.exp(-x)));

/** Run one graph; `feeds` maps input name -> [Float32Array, dims]. Returns output name -> ort Tensor. */
async function run(rt, name, feeds) {
  const session = await rt.session(name);
  const inputs = {};
  for (const [k, [data, dims]] of Object.entries(feeds)) inputs[k] = new rt.ort.Tensor("float32", data, dims);
  return session.run(inputs);
}

/** Split [T][280] features into the mel part [T][160] and the cq part [T][120]. */
export function splitFeatures(feats, T) {
  const mel = new Float32Array(T * N_MEL);
  const cq = new Float32Array(T * N_CQ);
  for (let t = 0; t < T; t++) {
    const row = t * N_FEAT;
    mel.set(feats.subarray(row, row + N_MEL), t * N_MEL);
    cq.set(feats.subarray(row + N_MEL, row + N_FEAT), t * N_CQ);
  }
  return { mel, cq };
}

/**
 * DrumNet.embed on a (possibly long) input: windows of `chunk` frames with `margin` frames of context each side, as
 * DrumNet.embed_long. Returns [T/2][88]. Used with `longAbove` = 9000 for the kit net and 8400 for the experts' bases.
 * @param {Runtime} rt
 * @param {string} model e.g. "drums/kit_embed"
 * @param {Float32Array} mel [T][160]
 * @param {number} T even frame count
 * @param {number} longAbove use the chunked path when T > longAbove
 * @param {number} [chunk]
 * @param {number} [margin]
 * @returns {Promise<Float32Array>}
 */
export async function embedLong(rt, model, mel, T, longAbove, chunk = CHUNK, margin = MARGIN) {
  const W = 88;
  if (T <= longAbove) return (await run(rt, model, { x: [mel, [1, T, N_MEL]] })).c.data;
  const out = new Float32Array((T / 2) * W);
  let filled = 0;
  for (let s = 0; s < T; s += chunk) {
    const a = Math.max(0, s - margin);
    const b = Math.min(T, s + chunk + margin);
    const h = (await run(rt, model, { x: [mel.subarray(a * N_MEL, b * N_MEL), [1, b - a, N_MEL]] })).c.data;
    const lo = (s - a) / 2;
    const n = (Math.min(s + chunk, T) - s) / 2;
    out.set(h.subarray(lo * W, (lo + n) * W), filled * W);
    filled += n;
  }
  return out;
}

/** Copy rows [r0, r1) of every head of a [H][L][D] array into a new [H][r1-r0][D] array. */
function headRows(x, H, L, D, r0, r1) {
  const n = r1 - r0;
  const out = new Float32Array(H * n * D);
  for (let h = 0; h < H; h++) out.set(x.subarray((h * L + r0) * D, (h * L + r1) * D), h * n * D);
  return out;
}

/**
 * softmax(q k^T / sqrt(D)) v over the whole sequence, computed on query chunks with the full k, v (bounded memory); the chunks
 * are concatenated along the query axis. Each query row is independent, so this equals the unchunked attention exactly.
 * @param {Runtime} rt
 * @param {Float32Array} q [H][L][D]
 * @param {Float32Array} k [H][L][D]
 * @param {Float32Array} v [H][L][D]
 * @param {number} H
 * @param {number} L
 * @param {number} D
 * @returns {Promise<Float32Array>} [H][L][D]
 */
export async function attention(rt, q, k, v, H, L, D) {
  const rows = Math.max(ATTN_MIN_ROWS, Math.min(ATTN_MAX_ROWS, Math.floor(ATTN_BUDGET / (H * L))));
  const out = new Float32Array(H * L * D);
  for (let r0 = 0; r0 < L; r0 += rows) {
    const r1 = Math.min(L, r0 + rows);
    const n = r1 - r0;
    const o = (
      await run(rt, "attn", {
        q: [headRows(q, H, L, D, r0, r1), [1, H, n, D]],
        k: [k, [1, H, L, D]],
        v: [v, [1, H, L, D]],
      })
    ).o.data;
    for (let h = 0; h < H; h++) out.set(o.subarray(h * n * D, (h + 1) * n * D), (h * L + r0) * D);
  }
  return out;
}

/** Attention between a "...A" graph's q/k/v outputs, in query chunks -> {att, H, D}. */
async function attend(rt, a, L) {
  const [, H, , D] = a.q.dims;
  return { att: await attention(rt, a.q.data, a.k.data, a.v.data, H, L, D), H, D };
}

/** DrumNet.trunk on frame embeddings c [T2][88] (attention in query chunks) -> {h, z, v} (trunk state, onset / velocity logits). */
async function trunk(rt, prefix, c, T2) {
  const a = await run(rt, `drums/${prefix}_trunkA`, { c: [c, [1, T2, 88]] });
  const { att, H, D } = await attend(rt, a, T2);
  const b = await run(rt, `drums/${prefix}_trunkB`, { h: [a.h.data, [1, T2, 88]], a: [att, [1, H, T2, D]] });
  return { h: b.h2.data, z: b.z.data, v: b.v.data };
}

/** The kit net: DrumNet.forward -> sigmoid onset / velocity probabilities [T/2][61]. */
async function kitProbabilities(rt, mel, T) {
  const c = await embedLong(rt, "drums/kit_embed", mel, T, LONG_KIT);
  const { z, v } = await trunk(rt, "kit", c, T / 2);
  return { kit: z.map(sigmoid32), kitVel: v.map(sigmoid32) };
}

/** Rows [0, n) of a [rows][width] array (the array itself when nothing is trimmed). */
const trimRows = (x, width, n) => (x.length === n * width ? x : x.subarray(0, n * width));

/**
 * OrchestralExpert.forward for expert i: base DrumNet (embed_long above 8400 frames) -> cq branch -> adapter.
 * @returns {Promise<{router: number, drum: Float32Array, timpani: Float32Array, vel: Float32Array, n: number}>}
 *   router/drum/timpani/vel as probabilities (drum = sigmoid(base + (drum - base)), vel = sigmoid of the base's velocity logits).
 */
async function expertProbabilities(rt, i, mel, cqIn, T) {
  const c = await embedLong(rt, `drums/ex${i}_embed`, mel, T, LONG_EXPERT);
  const base = await trunk(rt, `ex${i}`, c, T / 2);
  const cq = (await run(rt, `drums/ex${i}_cq`, { xc: [cqIn, [1, T, N_CQ]] })).cq.data;
  const n = Math.min(base.h.length / 88, cq.length / 128);
  const z0 = trimRows(base.z, N_KEYS, n);
  const a = await run(rt, `drums/ex${i}_adaptA`, {
    h0: [trimRows(base.h, 88, n), [1, n, 88]],
    z0: [z0, [1, n, N_KEYS]],
    cq: [trimRows(cq, 128, n), [1, n, 128]],
  });
  const { att, H, D } = await attend(rt, a, n);
  const b = await run(rt, `drums/ex${i}_adaptB`, {
    r: [a.r.data, [1, n, 2 * 128]],
    att: [att, [1, H, n, D]],
    z0: [z0, [1, n, N_KEYS]],
  });
  const drum = b.drum.data.map((d, j) => sigmoid32(Math.fround(z0[j] + Math.fround(d - z0[j]))));
  return {
    router: sigmoid32(b.router.data[0]),
    drum,
    timpani: b.timpani.data.map(sigmoid32),
    vel: trimRows(base.v, N_KEYS, n).map(sigmoid32),
    n,
  };
}

/** Mean of equally long float32 arrays, rounded to float32 (np.mean over the experts). */
function meanOf(arrays, n) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (const a of arrays) s += a[j];
    out[j] = s / arrays.length;
  }
  return out;
}

/**
 * Features -> probabilities, as DrumTranscriber.probabilities: T = floor(T/2)*2 frames, the kit net, then all three experts
 * (run sequentially to bound memory); the experts' outputs are averaged over their common n frames.
 * @param {Runtime} rt
 * @param {Float32Array} feats [T][280] (mel 160 | cq 120), as dsp.drumFeatures
 * @param {number} T number of feature rows
 * @param {(fraction: number) => void} [onProgress] called after each of the four networks
 * @returns {Promise<DrumProbs>}
 */
export async function drumProbabilities(rt, feats, T, onProgress) {
  const Te = Math.floor(T / 2) * 2;
  const { mel, cq } = splitFeatures(feats, Te);
  const { kit, kitVel } = await kitProbabilities(rt, mel, Te);
  onProgress?.(1 / (1 + N_EXPERTS));
  const outs = [];
  for (let i = 0; i < N_EXPERTS; i++) {
    outs.push(await expertProbabilities(rt, i, mel, cq, Te));
    onProgress?.((2 + i) / (1 + N_EXPERTS));
  }
  const n = Math.min(...outs.map((o) => o.n));
  const col = (key, w) =>
    meanOf(
      outs.map((o) => o[key].subarray(0, n * w)),
      n * w,
    );
  const router = outs.reduce((s, o) => s + o.router, 0) / outs.length;
  return {
    router,
    kit,
    kitVel,
    drum: col("drum", N_KEYS),
    timpani: col("timpani", N_TIMPANI),
    vel: col("vel", N_KEYS),
  };
}

/**
 * infer.peaks: local maxima >= th per lane (the edge row is repeated as its own neighbour, `>=` comparisons), at least `gap`
 * frames apart per lane. Returns frames / lanes ordered by frame, then lane.
 * @param {Float32Array} prob [T][L]
 * @param {number} L
 * @param {number} th
 * @param {number} [gap]
 * @returns {{frames: number[], lanes: number[]}}
 */
export function peaks(prob, L, th, gap = 3) {
  const T = prob.length / L;
  const thr = Math.fround(th);
  const last = new Int32Array(L).fill(-10);
  const frames = [];
  const lanes = [];
  for (let t = 0; t < T; t++) {
    const lo = Math.max(t - 1, 0) * L;
    const hi = Math.min(t + 1, T - 1) * L;
    for (let l = 0; l < L; l++) {
      const p = prob[t * L + l];
      if (!(p >= thr && p >= prob[lo + l] && p >= prob[hi + l] && t - last[l] > gap - 1)) continue;
      frames.push(t);
      lanes.push(l);
      last[l] = t;
    }
  }
  return { frames, lanes };
}

/** Timpani hits that do not repeat a key already hit within `repeatS` seconds; appended to the lists in place. */
function addTimpani(P, cfg, hits, repeatS) {
  const seen = new Map();
  for (const h of hits) (seen.get(h.key) ?? seen.set(h.key, []).get(h.key)).push(h.onset);
  const { frames, lanes } = peaks(P.timpani, N_TIMPANI, cfg.thresholds.timpani, cfg.decode?.peak_gap_frames ?? 3);
  const lo = Math.fround(0.3);
  frames.forEach((f, j) => {
    const onset = f / FPS;
    const key = cfg.timpani_keys[lanes[j]];
    if ((seen.get(key) ?? []).some((q) => Math.abs(onset - q) <= repeatS)) return;
    const p = P.timpani[f * N_TIMPANI + lanes[j]];
    hits.push({ onset, key, vel: Math.min(Math.max(p, lo), 1.0) });
  });
}

/**
 * DrumTranscriber.decode: router < threshold -> kit model peaks, else orchestral drum peaks + timpani peaks; sorted by onset
 * (stable).
 * @param {DrumProbs} P
 * @param {{keys: number[], timpani_keys: number[], thresholds: {drum_kit: number, orchestral_drum: number, timpani: number, router: number}, decode?: object}} cfg meta.json `drums`
 * @returns {{onsets: Float64Array, keys: Int32Array, vels: Float64Array, model: "drum-kit" | "orchestral"}}
 */
export function drumDecode(P, cfg) {
  const gap = cfg.decode?.peak_gap_frames ?? 3;
  const th = cfg.thresholds;
  const model = P.router < th.router ? "drum-kit" : "orchestral";
  const [prob, vel, thr] = model === "drum-kit" ? [P.kit, P.kitVel, th.drum_kit] : [P.drum, P.vel, th.orchestral_drum];
  const { frames, lanes } = peaks(prob, N_KEYS, thr, gap);
  const hits = frames.map((f, j) => ({ onset: f / FPS, key: cfg.keys[lanes[j]], vel: vel[f * N_KEYS + lanes[j]] }));
  if (model === "orchestral") addTimpani(P, cfg, hits, cfg.decode?.timpani_repeat_s ?? 0.04);
  const order = hits.map((_, i) => i).sort((a, b) => hits[a].onset - hits[b].onset || a - b);
  const sorted = order.map((i) => hits[i]);
  return {
    onsets: Float64Array.from(sorted, (h) => h.onset),
    keys: Int32Array.from(sorted, (h) => h.key),
    vels: Float64Array.from(sorted, (h) => h.vel),
    model,
  };
}

/**
 * Python's round(x, digits) for a double: the decimal nearest the exact binary value, exact ties to even. Number.toFixed picks
 * the larger candidate on an exact tie (0.03125 -> "0.0313", Python 0.0312), so the exact expansion (toFixed(100) is exact for
 * |x| >= 2^-47) is inspected for a tie and the even neighbour is chosen then; otherwise toFixed(digits) is already correct.
 * @param {number} x
 * @param {number} digits
 * @returns {number}
 */
export function pyRound(x, digits) {
  const s = Math.abs(x).toFixed(100);
  const dot = s.indexOf(".");
  const tail = s.slice(dot + 1 + digits);
  const isTie = /^50*$/.test(tail);
  const fixed = Math.abs(x).toFixed(digits);
  let r = fixed;
  if (isTie) {
    const truncated = s.slice(0, dot + 1 + digits).replace(/\.$/, "");
    const lastDigit = Number(truncated[truncated.length - 1]);
    r = lastDigit % 2 === 0 ? truncated : fixed;
  }
  return Math.sign(x) * Number(r) || 0;
}

/** Hits in the transcribe() output format: onset round 4, offset onset + 0.1 round 4, velocity clipped to [0.05, 1] round 3. */
function formatHits(D, hitLength) {
  return Array.from(D.onsets, (on, i) => ({
    onset: pyRound(on, 4),
    offset: pyRound(on + hitLength, 4),
    pitch: D.keys[i],
    isDrum: true,
    velocity: pyRound(Math.min(Math.max(D.vels[i], 0.05), 1.0), 3),
  }));
}

/**
 * DrumTranscriber.transcribe from features: inputs shorter than SR // 10 samples (at 44.1 kHz) give no hits.
 * @param {Runtime} rt
 * @param {Float32Array} feats [T][280]
 * @param {number} T number of feature rows
 * @param {number} nSamples44 length of the 44.1 kHz audio the features came from
 * @param {object} cfg meta.json `drums`
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<{hits: {onset: number, offset: number, pitch: number, isDrum: true, velocity: number}[], info: {router: number | null, model: string | null}}>}
 */
export async function transcribeDrums(rt, feats, T, nSamples44, cfg, onProgress) {
  if (nSamples44 < Math.floor(SR / 10)) return { hits: [], info: { router: null, model: null } };
  const P = await drumProbabilities(rt, feats, T, onProgress);
  const D = drumDecode(P, cfg);
  return { hits: formatHits(D, cfg.decode?.hit_length_s ?? 0.1), info: { router: P.router, model: D.model } };
}
