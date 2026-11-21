// Song X-ray page controller: file in -> decode at the file's own rate -> inference worker -> piano rolls + playback,
// then score-informed stems in a second worker.
//
// URL parameters:
//   ?mock=1             use demo/mock-result.json instead of running inference (synth-only until a file is dropped;
//                       a dropped file is decoded and used for playback + stems with the mock transcription)
//   ?audio=<rel url>    same-origin audio to load at start, as if dropped (e.g. ?mock=1&audio=clips/band.wav)
//   ?fast=1             preset the Fast mode toggle
//   ?webgpu=1           ask the worker to try WebGPU (optional, falls back to WASM)
//
// window.__songXray mirrors the state for automated tests:
//   { status: "idle"|"decoding"|"running"|"done"|"error", result, error, progress, runtime, stems: "pending"|"ready"|"error" }

import { Player } from "./audio/player.js";
import { mixGains } from "./audio/mix.js";
import { sniffSampleRate } from "./audio/sniff.js";
import { writeMidi } from "./midi.js";
import { View } from "./ui/view.js";
import { Lanes } from "./ui/lanes.js";
import { Transport } from "./ui/transport.js";
import { Progress } from "./ui/progress.js";
import { initDropZone } from "./ui/dropzone.js";
import { computePeaks } from "./ui/draw.js";

const params = new URLSearchParams(location.search);
const MOCK = params.get("mock") === "1";
const WEBGPU = params.get("webgpu") === "1";
// Stem output budget: (tracks + 1) x channels x samples x 4 bytes, held twice briefly (worker output + AudioBuffers).
const STEREO_STEMS_MAX_BYTES = 700e6;

const $ = (id) => document.getElementById(id);
const ui = {
  intro: $("intro"),
  results: $("results"),
  error: $("error"),
  errorText: $("error-text"),
  notice: $("notice"),
  summary: $("summary"),
  badge: $("runtime-badge"),
  mockBadge: $("mock-badge"),
  fast: $("fast-mode"),
};

/** Test / automation hook (see header). */
const status = {
  status: "idle",
  result: null,
  error: null,
  progress: null,
  runtime: null,
  stems: "pending",
};
window.__songXray = status;

let player = null;
const view = new View();
const progress = new Progress();
const lanes = new Lanes($("lanes"), view, {
  onSeek: (t) => seek(t),
  onToggle: (id, key) => toggleRow(id, key),
  onVolume: (id, v) => {
    rowState(id).volume = v;
    applyMix();
  },
});
const transport = new Transport(view, {
  onToggle: () => togglePlay(),
  onSeek: (t) => seek(t),
  onMaster: (v) => player?.setMaster(v),
});
const drop = initDropZone((f) => handleFile(f));

let song = null; // { result, buffer, name, mix }
let workers = [];
let runId = 0;

function getPlayer() {
  if (!player) {
    player = new Player();
    player.setMaster(Number($("master").value));
    player.onEnded = () => (lanes.dirty = true);
  }
  return player;
}

/* ---------- runtime badge ---------- */

function setBadge({ backend = "wasm", threads, isolated }) {
  const n = threads ?? 1;
  const name = String(backend).toLowerCase() === "webgpu" ? "WebGPU" : "WASM";
  ui.badge.textContent = `${name} · ${n} thread${n === 1 ? "" : "s"}${isolated ? "" : " (not cross-origin isolated)"}`;
  ui.badge.classList.toggle("warn", !isolated);
}

const hw = navigator.hardwareConcurrency || 4;
setBadge({ backend: "wasm", threads: crossOriginIsolated ? hw : 1, isolated: crossOriginIsolated });
if (params.get("fast") === "1") ui.fast.checked = true;

/* ---------- screens ---------- */

function show(screen) {
  ui.intro.hidden = screen !== "intro";
  ui.results.hidden = screen !== "results";
  ui.error.hidden = screen !== "error";
  if (screen !== "results") progress.compact(false);
}

function fail(err) {
  console.error(err);
  status.status = "error";
  status.error = String(err?.message || err);
  ui.errorText.textContent = status.error;
  progress.hide();
  show("error");
}

$("error-retry").addEventListener("click", () => reset());
$("new-file").addEventListener("click", () => reset());

function reset() {
  runId++;
  for (const w of workers) w.terminate();
  workers = [];
  player?.load({ original: null, tracks: [], duration: 0 });
  song = null;
  drop.reset();
  progress.hide();
  Object.assign(status, { status: "idle", result: null, error: null, progress: null, stems: "pending" });
  show("intro");
}

function setNotice(text) {
  ui.notice.hidden = !text;
  ui.notice.textContent = text || "";
}

/* ---------- decode ---------- */

/**
 * Decode at the file's own sample rate (sniffed from its header) so the browser does not resample.
 * @param {ArrayBuffer} bytes
 * @param {string} name
 */
async function decode(bytes, name) {
  const sniffed = sniffSampleRate(bytes, name);
  const ctx = getPlayer().ctx;
  let buffer = null;
  if (sniffed && sniffed >= 3000 && sniffed <= 768000) {
    try {
      buffer = await new OfflineAudioContext(1, 1, sniffed).decodeAudioData(bytes.slice(0));
    } catch (e) {
      console.warn("decode at the sniffed rate failed, using the context rate", e);
    }
  }
  if (!buffer) buffer = await ctx.decodeAudioData(bytes.slice(0));
  return { buffer, sourceRate: sniffed ?? buffer.sampleRate };
}

/** Channel mean as a fresh array (never a view of the AudioBuffer: it gets transferred to the worker). */
function monoMix(buffer) {
  const n = buffer.length;
  const out = new Float32Array(n);
  const c = buffer.numberOfChannels;
  for (let ch = 0; ch < c; ch++) {
    const d = buffer.getChannelData(ch);
    for (let i = 0; i < n; i++) out[i] += d[i];
  }
  if (c > 1) for (let i = 0; i < n; i++) out[i] /= c;
  return out;
}

/* ---------- inference ---------- */

function runInference(mono, rate, sourceRate, id) {
  return new Promise((resolve, reject) => {
    let w;
    try {
      w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    } catch (e) {
      reject(e);
      return;
    }
    workers.push(w);
    const done = () => {
      w.terminate();
      workers = workers.filter((x) => x !== w);
    };
    w.onerror = (e) => {
      done();
      reject(new Error(`Inference worker failed: ${e.message || "could not load web/js/worker.js"}`));
    };
    w.onmessage = (e) => {
      if (id !== runId) return done();
      const m = e.data || {};
      if (m.type === "progress") {
        status.progress = m;
        progress.message(m);
      } else if (m.type === "partial") {
        if (m.notes) progress.partial(m.notes);
      } else if (m.type === "runtime") {
        status.runtime = m;
        setBadge(m);
      } else if (m.type === "result") {
        done();
        resolve(m.result);
      } else if (m.type === "error") {
        done();
        reject(new Error(m.message || "inference failed"));
      }
    };
    const options = { fast: ui.fast.checked, threads: navigator.hardwareConcurrency || 4 };
    if (WEBGPU) options.webgpu = true;
    w.postMessage({ type: "run", audio: mono, rate, sourceRate, options }, [mono.buffer]);
  });
}

let mockResult = null;
async function loadMock() {
  if (!mockResult) {
    const res = await fetch(new URL("../demo/mock-result.json", import.meta.url));
    if (!res.ok) throw new Error(`could not load demo/mock-result.json (HTTP ${res.status})`);
    mockResult = await res.json();
  }
  return structuredClone(mockResult);
}

/* ---------- file flow ---------- */

async function handleFile(file) {
  if (status.status === "decoding" || status.status === "running") return;
  const id = ++runId;
  try {
    status.status = "decoding";
    status.error = null;
    status.result = null;
    status.stems = "pending";
    show("none");
    progress.start(`${file.name} · ${(file.size / 1e6).toFixed(1)} MB`, 0);
    progress.stage("decode", { label: "Decoding audio" });
    const bytes = await file.arrayBuffer();
    const { buffer, sourceRate } = await decode(bytes, file.name);
    if (id !== runId) return;
    progress.duration = buffer.duration;
    progress.mark(
      "decode",
      "done",
      `${(buffer.sampleRate / 1000).toFixed(1)} kHz ${buffer.numberOfChannels > 1 ? "stereo" : "mono"}`,
    );
    let result;
    if (MOCK) {
      for (const k of ["models", "tuning", "transcriber", "repair", "grouping", "drums"])
        progress.mark(k, "skipped", "mock");
      result = await loadMock();
    } else {
      status.status = "running";
      progress.stage("models", { label: "Loading models" });
      result = await runInference(monoMix(buffer), buffer.sampleRate, sourceRate, id);
    }
    if (id !== runId) return;
    if (typeof result.info?.drums === "string") progress.mark("drums", "skipped", "source < 40 kHz");
    status.result = result;
    status.status = "done";
    showResults(result, buffer, file.name, sourceRate);
    startStems(buffer, result, id);
  } catch (err) {
    if (id === runId) fail(err);
  }
}

/* ---------- results ---------- */

function rowState(id) {
  if (id === "original") return song.mix.original;
  if (id === "residual") return song.mix.residual;
  return song.mix.tracks[id];
}

function showResults(result, buffer, name, fileRate) {
  const tracks = result.tracks ?? [];
  const duration =
    buffer?.duration ?? result.seconds ?? Math.max(1, ...tracks.flatMap((t) => t.notes.map((n) => n[1])));
  const synthOnly = !buffer;
  song = {
    result,
    buffer,
    name,
    duration,
    mix: {
      original: { muted: synthOnly },
      residual: {},
      tracks: tracks.map(() => ({ synth: synthOnly })),
      stemsReady: false,
    },
  };
  const p = getPlayer();
  p.load({ original: buffer, tracks, duration });
  show("results");
  view.setDuration(duration);
  lanes.build({ tracks, hasAudio: !!buffer, sourceRate: fileRate ?? result.sourceRate });
  let overview;
  if (buffer) {
    const chans = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
    const peaks = computePeaks(chans, buffer.sampleRate);
    lanes.setPeaks("original", peaks);
    lanes.setPeaks("residual", null);
    lanes.setResidualMessage("Computing stems…");
    lanes.setRowStatus("residual", "computing stems", true);
    overview = overviewFromPeaks(peaks);
  } else {
    overview = overviewFromNotes(tracks, duration);
  }
  transport.setSong(duration, overview);
  const pitched = tracks.filter((t) => !t.isDrum);
  const nNotes = pitched.reduce((s, t) => s + t.notes.length, 0);
  const hits = tracks.filter((t) => t.isDrum).reduce((s, t) => s + t.notes.length, 0);
  const secs = result.info?.timings ? Object.values(result.info.timings).reduce((a, b) => a + (Number(b) || 0), 0) : 0;
  ui.summary.innerHTML = "";
  ui.summary.append(
    ...[
      [`${pitched.length}`, ` instrument${pitched.length === 1 ? "" : "s"}`],
      [` · ${nNotes}`, " notes"],
      hits ? [` · ${hits}`, " drum hits"] : null,
      [` · ${fmtDur(duration)}`, ""],
      secs && !MOCK ? [` · transcribed in ${fmtDur(secs)}`, ""] : null,
    ]
      .filter(Boolean)
      .map(([b, rest]) => {
        const s = document.createElement("span");
        const bold = document.createElement("b");
        bold.textContent = b;
        s.append(bold, rest);
        return s;
      }),
  );
  if (synthOnly) {
    setNotice(
      "Demo transcription (mock), synth-only playback. Drop clips/band.wav on the page to hear it with the original audio and stems.",
    );
  } else {
    setNotice(
      "Computing stems… until they are ready you hear the original mix; Solo or Synth on a track plays its synth.",
    );
  }
  applyMix();
  lanes.dirty = true;
}

function fmtDur(s) {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  return `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")} s`;
}

function overviewFromPeaks(peaks, cols = 600) {
  const out = new Float32Array(cols);
  const per = peaks.max.length / cols;
  let top = 1e-6;
  for (let c = 0; c < cols; c++) {
    let m = 0;
    for (let i = Math.floor(c * per); i < Math.min(peaks.max.length, Math.floor((c + 1) * per) + 1); i++)
      m = Math.max(m, peaks.max[i], -peaks.min[i]);
    out[c] = m;
    top = Math.max(top, m);
  }
  for (let c = 0; c < cols; c++) out[c] /= top;
  return out;
}

function overviewFromNotes(tracks, duration, cols = 600) {
  const out = new Float32Array(cols);
  for (const t of tracks)
    for (const n of t.notes) {
      const a = Math.floor((n[0] / duration) * cols);
      const b = Math.min(cols - 1, Math.floor((n[1] / duration) * cols));
      for (let c = Math.max(0, a); c <= b; c++) out[c] += 1;
    }
  const top = Math.max(1, ...out);
  for (let c = 0; c < cols; c++) out[c] = Math.sqrt(out[c] / top);
  return out;
}

function toggleRow(id, key) {
  if (!song) return;
  const r = rowState(id);
  r[key] = !r[key];
  applyMix();
}

function applyMix() {
  if (!song || !player) return;
  const g = mixGains(song.mix);
  player.setGains(g);
  transport.setMode(g.mode);
  const recorded = g.original > 0 || g.residual > 0 || g.stems.some((x) => x > 0);
  lanes.setRowState("original", { ...song.mix.original, audible: recorded });
  lanes.setRowState("residual", { ...song.mix.residual, audible: g.residual > 0 });
  song.mix.tracks.forEach((t, k) => lanes.setRowState(k, { ...t, audible: g.stems[k] > 0 || g.synths[k] > 0 }));
}

function seek(t) {
  if (!player || !song) return;
  player.seek(t);
  lanes.dirty = true;
}

function togglePlay() {
  if (!player || !song) return;
  player.toggle();
  lanes.dirty = true;
}

/* ---------- stems ---------- */

function startStems(buffer, result, id) {
  if (!buffer) return;
  const tracks = result.tracks ?? [];
  const n = buffer.length;
  const K = tracks.length;
  const stereo = buffer.numberOfChannels >= 2 && (K + 1) * 2 * n * 4 <= STEREO_STEMS_MAX_BYTES;
  const channels = stereo
    ? [buffer.getChannelData(0).slice(), buffer.getChannelData(1).slice()]
    : [buffer.numberOfChannels === 1 ? buffer.getChannelData(0).slice() : monoMix(buffer)];
  const t0 = performance.now();
  status.stems = "pending";
  progress.compact(true);
  progress.stage("stems", { label: "Separating stems (score-informed masks)…" });
  const w = new Worker(new URL("./stems-worker.js", import.meta.url), { type: "module" });
  workers.push(w);
  const done = () => {
    w.terminate();
    workers = workers.filter((x) => x !== w);
  };
  const stemsFailed = (msg) => {
    done();
    status.stems = "error";
    console.error("stems failed:", msg);
    lanes.setRowStatus("residual", "stems unavailable");
    lanes.setResidualMessage("Stems failed; the original mix and synths still work.");
    setNotice(
      `Stems could not be computed (${msg}). Playback uses the original mix; Solo or Synth plays a track's synth.`,
    );
    progress.mark("stems", "skipped", "failed");
    progress.finish("Transcription done (stems failed)");
  };
  w.onerror = (e) => stemsFailed(e.message || "worker error");
  w.onmessage = (e) => {
    if (id !== runId) return done();
    const m = e.data || {};
    if (m.type === "progress") {
      const pct = Math.round((100 * m.done) / m.total);
      progress.stage("stems", {
        label: `Separating stems: frame ${m.done} / ${m.total}`,
        done: m.done,
        total: m.total,
      });
      lanes.setRowStatus("residual", `computing stems ${pct}%`, true);
    } else if (m.type === "error") {
      stemsFailed(m.message);
    } else if (m.type === "result") {
      done();
      installStems(m, buffer.sampleRate);
      const s = (performance.now() - t0) / 1000;
      progress.mark("stems", "done", `${s.toFixed(1)} s`);
      progress.finish(
        `Done · ${K} track${K === 1 ? "" : "s"} · stems ${stereo ? "stereo" : "mono"} in ${s.toFixed(1)} s`,
      );
    }
  };
  w.postMessage(
    { type: "run", channels, rate: buffer.sampleRate, tracks, tuningCents: result.info?.tuningCents || 0 },
    channels.map((c) => c.buffer),
  );
}

function installStems(m, rate) {
  const ctx = getPlayer().ctx;
  const toBuffer = (chs) => {
    const b = ctx.createBuffer(chs.length, chs[0].length, rate);
    chs.forEach((c, i) => b.copyToChannel(c, i));
    return b;
  };
  const stems = m.stems.map(toBuffer);
  const residual = toBuffer(m.residual);
  player.setStems(stems, residual);
  song.mix.stemsReady = true;
  lanes.setPeaks("residual", computePeaks(m.residual, rate));
  lanes.setRowStatus("residual", "");
  status.stems = "ready";
  setNotice("");
  applyMix();
}

/* ---------- exports ---------- */

function download(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const baseName = () => (song?.name || "song-xray").replace(/\.[^.]+$/, "") || "song-xray";
$("export-midi").addEventListener("click", () => {
  if (!song) return;
  const bytes = writeMidi(song.result, { title: baseName() });
  download(new Blob([bytes], { type: "audio/midi" }), `${baseName()}.mid`);
});
$("export-json").addEventListener("click", () => {
  if (!song) return;
  download(new Blob([JSON.stringify(song.result)], { type: "application/json" }), `${baseName()}.json`);
});

/* ---------- keyboard ---------- */

const typing = (t) =>
  t instanceof HTMLElement &&
  (t.isContentEditable ||
    /^(TEXTAREA|SELECT)$/.test(t.tagName) ||
    (t.tagName === "INPUT" && !["range", "checkbox", "button"].includes(t.type)));
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || typing(e.target) || !song || ui.results.hidden) return;
  e.preventDefault();
  if (!e.repeat) togglePlay();
});
// Keep space from also "clicking" a focused button or toggling a focused checkbox.
window.addEventListener(
  "keyup",
  (e) => {
    if (e.code === "Space" && !typing(e.target) && song && !ui.results.hidden) e.preventDefault();
  },
  true,
);

/* ---------- render loop ---------- */

let lastTime = -1;
function frame() {
  if (song && !ui.results.hidden && player) {
    const t = player.time;
    if (player.playing) view.follow(t);
    if (lanes.dirty || t !== lastTime || player.playing) {
      lanes.render(t, player.playing);
      transport.render(t, player.playing);
      lastTime = t;
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ---------- start ---------- */

async function boot() {
  if (MOCK) ui.mockBadge.hidden = false;
  const audioParam = params.get("audio");
  if (audioParam) {
    const url = new URL(audioParam, location.href);
    if (url.origin !== location.origin) throw new Error("?audio= must be a same-origin URL");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`could not load ${audioParam} (HTTP ${res.status})`);
    const blob = await res.blob();
    await handleFile(new File([blob], url.pathname.split("/").pop() || "audio", { type: blob.type }));
    return;
  }
  if (MOCK) {
    const result = await loadMock();
    status.result = result;
    status.status = "done";
    showResults(result, null, "mock");
  }
}
boot().catch(fail);
