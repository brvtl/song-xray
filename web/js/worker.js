// Inference worker: loads the ONNX models from ../models/ (same origin, no network beyond the page's own server) and runs the
// pipeline off the main thread so the page stays responsive.
//
// in:  { type: "run", audio: Float32Array (mono), rate, sourceRate, options: { fast, threads, webgpu } }
// out: { type: "progress", stage, label, done, total, etaS, elapsedS } | { type: "partial", notes } |
//      { type: "runtime", backend, threads, isolated } | { type: "result", result } | { type: "error", message }

import * as ortWasm from "../vendor/ort/ort.wasm.min.mjs";
import { transcribe } from "./pipeline.js";

const MODELS = new URL("../models/", import.meta.url);
const ORT_DIR = new URL("../vendor/ort/", import.meta.url).href;

let meta = null;
const sessions = new Map();

/** Fetch a file from ../models/ with byte-level progress. */
async function fetchBytes(name, onBytes) {
  const res = await fetch(new URL(name, MODELS));
  if (!res.ok) throw new Error(`could not load ${name}: HTTP ${res.status} (did you run scripts/setup.sh?)`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onBytes?.(got, total);
  }
  return out;
}

async function makeRuntime(options, post) {
  let ort = ortWasm;
  let eps = ["wasm"];
  if (options.webgpu && self.navigator?.gpu) {
    // Optional and untested here (this project was built and verified on CPU only). Falls back to WASM per operator.
    ort = await import("../vendor/ort/ort.webgpu.min.mjs");
    eps = ["webgpu", "wasm"];
  }
  const isolated = !!self.crossOriginIsolated;
  const hw = self.navigator?.hardwareConcurrency || 4;
  const threads = isolated ? Math.max(1, Math.min(options.threads || hw, 16)) : 1;
  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = threads;
  post({ type: "runtime", backend: eps[0], threads, isolated });
  return {
    ort,
    async session(name) {
      if (!sessions.has(name)) {
        const bytes = await fetchBytes(`${name}.onnx`, (g, t) =>
          post({ type: "progress", stage: "models", label: `Loading ${name} model`, done: g, total: t }),
        );
        sessions.set(name, await ort.InferenceSession.create(bytes, { executionProviders: eps }));
      }
      return sessions.get(name);
    },
    async asset(name) {
      return (await fetchBytes(name)).buffer;
    },
  };
}

let lastPartial = 0;

self.onmessage = async (e) => {
  const m = e.data;
  if (!m || m.type !== "run") return;
  const post = (msg) => self.postMessage(msg);
  try {
    if (!meta) meta = JSON.parse(new TextDecoder().decode(await fetchBytes("meta.json")));
    const rt = await makeRuntime(m.options || {}, post);
    const result = await transcribe(rt, m.audio, m.rate, {
      meta,
      sourceRate: m.sourceRate || m.rate,
      fast: !!m.options?.fast,
      onProgress: (p) => post({ type: "progress", ...p }),
      onPartial: (p) => {
        const now = performance.now();
        if (now - lastPartial < 400) return;
        lastPartial = now;
        post({ type: "partial", notes: p.notes });
      },
    });
    if (!m.options?.debug) {
      delete result.info.repaired;
      delete result.info.transcriberNotes;
      delete result.info.tuning;
    }
    post({ type: "result", result });
  } catch (err) {
    post({ type: "error", message: String(err?.stack || err?.message || err) });
  }
};
