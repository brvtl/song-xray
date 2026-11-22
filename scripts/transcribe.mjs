// Run the browser pipeline (web/js/pipeline.js, the same modules the page loads) under Node with onnxruntime-web's WASM backend.
//
//   node scripts/transcribe.mjs clips/band.wav [--out result.json] [--fast] [--threads 8] [--same-samples fixtures/band]
//
// --same-samples DIR feeds the pipeline the reference's own 16 kHz / 44.1 kHz samples (DIR/x16.npy, DIR/x44.npy from
// scripts/dump_fixtures.py) instead of resampling here, so a comparison isolates the models + JS glue from the resampler.
// WAV is parsed here; other formats are decoded with ffmpeg at the file's own rate (like the reference's ffmpeg fallback). The
// page decodes with the browser's decodeAudioData instead (see parity/parity.py --browser for that path).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { transcribe } from "../web/js/pipeline.js";
import { parseNpy } from "../tests/fixtures.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const a = {
    file: null,
    out: null,
    fast: false,
    threads: Math.min(8, os.cpus().length),
    same: null,
    quiet: false,
    debugDir: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const v = argv[i];
    if (v === "--out") a.out = argv[++i];
    else if (v === "--fast") a.fast = true;
    else if (v === "--threads") a.threads = Number(argv[++i]);
    else if (v === "--same-samples") a.same = argv[++i];
    else if (v === "--quiet") a.quiet = true;
    else if (v === "--debug-dir") a.debugDir = argv[++i];
    else a.file = v;
  }
  if (!a.file)
    throw new Error("usage: transcribe.mjs <audio> [--out f.json] [--fast] [--threads n] [--same-samples dir]");
  return a;
}

/** Minimal RIFF/WAVE reader: PCM 8/16/24/32-bit and IEEE float 32/64 -> mono (channel mean) Float32Array + rate. */
export function readWav(buf) {
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE")
    throw new Error("not a WAV file");
  let off = 12;
  let fmt = null;
  while (off + 8 <= buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    const body = off + 8;
    if (id === "fmt ") {
      let format = buf.readUInt16LE(body);
      if (format === 0xfffe) format = buf.readUInt16LE(body + 24); // WAVE_FORMAT_EXTENSIBLE: sub-format GUID
      fmt = {
        format,
        ch: buf.readUInt16LE(body + 2),
        rate: buf.readUInt32LE(body + 4),
        bits: buf.readUInt16LE(body + 14),
      };
    } else if (id === "data") {
      if (!fmt) throw new Error("WAV data before fmt");
      const n = Math.floor(Math.min(size, buf.length - body) / (fmt.bits / 8) / fmt.ch);
      const out = new Float32Array(n);
      const bps = fmt.bits / 8;
      const read = sampleReader(buf, fmt);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < fmt.ch; c++) s += read(body + (i * fmt.ch + c) * bps);
        out[i] = s / fmt.ch;
      }
      return { samples: out, rate: fmt.rate };
    }
    off = body + size + (size & 1);
  }
  throw new Error("WAV without data chunk");
}

function sampleReader(buf, fmt) {
  if (fmt.format === 3 && fmt.bits === 32) return (o) => buf.readFloatLE(o);
  if (fmt.format === 3 && fmt.bits === 64) return (o) => buf.readDoubleLE(o);
  if (fmt.bits === 16) return (o) => buf.readInt16LE(o) / 32768;
  if (fmt.bits === 24) return (o) => buf.readIntLE(o, 3) / 8388608;
  if (fmt.bits === 32) return (o) => buf.readInt32LE(o) / 2147483648;
  if (fmt.bits === 8) return (o) => (buf.readUInt8(o) - 128) / 128;
  throw new Error(`unsupported WAV format ${fmt.format}/${fmt.bits}`);
}

function readWithFfmpeg(file) {
  const probe = String(
    execFileSync("ffprobe", [
      "-v",
      "error",
      "-select_streams",
      "a:0",
      "-show_entries",
      "stream=sample_rate,channels",
      "-of",
      "csv=p=0",
      file,
    ]),
  )
    .trim()
    .split(",");
  const rate = parseInt(probe[0], 10) || 44100;
  const ch = parseInt(probe[1], 10) || 1;
  // decode every channel and average them here: ffmpeg's own "-ac 1" downmix is not the plain channel mean that the reference
  // (soundfile, x.mean(1)) and the page (mean of the decoded channels) use
  const raw = execFileSync(
    "ffmpeg",
    ["-v", "error", "-i", file, "-f", "f32le", "-ac", String(ch), "-ar", String(rate), "-"],
    {
      maxBuffer: 1 << 31,
    },
  );
  const x = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  const n = Math.floor(x.length / ch);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < ch; c++) s += x[i * ch + c];
    out[i] = s / ch;
  }
  return { samples: out, rate };
}

export function loadAudio(file) {
  const buf = fs.readFileSync(file);
  if (buf.toString("ascii", 0, 4) === "RIFF") return readWav(buf);
  return readWithFfmpeg(file);
}

/** The runtime handle the pipeline expects: onnxruntime-web sessions + raw assets from web/models/. */
export function nodeRuntime(threads) {
  ort.env.wasm.numThreads = threads;
  const cache = new Map();
  return {
    ort,
    async session(name) {
      if (!cache.has(name)) {
        const bytes = fs.readFileSync(path.join(ROOT, "web", "models", `${name}.onnx`));
        cache.set(name, await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }));
      }
      return cache.get(name);
    },
    async asset(name) {
      const b = fs.readFileSync(path.join(ROOT, "web", "models", name));
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
    },
  };
}

/** JSON.stringify replacer: typed arrays -> plain arrays. */
function plain(_k, v) {
  return ArrayBuffer.isView(v) ? Array.from(v) : v;
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, "web", "models", "meta.json"), "utf8"));
  const { samples, rate } = loadAudio(a.file);
  const opts = { meta, sourceRate: rate, fast: a.fast, debug: !!a.debugDir };
  if (a.same) {
    const npy = (k) => parseNpy(fs.readFileSync(path.join(a.same, `${k}.npy`))).data;
    opts.x16 = Float32Array.from(npy("x16"));
    if (fs.existsSync(path.join(a.same, "x44.npy"))) opts.x44 = Float32Array.from(npy("x44"));
  }
  let last = 0;
  opts.onProgress = (p) => {
    if (a.quiet) return;
    const now = Date.now();
    if (now - last > 1000 || p.stage === "done") {
      last = now;
      process.stderr.write(`[${p.elapsedS.toFixed(1)}s] ${p.label}\n`);
    }
  };
  const t0 = Date.now();
  const res = await transcribe(nodeRuntime(a.threads), samples, rate, opts);
  res.info.wallS = (Date.now() - t0) / 1000;
  if (a.debugDir) {
    fs.mkdirSync(a.debugDir, { recursive: true });
    const m = res.debugMaps;
    fs.writeFileSync(path.join(a.debugDir, "maps.f32"), Buffer.from(m.buffer, m.byteOffset, m.byteLength));
  }
  delete res.debugMaps;
  res.info.runtime = `onnxruntime-web ${ort.env.versions?.web || ""} wasm, ${a.threads} threads (Node ${process.version})`;
  const json = JSON.stringify(res, plain);
  if (a.out) fs.writeFileSync(a.out, json);
  else process.stdout.write(json + "\n");
  process.stderr.write(
    `${path.basename(a.file)}: ${res.tracks.map((t) => `${t.name} (${t.notes.length})`).join(", ")} in ${res.info.wallS.toFixed(1)} s\n`,
  );
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
