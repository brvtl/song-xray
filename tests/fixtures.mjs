// Test helpers: read the reference arrays that scripts/dump_fixtures.py wrote (NumPy .npy, little-endian, C order).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DTYPES = {
  "<f4": Float32Array,
  "<f8": Float64Array,
  "<i8": BigInt64Array,
  "<i4": Int32Array,
  "<i2": Int16Array,
  "|i1": Int8Array,
  "|u1": Uint8Array,
  "|b1": Uint8Array,
  "<f2": Uint16Array, // raw float16 bits
};

/** Parse a .npy buffer -> { data: TypedArray, shape: number[], dtype: string }. int64 is converted to Float64Array. */
export function parseNpy(buf) {
  const major = buf[6];
  const hlen = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const hstart = major === 1 ? 10 : 12;
  const header = buf.toString("latin1", hstart, hstart + hlen);
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  if (fortran) throw new Error("fortran order not supported");
  const shape = /'shape':\s*\(([^)]*)\)/
    .exec(header)[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const Ctor = DTYPES[descr];
  if (!Ctor) throw new Error(`dtype ${descr} not supported`);
  const off = hstart + hlen;
  const bytes = buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + buf.length);
  let data = new Ctor(bytes);
  if (Ctor === BigInt64Array) data = Float64Array.from(data, (v) => Number(v));
  return { data, shape, dtype: descr };
}

/** Does the fixture directory for `clip` exist? (tests skip when it does not: run scripts/dump_fixtures.py first) */
export function hasFixture(clip) {
  return fs.existsSync(path.join(ROOT, "fixtures", clip, "manifest.json"));
}

export function manifest(clip) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, "fixtures", clip, "manifest.json"), "utf8"));
}

export function npy(clip, key) {
  return parseNpy(fs.readFileSync(path.join(ROOT, "fixtures", clip, `${key}.npy`)));
}

/** Max |a - b| / (max |b| + eps) over two array-likes of equal length. */
export function relDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs ${b.length}`);
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > num || Number.isNaN(d)) num = Number.isNaN(d) ? Infinity : d;
    const m = Math.abs(b[i]);
    if (m > den) den = m;
  }
  return num / (den + 1e-12);
}

/** float16 bits -> number */
export function f16(bits) {
  const s = bits & 0x8000 ? -1 : 1;
  const e = (bits >> 10) & 0x1f;
  const m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** An onnxruntime-web session factory for Node tests (WASM backend, the same runtime the page uses). */
export async function nodeSessions() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = Math.min(8, (await import("node:os")).cpus().length);
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
  };
}
