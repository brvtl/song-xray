// Copy the ONNX Runtime Web files the page loads into web/vendor/ort/, so the page never fetches them from a CDN.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "node_modules", "onnxruntime-web", "dist");
const dst = path.join(root, "web", "vendor", "ort");
const files = [
  "ort.wasm.min.mjs", // WASM execution provider only (the default path)
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort.webgpu.min.mjs", // optional WebGPU path (?webgpu=1), untested on this machine (CPU only)
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];
fs.mkdirSync(dst, { recursive: true });
for (const f of files) fs.copyFileSync(path.join(src, f), path.join(dst, f));
const lic = path.join(root, "licenses", "onnxruntime-web-LICENSE");
if (fs.existsSync(lic)) fs.copyFileSync(lic, path.join(dst, "LICENSE"));
const version = JSON.parse(fs.readFileSync(path.join(src, "..", "package.json"), "utf8")).version;
fs.writeFileSync(path.join(dst, "VERSION"), `onnxruntime-web ${version}\n`);
console.log(`copied ${files.length} files (onnxruntime-web ${version}) -> ${path.relative(root, dst)}`);
