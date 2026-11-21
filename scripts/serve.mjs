// Local static server for the demo. It only serves files; inference runs in the browser.
//
//   node scripts/serve.mjs [port]        (npm run serve)
//
// It sends Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy so the page is cross-origin isolated: that enables
// SharedArrayBuffer, which ONNX Runtime Web needs for multi-threaded WASM (about 5x faster than one thread). Any static server
// that sends these two headers works the same (see README).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mounts = [
  ["/clips/", path.join(root, "clips")],
  ["/licenses/", path.join(root, "licenses")],
  ["/docs/", path.join(root, "docs")],
  ["/", path.join(root, "web")],
];
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
};

const ROOT_FILES = new Set(["/README.md", "/LICENSE", "/NOTICE"]);

function resolve(urlPath) {
  if (ROOT_FILES.has(urlPath)) return path.join(root, urlPath.slice(1));
  for (const [prefix, dir] of mounts) {
    if (!urlPath.startsWith(prefix)) continue;
    const rel = decodeURIComponent(urlPath.slice(prefix.length)) || "index.html";
    const file = path.resolve(dir, rel);
    if (!file.startsWith(dir + path.sep) && file !== dir) return null; // no path traversal out of the mount
    return file;
  }
  return null;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let file = resolve(url.pathname);
  if (file && fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  const headers = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-cache",
  };
  if (!file || !fs.existsSync(file)) {
    res.writeHead(404, { ...headers, "Content-Type": "text/plain" });
    res.end("not found");
    return;
  }
  const stat = fs.statSync(file);
  res.writeHead(200, {
    ...headers,
    "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
    "Content-Length": stat.size,
  });
  if (req.method === "HEAD") return res.end();
  fs.createReadStream(file).pipe(res);
});

const port = Number(process.argv[2] || process.env.PORT || 8080);
server.listen(port, "127.0.0.1", () => {
  console.log(`Song X-ray: http://localhost:${port}/   (serving web/ with COOP/COEP; Ctrl-C to stop)`);
});
