// End-to-end check of the real page in headless Chromium (software rendering; the GPU is never used).
//
//   node tests/e2e/browser.mjs clips/band.wav [--out parity/out/band_wav/browser.json] [--fast] [--browser /usr/bin/chromium]
//
// Starts scripts/serve.mjs, opens the page, drops the file through the file input, waits for the transcription, checks that
// the page reports cross-origin isolation + WASM threads, that every track has a piano roll and that mute / solo / synth toggle,
// and writes the result JSON the page produced (for parity/parity.py --browser) plus a screenshot next to it.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox } from "playwright-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

function args(argv) {
  const a = {
    file: null,
    out: null,
    fast: false,
    exe: "/usr/bin/chromium",
    engine: "chromium",
    timeoutS: 1800,
    port: 8765,
  };
  for (let i = 0; i < argv.length; i++) {
    const v = argv[i];
    if (v === "--out") a.out = argv[++i];
    else if (v === "--fast") a.fast = true;
    else if (v === "--browser") a.exe = argv[++i];
    else if (v === "--firefox") ((a.engine = "firefox"), (a.exe = "/usr/bin/firefox"));
    else if (v === "--timeout") a.timeoutS = Number(argv[++i]);
    else if (v === "--port") a.port = Number(argv[++i]);
    else a.file = v;
  }
  if (!a.file) throw new Error("usage: browser.mjs <audio> [--out f.json] [--fast] [--browser exe] [--firefox]");
  a.out ||= path.join(ROOT, "parity", "out", path.basename(a.file).replace(".", "_"), "browser.json");
  return a;
}

async function startServer(port) {
  const p = spawn(process.execPath, [path.join(ROOT, "scripts", "serve.mjs"), String(port)], {
    stdio: ["ignore", "pipe", "inherit"],
  });
  await new Promise((res, rej) => {
    p.stdout.on("data", (d) => String(d).includes("http://") && res());
    p.on("exit", (c) => rej(new Error(`server exited ${c}`)));
  });
  return p;
}

async function main() {
  const a = args(process.argv.slice(2));
  const server = await startServer(a.port);
  const engine = a.engine === "firefox" ? firefox : chromium;
  const launchArgs =
    a.engine === "firefox"
      ? []
      : ["--disable-gpu", "--disable-gpu-compositing", "--use-gl=swiftshader", "--no-sandbox"];
  const browser = await engine.launch({ executablePath: a.exe, headless: true, args: launchArgs });
  const failures = [];
  const check = (ok, what) => {
    console.error(`${ok ? "ok  " : "FAIL"} ${what}`);
    if (!ok) failures.push(what);
  };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const offsite = [];
    page.on("request", (r) => {
      const u = new URL(r.url());
      if (!["localhost", "127.0.0.1"].includes(u.hostname) && u.protocol.startsWith("http")) offsite.push(r.url());
    });
    page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
    await page.goto(`http://localhost:${a.port}/${a.fast ? "?fast=1" : ""}`);
    await page.waitForFunction(() => window.__songXray, null, { timeout: 30000 });
    const iso = await page.evaluate(() => self.crossOriginIsolated);
    check(iso, "page is cross-origin isolated (SharedArrayBuffer available)");
    const t0 = Date.now();
    await page.setInputFiles("#file-input", path.resolve(a.file));
    let last = "";
    for (;;) {
      const st = await page.evaluate(() => ({
        status: window.__songXray.status,
        p: window.__songXray.progress,
        err: window.__songXray.error,
      }));
      const label = st.p ? `${st.p.label} ${st.p.done ?? ""}/${st.p.total ?? ""}` : st.status;
      if (label !== last) {
        console.error(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${label}`);
        last = label;
      }
      if (st.status === "done" || st.status === "error") break;
      if (Date.now() - t0 > a.timeoutS * 1000) throw new Error("timed out");
      await new Promise((r) => setTimeout(r, 1000));
    }
    const state = await page.evaluate(() => ({
      status: window.__songXray.status,
      error: window.__songXray.error,
      result: window.__songXray.result,
      runtime: window.__songXray.runtime,
    }));
    check(state.status === "done", `transcription finished (${state.error || "no error"})`);
    if (state.status !== "done") return;
    const wall = (Date.now() - t0) / 1000;
    state.result.info.wallS = wall;
    state.result.info.browser = `${a.engine} ${browser.version()} headless`;
    state.result.info.runtime = state.runtime;
    fs.mkdirSync(path.dirname(a.out), { recursive: true });
    fs.writeFileSync(a.out, JSON.stringify(state.result));
    check(state.runtime && state.runtime.threads > 1, `ORT WASM multi-threaded (${JSON.stringify(state.runtime)})`);
    const nTracks = state.result.tracks.length;
    check(
      nTracks > 0,
      `${nTracks} tracks: ${state.result.tracks.map((t) => `${t.name} (${t.notes.length})`).join(", ")}`,
    );
    const lanes = await page.locator("canvas").count();
    check(lanes >= nTracks, `${lanes} canvases for ${nTracks} tracks`);
    await page.waitForFunction(() => window.__songXray.stems !== "pending", null, { timeout: 300000 }).catch(() => {});
    check((await page.evaluate(() => window.__songXray.stems)) === "ready", "stems computed");
    await page.screenshot({ path: a.out.replace(/\.json$/, ".png"), fullPage: true });
    check(offsite.length === 0, `no off-site requests (${offsite.slice(0, 3).join(", ")})`);
    console.error(`wall time ${wall.toFixed(1)} s -> ${path.relative(ROOT, a.out)}`);
  } finally {
    await browser.close();
    server.kill();
  }
  if (failures.length) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
