// End-to-end check of the real page in headless Chromium (software rendering; the GPU is never used).
//
//   node tests/e2e/browser.mjs clips/band.wav [--out parity/out/band_wav/browser.json] [--fast] [--browser /usr/bin/chromium]
//
// Starts scripts/serve.mjs, opens the page, drops the file through the file input, waits for the transcription, checks that
// the page reports cross-origin isolation + WASM threads, that every track has a piano roll, that playback runs and that
// Synth / Solo / Mute change the applied gains (and schedule synth voices), with no page errors,
// and writes the result JSON the page produced (for parity/parity.py --browser) plus a screenshot next to it.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox } from "playwright-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
// Playwright drives Firefox only through its own patched build:
//   PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright npx playwright-core install firefox
const FIREFOX = path.join(ROOT, ".cache", "ms-playwright", "firefox-1572", "firefox", "firefox");

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
    else if (v === "--firefox") ((a.engine = "firefox"), (a.exe = FIREFOX));
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

/** Press play, then Synth / Solo / Mute on the first two tracks, and check the gains the page applies and that it plays. */
async function checkControls(page, check, nTracks) {
  const probe = () =>
    page.evaluate(() => ({
      time: window.__songXray.time,
      playing: window.__songXray.playing,
      gains: window.__songXray.gains,
      voices: window.__songXray.synthVoices,
    }));
  await page.click("#play");
  await page.waitForTimeout(2000);
  let p = await probe();
  check(p.playing && p.time > 0.8, `play: playhead advanced to ${p.time.toFixed(2)} s`);
  const lanes = page.locator(".lane:has(.tbtn.y)"); // the transcribed tracks (Original / Residual have no Synth button)
  check((await lanes.count()) === nTracks, `${nTracks} track rows with Mute / Solo / Synth`);
  await lanes.nth(0).locator(".tbtn.y").click();
  await page.waitForTimeout(1200);
  p = await probe();
  check(p.gains.stems[0] === 0 && p.gains.synths[0] > 0, "synth swap: track 1 stem silent, its synth audible");
  check(p.voices[0] > 0, `synth swap: ${p.voices[0]} synth voices scheduled for track 1`);
  if (nTracks > 1) {
    await lanes.nth(1).locator(".tbtn.s").click();
    await page.waitForTimeout(300);
    p = await probe();
    const others = p.gains.stems.some((g, k) => k !== 1 && g > 0) || p.gains.synths.some((g, k) => k !== 1 && g > 0);
    check(p.gains.stems[1] > 0 && !others && p.gains.residual === 0, "solo track 2: only track 2 audible");
    await lanes.nth(1).locator(".tbtn.s").click(); // un-solo
    await lanes.nth(1).locator(".tbtn.m").click();
    await page.waitForTimeout(300);
    p = await probe();
    check(
      p.gains.stems[1] === 0 && p.gains.synths[1] === 0 && p.gains.synths[0] > 0,
      "mute track 2: silent, others back",
    );
  }
  await page.click("#play");
  await page.waitForTimeout(300);
  p = await probe();
  check(!p.playing, "pause");
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
    const pageErrors = [];
    page.on("console", (m) => m.type() === "error" && (pageErrors.push(m.text()), console.error("[page]", m.text())));
    page.on("pageerror", (e) => (pageErrors.push(String(e)), console.error("[page]", String(e))));
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
    await checkControls(page, check, nTracks);
    check(pageErrors.length === 0, `no page errors (${pageErrors.slice(0, 2).join(" | ")})`);
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
