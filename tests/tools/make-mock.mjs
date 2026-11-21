// Build web/demo/mock-result.json from a reference fixture manifest (default: fixtures/band/manifest.json),
// in the page's result format, so the UI can be developed without running inference.
// Usage: node tests/tools/make-mock.mjs [manifest.json] [out.json]
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const src = resolve(root, process.argv[2] ?? "fixtures/band/manifest.json");
const out = resolve(root, process.argv[3] ?? "web/demo/mock-result.json");
const m = JSON.parse(readFileSync(src, "utf8"));

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const tracks = m.tracks.map((t) => ({
  name: String(t.instrument).replaceAll("_", " "),
  program: t.program | 0,
  isDrum: false,
  notes: t.notes.map(([on, off, p, v]) => [r4(on), r4(off), Math.round(p), Math.round(v)]),
}));
if (m.drum_hits?.length) {
  tracks.push({
    name: "Drums",
    program: 0,
    isDrum: true,
    notes: m.drum_hits.map((h) => [
      r4(h.onset),
      r4(h.offset),
      h.pitch | 0,
      Math.max(1, Math.min(127, Math.round(h.velocity * 127))),
    ]),
  });
}
const x16 = m.arrays?.x16?.[0]?.[0];
const seconds = x16 ? x16 / 16000 : Math.max(...tracks.flatMap((t) => t.notes.map((n) => n[1])));
const pitched = tracks.filter((t) => !t.isDrum).reduce((s, t) => s + t.notes.length, 0);
const result = {
  seconds,
  sourceRate: Number(m.source_rate),
  tracks,
  info: {
    mock: true,
    clip: m.clip,
    tuningCents: Math.round((m.tuning?.delta ?? 0) * 100) / 100,
    notes: { transcribed: m.repair_info?.transcribed, inserted: m.repair_info?.inserted, grouped: pitched },
    drums: { model: m.drum_model, hits: m.drum_hits?.length ?? 0 },
    timings: {
      transcriber: m.time_transcriber_s,
      grouper: m.time_grouper_s,
      drums: m.time_drums_s,
      total: m.time_total_s,
    },
  },
};
writeFileSync(out, JSON.stringify(result) + "\n");
console.log(`wrote ${out}: ${tracks.length} tracks, ${seconds}s`);
