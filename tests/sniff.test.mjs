import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, mkdtempSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { sniffSampleRate } from "../web/js/audio/sniff.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ab = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);

/** Minimal WAV: RIFF header, optional extra chunk before fmt, fmt (PCM or extensible), data. */
function wav(rate, { extraChunk = false, extensible = false, rf64 = false } = {}) {
  const parts = [];
  const str = (s) => [...s].map((c) => c.charCodeAt(0));
  const u32 = (v) => [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255];
  const u16 = (v) => [v & 255, (v >> 8) & 255];
  if (extraChunk) parts.push(...str("LIST"), ...u32(3), 1, 2, 3, 0); // odd size + pad byte
  const fmtLen = extensible ? 40 : 16;
  parts.push(...str("fmt "), ...u32(fmtLen), ...u16(extensible ? 0xfffe : 1), ...u16(2), ...u32(rate));
  parts.push(...u32(rate * 4), ...u16(4), ...u16(16));
  if (extensible) parts.push(...u16(22), ...new Array(22).fill(0));
  parts.push(...str("data"), ...u32(4), 0, 0, 0, 0);
  return ab(new Uint8Array([...str(rf64 ? "RF64" : "RIFF"), ...u32(parts.length + 4), ...str("WAVE"), ...parts]));
}

/** MPEG audio frame header bytes: version bits (3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5), layer III, bitrate idx, sr idx. */
function mpegHeader(versionBits, brIdx, srIdx) {
  return [0xff, 0xe0 | (versionBits << 3) | (1 << 1) | 1, (brIdx << 4) | (srIdx << 2), 0xc4];
}

/** A run of `frames` identical MP3 frames, optionally after an ID3v2 tag and some junk. */
function mp3(versionBits, brIdx, srIdx, { id3 = 0, junk = 0, frames = 3 } = {}) {
  const BR1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
  const BR2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
  const SR = { 3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000] }[versionBits];
  const br = (versionBits === 3 ? BR1 : BR2)[brIdx] * 1000;
  const len = Math.floor(((versionBits === 3 ? 144 : 72) * br) / SR[srIdx]);
  const out = [];
  if (id3) {
    const s = id3;
    out.push(0x49, 0x44, 0x33, 4, 0, 0, (s >> 21) & 127, (s >> 14) & 127, (s >> 7) & 127, s & 127);
    out.push(...new Array(id3).fill(0));
  }
  out.push(...new Array(junk).fill(0x11));
  for (let f = 0; f < frames; f++) out.push(...mpegHeader(versionBits, brIdx, srIdx), ...new Array(len - 4).fill(0));
  return ab(new Uint8Array(out));
}

test("WAV: plain, with a chunk before fmt, extensible, RF64", () => {
  assert.equal(sniffSampleRate(wav(44100), "a.wav"), 44100);
  assert.equal(sniffSampleRate(wav(48000, { extraChunk: true }), "a.wav"), 48000);
  assert.equal(sniffSampleRate(wav(96000, { extensible: true }), "a.wav"), 96000);
  assert.equal(sniffSampleRate(wav(22050, { rf64: true }), "a.wav"), 22050);
});

test("MP3: MPEG1 / MPEG2 / MPEG2.5 rates, ID3v2 skipped, junk before sync", () => {
  assert.equal(sniffSampleRate(mp3(3, 9, 0), "a.mp3"), 44100);
  assert.equal(sniffSampleRate(mp3(3, 9, 1, { id3: 300 }), "a.mp3"), 48000);
  assert.equal(sniffSampleRate(mp3(3, 9, 2, { junk: 17 }), "a.mp3"), 32000);
  assert.equal(sniffSampleRate(mp3(2, 8, 0), "a.mp3"), 22050);
  assert.equal(sniffSampleRate(mp3(2, 8, 1, { id3: 1000 }), "a.mp3"), 24000);
  assert.equal(sniffSampleRate(mp3(0, 8, 2), "a.mp3"), 8000);
});

test("MP3: a false sync inside the ID3 tag or junk is not taken as a frame", () => {
  // Junk that looks like a header (MPEG1 32 kHz) but is not followed by a second frame.
  const fake = [...mpegHeader(3, 9, 2), 1, 2, 3, 4, 5, 6];
  const real = new Uint8Array(mp3(3, 9, 1));
  const buf = new Uint8Array([...fake, ...real]);
  assert.equal(sniffSampleRate(ab(buf), "a.mp3"), 48000);
});

test("unknown or broken input returns null", () => {
  assert.equal(sniffSampleRate(new ArrayBuffer(0), "x.mp3"), null);
  assert.equal(sniffSampleRate(ab(new Uint8Array([0x66, 0x4c, 0x61, 0x43, 0, 0, 0, 34])), "x.flac"), null);
  assert.equal(sniffSampleRate(ab(new Uint8Array(4000).fill(0x22)), "x.ogg"), null);
  // Reserved sample-rate index (3) everywhere: no valid frame.
  assert.equal(sniffSampleRate(ab(new Uint8Array([0xff, 0xfb, 0x9c, 0x00, 0xff, 0xfb, 0x9c, 0x00])), "x.mp3"), null);
  // Truncated WAV header.
  assert.equal(sniffSampleRate(wav(44100).slice(0, 20), "x.wav"), null);
});

const clips = join(root, "clips");
const realWavs = existsSync(clips) ? readdirSync(clips).filter((f) => f.endsWith(".wav")) : [];
let hasFfmpeg = true;
try {
  execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
  execFileSync("ffprobe", ["-version"], { stdio: "ignore" });
} catch {
  hasFfmpeg = false;
}

test(
  "real files: clips/*.wav",
  { skip: realWavs.length && hasFfmpeg ? false : "needs clips/*.wav and ffprobe" },
  () => {
    for (const f of realWavs) {
      const args = ["-v", "error", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", join(clips, f)];
      const probe = execFileSync("ffprobe", args);
      const rate = Number(String(probe).trim());
      assert.equal(sniffSampleRate(ab(readFileSync(join(clips, f))), f), rate, f);
    }
  },
);

test(
  "real files: ffmpeg-encoded mp3 (44.1 kHz MPEG1 with ID3, 22.05 kHz MPEG2)",
  { skip: hasFfmpeg && realWavs.includes("band.wav") ? false : "needs ffmpeg and clips/band.wav" },
  () => {
    const dir = mkdtempSync(join(tmpdir(), "sniff-"));
    try {
      const src = join(clips, "band.wav");
      const a = join(dir, "band.mp3");
      const b = join(dir, "band22.mp3");
      execFileSync("ffmpeg", ["-v", "error", "-y", "-i", src, "-b:a", "192k", a]);
      execFileSync("ffmpeg", ["-v", "error", "-y", "-i", src, "-ar", "22050", "-b:a", "64k", b]);
      assert.equal(sniffSampleRate(ab(readFileSync(a)), "band.mp3"), 44100);
      assert.equal(sniffSampleRate(ab(readFileSync(b)), "band22.mp3"), 22050);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
);
