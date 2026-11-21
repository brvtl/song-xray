// Minimal Standard MIDI File (type 1) writer for a Song X-ray result. DOM-free.

/** Ticks per quarter note. */
export const PPQ = 480;
const TEMPO_US = 500000; // 120 bpm: one quarter = 0.5 s, so 1 s = 2 * PPQ ticks
const TICKS_PER_S = (PPQ * 1e6) / TEMPO_US;
const DRUM_CHANNEL = 9;
const CREDIT = "Transcription by Dettato (KobiMusic, Apache-2.0) - github.com/kobimusic/dettato, via Song X-ray";

function varLen(v) {
  const bytes = [v & 0x7f];
  while ((v >>= 7) > 0) bytes.unshift((v & 0x7f) | 0x80);
  return bytes;
}

function textBytes(s) {
  return [...new TextEncoder().encode(s)];
}

function metaText(type, s) {
  const t = textBytes(s);
  return [0xff, type, ...varLen(t.length), ...t];
}

function chunk(id, data) {
  const n = data.length;
  return [
    ...[...id].map((c) => c.charCodeAt(0)),
    (n >>> 24) & 255,
    (n >>> 16) & 255,
    (n >>> 8) & 255,
    n & 255,
    ...data,
  ];
}

/** Events [tick, order, bytes] -> delta-timed track bytes ending with End of Track. */
function trackBytes(events) {
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  let last = 0;
  for (const [tick, , bytes] of events) {
    out.push(...varLen(tick - last), ...bytes);
    last = tick;
  }
  out.push(0, 0xff, 0x2f, 0);
  return out;
}

const clamp7 = (v) => Math.max(0, Math.min(127, Math.round(v)));

/**
 * Encode a transcription as a type-1 SMF: a conductor track (tempo 120, 4/4, credit text) and one track per instrument
 * with its GM program; drum tracks go to channel 10, pitched tracks to channels 1-16 skipping 10 (reused cyclically).
 * Times in seconds are converted to ticks at 480 ppq.
 * @param {{tracks: {name: string, program: number, isDrum: boolean, notes: number[][]}[]}} result
 * @param {{title?: string}} [opts]
 * @returns {Uint8Array} the .mid file bytes
 */
export function writeMidi(result, opts = {}) {
  const tracks = result.tracks ?? [];
  const conductor = [
    [0, 0, metaText(0x03, opts.title ?? "Song X-ray")],
    [0, 0, metaText(0x01, CREDIT)],
    [0, 0, [0xff, 0x51, 3, (TEMPO_US >> 16) & 255, (TEMPO_US >> 8) & 255, TEMPO_US & 255]],
    [0, 0, [0xff, 0x58, 4, 4, 2, 24, 8]],
  ];
  const chunks = [chunk("MTrk", trackBytes(conductor))];
  const melodic = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
  let next = 0;
  for (const tr of tracks) {
    const ch = tr.isDrum ? DRUM_CHANNEL : melodic[next++ % melodic.length];
    const ev = [[0, 0, metaText(0x03, tr.name)]];
    if (!tr.isDrum) ev.push([0, 1, [0xc0 | ch, clamp7(tr.program)]]);
    for (const [on, off, pitch, vel] of tr.notes) {
      const a = Math.max(0, Math.round(on * TICKS_PER_S));
      const b = Math.max(a + 1, Math.round(off * TICKS_PER_S));
      const p = clamp7(pitch);
      // Order at equal ticks: note-offs (2) before note-ons (3), so a repeated pitch is not cut short.
      ev.push([a, 3, [0x90 | ch, p, Math.max(1, clamp7(vel ?? 100))]]);
      ev.push([b, 2, [0x80 | ch, p, 0]]);
    }
    chunks.push(chunk("MTrk", trackBytes(ev)));
  }
  const n = chunks.length;
  const header = chunk("MThd", [0, 1, (n >> 8) & 255, n & 255, (PPQ >> 8) & 255, PPQ & 255]);
  return new Uint8Array([...header, ...chunks.flat()]);
}
