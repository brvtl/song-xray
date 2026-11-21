import { test } from "node:test";
import assert from "node:assert/strict";
import { writeMidi, PPQ } from "../web/js/midi.js";

/** Tiny SMF reader for the subset writeMidi emits (no running status). */
function readMidi(bytes) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const id = (p) => String.fromCharCode(...bytes.subarray(p, p + 4));
  assert.equal(id(0), "MThd");
  const format = v.getUint16(8);
  const ntrks = v.getUint16(10);
  const division = v.getUint16(12);
  let p = 14;
  const tracks = [];
  for (let t = 0; t < ntrks; t++) {
    assert.equal(id(p), "MTrk");
    const len = v.getUint32(p + 4);
    let q = p + 8;
    const end = q + len;
    let tick = 0;
    const events = [];
    const vl = () => {
      let x = 0;
      for (;;) {
        const b = bytes[q++];
        x = (x << 7) | (b & 0x7f);
        if (!(b & 0x80)) return x;
      }
    };
    while (q < end) {
      tick += vl();
      const s = bytes[q++];
      if (s === 0xff) {
        const type = bytes[q++];
        const n = vl();
        events.push({ tick, meta: type, data: bytes.slice(q, q + n) });
        q += n;
      } else {
        const kind = s & 0xf0;
        const n = kind === 0xc0 || kind === 0xd0 ? 1 : 2;
        events.push({ tick, kind, ch: s & 15, data: [...bytes.subarray(q, q + n)] });
        q += n;
      }
    }
    assert.equal(q, end);
    tracks.push(events);
    p = end;
  }
  return { format, ntrks, division, tracks };
}

const result = {
  tracks: [
    {
      name: "piano",
      program: 0,
      isDrum: false,
      notes: [
        [0, 0.5, 60, 100],
        [0.5, 1, 60, 90],
      ],
    },
    { name: "electric bass", program: 33, isDrum: false, notes: [[1.25, 2, 40, 127]] },
    { name: "Drums", program: 0, isDrum: true, notes: [[0.25, 0.35, 36, 64]] },
  ],
};

test("type-1 SMF: conductor + one track per instrument, 480 ppq, tempo 120, credit", () => {
  const m = readMidi(writeMidi(result));
  assert.equal(m.format, 1);
  assert.equal(m.division, PPQ);
  assert.equal(m.ntrks, 4);
  const tempo = m.tracks[0].find((e) => e.meta === 0x51);
  assert.deepEqual([...tempo.data], [0x07, 0xa1, 0x20]);
  const text = new TextDecoder().decode(m.tracks[0].find((e) => e.meta === 0x01).data);
  assert.match(text, /Dettato/);
  for (const tr of m.tracks) assert.equal(tr.at(-1).meta, 0x2f);
});

test("notes: seconds -> ticks, programs, drums on channel 10, off-before-on at equal ticks", () => {
  const m = readMidi(writeMidi(result));
  const piano = m.tracks[1];
  assert.equal(new TextDecoder().decode(piano.find((e) => e.meta === 0x03).data), "piano");
  const notes = piano.filter((e) => e.kind === 0x90 || e.kind === 0x80);
  assert.deepEqual(
    notes.map((e) => [e.tick, e.kind, e.data[0]]),
    [
      [0, 0x90, 60],
      [480, 0x80, 60],
      [480, 0x90, 60],
      [960, 0x80, 60],
    ],
  );
  const bass = m.tracks[2];
  const pc = bass.find((e) => e.kind === 0xc0);
  assert.equal(pc.data[0], 33);
  assert.equal(pc.ch, 1);
  assert.equal(bass.find((e) => e.kind === 0x90).tick, 1200);
  const drums = m.tracks[3];
  assert.ok(drums.filter((e) => e.kind === 0x90 || e.kind === 0x80).every((e) => e.ch === 9));
  assert.ok(!drums.some((e) => e.kind === 0xc0));
});

test("an empty result still yields a valid file", () => {
  const m = readMidi(writeMidi({ tracks: [] }));
  assert.equal(m.ntrks, 1);
});
