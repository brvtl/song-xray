// Player scheduler + synth voice tests against a minimal fake WebAudio graph (no audio is rendered; we inspect the calls).
import { test } from "node:test";
import assert from "node:assert/strict";

/** An AudioParam that records its automation calls. */
class FakeParam {
  constructor(value = 1, { hold = true } = {}) {
    this.value = value;
    this.calls = [];
    if (!hold) this.cancelAndHoldAtTime = undefined;
  }
  setValueAtTime(v, t) {
    this.calls.push(["setValueAtTime", v, t]);
  }
  linearRampToValueAtTime(v, t) {
    this.calls.push(["linearRampToValueAtTime", v, t]);
  }
  exponentialRampToValueAtTime(v, t) {
    this.calls.push(["exponentialRampToValueAtTime", v, t]);
  }
  setTargetAtTime(v, t, tau) {
    this.calls.push(["setTargetAtTime", v, t, tau]);
  }
  cancelScheduledValues(t) {
    this.calls.push(["cancelScheduledValues", t]);
  }
  cancelAndHoldAtTime(t) {
    this.calls.push(["cancelAndHoldAtTime", t]);
  }
}

class FakeNode {
  constructor(ctx) {
    this.ctx = ctx;
  }
  connect(n) {
    return n;
  }
  disconnect() {}
}

class FakeSource extends FakeNode {
  constructor(ctx, kind) {
    super(ctx);
    this.kind = kind;
    this.frequency = new FakeParam(440);
    this.detune = new FakeParam(0);
    this.started = [];
    ctx.sources.push(this);
  }
  start(when, offset) {
    this.started.push([when, offset]);
  }
  stop() {}
}

let holdSupported = true;

class FakeAudioContext {
  constructor() {
    this.currentTime = 0;
    this.sampleRate = 48000;
    this.state = "suspended";
    this.destination = new FakeNode(this);
    this.sources = [];
    this.gains = [];
    this.resumeCalls = 0;
    this.waiters = [];
  }
  resume() {
    this.resumeCalls++;
    return new Promise((res) => this.waiters.push(res));
  }
  /** Let every pending resume() settle, like a real context starting. */
  pendingResume() {
    this.state = "running";
    for (const res of this.waiters.splice(0)) res();
  }
  createGain() {
    const g = new FakeNode(this);
    g.gain = new FakeParam(1, { hold: holdSupported });
    this.gains.push(g);
    return g;
  }
  createBufferSource() {
    return new FakeSource(this, "buffer");
  }
  createOscillator() {
    return new FakeSource(this, "osc");
  }
  createBiquadFilter() {
    const f = new FakeNode(this);
    f.frequency = new FakeParam(350);
    f.Q = new FakeParam(1);
    return f;
  }
  createBuffer(ch, n, rate) {
    const d = new Float32Array(n);
    return { duration: n / rate, length: n, getChannelData: () => d };
  }
}

globalThis.AudioContext = FakeAudioContext;
const { Player } = await import("../web/js/audio/player.js");
const { playPitched, playDrum } = await import("../web/js/audio/synths.js");

const fakeBuffer = (duration) => ({ duration });

/* ---------- synth voices ---------- */

for (const hold of [true, false]) {
  test(`voice stopped before its onset never leaves the default gain 1 (cancelAndHold ${hold ? "native" : "fallback"})`, () => {
    holdSupported = hold;
    try {
      for (const make of [
        (ctx, d) => playPitched(ctx, d, { program: 0, pitch: 60, velocity: 100, when: 1, duration: 0.5 }),
        (ctx, d) => playDrum(ctx, d, { key: 36, velocity: 100, when: 1 }),
      ]) {
        const ctx = new FakeAudioContext();
        const voice = make(ctx, ctx.destination);
        const env = ctx.gains[0];
        assert.equal(env.gain.value, 0, "envelope starts silent even with no automation event yet");
        env.gain.calls = [];
        voice.stop(0.5);
        const kinds = env.gain.calls.map((c) => c[0]);
        if (hold) assert.equal(kinds[0], "cancelAndHoldAtTime");
        else assert.deepEqual(kinds.slice(0, 2), ["cancelScheduledValues", "setValueAtTime"]);
        const last = env.gain.calls.at(-1);
        assert.equal(last[1], 0, "ends ramping to 0");
        assert.ok(last[2] >= 0.5, "ramp starts at the stop time");
      }
    } finally {
      holdSupported = true;
    }
  });
}

/* ---------- player ---------- */

test("play() twice while the context resumes starts every source once and one timer", async () => {
  const p = new Player();
  try {
    p.load({ original: fakeBuffer(10), tracks: [], duration: 10 });
    const a = p.play();
    const b = p.play();
    assert.equal(p.ctx.resumeCalls, 1);
    p.ctx.pendingResume();
    await Promise.all([a, b]);
    const buffers = p.ctx.sources.filter((s) => s.kind === "buffer");
    assert.equal(buffers.length, 1);
    assert.equal(buffers[0].started.length, 1);
    assert.ok(p.playing);
  } finally {
    p.stop();
  }
});

test("pause during the resume cancels the pending start", async () => {
  const p = new Player();
  try {
    p.load({ original: fakeBuffer(10), tracks: [], duration: 10 });
    const a = p.play();
    p.pause();
    p.ctx.pendingResume();
    await a;
    assert.equal(p.playing, false);
    assert.equal(p.ctx.sources.length, 0);
    assert.equal(p.timer, 0);
  } finally {
    p.stop();
  }
});

test("seek while a start is pending still starts (once) at the new position", async () => {
  const p = new Player();
  try {
    p.load({ original: fakeBuffer(10), tracks: [], duration: 10 });
    const a = p.play();
    p.seek(4);
    p.ctx.pendingResume();
    await a;
    await new Promise((r) => setTimeout(r, 0));
    const buffers = p.ctx.sources.filter((s) => s.kind === "buffer");
    assert.equal(buffers.length, 1);
    assert.equal(buffers[0].started[0][1], 4);
    assert.ok(p.playing);
  } finally {
    p.stop();
  }
});

test("playing from inside a long note sounds its remainder; past drum hits are not replayed", async () => {
  const p = new Player();
  p.ctx.state = "running";
  try {
    const tracks = [
      // a long held note (0..8 s) followed by a short one that already ended before 5 s, then one after 5 s
      {
        program: 0,
        isDrum: false,
        notes: [
          [0, 8, 60, 100],
          [1, 2, 64, 100],
          [5.1, 5.15, 67, 100],
        ],
      },
      {
        program: 0,
        isDrum: true,
        notes: [
          [4.9, 5.0, 36, 100],
          [5.05, 5.1, 38, 100],
        ],
      },
    ];
    p.load({ original: null, tracks, duration: 10 });
    p.setGains({ original: 0, residual: 0, stems: [0, 0], synths: [1, 1], mode: "" });
    p.seek(5);
    await p.play();
    const oscStarts = p.ctx.sources.filter((s) => s.kind === "osc").map((s) => s.started[0][0]);
    // keys family = 2 oscillators per note: the held note (from now) and the 5.1 s note; nothing for the 1..2 s note
    // drums: only the 5.05 s snare (osc + noise); the 4.9 s kick is in the past
    const voices = p.synth.map((s) => s.voices.length);
    assert.deepEqual(voices, [2, 1]);
    assert.ok(
      oscStarts.every((w) => w >= p.startedAt - 1e-9),
      "nothing scheduled before the audio starts",
    );
  } finally {
    p.stop();
  }
});

test("turning a synth on mid-note sounds the held note's remainder", async () => {
  const p = new Player();
  p.ctx.state = "running";
  try {
    const tracks = [{ program: 0, isDrum: false, notes: [[0, 8, 60, 100]] }];
    p.load({ original: fakeBuffer(10), tracks, duration: 10 });
    p.setGains({ original: 1, residual: 0, stems: [0], synths: [0], mode: "" });
    p.seek(3);
    await p.play();
    assert.equal(p.synth[0].voices.length, 0);
    p.setGains({ original: 1, residual: 0, stems: [0], synths: [1], mode: "" });
    p.schedule();
    assert.equal(p.synth[0].voices.length, 1);
  } finally {
    p.stop();
  }
});
