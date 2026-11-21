import { test } from "node:test";
import assert from "node:assert/strict";
import { separateStems, stemChannels } from "../web/js/stems.js";

const RATE = 22050;

function sine(n, freq, amp = 0.5, rate = RATE) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / rate);
  return out;
}

function noise(n, amp, seed = 1) {
  const out = new Float32Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = amp * (s / 2 ** 31 - 1);
  }
  return out;
}

function add(...xs) {
  const out = new Float32Array(xs[0].length);
  for (const x of xs) for (let i = 0; i < out.length; i++) out[i] += x[i];
  return out;
}

function energy(x) {
  let e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return e;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

const midiHz = (p) => 440 * 2 ** ((p - 69) / 12);

test("stems + residual reconstruct the input (stereo, pitched + drums, odd length)", () => {
  const n = RATE * 3 + 777;
  const left = add(sine(n, midiHz(57), 0.4), noise(n, 0.2, 7));
  const right = add(sine(n, midiHz(64), 0.3), noise(n, 0.1, 9));
  const tracks = [
    { name: "piano", program: 0, isDrum: false, notes: [[0.2, 2.0, 57, 100]] },
    { name: "bass", program: 33, isDrum: false, notes: [[1.0, 2.8, 40, 90]] },
    {
      name: "Drums",
      program: 0,
      isDrum: true,
      notes: [
        [0.5, 0.6, 36, 100],
        [1.5, 1.6, 38, 80],
      ],
    },
  ];
  const { stems, residual } = separateStems({ channels: [left, right], rate: RATE, tracks });
  assert.equal(stems.length, 3);
  for (const [c, x] of [left, right].entries()) {
    const sum = add(...stems.map((s) => s[c]), residual[c]);
    assert.equal(sum.length, n);
    assert.ok(maxAbsDiff(sum, x) < 1e-4, `channel ${c}: max diff ${maxAbsDiff(sum, x)}`);
  }
});

test("input shorter than one FFT frame still reconstructs", () => {
  const x = noise(1000, 0.5, 3);
  const { stems, residual } = separateStems({
    channels: [x],
    rate: RATE,
    tracks: [{ name: "piano", program: 0, isDrum: false, notes: [[0, 0.04, 69, 100]] }],
  });
  assert.ok(maxAbsDiff(add(stems[0][0], residual[0]), x) < 1e-4);
});

test("no tracks: everything goes to the residual", () => {
  const x = sine(RATE, 440);
  const { stems, residual } = separateStems({ channels: [x], rate: RATE, tracks: [] });
  assert.equal(stems.length, 0);
  assert.ok(maxAbsDiff(residual[0], x) < 1e-4);
});

test("two sines: each track receives most of its own sine", () => {
  const n = RATE * 2;
  const a = sine(n, midiHz(57), 0.5); // A3
  const b = sine(n, midiHz(76), 0.5); // E5
  const tracks = [
    { name: "low", program: 0, isDrum: false, notes: [[0, 2, 57, 100]] },
    { name: "high", program: 80, isDrum: false, notes: [[0, 2, 76, 100]] },
  ];
  const { stems, residual } = separateStems({ channels: [add(a, b)], rate: RATE, tracks });
  // Compare away from the clip edges.
  const mid = (x) => x.subarray(RATE * 0.25, RATE * 1.75);
  const errA =
    energy(
      mid(
        add(
          stems[0][0],
          a.map((v) => -v),
        ),
      ),
    ) / energy(mid(a));
  const errB =
    energy(
      mid(
        add(
          stems[1][0],
          b.map((v) => -v),
        ),
      ),
    ) / energy(mid(b));
  assert.ok(errA < 0.05, `low track error ratio ${errA}`);
  assert.ok(errB < 0.05, `high track error ratio ${errB}`);
  assert.ok(energy(mid(residual[0])) < 0.05 * energy(mid(add(a, b))), "residual should be small");
});

test("drum hits pull transient energy into the drum stem, not the pitched one", () => {
  const n = RATE * 2;
  const tone = sine(n, midiHz(69), 0.3);
  const clicks = new Float32Array(n);
  const hitTimes = [0.5, 1.0, 1.5];
  const burst = noise(400, 0.8, 11);
  for (const t of hitTimes) {
    const i0 = Math.round(t * RATE);
    for (let i = 0; i < burst.length; i++) clicks[i0 + i] += burst[i] * Math.exp(-i / 80);
  }
  const tracks = [
    { name: "organ", program: 16, isDrum: false, notes: [[0, 2, 69, 100]] },
    { name: "Drums", program: 0, isDrum: true, notes: hitTimes.map((t) => [t, t + 0.1, 42, 100]) },
  ];
  const { stems } = separateStems({ channels: [add(tone, clicks)], rate: RATE, tracks });
  const drumE = energy(stems[1][0]);
  assert.ok(drumE > 0.4 * energy(clicks), `drum stem energy ${drumE} vs clicks ${energy(clicks)}`);
  const leak = energy(
    add(
      stems[0][0],
      tone.map((v) => -v),
    ),
  );
  assert.ok(leak < 0.5 * energy(clicks), `pitched stem picked up ${leak} of click energy`);
});

test("progress callback reports monotonically to completion", () => {
  const x = noise(RATE * 2, 0.3);
  const seen = [];
  separateStems({
    channels: [x],
    rate: RATE,
    tracks: [{ name: "p", program: 0, isDrum: false, notes: [[0, 1, 60, 90]] }],
    onProgress: (done, total) => seen.push([done, total]),
  });
  assert.ok(seen.length > 0);
  const [lastDone, lastTotal] = seen.at(-1);
  assert.equal(lastDone, lastTotal);
  for (let i = 1; i < seen.length; i++) assert.ok(seen[i][0] >= seen[i - 1][0]);
});

test("without a drum track, percussive clicks go to the residual, not the pitched stem", () => {
  const n = RATE * 2;
  const tone = sine(n, midiHz(69), 0.3);
  const clicks = new Float32Array(n);
  const burst = noise(400, 0.8, 13);
  for (const t of [0.5, 1.0, 1.5]) {
    const i0 = Math.round(t * RATE);
    for (let i = 0; i < burst.length; i++) clicks[i0 + i] += burst[i] * Math.exp(-i / 80);
  }
  const tracks = [{ name: "organ", program: 16, isDrum: false, notes: [[0, 2, 69, 100]] }];
  const { stems, residual } = separateStems({ channels: [add(tone, clicks)], rate: RATE, tracks });
  const leak = energy(
    add(
      stems[0][0],
      tone.map((v) => -v),
    ),
  );
  assert.ok(leak < 0.25 * energy(clicks), `pitched stem picked up ${leak} of click energy ${energy(clicks)}`);
  assert.ok(energy(residual[0]) > 0.7 * energy(clicks), `residual ${energy(residual[0])}`);
});

test("stem memory budget: stereo when it fits, else mono, else no stems at all", () => {
  const min = 44100 * 60;
  assert.equal(stemChannels({ channels: 2, tracks: 5, samples: 4 * min }), 2);
  assert.equal(stemChannels({ channels: 1, tracks: 5, samples: 4 * min }), 1);
  // 4 min stereo with 12 tracks: (13 x 2 x 10.6 M x 4 B) = 1.1 GB > stereo budget, mono 550 MB fits
  assert.equal(stemChannels({ channels: 2, tracks: 12, samples: 4 * min }), 1);
  // 20 min with 12 tracks: even mono is (13 x 52.9 M x 4 B) = 2.75 GB
  assert.equal(stemChannels({ channels: 2, tracks: 12, samples: 20 * min }), 0);
  assert.equal(stemChannels({ channels: 1, tracks: 12, samples: 20 * min }), 0);
});
