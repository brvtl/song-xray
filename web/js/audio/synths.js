// Simple WebAudio voices that follow a track's piano roll: one recipe per GM program family, and synthesized drum hits.
// Every voice ends with a short release towards 0 (setTargetAtTime), so stopping never clicks.

import { programFamily, drumKind } from "../ui/gm.js";

const RELEASE_TAU = 0.03;
const noiseBuffers = new WeakMap();

/** One second of white noise for `ctx`, created once and shared. */
function noise(ctx) {
  let b = noiseBuffers.get(ctx);
  if (!b) {
    b = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    noiseBuffers.set(ctx, b);
  }
  return b;
}

const hz = (p) => 440 * 2 ** ((p - 69) / 12);

/**
 * A scheduled voice: its nodes can be released early (pause / seek / synth off).
 * @typedef {{end: number, stop: (at: number) => void}} Voice
 */

function makeVoice(ctx, out, sources, end) {
  for (const s of sources) s.onended = () => s.disconnect();
  return {
    end,
    stop(at) {
      // Hold the envelope at its value at `at` (0 if the note has not started yet), then release. A bare
      // cancelScheduledValues would also drop the setValueAtTime(0, when) and leave the gain at its intrinsic value.
      const g = out.gain;
      if (typeof g.cancelAndHoldAtTime === "function") g.cancelAndHoldAtTime(at);
      else {
        g.cancelScheduledValues(at);
        g.setValueAtTime(g.value, at);
      }
      g.setTargetAtTime(0, at, 0.01);
      for (const s of sources) {
        try {
          s.stop(at + 0.08);
        } catch {
          /* already stopped */
        }
      }
    },
  };
}

function osc(ctx, type, freq, when, detune = 0) {
  const o = ctx.createOscillator();
  o.type = type;
  o.frequency.setValueAtTime(freq, when);
  if (detune) o.detune.setValueAtTime(detune, when);
  return o;
}

/**
 * Schedule a pitched note on `dest`.
 * @param {BaseAudioContext} ctx
 * @param {AudioNode} dest
 * @param {{program: number, pitch: number, velocity: number, when: number, duration: number}} n velocity 1..127
 * @returns {Voice}
 */
export function playPitched(ctx, dest, { program, pitch, velocity, when, duration }) {
  const family = programFamily(program);
  const f = hz(pitch);
  const peak = 0.06 + 0.22 * (velocity / 127) ** 1.5;
  const dur = Math.max(0.05, duration);
  const off = when + dur;
  const env = ctx.createGain();
  env.gain.value = 0;
  env.gain.setValueAtTime(0, when);
  env.connect(dest);
  const srcs = [];
  let tail = 0.25;

  if (family === "keys") {
    const a = osc(ctx, "triangle", f, when);
    const b = osc(ctx, "sine", f * 2, when);
    const bg = ctx.createGain();
    bg.gain.value = 0.3;
    a.connect(env);
    b.connect(bg).connect(env);
    srcs.push(a, b);
    env.gain.linearRampToValueAtTime(peak, when + 0.005);
    env.gain.setTargetAtTime(peak * 0.05, when + 0.005, 0.25 + 0.6 * Math.max(0, (72 - pitch) / 48));
  } else if (family === "bass") {
    const a = osc(ctx, "sawtooth", f, when);
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.Q.value = 4;
    lp.frequency.setValueAtTime(Math.min(4000, f * 10), when);
    lp.frequency.setTargetAtTime(f * 2.5, when, 0.08);
    a.connect(lp).connect(env);
    srcs.push(a);
    env.gain.linearRampToValueAtTime(peak * 1.3, when + 0.008);
    env.gain.setTargetAtTime(peak * 0.7, when + 0.008, 0.2);
    tail = 0.15;
  } else if (family === "strings") {
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = Math.min(6000, f * 6);
    lp.connect(env);
    for (const d of [-7, 7]) {
      const a = osc(ctx, "sawtooth", f, when, d);
      a.connect(lp);
      srcs.push(a);
    }
    const attack = Math.min(0.12, dur * 0.5);
    env.gain.linearRampToValueAtTime(peak * 0.55, when + attack);
    tail = 0.4;
  } else if (family === "lead") {
    const a = osc(ctx, "square", f, when);
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = Math.min(5000, f * 5);
    const lfo = osc(ctx, "sine", 5.5, when);
    const depth = ctx.createGain();
    depth.gain.setValueAtTime(0, when);
    depth.gain.linearRampToValueAtTime(f * 0.006, when + 0.3);
    lfo.connect(depth).connect(a.frequency);
    a.connect(lp).connect(env);
    srcs.push(a, lfo);
    env.gain.linearRampToValueAtTime(peak * 0.5, when + 0.02);
  } else if (family === "organ") {
    for (const [mult, g] of [
      [1, 1],
      [2, 0.5],
      [4, 0.25],
    ]) {
      const a = osc(ctx, "sine", f * mult, when);
      const ag = ctx.createGain();
      ag.gain.value = g * 0.6;
      a.connect(ag).connect(env);
      srcs.push(a);
    }
    env.gain.linearRampToValueAtTime(peak * 0.6, when + 0.01);
    tail = 0.1;
  } else {
    const a = osc(ctx, "sine", f, when);
    const b = osc(ctx, "triangle", f, when);
    const bg = ctx.createGain();
    bg.gain.value = 0.4;
    a.connect(env);
    b.connect(bg).connect(env);
    srcs.push(a, b);
    env.gain.linearRampToValueAtTime(peak, when + 0.01);
    env.gain.setTargetAtTime(peak * 0.4, when + 0.01, 0.3);
  }
  env.gain.cancelScheduledValues(off);
  env.gain.setTargetAtTime(0, off, RELEASE_TAU);
  const end = off + tail + 6 * RELEASE_TAU;
  for (const s of srcs) {
    s.start(when);
    s.stop(end);
  }
  return makeVoice(ctx, env, srcs, end);
}

function noiseSource(ctx) {
  const s = ctx.createBufferSource();
  s.buffer = noise(ctx);
  s.loop = true;
  s.loopStart = Math.random() * 0.5;
  return s;
}

/**
 * Schedule a synthesized drum hit for GM key `key` on `dest`.
 * @param {BaseAudioContext} ctx
 * @param {AudioNode} dest
 * @param {{key: number, velocity: number, when: number}} hit velocity 1..127
 * @returns {Voice}
 */
export function playDrum(ctx, dest, { key, velocity, when }) {
  const kind = drumKind(key);
  const v = 0.15 + 0.6 * (velocity / 127);
  const env = ctx.createGain();
  env.gain.value = 0;
  env.gain.setValueAtTime(0, when);
  env.connect(dest);
  const srcs = [];
  let end = when + 0.3;
  const hit = (gain, decay) => {
    env.gain.linearRampToValueAtTime(gain, when + 0.002);
    env.gain.setTargetAtTime(0, when + 0.002, decay);
    end = when + 0.002 + decay * 7;
  };

  if (kind === "kick") {
    const o = osc(ctx, "sine", 150, when);
    o.frequency.exponentialRampToValueAtTime(45, when + 0.12);
    o.connect(env);
    srcs.push(o);
    hit(v * 1.1, 0.12);
  } else if (kind === "tom") {
    const base = 80 + (key - 41) * 18;
    const o = osc(ctx, "sine", base * 1.8, when);
    o.frequency.exponentialRampToValueAtTime(base, when + 0.15);
    o.connect(env);
    srcs.push(o);
    hit(v * 0.8, 0.15);
  } else if (kind === "snare" || kind === "clap") {
    const n = noiseSource(ctx);
    const bp = ctx.createBiquadFilter();
    bp.type = kind === "clap" ? "bandpass" : "highpass";
    bp.frequency.value = kind === "clap" ? 1500 : 1200;
    n.connect(bp).connect(env);
    srcs.push(n);
    if (kind === "snare") {
      const o = osc(ctx, "triangle", 190, when);
      o.frequency.exponentialRampToValueAtTime(140, when + 0.05);
      const og = ctx.createGain();
      og.gain.setValueAtTime(0.8, when);
      og.gain.setTargetAtTime(0, when, 0.04);
      o.connect(og).connect(env);
      srcs.push(o);
    }
    hit(v * 0.6, kind === "clap" ? 0.06 : 0.07);
  } else if (kind === "hat" || kind === "openhat" || kind === "cymbal") {
    const n = noiseSource(ctx);
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass";
    hp.frequency.value = kind === "cymbal" ? 5000 : 7000;
    n.connect(hp).connect(env);
    srcs.push(n);
    hit(v * (kind === "hat" ? 0.35 : 0.3), kind === "hat" ? 0.025 : kind === "openhat" ? 0.12 : 0.45);
  } else {
    const n = noiseSource(ctx);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 2500;
    bp.Q.value = 3;
    n.connect(bp).connect(env);
    srcs.push(n);
    hit(v * 0.5, 0.015);
  }
  for (const s of srcs) {
    s.start(when);
    s.stop(end);
  }
  return makeVoice(ctx, env, srcs, end);
}
