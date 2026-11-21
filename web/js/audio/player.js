// WebAudio transport: the original file, the stems + residual, and one synth bus per track, all behind a master gain.
// Every buffer source starts at the same `when` with the same offset (a seek restarts all of them). Synth notes are put on
// the audio clock by a lookahead scheduler (every 50 ms, 0.2 s ahead), only for tracks whose synth is audible.

import { playPitched, playDrum } from "./synths.js";

const LOOKAHEAD_S = 0.2;
const TICK_MS = 50;
const RAMP_TAU = 0.015;

/** First index i with key(i) >= t, for a non-decreasing key over [0, n). */
function lowerBound(n, key, t) {
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (key(m) < t) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/**
 * Index of the first note to schedule when (re)starting at song time `t`. Pitched notes still sounding at `t` play their
 * remainder, so this is the first note whose end is after `t` (via the running max of ends, since notes are sorted by
 * onset, not end); the scheduler skips any already-ended note after it. Drum hits are instantaneous: first onset >= t.
 */
function firstToSchedule(track, t) {
  if (track.isDrum) return lowerBound(track.notes.length, (i) => track.notes[i][0], t);
  return lowerBound(track.maxEnd.length, (i) => track.maxEnd[i], t + 1e-9);
}

/** Running maximum of note ends (non-decreasing, unlike the ends themselves). */
function runningMaxEnd(notes) {
  const out = new Float64Array(notes.length);
  let m = -Infinity;
  notes.forEach((n, i) => (out[i] = m = Math.max(m, n[1])));
  return out;
}

/** Playback engine for one song. */
export class Player {
  constructor() {
    /** @type {AudioContext} */
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.master = this.ctx.createGain();
    this.master.connect(this.ctx.destination);
    this.originalBus = this.bus();
    this.residualBus = this.bus();
    this.original = null;
    this.residual = null;
    this.stems = [];
    this.tracks = [];
    this.stemBuses = [];
    this.synthBuses = [];
    this.gains = null;
    this.duration = 0;
    this.playing = false;
    this.offset = 0;
    this.startedAt = 0;
    this.sources = [];
    this.synth = [];
    this.timer = 0;
    /** Token of a play() waiting for ctx.resume(); cleared by stop() to cancel it. */
    this.pendingStart = null;
    /** @type {(() => void) | null} called when playback reaches the end */
    this.onEnded = null;
  }

  bus() {
    const g = this.ctx.createGain();
    g.gain.value = 0;
    g.connect(this.master);
    return g;
  }

  /**
   * Set the song: the decoded file (or null for synth-only playback) and the transcribed tracks.
   * @param {{original: AudioBuffer | null, tracks: {program: number, isDrum: boolean, notes: number[][]}[], duration: number}} song
   */
  load({ original, tracks, duration }) {
    this.stop();
    for (const b of [...this.stemBuses, ...this.synthBuses]) b.disconnect();
    this.original = original;
    this.residual = null;
    this.stems = [];
    this.tracks = tracks.map((t) => {
      const notes = [...t.notes].sort((a, b) => a[0] - b[0]);
      return { ...t, notes, maxEnd: runningMaxEnd(notes) };
    });
    this.stemBuses = tracks.map(() => this.bus());
    this.synthBuses = tracks.map(() => this.bus());
    this.synth = tracks.map(() => ({ next: 0, voices: [], on: false }));
    this.duration = duration;
    this.offset = 0;
  }

  /**
   * Install separated stems (one AudioBuffer per track, same order as `load`) and the residual. Restarts if playing.
   * @param {AudioBuffer[]} stems
   * @param {AudioBuffer} residual
   */
  setStems(stems, residual) {
    const wasPlaying = this.playing || !!this.pendingStart;
    const t = this.time;
    this.stop();
    this.stems = stems;
    this.residual = residual;
    this.offset = t;
    if (wasPlaying) this.play();
  }

  /** Master volume, 0..1. */
  setMaster(v) {
    this.master.gain.setTargetAtTime(v, this.ctx.currentTime, RAMP_TAU);
  }

  /**
   * Apply gains from mixGains(); starts / stops synth scheduling for tracks whose synth became (in)audible.
   * @param {import("./mix.js").Gains} g
   */
  setGains(g) {
    const now = this.ctx.currentTime;
    const set = (bus, v) => bus.gain.setTargetAtTime(v, now, RAMP_TAU);
    set(this.originalBus, g.original);
    set(this.residualBus, g.residual);
    g.stems.forEach((v, k) => set(this.stemBuses[k], v));
    g.synths.forEach((v, k) => {
      set(this.synthBuses[k], v);
      const s = this.synth[k];
      const on = v > 0;
      if (on && !s.on && this.playing) s.next = firstToSchedule(this.tracks[k], this.time);
      if (!on && s.on) this.releaseVoices(k, now);
      s.on = on;
    });
    this.gains = g;
  }

  /** Current song position in seconds. */
  get time() {
    if (!this.playing) return this.offset;
    return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
  }

  /** Start playback at the current offset. */
  async play() {
    if (this.playing || this.pendingStart) return;
    const token = {};
    this.pendingStart = token;
    try {
      if (this.ctx.state !== "running") await this.ctx.resume();
    } catch (e) {
      if (this.pendingStart === token) this.pendingStart = null;
      throw e;
    }
    if (this.pendingStart !== token) return; // stopped, paused or restarted while resuming
    this.pendingStart = null;
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    const when = this.ctx.currentTime + 0.03;
    this.startedAt = when;
    this.playing = true;
    const start = (buffer, bus) => {
      if (!buffer || this.offset >= buffer.duration) return;
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(bus);
      src.start(when, this.offset);
      this.sources.push(src);
    };
    start(this.original, this.originalBus);
    start(this.residual, this.residualBus);
    this.stems.forEach((b, k) => start(b, this.stemBuses[k]));
    this.synth.forEach((s, k) => (s.next = firstToSchedule(this.tracks[k], this.offset)));
    this.schedule();
    clearInterval(this.timer);
    this.timer = setInterval(() => this.schedule(), TICK_MS);
  }

  /** Pause at the current position. */
  pause() {
    if (!this.playing) {
      this.pendingStart = null;
      return;
    }
    this.offset = this.time;
    this.stop();
  }

  /** Toggle play / pause. */
  toggle() {
    return this.playing || this.pendingStart ? this.pause() : this.play();
  }

  /** Jump to `t` seconds (keeps playing if playing). */
  seek(t) {
    const wasPlaying = this.playing || !!this.pendingStart;
    this.stop();
    this.offset = Math.max(0, Math.min(this.duration, t));
    if (wasPlaying) this.play();
  }

  stop() {
    this.pendingStart = null;
    clearInterval(this.timer);
    this.timer = 0;
    const now = this.ctx.currentTime;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* not started */
      }
      s.disconnect();
    }
    this.sources = [];
    this.synth.forEach((_, k) => this.releaseVoices(k, now));
    this.playing = false;
  }

  releaseVoices(k, at) {
    for (const v of this.synth[k].voices) v.stop(at);
    this.synth[k].voices = [];
  }

  /** Lookahead scheduler tick: queue synth notes with onsets in [cursor, now + LOOKAHEAD_S] (held notes: their remainder). */
  schedule() {
    if (!this.playing) return;
    const now = this.ctx.currentTime;
    const songNow = this.time;
    if (songNow >= this.duration) {
      this.offset = this.duration;
      this.stop();
      this.onEnded?.();
      return;
    }
    const horizon = songNow + LOOKAHEAD_S + Math.max(0, this.startedAt - now);
    this.synth.forEach((s, k) => {
      s.voices = s.voices.filter((v) => v.end > now);
      if (!s.on) return;
      const tr = this.tracks[k];
      const notes = tr.notes;
      while (s.next < notes.length && notes[s.next][0] < horizon) {
        const [on, off, pitch, vel = 100] = notes[s.next++];
        // Before startedAt the song clock reads below offset; a held note's remainder starts with the audio, not earlier.
        const from = Math.max(on, songNow, this.offset);
        const when = Math.max(now, this.startedAt + (from - this.offset));
        const duration = off - from;
        if (duration <= 0.01 && !tr.isDrum) continue;
        const v = tr.isDrum
          ? playDrum(this.ctx, this.synthBuses[k], { key: pitch, velocity: vel, when })
          : playPitched(this.ctx, this.synthBuses[k], { program: tr.program, pitch, velocity: vel, when, duration });
        s.voices.push(v);
      }
    });
  }
}
