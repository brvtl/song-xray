// WebAudio transport: the original file, the stems + residual, and one synth bus per track, all behind a master gain.
// Every buffer source starts at the same `when` with the same offset (a seek restarts all of them). Synth notes are put on
// the audio clock by a lookahead scheduler (every 50 ms, 0.2 s ahead), only for tracks whose synth is audible.

import { playPitched, playDrum } from "./synths.js";

const LOOKAHEAD_S = 0.2;
const TICK_MS = 50;
const RAMP_TAU = 0.015;

/** First index i with notes[i][0] >= t (notes sorted by onset). */
function firstOnsetAtOrAfter(notes, t) {
  let lo = 0;
  let hi = notes.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (notes[m][0] < t) lo = m + 1;
    else hi = m;
  }
  return lo;
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
    this.tracks = tracks.map((t) => ({ ...t, notes: [...t.notes].sort((a, b) => a[0] - b[0]) }));
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
    const wasPlaying = this.playing;
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
      if (on && !s.on && this.playing) s.next = firstOnsetAtOrAfter(this.tracks[k].notes, this.time);
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
    if (this.playing) return;
    if (this.ctx.state !== "running") await this.ctx.resume();
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
    this.synth.forEach((s, k) => (s.next = firstOnsetAtOrAfter(this.tracks[k].notes, this.offset)));
    this.schedule();
    this.timer = setInterval(() => this.schedule(), TICK_MS);
  }

  /** Pause at the current position. */
  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this.stop();
  }

  /** Toggle play / pause. */
  toggle() {
    return this.playing ? this.pause() : this.play();
  }

  /** Jump to `t` seconds (keeps playing if playing). */
  seek(t) {
    const wasPlaying = this.playing;
    this.stop();
    this.offset = Math.max(0, Math.min(this.duration, t));
    if (wasPlaying) this.play();
  }

  stop() {
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

  /** Lookahead scheduler tick: queue synth notes with onsets in [cursor, now + LOOKAHEAD_S]. */
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
        const when = Math.max(now, this.startedAt + (on - this.offset));
        const duration = off - Math.max(on, songNow);
        if (duration <= 0.01 && !tr.isDrum) continue;
        const v = tr.isDrum
          ? playDrum(this.ctx, this.synthBuses[k], { key: pitch, velocity: vel, when })
          : playPitched(this.ctx, this.synthBuses[k], { program: tr.program, pitch, velocity: vel, when, duration });
        s.voices.push(v);
      }
    });
  }
}
