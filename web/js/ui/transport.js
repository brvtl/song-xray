// Transport bar: play / pause, time, a whole-song scrubber (overview of the waveform or note density, the visible
// window, the playhead), the "what you hear" pill, master volume and the zoom slider.

import { fitCanvas, fmtTime, rgba, ORIGINAL_COLOR } from "./draw.js";
import { paintRange } from "./lanes.js";

/** Transport controls bound to existing DOM elements (see index.html). */
export class Transport {
  /**
   * @param {import("./view.js").View} view
   * @param {{onToggle: () => void, onSeek: (t: number) => void, onMaster: (v: number) => void}} handlers
   */
  constructor(view, handlers) {
    this.view = view;
    this.playBtn = document.getElementById("play");
    this.now = document.getElementById("time-now");
    this.total = document.getElementById("time-total");
    this.scrubber = document.getElementById("scrubber");
    this.mode = document.getElementById("mode");
    this.master = document.getElementById("master");
    this.zoom = document.getElementById("zoom");
    this.overview = null;
    this.duration = 1;

    this.playBtn.addEventListener("click", () => handlers.onToggle());
    paintRange(this.master);
    this.master.addEventListener("input", () => {
      paintRange(this.master);
      handlers.onMaster(Number(this.master.value));
    });
    this.zoom.addEventListener("input", () => {
      const t = view.time(view.width / 2);
      view.setZoom(Number(this.zoom.value) / 1000, t, view.width / 2);
    });
    view.onChange(() => {
      this.zoom.value = String(Math.round(view.zoom * 1000));
      paintRange(this.zoom);
    });
    paintRange(this.zoom);

    let down = false;
    const seek = (e) => {
      const r = this.scrubber.getBoundingClientRect();
      handlers.onSeek(((e.clientX - r.left) / r.width) * this.duration);
    };
    this.scrubber.addEventListener("pointerdown", (e) => {
      down = true;
      this.scrubber.setPointerCapture(e.pointerId);
      seek(e);
    });
    this.scrubber.addEventListener("pointermove", (e) => down && seek(e));
    this.scrubber.addEventListener("pointerup", () => (down = false));
    this.scrubber.addEventListener("pointercancel", () => (down = false));
  }

  /**
   * Song length and an overview: Float32Array of 0..1 amplitude per column (any length).
   * @param {number} duration
   * @param {Float32Array} overview
   */
  setSong(duration, overview) {
    this.duration = Math.max(0.1, duration);
    this.overview = overview;
    this.total.textContent = fmtTime(duration);
  }

  /** The "what you hear" label from mixGains(). */
  setMode(mode) {
    this.mode.textContent = mode;
    this.mode.classList.toggle("stems", mode.startsWith("Stems"));
    this.mode.classList.toggle("synth", mode.startsWith("Synth"));
  }

  /** Redraw time, play state and scrubber. */
  render(time, playing) {
    this.playBtn.classList.toggle("playing", playing);
    this.playBtn.setAttribute("aria-label", playing ? "Pause (space)" : "Play (space)");
    this.now.textContent = fmtTime(time, 1);
    const w = this.scrubber.clientWidth;
    const h = this.scrubber.clientHeight;
    if (!w || !h) return;
    const g = fitCanvas(this.scrubber, w, h);
    g.clearRect(0, 0, w, h);
    g.fillStyle = "#0d1016";
    g.beginPath();
    g.roundRect(0, 0, w, h, 8);
    g.fill();
    const xPlay = (time / this.duration) * w;
    const ov = this.overview;
    if (ov?.length) {
      const mid = h / 2;
      for (let x = 0; x < w; x += 2) {
        const i = Math.min(ov.length - 1, Math.floor((x / w) * ov.length));
        const a = Math.max(0.03, ov[i]) * (h / 2 - 5);
        g.fillStyle = rgba(ORIGINAL_COLOR, x < xPlay ? 0.85 : 0.35);
        g.fillRect(x, mid - a, 1.25, 2 * a);
      }
    }
    const v0 = (this.view.t0 / this.duration) * w;
    const v1 = (this.view.t1 / this.duration) * w;
    if (v1 - v0 < w - 1) {
      g.fillStyle = "rgba(139,123,255,0.14)";
      g.strokeStyle = "rgba(139,123,255,0.65)";
      g.lineWidth = 1;
      g.beginPath();
      g.roundRect(v0 + 0.5, 0.5, Math.max(4, v1 - v0) - 1, h - 1, 6);
      g.fill();
      g.stroke();
    }
    g.fillStyle = "#fff";
    g.fillRect(Math.round(xPlay) - 1, 2, 2, h - 4);
  }
}
