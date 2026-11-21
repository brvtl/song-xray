// The lanes: a time ruler, an "Original" waveform row, one piano-roll row per track, and a "Residual" row.
// Each row has a coloured header (name, count, Mute / Solo / Synth, volume) and a viewport-sized canvas.

import { displayName } from "./gm.js";
import {
  PALETTE,
  DRUM_COLOR,
  ORIGINAL_COLOR,
  RESIDUAL_COLOR,
  rgba,
  fitCanvas,
  laneLayout,
  drawNoteLane,
  drawWaveLane,
  drawRuler,
} from "./draw.js";

const WAVE_H = 64;
const RULER_H = 26;

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k in e && typeof v !== "string") e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

/** Paint a range input's filled part (CSS var --fill). */
export function paintRange(input) {
  const f = (Number(input.value) - Number(input.min || 0)) / (Number(input.max || 1) - Number(input.min || 0));
  input.style.setProperty("--fill", `${(f * 100).toFixed(1)}%`);
}

/** Colour of the k-th track. */
export function trackColor(track, k) {
  return track.isDrum ? DRUM_COLOR : PALETTE[k % PALETTE.length];
}

/** All lane rows of the results view. */
export class Lanes {
  /**
   * @param {HTMLElement} root
   * @param {import("./view.js").View} view
   * @param {{onSeek: (t: number) => void, onToggle: (id: string | number, key: "muted"|"soloed"|"synth") => void,
   *   onVolume: (id: string | number, v: number) => void}} handlers
   */
  constructor(root, view, handlers) {
    this.root = root;
    this.view = view;
    this.handlers = handlers;
    this.rows = new Map();
    this.ruler = null;
    this.peaks = { original: null, residual: null };
    this.residualMessage = "";
    this.dirty = true;
    new ResizeObserver(() => this.resize()).observe(root);
    view.onChange(() => (this.dirty = true));
  }

  /**
   * Build rows for a result.
   * @param {{tracks: {name: string, program: number, isDrum: boolean, notes: number[][]}[], hasAudio: boolean,
   *   sourceRate?: number}} song
   */
  build({ tracks, hasAudio, sourceRate }) {
    this.root.replaceChildren();
    this.rows.clear();
    const rulerCanvas = el("canvas", { class: "ruler" });
    this.root.append(
      el(
        "div",
        { class: "lane ruler-row" },
        el("div", { class: "lane-head" }, "Tracks"),
        el("div", { class: "lane-roll" }, rulerCanvas),
      ),
    );
    this.ruler = rulerCanvas;
    this.attachPointer(rulerCanvas);

    if (hasAudio) {
      const rate = sourceRate ? `${(sourceRate / 1000).toFixed(sourceRate % 1000 ? 1 : 0)} kHz source` : "source audio";
      this.addRow("original", { name: "Original mix", meta: rate, color: ORIGINAL_COLOR, kind: "wave", synth: false });
    }
    tracks.forEach((t, k) => {
      const notes = [...t.notes].sort((a, b) => a[0] - b[0]);
      const maxDur = notes.reduce((m, n) => Math.max(m, n[1] - n[0]), 0.12);
      const n = notes.length;
      const meta = t.isDrum ? `${n} hit${n === 1 ? "" : "s"}` : `${n} note${n === 1 ? "" : "s"}`;
      this.addRow(k, {
        name: t.isDrum ? "Drums" : displayName(t.name),
        meta,
        title: t.isDrum ? "GM drum kit (channel 10)" : `GM program ${t.program}`,
        color: trackColor(t, k),
        kind: "notes",
        track: { notes, maxDur },
        layout: laneLayout({ isDrum: t.isDrum, notes }),
        synth: true,
      });
    });
    if (hasAudio) {
      this.addRow("residual", {
        name: "Residual",
        meta: "what no track explains",
        color: RESIDUAL_COLOR,
        kind: "wave",
        synth: false,
      });
    }
    this.resize();
  }

  addRow(id, spec) {
    const canvas = el("canvas", { class: "roll" });
    const btn = (cls, label, key, title) => {
      const b = el("button", { class: `tbtn ${cls}`, type: "button", title, "aria-pressed": "false" }, label);
      b.addEventListener("click", () => this.handlers.onToggle(id, key));
      return b;
    };
    const vol = el("input", {
      type: "range",
      min: "0",
      max: "1",
      step: "0.01",
      value: "1",
      "aria-label": `${spec.name} volume`,
    });
    vol.style.setProperty("--c", rgba(spec.color));
    paintRange(vol);
    vol.addEventListener("input", () => {
      paintRange(vol);
      this.handlers.onVolume(id, Number(vol.value));
    });
    const buttons = {
      muted: btn("m", "M", "muted", "Mute"),
      soloed: btn("s", "S", "soloed", "Solo (several allowed)"),
      synth: spec.synth ? btn("y", "Synth", "synth", "Swap this track's audio for a synth following its notes") : null,
    };
    const status = el("div", { class: "lane-status" });
    status.hidden = true;
    const head = el(
      "div",
      { class: "lane-head" },
      el(
        "div",
        { class: "lane-title" },
        el("span", { class: "lane-name", title: spec.title ?? spec.name }, spec.name),
        el("span", { class: "lane-meta" }, spec.meta),
      ),
      el("div", { class: "lane-ctrls" }, buttons.muted, buttons.soloed, buttons.synth, vol),
      status,
    );
    const row = el(
      "div",
      { class: "lane", style: `--c: ${rgba(spec.color)}` },
      head,
      el("div", { class: "lane-roll" }, canvas),
    );
    this.root.append(row);
    this.attachPointer(canvas);
    this.rows.set(id, {
      ...spec,
      id,
      row,
      canvas,
      buttons,
      vol,
      status,
      height: spec.kind === "wave" ? WAVE_H : spec.layout.height,
    });
  }

  /** Seek on click / drag; ctrl/cmd + wheel zooms at the cursor, horizontal or shift + wheel scrolls. */
  attachPointer(canvas) {
    let down = false;
    let last = 0;
    const seekAt = (e) => {
      const r = canvas.getBoundingClientRect();
      this.handlers.onSeek(this.view.time(e.clientX - r.left));
    };
    canvas.addEventListener("pointerdown", (e) => {
      down = true;
      canvas.setPointerCapture(e.pointerId);
      seekAt(e);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!down || performance.now() - last < 60) return;
      last = performance.now();
      seekAt(e);
    });
    const up = () => (down = false);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener(
      "wheel",
      (e) => {
        const r = canvas.getBoundingClientRect();
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          this.view.zoomBy(Math.exp(-e.deltaY * 0.0025), e.clientX - r.left);
        } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
          e.preventDefault();
          const d = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
          this.view.scrollBy(d / this.view.pps);
        }
      },
      { passive: false },
    );
  }

  /**
   * Reflect mixer state on a row.
   * @param {string | number} id
   * @param {{muted?: boolean, soloed?: boolean, synth?: boolean, audible?: boolean, synthDisabled?: boolean}} s
   */
  setRowState(id, s) {
    const r = this.rows.get(id);
    if (!r) return;
    for (const key of ["muted", "soloed", "synth"]) r.buttons[key]?.setAttribute("aria-pressed", String(!!s[key]));
    r.row.classList.toggle("silent", s.audible === false);
  }

  /** Show a status line (with spinner) under a row's controls, or hide it with an empty message. */
  setRowStatus(id, message, busy = false) {
    const r = this.rows.get(id);
    if (!r) return;
    r.status.hidden = !message;
    r.status.replaceChildren(...(busy ? [el("span", { class: "spinner" })] : []), message);
  }

  /** Waveform peaks for "original" / "residual". */
  setPeaks(id, peaks) {
    this.peaks[id] = peaks;
    this.dirty = true;
  }

  /** Text drawn in the residual lane until its audio exists. */
  setResidualMessage(m) {
    this.residualMessage = m;
    this.dirty = true;
  }

  resize() {
    const anyRoll = this.root.querySelector(".lane-roll");
    if (!anyRoll) return;
    this.view.setWidth(anyRoll.clientWidth);
    this.dirty = true;
  }

  /**
   * Redraw every row (cheap: canvases are viewport-sized and notes are binary-searched).
   * @param {number} time playhead
   * @param {boolean} playing
   */
  render(time, playing) {
    const view = this.view;
    if (this.ruler) drawRuler(fitCanvas(this.ruler, view.width, RULER_H), view, RULER_H, time);
    for (const r of this.rows.values()) {
      const g = fitCanvas(r.canvas, view.width, r.height);
      r.canvas.style.height = `${r.height}px`;
      if (r.kind === "wave") {
        const peaks = this.peaks[r.id];
        const message = r.id === "residual" ? this.residualMessage : "";
        drawWaveLane(g, { view, h: r.height, peaks, color: r.color, time, message });
      } else {
        drawNoteLane(g, {
          view,
          h: r.height,
          track: r.track,
          layout: r.layout,
          color: r.color,
          time,
          highlight: playing || time > 0,
        });
      }
    }
    this.dirty = false;
  }
}
