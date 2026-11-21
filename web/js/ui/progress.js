// Progress panel: stage list, bar, elapsed / ETA on the page's own clock (so it never looks frozen between messages),
// and a live preview of the notes transcribed so far.

import { fitCanvas, fmtTime } from "./draw.js";

/** Stages in pipeline order; `match` maps the worker's free-form stage / label strings onto them. */
export const STAGES = [
  { key: "decode", name: "Decode", match: /decod/ },
  { key: "models", name: "Load models", match: /model|loading/ },
  { key: "tuning", name: "Tuning", match: /resampl|tun/ },
  { key: "transcriber", name: "Transcriber", match: /transcri/ },
  { key: "repair", name: "Evidence repair", match: /repair|evidence/ },
  { key: "grouping", name: "Grouping", match: /group/ },
  { key: "drums", name: "Drums", match: /drum/ },
  { key: "stems", name: "Stems", match: /stem/ },
];

/**
 * Stage key for a worker progress message, or null when unknown.
 * @param {{stage?: string, label?: string}} msg
 */
export function stageOf(msg) {
  const s = String(msg.stage ?? "").toLowerCase();
  if (s === "done") return "done";
  for (const st of STAGES) if (st.match.test(s)) return st.key;
  const l = String(msg.label ?? "").toLowerCase();
  for (const st of STAGES) if (st.match.test(l)) return st.key;
  return null;
}

/** Progress panel bound to #progress. */
export class Progress {
  constructor() {
    this.root = document.getElementById("progress");
    this.file = document.getElementById("progress-file");
    this.label = document.getElementById("progress-label");
    this.elapsed = document.getElementById("progress-elapsed");
    this.eta = document.getElementById("progress-eta");
    this.bar = document.getElementById("progress-bar");
    this.list = document.getElementById("stages");
    this.preview = document.getElementById("preview");
    this.items = new Map();
    this.t0 = 0;
    this.timer = 0;
    this.current = null;
    this.etaAt = null;
    this.notes = [];
    this.duration = 0;
  }

  /** Reset and show for a new file. */
  start(fileName, duration) {
    this.root.hidden = false;
    this.root.classList.remove("compact", "finished");
    this.file.textContent = fileName;
    this.duration = duration || 0;
    this.notes = [];
    this.items.clear();
    this.list.replaceChildren(
      ...STAGES.map((s) => {
        const li = document.createElement("li");
        li.className = "stage";
        li.innerHTML = `<span class="dot"></span><span class="txt">${s.name}<span class="sub"></span></span>`;
        this.items.set(s.key, li);
        return li;
      }),
    );
    this.current = null;
    this.etaAt = null;
    this.t0 = performance.now();
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 250);
    this.tick();
    this.drawPreview();
  }

  tick() {
    this.elapsed.textContent = fmtTime((performance.now() - this.t0) / 1000);
    if (this.etaAt == null) this.eta.textContent = "–";
    else this.eta.textContent = fmtTime(Math.max(0, (this.etaAt - performance.now()) / 1000));
  }

  /**
   * Activate a stage (earlier stages become done) and show its counters.
   * @param {string} key
   * @param {{label?: string, done?: number, total?: number, etaS?: number}} [p]
   */
  stage(key, p = {}) {
    const idx = STAGES.findIndex((s) => s.key === key);
    if (idx < 0) return;
    const curIdx = STAGES.findIndex((s) => s.key === this.current);
    // Never move backwards (e.g. a late model-loading message during the transcriber stage).
    if (idx < curIdx) return this.sub(key, p);
    STAGES.forEach((s, i) => {
      const li = this.items.get(s.key);
      if (i < idx && !li.classList.contains("skipped")) li.className = "stage done";
      if (i === idx) li.className = "stage active";
    });
    if (key !== this.current) this.etaAt = null;
    this.current = key;
    const frac = p.total > 0 ? Math.max(0, Math.min(1, p.done / p.total)) : 0;
    this.bar.style.width = `${(frac * 100).toFixed(1)}%`;
    const counter = p.total > 1 ? ` ${p.done} / ${p.total}` : "";
    this.label.textContent = p.label || `${STAGES[idx].name}${counter}`;
    if (Number.isFinite(p.etaS)) this.etaAt = performance.now() + p.etaS * 1000;
    this.sub(key, p);
    this.tick();
  }

  sub(key, p) {
    const li = this.items.get(key);
    if (!li) return;
    const sub = li.querySelector(".sub");
    if (p.total > 1 && p.total <= 100000 && key !== "models") sub.textContent = `${p.done} / ${p.total}`;
    else if (p.total > 1 && key === "models") sub.textContent = `${Math.round((100 * p.done) / p.total)}%`;
  }

  /** Mark a stage done or skipped with a short note. */
  mark(key, state, note = "") {
    const li = this.items.get(key);
    if (!li) return;
    li.className = `stage ${state}`;
    if (note) li.querySelector(".sub").textContent = note;
  }

  /** Handle a worker {type: "progress"} message. */
  message(msg) {
    const key = stageOf(msg);
    if (key === "done") {
      for (const s of STAGES)
        if (s.key !== "stems" && !this.items.get(s.key).classList.contains("skipped")) this.mark(s.key, "done");
      return;
    }
    if (key) this.stage(key, msg);
    else if (msg.label) this.label.textContent = msg.label;
  }

  /** Raw transcriber notes so far ([[on, off, pitch], ...]) for the live preview. */
  partial(notes) {
    if (Array.isArray(notes)) this.notes = notes;
    this.drawPreview();
  }

  drawPreview() {
    const c = this.preview;
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (!w || !h) return;
    const g = fitCanvas(c, w, h);
    g.clearRect(0, 0, w, h);
    const notes = this.notes;
    if (!notes.length) {
      g.fillStyle = "rgba(230,233,240,0.35)";
      g.font = "500 12px system-ui, -apple-system, sans-serif";
      g.textBaseline = "middle";
      g.fillText("Notes appear here as the transcriber works through the song…", 14, h / 2);
      return;
    }
    let lo = 127;
    let hi = 0;
    let end = 0;
    for (const n of notes) {
      lo = Math.min(lo, n[2]);
      hi = Math.max(hi, n[2]);
      end = Math.max(end, n[1]);
    }
    const dur = Math.max(this.duration, end, 1);
    const rows = hi - lo + 3;
    const rh = (h - 10) / rows;
    const grad = g.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "rgba(139,123,255,0.9)");
    grad.addColorStop(1, "rgba(55,214,196,0.9)");
    g.fillStyle = grad;
    for (const n of notes) {
      const x = (n[0] / dur) * w;
      const nw = Math.max(1.5, ((n[1] - n[0]) / dur) * w);
      const y = 5 + (hi + 1 - n[2]) * rh;
      g.fillRect(x, y, nw, Math.max(1.5, rh - 0.5));
    }
    const xf = (end / dur) * w;
    g.fillStyle = "rgba(255,255,255,0.5)";
    g.fillRect(xf, 0, 1, h);
  }

  /** Collapse to a slim strip (results are shown below). */
  compact(on = true) {
    this.root.classList.toggle("compact", on);
  }

  /** Stop the clock; show a final summary line. */
  finish(summary) {
    clearInterval(this.timer);
    this.tick();
    this.etaAt = null;
    this.eta.textContent = "–";
    this.bar.style.width = "100%";
    this.root.classList.add("finished");
    if (summary) this.label.textContent = summary;
  }

  /** Total seconds since start(). */
  get seconds() {
    return (performance.now() - this.t0) / 1000;
  }

  hide() {
    clearInterval(this.timer);
    this.root.hidden = true;
  }
}
