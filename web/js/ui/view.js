// Shared horizontal view (time window) of all lanes: start time, pixels per second, width. DOM-free.

const MAX_PPS = 800;

/** Time window shared by the ruler, the piano rolls and the scrubber. */
export class View {
  constructor() {
    this.duration = 1;
    this.width = 800;
    this.t0 = 0;
    this.pps = 800;
    this.listeners = new Set();
  }

  /** Pixels per second at which the whole song fits the width. */
  get fitPps() {
    return this.width / Math.max(0.1, this.duration);
  }

  /** Seconds visible. */
  get span() {
    return this.width / this.pps;
  }

  /** Right edge time. */
  get t1() {
    return this.t0 + this.span;
  }

  /** Zoom as 0 (fit) .. 1 (max), log-scaled. */
  get zoom() {
    const lo = Math.log(this.fitPps);
    const hi = Math.log(Math.max(MAX_PPS, this.fitPps));
    return hi > lo ? (Math.log(this.pps) - lo) / (hi - lo) : 0;
  }

  onChange(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit() {
    for (const fn of this.listeners) fn(this);
  }

  /** Set the song length; resets to the fitted view. */
  setDuration(d) {
    this.duration = Math.max(0.1, d);
    this.t0 = 0;
    this.pps = this.fitPps;
    this.emit();
  }

  /** Set the drawable width (CSS px), keeping the zoom relative to "fit". */
  setWidth(w) {
    if (!(w > 0) || w === this.width) return;
    const z = this.zoom;
    this.width = w;
    this.setZoom(z, this.t0, 0);
  }

  clamp() {
    this.pps = Math.min(Math.max(MAX_PPS, this.fitPps), Math.max(this.fitPps, this.pps));
    this.t0 = Math.max(0, Math.min(this.duration - this.span, this.t0));
    if (this.span >= this.duration) this.t0 = 0;
  }

  /** x (CSS px) of time t. */
  x(t) {
    return (t - this.t0) * this.pps;
  }

  /** Time at x (CSS px). */
  time(x) {
    return this.t0 + x / this.pps;
  }

  /**
   * Set zoom 0..1, keeping `anchorT` at pixel `anchorX`.
   * @param {number} z
   * @param {number} anchorT
   * @param {number} anchorX
   */
  setZoom(z, anchorT, anchorX) {
    const lo = Math.log(this.fitPps);
    const hi = Math.log(Math.max(MAX_PPS, this.fitPps));
    this.pps = Math.exp(lo + Math.max(0, Math.min(1, z)) * (hi - lo));
    this.t0 = anchorT - anchorX / this.pps;
    this.clamp();
    this.emit();
  }

  /** Multiply pixels-per-second by `factor` around pixel x. */
  zoomBy(factor, x) {
    const t = this.time(x);
    this.pps *= factor;
    this.clamp();
    this.t0 = t - x / this.pps;
    this.clamp();
    this.emit();
  }

  /** Scroll by dt seconds. */
  scrollBy(dt) {
    this.t0 += dt;
    this.clamp();
    this.emit();
  }

  /** Keep the playhead visible while playing: page forward when it nears the right edge. Returns true if moved. */
  follow(t) {
    if (this.span >= this.duration) return false;
    if (t >= this.t0 && t <= this.t0 + this.span * 0.92) return false;
    this.t0 = t - this.span * 0.08;
    this.clamp();
    this.emit();
    return true;
  }
}
