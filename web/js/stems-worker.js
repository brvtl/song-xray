// Stems worker: runs stems.separateStems off the main thread.
//
// in:  { type: "run", channels: Float32Array[], rate, tracks, tuningCents }   (channel buffers transferred)
// out: { type: "progress", done, total } | { type: "result", stems: Float32Array[][], residual: Float32Array[], info }
//      | { type: "error", message }                                         (result buffers transferred)

import { separateStems } from "./stems.js";

self.onmessage = (e) => {
  const m = e.data;
  if (!m || m.type !== "run") return;
  try {
    let last = 0;
    const { stems, residual, info } = separateStems({
      channels: m.channels,
      rate: m.rate,
      tracks: m.tracks,
      tuningCents: m.tuningCents || 0,
      onProgress: (done, total) => {
        const now = performance.now();
        if (done < total && now - last < 150) return;
        last = now;
        self.postMessage({ type: "progress", done, total });
      },
    });
    const transfer = [...stems.flat(), ...residual].map((a) => a.buffer);
    self.postMessage({ type: "result", stems, residual, info }, transfer);
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.stack || err?.message || err) });
  }
};
