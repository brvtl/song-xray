# Architecture

Song X-ray runs the four Dettato models (transcriber, evidence repairer, grouper, drums) in a browser tab. The neural networks are
ONNX graphs executed by ONNX Runtime Web (WASM, multi-threaded when the page is cross-origin isolated). Everything else
(spectrograms, filterbanks, peak picking, the repairer's edit logic, the grouper's clustering, the drum decode) is a JavaScript
port of the reference Python in `vendor/dettato/src/dettato/` (pinned commit, see `scripts/setup.sh`).

```
 drop mp3/wav ─► decodeAudioData (page) ─► mono Float32 at the context rate
                         │
                         ▼   Web Worker (web/js/worker.js) — DOM-free modules, also run under Node for the parity check
   resample → 16 kHz ─► tuning (dsp.retuneRatio) ─► retune resample
                         │
                         ▼
   transcriber windows (8 s, every 2 s): dsp.hcqt ─► transcriber.onnx ─► Hann-weighted overlap-add
        maps (all windows + every-other-window), taps hidden / sub-hidden / sub-stem in a bounded float16 TapStore
                         │   (per 8 s block, as soon as its frames are final)
                         ▼
   transcriber.notes ─► evidence (map numbers + hidden gathers) ─► repairer.onnx ─► evidence.apply
                         ─► grouper.gather (meta + int8 taps per note)
                         │   (after the last block)
                         ▼
   grouper windows (24 s, every 12 s): dsp.grouperFront ─► grouper.onnx ─► grouper.decode (clustering, naming)
   drums (source ≥ 40 kHz): dsp.drumFeatures ─► drums/*.onnx (+ chunked attn.onnx) ─► drums.decode
                         │
                         ▼
   result {tracks:[{name, program, isDrum, notes:[[on, off, pitch, vel]]}]}  ─► page: piano rolls, stems (score-informed masks),
                                                                               WebAudio synths, mute / solo / synth swap
```

## Module contracts (web/js/)

All modules are ES modules without DOM access. Arrays are row-major typed arrays; shapes are given as `[dims]`.

`rt` (the runtime handle) is `{ ort, session(name) -> Promise<InferenceSession> }`, where `ort` is the onnxruntime-web namespace
(used for `new ort.Tensor`) and `name` is a model path under `web/models/` without `.onnx` (`"transcriber"`, `"drums/kit_embed"`).
The page and the Node tools build it differently; the modules never import onnxruntime themselves.

| module                     | exports (main)                                                                        | reference                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `f16.js`                   | `toF16`, `fromF16`, `roundF16`, `F16_TABLE`                                           | —                                                                                                                              |
| `dsp.js`                   | `FFT`, `stftPower`, `hcqt`, `grouperFront`, `drumFeatures`, `resample`, `retuneRatio` | `transcriber/frontend.py`, `grouper/front.py` (Front, pcen, standardise), `drums/features.py`, `grouper/retune.py`, `audio.py` |
| `transcriber.js`           | `layout`, `decode`, `notes`                                                           | `transcriber/core.py`, `transcriber/decode.py`                                                                                 |
| `evidence.js`              | `mapEvidence`, `gatherHidden`, `assemble`, `inputNotes`, `repairWindows`, `apply`     | `evidence/features.py`, `evidence/repair.py`, `evidence/model.py:candidates`                                                   |
| `grouper.js`, `cluster.js` | `assignSlots`, `gather`, `songWindows`, `predict`, `decode`, `tracks`                 | `grouper/core.py`, `grouper/decode.py`                                                                                         |
| `drums.js`                 | `drumProbabilities`, `drumDecode`, `transcribeDrums`                                  | `drums/infer.py`, `drums/model.py` (split graphs)                                                                              |
| `tapstore.js`              | `TapStore`                                                                            | `transcriber/core.py:_Average`                                                                                                 |
| `pipeline.js`              | `transcribe(rt, audio, rate, opts)`                                                   | `pipeline.py`                                                                                                                  |
| `stems.js`                 | `separateStems`                                                                       | new (not in Dettato)                                                                                                           |

### Maps and taps

- `maps`: `{ onset, frame, offset, onset2, frame2, offset2 }`, each `Float32Array [T][88]`, `T = ceil(n / 16000 / 0.02)`.
- Tap vectors are 368-d. A tap store exposes `vec(tap, t, lane, out?) -> Float32Array(368)` for `tap` in
  `"hidden"` (average over all windows), `"subHidden"`, `"subStem"` (average over every other window, i.e. windows whose padded
  start is a multiple of 4 s), for any frame `t` it still holds. Values are float16-rounded, as in the reference.

### Result format

```js
{
  seconds, sourceRate,
  tracks: [{ name: "piano", program: 0, isDrum: false, notes: [[onsetS, offsetS, midiPitch, velocity1to127], ...] }, ...],
  info: { tuningCents, notes: { transcribed, dropped, inserted, grouped }, drums: {...}, timings: {...} }
}
```

## Fixtures and tests

`scripts/dump_fixtures.py clips/X.wav [--taps]` runs the PyTorch reference stage by stage and writes `fixtures/X/*.npy` plus
`manifest.json`. `tests/*.test.mjs` (`npm test`) feed each JS stage the reference's own input and compare its output with the
reference's, so a mismatch points at one stage. `parity/parity.py` is the end-to-end check.
