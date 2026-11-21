# Song X-ray in a tab

Drop a song (mp3 or wav) onto a web page and get a **per-instrument + drums piano-roll transcription computed entirely
inside the browser tab**. You can then mute or solo each track, or swap any track for a simple WebAudio synth that plays its
notes.

There is no server-side model and no inference API. The page is static files. After `scripts/setup.sh` has put the converted
models next to it, it works with the network switched off.

The transcription is [**Dettato**](https://github.com/kobimusic/dettato) by KobiMusic (Apache-2.0, 28.1M parameters, four
models). Dettato itself is Python/PyTorch only. Its four networks are converted here to ONNX, its non-neural processing is ported
to JavaScript, and everything runs with ONNX Runtime Web (WASM). As far as I know this is the first fully in-tab version of
Dettato-style per-instrument + drums transcription.

> This is a local demo, not a hosted service. It is built to run on your own machine from a local static server.

<!-- RESULTS -->

## What's new here

- **An in-browser Dettato.** The four networks (transcriber, evidence repairer, grouper, drum models) are exported to ONNX, and
  all of the reference's glue is ported to plain ES modules in `web/js/`: the HCQT / PCEN / mel front ends with exact FFTs
  (including the drum features' 1411- and 6144-point DFTs), peak picking, the evidence features, the repairer's edit logic, the
  grouper's clustering and naming, and the drum decode. It runs in a Web Worker.
- **A streaming transcriber pass with bounded memory.** The reference keeps whole-song float16 tap buffers (about 10 MB per
  second of audio). The browser version finishes each 8 s block as soon as no later window can touch it, so it holds only a
  sliding ~26 s region (about 250 MB).
- **Chunked whole-song drum attention.** The drum networks are split around their song-level attention, which runs on query
  chunks. A 4-minute song never materialises a T×T matrix in wasm32 memory.
- **Rough stems from the score.** Dettato transcribes but does not separate audio. To make "mute this instrument in the
  original mix" possible, the page builds score-informed soft masks from each track's notes: harmonic combs for pitched tracks,
  and a percussive (HPSS) estimate gated to the drum hits. Unexplained energy goes to a _Residual_ stem. All stems plus the
  residual add back up to the original exactly.
- **A parity harness.** `parity/parity.py` runs the official PyTorch Dettato and this JS/ONNX pipeline on the same clips and
  reports agreement stage by stage.

## Setup

Requirements: [mise](https://mise.jdx.dev), git, ffmpeg (only for the parity script's mp3 variant), about 1.5 GB of disk. CPU
only: nothing here installs or uses CUDA or GPU wheels.

```bash
scripts/setup.sh      # toolchain (mise), .venv with CPU-only PyTorch, Dettato code + weights (pinned), ONNX export, ORT web assets, clips
npm run serve         # http://localhost:8080  (static files + COOP/COEP headers)
```

`scripts/setup.sh` is idempotent. It runs these steps, each of which can also be run by hand:

| step         | command                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| toolchain    | `mise install` (Python 3.13.18, uv 0.13.6, Node 22.23.4, `CUDA_VISIBLE_DEVICES=""`)                      |
| Python env   | `uv venv .venv` + CPU wheels from `https://download.pytorch.org/whl/cpu`                                 |
| Dettato code | `git clone https://github.com/kobimusic/dettato vendor/dettato` @ `9b72425`                              |
| weights      | Hugging Face `kobimusic/dettato{,-evidence,-grouper,-drums}` at pinned revisions → `weights/`            |
| convert      | `DETTATO_WEIGHTS=weights .venv/bin/python scripts/export_onnx.py` → `web/models/` (~130 MB, CPU, ~1 min) |
| ORT assets   | `node scripts/copy_ort.mjs` → `web/vendor/ort/`                                                          |
| clips        | `.venv/bin/python scripts/make_clips.py` → `clips/*.wav`                                                 |

Raw weights, converted `.onnx` files and the ORT WASM binaries are gitignored. They are large, and the export regenerates them
on CPU with one command.

### Serving

The page needs a static server that sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. That makes the page cross-origin isolated, which enables `SharedArrayBuffer`, which
ONNX Runtime Web needs for multi-threaded WASM. `npm run serve` (`scripts/serve.mjs`, no dependencies) does this. A plain
`python -m http.server` also works but runs single-threaded and about 5× slower. The page shows which one you got in the badge
at the top right.

Opening `web/index.html` from `file://` does not work, because browsers do not let a `file://` page load WASM or workers that
way.

URL options: `?fast=1` (fast mode on), `?mock=1` (UI with a canned result, no inference), `?webgpu=1` (optional WebGPU
execution provider, untested, see limitations).

## How it works

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module contracts. In short:

```
decodeAudioData at the file's own rate ─► mono ─► resample 16 kHz ─► tuning estimate ─► (retune)
  ─► transcriber: 8 s windows every 2 s, HCQT in JS ─► transcriber.onnx ─► Hann overlap-add (maps + float16 taps)
       per 8 s block, once final: notes ─► evidence (769 numbers / note) ─► repairer.onnx ─► kept / moved / inserted notes
                                  ─► grouper features (12 scalars + 3×368 int8 taps / note)
  ─► grouper: 24 s windows every 12 s, PCEN HCQT in JS ─► grouper.onnx ─► clustering + naming (73 instruments) ─► tracks
  ─► drums (source ≥ 40 kHz): 280 mel/log bands in JS ─► drum-kit net + 3 orchestral experts (+ router) ─► GM drum hits
  ─► page: piano rolls · stems (score-informed masks, Web Worker) · WebAudio player · synths · MIDI / JSON export
```

Long songs are handled by streaming, not chunking. The transcriber pass is one continuous sequence of overlapping windows, as in
the reference, and only the memory is bounded.

## Parity check

```bash
.venv/bin/python parity/parity.py                    # 3 clips + an mp3 and a detuned variant; writes parity/REPORT.md
node tests/e2e/browser.mjs clips/band.wav            # the real page in headless Chromium -> parity/out/band_wav/browser.json
.venv/bin/python parity/parity.py clips/band.wav --browser --no-variants   # include that browser result in the report
```

Per-module tests against the reference's own intermediate arrays:

```bash
.venv/bin/python scripts/dump_fixtures.py clips/band.wav --taps   # (and duet, piano) -> fixtures/
npm test
```

<!-- PARITY -->

## Limitations

<!-- LIMITATIONS -->

## License and credit

- **Dettato** by KobiMusic (Nathan Mills): code and model weights under the Apache License 2.0.
  <https://github.com/kobimusic/dettato> · <https://huggingface.co/collections/kobimusic/dettato>. The converted ONNX models
  are derived from those weights and are covered by the same license. The JavaScript in `web/js/` that ports Dettato's
  processing is a derivative of its Python source. See [NOTICE](NOTICE), [licenses/dettato-LICENSE](licenses/dettato-LICENSE)
  and [licenses/dettato-NOTICE](licenses/dettato-NOTICE). Dettato's model cards note that some of the grouper's and the drum
  model's training datasets carry non-commercial terms. Read them before commercial use.
- **This project**: Apache License 2.0 ([LICENSE](LICENSE)).
- **ONNX Runtime Web** (Microsoft): MIT ([licenses/onnxruntime-web-LICENSE](licenses/onnxruntime-web-LICENSE)), loaded
  locally from `web/vendor/ort/`.
- The demo clips in `clips/` are synthesised by `scripts/make_clips.py` (no third-party audio).

If you use Dettato, cite it:

```bibtex
@software{dettato2026,
  title  = {Dettato: multi-instrument audio-to-MIDI transcription},
  author = {Mills, Nathan},
  year   = {2026},
  url    = {https://github.com/kobimusic/dettato}
}
```
