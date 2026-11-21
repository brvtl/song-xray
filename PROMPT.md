# Song X-ray in a tab

Build a finished, demo-ready project in this directory. Work here only. Commit as you go with clear messages. Do not create a GitHub repo, do not add a remote, do not push, and do not post anything anywhere.

## Goal

A static webpage you can open offline (even with Wi-Fi off after the first load of local assets) where you drop in any song (mp3 or wav) and get a per-instrument + drums piano-roll transcription, all running inside the browser tab. The user can mute or solo each track and swap a track for a simple WebAudio synth.

This is meant to be the first fully in-tab version of Dettato-style per-instrument + drums transcription. Dettato itself is Python-only today: https://github.com/kobimusic/dettato (Apache-2.0, about 28M parameters). Keep the Dettato license and credit visible in the README and in the UI.

## Hard constraints

- CPU only. Never use the GPU. `CUDA_VISIBLE_DEVICES` is already set to empty. If you install PyTorch, use CPU wheels only. Do not call CUDA, do not use onnxruntime-gpu, do not touch nvidia-smi to "just try" the 5090, and do not enable WebGPU if the only way to get it is a GPU conversion step that needs the 5090. ONNX Runtime Web WASM is the safe default. WebGPU in the *browser* on someone else's machine later is fine to support as an optional path, but all conversion, export, and parity checks on this machine must run on CPU.
- Stay in `/home/peter/code/build-in-public/song-xray`. Do not touch other projects or other tmux sessions.
- The page must work as static files (no required backend at demo time). A short local static server for development is fine (`npx serve`, Python http.server, etc.).
- Be honest in the README about quality, speed, browser limits, and anything that did not work.

## What to build

1. **Export / convert Dettato for the browser.** Get the official Dettato code and weights, then export a browser-runnable model (ONNX preferred; transformers.js is also fine if it is a better fit). Document the conversion steps in the repo so they can be re-run on CPU. Keep the conversion script in the repo. Cache or vendor the converted artifacts locally; do not require a network call at demo time if you can avoid it. If weights are large, document how to obtain them and gitignore the raw weights, but make sure a converted artifact the demo needs is either in the repo or produced by a one-command CPU script.

2. **In-tab inference.** After a user drops an mp3/wav onto the page, decode the audio in the browser, run Dettato (ONNX Runtime Web WASM, or transformers.js) fully in the tab, and render a piano roll per instrument plus drums. Show progress so the page does not look frozen. Handle long-ish songs reasonably (chunking is fine; say so in the README).

3. **Playback controls.** Mute and solo each track. Swap a track for a simple WebAudio synth that follows that track's piano roll (a basic oscillator or sampled piano/drum hit is enough; it does not need to sound like the original instrument). The original mix should still be playable, with muted stems silent and solo meaning "only this track."

4. **Parity check.** Add a script (Python, CPU) that runs the official Dettato reference on 2–3 short clips and compares that output to the browser/ONNX (or the exported model run under onnxruntime CPU) output. Print a clear pass/fail-style report: agreement rate, any systematic offsets, and where they diverge. Include the clips or a script to generate/download short legal/demo clips. It is fine if this comparison runs outside the browser against the same ONNX file the page uses, as long as you also sanity-check that the page's preprocessing matches.

5. **Demo-ready UI.** One clean screen: drop zone, piano rolls, mute/solo/synth controls, and credit for Dettato. It should look good in a screen recording. No login, no cloud API for inference.

6. **README.** Setup (`mise install`, how to convert if needed, how to serve the page, how to run the parity script), what's new, honest limitations (speed on WASM, instrument coverage, accuracy vs the Python original, browsers tested), license/credit for Dettato (Apache-2.0) and this project's license, and a short note that this is a local demo, not a hosted service.

`mise.toml` is already pinned (Python 3.13.18, uv 0.13.6, Node 22.23.4, `CUDA_VISIBLE_DEVICES=""`). Use it. Create a `.venv` via mise/uv. Do not install CUDA or GPU torch.

## Done when

- [ ] Dropping an mp3 or wav into the static page produces per-instrument + drums piano rolls without a server-side model.
- [ ] The page works offline once its local assets are present (no inference API).
- [ ] Mute, solo, and "swap this track for a simple synth" all work.
- [ ] A CPU parity script has been run on 2–3 clips and the README (or a small report file) records the results honestly.
- [ ] Dettato is credited and its Apache-2.0 license is preserved.
- [ ] README is complete and honest.
- [ ] The work is committed in this git repo (no remote, no push).

If something is blocked (export fails, weights missing, accuracy is poor), still leave a working UI, the conversion attempt, the parity script, and an honest README of what worked and what did not. Prefer a slightly slower WASM path that actually runs over a broken GPU path.

When you are finished, print a short summary: what you built, how to run the demo, parity results, and any remaining caveats.
