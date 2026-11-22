#!/usr/bin/env bash
# One command from a fresh clone to a runnable demo (CPU only; no GPU, no CUDA wheels):
#
#   scripts/setup.sh            # toolchain, Python venv (CPU torch), Dettato code + weights, ONNX export, ORT web assets, clips
#
# Re-running is safe: finished steps are skipped. Needs network the first time (PyPI, GitHub, Hugging Face, npm); afterwards the
# page itself needs no network at all.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""

# Pinned upstream revisions (the versions this demo was converted and parity-checked against).
DETTATO_REPO="https://github.com/kobimusic/dettato"
DETTATO_COMMIT="9b724251794f2bbd5eaa43e54734e5a251829130"
declare -A HF_REVS=(
  [dettato]="a0d1e354d282c4ed458b83b287065587979b053e"
  [dettato-evidence]="5633f28ee330a48bc31bdd3f9f98bb1a71206141"
  [dettato-grouper]="5a94cea2329f53228d7bcf1fd6a9f8d9778cc75b"
  [dettato-drums]="97801f2328e32e4c87b12586aa81121f62cacacb"
)

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "toolchain (mise: Python 3.13, uv, Node 22)"
mise trust --quiet "$PWD/mise.toml"   # a fresh clone's mise.toml is untrusted until this
mise install

step "Python venv with CPU-only PyTorch"
[ -x .venv/bin/python ] || mise exec -- uv venv .venv --python 3.13
mise exec -- uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
mise exec -- uv pip install --python .venv/bin/python numpy scipy soundfile soxr pretty_midi huggingface_hub safetensors \
  onnx onnxruntime onnxscript ruff

step "Dettato reference code @ ${DETTATO_COMMIT:0:7}"
if [ ! -d vendor/dettato/.git ]; then
  git clone --quiet "$DETTATO_REPO" vendor/dettato
fi
git -C vendor/dettato fetch --quiet origin "$DETTATO_COMMIT" 2>/dev/null || true
git -C vendor/dettato checkout --quiet "$DETTATO_COMMIT"
mise exec -- uv pip install --python .venv/bin/python --no-deps -e vendor/dettato

step "Dettato weights (Hugging Face, Apache-2.0)"
for repo in "${!HF_REVS[@]}"; do
  if [ ! -f "weights/$repo/model.safetensors" ]; then
    .venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('kobimusic/$repo', revision='${HF_REVS[$repo]}', local_dir='weights/$repo')"
  fi
done

step "npm packages (onnxruntime-web, dev tools)"
mise exec -- npm ci --no-audit --no-fund

step "ONNX export (CPU) -> web/models/"
if [ ! -f web/models/meta.json ] || [ "${FORCE_EXPORT:-0}" = 1 ]; then
  DETTATO_WEIGHTS=weights .venv/bin/python scripts/export_onnx.py
fi

step "ONNX Runtime Web assets -> web/vendor/ort/ (served locally, no CDN)"
mise exec -- node scripts/copy_ort.mjs

step "demo / parity clips -> clips/"
[ -f clips/band.wav ] || .venv/bin/python scripts/make_clips.py

printf '\nDone. Start the page with:  npm run serve   (then open http://localhost:8080)\n'
