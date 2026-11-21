# Lessons

- Never invoke GPU tooling (nvidia-smi, CUDA probes) in this repo, not even with output discarded: the brief forbids touching
  the GPU. To check hardware, use lscpu / free only.
- `rm` is aliased to interactive in this shell; use `command rm -f` in scripts.
