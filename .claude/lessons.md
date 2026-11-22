# Lessons

- Never invoke GPU tooling (nvidia-smi, CUDA probes) in this repo, not even with output discarded: the brief forbids touching
  the GPU. To check hardware, use lscpu / free only.
- `rm` is aliased to interactive in this shell; use `command rm -f` in scripts.
- `pgrep -f` / `pkill -f` patterns match the invoking shell's own command line when the pattern text appears in it (this
  killed a background job and hung samplers). Bind waits to a PID (`$!`) or match a string the shell line cannot contain.
