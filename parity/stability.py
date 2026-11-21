"""How stable is the reference's track grouping to tiny perturbations? Runs dettato's pitched pipeline on one file as shipped,
with -100 dBFS dither added to the audio (two seeds), and with the grouper's PCEN FFT convolution padded to twice the length
(mathematically the same, different float32 rounding).

    python parity/stability.py clips/sousa_full.mp3

If the reference's own tracks change under this perturbation, track-level differences of the same size between the browser port
and the reference are float noise, not port bugs. CPU only.
"""

import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from parity import track_stats

from dettato import Dettato
from dettato import audio as A
from dettato.grouper import front as FR


def pcen_padded(X, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, K=200):
    B, Cc, T, Fq = X.shape
    E = torch.exp(X.float()).permute(0, 1, 3, 2)
    k = s * (1 - s) ** torch.arange(K, device=X.device, dtype=torch.float32)
    n = 2 << int(math.ceil(math.log2(T + K)))  # the only change: twice the FFT length
    M = torch.fft.irfft(torch.fft.rfft(E, n) * torch.fft.rfft(k, n), n)[..., :T].clamp(min=0)
    out = (E / (eps + M) ** alpha + delta) ** r - delta**r
    return torch.log(out + 1e-4).permute(0, 1, 3, 2)


def tracks_of(m, x):
    with torch.no_grad():
        _, _, _, groups, _ = m.pitched(x)
    return [dict(name=g["instrument"], isDrum=False, notes=np.asarray(g["notes"]).tolist()) for g in groups]


def main():
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    m = Dettato.from_pretrained("cpu")
    x = A.load(sys.argv[1]).at(16000)
    a = tracks_of(m, x)
    for seed in (1, 2):  # -100 dBFS dither: far below what different mp3 decoders or resamplers change
        xd = (x + np.random.default_rng(seed).normal(0, 1e-5, len(x))).astype(np.float32)
        s = track_stats(a, tracks_of(m, xd))
        print(f"dither seed {seed}: {s['js_tracks']}  agreement {s['agreement']:.3f}, names equal: {s['names_equal']}")
    orig = FR.pcen
    FR.pcen = pcen_padded
    try:
        b = tracks_of(m, x)
    finally:
        FR.pcen = orig
    s = track_stats(a, b)
    print(f"reference as shipped : {s['ref_tracks']}")
    print(f"reference, padded FFT: {s['js_tracks']}")
    print(f"track agreement between the two reference runs: {s['agreement']:.3f}, names equal: {s['names_equal']}")


if __name__ == "__main__":
    main()
