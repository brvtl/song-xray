"""Generate short, license-free demo clips (synthesised here, so no third-party rights) into clips/.

    python scripts/make_clips.py            # -> clips/*.wav (44.1 kHz mono) + clips/*.truth.json (the notes that were rendered)

The clips are simple additive / FM / noise synthesis. They are not realistic recordings: they exist so the parity check and the
demo have something reproducible to run on. Real recordings are a better test of quality; see README.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clips")


def hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)


def env_adsr(n, a=0.005, d=0.15, s=0.5, r=0.08, sr=SR):
    t = np.arange(n) / sr
    dur = n / sr
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    rel = np.clip((dur - t) / r, 0, 1)
    return e * rel


def piano(p, dur, vel):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(p)
    x = sum((0.6 ** (k - 1)) * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k * k)) * np.exp(-t * (1.5 + 0.8 * k)) for k in range(1, 9))
    return vel * x * env_adsr(n, d=0.4, s=0.0, r=0.05)


def bass(p, dur, vel):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(p)
    x = np.sin(2 * np.pi * f * t + 1.5 * np.exp(-t * 6) * np.sin(2 * np.pi * f * t))
    return vel * x * env_adsr(n, a=0.01, d=0.3, s=0.6, r=0.05)


def flute(p, dur, vel):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(p)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    x = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    return vel * x * env_adsr(n, a=0.06, d=0.2, s=0.8, r=0.1)


def kick(vel):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 50 + 110 * np.exp(-t * 30)
    return vel * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def snare(vel, rng):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    return vel * (0.6 * noise * np.exp(-t * 18) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25))


def hat(vel, rng):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    x = np.diff(np.concatenate([[0], x]))          # crude high-pass
    return vel * 0.35 * x * np.exp(-t * 60)


def place(buf, x, t0):
    i = int(t0 * SR)
    j = min(len(buf), i + len(x))
    buf[i:j] += x[:j - i]


def clip_band(seconds=12.0, bpm=100):
    """Piano chords + FM bass + flute melody + drum kit."""
    rng = np.random.default_rng(1)
    beat = 60 / bpm
    buf = np.zeros(int(seconds * SR) + SR)
    truth = {"piano": [], "bass": [], "flute": [], "drums": []}
    prog = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]), (55, [55, 59, 62])]   # Am F C G
    melody = [76, 74, 72, 74, 76, 76, 76, None, 74, 74, 74, None, 76, 79, 79, None]
    nbeats = int(seconds / beat)
    for b in range(nbeats):
        t = b * beat
        root, chord = prog[(b // 4) % 4]
        if b % 2 == 0:
            for p in chord:
                place(buf, piano(p + 12, beat * 1.9, 0.18), t)
                truth["piano"].append([t, t + beat * 1.9, p + 12])
        place(buf, bass(root - 12, beat * 0.9, 0.35), t)
        truth["bass"].append([t, t + beat * 0.9, root - 12])
        m = melody[b % len(melody)]
        if m is not None:
            place(buf, flute(m, beat * 0.95, 0.16), t)
            truth["flute"].append([t, t + beat * 0.95, m])
        place(buf, kick(0.7), t) if b % 2 == 0 else place(buf, snare(0.5, rng), t)
        truth["drums"].append([t, t + 0.1, 36 if b % 2 == 0 else 38])
        for h in (0, 0.5):
            place(buf, hat(0.6, rng), t + h * beat)
            truth["drums"].append([t + h * beat, t + h * beat + 0.1, 42])
    return buf[:int(seconds * SR)], truth


def clip_piano(seconds=10.0):
    """Solo 'piano': an arpeggiated progression with a bass line (one instrument)."""
    buf = np.zeros(int(seconds * SR) + SR)
    truth = {"piano": []}
    seq = [60, 64, 67, 72, 67, 64] * 2 + [57, 60, 64, 69, 64, 60] * 2 + [53, 57, 60, 65, 60, 57] * 2 + [55, 59, 62, 67, 62, 59] * 2
    step = 0.2
    for i, p in enumerate(seq):
        t = i * step
        if t + 0.6 > seconds:
            break
        place(buf, piano(p, 0.6, 0.25), t)
        truth["piano"].append([t, t + 0.6, p])
        if i % 12 == 0:
            place(buf, piano(p - 24, 2.4, 0.3), t)
            truth["piano"].append([t, t + 2.4, p - 24])
    return buf[:int(seconds * SR)], truth


def clip_duet(seconds=10.0):
    """Flute melody over FM bass, no drums: two clearly different timbres."""
    buf = np.zeros(int(seconds * SR) + SR)
    truth = {"flute": [], "bass": []}
    mel = [72, 74, 76, 77, 79, 77, 76, 74, 72, 71, 72, 74, 76, 74, 72, 72]
    for i, p in enumerate(mel):
        t = i * 0.6
        if t + 0.55 > seconds:
            break
        place(buf, flute(p, 0.55, 0.2), t)
        truth["flute"].append([t, t + 0.55, p])
    for i, p in enumerate([48, 43, 45, 41, 48, 43, 45, 41]):
        t = i * 1.2
        if t + 1.1 > seconds:
            break
        place(buf, bass(p, 1.1, 0.4), t)
        truth["bass"].append([t, t + 1.1, p])
    return buf[:int(seconds * SR)], truth


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("band", clip_band), ("piano", clip_piano), ("duet", clip_duet)):
        x, truth = fn()
        x = 0.5 * x / max(1e-9, np.abs(x).max())
        path = os.path.join(OUT, f"{name}.wav")
        sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")
        with open(os.path.join(OUT, f"{name}.truth.json"), "w") as fh:
            json.dump({k: [[round(a, 4), round(b, 4), int(p)] for a, b, p in v] for k, v in truth.items()}, fh)
        print(path, f"{len(x) / SR:.1f}s", {k: len(v) for k, v in truth.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
