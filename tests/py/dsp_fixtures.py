"""Extra reference arrays for tests/dsp.test.mjs, written under fixtures/<clip>/dsp/
(case table: fixtures/band/dsp/manifest.json). Run after scripts/dump_fixtures.py.

    .venv/bin/python tests/py/dsp_fixtures.py

- resample cases: soxr.resample(..., quality="HQ") outputs for the rate pairs the app uses (incl. retune r != 1), plus a table
  of output lengths for many (n, rate in, rate out) combinations
- drum_feats in C order, and soxr's 16 kHz version of the level-normalised signal (isolates the resampler from the features)
- a deliberately detuned clip (band x16 played 2^(35/1200) faster) and dettato.grouper.retune.ratio on it
"""

from __future__ import annotations

import importlib
import json
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))

import numpy as np  # noqa: E402
import soxr  # noqa: E402
from dettato.grouper import retune  # noqa: E402

DF = importlib.import_module("dettato.drums.features")


def out_dir(clip):
    d = os.path.join(ROOT, "fixtures", clip, "dsp")
    os.makedirs(d, exist_ok=True)
    return d


def save(clip, key, arr):
    np.save(os.path.join(out_dir(clip), f"{key}.npy"), np.ascontiguousarray(arr))


def resample_cases(clip):
    """Inputs are slices of the clip's own audio, labelled with the rate they are treated as."""
    base = os.path.join(ROOT, "fixtures", clip)
    x16 = np.load(os.path.join(base, "x16.npy"))
    cases = {
        "r16_up30": (x16, 16000, 16000 / 2 ** (30 / 1200)),
        "r16_dn25": (x16, 16000, 16000 / 2 ** (-25 / 1200)),
        "u16_44": (x16[:160000], 16000, 44100),
    }
    x44p = os.path.join(base, "x44.npy")
    if os.path.exists(x44p):
        x44 = np.load(x44p)[: 44100 * 10]
        cases["d48_16"] = (x44, 48000, 16000)
        cases["d48_44"] = (x44, 48000, 44100)
    meta = {}
    for name, (x, a, b) in cases.items():
        save(clip, f"rs_{name}_in", x.astype(np.float32))
        save(clip, f"rs_{name}_out", soxr.resample(x, a, b, quality="HQ").astype(np.float32))
        meta[name] = [a, b]
    return meta


def length_table():
    random.seed(7)
    rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 16000 / 2 ** (30 / 1200), 16000 / 2 ** (-25 / 1200)]
    rows = []
    for i in range(400):
        a, b = random.choice(rates), random.choice(rates)
        n = random.randint(1, 3000) if i % 2 else random.randint(1, 3_000_000)
        rows.append([n, a, b])
    rows += [[480, 24000, 11025], [800, 24000, 11025], [160, 8000, 11025], [1058400, 96000, 44100], [3, 16000, 8000]]
    out = []
    for n, a, b in rows:
        out.append([n, a, b, len(soxr.resample(np.zeros(n, np.float32), a, b, quality="HQ"))])
    return out


def main():
    man = {"resample": {}, "lengths": length_table()}
    for clip in ("band", "duet", "piano"):
        if not os.path.exists(os.path.join(ROOT, "fixtures", clip, "manifest.json")):
            continue
        man["resample"][clip] = resample_cases(clip)
        x44p = os.path.join(ROOT, "fixtures", clip, "x44.npy")
        if os.path.exists(x44p):
            x44 = np.load(x44p)
            save(clip, "drum_feats", np.load(os.path.join(ROOT, "fixtures", clip, "drum_feats.npy")).astype(np.float32))
            x = (x44 * DF.level_gain(x44)).astype(np.float32)
            save(clip, "drum_x16", soxr.resample(x, 44100, 16000).astype(np.float32))
    x16 = np.load(os.path.join(ROOT, "fixtures", "band", "x16.npy"))
    det = retune.resample(x16, 2 ** (35 / 1200))
    save("band", "detuned35", det)
    d, s, sh, r = retune.ratio(det)
    man["detuned35"] = dict(delta=d, strength=s, shift=sh, r=r)
    with open(os.path.join(out_dir("band"), "manifest.json"), "w") as fh:
        json.dump(man, fh)
    print(json.dumps(man["detuned35"]))


if __name__ == "__main__":
    main()
