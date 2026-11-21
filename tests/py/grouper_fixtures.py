"""Extra reference arrays for tests/grouper.test.mjs and tests/cluster.test.mjs (written under fixtures/<clip>/grouper/).

    CUDA_VISIBLE_DEVICES= DETTATO_WEIGHTS=weights .venv/bin/python tests/py/grouper_fixtures.py

Reads the arrays scripts/dump_fixtures.py wrote (it does not rerun the transcriber) and adds:
  - per clip: the grouper network's raw outputs on window 0 (collated as Grouper.predict does): w0_probs / w0_emb / w0_h;
  - band: a multi-window song (band's notes three times, 24 s apart, plus two stray notes so the last windows take the
    "fewer than 4 notes" fallback) run through Grouper.predict with the front end pinned to window 0's X / P (the JS test pins
    its frontFn the same way), and its decode;
  - band: synthetic multi-window decode inputs (many windows, > 600 tracklets, > 3000 notes) and their groups, also with
    np.argsort made stable (the reference's argsort is not stable on AVX-512 machines; JS sorts stably);
  - band: scipy references for cluster.js (average linkage + fcluster 'distance', linear_sum_assignment, erf) and
    song_windows cases.
CPU only.
"""
from __future__ import annotations

import json
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.special import erf  # noqa: E402

from dettato.grouper import Grouper  # noqa: E402
from dettato.grouper import core as GC  # noqa: E402
from dettato.grouper import decode as GD  # noqa: E402

CLIPS = ("band", "duet", "piano")


def load(clip, key):
    return np.load(os.path.join(ROOT, "fixtures", clip, f"{key}.npy"))


def saver(clip):
    out = os.path.join(ROOT, "fixtures", clip, "grouper")
    os.makedirs(out, exist_ok=True)

    def save(key, arr):
        np.save(os.path.join(out, f"{key}.npy"), np.asarray(arr))

    def save_json(key, obj):
        with open(os.path.join(out, f"{key}.json"), "w") as fh:
            json.dump(obj, fh)

    return save, save_json


def song_of(clip):
    return GC._Song(load(clip, "y"), load(clip, "grp_meta"), load(clip, "grp_feat"), load(clip, "grp_fsc"))


@torch.no_grad()
def window0_outputs(grp, clip):
    save, _ = saver(clip)
    song = song_of(clip)
    jobs = GC.song_windows(song.meta[:, 0], grp.window_s, grp.window_s * grp.window_hop, grp.max_notes)
    t0, sel = jobs[0]
    it = song.window(t0, grp.window_s, sel) or song.window(t0, grp.window_s, np.concatenate([sel] * 4)[:4])
    o = grp.model(GC.collate([it], grp.max_notes, "cpu"))
    m = len(it["notes"])
    save("w0_probs", torch.softmax(o["inst"].float(), -1)[0, :m].numpy())
    save("w0_emb", o["emb"].float()[0, :m].numpy())
    save("w0_h", o["h"].float()[0, :m].numpy())


def pred_out(save, save_json, prefix, pred, groups):
    save(f"{prefix}_probs", pred["probs"])
    save(f"{prefix}_emb", pred["emb"])
    save(f"{prefix}_h", np.asarray(pred["h"], np.float32))
    save_json(f"{prefix}_wins", [[float(t), np.asarray(s).tolist()] for t, s, _, _ in pred["wins"]])
    for i, (_, _, e, p) in enumerate(pred["wins"]):
        save(f"{prefix}_win{i}_emb", e.astype(np.float32))
        save(f"{prefix}_win{i}_probs", p.astype(np.float32))
    save_json(f"{prefix}_groups", [[int(k), np.asarray(ids).tolist()] for k, ids in groups])


def multi_window_song(grp):
    """band's notes three times (24 s apart) + two stray notes; the front end pinned to window 0's X / P."""
    save, save_json = saver("band")
    meta, feat, fsc = load("band", "grp_meta"), load("band", "grp_feat"), load("band", "grp_fsc")
    keep = meta[:, 0] >= 0
    meta, feat, fsc = meta[keep], feat[keep], fsc[keep]
    parts = []
    for k in range(3):
        m = meta.copy()
        m[:, :2] += np.float32(24.0 * k)
        parts.append(m)
    stray = meta[[3, 7]].copy()
    stray[:, 0] = [84.0, 85.0]
    stray[:, 1] = stray[:, 0] + 0.5
    M = np.concatenate(parts + [stray]).astype(np.float32)
    F = np.concatenate([feat] * 3 + [feat[[3, 7]]])
    S = np.concatenate([fsc] * 3 + [fsc[[3, 7]]])
    X = torch.from_numpy(load("band", "grp_w0_X"))[None]
    P = torch.from_numpy(load("band", "grp_w0_P"))[None]
    front = grp.model.front
    front.forward = lambda audio: (X.expand(audio.shape[0], -1, -1, -1), P.expand(audio.shape[0], -1, -1))
    try:
        song = GC._Song(np.zeros(16000 * 90, np.float32), M, F, S)
        pred = grp.predict(song)
    finally:
        del front.forward
    on, off, pitch = M[:, 0].astype(np.float64), M[:, 1].astype(np.float64), M[:, 2].round().astype(np.float64)
    groups = GD.decode(pred, (on, off, pitch), grp.th, **grp.params)
    save("mw_meta", M)
    save("mw_feat", F)
    save("mw_fsc", S)
    pred_out(save, save_json, "mw", pred, groups)
    print("multi-window:", len(pred["wins"]), "windows,", [(k, len(i)) for k, i in groups], file=sys.stderr)


def synthetic_pred(rng, n, dur, n_src, noise, W=24.0, nmax=768):
    """Predictions shaped like Grouper.predict's: per-window float16 emb / probs around per-source centres."""
    on = np.sort(rng.uniform(0, dur, n)).astype(np.float32)
    src = rng.integers(0, n_src, n)
    cen = rng.standard_normal((n_src, 32))
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    inst = rng.integers(0, 73, n_src)
    jobs = GC.song_windows(on, W, W * 0.5, nmax)
    acc, wsum, eacc = np.zeros((n, 73)), np.zeros(n), np.zeros((n, 32))
    wins = []
    for t0, sel in jobs:
        e = cen[src[sel]] + noise * rng.standard_normal((len(sel), 32))
        e = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16)
        lg = rng.standard_normal((len(sel), 73))
        lg[np.arange(len(sel)), inst[src[sel]]] += 4.0
        p = np.exp(lg)
        p = (p / p.sum(1, keepdims=True)).astype(np.float16)
        w = np.clip(1.0 - np.abs((on[sel] - t0) / W - 0.5) * 2, 0.05, 1.0)
        acc[sel] += p * w[:, None]
        wsum[sel] += w
        eacc[sel] += e * w[:, None]
        wins.append((float(t0), sel, e, p))
    emb = eacc / np.maximum(np.linalg.norm(eacc, axis=1, keepdims=True), 1e-9)
    h = (0.5 * rng.standard_normal((n, 176))).astype(np.float16)
    pred = dict(probs=(acc / np.maximum(wsum, 1e-9)[:, None]).astype(np.float32), emb=emb.astype(np.float32), h=h, wins=wins, W=W)
    off = on.astype(np.float64) + rng.uniform(0.05, 1.5, n)
    pitch = np.clip(40 + 8 * src + rng.integers(-6, 7, n), 21, 108).astype(np.float64)
    return pred, (on.astype(np.float64), off, pitch)


def stable_np():
    """numpy with a stable argsort (what the JS port does)."""
    ns = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    ns.argsort = lambda a, *args, **kw: np.argsort(a, kind="stable")
    return ns


def synthetic_decodes(grp):
    save, save_json = saver("band")
    rng = np.random.default_rng(7)
    cases = dict(synA=(420, 70.0, 4, 0.03), synB=(1500, 140.0, 40, 0.055), synC=(3300, 300.0, 3, 0.03))
    info = {}
    for name, (n, dur, n_src, noise) in cases.items():
        pred, notes = synthetic_pred(rng, n, dur, n_src, noise)
        groups = GD.decode(pred, notes, grp.th, **grp.params)
        real_np = GD.np
        GD.np = stable_np()
        try:
            groups_st = GD.decode(pred, notes, grp.th, **grp.params)
        finally:
            GD.np = real_np
        tr = GD.group_chain(pred, notes[0], grp.params["tau"], grp.params["lam"])
        save(f"{name}_on", notes[0])
        save(f"{name}_off", notes[1])
        save(f"{name}_pitch", notes[2])
        pred_out(save, save_json, name, pred, groups)
        save_json(f"{name}_groups_stable", [[int(k), np.asarray(ids).tolist()] for k, ids in groups_st])
        save_json(f"{name}_chain", tr.tolist())
        same = [[k, list(map(int, i))] for k, i in groups] == [[k, list(map(int, i))] for k, i in groups_st]
        info[name] = dict(n=n, windows=len(pred["wins"]), tracklets=int(len(np.unique(tr[tr >= 0]))), stable_same=same,
                          groups=[(int(k), len(i)) for k, i in groups])
    save_json("syn_info", info)
    print("synthetic:", json.dumps(info), file=sys.stderr)


def cluster_refs():
    """scipy references: average linkage + fcluster, linear_sum_assignment(-ov), erf; song_windows cases."""
    _, save_json = saver("band")
    rng = np.random.default_rng(3)
    link = []
    for n in [2, 3, 4, 5, 7, 10, 16, 30, 64, 120]:
        for kind in ("cont", "int", "f32"):
            m = n * (n - 1) // 2
            if kind == "cont":
                y = rng.uniform(0, 1, m)
            elif kind == "int":
                y = rng.integers(0, 4, m).astype(np.float64) / 10.0
            else:
                y = rng.uniform(0, 0.4, m).astype(np.float32).astype(np.float64)
            Z = linkage(y, "average")
            ts = [0.0, 0.1, 0.2, 0.3, float(np.median(Z[:, 2])), 10.0]
            link.append(dict(n=n, y=y.tolist(), Z=Z.tolist(), t=ts, labels=[fcluster(Z, t, "distance").tolist() for t in ts]))
    lsap = []
    for r, c in [(1, 1), (1, 3), (3, 1), (2, 2), (3, 3), (4, 2), (2, 5), (5, 5), (6, 3), (3, 7), (8, 8), (10, 4), (12, 12)]:
        for hi in (2, 4, 30):
            ov = rng.integers(0, hi, (r, c)).astype(np.float64)
            if rng.uniform() < 0.3:
                ov[rng.integers(0, r)] = 0
            a, b = linear_sum_assignment(-ov)
            lsap.append(dict(cost=(-ov).tolist(), rows=a.tolist(), cols=b.tolist()))
    xs = np.concatenate([np.linspace(-8, 8, 801), rng.standard_normal(400) * 3, [0.0, 1e-8, -1e-8, 0.5, 2.5, 5.9, 6.1, 27.0]])
    sw = []
    for n, dur, nmax in [(0, 10, 768), (1, 10, 768), (50, 30, 768), (300, 60, 40), (500, 100, 64), (200, 20, 30), (90, 200, 768)]:
        on = np.sort(rng.uniform(-0.05, dur, n)).astype(np.float32)
        if n > 20:
            on[5:9] = on[5]          # duplicate onsets
        jobs = GC.song_windows(on, 24.0, 12.0, nmax)
        sw.append(dict(on=on.astype(np.float64).tolist(), nmax=nmax, jobs=[[float(t), s.tolist()] for t, s in jobs]))
    save_json("cluster_refs", dict(linkage=link, lsap=lsap, erf=dict(x=xs.tolist(), y=erf(xs).tolist()), windows=sw))


def main():
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    grp = Grouper.from_pretrained("cpu")
    for clip in CLIPS:
        if os.path.exists(os.path.join(ROOT, "fixtures", clip, "manifest.json")):
            window0_outputs(grp, clip)
    cluster_refs()
    multi_window_song(grp)
    synthetic_decodes(grp)


if __name__ == "__main__":
    main()
