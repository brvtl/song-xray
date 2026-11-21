"""Run the official Dettato reference (PyTorch, CPU) stage by stage on one clip and save every intermediate as .npy.

    python scripts/dump_fixtures.py clips/band.wav            # -> fixtures/band/*.npy + fixtures/band/manifest.json

The JavaScript port (web/js) is tested module by module against these arrays (tests/*.test.mjs): each test feeds a stage the
reference's own input and compares the output, so a mismatch points at one stage. The end-to-end comparison is parity/parity.py.

This mirrors dettato.pipeline.Dettato.pitched / transcribe step for step (same calls, same arguments), only keeping the pieces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))

import importlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dettato import audio as A  # noqa: E402
from dettato.drums import DrumTranscriber  # noqa: E402

DF = importlib.import_module("dettato.drums.features")
from dettato.evidence import EvidenceRepairer  # noqa: E402
from dettato.evidence import features as EF  # noqa: E402
from dettato.evidence import repair as ER  # noqa: E402
from dettato.grouper import Grouper, retune  # noqa: E402
from dettato.grouper import core as GC  # noqa: E402
from dettato.grouper.decode import decode as gdecode  # noqa: E402
from dettato.pipeline import MIN_END_S  # noqa: E402
from dettato.transcriber import HEADS, Transcriber  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--taps", action="store_true", help="also save the (big) tap buffers hidden / sub_hidden / sub_stem")
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    name = os.path.splitext(os.path.basename(a.clip))[0]
    out = a.out or os.path.join(ROOT, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    man: dict = {"clip": os.path.relpath(a.clip, ROOT)}

    def save(key, arr):
        arr = np.ascontiguousarray(arr)
        np.save(os.path.join(out, f"{key}.npy"), arr)
        man.setdefault("arrays", {})[key] = [list(arr.shape), str(arr.dtype)]

    t_all = time.time()
    tr = Transcriber.from_pretrained("cpu")
    rep = EvidenceRepairer.from_pretrained("cpu")
    grp = Grouper.from_pretrained("cpu")
    drums = DrumTranscriber.from_pretrained("cpu")

    au = A.load(a.clip)
    man["source_rate"] = au.rate
    x = au.at(16000)
    save("x16", x)
    with torch.no_grad():
        # ---------------------------------------------------------------- tuning
        delta, strength, shift, r = retune.ratio(x)
        y = retune.resample(x, r) if shift else x
        man["tuning"] = dict(delta=delta, strength=strength, shift=shift, r=r)
        save("y", y)
        # ---------------------------------------------------------------- transcriber: window 0's HCQT + net outputs
        W, Fw, T, starts, pad, g_pad = tr._layout(len(y), 0.25)
        man["layout"] = dict(W=W, Fw=Fw, T=T, starts=starts, pad=pad, g_pad=g_pad)
        x_all = np.concatenate([np.zeros(pad, np.float32), y.astype(np.float32), np.zeros(pad + W, np.float32)])
        k_probe = min(2, len(starts) - 1)
        s0 = starts[k_probe]
        win = torch.from_numpy(x_all[s0 : s0 + W][None])
        hc = tr.model.front(win)
        save("win_audio", win[0].numpy())
        save("win_hcqt", hc[0].numpy())
        box = {}
        res_w = tr.model(win, box)
        save("win_probs", np.stack([torch.sigmoid(res_w[k])[0].numpy() for k in HEADS]))
        save("win_hidden_lane40", box["hidden"][0, :, :, 40].numpy())  # (C, Fw) one lane, to keep the file small
        man["win_probe"] = k_probe
        # ---------------------------------------------------------------- one pass, as Dettato.pitched
        t0 = time.time()
        maps, bufs, (smaps, sbufs) = tr.run(y, hop_frac=0.25, batch=2, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"))
        man["time_transcriber_s"] = time.time() - t0
        save("maps", np.stack([maps[k] for k in HEADS]))
        save("smaps", np.stack([smaps[k] for k in HEADS]))
        H = bufs["hidden"]
        if a.taps:
            save("hidden", H.numpy())
            save("sub_hidden", sbufs["hidden"].numpy())
            save("sub_stem", sbufs["stem"].numpy())
        # ---------------------------------------------------------------- transcriber notes + evidence
        res, L, Aa, B, mf = EF._decode_maps(tr, maps)
        for k_ in ("onset", "offset", "pitch", "conf", "slot"):
            save(f"tnotes_{k_}", res[k_])
        save("ev_map", mf)
        at, mean = EF.gather_hidden(H, L, Aa, B, tr.device)
        save("ev_at", at)
        save("ev_mean", mean)
        ev = EF._assemble(mf, at, mean)
        save("ev", ev)
        # ---------------------------------------------------------------- repairer
        k, o = ER.input_notes(res)
        save("rep_in_order", o)
        for f in ("pitch", "vel", "on", "off"):
            save(f"rep_in_{f}", k[f])
        e = rep.edits(k, ev[o])
        pk, on_r, off_r, ins = e
        save("rep_pk", pk)
        save("rep_on_r", on_r)
        save("rep_off_r", off_r)
        save("rep_ins", np.array(ins, np.float64).reshape(-1, 4))
        # one window's raw network inputs and outputs, for the ONNX graph test
        lo, hi = int(np.searchsorted(k["on"], 0.0)), int(np.searchsorted(k["on"], 8.0))
        if hi > lo:
            b = ER._window_batch(k[lo:hi], ev[o][lo:hi], 0.0, "cpu")
            c = rep.model.candidates(b)[0]
            save("rep_w0_score", c["score"])
            save("rep_w0_on", c["on"])
            save("rep_w0_off", c["off"])
            save("rep_w0_pitch", c["pitch"])
        repn = rep.apply(k, e)
        for k_ in ("onset", "offset", "pitch", "inserted", "score"):
            save(f"rep_{k_}", repn[k_])
        man["repair_info"] = dict(transcribed=len(k), kept=int((~repn["inserted"]).sum()), inserted=int(repn["inserted"].sum()))
        # ---------------------------------------------------------------- grouper
        notes = np.stack([repn["onset"], np.maximum(repn["offset"], repn["onset"] + MIN_END_S), repn["pitch"].astype(np.float64)], 1)
        save("grp_notes_in", notes)
        tn = tr.notes(smaps)
        own = np.stack([tn["onset"], tn["offset"], tn["pitch"].astype(np.float64), tn["slot"] - 1.0], 1).reshape(-1, 4)
        save("grp_own", own)
        nt = GC.assign_slots(notes, own)
        save("grp_nt", nt)
        P = {kk: torch.from_numpy(v) for kk, v in smaps.items()}
        meta, feat, fsc = GC.gather(P, sbufs["hidden"], sbufs["stem"], nt, tr.hop)
        save("grp_meta", meta)
        save("grp_feat", feat)
        save("grp_fsc", fsc)
        song = GC._Song(y, meta, feat, fsc)
        jobs = GC.song_windows(meta[:, 0], grp.window_s, grp.window_s * grp.window_hop, grp.max_notes)
        man["grp_jobs"] = [[float(t), s.tolist()] for t, s in jobs]
        if jobs:
            t_0, sel = jobs[0]
            it = song.window(t_0, grp.window_s, sel) or song.window(t_0, grp.window_s, np.concatenate([sel] * 4)[:4])
            save("grp_w0_audio", it["audio"])
            Xf, Pf = grp.model.front(torch.from_numpy(it["audio"][None]))
            save("grp_w0_X", Xf[0].numpy())
            save("grp_w0_P", Pf[0].numpy())
            save("grp_w0_notes", it["notes"])
            save("grp_w0_feat", it["feat"])
        t0 = time.time()
        pred = grp.predict(song)
        man["time_grouper_s"] = time.time() - t0
        save("grp_probs", pred["probs"])
        save("grp_emb", pred["emb"])
        save("grp_h", pred["h"].astype(np.float32))
        man["grp_wins"] = [[t_, s.tolist()] for t_, s, _, _ in pred["wins"]]
        for i, (_, _, e_, p_) in enumerate(pred["wins"]):
            save(f"grp_win{i}_emb", e_.astype(np.float32))
            save(f"grp_win{i}_probs", p_.astype(np.float32))
        on_, off_, pitch_ = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64), meta[:, 2].round().astype(np.int64)
        groups = gdecode(pred, (on_, off_, pitch_.astype(np.float64)), grp.th, **grp.params) if len(on_) else []
        man["grp_groups"] = [[int(kk), np.asarray(ids).tolist()] for kk, ids in groups]
        conf = np.where(meta[:, 3] > 0.5, meta[:, 5], meta[:, 4])
        tracks = grp._tracks(on_ * r, off_ * r, pitch_, conf, groups)
        man["tracks"] = [dict(instrument=t_["instrument"], program=t_["program"], notes=t_["notes"].tolist()) for t_ in tracks]
        # ---------------------------------------------------------------- drums
        if au.rate >= 40000:
            x44 = au.at(44100)
            save("x44", x44)
            t0 = time.time()
            feats = DF.features(x44)
            save("drum_feats", feats.astype(np.float32))
            Pd = drums.probabilities(feats.astype(np.float32))
            for kk in ("kit", "kit_vel", "drum", "timpani", "vel"):
                save(f"drum_{kk}", Pd[kk])
            man["drum_router"] = Pd["router"]
            hits = drums.transcribe(x44)
            man["time_drums_s"] = time.time() - t0
            man["drum_model"] = drums.last_info.get("model")
            man["drum_hits"] = hits
    man["time_total_s"] = time.time() - t_all
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(man, fh, default=float)
    print(f"{name}: {len(man['arrays'])} arrays -> {out}  ({man['time_total_s']:.1f} s)", file=sys.stderr)
    print(
        json.dumps(
            {
                "tracks": [(t_["instrument"], len(t_["notes"])) for t_ in man["tracks"]],
                "drum_hits": len(man.get("drum_hits", [])),
                "repair": man["repair_info"],
                "tuning": man["tuning"],
            }
        )
    )


if __name__ == "__main__":
    main()
