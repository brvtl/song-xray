"""Parity check: the official Dettato reference (PyTorch, CPU) vs this project's in-browser pipeline, on the same clips.

    python parity/parity.py                       # clips/band.wav clips/duet.wav clips/piano.wav (+ an mp3 and a detuned variant)
    python parity/parity.py clips/x.wav --browser # also compare the result the real page produced (tests/e2e/browser.mjs)

For every clip it runs
  ref      dettato.Dettato.transcribe (the Python package at the pinned commit, CPU, float32)
  js-same  web/js/pipeline.js under Node + onnxruntime-web WASM, fed the reference's own 16 kHz / 44.1 kHz samples
           (isolates the ONNX models and the JS port from resampling)
  js       the same pipeline doing its own decode + resampling (what the page does, minus the browser's audio decoder)
  browser  (optional) the result JSON the page itself produced in headless Chromium for the same file
and compares, per stage: the transcriber's probability maps (max abs diff), the transcriber notes, the repaired notes, the final
pitched notes and their track assignment, track names, and the drum hits. Notes match when the pitch is equal and the onsets are
within 50 ms (the usual transcription tolerance); "exact" means onset and offset within 1 ms.

Writes parity/REPORT.md and parity/out/report.json and prints a PASS / FAIL table. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))
OUT = os.path.join(ROOT, "parity", "out")

import numpy as np
import soundfile as sf
import soxr
from scipy.optimize import linear_sum_assignment

TOL_ON = 0.05
THRESH = {  # minimum F1 / agreement for PASS, per comparison mode
    "js-same": dict(maps=1e-3, notes=0.99, tracks=0.95, drums=0.99),
    "js": dict(maps=2e-2, notes=0.95, tracks=0.90, drums=0.95),
    "browser": dict(maps=None, notes=0.95, tracks=0.90, drums=0.95),
}


# ----------------------------------------------------------------------------- clips
def ensure_variants(clips):
    """An mp3 of band.wav (decoder delay / padding shows up as a systematic offset) and a detuned copy (exercises retune)."""
    out = list(clips)
    band = os.path.join(ROOT, "clips", "band.wav")
    if band in [os.path.abspath(c) for c in clips]:
        mp3 = os.path.join(ROOT, "clips", "band.mp3")
        if not os.path.exists(mp3):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", band, "-b:a", "192k", mp3], check=True)
        det = os.path.join(ROOT, "clips", "band_detuned.wav")
        if not os.path.exists(det):
            x, sr = sf.read(band, dtype="float32")
            r = 2 ** (35 / 1200)  # 35 cents sharp: played r times faster
            y = soxr.resample(x, sr, sr / r, quality="HQ")
            sf.write(det, y.astype(np.float32), sr, subtype="PCM_16")
        out += [mp3, det]
    return out


# ----------------------------------------------------------------------------- runs
def run_reference(path, outdir):
    import torch
    from dettato import Dettato
    from dettato import audio as A

    torch.set_num_threads(min(16, os.cpu_count() or 4))
    if not hasattr(run_reference, "model"):
        run_reference.model = Dettato.from_pretrained("cpu")
    m = run_reference.model
    t0 = time.time()
    au = A.load(path)
    x = au.at(16000)
    np.save(os.path.join(outdir, "x16.npy"), x)
    if au.rate >= 40000:
        np.save(os.path.join(outdir, "x44.npy"), au.at(44100))
    with torch.no_grad():
        rep, res, rinfo, groups, ginfo = m.pitched(x)
        tracks = [
            dict(name=g["instrument"].replace("_", " "), program=int(g["program"]), isDrum=False, notes=np.asarray(g["notes"]).tolist())
            for g in groups
        ]
        info = {"notes": {k: rinfo[k] for k in ("transcribed", "dropped", "inserted")}, "tuning_cents": ginfo.get("tuning_cents")}
        if au.rate >= 40000:
            hits = m.drums.transcribe(au.at(44100))
            if hits:
                d = [[h["onset"], h["offset"], h["pitch"], max(1, min(127, int(round(h["velocity"] * 127))))] for h in hits]
                tracks.append(dict(name="Drums", program=0, isDrum=True, notes=d))
            info["drums"] = {"hits": len(hits), **m.drums.last_info}
        else:
            info["drums"] = f"skipped: source recorded at {au.rate} Hz"
        ref_time = time.time() - t0
        # the transcriber's maps for the stage comparison (one more pass: the pipeline does not return them)
        from dettato.grouper import retune

        _, _, shift, ratio = retune.ratio(x)
        y = retune.resample(x, ratio) if shift else x
        maps, _ = m.transcriber.run(y, hop_frac=0.25, batch=2)
    heads = ("onset", "frame", "offset", "onset2", "frame2", "offset2")
    np.save(os.path.join(outdir, "ref_maps.npy"), np.stack([maps[k] for k in heads]))
    ref = dict(
        seconds=au.seconds,
        time_s=ref_time,
        tnotes=np.stack([np.round(res["onset"], 4), res["offset"], res["pitch"]], 1).tolist() if len(res["onset"]) else [],
        repaired=np.stack([rep["onset"], rep["offset"], rep["pitch"]], 1).tolist() if len(rep["onset"]) else [],
        tracks=tracks,
        info=json.loads(json.dumps(info, default=float)),
        ratio=ratio,
    )
    return ref


def run_js(path, outdir, mode, threads):
    out = os.path.join(outdir, f"{mode}.json")
    cmd = [
        "node",
        os.path.join(ROOT, "scripts", "transcribe.mjs"),
        path,
        "--out",
        out,
        "--threads",
        str(threads),
        "--quiet",
        "--debug-dir",
        outdir,
    ]
    if mode == "js-same":
        cmd += ["--same-samples", outdir]
    t0 = time.time()
    p = subprocess.run(["mise", "exec", "--", *cmd], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{mode} failed:\n{p.stderr[-3000:]}")
    res = json.load(open(out))
    res["time_s"] = time.time() - t0
    os.replace(os.path.join(outdir, "maps.f32"), os.path.join(outdir, f"{mode}_maps.f32")) if os.path.exists(
        os.path.join(outdir, "maps.f32")
    ) else None
    return res


# ----------------------------------------------------------------------------- comparisons
def match_notes(a, b, tol=TOL_ON):
    """Optimal one-to-one matching of notes (on, off, pitch, ...) with equal pitch and |onset diff| <= tol.
    -> list of (i, j) pairs."""
    a = np.asarray(a, np.float64).reshape(-1, max(3, np.asarray(a).shape[-1] if len(a) else 3))
    b = np.asarray(b, np.float64).reshape(-1, max(3, np.asarray(b).shape[-1] if len(b) else 3))
    pairs = []
    for p in np.unique(np.concatenate([a[:, 2], b[:, 2]])) if len(a) and len(b) else []:
        ia, ib = np.nonzero(a[:, 2] == p)[0], np.nonzero(b[:, 2] == p)[0]
        if not len(ia) or not len(ib):
            continue
        d = np.abs(a[ia, 0][:, None] - b[ib, 0][None, :])
        cost = np.where(d <= tol, d, 1e6)
        r, c = linear_sum_assignment(cost)
        pairs += [(int(ia[i]), int(ib[j])) for i, j in zip(r, c) if cost[i, j] < 1e6]
    return pairs


def note_stats(ref, got):
    ref = np.asarray(ref, np.float64).reshape(-1, 3 if not len(ref) else np.asarray(ref).shape[1])
    got = np.asarray(got, np.float64).reshape(-1, 3 if not len(got) else np.asarray(got).shape[1])
    pairs = match_notes(ref, got)
    n_r, n_g, n_m = len(ref), len(got), len(pairs)
    prec = n_m / n_g if n_g else 1.0
    rec = n_m / n_r if n_r else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    s = dict(ref=n_r, got=n_g, matched=n_m, precision=prec, recall=rec, f1=f1 if (n_r or n_g) else 1.0)
    if pairs:
        i, j = np.array(pairs).T
        d_on = got[j, 0] - ref[i, 0]
        d_off = got[j, 1] - ref[i, 1]
        s.update(
            exact=int(np.sum((np.abs(d_on) < 1e-3) & (np.abs(d_off) < 1e-3))),
            onset_mean_ms=float(d_on.mean() * 1000),
            onset_abs_p95_ms=float(np.percentile(np.abs(d_on), 95) * 1000),
            offset_mean_ms=float(d_off.mean() * 1000),
            offset_abs_p95_ms=float(np.percentile(np.abs(d_off), 95) * 1000),
        )
        # where they diverge: unmatched notes by time
        um_r = sorted(set(range(n_r)) - set(i.tolist()))
        um_g = sorted(set(range(n_g)) - set(j.tolist()))
        s["only_ref"] = [[round(ref[k, 0], 3), int(ref[k, 2])] for k in um_r[:12]]
        s["only_js"] = [[round(got[k, 0], 3), int(got[k, 2])] for k in um_g[:12]]
    return s


def flatten(tracks, drums):
    rows = []
    for ti, t in enumerate(tracks):
        if bool(t.get("isDrum")) != drums:
            continue
        for n in t["notes"]:
            rows.append([n[0], n[1], n[2], ti])
    return np.asarray(rows, np.float64).reshape(-1, 4)


def track_stats(ref_tracks, got_tracks):
    """Match the pitched notes across all tracks, then map JS tracks to reference tracks (Hungarian on the contingency table);
    agreement = share of matched notes whose JS track maps to their reference track."""
    R, G = flatten(ref_tracks, False), flatten(got_tracks, False)
    pairs = match_notes(R, G)
    rt = [t for t in ref_tracks if not t.get("isDrum")]
    gt = [t for t in got_tracks if not t.get("isDrum")]
    s = dict(ref_tracks=[f"{t['name']} ({len(t['notes'])})" for t in rt], js_tracks=[f"{t['name']} ({len(t['notes'])})" for t in gt])
    if not pairs:
        s.update(agreement=1.0 if not len(R) and not len(G) else 0.0, names_equal=len(rt) == len(gt) == 0)
        return s
    ri = {i: k for k, i in enumerate([i for i, t in enumerate(ref_tracks) if not t.get("isDrum")])}
    gi = {i: k for k, i in enumerate([i for i, t in enumerate(got_tracks) if not t.get("isDrum")])}
    C = np.zeros((len(rt), len(gt)))
    for i, j in pairs:
        C[ri[int(R[i, 3])], gi[int(G[j, 3])]] += 1
    r, c = linear_sum_assignment(-C)
    s["agreement"] = float(C[r, c].sum() / C.sum())
    s["name_pairs"] = [[rt[a]["name"], gt[b]["name"]] for a, b in zip(r, c)]
    s["names_equal"] = len(rt) == len(gt) and all(rt[a]["name"] == gt[b]["name"] for a, b in zip(r, c))
    return s


def maps_diff(outdir, mode):
    ref_p, got_p = os.path.join(outdir, "ref_maps.npy"), os.path.join(outdir, f"{mode}_maps.f32")
    if not (os.path.exists(ref_p) and os.path.exists(got_p)):
        return None
    ref = np.load(ref_p)
    got = np.fromfile(got_p, np.float32)
    if got.size != ref.size:
        return dict(error=f"size {got.size} vs {ref.size}")
    got = got.reshape(ref.shape)
    d = np.abs(got - ref)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), p999_abs=float(np.quantile(d, 0.999)))


def compare(ref, got, outdir, mode):
    c = dict(mode=mode, time_s=got.get("time_s"))
    c["maps"] = maps_diff(outdir, mode)
    dbg = got.get("info", {})
    if "transcriberNotes" in dbg:
        c["transcriber_notes"] = note_stats(ref["tnotes"], dbg["transcriberNotes"])
    if "repaired" in dbg and isinstance(dbg["repaired"], dict):
        rp = dbg["repaired"]
        c["repaired_notes"] = note_stats(
            ref["repaired"], np.stack([rp["onset"], rp["offset"], rp["pitch"]], 1).tolist() if rp["onset"] else []
        )
    c["pitched_notes"] = note_stats(flatten(ref["tracks"], False)[:, :3], flatten(got["tracks"], False)[:, :3])
    c["tracks"] = track_stats(ref["tracks"], got["tracks"])
    c["drums"] = note_stats(flatten(ref["tracks"], True)[:, :3], flatten(got["tracks"], True)[:, :3])
    c["drums"]["ref_model"] = (
        (ref["info"].get("drums") or {}).get("model") if isinstance(ref["info"].get("drums"), dict) else ref["info"].get("drums")
    )
    gd = dbg.get("drums")
    c["drums"]["js_model"] = gd.get("model") if isinstance(gd, dict) else gd
    th = THRESH[mode]
    checks = {
        "notes": c["pitched_notes"]["f1"] >= th["notes"],
        "tracks": c["tracks"]["agreement"] >= th["tracks"] and c["tracks"].get("names_equal", False),
        "drums": c["drums"]["f1"] >= th["drums"] and c["drums"]["ref_model"] == c["drums"]["js_model"],
    }
    if th["maps"] is not None and c["maps"] and "max_abs" in c["maps"]:
        checks["maps"] = c["maps"]["max_abs"] <= th["maps"]
    c["checks"] = checks
    c["pass"] = all(checks.values())
    return c


# ----------------------------------------------------------------------------- report
def fmt_notes(s):
    if not s:
        return "–"
    t = f"F1 {s['f1']:.3f} ({s['matched']}/{s['ref']} ref, {s['got']} js)"
    if "onset_mean_ms" in s:
        t += f", Δon mean {s['onset_mean_ms']:+.2f} ms, p95 {s['onset_abs_p95_ms']:.2f} ms, exact {s.get('exact', 0)}"
    return t


def write_report(results, path_md, meta):
    lines = [
        "# Parity report: Dettato reference (PyTorch) vs Song X-ray (ONNX Runtime Web + JS)",
        "",
        f"Generated by `parity/parity.py` on {time.strftime('%Y-%m-%d %H:%M')} — CPU only. {meta}",
        "",
        "Modes: **js-same** = the browser pipeline under Node/onnxruntime-web WASM fed the reference's own resampled samples (isolates",
        "models + JS port); **js** = the same pipeline doing its own decode/resampling; **browser** = the result the page produced in",
        "headless Chromium. Notes match on equal pitch and onsets within 50 ms; *exact* = onset and offset within 1 ms.",
        "",
        "| clip | mode | maps max abs diff | pitched notes | track agreement | track names | drums | result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for clip, r in results.items():
        for c in r["comparisons"]:
            m = c["maps"]
            ms = f"{m['max_abs']:.1e}" if m and "max_abs" in m else "–"
            tr = c["tracks"]
            lines.append(
                f"| {clip} | {c['mode']} | {ms} | {c['pitched_notes']['f1']:.3f} | {tr['agreement']:.3f} | "
                f"{'same' if tr.get('names_equal') else 'differ'} | {c['drums']['f1']:.3f} ({c['drums']['js_model']}) | "
                f"{'PASS' if c['pass'] else 'FAIL'} |"
            )
    lines.append("")
    for clip, r in results.items():
        lines += [f"## {clip}", "", f"Reference: {r['ref_summary']} ({r['ref_time_s']:.1f} s on CPU)", ""]
        for c in r["comparisons"]:
            lines += [f"**{c['mode']}** ({c['time_s']:.1f} s):", ""]
            if c["maps"]:
                lines.append(
                    f"- transcriber maps: {json.dumps({k: round(v, 7) if isinstance(v, float) else v for k, v in c['maps'].items()})}"
                )
            for key in ("transcriber_notes", "repaired_notes", "pitched_notes", "drums"):
                if key in c:
                    lines.append(f"- {key.replace('_', ' ')}: {fmt_notes(c[key])}")
                    s = c[key]
                    if s.get("only_ref") or s.get("only_js"):
                        lines.append(
                            f"  - diverge: only in ref {s.get('only_ref', [])} · only in JS {s.get('only_js', [])} ([onset s, pitch])"
                        )
            tr = c["tracks"]
            lines.append(f"- tracks: ref {tr['ref_tracks']} · JS {tr['js_tracks']} · agreement {tr['agreement']:.3f}")
            lines.append(f"- drums model: ref {c['drums']['ref_model']} · JS {c['drums']['js_model']}")
            lines.append(f"- checks: {c['checks']} → **{'PASS' if c['pass'] else 'FAIL'}**")
            lines.append("")
    with open(path_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--browser", action="store_true", help="also compare parity/out/<clip>/browser.json (tests/e2e/browser.mjs)")
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--report", default=os.path.join(ROOT, "parity", "REPORT.md"))
    a = ap.parse_args()
    clips = a.clips or [os.path.join(ROOT, "clips", f) for f in ("band.wav", "duet.wav", "piano.wav")]
    if not a.no_variants:
        clips = ensure_variants(clips)
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for path in clips:
        name = os.path.basename(path)
        outdir = os.path.join(OUT, name.replace(".", "_"))
        os.makedirs(outdir, exist_ok=True)
        print(f"== {name}", file=sys.stderr)
        ref = run_reference(path, outdir)
        json.dump(ref, open(os.path.join(outdir, "ref.json"), "w"))
        comps = []
        for mode in ("js-same", "js"):
            got = run_js(path, outdir, mode, a.threads)
            comps.append(compare(ref, got, outdir, mode))
        bpath = os.path.join(outdir, "browser.json")
        if a.browser and os.path.exists(bpath):
            got = json.load(open(bpath))
            got.setdefault("time_s", got.get("info", {}).get("wallS", 0.0))
            comps.append(compare(ref, got, outdir, "browser"))
        summary = ", ".join(f"{t['name']} ({len(t['notes'])})" for t in ref["tracks"])
        results[name] = dict(ref_summary=summary, ref_time_s=ref["time_s"], comparisons=comps)
        for c in comps:
            print(
                f"  {c['mode']:<8} notes F1 {c['pitched_notes']['f1']:.3f}  tracks {c['tracks']['agreement']:.3f}  "
                f"drums F1 {c['drums']['f1']:.3f}  {'PASS' if c['pass'] else 'FAIL'}  {c['checks']}",
                file=sys.stderr,
            )
    json.dump(results, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    import torch

    write_report(results, a.report, f"torch {torch.__version__}, {os.cpu_count()} CPU threads available.")
    n_pass = sum(c["pass"] for r in results.values() for c in r["comparisons"])
    n_all = sum(len(r["comparisons"]) for r in results.values())
    print(f"\n{n_pass}/{n_all} comparisons PASS  -> {os.path.relpath(a.report, ROOT)}", file=sys.stderr)
    return 0 if n_pass == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
