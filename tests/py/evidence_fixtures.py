"""Extra reference arrays for tests/evidence.test.mjs: EvidenceRepairer.apply on each clip's saved edits at lowered thresholds,
so the insertion path (insert filter, kept-vs-insertion suppression, dedup among insertions) is exercised on real proposals.

    .venv/bin/python tests/py/evidence_fixtures.py        # -> fixtures/<clip>/evidence/apply_<tag>_*.npy
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
from dettato.evidence.repair import NOTE_DT, EvidenceRepairer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CASES = {"lo": (0.2, 0.05), "all": (0.0, 0.0)}


def main():
    for clip in sys.argv[1:] or ["band", "duet", "piano"]:
        d = os.path.join(ROOT, "fixtures", clip)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        ld = lambda key: np.load(os.path.join(d, f"{key}.npy"))  # noqa: E731
        k = np.zeros(len(ld("rep_in_on")), NOTE_DT)
        for f in ("pitch", "vel", "on", "off"):
            k[f] = ld(f"rep_in_{f}")
        ins = [tuple(r) for r in ld("rep_ins").tolist()]
        ins = [(a, b, int(p), s) for a, b, p, s in ins]
        e = (ld("rep_pk"), ld("rep_on_r"), ld("rep_off_r"), ins)
        out = os.path.join(d, "evidence")
        os.makedirs(out, exist_ok=True)
        for tag, (kt, it) in CASES.items():
            rep = EvidenceRepairer.apply(SimpleNamespace(keep_th=kt, insert_th=it), k, e)
            for f, v in rep.items():
                np.save(os.path.join(out, f"apply_{tag}_{f}.npy"), v)
            print(clip, tag, len(rep["onset"]), "notes,", int(rep["inserted"].sum()), "inserted", file=sys.stderr)


if __name__ == "__main__":
    main()
