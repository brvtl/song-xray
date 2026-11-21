// Which precision of the grouper's PCEN front reproduces the reference grouping best? For each fixture clip, run the JS grouper
// (front + grouper.onnx + decode) on the reference's own grouper inputs (fixtures/<clip>: y, grp_meta, grp_feat, grp_fsc) and
// compare the decoded groups with the reference's (share of notes in the same group, after matching groups one-to-one).
//
//   node parity/grouper_front_check.mjs band duet piano sousa_30s sousa_full
import fs from "node:fs";
import { npy, manifest, f16 } from "../tests/fixtures.mjs";
import * as GR from "../web/js/grouper.js";
import { linearSumAssignment } from "../web/js/cluster.js";
import { grouperFront, setPcenPrecision } from "../web/js/dsp.js";
import { nodeRuntime } from "../scripts/transcribe.mjs";

const meta = JSON.parse(fs.readFileSync(new URL("../web/models/meta.json", import.meta.url)));
const th = GR.loadTrackHead(
  fs.readFileSync(new URL("../web/models/track_head.bin", import.meta.url)).buffer.slice(0),
  meta.grouper.track_head_shapes,
);
const rt = nodeRuntime(16);

function agreement(refGroups, gotGroups, n) {
  const lab = (groups) => {
    const l = new Int32Array(n).fill(-1);
    groups.forEach(([, ids], g) => ids.forEach((i) => (l[i] = g)));
    return l;
  };
  const a = lab(refGroups);
  const b = lab(gotGroups);
  const R = refGroups.length;
  const G = gotGroups.length;
  const C = new Float64Array(R * G);
  for (let i = 0; i < n; i++) if (a[i] >= 0 && b[i] >= 0) C[a[i] * G + b[i]] += 1;
  const { rows, cols } = linearSumAssignment(
    C.map((v) => -v),
    R,
    G,
  );
  let s = 0;
  for (let i = 0; i < rows.length; i++) s += C[rows[i] * G + cols[i]];
  return s / n;
}

for (const clip of process.argv.slice(2)) {
  const man = manifest(clip);
  const m = npy(clip, "grp_meta");
  const N = m.shape[0];
  const s = npy(clip, "grp_fsc");
  const fsc = s.dtype === "<f2" ? Float32Array.from(s.data, f16) : Float32Array.from(s.data);
  const song = GR.makeSong(
    Float32Array.from(npy(clip, "y").data),
    Float32Array.from(m.data),
    Int8Array.from(npy(clip, "grp_feat").data),
    fsc,
  );
  const on = Float64Array.from({ length: N }, (_, i) => m.data[12 * i]);
  const off = Float64Array.from({ length: N }, (_, i) => m.data[12 * i + 1]);
  const pitch = Int32Array.from({ length: N }, (_, i) => Math.round(m.data[12 * i + 2]));
  const name = (g) => g.map(([k, ids]) => `${meta.instruments.names[k]} ${ids.length}`).join(", ");
  console.log(`${clip}: reference ${name(man.grp_groups)}`);
  for (const prec of ["float32", "float64"]) {
    setPcenPrecision(prec);
    const P = await GR.predict(rt, song, grouperFront, { ...meta.grouper });
    const g = GR.decode(P, [on, off, pitch], th, meta.grouper.decode, meta.instruments.credit);
    console.log(`  ${prec}: agreement ${agreement(man.grp_groups, g, N).toFixed(3)}  ${name(g)}`);
  }
}
