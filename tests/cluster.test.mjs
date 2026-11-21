// cluster.js against scipy / NumPy references (fixtures/band/grouper/cluster_refs.json from tests/py/grouper_fixtures.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { ROOT } from "./fixtures.mjs";
import {
  erf,
  fclusterDistance,
  linearSumAssignment,
  linkageAverage,
  npSum,
  roundHalfEven,
  argsortStable,
} from "../web/js/cluster.js";

const REFS = path.join(ROOT, "fixtures", "band", "grouper", "cluster_refs.json");
const refs = fs.existsSync(REFS) ? JSON.parse(fs.readFileSync(REFS, "utf8")) : null;
const skip = refs ? false : "run tests/py/grouper_fixtures.py first";

test("roundHalfEven", () => {
  assert.deepEqual([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, -2.6].map(roundHalfEven), [0, 2, 2, -0, -2, -2, 2, -3]);
});

test("npSum: pairwise blocks of 8 / 128 (reference values computed by NumPy's algorithm)", () => {
  const a = Float64Array.from({ length: 300 }, (_, i) => 1 / (i + 1));
  let seq = 0;
  for (const v of a) seq += v;
  assert.ok(Math.abs(npSum(a) - seq) < 1e-13);
  assert.equal(npSum([1, 2, 3]), 6);
  assert.equal(npSum(Float32Array.of(0.1, 0.2, 0.3), 0, 3, 1, true), Math.fround(Math.fround(0.1 + 0.2) + 0.3));
});

test("argsortStable keeps ties in index order", () => {
  assert.deepEqual([...argsortStable(5, (i) => [2, 1, 2, 1, 0][i])], [4, 1, 3, 0, 2]);
});

test("linkageAverage + fclusterDistance match scipy (continuous, tied and float32 distances)", { skip }, () => {
  let nZ = 0;
  for (const c of refs.linkage) {
    const Z = linkageAverage(c.y, c.n);
    const flat = c.Z.flat();
    for (let i = 0; i < flat.length; i++) {
      assert.ok(Math.abs(Z[i] - flat[i]) <= 1e-12 * Math.max(1, Math.abs(flat[i])), `n=${c.n} Z[${i}]`);
    }
    c.t.forEach((t, k) => assert.deepEqual([...fclusterDistance(Z, c.n, t)], c.labels[k], `n=${c.n} t=${t}`));
    nZ++;
  }
  assert.ok(nZ >= 30);
});

test("linearSumAssignment matches scipy (square, wide, tall, ties, zero rows)", { skip }, () => {
  for (const c of refs.lsap) {
    const nr = c.cost.length;
    const nc = c.cost[0].length;
    const { rows, cols } = linearSumAssignment(c.cost.flat(), nr, nc);
    assert.deepEqual([...rows], c.rows, `${nr}x${nc}`);
    assert.deepEqual([...cols], c.cols, `${nr}x${nc} ${JSON.stringify(c.cost)}`);
  }
});

test("erf matches scipy.special.erf to 5e-15", { skip }, () => {
  let mx = 0;
  refs.erf.x.forEach((x, i) => (mx = Math.max(mx, Math.abs(erf(x) - refs.erf.y[i]))));
  assert.ok(mx < 5e-15, `max |diff| ${mx}`);
});
