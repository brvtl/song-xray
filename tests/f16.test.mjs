import test from "node:test";
import assert from "node:assert/strict";
import { toF16, fromF16, roundF16 } from "../web/js/f16.js";

test("float16 round trip of representable values is exact", () => {
  for (const v of [0, 1, -1, 0.5, 65504, -65504, 6.103515625e-5, 5.960464477539063e-8, 1.5, 1000.5]) {
    assert.equal(fromF16(toF16(v)), v);
  }
});

test("float16 rounds to nearest even and saturates to infinity", () => {
  assert.equal(roundF16(1 + 2 ** -11), 1); // halfway between 1 and 1+2^-10 -> even (1)
  assert.equal(roundF16(1 + 3 * 2 ** -11), 1 + 2 * 2 ** -10); // halfway -> even mantissa
  assert.equal(roundF16(1e6), Infinity);
  assert.equal(roundF16(-1e6), -Infinity);
  assert.ok(Number.isNaN(roundF16(NaN)));
});
