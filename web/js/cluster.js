// Clustering and numeric helpers for the grouper's decode, ported from the exact algorithms the reference calls:
// scipy.cluster.hierarchy.linkage(y, "average") (nearest-neighbour chain, scipy/cluster/hierarchy/_hierarchy.pyx),
// fcluster(Z, t, "distance"), scipy.optimize.linear_sum_assignment (rectangular_lsap.cpp, shortest augmenting path),
// scipy.special.erf, and NumPy's pairwise summation (the order np.sum / np.linalg.norm add a contiguous row in).

const PW_BLOCKSIZE = 128;

/** Round half to even (NumPy / torch .round(), Python round()). */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function pairwise64(a, off, n, stride) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[off + i * stride];
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    const r = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let j = 0; j < 8; j++) r[j] = a[off + j * stride];
    const m = n - (n % 8);
    for (let i = 8; i < m; i += 8) for (let j = 0; j < 8; j++) r[j] += a[off + (i + j) * stride];
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]));
    for (let i = m; i < n; i++) res += a[off + i * stride];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwise64(a, off, n2, stride) + pairwise64(a, off + n2 * stride, n - n2, stride);
}

function pairwise32(a, off, n, stride) {
  const f = Math.fround;
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f(res + a[off + i * stride]);
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    const r = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let j = 0; j < 8; j++) r[j] = f(a[off + j * stride]);
    const m = n - (n % 8);
    for (let i = 8; i < m; i += 8) for (let j = 0; j < 8; j++) r[j] = f(r[j] + a[off + (i + j) * stride]);
    let res = f(f(f(r[0] + r[1]) + f(r[2] + r[3])) + f(f(r[4] + r[5]) + f(r[6] + r[7])));
    for (let i = m; i < n; i++) res = f(res + a[off + i * stride]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f(pairwise32(a, off, n2, stride) + pairwise32(a, off + n2 * stride, n - n2, stride));
}

/**
 * NumPy's sum of `n` elements of `a` starting at `off` with `stride` (np.add.reduce over a contiguous axis: pairwise
 * summation from 0, blocks of 128 with 8 accumulators). `f32` sums in float32, as NumPy does for float32 arrays.
 * @param {ArrayLike<number>} a
 * @param {number} [off]
 * @param {number} [n]
 * @param {number} [stride]
 * @param {boolean} [f32]
 * @returns {number}
 */
export function npSum(a, off = 0, n = a.length - off, stride = 1, f32 = false) {
  return f32 ? pairwise32(a, off, n, stride) : pairwise64(a, off, n, stride);
}

/**
 * np.linalg.norm of one row (float64): sqrt of the pairwise sum of squares.
 * @param {ArrayLike<number>} a
 * @param {number} off
 * @param {number} n
 * @returns {number}
 */
export function rowNorm(a, off, n) {
  const sq = new Float64Array(n);
  for (let i = 0; i < n; i++) sq[i] = a[off + i] * a[off + i];
  return Math.sqrt(pairwise64(sq, 0, n, 1));
}

/**
 * The same in float32 (np.linalg.norm of a float32 row).
 * @param {ArrayLike<number>} a
 * @param {number} off
 * @param {number} n
 * @returns {number}
 */
export function rowNorm32(a, off, n) {
  const sq = new Float32Array(n);
  for (let i = 0; i < n; i++) sq[i] = Math.fround(a[off + i] * a[off + i]);
  return Math.fround(Math.sqrt(pairwise32(sq, 0, n, 1)));
}

/**
 * Stable argsort of `n` keys by `key(i)` ascending.
 * @param {number} n
 * @param {(i: number) => number} key
 * @returns {Int32Array}
 */
export function argsortStable(n, key) {
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => key(a) - key(b) || a - b);
  return Int32Array.from(idx);
}

// ----------------------------------------------------------------------------- erf

const TWO_OVER_SQRT_PI = 1.1283791670955126;

/**
 * Error function, |error| < 5e-15 (scipy.special.erf to the last few bits). Uses the all-positive series
 * erf(x) = 2/sqrt(pi) e^(-x^2) sum_n (2x^2)^n x / (1 3 5 ... (2n+1)), so there is no cancellation; |x| >= 6 is +-1 in double.
 * @param {number} x
 * @returns {number}
 */
export function erf(x) {
  if (Number.isNaN(x)) return NaN;
  const ax = Math.abs(x);
  if (ax >= 6) return x > 0 ? 1 : -1;
  const x2 = 2 * ax * ax;
  let term = ax;
  let sum = ax;
  for (let n = 1; n < 500; n++) {
    term *= x2 / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  const r = TWO_OVER_SQRT_PI * Math.exp(-ax * ax) * sum;
  return x < 0 ? -r : r;
}

// ----------------------------------------------------------------------------- average linkage

function condensedIndex(n, i, j) {
  if (i < j) return n * i - (i * (i + 1)) / 2 + (j - i - 1);
  return n * j - (j * (j + 1)) / 2 + (i - j - 1);
}

function chainMerges(D, n) {
  const Z = new Float64Array((n - 1) * 4);
  const size = new Int32Array(n).fill(1);
  const chain = new Int32Array(n);
  let len = 0;
  for (let k = 0; k < n - 1; k++) {
    if (len === 0) {
      len = 1;
      for (let i = 0; i < n; i++) {
        if (size[i] > 0) {
          chain[0] = i;
          break;
        }
      }
    }
    let x;
    let y = 0;
    let cur;
    for (;;) {
      x = chain[len - 1];
      if (len > 1) {
        y = chain[len - 2];
        cur = D[condensedIndex(n, x, y)];
      } else cur = Infinity;
      for (let i = 0; i < n; i++) {
        if (size[i] === 0 || x === i) continue;
        const d = D[condensedIndex(n, x, i)];
        if (d < cur) {
          cur = d;
          y = i;
        }
      }
      if (len > 1 && y === chain[len - 2]) break;
      chain[len++] = y;
    }
    len -= 2;
    if (x > y) [x, y] = [y, x];
    const nx = size[x];
    const ny = size[y];
    Z.set([x, y, cur, nx + ny], k * 4);
    size[x] = 0;
    size[y] = nx + ny;
    for (let i = 0; i < n; i++) {
      const ni = size[i];
      if (ni === 0 || i === y) continue;
      const a = condensedIndex(n, i, x);
      const b = condensedIndex(n, i, y);
      D[b] = (nx * D[a] + ny * D[b]) / (nx + ny);
    }
  }
  return Z;
}

function relabel(Z, n) {
  const parent = Int32Array.from({ length: 2 * n - 1 }, (_, i) => i);
  const usize = new Int32Array(2 * n - 1).fill(1);
  let next = n;
  const find = (x0) => {
    let x = x0;
    while (parent[x] !== x) x = parent[x];
    let p = x0;
    while (parent[p] !== x) {
      const q = parent[p];
      parent[p] = x;
      p = q;
    }
    return x;
  };
  for (let i = 0; i < n - 1; i++) {
    const xr = find(Z[i * 4]);
    const yr = find(Z[i * 4 + 1]);
    Z[i * 4] = Math.min(xr, yr);
    Z[i * 4 + 1] = Math.max(xr, yr);
    parent[xr] = next;
    parent[yr] = next;
    usize[next] = usize[xr] + usize[yr];
    Z[i * 4 + 3] = usize[next];
    next++;
  }
}

/**
 * scipy.cluster.hierarchy.linkage(y, "average"): nearest-neighbour chain, merges stably sorted by height, relabelled.
 * @param {ArrayLike<number>} y condensed distances (n (n - 1) / 2, row-major upper triangle)
 * @param {number} n number of observations (>= 2)
 * @returns {Float64Array} Z [(n - 1)][4]: cluster a, cluster b, height, size
 */
export function linkageAverage(y, n) {
  const Zu = chainMerges(Float64Array.from(y), n);
  const order = argsortStable(n - 1, (i) => Zu[i * 4 + 2]);
  const Z = new Float64Array((n - 1) * 4);
  for (let r = 0; r < n - 1; r++) Z.set(Zu.subarray(order[r] * 4, order[r] * 4 + 4), r * 4);
  relabel(Z, n);
  return Z;
}

function maxDists(Z, n) {
  const MD = new Float64Array(n);
  const visited = new Uint8Array(2 * n - 1);
  const stack = new Int32Array(n);
  let k = 0;
  stack[0] = 2 * n - 2;
  while (k >= 0) {
    const root = stack[k] - n;
    const lc = Z[root * 4];
    const rc = Z[root * 4 + 1];
    if (lc >= n && !visited[lc]) {
      visited[lc] = 1;
      stack[++k] = lc;
      continue;
    }
    if (rc >= n && !visited[rc]) {
      visited[rc] = 1;
      stack[++k] = rc;
      continue;
    }
    let md = Z[root * 4 + 2];
    if (lc >= n && MD[lc - n] > md) md = MD[lc - n];
    if (rc >= n && MD[rc - n] > md) md = MD[rc - n];
    MD[root] = md;
    k--;
  }
  return MD;
}

/**
 * scipy.cluster.hierarchy.fcluster(Z, t, "distance"): flat clusters numbered 1.. in scipy's traversal order.
 * @param {Float64Array} Z linkage matrix from linkageAverage
 * @param {number} n number of observations
 * @param {number} t cutoff (a cluster's maximal merge height <= t)
 * @returns {Int32Array} labels (1-based)
 */
export function fclusterDistance(Z, n, t) {
  const MC = maxDists(Z, n);
  const T = new Int32Array(n);
  const visited = new Uint8Array(2 * n - 1);
  const stack = new Int32Array(n);
  let nc = 0;
  let leader = -1;
  let k = 0;
  stack[0] = 2 * n - 2;
  while (k >= 0) {
    const root = stack[k] - n;
    const lc = Z[root * 4];
    const rc = Z[root * 4 + 1];
    if (leader === -1 && MC[root] <= t) {
      leader = root;
      nc++;
    }
    if (lc >= n && !visited[lc]) {
      visited[lc] = 1;
      stack[++k] = lc;
      continue;
    }
    if (rc >= n && !visited[rc]) {
      visited[rc] = 1;
      stack[++k] = rc;
      continue;
    }
    if (lc < n) {
      if (leader === -1) nc++;
      T[lc] = nc;
    }
    if (rc < n) {
      if (leader === -1) nc++;
      T[rc] = nc;
    }
    if (leader === root) leader = -1;
    k--;
  }
  return T;
}

// ----------------------------------------------------------------------------- linear sum assignment

function augmentingPath(nc, cost, st, i0) {
  const { u, v, path, row4col, spc, SR, SC, remaining } = st;
  let i = i0;
  let minVal = 0;
  let numRemaining = nc;
  for (let it = 0; it < nc; it++) remaining[it] = nc - it - 1;
  SR.fill(0);
  SC.fill(0);
  spc.fill(Infinity);
  let sink = -1;
  while (sink === -1) {
    let index = -1;
    let lowest = Infinity;
    SR[i] = 1;
    for (let it = 0; it < numRemaining; it++) {
      const j = remaining[it];
      const r = minVal + cost[i * nc + j] - u[i] - v[j];
      if (r < spc[j]) {
        path[j] = i;
        spc[j] = r;
      }
      if (spc[j] < lowest || (spc[j] === lowest && row4col[j] === -1)) {
        lowest = spc[j];
        index = it;
      }
    }
    minVal = lowest;
    if (minVal === Infinity) return [-1, minVal];
    const j = remaining[index];
    if (row4col[j] === -1) sink = j;
    else i = row4col[j];
    SC[j] = 1;
    remaining[index] = remaining[--numRemaining];
  }
  return [sink, minVal];
}

function lsapSolve(nr, nc, cost) {
  const st = {
    u: new Float64Array(nr),
    v: new Float64Array(nc),
    path: new Int32Array(nc).fill(-1),
    col4row: new Int32Array(nr).fill(-1),
    row4col: new Int32Array(nc).fill(-1),
    spc: new Float64Array(nc),
    SR: new Uint8Array(nr),
    SC: new Uint8Array(nc),
    remaining: new Int32Array(nc),
  };
  const { u, v, path, col4row, row4col, spc, SR, SC } = st;
  for (let cur = 0; cur < nr; cur++) {
    const [sink, minVal] = augmentingPath(nc, cost, st, cur);
    if (sink < 0) throw new Error("linear_sum_assignment: cost matrix is infeasible");
    u[cur] += minVal;
    for (let i = 0; i < nr; i++) if (SR[i] && i !== cur) u[i] += minVal - spc[col4row[i]];
    for (let j = 0; j < nc; j++) if (SC[j]) v[j] -= minVal - spc[j];
    let j = sink;
    for (;;) {
      const i = path[j];
      row4col[j] = i;
      const t = col4row[i];
      col4row[i] = j;
      j = t;
      if (i === cur) break;
    }
  }
  return col4row;
}

/**
 * scipy.optimize.linear_sum_assignment(cost) (minimisation; tie behaviour as scipy's rectangular_lsap.cpp).
 * @param {ArrayLike<number>} cost [nr][nc] row-major
 * @param {number} nr
 * @param {number} nc
 * @returns {{rows: Int32Array, cols: Int32Array}} matched (row, col) pairs, rows ascending
 */
export function linearSumAssignment(cost, nr, nc) {
  if (nr === 0 || nc === 0) return { rows: new Int32Array(0), cols: new Int32Array(0) };
  for (let i = 0; i < nr * nc; i++) {
    if (Number.isNaN(cost[i]) || cost[i] === -Infinity) throw new Error("linear_sum_assignment: invalid cost entry");
  }
  if (nc >= nr) {
    const c4r = lsapSolve(nr, nc, Float64Array.from(cost));
    return { rows: Int32Array.from({ length: nr }, (_, i) => i), cols: c4r };
  }
  const t = new Float64Array(nr * nc);
  for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) t[j * nr + i] = cost[i * nc + j];
  const c4r = lsapSolve(nc, nr, t);
  const order = argsortStable(nc, (i) => c4r[i]);
  return { rows: Int32Array.from(order, (o) => c4r[o]), cols: Int32Array.from(order) };
}
