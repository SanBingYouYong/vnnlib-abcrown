"""Turning linear input constraints into a box that α,β-CROWN accepts.

α,β-CROWN only accepts input regions that are boxes. Vehicle's queries can
constrain linear combinations of the inputs, for example `y[i] - x[i] <= eps`
when a property relates two network applications.

Let `v` be the free input variables. We pick `n = len(v)` linearly independent
constraint directions `a_1 .. a_n` that are bounded on both sides, preferring
narrow ones, and use `z = A v` (rows of `A` are the `a_i`) as the new input.
The constraints on these directions are exactly the box on `z`, and the
network gets `v = A⁻¹ z` as an extra first layer.

Every other constraint direction is either implied by the box, so it is
dropped, or kept as a "leftover" constraint that the caller must enforce
some other way (we route `v` to extra outputs and check it there).
"""

import heapq

import numpy as np

from .vnnlib import Unsupported

# Coefficients smaller than this are treated as zero during elimination.
EPS = 1e-12
# Slack when deciding that a bound is implied by the box.
IMPLIED_TOL = 1e-9


class InputRegion:
    """The input region `{v | lo <= A v <= hi and leftover constraints}`."""

    def __init__(self, keys, a, lo, hi, leftover):
        self.keys = keys
        self.a = a
        self.lo = lo
        self.hi = hi
        self.leftover = leftover  # list of (row over v as {index: coeff}, op, bound)
        self.identity = bool(np.array_equal(a, np.eye(len(keys))))
        self.a_inv = np.eye(len(keys)) if self.identity else np.linalg.inv(a)

    def to_v(self, z):
        z = np.clip(np.asarray(z, dtype=np.float64), self.lo, self.hi)
        return z if self.identity else self.a_inv @ z


def _normalise(row, bound_upper):
    """Scale `row . v <= bound` so that its first coefficient is 1.

    Returns (direction key, row, lower, upper)."""
    first = row[min(row)]
    s = 1.0 / first
    row = {i: c * s for i, c in row.items()}
    key = tuple(sorted((i, round(c, 12)) for i, c in row.items()))
    if s > 0:
        return key, row, -np.inf, bound_upper * s
    return key, row, bound_upper * s, np.inf


def analyse(constraints, keys):
    """Build the input region.

    `constraints` is a list of (coeffs, const, op) meaning `coeffs . v + const op 0`
    with `op` one of "<=", "<", "==" and every variable in `keys`. Strict
    inequalities are relaxed to non-strict ones.

    Returns None if the constraints are obviously infeasible.
    """
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    directions = {}  # key -> [row, lo, hi]
    for coeffs, const, op in constraints:
        row = {index[k]: c for k, c in coeffs.items()}
        sides = [(row, -const)]
        if op == "==":
            sides.append(({i: -c for i, c in row.items()}, const))
        for r, b in sides:
            key, r, lo, hi = _normalise(r, b)
            d = directions.setdefault(key, [r, -np.inf, np.inf])
            d[1], d[2] = max(d[1], lo), min(d[2], hi)

    for _, lo, hi in directions.values():
        if lo > hi + IMPLIED_TOL * max(1.0, abs(lo), abs(hi)):
            return None

    # Choose a basis of bounded directions, narrowest first, by sparse
    # Gaussian elimination. Pivot rows are stored reduced against all earlier
    # pivots, so eliminating pivots in creation order terminates.
    candidates = sorted(
        (key for key, (_, lo, hi) in directions.items() if np.isfinite(lo) and np.isfinite(hi)),
        key=lambda key: (directions[key][2] - directions[key][1], len(key), key),
    )
    pivot_of_var = {}  # variable index -> pivot number
    pivot_rows = []  # (pivot variable, reduced row with pivot coefficient 1)
    basis = []
    for key in candidates:
        if len(basis) == n:
            break
        r = dict(directions[key][0])
        heap = [pivot_of_var[i] for i in r if i in pivot_of_var]
        heapq.heapify(heap)
        seen = set(heap)
        while heap:
            p = heapq.heappop(heap)
            var, prow = pivot_rows[p]
            c = r.get(var, 0.0)
            if abs(c) <= EPS:
                r.pop(var, None)
                continue
            for i, pc in prow.items():
                new = r.get(i, 0.0) - c * pc
                if abs(new) <= EPS:
                    r.pop(i, None)
                else:
                    r[i] = new
                    q = pivot_of_var.get(i)
                    if q is not None and q not in seen:
                        seen.add(q)
                        heapq.heappush(heap, q)
        if not r:
            continue
        var = max(r, key=lambda i: abs(r[i]))
        c = r[var]
        pivot_of_var[var] = len(pivot_rows)
        pivot_rows.append((var, {i: v / c for i, v in r.items()}))
        basis.append(key)

    if len(basis) < n:
        free = sorted(set(range(n)) - set(pivot_of_var))
        names = ", ".join(str(keys[i]) for i in free[:5])
        raise Unsupported(
            f"the input region is unbounded in {n - len(basis)} direction(s) "
            f"(e.g. involving {names}); α,β-CROWN needs bounded inputs"
        )

    # Single-variable directions make A a permutation matrix. Order them by
    # variable so that A is the identity and no extra layer is needed.
    if all(len(key) == 1 for key in basis):
        basis.sort(key=lambda key: key[0][0])

    a = np.zeros((n, n))
    lo = np.empty(n)
    hi = np.empty(n)
    for j, key in enumerate(basis):
        row, lo[j], hi[j] = directions[key]
        for i, c in row.items():
            a[j, i] = c

    region = InputRegion(keys, a, lo, hi, [])

    # Keep the sides of non-basis directions that the box does not imply.
    in_basis = set(basis)
    for key, (row, dlo, dhi) in directions.items():
        if key in in_basis:
            continue
        w = np.zeros(n)
        for i, c in row.items():
            w += c * region.a_inv[i]
        zmin = np.sum(np.where(w > 0, w * lo, w * hi))
        zmax = np.sum(np.where(w > 0, w * hi, w * lo))
        if np.isfinite(dlo) and zmin < dlo - IMPLIED_TOL * max(1.0, abs(dlo)):
            region.leftover.append((row, ">=", dlo))
        if np.isfinite(dhi) and zmax > dhi + IMPLIED_TOL * max(1.0, abs(dhi)):
            region.leftover.append((row, "<=", dhi))
    return region
