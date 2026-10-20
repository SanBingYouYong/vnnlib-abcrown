import numpy as np
import pytest

from vnnlib_abcrown.inputs import analyse
from vnnlib_abcrown.query import Unsupported


def box(key, lo, hi):
    return [({key: 1.0}, -hi, "<="), ({key: -1.0}, lo, "<=")]


def linked(n, eps):
    """x, y in [0, 1]^n with |y_i - x_i| <= eps, as Vehicle emits it."""
    keys = [("f", "X", i) for i in range(n)] + [("g", "X", i) for i in range(n)]
    cons = []
    for i in range(n):
        x, y = keys[i], keys[n + i]
        cons += box(x, 0.0, 1.0) + box(y, 0.0, 1.0)
        cons.append(({y: 1.0, x: -1.0}, -eps, "<="))
        cons.append(({y: -1.0, x: 1.0}, -eps, "<="))
    return keys, cons


def test_plain_box_needs_no_change_of_variables():
    keys = [("f", "X", i) for i in range(3)]
    cons = [c for i, k in enumerate(keys) for c in box(k, -i, i + 1.0)]
    region = analyse(cons, keys)
    assert region.identity
    assert np.allclose(region.lo, [0, -1, -2]) and np.allclose(region.hi, [1, 2, 3])
    assert region.leftover == []


def test_linked_inputs_are_reparameterised():
    n, eps = 4, 0.1
    keys, cons = linked(n, eps)
    region = analyse(cons, keys)
    assert not region.identity
    # The n narrow difference directions must be in the basis.
    widths = np.sort(region.hi - region.lo)
    assert np.allclose(widths[:n], 2 * eps) and np.allclose(widths[n:], 1.0)
    # y in [0, 1] is not implied by x in [0, 1], |y - x| <= eps: one side each.
    assert len(region.leftover) == 2 * n
    # Every point of the box maps to a point satisfying the basis constraints.
    rng = np.random.default_rng(0)
    for _ in range(100):
        z = rng.uniform(region.lo, region.hi)
        v = region.to_v(z)
        assert np.all(np.abs(v[n:] - v[:n]) <= eps + 1e-12)
        assert np.allclose(region.a @ v, z)


def test_implied_constraints_are_dropped():
    keys = [("f", "X", 0), ("f", "X", 1)]
    cons = box(keys[0], 0, 1) + box(keys[1], 0, 1)
    cons.append(({keys[0]: 1.0, keys[1]: 1.0}, -3.0, "<="))  # x0 + x1 <= 3: implied
    cons.append(({keys[0]: 1.0, keys[1]: 1.0}, -1.5, "<="))  # x0 + x1 <= 1.5: not implied
    region = analyse(cons, keys)
    assert region.identity
    assert len(region.leftover) == 1
    row, op, bound = region.leftover[0]
    assert op == "<=" and bound == pytest.approx(1.5)


def test_equalities_and_infeasibility():
    keys = [("f", "X", 0)]
    region = analyse([({keys[0]: 2.0}, -1.0, "==")], keys)
    assert region.lo[0] == pytest.approx(0.5) and region.hi[0] == pytest.approx(0.5)
    assert analyse(box(keys[0], 1.0, 0.0), keys) is None


def test_unbounded_inputs_are_rejected():
    keys = [("f", "X", 0), ("f", "X", 1)]
    with pytest.raises(Unsupported):
        analyse(box(keys[0], 0, 1), keys)
