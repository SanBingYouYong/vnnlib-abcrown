import pytest

from vnnlib_abcrown.query import (
    FALSE, TRUE, Unsupported, conjuncts, evaluate, parse_query, substitute, to_dnf)

from helpers import global_query

HEADER = "(vnnlib-version <2.0>)\n"


def net(dtype="real", extra=""):
    return (f"{HEADER}(declare-network n (declare-input n_X {dtype} [2, 3]) {extra}"
            f"(declare-output n_Y {dtype} [1]))\n")


def test_parse_two_network_query():
    q = parse_query(global_query(2, 3, 0.1, 0.5, output_index=1, upper=True))
    assert set(q.networks) == {"f", "f_2"}
    assert q.networks["f_2"].equal_to == "f"
    assert q.networks["f"].equal_to is None
    assert q.networks["f"].input.shape == [2]
    assert q.networks["f"].output.size == 3
    parts = conjuncts(q.formula)
    assert len(parts) == 2 * 2 + 1 + 2 * 2 * 2
    # (> (+ f_2_Y[1] (- f_Y[1])) 0.5) becomes 0.5 - f_2_Y[1] + f_Y[1] < 0
    out = [g for g in parts if any(k[1] == "Y" for k in g[1][0])]
    assert out == [("atom", ({("f_2", "Y", 1): -1.0, ("f", "Y", 1): 1.0}, 0.5), "<")]


def test_multi_dimensional_indices_are_row_major():
    q = parse_query(net() + "(assert (<= n_X[1, 2] 0.0))")
    assert conjuncts(q.formula) == [("atom", ({("n", "X", 5): 1.0}, 0.0), "<=")]


def test_vehicle_capitalised_real_is_accepted():
    q = parse_query(net("Real") + "(assert (>= n_Y[0] 1.0))")
    assert conjuncts(q.formula) == [("atom", ({("n", "Y", 0): -1.0}, 1.0), "<=")]


def test_disjunction_is_kept_per_assertion():
    q = parse_query(net() + "(assert (or (<= n_Y[0] 1.0) (and (<= n_X[0, 0] 0.0) (== n_Y[0] 2.0))))")
    (part,) = conjuncts(q.formula)
    assert part[0] == "or" and [len(c[1]) for c in part[1]] == [1, 2]
    assert part[1][1][1][1] == ("atom", ({("n", "Y", 0): 1.0}, -2.0), "==")


@pytest.mark.parametrize("text", [
    net("float32") + "(assert (<= n_Y[0] 1.0))",
    net() + "(assert (!= n_Y[0] 1.0))",
    net() + "(assert (<= (* n_X[0, 0] n_X[0, 1]) 1.0))",
    net(extra='(declare-hidden n_H real [1] "h") ') + "(assert (<= n_H[0] 1.0))",
    "(declare-network n (declare-input n_X real [1]) (declare-output n_Y real [1]))",
])
def test_unsupported_queries_are_rejected(text):
    with pytest.raises(Unsupported):
        parse_query(text)


def test_substitute_evaluate_and_dnf():
    a = ("atom", ({"x": 1.0}, -1.0), "<=")   # x <= 1
    b = ("atom", ({"y": 1.0}, 0.0), "<")     # y < 0
    not_b = ("atom", ({"y": -1.0}, 0.0), "<=")
    f = ("and", [("or", [a, b]), ("or", [a, not_b])])
    assert len(to_dnf(f)) == 4
    assert substitute(f, {"x": 0.5}) == TRUE
    assert substitute(("and", [a, b]), {"x": 2.0}) == FALSE
    assert evaluate(f, {"x": 2.0, "y": -1.0}) is False
    assert evaluate(b, {"y": 0.0}) is False
    assert evaluate(b, {"y": 0.0}, tol=1e-9) is True
