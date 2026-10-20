from vnnlib_abcrown.vnnlib import (
    FALSE, TRUE, conjuncts, evaluate, negate, parse_query, substitute, to_dnf, tokenize)

from helpers import global_query


def test_tokenizer_keeps_indexed_variables_together():
    assert tokenize("(<= x_X[0, 1] 2.5) ; comment") == ["(", "<=", "x_X[0, 1]", "2.5", ")"]


def test_parse_two_network_query():
    q = parse_query(global_query(2, 3, 0.1, 0.5, output_index=1, upper=True))
    assert set(q.networks) == {"f", "f_2"}
    assert q.networks["f_2"].equal_to == "f"
    assert q.networks["f"].input.shape == [2]
    assert q.networks["f"].output.size == 3
    parts = conjuncts(q.formula)
    assert len(parts) == 2 * 2 + 1 + 2 * 2 * 2
    # (> (+ f_2_Y[1] (- f_Y[1])) 0.5) becomes 0.5 - f_2_Y[1] + f_Y[1] < 0
    out = [g for g in parts if any(k[1] == "Y" for k in g[1][0])]
    assert out == [("atom", ({("f_2", "Y", 1): -1.0, ("f", "Y", 1): 1.0}, 0.5), "<")]


def test_multi_dimensional_indices_are_row_major():
    q = parse_query("(declare-network n (declare-input n_X Real [2, 3]) "
                    "(declare-output n_Y Real [1])) (assert (<= n_X[1, 2] 0.0))")
    assert q.formula == ("and", [("atom", ({("n", "X", 5): 1.0}, 0.0), "<=")])


def test_substitute_evaluate_and_dnf():
    a = ("atom", ({"x": 1.0}, -1.0), "<=")   # x <= 1
    b = ("atom", ({"y": 1.0}, 0.0), "<")     # y < 0
    f = ("and", [("or", [a, b]), ("or", [a, negate(b)])])
    assert len(to_dnf(f)) == 4
    assert substitute(f, {"x": 0.5}) == TRUE
    assert substitute(("and", [a, b]), {"x": 2.0}) == FALSE
    assert evaluate(f, {"x": 2.0, "y": -1.0}) is False
    assert evaluate(b, {"y": 0.0}) is False
    assert evaluate(b, {"y": 0.0}, tol=1e-9) is True
