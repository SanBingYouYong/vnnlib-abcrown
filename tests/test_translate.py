"""Translation tests with a fake α,β-CROWN, so they run without a GPU."""

import io
import re

import numpy as np
import onnxruntime as ort

from vnnlib_abcrown.cli import print_solution
from vnnlib_abcrown.translate import solve
from vnnlib_abcrown.vnnlib import parse_query

from helpers import global_query, make_mlp, mlp_forward, mlp_weights


def read_vnnlib_box(path, n):
    lo, hi = np.empty(n), np.empty(n)
    for line in open(path):
        if line.startswith("(assert (>= X_"):
            i, v = line[len("(assert (>= X_"):].rstrip(")\n").split()
            lo[int(i)] = float(v)
        elif line.startswith("(assert (<= X_"):
            i, v = line[len("(assert (<= X_"):].rstrip(")\n").split()
            hi[int(i)] = float(v)
    return lo, hi


ATOM_RE = re.compile(r"\((<=|>=) Y_(\d+) (\S+?)\)")


def parse_output_spec(text):
    """The clauses of a VNN-LIB 1.0 output specification as written by `write_vnnlib`."""
    lines = [l for l in text.splitlines() if "Y_" in l and not l.startswith("(declare")]
    if "(assert (or" in text:
        return [[(op, int(j), float(c)) for op, j, c in ATOM_RE.findall(l)]
                for l in lines if l.strip().startswith("(and")]
    return [[(op, int(j), float(c)) for l in lines for op, j, c in ATOM_RE.findall(l)]]


class FakeAbcrown:
    """Answers `status`; for "sat", searches the box by random sampling for a
    point where the spec network violates every atom of some clause."""

    def __init__(self, status="sat", seed=0):
        self.status = status
        self.calls = []
        self.rng = np.random.default_rng(seed)

    def __call__(self, onnx_path, vnnlib_path, n):
        self.calls.append((onnx_path, open(vnnlib_path).read()))
        if self.status != "sat":
            return self.status, None
        lo, hi = read_vnnlib_box(vnnlib_path, n)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        shape = [1] + list(sess.get_inputs()[0].shape[1:])
        clauses = parse_output_spec(open(vnnlib_path).read())
        for _ in range(20000):
            z = self.rng.uniform(lo, hi)
            if self.rng.random() < 0.5:
                z = np.where(self.rng.random(n) < 0.5, lo, hi)
            y = sess.run(None, {name: z.reshape(shape).astype(np.float32)})[0][0]
            if any(all(y[j] <= c if op == "<=" else y[j] >= c for op, j, c in clause)
                   for clause in clauses):
                return "sat", list(z)
        return "timeout", None


def mlp(tmp_path, seed=0):
    w = mlp_weights([2, 6, 2], seed=seed)
    return w, make_mlp(str(tmp_path / f"mlp{seed}.onnx"), w)


def test_global_query_counterexample_is_valid(tmp_path):
    w, path = mlp(tmp_path)
    q = parse_query(global_query(2, 2, 0.2, 0.01, output_index=0, upper=True))
    fake = FakeAbcrown()
    sol = solve(q, {"f": path, "f_2": path}, str(tmp_path), fake)
    assert sol.status == "sat"
    x = [sol.values[("f", "X", i)] for i in range(2)]
    y = [sol.values[("f_2", "X", i)] for i in range(2)]
    assert np.all(np.abs(np.subtract(y, x)) <= 0.2 + 1e-9)
    assert np.all((0 <= np.array(x + y)) & (np.array(x + y) <= 1))
    d = mlp_forward(w, y)[0] - mlp_forward(w, x)[0]
    assert d > 0.01 - 1e-4
    out = io.StringIO()
    print_solution(sol, q, out)
    lines = out.getvalue().splitlines()
    assert lines[0] == "sat" and lines[1] == "f_X Real [2]" and len(lines) == 1 + 4 * 3


def test_refinement_adds_violated_input_constraints(tmp_path):
    _, path = mlp(tmp_path)
    # A box plus x0 + x1 <= 1.5, which does not fit the box.
    q = parse_query(
        "(declare-network f (declare-input f_X Real [2]) (declare-output f_Y Real [2]))\n"
        "(assert (and (<= 0.0 f_X[0]) (<= f_X[0] 1.0)))\n"
        "(assert (and (<= 0.0 f_X[1]) (<= f_X[1] 1.0)))\n"
        "(assert (<= (+ f_X[0] f_X[1]) 1.5))\n"
        "(assert (>= f_Y[0] -100.0))\n")
    calls = []

    def corner_then_unsat(onnx_path, vnnlib_path, n):
        calls.append(open(vnnlib_path).read())
        if len(calls) == 1:
            return "sat", [1.0, 1.0]  # violates x0 + x1 <= 1.5
        return "unsat", None

    sol = solve(q, {"f": path}, str(tmp_path), corner_then_unsat)
    assert sol.status == "unsat"
    # Round 1 checks only the output; round 2 adds the violated input constraint.
    assert [text.count("(declare-const Y_") for text in calls] == [1, 2]
    assert " 1.5))" in calls[1] and " 1.5))" not in calls[0]


def test_unsat_and_timeout_are_passed_through(tmp_path):
    _, path = mlp(tmp_path)
    q = parse_query(global_query(2, 2, 0.2, 0.01, output_index=1, upper=False))
    for status in ("unsat", "timeout"):
        assert solve(q, {"f": path, "f_2": path}, str(tmp_path), FakeAbcrown(status)).status == status


def test_pinned_network_is_evaluated_not_verified(tmp_path):
    w, path = mlp(tmp_path)
    text = global_query(2, 2, 10.0, 0.0, output_index=0, upper=True)
    text += "(assert (== f_X[0] 0.25))\n(assert (== f_X[1] 0.75))\n"
    q = parse_query(text)
    fake = FakeAbcrown("unsat")
    solve(q, {"f": path, "f_2": path}, str(tmp_path), fake)
    # Only f_2 is free, with a plain box, so the spec network is f_2 itself plus a layer.
    (onnx_path, vnnlib), = fake.calls
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].name == "input"
    assert vnnlib.count("declare-const X_") == 2
    # f_2_Y[0] - f_Y[0] > 0 becomes the row -f_2_Y[0] <= -f([0.25, 0.75])[0].
    c = float(vnnlib.strip().splitlines()[-1].split()[-1].rstrip(")"))
    assert np.isclose(c, -mlp_forward(w, [0.25, 0.75])[0], atol=1e-5)


def test_all_networks_pinned_needs_no_solver(tmp_path):
    w, path = mlp(tmp_path)
    y0 = mlp_forward(w, [0.5, 0.5])[0]
    for bound, expected in ((y0 - 1, "sat"), (y0 + 1, "unsat")):
        q = parse_query(
            "(declare-network f (declare-input f_X Real [2]) (declare-output f_Y Real [2]))\n"
            f"(assert (== f_X[0] 0.5))\n(assert (== f_X[1] 0.5))\n(assert (>= f_Y[0] {bound}))\n")
        assert solve(q, {"f": path}, str(tmp_path), None).status == expected


def test_mixed_input_output_constraint(tmp_path):
    w, path = mlp(tmp_path)
    q = parse_query(
        "(declare-network f (declare-input f_X Real [2]) (declare-output f_Y Real [2]))\n"
        "(assert (and (<= 0.0 f_X[0]) (<= f_X[0] 1.0)))\n"
        "(assert (and (<= 0.0 f_X[1]) (<= f_X[1] 1.0)))\n"
        "(assert (or (>= (- f_Y[0] f_X[0]) 0.1) (<= (+ f_Y[1] f_X[1]) -5.0)))\n")
    sol = solve(q, {"f": path}, str(tmp_path), FakeAbcrown())
    assert sol.status == "sat"
    x = np.array([sol.values[("f", "X", i)] for i in range(2)])
    y = mlp_forward(w, x)
    assert y[0] - x[0] >= 0.1 - 1e-4 or y[1] + x[1] <= -5 + 1e-4


def identity_net(tmp_path):
    w = [(np.eye(2), np.zeros(2))]
    return make_mlp(str(tmp_path / "identity.onnx"), w)


STRICT_QUERY = (
    "(declare-network f (declare-input f_X Real [2]) (declare-output f_Y Real [2]))\n"
    "(assert (and (<= 0.0 f_X[0]) (<= f_X[0] 1.0)))\n"
    "(assert (and (<= 0.0 f_X[1]) (<= f_X[1] 1.0)))\n"
    "(assert (> f_Y[0] f_Y[1]))\n")


def test_tied_counterexample_is_not_reported_as_sat(tmp_path):
    # A point with f_Y[0] == f_Y[1] satisfies the relaxed `>=` that α,β-CROWN
    # checks, but not the strict `>` of the query. Then no counterexample with
    # a margin exists either, so the answer must be unknown, not sat or unsat.
    calls = []

    def tie_then_unsat(onnx_path, vnnlib_path, n):
        calls.append(open(vnnlib_path).read())
        return ("sat", [0.5, 0.5]) if len(calls) == 1 else ("unsat", None)

    q = parse_query(STRICT_QUERY)
    assert solve(q, {"f": identity_net(tmp_path)}, str(tmp_path), tie_then_unsat).status == "timeout"
    assert len(calls) == 2
    assert "(<= Y_0 -0.0)" in calls[0] and "(<= Y_0 -1e-06)" in calls[1]


def test_margin_round_finds_a_genuine_counterexample(tmp_path):
    answers = iter([("sat", [0.5, 0.5]), ("sat", [0.6, 0.5])])
    q = parse_query(STRICT_QUERY)
    sol = solve(q, {"f": identity_net(tmp_path)}, str(tmp_path), lambda *a: next(answers))
    assert sol.status == "sat"
    assert sol.values[("f", "Y", 0)] > sol.values[("f", "Y", 1)]
