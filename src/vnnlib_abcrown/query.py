"""Reading VNN-LIB 2.0 queries.

Parsing, type checking, linearisation and conversion to DNF are done by the
`vnnlib` package (https://github.com/VNNLIB/VNNLIB-Python). This module turns
its AST into the linear form the rest of the adapter works with.

Supported: networks with one input and one output and no hidden nodes
(theories NH and SIO), element type `real`, and linear assertions without `!=`.
"""

import re

import numpy as np
import vnnlib


class Unsupported(Exception):
    """The query uses something this backend cannot translate."""


################################################################################
# Networks


class Tensor:
    def __init__(self, network, kind, name, shape):
        self.network = network
        self.kind = kind  # "X" for the input, "Y" for the output
        self.name = name
        self.shape = shape
        self.size = int(np.prod(shape)) if shape else 1

    def keys(self):
        return [(self.network, self.kind, i) for i in range(self.size)]


class Network:
    def __init__(self, name, equal_to, input, output):
        self.name = name
        self.equal_to = equal_to
        self.input = input
        self.output = output
        self.onnx_path = None


def _network(n):
    if n.hidden:
        raise Unsupported(f"network {n.name} declares hidden nodes")
    if len(n.inputs) != 1 or len(n.outputs) != 1:
        raise Unsupported(f"network {n.name} must have exactly one input and one output")
    tensors = []
    for kind, decl in (("X", n.inputs[0]), ("Y", n.outputs[0])):
        # α,β-CROWN computes in float32, so its analysis is only claimed to be
        # sound for real arithmetic (see "supports --onnx-element-types").
        if decl.dtype != vnnlib.DType.Real:
            raise Unsupported(f"element type of {decl.name} is {decl.dtype.name}; only real "
                              "is supported")
        tensors.append(Tensor(n.name, kind, decl.name, list(decl.shape)))
    return Network(n.name, n.equal_to or None, *tensors)


################################################################################
# Linear expressions
#
# A linear expression is a pair (coeffs, const) where coeffs maps a variable
# key (network, kind, flat index) to a non-zero float.


def lin_add(a, b, scale=1.0):
    coeffs = dict(a[0])
    for k, v in b[0].items():
        coeffs[k] = coeffs.get(k, 0.0) + scale * v
    return ({k: v for k, v in coeffs.items() if v != 0.0}, a[1] + scale * b[1])


def lin_scale(a, s):
    return ({k: v * s for k, v in a[0].items() if v * s != 0.0}, a[1] * s)


def _linear(expr):
    try:
        lin = expr.to_linear_expr()
    except vnnlib.VNNLibException as e:
        raise Unsupported(str(e)) from None
    coeffs = {}
    for t in lin.terms:
        v = t.var
        kind = "X" if v.kind == vnnlib.SymbolKind.Input else "Y"
        flat = int(np.ravel_multi_index(list(v.indices), list(v.shape))) if v.indices else 0
        key = (v.network_name, kind, flat)
        coeffs[key] = coeffs.get(key, 0.0) + t.coeff
    return ({k: c for k, c in coeffs.items() if c != 0.0}, lin.constant)


################################################################################
# Formulas
#
# ("and", [...]), ("or", [...]), ("atom", lin, op) meaning `lin op 0` with op
# one of "<=", "<", "==", and the constants TRUE and FALSE.

TRUE, FALSE = ("true",), ("false",)

# Comparison class -> (sign of lhs - rhs, operator)
COMPARISONS = {
    vnnlib.LessEqual: (1.0, "<="),
    vnnlib.LessThan: (1.0, "<"),
    vnnlib.GreaterEqual: (-1.0, "<="),
    vnnlib.GreaterThan: (-1.0, "<"),
    vnnlib.Equal: (1.0, "=="),
}


def _atom(c):
    if type(c) not in COMPARISONS:
        raise Unsupported(f"comparison {c}")
    sign, op = COMPARISONS[type(c)]
    return ("atom", lin_scale(lin_add(_linear(c.lhs), _linear(c.rhs), -1.0), sign), op)


def conjuncts(f):
    """The top-level conjuncts of a formula."""
    if f[0] == "and":
        return [c for a in f[1] for c in conjuncts(a)]
    if f == TRUE:
        return []
    return [f]


def substitute(f, values):
    """Replace variables whose values are known by constants and simplify."""
    if f[0] in ("and", "or"):
        parts = [substitute(a, values) for a in f[1]]
        unit, zero = (TRUE, FALSE) if f[0] == "and" else (FALSE, TRUE)
        if zero in parts:
            return zero
        parts = [p for p in parts if p != unit]
        if not parts:
            return unit
        return parts[0] if len(parts) == 1 else (f[0], parts)
    if f[0] == "atom":
        _, (coeffs, const), op = f
        new_coeffs = {}
        for k, v in coeffs.items():
            if k in values:
                const += v * values[k]
            else:
                new_coeffs[k] = v
        if not new_coeffs:
            holds = {"<=": const <= 0, "<": const < 0, "==": const == 0}[op]
            return TRUE if holds else FALSE
        return ("atom", (new_coeffs, const), op)
    return f


def evaluate(f, values, tol=0.0):
    """Evaluate a formula, allowing each atom to be violated by up to `tol`,
    a number or a function from an atom to a number."""
    if f[0] == "and":
        return all(evaluate(a, values, tol) for a in f[1])
    if f[0] == "or":
        return any(evaluate(a, values, tol) for a in f[1])
    if f[0] == "atom":
        _, (coeffs, const), op = f
        t = tol(f) if callable(tol) else tol
        v = const + sum(c * values[k] for k, c in coeffs.items())
        return {"<=": v <= t, "<": v < t, "==": abs(v) <= t}[op]
    return f == TRUE


def to_dnf(f):
    """A list of clauses, each a list of atoms."""
    if f == TRUE:
        return [[]]
    if f == FALSE:
        return []
    if f[0] == "atom":
        return [[f]]
    if f[0] == "or":
        return [c for a in f[1] for c in to_dnf(a)]
    clauses = [[]]
    for a in f[1]:
        clauses = [c1 + c2 for c1 in clauses for c2 in to_dnf(a)]
    return clauses


################################################################################
# Queries


class Query:
    def __init__(self, networks, formula):
        self.networks = networks
        self.formula = formula

    def tensors(self):
        return [t for n in self.networks.values() for t in (n.input, n.output)]


# Vehicle (up to at least 0.28) declares tensors as `Real`, but the standard
# and its type checker spell the type `real`.
REAL_RE = re.compile(r"(\(declare-(?:input|output|hidden)\s+\S+\s+)Real\b")


def parse_query(text):
    try:
        q = vnnlib.parse_query_string(REAL_RE.sub(r"\1real", text))
    except vnnlib.VNNLibException as e:
        raise Unsupported(f"invalid VNN-LIB query: {e}") from None
    networks = {n.name: _network(n) for n in q.networks}
    parts = [_assertion(a.expr) for a in q.assertions]
    return Query(networks, ("and", parts) if parts else TRUE)


def _assertion(expr):
    # A single clause stays a conjunction, so `conjuncts` sees its atoms (the
    # solver looks there for bounds on single inputs).
    clauses = [("and", [_atom(c) for c in clause]) for clause in expr.to_dnf()]
    return clauses[0] if len(clauses) == 1 else ("or", clauses)


def read_query(path):
    with open(path) as f:
        return parse_query(f.read())
