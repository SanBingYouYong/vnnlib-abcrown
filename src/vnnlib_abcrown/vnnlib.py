"""Parsing the VNN-LIB 2.0 queries that Vehicle generates.

Only the subset that Vehicle emits is supported: network declarations with one
input and one output tensor, and assertions built from `and`, `or`, `not`,
comparisons and linear arithmetic.
"""

import re

import numpy as np


class Unsupported(Exception):
    """The query uses something this backend cannot translate."""


################################################################################
# S-expressions

# Indexed variables such as `classifier_X[0, 3]` contain a space, so they must
# be matched as a single token.
TOKEN_RE = re.compile(r"\(|\)|[A-Za-z_][\w.]*\[[^\]]*\]|[^\s()]+")


def tokenize(text):
    text = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    return TOKEN_RE.findall(text)


def parse_sexprs(tokens):
    stack = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced parentheses in query")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses in query")
    return stack[0]


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
    def __init__(self, name):
        self.name = name
        self.equal_to = None
        self.input = None
        self.output = None
        self.onnx_path = None


def parse_shape(text):
    inner = text.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        raise Unsupported(f"tensor shape {text}")
    inner = inner[1:-1].strip()
    return [int(d) for d in inner.split(",")] if inner else []


def parse_networks(forms):
    networks = {}
    for form in forms:
        if not (isinstance(form, list) and form and form[0] == "declare-network"):
            continue
        net = Network(form[1])
        for item in form[2:]:
            head = item[0]
            if head == "equal-to":
                net.equal_to = item[1]
            elif head in ("declare-input", "declare-output"):
                kind = "X" if head == "declare-input" else "Y"
                if item[2] != "Real":
                    raise Unsupported(f"tensor element type {item[2]}")
                # The tokenizer splits shapes like "[28, 28]" into "[28," "28]".
                tensor = Tensor(net.name, kind, item[1], parse_shape(" ".join(item[3:])))
                if (net.input if kind == "X" else net.output) is not None:
                    raise Unsupported(f"network {net.name} has several inputs or outputs")
                if kind == "X":
                    net.input = tensor
                else:
                    net.output = tensor
            else:
                raise Unsupported(f"network declaration item {head}")
        if net.input is None or net.output is None:
            raise Unsupported(f"network {net.name} lacks an input or output declaration")
        networks[net.name] = net
    return networks


################################################################################
# Linear expressions
#
# A linear expression is a pair (coeffs, const) where coeffs maps a variable
# key (network, kind, flat index) to a non-zero float.


def lin_const(c):
    return ({}, float(c))


def lin_add(a, b, scale=1.0):
    coeffs = dict(a[0])
    for k, v in b[0].items():
        coeffs[k] = coeffs.get(k, 0.0) + scale * v
    return ({k: v for k, v in coeffs.items() if v != 0.0}, a[1] + scale * b[1])


def lin_scale(a, s):
    return ({k: v * s for k, v in a[0].items() if v * s != 0.0}, a[1] * s)


VAR_RE = re.compile(r"^([A-Za-z_][\w.]*)\[([^\]]*)\]$")


def is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_term(t, tensors):
    if isinstance(t, str):
        if is_number(t):
            return lin_const(t)
        m = VAR_RE.match(t)
        name, idx = (m.group(1), m.group(2)) if m else (t, "")
        if name not in tensors:
            raise Unsupported(f"unknown variable {t}")
        tensor = tensors[name]
        indices = [int(i) for i in idx.split(",")] if idx.strip() else []
        flat = int(np.ravel_multi_index(indices, tensor.shape)) if indices else 0
        return ({(tensor.network, tensor.kind, flat): 1.0}, 0.0)
    op, args = t[0], t[1:]
    if op == "+":
        acc = lin_const(0)
        for a in args:
            acc = lin_add(acc, parse_term(a, tensors))
        return acc
    if op == "-":
        if len(args) == 1:
            return lin_scale(parse_term(args[0], tensors), -1.0)
        acc = parse_term(args[0], tensors)
        for a in args[1:]:
            acc = lin_add(acc, parse_term(a, tensors), -1.0)
        return acc
    if op == "*":
        acc = lin_const(1)
        for p in (parse_term(a, tensors) for a in args):
            if not acc[0]:
                acc = lin_scale(p, acc[1])
            elif not p[0]:
                acc = lin_scale(acc, p[1])
            else:
                raise Unsupported("non-linear multiplication")
        return acc
    if op == "/":
        num, den = parse_term(args[0], tensors), parse_term(args[1], tensors)
        if den[0]:
            raise Unsupported("division by a variable")
        return lin_scale(num, 1.0 / den[1])
    raise Unsupported(f"arithmetic operator {op}")


################################################################################
# Formulas
#
# ("and", [...]), ("or", [...]), ("atom", lin, op) meaning `lin op 0` with op
# one of "<=", "<", "==", and the constants TRUE and FALSE.

TRUE, FALSE = ("true",), ("false",)


def parse_formula(f, tensors):
    op, args = f[0], f[1:]
    if op in ("and", "or"):
        return (op, [parse_formula(a, tensors) for a in args])
    if op == "not":
        return negate(parse_formula(args[0], tensors))
    if op in ("<=", "<", ">=", ">", "==", "="):
        if len(args) != 2:
            raise Unsupported(f"chained comparison {f}")
        lhs, rhs = parse_term(args[0], tensors), parse_term(args[1], tensors)
        if op in ("<=", "<", "==", "="):
            lin = lin_add(lhs, rhs, -1.0)
        else:
            lin = lin_add(rhs, lhs, -1.0)
        return ("atom", lin, {">=": "<=", ">": "<", "=": "=="}.get(op, op))
    raise Unsupported(f"logical operator {op}")


def negate(f):
    if f[0] == "and":
        return ("or", [negate(a) for a in f[1]])
    if f[0] == "or":
        return ("and", [negate(a) for a in f[1]])
    if f[0] == "atom":
        _, lin, op = f
        if op == "<=":
            return ("atom", lin_scale(lin, -1.0), "<")
        if op == "<":
            return ("atom", lin_scale(lin, -1.0), "<=")
        raise Unsupported("negated equality")
    return FALSE if f == TRUE else TRUE


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


def variables(f):
    if f[0] in ("and", "or"):
        return set().union(*(variables(a) for a in f[1])) if f[1] else set()
    if f[0] == "atom":
        return set(f[1][0])
    return set()


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


def parse_query(text):
    forms = parse_sexprs(tokenize(text))
    networks = parse_networks(forms)
    tensors = {t.name: t for n in networks.values() for t in (n.input, n.output)}
    parts = [parse_formula(form[1], tensors)
             for form in forms if isinstance(form, list) and form and form[0] == "assert"]
    return Query(networks, ("and", parts) if parts else TRUE)


def read_query(path):
    with open(path) as f:
        return parse_query(f.read())
