"""Solving a Vehicle VNN-LIB 2.0 query with α,β-CROWN.

1. Network applications whose inputs are all pinned to constants (e.g. an
   image from a dataset) are evaluated, and their outputs become constants.
2. The remaining ("free") applications are put side by side in one network.
   Their inputs `v` are re-parameterised so the input region is a box
   (see `inputs.py`).
3. Every linear expression the query needs (over outputs, and over inputs
   that are routed through) becomes an output of a final linear layer, so the
   specification is a disjunction of conjunctions of `Y_j <= c` / `Y_j >= c`,
   which is what α,β-CROWN's VNN-LIB 1.0 reader understands.
4. Any counterexample is mapped back, the networks are re-run with
   onnxruntime and the original (strict, unrelaxed) query is re-checked.
"""

import os

import numpy as np

from . import inputs
from .onnx_tools import Block, build_spec_network, run_onnx
from .query import FALSE, TRUE, Unsupported, conjuncts, evaluate, substitute, to_dnf

# Counterexamples are re-checked against the original query: constraints on
# network outputs exactly (a tolerance there once let a near-tie through as a
# spurious ACAS Xu counterexample), constraints on inputs alone up to float64
# rounding of `v = A⁻¹ z`.
INPUT_TOL = 1e-9
# When a counterexample fails the exact check, α,β-CROWN is asked again for one
# that violates every output constraint by at least `margin * (1 + |bound|)`,
# with these margins in turn.
MARGINS = (1e-6, 1e-4)
MAX_CLAUSES = 10000


def log(*args):
    import sys
    print("[vnnlib-abcrown]", *args, file=sys.stderr, flush=True)


def fmt(v):
    return repr(float(v))


class Solution:
    def __init__(self, status, values=None):
        self.status = status  # "sat", "unsat" or "timeout"
        self.values = values


def _is_input_atom(g):
    return g[0] == "atom" and all(k[1] == "X" for k in g[1][0])


def solve(query, network_paths, work_dir, run_abcrown, exact_inputs=False, max_rounds=5):
    """`run_abcrown(onnx_path, vnnlib_path, num_inputs)` returns
    (status, z) with status in "sat"/"unsat"/"timeout" and z the counterexample.

    Input constraints that do not fit the box ("leftover" constraints) each
    cost a specification row, which can make α,β-CROWN run out of memory. So
    unless `exact_inputs` is set they are first dropped, which only enlarges the
    region searched (`unsat` stays sound), and added back one round at a time
    when a counterexample violates them, for at most `max_rounds` rounds."""
    networks = query.networks
    for name, path in network_paths.items():
        if name not in networks:
            raise Unsupported(f"--network {name} is not declared in the query")
        networks[name].onnx_path = path
    for net in networks.values():
        if net.onnx_path is None and net.equal_to:
            net.onnx_path = networks[net.equal_to].onnx_path
        if net.onnx_path is None:
            raise Unsupported(f"no ONNX file given for network {net.name}")

    parts = conjuncts(query.formula)
    input_parts = [g for g in parts if _is_input_atom(g)]
    rest = [g for g in parts if not _is_input_atom(g)]

    # Find the inputs pinned to a single value by single-variable constraints.
    lower, upper = {}, {}
    for _, (coeffs, const), op in input_parts:
        if len(coeffs) != 1:
            continue
        (key, c), = coeffs.items()
        bound = -const / c
        if op == "==" or c > 0:
            upper[key] = min(upper.get(key, np.inf), bound)
        if op == "==" or c < 0:
            lower[key] = max(lower.get(key, -np.inf), bound)

    values = {}
    free = []
    for net in networks.values():
        keys = net.input.keys()
        if all(k in lower and lower[k] == upper.get(k) for k in keys):
            x = np.array([lower[k] for k in keys])
            values.update(zip(keys, x))
            values.update(zip(net.output.keys(), run_onnx(net.onnx_path, x)))
        else:
            free.append(net)

    input_atoms = []
    for g in input_parts:
        g = substitute(g, values)
        if g == FALSE:
            return Solution("unsat")
        if g != TRUE:
            input_atoms.append(g)
    formula = substitute(("and", rest) if rest else TRUE, values)

    if not free:
        return Solution("sat", values) if evaluate(formula, values) else Solution("unsat")

    keys = [k for net in free for k in net.input.keys()]
    region = inputs.analyse([(g[1][0], g[1][1], g[2]) for g in input_atoms], keys)
    if region is None:
        return Solution("unsat")

    clauses = to_dnf(formula)
    if not clauses:
        return Solution("unsat")
    if len(clauses) > MAX_CLAUSES:
        raise Unsupported(f"the query has {len(clauses)} disjuncts after DNF conversion")

    blocks = []
    start = 0
    for net in free:
        blocks.append(Block(net.onnx_path, list(range(start, start + net.input.size)),
                            net.output.size))
        start += net.input.size

    def witness(z):
        v = region.to_v(z)
        result = dict(values)
        result.update(zip(keys, v))
        start = 0
        for net in free:
            x = v[start:start + net.input.size]
            start += net.input.size
            result.update(zip(net.output.keys(), run_onnx(net.onnx_path, x)))
        return result

    # Leftover input constraints are enforced lazily: start without them (or
    # with all of them if `exact_inputs`), and when a counterexample violates
    # some, add exactly those and try again.
    pending = list(region.leftover)
    active = pending if exact_inputs else []
    pending = [] if exact_inputs else pending
    if pending:
        log(f"{len(pending)} input constraint(s) do not fit the input box; "
            "they are added only if a counterexample violates them")

    margins = iter(MARGINS)
    margin = 0.0
    for round_ in range(1, max_rounds + 1):
        problem = _encode(free, keys, region, clauses, active, margin)
        if problem is None:
            # Nothing to check on the outputs: any point of the box will do.
            result = witness((region.lo + region.hi) / 2)
        else:
            rows, pass_rows, spec = problem
            onnx_path = os.path.join(work_dir, "spec.onnx")
            vnnlib_path = os.path.join(work_dir, "spec.vnnlib")
            build_spec_network(onnx_path, blocks, region.a_inv, region.identity, pass_rows, rows)
            write_vnnlib(vnnlib_path, region.lo, region.hi, len(rows), spec)
            log(f"round {round_}: {len(free)} free network application(s), {len(keys)} input "
                f"variables{'' if region.identity else ' (re-parameterised)'}, "
                f"{len(rows)} specification rows, {len(spec)} clause(s), "
                f"{len(active)} input constraint(s) as rows"
                + (f", output margin {margin:g}" if margin else ""))
            status, z = run_abcrown(onnx_path, vnnlib_path, len(keys))
            if status == "unsat" and margin:
                # Only counterexamples by at least the margin were ruled out.
                log(f"no counterexample violates the property by margin {margin:g}; it may hold "
                    "or fail only within the margin, so reporting 'timeout'")
                return Solution("timeout")
            if status != "sat":
                return Solution(status)
            result = witness(z)

        if evaluate(query.formula, result, _tolerance):
            return Solution("sat", result)
        v = np.array([result[k] for k in keys])
        violated = [c for c in pending if _violation(c, v) > INPUT_TOL * (1 + abs(c[2]))]
        if violated:
            log(f"the counterexample violates {len(violated)} relaxed input constraint(s); "
                "adding them and trying again")
            active = active + violated
            pending = [c for c in pending if c not in violated]
            continue
        # The candidate is on (or within float error of) the boundary of an
        # output constraint, e.g. two exactly tied scores.
        margin = next(margins, None)
        if margin is None:
            log("no counterexample that passes the exact re-check; "
                "reporting 'timeout' instead of an unchecked 'sat'")
            return Solution("timeout")
        log(f"the counterexample does not satisfy the original query exactly; asking for "
            f"one that violates it by margin {margin:g}")

    log(f"no valid counterexample after {max_rounds} rounds; reporting 'timeout'")
    return Solution("timeout")


def _tolerance(atom):
    coeffs, const = atom[1]
    if any(k[1] == "Y" for k in coeffs):
        return 0.0
    return INPUT_TOL * (1 + abs(const))


def _violation(constraint, v):
    row, op, bound = constraint
    value = sum(c * v[i] for i, c in row.items())
    return value - bound if op == "<=" else bound - value


def _encode(free, keys, region, clauses, leftover, margin=0.0):
    """The specification rows, the inputs routed to outputs, and the clauses
    over rows, or None if there is nothing to check. With a margin, the
    clauses' constraints must hold by `margin * (1 + |bound|)`."""
    # Layout of the vector the final linear layer is applied to: the outputs of
    # every free network, then the inputs that constraints refer to.
    v_index = {k: i for i, k in enumerate(keys)}
    position = {}
    offset = 0
    for net in free:
        for i, k in enumerate(net.output.keys()):
            position[k] = offset + i
        offset += net.output.size
    passed = {v_index[k] for clause in clauses for atom in clause
              for k in atom[1][0] if k[1] == "X"}
    passed |= {i for row, _, _ in leftover for i in row}
    pass_rows = sorted(passed)
    for j, i in enumerate(pass_rows):
        position[keys[i]] = offset + j
    width = offset + len(pass_rows)

    rows, row_index = [], {}

    def spec_row(coeffs):
        r = np.zeros(width)
        for k, c in coeffs.items():
            r[position[k]] += c
        key = tuple(r)
        if key not in row_index:
            row_index[key] = len(rows)
            rows.append(r)
        return row_index[key]

    extra = [(op, spec_row({keys[i]: c for i, c in row.items()}), bound)
             for row, op, bound in leftover]
    spec = []
    for clause in clauses:
        atoms = []
        for _, (coeffs, const), op in clause:
            if any(k not in position for k in coeffs):
                raise Unsupported("a constraint refers to a variable that is not free")
            j = spec_row(coeffs)
            # coeffs . o + const (op) 0, i.e. Y_j (op) -const. Strict becomes non-strict.
            slack = margin * (1 + abs(const))
            atoms.append(("<=", j, -const - slack))
            if op == "==":
                atoms.append((">=", j, -const + slack))
        spec.append(atoms + extra)
    if not rows:
        return None
    return rows, pass_rows, spec


def write_vnnlib(path, lo, hi, num_outputs, spec):
    with open(path, "w") as f:
        for i in range(len(lo)):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(num_outputs):
            f.write(f"(declare-const Y_{j} Real)\n")
        for i in range(len(lo)):
            f.write(f"(assert (>= X_{i} {fmt(lo[i])}))\n")
            f.write(f"(assert (<= X_{i} {fmt(hi[i])}))\n")
        if len(spec) == 1:
            for op, j, c in spec[0]:
                f.write(f"(assert ({op} Y_{j} {fmt(c)}))\n")
        else:
            f.write("(assert (or\n")
            for clause in spec:
                if not clause:
                    raise Unsupported("an empty clause next to other clauses")
                atoms = "".join(f"({op} Y_{j} {fmt(c)})" for op, j, c in clause)
                f.write(f"  (and {atoms})\n")
            f.write("))\n")
