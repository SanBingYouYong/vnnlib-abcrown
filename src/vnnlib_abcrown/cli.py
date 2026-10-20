"""Command-line interface of the VNN-LIB 2.0 standard (chapter "Solver
Command-line Interface"):

    vnnlib-abcrown --name
    vnnlib-abcrown --version
    vnnlib-abcrown supports --CAPABILITY
    vnnlib-abcrown verify QUERY --network NAME=FILE ... [--timeout SECONDS]
                   [options] [α,β-CROWN arguments]

`verify` prints `sat`, `unsat`, `timed-out` or `unknown`; `sat` is followed by
every tensor of the query as a `NAME real [d1,d2]` line and one value per line
in row-major order.

Vehicle (0.28 and its current master) still uses an earlier form of this
interface: it leaves out `verify` and only understands `timeout`. Without
`verify`, `timed-out` and `unknown` are therefore both printed as `timeout`.
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback

from . import __version__
from .query import Unsupported, read_query
from .runner import Runner
from .translate import fmt, log, solve

NAME = "alpha-beta-CROWN"

# Answers to `supports`. The ONNX opsets and operators are those of the
# networks this has been tested on (the Vehicle tutorial), not everything
# α,β-CROWN can handle.
SUPPORTS = {
    "--onnx-opset-versions": ["8", "20"],
    # α,β-CROWN computes in float32 without sound rounding, so no ONNX
    # element type can be claimed; only `real`, as the standard asks.
    "--onnx-element-types": ["real"],
    "--onnx-operators": ["Add", "Conv", "Flatten", "Gemm", "MatMul", "MaxPool", "Relu",
                         "Reshape", "Sub"],
    "--onnx-operators-unsat-proofs": [],
    "--vnnlib-versions": ["2.0", "2.0"],
    "--hidden-node-theories": ["NH"],
    "--multiple-input-output-theories": ["SIO"],
    "--multiple-network-theories": ["SNET", "MENET", "MINET", "MNET"],
    "--multiple-node-comparison-theories": ["SNC", "MNC"],
    "--arithmetic-complexity-theories": ["BND", "OUTC", "LIN"],
    # Queries are converted to DNF.
    "--optimised-disjunctive-reasoning": ["false"],
    "--serialise-assignments": ["false"],
    "--unsat-proofs": ["false"],
}

LEGACY_STATUS = {"timed-out": "timeout", "unknown": "timeout"}


def print_solution(solution, query, legacy=False, out=None):
    out = out or sys.stdout
    print(LEGACY_STATUS.get(solution.status, solution.status) if legacy else solution.status,
          file=out)
    if solution.status != "sat":
        return
    for t in query.tensors():
        shape = "[" + ",".join(str(d) for d in t.shape) + "]"
        print(f"{t.name} real {shape}", file=out)
        for k in t.keys():
            print(fmt(solution.values[k]), file=out)


def supports(argv):
    if len(argv) != 1 or argv[0] not in SUPPORTS:
        log(f"supports: expected one of {', '.join(SUPPORTS)}")
        return 1
    for line in SUPPORTS[argv[0]]:
        print(line)
    return 0


def verify(argv, legacy):
    parser = argparse.ArgumentParser(
        prog="vnnlib-abcrown verify", description="Verify a VNN-LIB 2.0 query with α,β-CROWN. "
        "Unrecognised arguments are passed to α,β-CROWN.")
    parser.add_argument("query")
    parser.add_argument("--network", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--timeout", type=float,
                        help="time limit in seconds for the whole query")
    parser.add_argument("--keep", action="store_true",
                        help="keep the generated network, query and α,β-CROWN log")
    parser.add_argument("--exact-inputs", action="store_true",
                        help="enforce input constraints that do not fit the input box as "
                        "extra specification rows from the start instead of adding them "
                        "only when a counterexample violates them")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="how many times to re-run α,β-CROWN after a counterexample "
                        "violates relaxed input constraints (default: 5)")
    args, abcrown_args = parser.parse_known_args(argv)

    # Vehicle also passes networks declared `equal-to` another one, which the
    # standard calls unnecessary; they are accepted.
    network_paths = dict(n.split("=", 1) for n in args.network)
    keep = args.keep or bool(os.environ.get("VNNLIB_ABCROWN_KEEP"))
    work_dir = tempfile.mkdtemp(prefix="vnnlib-abcrown-")
    try:
        query = read_query(args.query)
        runner = Runner(work_dir, abcrown_args, timeout=args.timeout)
        solution = solve(query, network_paths, work_dir, runner,
                         exact_inputs=args.exact_inputs or bool(os.environ.get("VNNLIB_ABCROWN_EXACT_INPUTS")),
                         max_rounds=args.max_rounds)
        print_solution(solution, query, legacy)
        return 0
    except Unsupported as e:
        log(f"cannot verify the query: {e}")
        keep = True
        return 1
    except Exception:
        log("error:\n" + traceback.format_exc())
        keep = True
        return 1
    finally:
        if keep:
            log(f"work files in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--name"]:
        print(NAME)
        return 0
    if argv == ["--version"]:
        print(__version__)
        return 0
    if argv[:1] == ["supports"]:
        return supports(argv[1:])
    if argv[:1] == ["verify"]:
        return verify(argv[1:], legacy=False)
    return verify(argv, legacy=True)
