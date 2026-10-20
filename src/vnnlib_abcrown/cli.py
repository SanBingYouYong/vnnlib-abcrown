"""Command-line entry point implementing Vehicle's VNN-LIB solver protocol:

    vnnlib-abcrown --name
    vnnlib-abcrown --version
    vnnlib-abcrown QUERY --network NAME=FILE ... [α,β-CROWN arguments]

prints `unsat`, `timeout`, or `sat` followed by every tensor of the query as
a `NAME Real [d1,d2]` line and one value per line in row-major order.
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback

from . import __version__
from .runner import Runner
from .translate import fmt, log, solve
from .query import Unsupported, read_query

NAME = "abcrown"


def print_solution(solution, query, out=sys.stdout):
    print(solution.status, file=out)
    if solution.status != "sat":
        return
    for t in query.tensors():
        shape = "[" + ",".join(str(d) for d in t.shape) + "]"
        print(f"{t.name} Real {shape}", file=out)
        for k in t.keys():
            print(fmt(solution.values[k]), file=out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--name"]:
        print(NAME)
        return 0
    if argv == ["--version"]:
        print(__version__)
        return 0

    parser = argparse.ArgumentParser(
        prog="vnnlib-abcrown", description="Vehicle solver backend using α,β-CROWN. "
        "Unrecognised arguments are passed to α,β-CROWN.")
    parser.add_argument("query")
    parser.add_argument("--network", action="append", default=[], metavar="NAME=FILE")
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

    network_paths = dict(n.split("=", 1) for n in args.network)
    keep = args.keep or bool(os.environ.get("VNNLIB_ABCROWN_KEEP"))
    work_dir = tempfile.mkdtemp(prefix="vnnlib-abcrown-")
    try:
        query = read_query(args.query)
        solution = solve(query, network_paths, work_dir, Runner(work_dir, abcrown_args),
                         exact_inputs=args.exact_inputs or bool(os.environ.get("VNNLIB_ABCROWN_EXACT_INPUTS")),
                         max_rounds=args.max_rounds)
        print_solution(solution, query)
        return 0
    except Unsupported as e:
        log(f"unsupported query: {e}")
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
