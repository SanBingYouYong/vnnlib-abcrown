# Notes for reviewers

Thank you for taking a look. This repository is a thin wrapper that makes
[α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) a
[VNN-LIB 2.0](https://www.vnnlib.org/) solver, so that any VNN-LIB 2.0 client
can use it. It was written while working through the
[Vehicle tutorial](https://vehicle-lang.github.io/tutorial/), whose MNIST
exercise on standard robustness cannot be run with Vehicle's Marabou backend,
and Vehicle is the client it has been tested with.

**How it was made.** The code, tests and documents were written by an AI
coding assistant (Claude, by Anthropic), working with a student who is new to
the field. A first round of feedback from the VNN-LIB/Vehicle side led to
three changes: it is presented as a VNN-LIB 2.0 wrapper rather than a Vehicle
backend, queries are parsed with the reference
[`vnnlib`](https://github.com/VNNLIB/VNNLIB-Python) package instead of a
hand-written parser, and the standard's `verify` and `supports` commands are
implemented (see `PROGRESS.md`). The verification approach itself has not yet
been reviewed by anyone with verification expertise; that is what we are
asking for.

## What we claim

1. The VNN-LIB 2.0 solver interface is enough to plug α,β-CROWN into Vehicle
   without changing either tool. The wrapper implements the standard's
   `verify` and `supports` commands, and also the earlier form of `verify`
   that Vehicle 0.28 uses.
2. The adapter handles queries with **several network applications**:
   * applications with dataset-fixed inputs are evaluated to constants;
   * applications with free inputs are verified together as one product
     network;
   * linked input constraints (e.g. `|y − x| ≤ ε`) become a box through a
     linear change of variables.
3. `unsat` answers rest only on α,β-CROWN's proof of a relaxation of the
   query. `sat` answers are only given for points re-checked against the
   original query with onnxruntime.
4. On the whole Vehicle tutorial (chapters 2–6), it agrees with Marabou on all
   185 results that both decide.

## Where to look

| | |
|---|---|
| `docs/how-it-works.md` | the pipeline, step by step, with a worked example |
| `src/vnnlib_abcrown/` | the code (~1250 lines); `inputs.py` and `translate.py` hold the main ideas |
| `docs/tutorial-check/README.md` | Marabou vs α,β-CROWN on the tutorial: method and results |
| `docs/tutorial-check/results/` | every transcript from that run, and `report.md` with the per-case table |
| `PROGRESS.md` | development log, including bugs found and how they were fixed |
| `tests/` | 52 fast tests (fake solver) and 7 end-to-end tests (α,β-CROWN + Vehicle) |

## Questions we would most like answered

1. **Soundness of the input re-parameterisation (`inputs.py`).**
   * Choosing narrow directions and using `v = A⁻¹ z` as a first layer: is
     this a reasonable way to handle linked inputs, or is there a standard
     technique we have missed?
   * Is dropping leftover constraints at first (sound for `unsat`) and adding
     back only the ones a counterexample violates a sensible scheme?
2. **Strict inequalities and float arithmetic.**
   * Strict `<` becomes `≤` for α,β-CROWN. A counterexample that fails an
     *exact* re-check triggers a retry demanding a violation margin. When no
     such counterexample exists, we report `unknown` rather than a verdict.
   * Is this the right behaviour? Is there an established way to handle the
     strict/non-strict gap with a floating-point verifier?
3. **Marabou's counterexamples.** On two tutorial cases, Marabou reports
   counterexamples that sit exactly on the boundary and do not violate the
   strict property when the network is re-evaluated:
   * ACAS Xu N1,7 property 8 (scores tied to 2e-8);
   * `chapter-3/solutions/repro/linear.vcl` (`f(0) = 0` reported as `f < 0`).

   Is this a known Marabou/Vehicle behaviour, or a mistake in our checking?
4. **α,β-CROWN configuration.** We switch to input-space branch-and-bound for
   ≤ 16 input variables and use `conv_mode: matrix` otherwise. Is that
   threshold sensible, and are there better defaults for the tutorial's
   networks?
5. **New results.** α,β-CROWN decides five ACAS Xu cases Marabou did not:
   * N1,8 properties 6, 8 verified and 7 falsified (the tutorial's Marabou
     runs timed out after 900–1588 s);
   * N1,7 property 6 verified and 7 falsified.

   The falsifications were re-checked independently; the verifications rest
   on α,β-CROWN. Do these look plausible?
6. **Global robustness.** With two free applications of the MNIST CNN, the
   bound is about −1000 where 0 is needed, so nothing can be proved. Is this
   expected, and would a different encoding help (e.g. relational /
   "twin network" methods)?
7. **Upstreaming.** The suggestion so far is that a tidied-up version
   belongs in α,β-CROWN itself. What would be missing for that?

## Known limitations

See the README's Limitations section. In short:

* `!=` is rejected by Vehicle's VNN-LIB compiler before any solver runs.
* Exact-boundary properties come back as `unknown` (`timeout` through Vehicle).
* There is one α,β-CROWN process per query (about 5 s start-up each).
* Everything was tested with Vehicle 0.28.0 on one machine (2× RTX 3090).
* No licence has been chosen yet.
