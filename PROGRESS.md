# Progress log

Development notes for `vehicle-abcrown`. Newest entries first. Each entry
records what changed, how it was checked, and what is still open.

## Status

| Area | State |
|---|---|
| VNN-LIB 2.0 solver CLI (`--name`, `--version`, `verify`, `supports`) and Vehicle 0.28's earlier form | done |
| Parsing with the reference `vnnlib` package | done |
| Fixed network applications (inputs pinned by dataset) evaluated to constants | done |
| Several free network applications (product network) | done, tested |
| Linked input constraints (change of variables) | done, tested |
| Leftover input constraints, lazily added in rounds | done, tested |
| Constraints mixing inputs and outputs | done, tested (fake solver only) |
| `exists` properties | done, tested through Vehicle |
| Exact re-check of counterexamples, margin retries for near-ties | done, tested |
| Input splitting for ≤ 16 inputs; OOM retry with smaller batches | done, tested |
| ONNX: `auto_pad`, conv bias `Add`, batch-1 `Reshape`, IR < 4, no batch dimension | done, tested |
| Whole Vehicle tutorial (chapters 2–6) vs Marabou | done: no disagreements |
| Standalone packaging, tests, examples | done |
| Licence | **open**: not chosen; provenance of the MNIST example files to check |

### Open items / ideas

* One α,β-CROWN process per query costs ~5 s of start-up; Vehicle splits
  properties into many queries (9 per image for classification robustness).
  On the tutorial's 150 Fashion MNIST images this made α,β-CROWN take 1.7 h
  against Marabou's 0.35 h. A persistent worker would help most.
* Global properties of the MNIST CNN are out of reach (bound ≈ −1000 vs 0 needed,
  4 BaB domains in 300 s). α,β-CROWN's input-space branch-and-bound or
  tighter relational methods might help; untried.
* Mixed input/output constraints are only tested with the fake solver, not
  with real α,β-CROWN or through Vehicle.
* Properties that hold only up to an exact boundary (`f x >= 0` with
  `f x = 0` reachable) come back as `timeout`; see README.
* The automatic input-split threshold (16 inputs) is a guess based on
  VNN-COMP practice; only tested on 2–5 inputs (and 784/1568 the other way).
* Rocq/Agda proof checking of exported files was not run (neither installed);
  the exported files were compared with those from Marabou caches.
* ONNX coverage beyond the networks in the tutorial is untested.
* `pip install` into α,β-CROWN's environment was checked with
  `uv pip install --target`, not with a real install into the environment.

## 2026-10-20: VNN-LIB 2.0 wrapper (first feedback)

Feedback from the VNN-LIB/Vehicle side: (1) it is a VNN-LIB 2.0 wrapper, not
a Vehicle wrapper, and should be presented as a community tool; (2) the
hand-written parser duplicates the `vnnlib` package (VNNLIB-Python); (3) the
standard's `supports` command was missing, and future Vehicle versions will
rely on it. Changes:

* **Renamed** to `vnnlib-abcrown` (package `vnnlib_abcrown`, `bin/vnnlib-abcrown`,
  `VNNLIB_ABCROWN_*` variables). README/REVIEW reframed; version 0.3.0.
* **Parser:** queries are parsed, type-checked, linearised and put into DNF
  per assertion by `vnnlib` ≥ 1.0.2 (`query.py` replaces `vnnlib.py`; the
  parsing part shrank from ~200 to ~70 lines). Element types other than
  `real`, hidden nodes and `!=` are rejected explicitly.
* **CLI** per the standard: `verify QUERY --network ... [--timeout S]` with
  `sat` / `unsat` / `timed-out` / `unknown`, and `supports` for every
  capability. `--timeout` now covers the whole query (each α,β-CROWN run
  gets the time left), which also tells α,β-CROWN's `timeout` apart into
  `timed-out` and `unknown`.

Found while doing this (worth reporting upstream):

* Vehicle (0.28 and master at 1c97857) writes the element type `Real`; the
  standard and `vnnlib`'s type checker require `real`, so the reference
  parser rejected all 64 queries Vehicle produced for the examples. The
  wrapper rewrites `Real` to `real` before parsing.
* Vehicle still calls solvers without `verify`, only understands `timeout`,
  and passes `--network` also for `equal-to` networks (which the standard
  says must be an error). The wrapper keeps that form working: without
  `verify`, `timed-out` and `unknown` are printed as `timeout`.

Checked: 52 fast tests and 7 end-to-end tests (α,β-CROWN + Vehicle 0.28) pass.
46 of the 51 tutorial cases were re-run through Vehicle with the new wrapper
(all but ACAS Xu N1,7 p8 and N1,8 p8 and the three Fashion MNIST cases, the
slowest): verdicts
identical to the recorded run, total time 2223 s vs 2252 s. The `supports`
answers for ONNX opsets (8–20) and operators are those of the tutorial's
networks, not α,β-CROWN's full coverage.

Open: rename the GitHub repository; `--serialise-assignments` and UNSAT
proofs are not implemented (reported as unsupported); move to
`vnnlib.query` once VNNLIB-Python releases it.

## 2026-10-06: prepared for review

* Moved to its own repository, with `docs/how-it-works.md`, the tutorial
  comparison (`docs/tutorial-check/`, including all transcripts), and
  `REVIEW.md` with questions for researchers in the field.
* No code changes.

## 2026-10-05 (later): the whole Vehicle tutorial

Ran every `vehicle verify` command of the tutorial (chapters 2–6: chapter
code, exercises, solutions; 51 cases) with Marabou and with this backend,
using a harness in the tutorial repository (copied to `docs/tutorial-check/`). Chapter 6's
files were fetched from the 5 GB Zenodo artefact with HTTP range requests.

Bugs found and fixed:

* **ACAS Xu crashed the adapter.** The networks use ONNX IR version 3, where
  initializers must also be graph inputs; the generated network failed the
  ONNX checker. Now at least IR 4 is declared.
* **ACAS Xu property 2 never finished** (ReLU-split BaB, bounds −23 after
  65k domains). Problems with ≤ 16 inputs now use α,β-CROWN's input-split
  configuration: 3.5 s.
* **Spurious counterexample (soundness bug).** On ACAS Xu N1,7 property 8,
  α,β-CROWN returned a point where two scores are tied to 2e-9, and the 1e-4
  re-check tolerance accepted it. Output constraints are now re-checked
  exactly, and a failing candidate triggers a re-run that asks for a violation
  by a margin (1e-6, then 1e-4, relative). Now `timeout`. Marabou's
  counterexample for the same property is also a near-tie: it does not
  violate the property when re-evaluated.
* **Networks without a batch dimension** (`repro/tiny.onnx`, input `[2]`)
  were rejected; a batch dimension is now added.
* **CUDA out of memory** (my own parallel test run held 14 GB of the GPU)
  made α,β-CROWN error; the runner now retries with batch sizes 64 and 8.

Results (details: `docs/tutorial-check/README.md`):

* 185 results decided by both solvers, 0 disagreements, including every one
  of the 150 Fashion MNIST images. All documented results reproduced.
* Only α,β-CROWN decides: ACAS Xu N1,7 properties 6 (verified) and 7
  (falsified); N1,8 properties 6 (verified), 7 (falsified), 8 (verified). The
  tutorial's Marabou runs timed out on N1,8 6–8 after 900–1588 s. Both
  falsifications were re-checked independently with onnxruntime.
* Only Marabou "decides": ACAS Xu N1,7 property 8 and the `repro/linear`
  specs. Both of its counterexamples are wrong: they sit exactly on a boundary
  and do not violate the strict property. This backend says `timeout`.
* Vehicle-side limitation: `!=` is rejected for VNN-LIB solvers (chapter 6,
  `Ka != Ke`); with `Ka < Ke or Ka > Ke` chapter 6 verifies and the Rocq
  export matches Marabou's.
* `vehicle export` (Rocq, Agda, Isabelle, Imandra) from α,β-CROWN caches gives
  the same files as from Marabou caches, apart from the cache path.

Tests: 30 pass (23 fast, 7 with α,β-CROWN/Vehicle).

## 2026-10-05: general backend and standalone project

Goal: lift the restriction to one network application with a free input,
and make the project standalone.

Finding before starting: a global robustness spec (`forall image perturbation`)
compiles to VNN-LIB 2.0 queries with two networks, `classifier` and
`classifier_2`, each with a box `[0, 1]` on its input and two one-sided
constraints per pixel on `classifier_2_X[i] - classifier_X[i]`. So the input
region is a box in the coordinates `(x, y - x)`, but not in `(x, y)`.

Done:

* Restructured into a package (`src/vehicle_abcrown/`: `vnnlib`, `inputs`,
  `onnx_tools`, `translate`, `runner`, `cli`), with a launcher in
  `bin/vehicle-abcrown`, `pyproject.toml`, tests and examples.
* Product network: free network applications side by side in one ONNX graph,
  input `z`, `v = A⁻¹ z` as the first layer, a final linear layer over
  `concat(outputs, passed-through inputs)`.
* Change of variables (`inputs.analyse`): bounded constraint directions are
  sorted narrowest first and picked by sparse Gaussian elimination. If they
  are all single variables, `A` is the identity and the original network
  input is used directly (same result as the first version).
* Leftover input constraints: first attempt passed all of them to α,β-CROWN
  as specification rows. For MNIST global robustness that was 1568 rows and
  α,β-CROWN tried to allocate 80 GB. Now they are dropped at first (sound for
  `unsat`) and added only when a counterexample violates them, for at most
  `--max-rounds` rounds. `--exact-inputs` restores adding all of them.
* Counterexamples are re-run with onnxruntime and checked against the
  original query (tolerance 1e-4). Failing ones are reported as `timeout`,
  never as `sat`.

Checked:

* 23 tests pass: 18 fast (fake solver), 5 with real α,β-CROWN/Vehicle
  (`pytest`, ~1.5 min). Three test failures along the way were all bugs in
  the tests (sign of a constant, an unreachable threshold, the fake's
  VNN-LIB parser), not in the adapter.
* `examples/toy` (MLP 2→16→16→2, ε = 0.05, empirical max difference ≈ 0.054,
  Lipschitz bound gives 0.55), through `vehicle verify`:
  * δ = 0.03: falsified in 5.5 s; the counterexample was checked
    independently (`|x - y| = 0.05`, `|f(x)_0 - f(y)_0| = 0.039`).
  * δ = 0.1: verified in 144 s; δ = 0.6: verified in 34 s.
* MNIST global robustness (ε = 0.005, δ = 1.0, one of 20 queries):
  * One run (before the refinement loop existed, all leftover constraints
    dropped): PGD found a counterexample that the re-check rejected, most
    likely because a perturbed pixel was outside [0, 1]. I did not inspect it.
  * A second run with the refinement loop timed out at 300 s, with PGD
    within 0.03 of a counterexample and the best bound at about −1007. So it
    is probably false but unconfirmed.

## 2026-10-03: first working version

* Found Vehicle's generic solver protocol by pointing it at a logging dummy
  executable, then confirmed it against `src/Vehicle/Verify/Solver/VNNLIB.hs`
  (Vehicle v0.28.0).
* Marabou's query format rejects properties with two network applications.
  Vehicle's VNN-LIB format accepts them.
* Adapter: evaluate fully-pinned networks, append a linear layer for all
  output expressions, write VNN-LIB 1.0, run α,β-CROWN.
* ONNX fixes needed for the tutorial MNIST model: explicit padding instead of
  `auto_pad`, folding conv biases into `Conv` (auto_LiRPA failed on the separate
  `Add`), `Reshape [1, 256]` → `[-1, 256]` (broke batched PGD), and
  `conv_mode: matrix` (MaxPool is not supported in patches mode).
* Results on 3 MNIST test images, ε = 0.005, corrected standard robustness
  `|f(x') - f(x)| <= δ` for every output:
  * δ = 5.0: verified 3/3 (341 s).
  * δ = 1.0: timed out 3/3 (1028 s). The label logit's query is close to the
    margin; PGD gets within about 0.3 and α-CROWN's bound misses by 0.02–0.2.
  * The tutorial's own spec (`f(x')!l - f(x)!l >= δ`) is falsified for every
    image by the zero perturbation, so it was wrong as written, not the solver.
