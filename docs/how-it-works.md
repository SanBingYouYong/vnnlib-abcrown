# How vnnlib-abcrown works

This document describes the pipeline from `vehicle verify` to α,β-CROWN and
back. The code is in `src/vnnlib_abcrown/`.

## Vehicle's side (unchanged)

```bash
vehicle verify -s spec.vcl -n classifier:net.onnx ... --solver bin/vnnlib-abcrown
```

Vehicle (0.28) treats any solver it does not recognise as Marabou as a generic
"VNN-LIB solver" (`src/Vehicle/Verify/Solver/VNNLIB.hs` in Vehicle):

1. It calls `vnnlib-abcrown --name` (we answer `abcrown`) and `--version`.
2. It compiles each property into **queries** in VNN-LIB 2.0. Each query
   asks for an assignment that *violates* the property (or, for `exists`,
   satisfies it). Vehicle performs the quantifier and disjunction splitting
   itself, so one property can produce many queries; classification
   robustness of one image gives 9.
3. For each query it runs

   ```
   vnnlib-abcrown QUERY.vnnlib --network classifier=net.onnx --network classifier_2=net.onnx <solver args>
   ```

   with one `--network` per *application* of a network in the property.
4. It expects on standard output one of:
   * `unsat`;
   * `timeout`;
   * `sat`, followed by every declared tensor as a line `name Real [d1,d2]`
     and then one value per line.

   It then reports ✓ / ✗ / timeout and writes the verification cache that
   `vehicle export` uses.

Unlike Vehicle's Marabou query format, the VNN-LIB format allows several
network applications in one query, so properties such as standard robustness
(`|f(x') − f(x)| ≤ δ`) can be compiled at all.

## The adapter, per query (`translate.solve`)

### a. Parse (`vnnlib.py`)

The adapter reads the network declarations (one input and one output tensor
each, as Vehicle emits them) and the assertions. Assertions are linear
(in)equalities over input variables `X` and output variables `Y`, combined
with `and`, `or` and `not`.

### b. Evaluate network applications with fixed inputs

If every input element of an application is pinned by equalities, it is
evaluated once with onnxruntime and its outputs become constants. An example
is `classifier image` with `image` taken from a dataset. Only the
applications with free inputs remain.

### c. Turn the input region into a box (`inputs.py`)

α,β-CROWN accepts only box input regions. Vehicle's queries can link inputs.
Global robustness, for example, gives `x ∈ [0,1]ⁿ`, `y ∈ [0,1]ⁿ` and
`|yᵢ − xᵢ| ≤ ε`.

1. Let `v` be all free input variables (concatenated over applications). All
   constraints `a·v ≤ b` are grouped by direction `a`, each giving an interval.
2. Choose `n = dim(v)` linearly independent directions with finite intervals,
   **narrowest first**, using sparse Gaussian elimination. With rows `aᵢ`
   forming `A`, the new input is `z = A v`. Its region is exactly the box of
   those intervals, and the network gets `v = A⁻¹ z` as a first layer. For the
   example above, the chosen coordinates are `x` and `d = y − x`. When all
   chosen directions are single variables, `A = I` and nothing is added.
3. Every other direction is either implied by the box (checked by interval
   arithmetic, then dropped) or a **leftover** constraint. Leftovers are
   *not* given to α,β-CROWN at first. That only enlarges the region searched,
   so `unsat` remains sound. They are added back in step g if a
   counterexample violates them, because each leftover costs a specification
   row: 1568 rows for MNIST global robustness made α,β-CROWN try to allocate
   80 GB.

If some direction stays unbounded, the query is rejected as unsupported.

### d. Build one network (`onnx_tools.py`)

A single ONNX graph is built from:

```
z ──[A⁻¹ rows]──► v₁ ──► f (copy 1) ──┐
  ──[A⁻¹ rows]──► v₂ ──► f (copy 2) ──┼─► concat ──[W]──► spec outputs Y₀..Y_k
  ──[A⁻¹ rows]──► (inputs referenced by constraints) ──┘
```

Each row of `W` is one distinct linear expression the query needs, e.g.
`f(y)₃ − f(x)₃` or `f(x)₀ − x₀`.

The networks are also rewritten in ways that do not change the function
computed, because α,β-CROWN's ONNX loader or auto_LiRPA otherwise fails:

* `auto_pad=SAME_*` is replaced by explicit padding;
* conv-bias `Add` nodes are folded into `Conv`;
* `Reshape` to a hard-coded batch size of 1 uses `-1` instead;
* the generated graph declares at least ONNX IR version 4;
* a batch dimension is added for networks without one.

### e. Write VNN-LIB 1.0

The adapter writes box bounds on `X_i` (the `z` coordinates), then the
specification in disjunctive normal form over `Yⱼ ≤ c` / `Yⱼ ≥ c`, which is
the form α,β-CROWN's reader accepts. **Strict inequalities become
non-strict.**

### f. Run α,β-CROWN (`runner.py`)

* **Configuration**, when the user passes no `--config`:
  * at most 16 input variables: α,β-CROWN's VNN-COMP ACAS Xu configuration,
    which uses input-space branch-and-bound;
  * otherwise: the default (ReLU splitting), with `conv_mode: matrix`.
* **Other arguments:** everything in Vehicle's `--solver-args` is passed to
  `abcrown.py`.
* **CUDA out of memory:** the run is repeated with batch size 64, then 8,
  with automatic batch enlarging off.
* **`unknown`** is reported to Vehicle as `timeout`.

### g. Check the answer

* **`unsat`** is passed on to Vehicle.
* **A counterexample `z`** is mapped back (`v = A⁻¹ z`, clipped to the box).
  The original networks are re-run with onnxruntime and the **original**
  query is evaluated: output constraints exactly, input-only constraints up
  to 1e-9 for float64 rounding.
  * If the check passes, the adapter reports `sat` with every tensor.
  * If the point violates leftover input constraints, those are added as
    specification rows and α,β-CROWN runs again (up to 5 rounds in total).
  * Otherwise the point lies on, or within float error of, an output
    boundary, e.g. two exactly tied ACAS Xu scores. α,β-CROWN is asked again
    for a point violating every output constraint by `m·(1+|c|)`, with
    `m = 1e-6`, then `1e-4`. A new `unsat` at this stage only rules out
    violations larger than the margin, so the answer is `timeout`.

So `sat` is only reported for points that satisfy the original query when
the networks are evaluated with onnxruntime, and `unsat` is only reported
when α,β-CROWN proved infeasibility of a relaxation of the query.

## Worked example: standard robustness around a dataset image

Property: `forall p . |p| ≤ ε and valid(image − p) => |f(image − p)ᵢ − f(image)ᵢ| ≤ δ`.
One of Vehicle's queries asks for `p` with `f(image − p)₃ − f(image)₃ > δ`.

* **b.** `f(image)` is evaluated, say `f(image)₃ = c`.
* **c.** Vehicle has already eliminated `p`. The query bounds the free
  network input `x = image − p` by a box (`max(0, image−ε) ≤ x ≤ min(1, image+ε)`),
  so `A = I`.
* **d.** The network is `f` followed by a 10→1 layer selecting output 3.
* **e.** The VNN-LIB 1.0 query is that box and `(<= Y_0 -(c+δ))` for the
  row `−f(x)₃`. The sign comes from normalising `c + δ − f(x)₃ < 0`.
* **f, g.** α,β-CROWN proves `unsat`, or returns `x`. The adapter re-checks
  `x`, and Vehicle prints `perturbation = image − x`.

## Code map

| File | Role |
|---|---|
| `cli.py` | Vehicle's solver interface, options, output format |
| `vnnlib.py` | VNN-LIB 2.0 parsing; linear expressions; formulas, DNF |
| `inputs.py` | step c |
| `onnx_tools.py` | step d; running networks |
| `translate.py` | steps b, e, g; the refinement loop |
| `runner.py` | step f |
