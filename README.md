# vnnlib-abcrown

Use [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) as a
verification backend for the [Vehicle](https://github.com/vehicle-lang/vehicle)
specification language.

```bash
vehicle verify -s spec.vcl -n net:net.onnx ... \
  --solver /path/to/vnnlib-abcrown/bin/vnnlib-abcrown \
  --solver-args "--timeout 300"
```

Compared with Vehicle's built-in Marabou backend, this backend supports:

* **Properties with several network applications**, e.g. comparing
  `f image` with `f perturbedImage` (standard robustness), or global
  properties over two free inputs (`forall x y . close x y => ...`).
* **Input regions that are not boxes**, such as constraints linking two
  inputs (`|x - y| <= eps`) and constraints that mix inputs and outputs.
* α,β-CROWN's GPU-accelerated bound propagation and branch-and-bound.

## Requirements

* Vehicle ≥ 0.28 (tested with 0.28.0).
* An α,β-CROWN checkout with its own virtual environment
  (`<checkout>/.venv`). The adapter runs in that environment and needs only
  `numpy`, `onnx` and `onnxruntime`, which α,β-CROWN already installs.
* Normally a CUDA GPU. Pass `--solver-args "--device cpu"` otherwise.

Configuration is through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ABCROWN_ROOT` | `~/alpha-beta-CROWN` | α,β-CROWN checkout |
| `ABCROWN_PYTHON` | `$ABCROWN_ROOT/.venv/bin/python` | Python used for the adapter and α,β-CROWN |
| `VNNLIB_ABCROWN_KEEP` | unset | keep the generated files in `/tmp/vnnlib-abcrown-*` |
| `VNNLIB_ABCROWN_EXACT_INPUTS` | unset | same as `--exact-inputs` |
| `VNNLIB_ABCROWN_INPUT_SPLIT_MAX` | `16` | use input splitting for problems with at most this many inputs |

Alternatively, `pip install` this project into α,β-CROWN's environment and use
its `vnnlib-abcrown` script as the solver.

## Options

`--solver-args` are passed to α,β-CROWN's `abcrown.py`, except for these:

| Option | Meaning |
|---|---|
| `--keep` | keep the generated network, query and α,β-CROWN log |
| `--exact-inputs` | enforce all input constraints that do not fit the input box from the start (see below) |
| `--max-rounds N` | rounds of the refinement loop (default 5) |

Without `--config`, the adapter chooses α,β-CROWN's configuration:

* **At most 16 input variables** (ACAS Xu, Iris, the wind controller): it
  splits the input space, using α,β-CROWN's own VNN-COMP configuration for
  ACAS Xu. Splitting ReLUs is hopeless on such problems: ACAS Xu property 2
  took 3.5 s instead of timing out.
* **Larger problems:** `conv_mode: matrix` (patches mode does not support
  MaxPool).

Both run on CUDA. If α,β-CROWN runs out of GPU memory, the adapter retries
with smaller fixed batch sizes (64, then 8). Any α,β-CROWN option can be given
too, e.g. `--timeout 300`, `--conv_mode patches` or `--config my.yaml`.

## How it works

Vehicle calls any solver other than Marabou as

```
vnnlib-abcrown --name
vnnlib-abcrown --version
vnnlib-abcrown QUERY.vnnlib --network NAME=FILE.onnx ... <solver args>
```

with a VNN-LIB 2.0 query, and expects `unsat`, `timeout`, or `sat` followed by
a value for every input and output tensor. For each query the adapter:

1. **Evaluates fixed network applications.** Applications whose inputs are
   all pinned to constants (e.g. `f image` with `image` from a dataset) are
   evaluated with onnxruntime, and their outputs become constants.
2. **Builds one network.** The remaining applications are placed side by side
   in a single ONNX graph. The adapter also rewrites ONNX constructs that
   α,β-CROWN cannot handle: `auto_pad=SAME_*`, conv biases as separate `Add`
   nodes, and `Reshape` with a hard-coded batch size of 1. These rewrites
   don't change what the network computes.
3. **Makes the input region a box.** α,β-CROWN needs box inputs. Out of the
   query's linear input constraints, the adapter picks `n` linearly
   independent directions that are bounded on both sides (narrowest first),
   and uses `z = A v` as the new input, prepending `v = A⁻¹ z` to the network.
   For `|y - x| <= eps` this gives coordinates `(x, y - x)`.
4. **Handles constraints that are still left over.** If a remaining input
   constraint is implied by the box, it is dropped. Otherwise the inputs it
   needs are passed through to extra outputs, so it can be checked like an
   output constraint.
5. **Encodes the specification.** Every linear expression the query needs
   becomes an output of a final linear layer, so α,β-CROWN sees a VNN-LIB 1.0
   disjunction of conjunctions of `Y_j <= c` / `Y_j >= c`.
6. **Checks results.** `unsat` is passed on. A counterexample is mapped back
   (`v = A⁻¹ z`), the original networks are re-run, and the original query is
   re-checked: output constraints exactly, input constraints up to float64
   rounding. A counterexample that fails this check is never reported as
   `sat`.
7. **Retries near-ties with a margin.** α,β-CROWN works in float32 and can
   return a point on a boundary, e.g. two exactly tied ACAS Xu scores, that
   does not violate the strict property. In that case the adapter asks
   again for a counterexample that violates every output constraint by a
   margin (1e-6, then 1e-4, relative). If none exists, the answer is
   `timeout`: the property might hold, or fail only within the margin.

Each leftover input constraint costs α,β-CROWN one specification row, and
thousands of them run out of memory. By default they are therefore enforced
lazily, in rounds:

1. Verify without them. This searches a larger region, so `unsat` is still
   sound.
2. If the counterexample violates some of them, add exactly those and run
   again.

`--exact-inputs` adds all of them from the start.

## Limitations

* **At most one input and one output per network.** Vehicle itself does not
  emit networks with several inputs or outputs.
* **Linear specifications only**, as with any Vehicle VNN-LIB backend.
* **Bounded inputs.** Every input must be bounded in some direction of the
  constraints; otherwise the query is rejected as unsupported.
* **Some results are reported as `timeout`:** α,β-CROWN's `unknown`, and any
  counterexample that still fails the re-check after `--max-rounds` rounds.
* **Strict inequalities are checked as `<=`.** This is sound for `unsat`.
  Counterexamples are re-checked exactly, and retried with a margin as
  described above.
* **Networks without a batch dimension** (input `[2]` rather than `[1, 2]`)
  get one added. This assumes their operators broadcast over a leading
  dimension, as `MatMul`, `Add` and `Relu` do.
* **No proofs on exact boundaries.** A property whose only "violations" lie
  exactly on a boundary, e.g. `f x >= 0` where `f x = 0` is reachable, comes
  back as `timeout`. A float verifier cannot separate `< 0` from `<= 0`.
  Marabou reports such points as counterexamples, wrongly.
* **`!=` is rejected by Vehicle** for every solver except Marabou, even
  between parameters. This happens in Vehicle's VNN-LIB compiler, before the
  backend runs. Write `a != b` as `a < b or a > b`.
* **One α,β-CROWN process per query** (about 5 s of startup each). Vehicle
  splits disjunctive properties into many queries: Fashion MNIST robustness
  takes ~5× longer than with Marabou for this reason.
* **Hard global properties.** Global properties of large networks are usually
  out of reach. On the MNIST CNN, global robustness gives α,β-CROWN a bound of
  about −1000 where it would need something above 0.

## Examples

* `examples/toy`: global stability of a small MLP (two free network
  applications). With ε = 0.05: δ = 0.03 is falsified (~5 s), and δ = 0.1
  (~145 s) and δ = 0.6 (~35 s) are verified.
* `examples/mnist`: standard robustness around dataset images
  (`standard-robustness.vcl`), and global robustness (`global-robustness.vcl`).

```bash
cd examples/toy
vehicle verify -s global-stability.vcl -n toy:toy.onnx \
  -p epsilon:0.05 -p delta:0.1 -v ../../bin/vnnlib-abcrown -a "--timeout 120"

cd examples/mnist
vehicle verify -s standard-robustness.vcl -n classifier:mnist-classifier.onnx \
  -d images:t3-images.idx \
  -p epsilon:0.005 -p delta:5.0 -v ../../bin/vnnlib-abcrown -a "--timeout 300"
```

## Tests

```bash
$ABCROWN_PYTHON -m pytest -m "not abcrown and not vehicle"   # fast, no GPU
ABCROWN_ROOT=... VEHICLE=$(which vehicle) $ABCROWN_PYTHON -m pytest   # everything
```

The fast tests use a fake α,β-CROWN. The `abcrown` and `vehicle` tests run the
real tools on small networks and the MNIST example (about 1.5 minutes).

See `PROGRESS.md` for the development log, results and open items. Every
`vehicle verify` command of the Vehicle tutorial (chapters 2–6) has been run
with this backend and with Marabou. They agree on all 185 results both
decide; see `docs/tutorial-check/`.

## Documentation

* `docs/how-it-works.md`: the pipeline step by step, with a worked example.
* `docs/tutorial-check/`: the comparison with Marabou on the Vehicle tutorial,
  with all transcripts.
* `REVIEW.md`: what we claim, where the evidence is, and open questions for
  reviewers.
* `PROGRESS.md`: development log.

## Licence

No licence chosen yet. The MNIST network and images in `examples/mnist` were
copied from the [Vehicle tutorial repository](https://github.com/vehicle-lang/tutorial).
The network's node names match the [ONNX model zoo](https://github.com/onnx/models)
MNIST model. Check their licences before redistributing them.
