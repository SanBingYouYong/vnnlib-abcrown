"""End-to-end tests that run α,β-CROWN (and, if available, Vehicle).

Run with `pytest -m abcrown` / `pytest -m vehicle`; set ABCROWN_ROOT to the
α,β-CROWN checkout and VEHICLE to the vehicle executable if they are not in
the default places.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import (MNIST_DIR, PROJECT, global_query, have_abcrown, linf_lipschitz_bound,
                     make_mlp, mlp_forward, mlp_weights, vehicle_executable)

LAUNCHER = os.path.join(PROJECT, "bin", "vnnlib-abcrown")
TOY_DIR = os.path.join(PROJECT, "examples", "toy")

needs_abcrown = pytest.mark.skipif(not have_abcrown(), reason="α,β-CROWN not found (set ABCROWN_ROOT)")


def run_backend(query_path, networks, *extra, cwd=None):
    env = dict(os.environ)
    env.setdefault("ABCROWN_PYTHON", sys.executable)
    args = [LAUNCHER, query_path] + [f"--network={n}={p}" for n, p in networks.items()]
    proc = subprocess.run(args + ["--timeout", "120"] + list(extra), capture_output=True,
                          text=True, env=env, cwd=cwd, timeout=900)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def parse_witness(lines):
    values, i = {}, 1
    while i < len(lines):
        name, _, shape = lines[i].split()
        n = int(np.prod([int(d) for d in shape[1:-1].split(",")]))
        values[name] = np.array([float(v) for v in lines[i + 1:i + 1 + n]])
        i += 1 + n
    return values


@pytest.mark.abcrown
@needs_abcrown
@pytest.mark.parametrize("seed", [0, 1])
def test_global_property_on_small_network(tmp_path, seed):
    w = mlp_weights([2, 8, 2], seed=seed)
    net = make_mlp(str(tmp_path / "net.onnx"), w)
    eps = 0.1
    nets = {"f": net, "f_2": net}

    # Above a sound Lipschitz bound: no counterexample can exist.
    delta = 1.01 * eps * linf_lipschitz_bound(w)
    q = tmp_path / "true.vnnlib"
    q.write_text(global_query(2, 2, eps, delta, output_index=0, upper=True))
    assert run_backend(str(q), nets) == ["unsat"]

    # Below a difference that sampling finds: the property is false.
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(20000, 2))
    y = np.clip(x + rng.choice([-eps, eps], size=x.shape), 0, 1)
    seen = max(mlp_forward(w, b)[0] - mlp_forward(w, a)[0] for a, b in zip(x, y))
    q = tmp_path / "false.vnnlib"
    q.write_text(global_query(2, 2, eps, 0.5 * seen, output_index=0, upper=True))
    lines = run_backend(str(q), nets)
    assert lines[0] == "sat"
    v = parse_witness(lines)
    assert np.all(np.abs(v["f_2_X"] - v["f_X"]) <= eps + 1e-6)
    assert v["f_2_Y"][0] - v["f_Y"][0] > 0.5 * seen - 1e-4
    assert np.allclose(v["f_Y"], mlp_forward(w, v["f_X"]), atol=1e-4)


VEHICLE = vehicle_executable()
needs_vehicle = pytest.mark.skipif(VEHICLE is None, reason="vehicle not found (set VEHICLE)")


def vehicle_verify(spec, networks, params, datasets=(), prop=None, cwd=None):
    env = dict(os.environ)
    env.setdefault("ABCROWN_PYTHON", sys.executable)
    args = [VEHICLE, "verify", "-s", spec, "-v", LAUNCHER, "-a", "--timeout 120"]
    for n, p in networks.items():
        args += ["-n", f"{n}:{p}"]
    for n, p in params.items():
        args += ["-p", f"{n}:{p}"]
    for n, p in datasets:
        args += ["-d", f"{n}:{p}"]
    if prop:
        args += ["-y", prop]
    proc = subprocess.run(args, capture_output=True, text=True, env=env, cwd=cwd, timeout=1800)
    return proc.stdout + proc.stderr


@pytest.mark.abcrown
@pytest.mark.vehicle
@needs_abcrown
@needs_vehicle
@pytest.mark.parametrize("delta,expected", [(0.03, "found a counterexample"),
                                            (0.6, "proved no counterexample exists")])
def test_vehicle_toy_global_stability(delta, expected):
    out = vehicle_verify("global-stability.vcl", {"toy": "toy.onnx"},
                         {"epsilon": 0.05, "delta": delta}, cwd=TOY_DIR)
    assert expected in out, out


@pytest.mark.abcrown
@pytest.mark.vehicle
@needs_abcrown
@needs_vehicle
def test_vehicle_mnist_standard_robustness_is_falsified_for_tiny_delta():
    # With delta = 0 and a strict bound the zero perturbation is fine, but any
    # change in the logits is not, which PGD finds immediately.
    out = vehicle_verify("standard-robustness.vcl", {"classifier": "mnist-classifier.onnx"},
                         {"epsilon": 0.005, "delta": 0.0001},
                         datasets=[("images", "t3-images.idx")],
                         prop="standardRobust", cwd=MNIST_DIR)
    assert "falsified: 3/3" in out, out


@pytest.mark.abcrown
@pytest.mark.vehicle
@needs_abcrown
@needs_vehicle
@pytest.mark.parametrize("c,expected", [(0.0, "found a witness"), (100.0, "proved no witness exists")])
def test_vehicle_exists_properties(c, expected):
    spec = os.path.join(PROJECT, "tests", "specs", "exists.vcl")
    out = vehicle_verify(spec, {"toy": os.path.join(TOY_DIR, "toy.onnx")}, {"c": c})
    assert out.count(expected) == 2, out
