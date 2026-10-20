import numpy as np
import onnxruntime as ort
import pytest

from vnnlib_abcrown.onnx_tools import Block, build_spec_network, prepare, run_onnx

from helpers import MNIST_ONNX, make_mlp, mlp_weights


def run(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]


@pytest.mark.skipif(not __import__("os").path.exists(MNIST_ONNX), reason="MNIST example missing")
def test_rewrites_preserve_mnist_and_allow_batches(tmp_path):
    model = prepare(MNIST_ONNX)
    path = str(tmp_path / "rewritten.onnx")
    import onnx
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.save(model, path)
    x = np.random.default_rng(0).uniform(size=(5, 1, 28, 28))
    expected = np.concatenate([run(MNIST_ONNX, x[i:i + 1]) for i in range(5)])
    assert np.allclose(run(path, x), expected, atol=1e-5)
    ops = [n.op_type for n in model.graph.node]
    assert "Add" in ops and ops.count("Add") == 1  # conv biases folded, dense bias kept


def test_spec_network_two_blocks_with_passthrough(tmp_path):
    w1, w2 = mlp_weights([3, 5, 2], seed=1), mlp_weights([3, 4, 2], seed=2)
    p1, p2 = make_mlp(str(tmp_path / "a.onnx"), w1), make_mlp(str(tmp_path / "b.onnx"), w2)
    rng = np.random.default_rng(3)
    a_inv = rng.normal(size=(6, 6))
    pass_rows = [0, 4]
    weights = rng.normal(size=(3, 2 + 2 + len(pass_rows)))
    dst = str(tmp_path / "spec.onnx")
    build_spec_network(dst, [Block(p1, [0, 1, 2], 2), Block(p2, [3, 4, 5], 2)],
                       a_inv, False, pass_rows, weights)
    for _ in range(5):
        z = rng.uniform(-1, 1, size=6)
        v = a_inv @ z
        o = np.concatenate([run_onnx(p1, v[:3]), run_onnx(p2, v[3:]), v[pass_rows]])
        assert np.allclose(run(dst, z[None])[0], weights @ o, atol=1e-4)


def test_spec_network_direct_input(tmp_path):
    w = mlp_weights([3, 5, 2], seed=4)
    p = make_mlp(str(tmp_path / "a.onnx"), w)
    weights = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 2.0]])
    dst = str(tmp_path / "spec.onnx")
    build_spec_network(dst, [Block(p, [0, 1, 2], 2)], np.eye(3), True, [1], weights)
    x = np.array([0.2, -0.3, 0.7])
    o = np.concatenate([run_onnx(p, x), x[[1]]])
    assert np.allclose(run(dst, x[None])[0], weights @ o, atol=1e-5)


def test_old_ir_version_with_initializers_as_inputs(tmp_path):
    # Like the ACAS Xu networks: IR version 3, initializers listed as inputs.
    import onnx
    from onnx import helper
    w = mlp_weights([3, 4, 2], seed=5)
    p = make_mlp(str(tmp_path / "a.onnx"), w)
    model = onnx.load(p)
    model.ir_version = 3
    for init in model.graph.initializer:
        model.graph.input.append(helper.make_tensor_value_info(
            init.name, init.data_type, list(init.dims)))
    old = str(tmp_path / "old.onnx")
    onnx.save(model, old)
    dst = str(tmp_path / "spec.onnx")
    weights = np.array([[1.0, -1.0]])
    build_spec_network(dst, [Block(old, [0, 1, 2], 2)], np.eye(3), True, [], weights)
    x = np.array([0.1, 0.2, 0.3])
    assert np.allclose(run(dst, x[None])[0], weights @ run_onnx(p, x), atol=1e-5)


def test_network_without_batch_dimension(tmp_path):
    # Like chapter 3's tiny.onnx: input [2], output [1], a single MatMul.
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = np.array([[0.5], [-2.0]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(w, "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    p = str(tmp_path / "tiny.onnx")
    onnx.save(model, p)
    for identity in (True, False):
        dst = str(tmp_path / f"spec{identity}.onnx")
        a_inv = np.eye(2) if identity else np.array([[1.0, 1.0], [0.0, 1.0]])
        build_spec_network(dst, [Block(p, [0, 1], 1)], a_inv, identity, [], np.array([[3.0]]))
        for z in (np.array([0.3, -0.7]), np.array([1.0, 2.0])):
            expected = 3.0 * (a_inv @ z) @ w
            assert np.allclose(run(dst, z[None])[0], expected, atol=1e-5)
