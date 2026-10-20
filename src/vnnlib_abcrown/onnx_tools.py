"""ONNX manipulation: running networks, rewrites that α,β-CROWN needs, and
building the single "specification network" that α,β-CROWN verifies."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .vnnlib import Unsupported


################################################################################
# Running networks


_sessions = {}


def run_onnx(path, flat_input):
    """Evaluate the network at `path` on one input given as a flat vector."""
    if path not in _sessions:
        _sessions[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sess = _sessions[path]
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.asarray(flat_input, dtype=np.float32).reshape(shape)
    (y,) = sess.run(None, {inp.name: x})
    return y.reshape(-1).astype(np.float64)


def input_dims(model):
    init_names = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in init_names]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise Unsupported("networks must have exactly one input and one output")
    return inputs[0].name, [d.dim_value if d.dim_value > 0 else -1
                            for d in inputs[0].type.tensor_type.shape.dim]


################################################################################
# Rewrites. None of them change what the network computes.


def make_padding_explicit(model):
    """Replace `auto_pad=SAME_*` on Conv/pooling nodes by the equivalent explicit
    `pads`, since α,β-CROWN's ONNX converter does not support auto_pad."""
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for node in model.graph.node:
        if node.op_type not in ("Conv", "MaxPool", "AveragePool"):
            continue
        attrs = {a.name: a for a in node.attribute}
        if "auto_pad" not in attrs:
            continue
        mode = helper.get_attribute_value(attrs["auto_pad"]).decode()
        if mode not in ("SAME_UPPER", "SAME_LOWER"):
            continue
        spatial = shapes.get(node.input[0], [])[2:]
        if not spatial or 0 in spatial:
            raise Unsupported(f"cannot infer input shape of {node.op_type} with auto_pad")
        kernel = helper.get_attribute_value(attrs["kernel_shape"])
        strides = helper.get_attribute_value(attrs["strides"]) if "strides" in attrs else [1] * len(spatial)
        dilations = helper.get_attribute_value(attrs["dilations"]) if "dilations" in attrs else [1] * len(spatial)
        begins, ends = [], []
        for size, k, s, d in zip(spatial, kernel, strides, dilations):
            out = -(-size // s)
            total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
            small, big = total // 2, total - total // 2
            begins.append(small if mode == "SAME_UPPER" else big)
            ends.append(big if mode == "SAME_UPPER" else small)
        for name in ("auto_pad", "pads"):
            if name in attrs:
                node.attribute.remove(attrs[name])
        node.attribute.append(helper.make_attribute("pads", begins + ends))


def make_reshapes_batchable(model):
    """Constant Reshape targets with a hard-coded batch size of 1 (e.g. [1, 256])
    break α,β-CROWN's batched attacks and bounding, so use -1 instead."""
    inits = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Reshape" or node.input[1] not in inits:
            continue
        shape = numpy_helper.to_array(inits[node.input[1]]).copy()
        if len(shape) >= 2 and shape[0] == 1 and -1 not in shape:
            shape[0] = -1
            # The initializer may be shared, so give this node its own copy.
            new_name = node.input[1] + "_batchable"
            if new_name not in inits:
                inits[new_name] = numpy_helper.from_array(shape, new_name)
                model.graph.initializer.append(inits[new_name])
            node.input[1] = new_name


def fold_conv_bias_adds(model):
    """Fold `Add(Conv(x, W), b)` with a constant per-channel `b` (shape [C,1,1]
    or [1,C,1,1]) into `Conv(x, W, b)`. auto_LiRPA fails to bound the separate
    Add when it computes bounds in patches mode."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_outputs = {o.name for o in graph.output}
    for conv in list(graph.node):
        if conv.op_type != "Conv" or len(conv.input) > 2 or conv.input[1] not in inits:
            continue
        users = consumers.get(conv.output[0], [])
        if len(users) != 1 or users[0].op_type != "Add" or conv.output[0] in graph_outputs:
            continue
        add = users[0]
        other = add.input[1] if add.input[0] == conv.output[0] else add.input[0]
        if other not in inits:
            continue
        bias = numpy_helper.to_array(inits[other])
        channels = numpy_helper.to_array(inits[conv.input[1]]).shape[0]
        # Only a bias that varies along the channel axis alone can be folded.
        if bias.size != channels or (bias.ndim >= 3 and bias.shape[-3] != channels):
            continue
        bias_name = other + "_conv_bias"
        graph.initializer.append(numpy_helper.from_array(bias.reshape(-1), bias_name))
        conv.input.append(bias_name)
        conv.output[0] = add.output[0]
        graph.node.remove(add)


def remove_unused_initializers(model):
    used = {name for node in model.graph.node for name in node.input}
    used |= {o.name for o in model.graph.output}
    for init in [i for i in model.graph.initializer if i.name not in used]:
        model.graph.initializer.remove(init)


def prepare(path):
    model = onnx.load(path)
    make_padding_explicit(model)
    make_reshapes_batchable(model)
    fold_conv_bias_adds(model)
    remove_unused_initializers(model)
    return model


################################################################################
# The specification network


def _prefix_graph(graph, prefix, keep=()):
    """Rename every value in `graph` (no subgraphs) by adding `prefix`."""
    def rn(name):
        return name if (not name or name in keep) else prefix + name

    for node in graph.node:
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for a in node.attribute):
            raise Unsupported(f"ONNX control flow ({node.op_type})")
        node.name = rn(node.name)
        node.input[:] = [rn(n) for n in node.input]
        node.output[:] = [rn(n) for n in node.output]
    for coll in (graph.initializer, graph.input, graph.output, graph.value_info):
        for item in coll:
            item.name = rn(item.name)


def _add_batch_dimension(model, input_name):
    """Declare a leading batch dimension on the graph input and output."""
    inp = next(i for i in model.graph.input if i.name == input_name)
    for vi in (inp, model.graph.output[0]):
        shape = vi.type.tensor_type.shape
        old = [(d.dim_value, d.dim_param) for d in shape.dim]
        del shape.dim[:]
        shape.dim.add().dim_param = "batch"
        for value, param in old:
            d = shape.dim.add()
            if param:
                d.dim_param = param
            else:
                d.dim_value = value
    # Inferred intermediate shapes no longer hold.
    del model.graph.value_info[:]


class Block:
    """A network inside the specification network: its ONNX file and which
    rows of `A⁻¹` compute its (flattened) input from `z`."""

    def __init__(self, onnx_path, input_rows, output_size):
        self.onnx_path = onnx_path
        self.input_rows = input_rows
        self.output_size = output_size


def build_spec_network(dst, blocks, a_inv, identity, pass_rows, weights):
    """Write the specification network to `dst`.

    The network takes `z` and returns `weights @ concat(f_1(v_1), ..., f_k(v_k), v[pass_rows])`
    where `v = A⁻¹ z` and `v_i = v[block_i.input_rows]`, reshaped to network
    `i`'s input shape.

    If `identity` is true and there is a single block, `z = v` is fed to the
    network directly with its original input shape, so the result is just the
    original network with a linear layer on top.
    """
    n = a_inv.shape[0]
    nodes, inits, opsets = [], [], {}
    ir_version = 0

    def const(name, array):
        inits.append(numpy_helper.from_array(array, name))
        return name

    direct = identity and len(blocks) == 1
    pieces = []
    graph_input = None
    for k, block in enumerate(blocks):
        model = prepare(block.onnx_path)
        ir_version = max(ir_version, model.ir_version)
        for imp in model.opset_import:
            opsets[imp.domain] = max(opsets.get(imp.domain, 0), imp.version)
        in_name, dims = input_dims(model)
        size = len(block.input_rows)
        if len(dims) >= 2 and dims[0] in (1, -1) and int(np.prod(dims[1:])) == size:
            item = dims[1:]  # the first dimension is the batch
        elif int(np.prod(dims)) == size:
            # No batch dimension (e.g. input [2]). Add one: the usual operators
            # (MatMul, Add, Relu, ...) broadcast over a leading dimension.
            item = dims
            _add_batch_dimension(model, in_name)
        else:
            raise Unsupported(f"cannot match ONNX input shape {dims} to {size} input variables")
        reshaped = [-1] + item

        prefix = f"n{k}_"
        if direct:
            _prefix_graph(model.graph, prefix, keep=(in_name,))
            graph_input = helper.make_tensor_value_info(
                in_name, TensorProto.FLOAT, [1] + item)
            if len(pass_rows):
                nodes.append(helper.make_node(
                    "Reshape", [in_name, const("v_flat_shape", np.array([-1, n], dtype=np.int64))],
                    ["v_flat"]))
        else:
            _prefix_graph(model.graph, prefix)
            x_flat = prefix + "x_flat"
            w = const(prefix + "a_inv", a_inv[block.input_rows].T.astype(np.float32).copy())
            nodes.append(helper.make_node("MatMul", ["z", w], [x_flat]))
            nodes.append(helper.make_node(
                "Reshape", [x_flat, const(prefix + "x_shape", np.array(reshaped, dtype=np.int64))],
                [prefix + in_name]))
        nodes.extend(model.graph.node)
        inits.extend(model.graph.initializer)
        out = model.graph.output[0].name
        out_flat = prefix + "out_flat"
        nodes.append(helper.make_node(
            "Reshape", [out, const(prefix + "out_shape", np.array([-1, block.output_size], dtype=np.int64))],
            [out_flat]))
        pieces.append(out_flat)

    if not direct:
        graph_input = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, n])

    if len(pass_rows):
        sel = a_inv[pass_rows].T.astype(np.float32).copy()
        src = "v_flat" if direct else "z"
        nodes.append(helper.make_node("MatMul", [src, const("pass_w", sel)], ["passthrough"]))
        pieces.append("passthrough")

    if len(pieces) > 1:
        nodes.append(helper.make_node("Concat", pieces, ["spec_in"], axis=1))
        spec_in = "spec_in"
    else:
        spec_in = pieces[0]
    w = const("spec_w", np.asarray(weights, dtype=np.float32).T.copy())
    nodes.append(helper.make_node("MatMul", [spec_in, w], ["spec_out"]))

    graph = helper.make_graph(
        nodes, "vehicle_spec", [graph_input],
        [helper.make_tensor_value_info("spec_out", TensorProto.FLOAT, [1, len(weights)])],
        inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
    # Before IR version 4 every initializer also had to be a graph input
    # (e.g. the ACAS Xu networks); the generated graph lists only `z`.
    model.ir_version = max(ir_version, 4)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
