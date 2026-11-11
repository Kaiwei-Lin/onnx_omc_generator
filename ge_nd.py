"""Build uncompiled GE AIR for ND cores; OMG still compiles every kernel."""
import numpy as np
from onnx import TensorProto, numpy_helper

from omc_reader import schema


def build_air(case, model, ge_proto):
    """Use the checked ONNX's shapes, types and exact initializer bytes."""
    op = case["op"]
    if op not in {"Add", "Mul", "LayerNorm", "Activation", "PRelu", "Softmax", "ReduceSum"}:
        raise ValueError("Unsupported ND AIR operator: " + op)
    if op == "Activation" and case.get("activation", "Sigmoid") != "Sigmoid":
        raise ValueError("ND AIR Activation currently supports Sigmoid")
    Model, *_ = schema(ge_proto)
    ir = Model(name=op, version=0)
    graph = ir.graph.add(name=op + "_nd")
    tensors = {t.name: t for t in model.graph.initializer}
    infos = {t.name: t for t in [*model.graph.input, *model.graph.output]}
    types = {TensorProto.FLOAT: 1, TensorProto.FLOAT16: 2}

    def describe(name):
        if name in tensors:
            t = tensors[name]
            return list(t.dims), t.data_type
        t = infos[name].type.tensor_type
        return [d.dim_value for d in t.shape.dim], t.elem_type

    def desc(destination, name, layout="ND"):
        shape, dtype = describe(name)
        if dtype not in types:
            raise ValueError("ND AIR requires float16 or float32 tensors")
        destination.shape.dim.extend(shape)
        destination.dtype = types[dtype]
        destination.layout = layout

    if op == "LayerNorm":
        dtype = describe(case["inputs"][0]["name"])[1]
        acc = case.get("accumulation_dtype", "float32")
        if acc not in ("float32", TensorProto.FLOAT) or case.get("attributes", {}).get("stash_type", TensorProto.FLOAT) != TensorProto.FLOAT:
            raise ValueError("ND LayerNorm requires float32 accumulation")
        if any(describe(t["name"])[1] != dtype for t in case["inputs"]):
            raise ValueError("ND LayerNorm requires matching X/scale/bias types")
        affine = case.get("affine_dtype", dtype)
        if affine not in (dtype, "float16" if dtype == TensorProto.FLOAT16 else "float32"):
            raise ValueError("ND LayerNorm requires affine_dtype to match X")

    for item in case["inputs"]:
        name = item["name"]
        constant = name in tensors
        node = graph.op.add(name=name, type="Const" if constant else "Data")
        # OMG's public AIR interfaces require NCHW; the core descriptors are ND.
        # In this path both describe the same contiguous logical tensor bytes.
        desc(node.output_desc.add(), name, "ND" if constant else "NCHW")
        if constant:
            value = node.attr["value"].t
            desc(value.desc, name)
            value.data = numpy_helper.to_array(tensors[name]).tobytes(order="C")
        else:
            node.attr["index"].i = len(graph.input)
            node.input_desc.add().CopyFrom(node.output_desc[0])
            graph.input.append(name + ":0")

    if op == "ReduceSum":
        attrs = case.get("attributes", {})
        rank = len(describe(case["inputs"][0]["name"])[0])
        axes = [a % rank for a in (attrs.get("axes") or [])]
        if not axes and not attrs.get("noop_with_empty_axes", 0):
            axes = list(range(rank))
        const = graph.op.add(name="__nd_axes", type="Const")
        axes_desc = const.output_desc.add()
        axes_desc.shape.dim.append(len(axes))
        axes_desc.layout = "ND"
        axes_desc.dtype = axes_desc.DESCRIPTOR.fields_by_name["dtype"].enum_type.values_by_name["DT_INT32"].number
        const.attr["value"].t.desc.CopyFrom(axes_desc)
        # Normalized axis indices fit INT32; GE consumes these as its Tidx=3 type.
        const.attr["value"].t.data = np.asarray(axes, dtype=np.int32).tobytes()

    node = graph.op.add(name="Y", type=op)
    node.input.extend(t["name"] + ":0" for t in case["inputs"])
    for item in case["inputs"]:
        desc(node.input_desc.add(), item["name"])
    desc(node.output_desc.add(), "Y")
    node.attr["version"].i = 5 if op == "LayerNorm" else 6
    node.attr["transformer_nd"].b = True
    if op == "ReduceSum":
        node.input.append("__nd_axes:0")
        node.input_desc.add().CopyFrom(axes_desc)
        node.attr["axis"].list.i.extend(axes)
        node.attr["axis"].list.val_type = 2
        node.attr["keep_dims"].b = bool(attrs.get("keepdims", 1))
        node.attr["noop_with_empty_axes"].b = bool(attrs.get("noop_with_empty_axes", 0))
        node.attr["Tidx"].i = 3
        node.attr["alpha"].f = 1
        node.attr["beta"].f = 0
    if op == "Activation":
        node.attr["mode"].i = 0  # GE Activation mode 0 is Sigmoid.
    if op == "PRelu":
        slope_shape, _ = describe(case["inputs"][1]["name"])
        node.attr["data_format"].s = b"NCHW"
        node.attr["channel_shared"].b = all(d == 1 for d in slope_shape)
    if op == "Softmax":
        rank = len(describe(case["inputs"][0]["name"])[0])
        axis = case.get("attributes", {}).get("axis", -1) % rank
        if case.get("opset", 13) < 13 and axis != rank - 1:
            raise ValueError("ND Softmax requires opset >=13 for a non-last axis")
        node.attr["axis"].i = axis
    if op == "LayerNorm":
        attrs = case.get("attributes", {})
        rank = len(describe(case["inputs"][0]["name"])[0])
        axis = attrs.get("axis", -1) % rank
        node.attr["begin_norm_axis"].i = axis
        node.attr["begin_params_axis"].i = axis
        node.attr["end_norm_axis"].i = rank - 1
        node.attr["epsilon"].f = attrs.get("epsilon", 1e-5)
        node.attr["norm_type"].s = b"LayerNorm"
    graph.output.append("Y:0")
    output = graph.op.add(name="NetOutput", type="NetOutput")
    output.input.append("Y:0")
    desc(output.input_desc.add(), "Y", "NCHW")
    return ir.SerializeToString(deterministic=True)
