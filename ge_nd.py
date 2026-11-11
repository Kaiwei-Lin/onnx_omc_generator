"""Build uncompiled GE AIR for ND Mul/LayerNorm; OMG still compiles every kernel."""
from onnx import TensorProto, numpy_helper

from omc_reader import schema


def build_air(case, model, ge_proto):
    """Use the checked ONNX's shapes, types and exact initializer bytes."""
    op = case["op"]
    if op not in {"Mul", "LayerNorm"}:
        raise ValueError("ND AIR supports Mul and LayerNorm")
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

    node = graph.op.add(name="Y", type=op)
    node.input.extend(t["name"] + ":0" for t in case["inputs"])
    for item in case["inputs"]:
        desc(node.input_desc.add(), item["name"])
    desc(node.output_desc.add(), "Y")
    node.attr["version"].i = 5 if op == "LayerNorm" else 6
    node.attr["transformer_nd"].b = True
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
