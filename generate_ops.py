#!/usr/bin/env python3
"""Build configurable ONNX operator tests and compile/audit them with Huawei OMG."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OPS = ("MatMul", "GemmD", "BatchMatMul", "Convolution", "ConvTranspose", "Add", "Mul",
       "Activation", "PRelu", "ReduceSum", "LayerNorm", "Softmax", "CastT")
ALIASES = {x.lower(): x for x in OPS}
ALIASES.update(matmul="MatMul", gemm="GemmD", gemmv3="GemmD", conv="Convolution",
               convalution="Convolution", layernormalization="LayerNorm", cast="CastT")
DTYPES = {"float16": TensorProto.FLOAT16, "float32": TensorProto.FLOAT,
          "float64": TensorProto.DOUBLE, "int8": TensorProto.INT8, "uint8": TensorProto.UINT8,
          "int16": TensorProto.INT16, "uint16": TensorProto.UINT16, "int32": TensorProto.INT32,
          "uint32": TensorProto.UINT32, "int64": TensorProto.INT64, "uint64": TensorProto.UINT64,
          "bool": TensorProto.BOOL}
OMG_TYPES = {v: k.upper() for k, v in DTYPES.items()}
OMG_TYPES.update({TensorProto.FLOAT16: "FP16", TensorProto.FLOAT: "FP32", TensorProto.DOUBLE: "FP64"})
SCHEMA_RELATIVE = "tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py"
BLOCKED_IO_OPS = {"Convolution", "ConvTranspose", "Add", "Mul", "Activation", "PRelu", "LayerNorm", "Softmax"}


def canonical(op):
    try:
        return ALIASES[op.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unknown operator {op!r}; choose from {', '.join(OPS)}") from None


def json_text(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(json_text(obj).encode("utf-8")).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def model_config(case):
    return {k: v for k, v in case.items() if k not in {"id", "source", "expect_omc_type"}}


def operator_config(case):
    """The editable sidecar contains only operator type and model parameters."""
    params = copy.deepcopy(model_config(case))
    op = params.pop("op")
    return {"op": op, "params": params}


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    compiler = {}
    if isinstance(value, list):
        cases = value
    elif "cases" in value:
        unknown_keys(value, {"compiler", "cases"}, "configuration")
        cases, compiler = value["cases"], value.get("compiler", {})
    else:
        cases = [value]
    result = []
    for case in cases:
        if "params" in case:
            unknown_keys(case, {"op", "params"}, "operator configuration")
            if "op" in case["params"]:
                raise ValueError("Put op outside params")
            case = {"op": case["op"], **case["params"]}
        result.append(case)
    return result, compiler


def save_operator_config(path, case, base=None):
    value = operator_config(case)
    if base is not None:
        for item in value["params"]["inputs"]:
            spec = item.get("init", {})
            if spec.get("kind") == "npy":
                source = (Path(base) / spec["path"]).resolve()
                try:
                    spec["path"] = os.path.relpath(source, path.parent)
                except ValueError:
                    spec["path"] = str(source)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8-sig"))
        if "params" in old and old != value:
            raise FileExistsError(f"Edited configuration already exists: {path}; choose a new --out")
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    text = re.sub(r"\[\s*([-+\d.eE]+(?:\s*,\s*[-+\d.eE]+)*)\s*\]",
                  lambda m: "[" + re.sub(r"\s+", "", m[1]).replace(",", ", ") + "]", text)
    path.write_text(text + "\n", encoding="utf-8")


def unknown_keys(obj, allowed, location):
    extra = set(obj) - set(allowed)
    if extra:
        raise ValueError(f"Unknown {location} fields: {sorted(extra)}")


def dtype_id(value):
    if isinstance(value, str) and value.lower() in DTYPES:
        return DTYPES[value.lower()]
    if type(value) is int and value in DTYPES.values():
        return value
    raise ValueError(f"Unsupported dtype: {value!r}; choose {list(DTYPES)}")


def np_dtype(value):
    return np.dtype(next(k for k, v in DTYPES.items() if v == dtype_id(value)))


def input_dtype(case, item):
    return item.get("dtype", case.get("dtype", "float16"))


def broadcast_into(small, large, description):
    if list(np.broadcast_shapes(tuple(small), tuple(large))) != list(large):
        raise ValueError(f"{description} {small} cannot broadcast into {large}")


def axis_index(value, rank):
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError(f"axis {value} outside rank {rank}")
    return value % rank


def validate_case(case):
    op = canonical(case["op"])
    special = {"MatMul": {"transpose_a", "transpose_b", "constant_left_rewrite"},
               "BatchMatMul": {"transpose_a", "transpose_b", "constant_left_rewrite"},
               "Activation": {"activation"}, "LayerNorm": {"implementation", "accumulation_dtype", "affine_dtype"}}
    unknown_keys(case, {"id", "op", "dtype", "opset", "seed", "inputs", "attributes", "source",
                        "expect_omc_type"} | special.get(op, set()), "case")
    if type(case.get("opset", 13)) is not int or case.get("opset", 13) < 11:
        raise ValueError("Use ONNX opset >= 11")
    if type(case.get("seed", 0)) is not int or case.get("seed", 0) < 0:
        raise ValueError("seed must be a nonnegative integer")
    items = case["inputs"]
    counts = {"GemmD": (2, 3), "Convolution": (2, 3), "ConvTranspose": (2, 3),
              "LayerNorm": (2, 3), "Activation": (1, 2, 3)}
    allowed = counts.get(op, (2,) if op in {"MatMul", "BatchMatMul", "Add", "Mul", "PRelu"} else (1,))
    if len(items) not in allowed:
        raise ValueError(f"{op} requires input counts {allowed}")
    names, types = [], []
    for item in items:
        unknown_keys(item, {"name", "shape", "dtype", "init"}, "input")
        name, shape = item["name"], item["shape"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name == "Y":
            raise ValueError(f"Invalid/reserved input name {name!r}; use letters, digits, underscore; Y is output")
        if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"{name}: shape must contain positive integers (use [] for scalar)")
        names.append(name)
        types.append(dtype_id(input_dtype(case, item)))
    if len(names) != len(set(names)):
        raise ValueError("Input names must be unique")
    if all("init" in x for x in items):
        raise ValueError("At least one runtime input is required; otherwise OMG can fold the entire graph")
    if len(set(types)) != 1:
        raise ValueError("This operator requires equal input dtypes; change dtype for all operands together")
    shapes = [x["shape"] for x in items]
    attrs = case.get("attributes", {})
    json_text(case)  # Reject nonfinite JSON parameters.
    if op in {"MatMul", "BatchMatMul"}:
        for key in ("transpose_a", "transpose_b", "constant_left_rewrite"):
            if key in case and type(case[key]) is not bool:
                raise ValueError(f"{key} must be true/false")
        aa, bb = copy.deepcopy(shapes)
        for s, flag in zip((aa, bb), (case.get("transpose_a"), case.get("transpose_b"))):
            if flag:
                if len(s) < 2:
                    raise ValueError("Transpose requires rank >= 2")
                s[-2:] = s[-2:][::-1]
        if not aa or not bb or aa[-1] != bb[-2 if len(bb) > 1 else 0]:
            raise ValueError("MatMul contraction dimensions do not match")
        np.broadcast_shapes(tuple(aa[:-2]), tuple(bb[:-2]))
        if op == "BatchMatMul" and max(len(aa), len(bb)) < 3:
            raise ValueError("BatchMatMul requires at least one batched input (rank >= 3)")
        if case.get("constant_left_rewrite") and (len(aa) != 2 or len(bb) != 2 or "init" not in items[0] or "init" in items[1]):
            raise ValueError("constant_left_rewrite requires 2-D constant A and runtime B")
    elif op == "GemmD":
        for key in ("transA", "transB"):
            if attrs.get(key, 0) not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
        aa = shapes[0][::-1] if attrs.get("transA", 0) else shapes[0]
        bb = shapes[1][::-1] if attrs.get("transB", 0) else shapes[1]
        if len(aa) != 2 or len(bb) != 2 or aa[1] != bb[0]:
            raise ValueError("Gemm requires compatible 2-D matrices after transA/transB")
        if len(shapes) == 3:
            broadcast_into(shapes[2], [aa[0], bb[1]], "Gemm bias")
    elif op in {"Add", "Mul"}:
        np.broadcast_shapes(*map(tuple, shapes))
    elif op == "PRelu":
        broadcast_into(shapes[1], shapes[0], "PRelu slope")
    elif op in {"Convolution", "ConvTranspose"}:
        x, w = shapes[:2]
        rank = len(x) - 2
        group = attrs.get("group", 1)
        if rank < 1 or len(w) != len(x) or type(group) is not int or group < 1:
            raise ValueError("Convolution requires matching N,C,spatial ranks and positive group")
        if op == "Convolution":
            if x[1] != w[1] * group or w[0] % group:
                raise ValueError("Conv channels/group disagree with W=[Cout,Cin/group,k...]")
            channels = w[0]
        else:
            if x[1] != w[0] or w[0] % group:
                raise ValueError("ConvTranspose channels/group disagree with W=[Cin,Cout/group,k...]")
            channels = w[1] * group
        if len(shapes) == 3 and shapes[2] != [channels]:
            raise ValueError(f"Convolution bias shape must be [{channels}]")
        if attrs.get("kernel_shape", w[2:]) != w[2:]:
            raise ValueError("kernel_shape must match weight spatial dimensions")
        for key, count, minimum in (("strides", rank, 1), ("dilations", rank, 1),
                                     ("pads", 2 * rank, 0), ("output_padding", rank, 0),
                                     ("output_shape", rank, 1)):
            if key in attrs and (len(attrs[key]) != count or any(type(d) is not int or d < minimum for d in attrs[key])):
                raise ValueError(f"Invalid {key}: expected {count} integers >= {minimum}")
        if "pads" in attrs and attrs.get("auto_pad", "NOTSET") != "NOTSET":
            raise ValueError("Remove pads when using auto_pad")
        if op == "ConvTranspose":
            limits = zip(attrs.get("output_padding", [0] * rank), attrs.get("strides", [1] * rank), attrs.get("dilations", [1] * rank))
            if any(p >= max(s, d) for p, s, d in limits):
                raise ValueError("output_padding must be smaller than stride or dilation on each axis")
    elif op == "LayerNorm":
        axis = axis_index(attrs.get("axis", -1), len(shapes[0]))
        if attrs.get("epsilon", 1e-5) <= 0:
            raise ValueError("LayerNorm epsilon must be positive")
        for shape in shapes[1:]:
            broadcast_into(shape, [1] * axis + shapes[0][axis:], "LayerNorm scale/bias")
    elif op == "ReduceSum":
        axes = attrs.get("axes")
        if axes is not None:
            normalized = [axis_index(a, len(shapes[0])) for a in axes]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate reduction axes")
        for key in ("keepdims", "noop_with_empty_axes"):
            if attrs.get(key, 0) not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
    elif op == "Softmax":
        axis_index(attrs.get("axis", -1), len(shapes[0]))


def constant_array(spec, shape, dtype, rng, base):
    kind = spec.get("kind", "normal")
    allowed = {"normal": {"mean", "std"}, "uniform": {"low", "high"}, "fill": {"value"},
               "values": {"values"}, "npy": {"path", "sha256"}}
    if kind not in allowed:
        raise ValueError(f"Unknown initializer kind {kind!r}")
    unknown_keys(spec, {"kind"} | allowed[kind], "initializer")
    dtype = np_dtype(dtype)
    if kind in {"normal", "uniform"} and dtype.kind != "f":
        raise ValueError("Use fill/values/npy for integer or bool constants")
    if kind == "normal":
        data = rng.normal(spec.get("mean", 0), spec.get("std", 0.02), shape)
    elif kind == "uniform":
        data = rng.uniform(spec.get("low", -1), spec.get("high", 1), shape)
    elif kind == "fill":
        data = np.full(shape, spec["value"])
    elif kind == "values":
        data = np.asarray(spec["values"]).reshape(shape)
    else:
        path = (base / spec["path"]).resolve()
        if "sha256" in spec and sha256(path) != spec["sha256"]:
            raise ValueError(f"Initializer changed: {path}")
        data = np.load(path, allow_pickle=False)
        if list(data.shape) != shape:
            raise ValueError(f"{path}: expected shape {shape}, got {data.shape}")
    data = np.asarray(data, dtype=dtype)
    if not np.isfinite(data).all():
        raise ValueError("Initializer has nonfinite values after dtype conversion")
    return data


def build_model(case, base=HERE):
    """Build standard ONNX; the DDK decides its compiled kernel names."""
    validate_case(case)
    op, opset = canonical(case["op"]), case.get("opset", 13)
    attrs = copy.deepcopy(case.get("attributes", {}))
    rng = np.random.default_rng(case.get("seed", 0))
    initializers, graph_inputs, nodes, args = [], [], [], []
    arrays = {}
    for item in case["inputs"]:
        name, dtype = item["name"], input_dtype(case, item)
        args.append(name)
        if "init" in item:
            arrays[name] = constant_array(item["init"], item["shape"], dtype, rng, Path(base))
        else:
            graph_inputs.append(helper.make_tensor_value_info(name, dtype_id(dtype), item["shape"]))

    def node(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], name=output + "_" + kind, **attributes))

    def const(name, value, dtype):
        arrays[name] = np.asarray(value, dtype=np_dtype(dtype))
        return name

    def transpose(name, rank):
        target = "__g_" + name.strip("_") + "_T"
        if name in arrays:
            arrays[target] = np.ascontiguousarray(np.swapaxes(arrays.pop(name), -1, -2))
        else:
            perm = list(range(rank))
            perm[-2:] = perm[-2:][::-1]
            node("Transpose", [name], target, perm=perm)
        return target

    def reduce_mean(name, output, axes):
        if opset >= 18:
            ax = const("__g_" + output.strip("_") + "_axes", axes, "int64")
            node("ReduceMean", [name, ax], output, keepdims=1)
        else:
            node("ReduceMean", [name], output, axes=axes, keepdims=1)

    out_dtype = dtype_id(input_dtype(case, case["inputs"][0]))
    if op in {"MatMul", "BatchMatMul"}:
        for i, key in enumerate(("transpose_a", "transpose_b")):
            if case.get(key):
                args[i] = transpose(args[i], len(case["inputs"][i]["shape"]))
        if case.get("constant_left_rewrite"):
            left, right = transpose(args[0], 2), transpose(args[1], 2)
            node("MatMul", [right, left], "__g_product_T", **attrs)
            node("Transpose", ["__g_product_T"], "Y", perm=[1, 0])
        else:
            node("MatMul", args, "Y", **attrs)
    elif op == "LayerNorm":
        implementation = case.get("implementation", "decomposed")
        acc = dtype_id(case.get("accumulation_dtype", "float32"))
        if acc not in {TensorProto.FLOAT, TensorProto.FLOAT16}:
            raise ValueError("LayerNorm accumulation_dtype must be float32 or float16")
        if implementation == "native":
            if opset < 17:
                raise ValueError("Native LayerNormalization requires opset >= 17")
            if "affine_dtype" in case and dtype_id(case["affine_dtype"]) != out_dtype:
                raise ValueError("Native LayerNorm affine_dtype must match the input dtype")
            attrs.setdefault("stash_type", acc)
            node("LayerNormalization", args, "Y", **attrs)
        elif implementation == "decomposed":
            unknown_keys(attrs, {"axis", "epsilon"}, "decomposed LayerNorm attributes")
            rank = len(case["inputs"][0]["shape"])
            axes = list(range(axis_index(attrs.get("axis", -1), rank), rank))
            x = args[0]
            if acc != out_dtype:
                node("Cast", [x], "__g_acc", to=acc)
                x = "__g_acc"
            reduce_mean(x, "__g_mean", axes)
            node("Sub", [x, "__g_mean"], "__g_centered")
            two = const("__g_two", 2.0, acc)
            node("Pow", ["__g_centered", two], "__g_squared")
            reduce_mean("__g_squared", "__g_variance", axes)
            epsilon = const("__g_epsilon", attrs.get("epsilon", 1e-5), acc)
            node("Add", ["__g_variance", epsilon], "__g_var_eps")
            node("Sqrt", ["__g_var_eps"], "__g_std")
            node("Div", ["__g_centered", "__g_std"], "__g_normalized")
            norm = "__g_normalized"
            affine = dtype_id(case.get("affine_dtype", out_dtype))
            if affine not in {TensorProto.FLOAT, TensorProto.FLOAT16}:
                raise ValueError("LayerNorm affine_dtype must be float32 or float16")
            if acc != affine:
                node("Cast", [norm], "__g_norm_cast", to=affine)
                norm = "__g_norm_cast"
            weights = list(args[1:])
            if affine != out_dtype:
                for i, weight in enumerate(weights):
                    if weight in arrays:
                        arrays[weight] = arrays[weight].astype(np_dtype(affine))
                    else:
                        weights[i] = "__g_affine_" + weight
                        node("Cast", [weight], weights[i], to=affine)
            result = "__g_affine_result" if affine != out_dtype else "Y"
            node("Mul", [norm, weights[0]], "__g_scaled" if len(args) == 3 else result)
            if len(args) == 3:
                node("Add", ["__g_scaled", weights[1]], result)
            if affine != out_dtype:
                node("Cast", [result], "Y", to=out_dtype)
        else:
            raise ValueError("LayerNorm implementation must be native or decomposed")
    elif op == "ReduceSum":
        axes = attrs.pop("axes", None)
        if opset >= 13:
            if axes is not None:
                args.append(const("__g_axes", axes, "int64"))
        elif axes is not None:
            attrs["axes"] = axes
        node("ReduceSum", args, "Y", **attrs)
    elif op == "CastT":
        attrs["to"] = dtype_id(attrs["to"])
        out_dtype = attrs["to"]
        node("Cast", args, "Y", **attrs)
    else:
        kind = {"GemmD": "Gemm", "Convolution": "Conv", "PRelu": "PRelu"}.get(op, op)
        if op == "Activation":
            kind = case.get("activation", "Sigmoid")
            if kind not in {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu", "HardSigmoid",
                            "Softplus", "Softsign", "ThresholdedRelu", "HardSwish", "Gelu", "Clip"}:
                raise ValueError(f"Unsupported activation {kind!r}")
        node(kind, args, "Y", **attrs)
    initializers = [numpy_helper.from_array(a, name) for name, a in arrays.items()]
    graph = helper.make_graph(nodes, op + "_parameter_test", graph_inputs,
                              [helper.make_tensor_value_info("Y", out_dtype, None)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                              producer_name="onnx_omc_generator", producer_version="1", ir_version=8)
    model.doc_string = "Parameterized operator test. Handfree defaults describe logical structure; synthetic constants are not original trained weights."
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)
    for output in model.graph.output:
        shape = output.type.tensor_type.shape
        if not output.type.tensor_type.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in shape.dim):
            raise ValueError("Output shape is unresolved/nonpositive; check shapes, axes, convolution parameters")
    return model


def tensor_info(value):
    return {"name": value.name, "shape": [d.dim_value for d in value.type.tensor_type.shape.dim],
            "dtype": TensorProto.DataType.Name(value.type.tensor_type.elem_type)}


def describe(model):
    return {"inputs": [tensor_info(t) for t in model.graph.input],
            "outputs": [tensor_info(t) for t in model.graph.output],
            "initializers": [{"name": t.name, "shape": list(t.dims), "dtype": TensorProto.DataType.Name(t.data_type)} for t in model.graph.initializer],
            "node_types": dict(Counter(n.op_type for n in model.graph.node))}


def check_runtime(model, seed=0):
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    feed = {}
    for tensor in model.graph.input:
        shape = [d.dim_value for d in tensor.type.tensor_type.shape.dim]
        dtype = np_dtype(tensor.type.tensor_type.elem_type)
        feed[tensor.name] = (rng.normal(0, 0.25, shape) if dtype.kind == "f" else rng.integers(0, 2, shape)).astype(dtype)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    # Validate exported primitives; older ORT fusion can introduce unsupported FP16 LayerNorm kernels.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    for spec, data in zip(model.graph.output, session.run(None, feed)):
        expected = [d.dim_value for d in spec.type.tensor_type.shape.dim]
        if list(data.shape) != expected or data.dtype != np_dtype(spec.type.tensor_type.elem_type) or not np.isfinite(data).all():
            raise ValueError("Runtime output shape/dtype/finite-value check failed")
    return {"status": "passed", "provider": "CPUExecutionProvider", "onnxruntime_version": ort.__version__,
            "graph_optimizations": "disabled",
            "scope": "random-input smoke check; not a comparison with original OMC or NPU"}


def filename(case, model):
    def dims(shape):
        return "x".join(map(str, shape)) or "scalar"
    def token(value):
        if isinstance(value, list):
            return "x".join(map(str, value)) or "empty"
        return format(value, ".6g") if isinstance(value, float) else str(value)
    dtype = np_dtype(case.get("dtype", "float16")).name.replace("float", "f")
    parts = [canonical(case["op"]), dtype]
    for i, item in enumerate(case["inputs"]):
        parts.append(("w" if "init" in item else "in") + str(i) + "-" + dims(item["shape"]))
    parts.append("out-" + dims(describe(model)["outputs"][0]["shape"]))
    if "activation" in case:
        parts.append(case["activation"])
    for k, v in sorted(case.get("attributes", {}).items()):
        parts.append(k + "-" + token(v))
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(parts))
    # Keep path components short on Windows; full parameters and seed stay in JSON.
    return slug[:148].rstrip("_.-") + "_" + digest(model_config(case))[:12]


def apply_override(case, expression):
    key, sep, value = expression.partition("=")
    if not sep:
        raise ValueError("--set uses dotted.path=JSON_VALUE")
    value = json.loads(value)
    target = case
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def resolve_npy_hashes(case, base):
    for item in case["inputs"]:
        spec = item.get("init", {})
        if spec.get("kind") == "npy":
            actual = sha256(base / spec["path"])
            if spec.get("sha256", actual) != actual:
                raise ValueError(f"npy sha256 mismatch: {spec['path']}")
            spec["sha256"] = actual


def compiler_setup(config, ddk_root=None, ge_proto=None):
    unknown_keys(config, {"platform", "weight_data_type", "use_origin_format", "io_types", "extra_args", "timeout",
                          "ddk_root", "omg", "loader", "command", "env", "ge_proto", "version_label", "io_layout"}, "compiler")
    root = ddk_root or config.get("ddk_root") or os.environ.get("DDK_PATH") or os.environ.get("DDK_INSTALL_PATH")
    root = Path(root).expanduser().resolve() if root else None
    if os.name == "nt" and not config.get("command"):
        raise ValueError("Linux OMG must run inside WSL/Linux. Use --onnx-only on Windows; see README.")
    platform = config.get("platform", "kirin9030")
    env = os.environ.copy()
    binary = Path(config["omg"]).resolve() if config.get("omg") else root / "tools/tools_omg/master/omg" if root else None
    if root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        lib = binary.parent / "lib64"
        platform_dir = root / "tools/platform" / platform
        env.update(DDK_INSTALL_PATH=str(root), DDK_PATH=str(root), TOOLCHAIN_HOME=str(root),
                   CCE_LIB_DIR=str(lib) + "/", SOC_VERSION=platform, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
        for key, prefixes in {
            "LD_LIBRARY_PATH": [lib, platform_dir / "lib64"],
            "PYTHONPATH": [root / "tools/tools_ascendc/package/python", platform_dir / "ops/impl"],
            "PATH": [root / "ddk/ccec_compiler/bin", Path(sys.executable).parent],
        }.items():
            env[key] = os.pathsep.join([*map(str, prefixes), *([env[key]] if env.get(key) else [])])
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})
    schema = ge_proto or config.get("ge_proto") or os.environ.get("OMC_GE_PROTO") or (root / SCHEMA_RELATIVE if root else None)
    if not schema or not Path(schema).is_file():
        raise ValueError("Set --ddk-root or --ge-proto to the matching DDK ge_ir_pb2.py; required for actual OMC type audit")
    if not config.get("command") and (not binary or not binary.is_file()):
        raise ValueError("Set --ddk-root or compiler.omg to an installed OMG")
    if config.get("command") and (not isinstance(config["command"], list) or not all(isinstance(s, str) for s in config["command"])):
        raise ValueError("compiler.command must be an argv array, not a shell string")
    if not isinstance(config.get("extra_args", []), list) or not all(isinstance(s, str) for s in config.get("extra_args", [])):
        raise ValueError("compiler.extra_args must be an array of strings")
    if config.get("timeout", 300) <= 0:
        raise ValueError("compiler.timeout must be positive")
    if config.get("io_types", "model") not in {"model", "auto"}:
        raise ValueError("compiler.io_types must be model or auto")
    if config.get("io_layout", "standard") not in {"standard", "native"}:
        raise ValueError("compiler.io_layout must be standard or native")
    evidence = {"config": config, "ddk_root": str(root) if root else None,
                "omg": str(binary) if binary else None, "omg_sha256": sha256(binary) if binary and binary.is_file() else None,
                "ge_proto": str(Path(schema).resolve()), "ge_proto_sha256": sha256(schema),
                "python": sys.executable, "onnx_version": onnx.__version__,
                "environment_sha256": digest({k: env.get(k) for k in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "SOC_VERSION")})}
    evidence["fingerprint"] = digest(evidence)
    return {"root": root, "binary": binary, "schema": str(Path(schema).resolve()), "env": env,
            "config": config, "evidence": evidence, "platform": platform}


def compiler_command(setup, model, source, prefix):
    cfg = setup["config"]
    native = (cfg.get("io_layout", "standard") == "native"
              and model.graph.name.removesuffix("_parameter_test") in BLOCKED_IO_OPS
              and not any(n.op_type == "LayerNormalization" for n in model.graph.node))
    input_format, output_format = "", ""
    if native:
        if any(len(t.type.tensor_type.shape.dim) != 4 for t in [*model.graph.input, *model.graph.output]):
            raise ValueError("Native NC1HWC0 interfaces require rank 4 in this DDK; pad singleton dimensions and adjust axes, or use --io-layout standard")
        input_format = ";".join(t.name + ":NC1HWC0" for t in model.graph.input)
        # OMG output_format addresses producer node names, not ONNX tensor names.
        producers = {output: n.name for n in model.graph.node for output in n.output}
        output_format = ";".join(producers[t.name] + ":NC1HWC0" for t in model.graph.output)
    if cfg.get("command"):
        mapping = {"model": str(source), "output": str(prefix), "platform": setup["platform"],
                   "ddk_root": str(setup["root"] or ""), "input_format": input_format, "output_format": output_format}
        command = [part.format(**mapping) for part in cfg["command"]]
    else:
        loader = cfg.get("loader", "auto")
        command = [str(setup["binary"])]
        if loader == "auto":
            candidate = Path("/lib64/ld-linux-x86-64.so.2")
            loader = str(candidate) if candidate.exists() else None
        if loader:
            command.insert(0, loader)
        command += ["--framework", "5", "--model", str(source), "--output", str(prefix),
                    "--target", "omc", "--platform", setup["platform"],
                    "--use_origin_format=" + str(False if native else cfg.get("use_origin_format", True)).lower()]
        if native:
            command += ["--input_format", input_format, "--output_format", output_format]
        if cfg.get("weight_data_type", "FP16"):
            command += ["--weight_data_type", cfg.get("weight_data_type", "FP16")]
        if cfg.get("io_types", "model") == "model":
            for option, tensors in (("--input_type", model.graph.input), ("--output_type", model.graph.output)):
                command += [option, ";".join(t.name + ":" + OMG_TYPES[t.type.tensor_type.elem_type] for t in tensors)]
    return command + cfg.get("extra_args", [])


def run_logged(command, log, env, cwd, timeout):
    start = time.monotonic()
    with Path(log).open("w", encoding="utf-8") as stream:
        stream.write(json_text({"argv": command, "cwd": str(cwd)}) + "\n")
        stream.flush()
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                              start_new_session=(os.name != "nt")) as process:
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                stream.write("\nTIMEOUT\n")
                rc = 124
        stream.write(f"\nreturncode={rc} elapsed_seconds={time.monotonic() - start:.3f}\n")
    return rc


def audit_omc(path, schema):
    from omc_reader import read_omc, STRUCTURAL
    graph = read_omc(path, schema)
    def short(t):
        return {k: t[k] for k in ("name", "shape", "dtype", "layout")}
    records = []
    for identifier, data in graph.nodes(data=True):
        if data["op_type"] in STRUCTURAL:
            continue
        inputs = [short(t) for t in data["input_tensors"]]
        for parent, _, edge in graph.in_edges(identifier, data=True):
            producer, port = graph.nodes[parent], edge["input_port"]
            if not edge["control"] and producer["op_type"] == "Const" and port < len(inputs) and producer["constant_tensor"]:
                inputs[port]["constant_tensor"] = short(producer["constant_tensor"])
        records.append({"name": data["name"], "type": data["op_type"], "region": data["region"],
                        "inputs": inputs, "outputs": [short(t) for t in data["output_tensors"]],
                        "attributes": data["attributes"]})
    interfaces = []
    for _, data in graph.nodes(data=True):
        if data["op_type"] in {"Data", "NetOutput"} and graph.graph["regions"][data["region"]]["parent_node"] is None:
            tensors = data["output_tensors"] if data["op_type"] == "Data" else data["input_tensors"]
            interfaces.append({"name": data["name"], "kind": data["op_type"],
                               "tensors": [{**short(t), "buffer_bytes": int(t["descriptor"].get("size", 0))} for t in tensors]})
    return {"types": dict(sorted(Counter(r["type"] for r in records).items())), "nodes": records, "interfaces": interfaces,
            "schema_sha256": graph.graph["schema_sha256"], "warnings": graph.graph["warnings"]}


def generate_case(case, base, out, setup=None, runtime=False):
    case = copy.deepcopy(case)
    case["op"] = canonical(case["op"])
    resolve_npy_hashes(case, base)
    model = build_model(case, base)
    name = filename(case, model)
    directory = out / case["op"]
    directory.mkdir(parents=True, exist_ok=True)
    source, compiled, config_path = [directory / (name + suffix) for suffix in (".onnx", ".omc", ".json")]
    record_path = directory / "logs" / digest(model_config(case))[:12] / "record.json"
    payload = model.SerializeToString(deterministic=True)
    model_hash = hashlib.sha256(payload).hexdigest()
    if source.exists() and sha256(source) != model_hash:
        raise FileExistsError(f"Different ONNX already exists: {source}; use a new --out directory")
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    if previous and previous["config_sha256"] != digest(model_config(case)):
        raise ValueError("Filename hash collision; use another output directory")
    if not source.exists():
        source.write_bytes(payload)
    save_operator_config(config_path, case, base)
    record = {"id": case.get("id", case["op"]), "op": case["op"], "case": case,
              "config_sha256": digest(model_config(case)), "onnx": source.name, "onnx_sha256": model_hash,
              "onnx_info": describe(model), "status": "ONNX_READY", "original_omc_numerical_equivalence": "not_verified"}
    if case.get("source", {}).get("preset_sha256"):
        record["matches_handfree_preset"] = digest(model_config(case)) == case["source"]["preset_sha256"]
    if previous and previous["onnx_sha256"] == model_hash:
        if "runtime_check" in previous:
            record["runtime_check"] = previous["runtime_check"]
        old = previous.get("compile", {})
        if compiled.is_file() and old.get("status") == "COMPILED" and old.get("omc_sha256") == sha256(compiled):
            record.update(compile=old, status="COMPILED")
    if runtime:
        record["runtime_check"] = check_runtime(model, case.get("seed", 0))
    if setup:
        old = record.get("compile")
        fingerprint = setup["evidence"]["fingerprint"]
        if compiled.exists():
            if not old or old.get("status") != "COMPILED" or old.get("fingerprint") != fingerprint or old.get("omc_sha256") != sha256(compiled):
                raise FileExistsError(f"Existing OMC is unverified or compiler changed: {compiled}; use a new --out")
        else:
            # OMG has a stricter path-length limit than ONNX; keep temporary paths short.
            jobs = directory / "logs" / record["config_sha256"][:12]
            jobs.mkdir(parents=True, exist_ok=True)
            job = jobs / ("attempt_" + str(time.time_ns()))
            job.mkdir()
            # Publish only after return code, IMOD parsing, and ONNX integrity all pass.
            temporary_prefix = job / "compiled"
            command = compiler_command(setup, model, source.resolve(), temporary_prefix.resolve())
            save_json(job / "command.json", {"argv": command, "compiler": setup["evidence"], "onnx_sha256": model_hash})
            attempt = {"status": "FAILED", "fingerprint": fingerprint, "compiler": setup["evidence"],
                       "command": command, "log": (job / "convert.log").relative_to(directory).as_posix()}
            try:
                rc = run_logged(command, job / "convert.log", setup["env"], job, setup["config"].get("timeout", 300))
                attempt["returncode"] = rc
                produced = Path(str(temporary_prefix) + ".omc")
                if rc or not produced.is_file():
                    raise RuntimeError(f"OMG failed (returncode={rc}); see {job / 'convert.log'}")
                audit = audit_omc(produced, setup["schema"])
                attempt["actual_omc"] = audit
                if setup["config"].get("io_layout") == "native" and audit["types"].get("TransData", 0):
                    raise RuntimeError(f"Native I/O still contains {audit['types']['TransData']} TransData nodes; inspect temporary OMC/logs or use --io-layout standard")
                if sha256(source) != model_hash:
                    raise RuntimeError("ONNX changed during compilation")
                attempt.update(status="COMPILED", omc=compiled.name, omc_sha256=sha256(produced),
                               omc_bytes=produced.stat().st_size, actual_omc=audit)
                produced.replace(compiled)
            except Exception as exc:
                attempt["status"] = "FAILED"
                attempt["error"] = str(exc)
                record.update(status="FAILED", compile=attempt)
                save_json(record_path, record)
                raise
            record.update(status="COMPILED", compile=attempt)
    expected = case.get("expect_omc_type")
    if expected and "compile" in record and record["compile"]["status"] == "COMPILED":
        actual = record["compile"]["actual_omc"]["types"]
        record["expected_type_present"] = expected in actual
        if expected not in actual:
            record["status"] = "TYPE_MISMATCH"
            save_json(record_path, record)
            raise ValueError(f"Expected {expected}; actual compiled types {actual}; OMC retained for inspection")
    save_json(record_path, record)
    return {"id": record["id"], "op": case["op"], "status": record["status"],
            "onnx": source.relative_to(out).as_posix(), "config": config_path.relative_to(out).as_posix(),
            "record": record_path.relative_to(out).as_posix(),
            "omc": compiled.relative_to(out).as_posix() if compiled.is_file() else None,
            "actual_omc_types": record.get("compile", {}).get("actual_omc", {}).get("types", {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "defaults_handfree.json")
    parser.add_argument("--compiler-config", type=Path, default=HERE / "compiler.json")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--op", nargs="+", help="Operator family or case id; default: all cases")
    parser.add_argument("--all", action="store_true", help="Explicitly select all cases")
    parser.add_argument("--list", action="store_true", help="List cases without generating")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=JSON", help="Override one selected case, e.g. attributes.transB=1")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--check-runtime", action="store_true", help="Also run ONNX Runtime CPU smoke checks")
    parser.add_argument("--ddk-root", type=Path)
    parser.add_argument("--ge-proto", type=Path)
    parser.add_argument("--platform", help="Override compiler platform, e.g. kirin9030")
    parser.add_argument("--io-layout", choices=("native", "standard"), help="Native blocked interfaces remove boundary TransData; standard keeps normal tensor I/O")
    args = parser.parse_args(argv)
    cases, compiler = load_config(args.config)
    if args.op and args.all:
        parser.error("Use --op or --all, not both")
    if args.op:
        chosen = []
        for selector in args.op:
            matches = [c for c in cases if c.get("id", c["op"]).lower() == selector.lower()]
            if not matches:
                matches = [c for c in cases if canonical(c["op"]) == canonical(selector)]
            if not matches:
                parser.error(f"No configured case for {selector}")
            for case in matches:
                if case not in chosen:
                    chosen.append(case)
        cases = chosen
    if not cases:
        parser.error("No cases selected")
    if args.list:
        for case in cases:
            print(case.get("id", case["op"]), canonical(case["op"]), [i["shape"] for i in case["inputs"]], flush=True)
        return 0
    if args.set and len(cases) != 1:
        parser.error("--set requires exactly one selected case")
    for expression in args.set:
        if expression.startswith("params."):
            expression = expression[len("params."):]
        apply_override(cases[0], expression)
    if args.compiler_config.is_file():
        compiler = {**json.loads(args.compiler_config.read_text(encoding="utf-8-sig")), **compiler}
    if args.platform:
        compiler["platform"] = args.platform
    if args.io_layout:
        compiler["io_layout"] = args.io_layout
    setup = None if args.onnx_only else compiler_setup(compiler, args.ddk_root, args.ge_proto)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": str(args.config.resolve()), "onnx_version": onnx.__version__, "cases": []}
    failures = 0
    for case in cases:
        try:
            row = generate_case(case, args.config.resolve().parent, out, setup, args.check_runtime)
            print(row["status"], row["op"], row["actual_omc_types"], flush=True)
        except Exception as exc:
            failures += 1
            row = {"id": case.get("id", case.get("op")), "op": case.get("op"), "status": "FAILED", "error": str(exc)}
            print("FAILED", row["id"], str(exc), file=sys.stderr, flush=True)
        summary["cases"].append(row)
    summary["failures"] = failures
    manifest = out / ("run_" + str(time.time_ns()) + ".json")
    save_json(manifest, summary)
    print("Manifest:", manifest, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
