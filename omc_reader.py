"""Read-only IMOD/GE reader. No ONNX import, reconstruction, or binary rewriting.

The partition/descriptor approach follows tools/onnx_graph/omc.py. Kept independent
because that legacy package imports ONNX and performs ONNX name repair.
"""

import ast
import base64
import hashlib
import os
from pathlib import Path
import struct

import networkx as nx
from google.protobuf import descriptor_pool, json_format, message_factory
from google.protobuf.message import DecodeError

STRUCTURAL = {"Const", "Data", "NetOutput", "GraphOp"}
SCHEMA_RELATIVE = Path("tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py")


def schema(path=None):
    path = path or os.environ.get("OMC_GE_PROTO")
    if path is None and os.environ.get("DDK_PATH"):
        path = Path(os.environ["DDK_PATH"]) / SCHEMA_RELATIVE
    if path is None:
        local = Path(__file__).resolve().parents[2] / "spatialnet_field_kit/microbenchmark/.local"
        found = sorted(p / SCHEMA_RELATIVE for p in local.glob("DDK-tools-*")
                       if (p / SCHEMA_RELATIVE).is_file())
        if len(found) != 1:
            raise ValueError("Set DDK_PATH or --ge-proto to the installed SDK ge_ir_pb2.py")
        path = found[0]
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    descriptors = set()
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.keyword) and node.arg == "serialized_pb":
            value = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "AddSerializedFile" and len(node.args) == 1):
            value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, bytes):
            descriptors.add(value.value)
    if len(descriptors) != 1:
        raise ValueError("GE schema must contain exactly one literal protobuf descriptor")
    payload = descriptors.pop()
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(payload)
    # Support both modern protobuf and the protobuf 3.x bundled with older DDKs.
    factory = getattr(message_factory, "GetMessageClass", None) or message_factory.MessageFactory(pool).GetPrototype
    classes = [factory(pool.FindMessageTypeByName("ge.proto." + name)) for name in ("ModelDef", "GraphDef")]
    return *classes, payload, str(path.resolve())


def partitions(raw):
    if len(raw) < 260 or raw[:4] != b"IMOD":
        raise ValueError("Only unencrypted IMOD OMC/OM is supported")
    header, version = struct.unpack_from("<II", raw, 4)
    if header != 256 or version != 0x10000000 or raw[80]:
        raise ValueError("Unverified IMOD version/header or encrypted model")
    if struct.unpack_from("<I", raw, 76)[0] != len(raw) - header:
        raise ValueError("IMOD length mismatch (truncated or unsupported model)")
    count = struct.unpack_from("<I", raw, header)[0]
    for width, fmt in ((4, "<III"), (8, "<QQQ")):
        start = header + width + count * 3 * width
        if not 1 <= count <= 32 or start > len(raw):
            continue
        rows = []
        for i in range(count):
            kind, offset, size = struct.unpack_from(fmt, raw, header + width + i * 3 * width)
            rows.append(dict(type=kind, offset=start + offset, size=size))
        ordered = sorted(rows, key=lambda p: p["offset"])
        if (len({p["type"] for p in rows}) != count or not any(p["type"] == 0 and p["size"] > 0 for p in rows)
                or any(p["type"] >= 32 or p["offset"] + p["size"] > len(raw) for p in rows)
                or any(a["offset"] + a["size"] > b["offset"] for a, b in zip(ordered, ordered[1:]))):
            continue
        return rows
    raise ValueError("Invalid or unsupported IMOD partition table")


def dictionary(message):
    return json_format.MessageToDict(message, preserving_proto_field_name=True,
                                    use_integers_for_enums=True)


def attribute(value):
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    field = getattr(value, kind)
    if kind == "s":
        try:
            return field.decode("utf-8")
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(field).decode("ascii")}
    if kind == "bt":
        return {"base64": base64.b64encode(field).decode("ascii")}
    if kind in {"i", "f", "b", "dt"}:
        return field
    return dictionary(field)


def tensor(desc):
    dtype = desc.DESCRIPTOR.fields_by_name["dtype"].enum_type.values_by_number.get(desc.dtype)
    return dict(name=desc.name, shape=list(desc.shape.dim) if desc.HasField("shape") else None,
                dtype=dtype.name if dtype else f"UNKNOWN({desc.dtype})", layout=desc.layout,
                descriptor=dictionary(desc))


def read_omc(source, ge_proto=None):
    path = Path(source).resolve()
    raw = path.read_bytes()
    parts = partitions(raw)
    Model, Graph, descriptor, schema_path = schema(ge_proto)
    definition = next(p for p in parts if p["type"] == 0)
    try:
        model = Model.FromString(raw[definition["offset"]:definition["offset"] + definition["size"]])
    except DecodeError as exc:
        raise ValueError("Model definition is incompatible with the installed GE schema") from exc
    if not model.graph:
        raise ValueError("No GE GraphDef in model")
    graph = nx.MultiDiGraph()
    graph.graph.update(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                       format="IMOD/GE", header_version="0x10000000", partitions=parts,
                       schema_path=schema_path, schema_sha256=hashlib.sha256(descriptor).hexdigest(),
                       model_name=model.name, model_version=model.version, model_custom_version=model.custom_version,
                       model_definition_base64=base64.b64encode(raw[definition["offset"]:definition["offset"] + definition["size"]]).decode("ascii"),
                       model_attributes={k: attribute(v) for k, v in model.attr.items()},
                       regions=[], warnings=[], export_capability={
                           "json": True, "executable_omc": False, "backend": None,
                           "reason": "No validated supported compiled-OMC subgraph rebuild backend",
                           "load_validated": False, "inference_validated": False})
    weights = next((p for p in parts if p["type"] == 1), None)

    def visit(region, parent=None, depth=0, weight_base=0):
        if depth > 64:
            raise ValueError("Nested GraphOp depth exceeds 64")
        rid = len(graph.graph["regions"])
        ids = list(range(len(graph), len(graph) + len(region.op)))
        graph.graph["regions"].append(dict(id=rid, name=region.name, parent_node=parent,
                                           nodes=ids, inputs=list(region.input), outputs=list(region.output),
                                           attributes={k: attribute(v) for k, v in region.attr.items()}))
        names = {}
        for n, op in zip(ids, region.op):
            if not op.name or op.name in names:
                raise ValueError(f"Region {rid}: empty/duplicate OMC name {op.name!r}; dependencies ambiguous")
            names[op.name] = n
            attrs = {k: attribute(v) for k, v in op.attr.items()}
            graph.add_node(n, name=op.name, op_type=op.type, region=rid, attributes=attrs,
                           input_tensors=[tensor(d) for d in op.input_desc],
                           output_tensors=[tensor(d) for d in op.output_desc],
                           input_refs=list(op.input), constant_tensor=None, weight=None,
                           proto_base64=base64.b64encode(op.SerializeToString()).decode("ascii"),
                           time_us=None, profile_samples=0)
            if op.type == "Const" and "value" in op.attr and op.attr["value"].WhichOneof("value") == "t":
                value = op.attr["value"].t
                graph.nodes[n]["constant_tensor"] = tensor(value.desc)
                if value.data:
                    graph.nodes[n]["weight"] = dict(storage="inline", bytes=len(value.data),
                                                    sha256=hashlib.sha256(value.data).hexdigest(),
                                                    base64=base64.b64encode(value.data).decode("ascii"))
                elif value.desc.weight_size and weights:
                    offset = weights["offset"] + weight_base + value.desc.data_offset
                    size = value.desc.weight_size
                    if (offset < weights["offset"] or offset + size > weights["offset"] + weights["size"]):
                        raise ValueError(f"Weight span outside partition for {op.name}")
                    graph.nodes[n]["weight"] = dict(storage="compiled_weight_partition", offset=offset,
                                                    bytes=size, sha256=hashlib.sha256(raw[offset:offset + size]).hexdigest(),
                                                    numeric_unpack_verified=False)
                else:
                    graph.graph["warnings"].append(f"Const {n}: weight payload unresolved")
        for n, op in zip(ids, region.op):
            for port, ref in enumerate(op.input):
                if not ref:
                    continue
                producer, sep, index = ref.rpartition(":")
                if not sep or producer not in names:
                    raise ValueError(f"Region {rid}: unresolved dependency {ref!r}")
                try:
                    index = int(index)
                except ValueError as exc:
                    raise ValueError(f"Invalid tensor port: {ref!r}") from exc
                u = names[producer]
                if index < -1 or index >= len(graph.nodes[u]["output_tensors"]):
                    raise ValueError(f"Out-of-range producer port: {ref!r}")
                graph.add_edge(u, n, output_port=index, input_port=port, control=index == -1,
                               tensor=ref, region=rid)
            if op.type == "GraphOp":
                if "subgraph" not in op.attr:
                    raise ValueError(f"GraphOp {op.name} has no readable subgraph")
                sub = op.attr["subgraph"]
                kind = sub.WhichOneof("value")
                child = Graph.FromString(sub.bt) if kind == "bt" else sub.g if kind == "g" else None
                if child is None or not child.op:
                    raise ValueError(f"GraphOp {op.name} has an unsupported/empty subgraph")
                visit(child, n, depth + 1, weight_base + int(graph.nodes[n]["attributes"].get("graphop_weight_offset", 0)))

    for region in model.graph:
        visit(region)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Cyclic GE graph: this bounded DAG discovery version does not support control-flow cycles")
    graph.graph["warnings"].append("GraphOp regions remain separate; no cross-region port mapping is invented")
    return graph


def computational_nodes(graph):
    return {n for n, d in graph.nodes(data=True) if d["op_type"] not in STRUCTURAL}


def with_constants(graph, core):
    result, pending = set(core), list(core)
    while pending:
        for u in graph.predecessors(pending.pop()):
            if u not in result and graph.nodes[u]["op_type"] == "Const":
                result.add(u)
                pending.append(u)
    return frozenset(result)
