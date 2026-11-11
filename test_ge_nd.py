"""AIR contract tests; actual OMG/device evidence is in verification/boundary_31us.json."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from onnx import numpy_helper

import generate_ops as g
from ge_nd import build_air
from omc_reader import schema


class LayoutTests(unittest.TestCase):
    def test_core_ports_not_model_input_labels(self):
        audit = {"nodes": [{"type": "Mul", "inputs": [{"layout": "ND"}, {"layout": "ND"}]}],
                 "interfaces": [{"kind": "Data", "tensors": [{"layout": "NCHW"}]}]}
        g.check_handfree_layouts({"op": "Mul"}, audit)
        audit["nodes"][0]["inputs"][1]["layout"] = "NC1HWC0"
        with self.assertRaisesRegex(RuntimeError, "input 1 must be ND"):
            g.check_handfree_layouts({"op": "Mul"}, audit)

    def test_bias_uses_actual_constant_descriptor(self):
        bias = {"shape": [1], "layout": "ND", "constant_tensor": {"shape": [1, 1, 1, 1], "layout": "ND"}}
        audit = {"nodes": [{"type": "ConvTranspose", "inputs": [{}, {}, bias]}]}
        case = {"op": "ConvTranspose", "inputs": [{}, {}, {}]}
        with self.assertRaisesRegex(RuntimeError, "bias must be 1-D ND"):
            g.check_handfree_layouts(case, audit)
        bias["constant_tensor"]["shape"] = [16]
        g.check_handfree_layouts(case, audit)
        bias["layout"] = "NC1HWC0"
        with self.assertRaises(RuntimeError):
            g.check_handfree_layouts(case, audit)

    def test_incorrect_bias_is_not_published(self):
        case = next(c for c in g.load_config(g.HERE / "defaults_handfree.json")[0] if c["op"] == "ConvTranspose")
        case.pop("attributes")  # Keep filenames short under Windows sandbox temp paths.
        audit = {"types": {"ConvTranspose": 1}, "nodes": [{"type": "ConvTranspose", "inputs": [
            {}, {}, {}, {"layout": "NC1HWC0", "shape": [1],
                         "constant_tensor": {"layout": "NC1HWC0", "shape": [1, 1, 1, 1]}}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            command = [sys.executable, "-c", "import pathlib,sys;pathlib.Path(sys.argv[1]+'.omc').write_bytes(b'test')", "{output}"]
            setup = {"config": {"command": command, "io_layout": "handfree"}, "root": None,
                     "platform": "test", "schema": "unused", "env": None, "evidence": {"fingerprint": "test"}}
            with patch.object(g, "audit_omc", return_value=audit), self.assertRaisesRegex(RuntimeError, "bias must be 1-D ND"):
                g.generate_case(case, out, out, setup)
            self.assertEqual(list((out / "ConvTranspose").glob("*.omc")), [])

    def test_air_command_uses_framework_one_without_blocked_interface(self):
        case = {"op": "Mul", "inputs": [{"name": "X", "shape": [2, 3]}, {"name": "W", "shape": [2, 3]}]}
        setup = {"config": {"io_layout": "handfree", "loader": None}, "binary": Path("omg"), "platform": "kirin9030"}
        command = g.compiler_command(setup, g.build_model(case), Path("model.air"), Path("result"), framework=1)
        self.assertEqual(command[command.index("--framework") + 1], "1")
        self.assertNotIn("--input_format", command)
        self.assertEqual(command[command.index("--model") + 1], "model.air")


class AirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.Model, _, _, cls.schema_path = schema()
        except (ValueError, FileNotFoundError) as exc:
            raise unittest.SkipTest("Set OMC_GE_PROTO or DDK_PATH to run AIR schema checks: " + str(exc))

    def test_layernorm_constants_axes_and_types_match_onnx(self):
        case = next(c for c in g.load_config(g.HERE / "defaults_handfree.json")[0] if c["op"] == "LayerNorm")
        case = copy.deepcopy(case)
        case["attributes"]["axis"] = -1
        model = g.build_model(case)
        raw = build_air(case, model, self.schema_path)
        self.assertEqual(raw, build_air(case, model, self.schema_path))
        graph = self.Model.FromString(raw).graph[0]
        nodes = {n.name: n for n in graph.op}
        self.assertEqual(nodes["Y"].attr["begin_norm_axis"].i, 1)
        self.assertEqual(nodes["Y"].attr["end_norm_axis"].i, 1)
        self.assertTrue(all(d.layout == "ND" for d in nodes["Y"].input_desc))
        self.assertEqual(nodes["X"].output_desc[0].layout, "NCHW")
        for tensor in model.graph.initializer:
            value = nodes[tensor.name].attr["value"].t
            self.assertEqual(list(value.desc.shape.dim), list(tensor.dims))
            self.assertEqual(value.desc.dtype, 2)
            np.testing.assert_array_equal(np.frombuffer(value.data, dtype=np.float16).reshape(tensor.dims),
                                          numpy_helper.to_array(tensor))

    def test_mul_broadcast_inputs_keep_individual_shapes(self):
        case = {"op": "Mul", "dtype": "float16", "inputs": [
            {"name": "X", "shape": [2, 3, 4]}, {"name": "W", "shape": [4]}]}
        graph = self.Model.FromString(build_air(case, g.build_model(case), self.schema_path)).graph[0]
        core = next(n for n in graph.op if n.type == "Mul")
        self.assertEqual([list(d.shape.dim) for d in core.input_desc], [[2, 3, 4], [4]])
        self.assertEqual(list(core.output_desc[0].shape.dim), [2, 3, 4])

    def test_layernorm_per_input_dtype_override_is_preserved(self):
        case = copy.deepcopy(next(c for c in g.load_config(g.HERE / "defaults_handfree.json")[0] if c["op"] == "LayerNorm"))
        for item in case["inputs"]:
            item["dtype"] = "float32"
        model = g.build_model(case)
        graph = self.Model.FromString(build_air(case, model, self.schema_path)).graph[0]
        self.assertEqual([d.dtype for n in graph.op if n.type == "LayerNorm" for d in n.input_desc], [1, 1, 1])
        case["attributes"]["stash_type"] = 10
        with self.assertRaisesRegex(ValueError, "float32 accumulation"):
            build_air(case, model, self.schema_path)


if __name__ == "__main__":
    unittest.main()
