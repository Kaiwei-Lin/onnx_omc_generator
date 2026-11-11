"""Portable checks: python -m unittest discover -s . -p 'test_*.py' -v"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnxruntime as ort

import generate_ops as g

DEFAULTS = json.loads((g.HERE / "defaults_handfree.json").read_text(encoding="utf-8"))


def case(op, shapes, attrs=None, **extra):
    return {"op": op, "dtype": "float32", "opset": 13, "seed": 1,
            "inputs": [{"name": "input" + str(i), "shape": s} for i, s in enumerate(shapes)],
            "attributes": attrs or {}, **extra}


def evaluate(config, values):
    model = g.build_model(config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    feed = {info.name: np.asarray(value, dtype=g.np_dtype(info.type.tensor_type.elem_type))
            for info, value in zip(model.graph.input, values)}
    return session.run(None, feed)[0]


class GeneratorTests(unittest.TestCase):
    def test_all_handfree_defaults(self):
        self.assertEqual({c["op"] for c in DEFAULTS["cases"]}, set(g.OPS))
        for config in DEFAULTS["cases"]:
            with self.subTest(op=config["op"]):
                self.assertEqual(g.digest(g.model_config(config)), config["source"]["preset_sha256"])
                model = g.build_model(config)
                g.check_runtime(model, 42)
                if "outputs" in config["source"]:
                    self.assertEqual(g.describe(model)["outputs"][0]["shape"], config["source"]["outputs"][0]["shape"])
        softmax = next(c for c in DEFAULTS["cases"] if c["op"] == "Softmax")
        self.assertEqual(softmax["source"]["kind"], "generic_example")

    def test_gemm_transpose_scaling_bias(self):
        a, b, c = np.arange(6).reshape(3, 2) / 10, np.arange(12).reshape(4, 3) / 20, np.arange(4) / 5
        cfg = case("GemmD", [[3, 2], [4, 3], [4]], {"transA": 1, "transB": 1, "alpha": 0.5, "beta": 2.0})
        np.testing.assert_allclose(evaluate(cfg, [a, b, c]), 0.5 * a.T @ b.T + 2 * c, rtol=1e-6)
        cfg["inputs"].pop()
        np.testing.assert_allclose(evaluate(cfg, [a, b]), 0.5 * a.T @ b.T, rtol=1e-6)

    def test_constant_left_matmul_rewrite(self):
        cfg = case("MatMul", [[2, 3], [3, 4]], constant_left_rewrite=True)
        a = np.arange(6).reshape(2, 3) / 10
        b = np.arange(12).reshape(3, 4) / 10
        cfg["inputs"][0]["init"] = {"kind": "values", "values": a.tolist()}
        model = g.build_model(cfg)
        self.assertEqual(list(model.graph.initializer[0].dims), [3, 2])
        self.assertEqual([n.op_type for n in model.graph.node], ["Transpose", "MatMul", "Transpose"])
        np.testing.assert_allclose(evaluate(cfg, [b]), a @ b, rtol=1e-6)
        cfg["constant_left_rewrite"] = False
        np.testing.assert_allclose(evaluate(cfg, [b]), a @ b, rtol=1e-6)

    def test_batched_transpose_and_broadcast(self):
        cfg = case("BatchMatMul", [[2, 3, 4], [1, 5, 3]], transpose_a=True, transpose_b=True)
        a, b = np.arange(24).reshape(2, 3, 4) / 10, np.arange(15).reshape(1, 5, 3) / 10
        np.testing.assert_allclose(evaluate(cfg, [a, b]), a.swapaxes(-1, -2) @ b.swapaxes(-1, -2), rtol=1e-6)

    def test_matmul_vector_and_scalar_output(self):
        cfg = case("MatMul", [[3], [3]])
        self.assertEqual(evaluate(cfg, [[1, 2, 3], [2, 3, 4]]).shape, ())
        self.assertEqual(float(evaluate(cfg, [[1, 2, 3], [2, 3, 4]])), 20)

    def test_broadcast_binary(self):
        x, b = np.arange(24).reshape(2, 3, 4) / 10, np.arange(3).reshape(3, 1) / 10
        for op, fn in (("Add", np.add), ("Mul", np.multiply)):
            np.testing.assert_allclose(evaluate(case(op, [[2, 3, 4], [3, 1]]), [x, b]), fn(x, b), rtol=1e-6, atol=1e-7)

    def test_prelu_channel_slope(self):
        x = np.linspace(-2, 2, 24).reshape(1, 3, 2, 4)
        slope = np.array([0.1, 0.2, 0.3]).reshape(3, 1, 1)
        cfg = case("PRelu", [[1, 3, 2, 4], [3, 1, 1]])
        np.testing.assert_allclose(evaluate(cfg, [x, slope]), np.where(x < 0, x * slope, x), rtol=1e-6)

    def test_reduce_axes_keepdims_and_empty(self):
        x = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        cfg = case("ReduceSum", [[2, 3, 4]], {"axes": [0, -1], "keepdims": 0})
        np.testing.assert_array_equal(evaluate(cfg, [x]), x.sum(axis=(0, 2)))
        cfg["attributes"] = {"axes": [], "keepdims": 0, "noop_with_empty_axes": 1}
        np.testing.assert_array_equal(evaluate(cfg, [x]), x)
        cfg["attributes"]["noop_with_empty_axes"] = 0
        self.assertEqual(float(evaluate(cfg, [x])), float(x.sum()))

    def test_softmax_axis(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
        reference = np.exp(x - x.max(axis=1, keepdims=True))
        reference /= reference.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(evaluate(case("Softmax", [[2, 3, 4]], {"axis": 1}), [x]), reference, rtol=1e-6)

    def test_layernorm_native_and_decomposed(self):
        rng = np.random.default_rng(4)
        x, scale, bias = rng.normal(size=(2, 3, 4)).astype("float32"), rng.normal(size=(3, 4)).astype("float32"), rng.normal(size=(3, 4)).astype("float32")
        centered = x - x.mean(axis=(1, 2), keepdims=True)
        expected = centered / np.sqrt((centered ** 2).mean(axis=(1, 2), keepdims=True) + 1e-4) * scale + bias
        for implementation, opset in (("decomposed", 13), ("decomposed", 18), ("native", 17)):
            cfg = case("LayerNorm", [[2, 3, 4], [3, 4], [3, 4]], {"axis": 1, "epsilon": 1e-4}, implementation=implementation, accumulation_dtype="float32")
            cfg["opset"] = opset
            np.testing.assert_allclose(evaluate(cfg, [x, scale, bias]), expected, rtol=2e-5, atol=2e-6)

    def test_layernorm_fp32_accumulation_for_fp16_input(self):
        cfg = case("LayerNorm", [[2, 4], [4]], implementation="decomposed", accumulation_dtype="float32")
        cfg["dtype"] = "float16"
        model = g.build_model(cfg)
        self.assertEqual(g.describe(model)["node_types"]["Cast"], 2)
        g.check_runtime(model)

    def test_layernorm_full_fp32_with_fp16_interface(self):
        cfg = case("LayerNorm", [[2, 4], [4], [4]], {"axis": -1, "epsilon": 1e-5},
                   implementation="decomposed", accumulation_dtype="float32", affine_dtype="float32")
        cfg["dtype"] = "float16"
        x = np.array([[0.1, -0.4, 0.5, 0.7], [-0.2, 0.8, 0.2, 0.4]], dtype="float16")
        scale, bias = np.array([0.8, 1.1, 0.7, 0.9], dtype="float16"), np.array([0.1, -0.2, 0.3, 0], dtype="float16")
        centered = x.astype("float32") - x.astype("float32").mean(axis=-1, keepdims=True)
        expected = ((centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)) * scale.astype("float32") + bias.astype("float32")).astype("float16")
        np.testing.assert_allclose(evaluate(cfg, [x, scale, bias]), expected, rtol=1e-3, atol=1e-3)
        cfg["inputs"][1]["init"] = {"kind": "values", "values": scale.tolist()}
        cfg["inputs"][2]["init"] = {"kind": "values", "values": bias.tolist()}
        np.testing.assert_allclose(evaluate(cfg, [x]), expected, rtol=1e-3, atol=1e-3)

    def test_compiler_preserves_cast_interfaces(self):
        cfg = case("CastT", [[2]], {"to": "float16"})
        setup = {"config": {"loader": None}, "binary": Path("omg"), "platform": "kirin9030"}
        command = g.compiler_command(setup, g.build_model(cfg), Path("model.onnx"), Path("output"))
        self.assertEqual(command[command.index("--input_type") + 1], "input0:FP32")
        self.assertEqual(command[command.index("--output_type") + 1], "Y:FP16")

    def test_grouped_conv_stride_padding_dilation(self):
        cfg = case("Convolution", [[1, 2, 5, 5], [2, 1, 2, 2], [2]],
                   {"group": 2, "strides": [2, 1], "pads": [1, 0, 0, 1], "dilations": [2, 1]})
        x, w, bias = np.arange(50).reshape(1, 2, 5, 5) / 10, np.arange(8).reshape(2, 1, 2, 2) / 10, np.array([0.2, -0.3])
        padded = np.pad(x, ((0, 0), (0, 0), (1, 0), (0, 1)))
        expected = np.empty((1, 2, 2, 5))
        for channel in range(2):
            for y in range(2):
                for z in range(5):
                    expected[0, channel, y, z] = np.sum(padded[0, channel, y * 2:y * 2 + 3:2, z:z + 2] * w[channel, 0]) + bias[channel]
        np.testing.assert_allclose(evaluate(cfg, [x, w, bias]), expected, rtol=2e-6, atol=2e-6)

    def test_convtranspose_stride_padding_output_padding(self):
        cfg = case("ConvTranspose", [[1, 1, 2, 3], [1, 1, 2, 2]],
                   {"strides": [2, 2], "pads": [1, 0, 0, 1], "output_padding": [1, 0]})
        x, w = np.arange(6).reshape(1, 1, 2, 3) / 10, np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2)
        full = np.zeros((5, 6))
        for y in range(2):
            for z in range(3):
                full[y * 2:y * 2 + 2, z * 2:z * 2 + 2] += x[0, 0, y, z] * w[0, 0]
        np.testing.assert_allclose(evaluate(cfg, [x, w]), full[1:, :-1].reshape(1, 1, 4, 5), rtol=1e-6)

    def test_activation_and_cast(self):
        cfg = case("Activation", [[2, 3]], {"alpha": 0.125}, activation="LeakyRelu")
        x = np.linspace(-1, 1, 6).reshape(2, 3)
        np.testing.assert_allclose(evaluate(cfg, [x]), np.where(x < 0, x * 0.125, x), rtol=1e-6)
        cast = case("CastT", [[2, 3]], {"to": "int32"})
        np.testing.assert_array_equal(evaluate(cast, [x * 10]), (x * 10).astype(np.float32).astype(np.int32))

    def test_bad_config_is_rejected(self):
        bad = [case("GemmD", [[2, 3], [4, 3]]), case("Add", [[2, 3], [4]]),
               case("Convolution", [[1, 3, 4, 4], [2, 2, 3, 3]], {"group": 2}),
               case("ReduceSum", [[2, 3]], {"axes": [2]}), case("ReduceSum", [[2, 3]], {"axes": [1, -1]}),
               case("Softmax", [[2, 3]], {"axis": 0, "typo": 1}),
               case("Activation", [[2, 3]], activation="NotAnActivation"),
               case("LayerNorm", [[2, 3], [3]], implementation="native"),
               case("MatMul", [[2, 3], [3, 4]], constant_left_rewrite=True)]
        for cfg in bad:
            with self.subTest(config=cfg):
                with self.assertRaises(Exception):
                    g.build_model(cfg)

    def test_seed_and_float_attributes_distinguish_filenames(self):
        a = case("GemmD", [[2, 3], [3, 4]], {"alpha": 0.123456781})
        b = copy.deepcopy(a)
        b["attributes"]["alpha"] = 0.123456782
        self.assertNotEqual(g.filename(a, g.build_model(a)), g.filename(b, g.build_model(b)))
        b = copy.deepcopy(a)
        b["seed"] = 2
        self.assertNotEqual(g.filename(a, g.build_model(a)), g.filename(b, g.build_model(b)))
        self.assertLessEqual(len(g.filename(a, g.build_model(a))), 161)

    def test_overrides_and_alias(self):
        cfg = case("gemm", [[2, 3], [3, 4]])
        g.apply_override(cfg, "inputs.1.shape=[4,3]")
        g.apply_override(cfg, "attributes.transB=1")
        self.assertEqual(g.describe(g.build_model(cfg))["outputs"][0]["shape"], [2, 4])
        self.assertEqual(g.canonical("Convalution"), "Convolution")

    def test_npy_change_updates_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = case("MatMul", [[2, 3], [3, 4]])
            cfg["inputs"][1]["init"] = {"kind": "npy", "path": "w.npy"}
            np.save(base / "w.npy", np.ones((3, 4), dtype="float32"))
            first = copy.deepcopy(cfg)
            g.resolve_npy_hashes(first, base)
            g.build_model(first, base)
            np.save(base / "w.npy", np.zeros((3, 4), dtype="float32"))
            second = copy.deepcopy(cfg)
            g.resolve_npy_hashes(second, base)
            self.assertNotEqual(g.digest(g.model_config(first)), g.digest(g.model_config(second)))
            with self.assertRaises(ValueError):
                g.build_model(first, base)

    def test_reuse_and_protect_existing_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cfg = case("Add", [[2], [2]])
            row = g.generate_case(cfg, out, out)
            self.assertEqual(row, g.generate_case(cfg, out, out))
            (out / row["onnx"]).write_bytes(b"user-edited")
            with self.assertRaises(FileExistsError):
                g.generate_case(cfg, out, out)
            self.assertEqual((out / row["onnx"]).read_bytes(), b"user-edited")

    def test_failed_compiler_never_publishes_omc(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                cfg = case("CastT", [[2]], {"to": "float16"})
                command = [sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]+'.omc').write_bytes(b'partial'); sys.exit(" + str(returncode) + ")", "{output}"]
                setup = {"config": {"command": command, "timeout": 10}, "root": None, "platform": "test",
                         "schema": "unused", "env": None, "evidence": {"fingerprint": "test"}}
                with self.assertRaises((ValueError, RuntimeError)):
                    g.generate_case(cfg, out, out, setup)
                self.assertEqual(list((out / "CastT").glob("*.omc")), [])
                records = list((out / "CastT").glob("*.json"))
                self.assertEqual(len(records), 1)
                self.assertEqual(json.loads(records[0].read_text())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
