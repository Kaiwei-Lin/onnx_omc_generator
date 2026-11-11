import unittest

import numpy as np

from profile_ops import Device, durations, pack, unpack, target_statistics, numerical_check


class ProfileTests(unittest.TestCase):
    def test_blocked_order_padding_and_roundtrip(self):
        x = np.arange(34, dtype=np.float16).reshape(1, 17, 1, 2)
        t = dict(shape=list(x.shape), dtype="DT_FLOAT16", layout="NC1HWC0", buffer_bytes=160)
        data = pack(x, t)
        self.assertEqual(len(data), 160)
        np.testing.assert_array_equal(np.frombuffer(data, np.float16)[:16], x[0, :16, 0, 0])
        np.testing.assert_array_equal(unpack(data, t), x)
        self.assertEqual(data[-32:], bytes(32))

    def test_plain_romc_buffer_uses_logical_size(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = dict(shape=[3, 4], dtype="DT_FLOAT", layout="NCHW", buffer_bytes=80)
        payload = pack(x, t)
        self.assertEqual(len(payload), 48)
        np.testing.assert_array_equal(unpack(payload, t), x)
        with self.assertRaises(ValueError):
            pack(x, dict(t, buffer_bytes=32))

    def test_core_timing_excludes_other_nodes_and_rejects_bad_samples(self):
        rows = [{"op name": "core(LayerNorm)", "total time(us)": "35"},
                {"op name": "cast(CastT)", "total time(us)": "100"},
                {"op name": "core(LayerNorm)", "total time(us)": "36"}]
        self.assertEqual(durations(rows, "op name", "core(LayerNorm)", 2)[1], [35, 36])
        with self.assertRaises(ValueError):
            durations(rows, "op name", "core(LayerNorm)", 3)
        with self.assertRaises(ValueError):
            durations([{"op name": "x", "total time(us)": "nan"}], "op name", "x", 1)

    def test_remote_failure_is_not_hidden_by_successful_hdc_exit(self):
        device = Device("hdc", "test", "/test")
        device.call = lambda *args, **kwargs: "remote failed\n__BENCH_EXIT__=255"
        with self.assertRaises(RuntimeError):
            device.shell("test")
        device.call = lambda *args, **kwargs: "missing remote status"
        with self.assertRaises(RuntimeError):
            device.shell("test")

    def test_multiple_cores_are_not_summed_and_each_drops_its_warmup(self):
        cores = [dict(name="a", type="Convolution"), dict(name="b", type="Convolution")]
        rows = [{"op name": c["name"] + "(Convolution)", "total time(us)": str(v), "device type": "npu_aicore"}
                for c in cores for v in (99, 20, 22)]
        result = target_statistics(rows, cores, 3, 1, 31)
        self.assertFalse(result["all_samples_above_threshold"])
        self.assertEqual(result["core_count"], 2)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["values_us"][result["warmup"]:], [20, 22, 20, 22])
        self.assertEqual(result["max_us"], 22)
        rows[-1]["device type"] = "cpu"
        with self.assertRaises(ValueError):
            target_statistics(rows, cores, 3, 1, 31)

    def test_half_reduction_requires_exact_secondary_reference(self):
        x = np.array([[2048], [1], [-2048]], dtype=np.float16)
        case = dict(op="ReduceSum", inputs=[dict(name="X")], attributes=dict(axes=[0], keepdims=1))
        expected = x.astype(np.float32).sum(axis=0, keepdims=True)
        result = numerical_check(np.array([[0]], np.float32), expected, case, {"X": x})
        self.assertTrue(result["passed"])
        self.assertFalse(result["onnxruntime_passed"])
        self.assertTrue(result["fp16_sequential_exact"])
        wrong = numerical_check(np.array([[0.25]], np.float32), expected, case, {"X": x})
        self.assertFalse(wrong["passed"])
        nonfinite = numerical_check(np.array([[np.inf]], np.float32), expected, case, {"X": x})
        self.assertFalse(nonfinite["passed"])
        self.assertIsNone(nonfinite["max_abs_error"])
        mismatched = numerical_check(np.ones(1, np.float32), np.ones((1,1), np.float32), case, {"X":x})
        self.assertFalse(mismatched["passed"])
        self.assertEqual(mismatched["reason"], "output_shape_mismatch")


if __name__ == "__main__":
    unittest.main()
