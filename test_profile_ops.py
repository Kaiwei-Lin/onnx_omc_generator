import unittest

import numpy as np

from profile_ops import Device, durations, pack, unpack


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


if __name__ == "__main__":
    unittest.main()
