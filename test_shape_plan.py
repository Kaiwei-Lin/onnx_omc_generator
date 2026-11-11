import unittest

import generate_ops as g
from shape_plan import build_case, plan
from adapt_shapes import scaled_parameters, proposals, memory_bytes, merge_measurements


class ShapePlanTests(unittest.TestCase):
    def test_requested_matrix_is_complete_and_valid(self):
        groups = plan()
        self.assertEqual(len(groups), 59)
        self.assertEqual(sum(len(x["cases"]) for x in groups), 492)
        self.assertEqual(len({c["hash"] for group in groups for c in group["cases"]}), 462)
        for group in groups:
            for row in group["cases"]:
                g.validate_case({"op": row["config"]["op"], **row["config"]["params"]})
        same = next(x for x in groups if x["id"] == "MatMul_same_flops")
        self.assertEqual({c["parameters"]["M"] * c["parameters"]["N"] * c["parameters"]["K"] for c in same["cases"]}, {67108864})

    def test_parameter_changes_keep_weights_and_constant_roles_consistent(self):
        mm = build_case("MatMul", dict(M=8, N=512, K=4096))
        self.assertIn("init", mm["inputs"][0])
        self.assertNotIn("init", mm["inputs"][1])
        gemm = build_case("GemmD", dict(M=128, N=33, K=288))
        self.assertEqual([x["shape"] for x in gemm["inputs"]], [[128,288],[288,33],[33]])
        conv = build_case("Convolution", dict(Cin=96, Cout=96, H=32, W=32, G=96))
        self.assertEqual(conv["inputs"][1]["shape"], [96,1,3,3])

    def test_nd_route_does_not_pad_user_rank(self):
        one = build_case("Add", dict(a=[1024]))
        model = g.build_model(one)
        self.assertTrue(g.needs_nd_air(one, model))
        self.assertEqual(g.describe(model)["outputs"][0]["shape"], [1024])
        four = build_case("Add", dict(a=[1,16,8,8]))
        self.assertFalse(g.needs_nd_air(four, g.build_model(four)))
        broadcast = build_case("Add", dict(a=[1,16,256,128], b=[1,16,256,1]))
        self.assertTrue(g.needs_nd_air(broadcast, g.build_model(broadcast)))

    def test_scaling_preserves_broadcast_and_marks_fixed_dimension_change(self):
        group = {"op": "Mul", "id": "Mul_broadcast", "grow": ["tensor"]}
        p, _ = scaled_parameters(group, dict(a=[1,256,2048,1], b=[1,1,2048,1], constant=True), True)
        self.assertEqual(p["a"], [1,256,4096,1])
        self.assertEqual(p["b"], [1,1,4096,1])
        group = {"op": "MatMul", "id": "MatMul_small_m", "grow": ["M"]}
        p, changes = scaled_parameters(group, dict(M=1,N=512,K=4096), False)
        self.assertEqual(p, dict(M=1,N=256,K=4096))
        self.assertEqual(changes, ["fixed:N"])

    def test_adjustment_reuses_measured_larger_case(self):
        group = next(x for x in plan() if x["id"] == "Add_small")
        known = {c["hash"]: {"status": "above_31"} for c in group["cases"]}
        known[group["cases"][0]["hash"]] = {"status": "at_or_below_31"}
        pending, decisions = proposals([group], known)
        self.assertEqual(pending, {})
        self.assertEqual(decisions[0]["result"]["status"], "replacement_above_31")
        self.assertEqual(decisions[0]["result"]["parameters"]["a"], [64])

    def test_memory_bound_includes_large_matmul_output(self):
        c = build_case("MatMul", dict(M=8192,N=8192,K=1))
        self.assertGreater(memory_bytes(c), 128 * 1024 * 1024)

    def test_confirmation_keeps_previous_below_threshold_sample(self):
        a = dict(target="Y(Mul)", threshold_us=31, values_us=[99, 31, 32], warmup=1)
        b = dict(target="Y(Mul)", threshold_us=31, values_us=[99, 33, 34], warmup=1)
        measured = merge_measurements([a, b])
        self.assertFalse(measured["all_samples_above_threshold"])
        self.assertEqual(measured["samples"], 4)
        self.assertEqual(measured["rounds"], 2)

    def test_small_normalization_control_preserves_nontrivial_reduction(self):
        group = dict(op="LayerNorm", id="LayerNorm_short", grow=["tensor"])
        p, changes = scaled_parameters(group, dict(a=[1024,2], axis=1), False)
        self.assertEqual(p, dict(a=[512,2], axis=1))
        self.assertIn("fixed-dimension", changes)

    def test_failed_length_growth_tries_more_independent_rows(self):
        parameters = dict(a=[16,2048], axis=1)
        cases = [build_case("LayerNorm", p) for p in (parameters, dict(a=[16,4096],axis=1), dict(a=[32,2048],axis=1))]
        keys = [g.digest(g.model_config(c))[:12] for c in cases]
        group = dict(op="LayerNorm", id="LayerNorm_long", grow=["tensor"],
                     cases=[dict(parameters=parameters, hash=keys[0])])
        known = {k: dict(status=s) for k,s in zip(keys, ("at_or_below_31", "profile_failed", "above_31"))}
        pending, decisions = proposals([group], known)
        self.assertFalse(pending)
        result = decisions[0]["result"]
        self.assertEqual(result["parameters"]["a"], [32,2048])
        self.assertEqual(result["fallback_from_failed_hash"], keys[1])
        self.assertEqual(result["changed"], ["fixed:a.0"])

    def test_small_control_can_shrink_rows_when_shorter_length_is_invalid(self):
        p = dict(a=[1024,8],axis=1)
        keys = [g.digest(g.model_config(build_case("LayerNorm", x)))[:12] for x in (p, dict(a=[1024,4],axis=1), dict(a=[512,8],axis=1))]
        group = dict(op="LayerNorm",id="LayerNorm_short",grow=["tensor"],cases=[dict(parameters=p,hash=keys[0])])
        known = {k:dict(status=s) for k,s in zip(keys,("above_31","profile_failed","at_or_below_31"))}
        pending, decisions = proposals([group],known)
        self.assertFalse(pending)
        self.assertEqual(decisions[0]["result"]["parameters"]["a"], [512,8])
        self.assertEqual(decisions[0]["result"]["status"], "smaller_control")


if __name__ == "__main__":
    unittest.main()
