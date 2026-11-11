"""Append larger replacements for slow-to-cross cases and smaller controls for all-pass groups."""
import argparse
import copy
import json
import math
from pathlib import Path

import generate_ops as g
from shape_plan import plan, build_case
from profile_ops import stats

HERE = Path(__file__).resolve().parent


def merge_measurements(measurements):
    """All successful rounds count, including earlier samples at or below 31 us."""
    first = measurements[0]
    keys = ("target", "threshold_us")
    if any(any(m[k] != first[k] for k in keys) for m in measurements):
        raise ValueError("Cannot merge different target kernels or thresholds")
    values = [v for m in measurements for v in m["values_us"][m["warmup"]:]]
    result = {**first, **stats(values), "values_us": values, "warmup": 0, "rounds": len(measurements),
              "all_samples_above_threshold": min(values) > first["threshold_us"]}
    grouped = {}
    for m in measurements:
        for core in m.get("per_core", [m]):
            grouped.setdefault(core["target"], []).extend(core["values_us"][core["warmup"]:])
    result["core_count"] = len(grouped)
    result["per_core"] = [dict(target=name, **stats(v), all_samples_above_threshold=min(v) > first["threshold_us"])
                          for name, v in grouped.items()]
    return result


def load_known(roots):
    known, measurements = {}, {}
    for root in roots:
        for path in (root / "status").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8")); key = row["hash"]
            known[key] = row
            for m in [*row.get("previous_measurements", []), *([row["measurement"]] if "measurement" in row else [])]:
                measurements.setdefault(key, {})[m["op_csv"]] = m
    for key, rounds in measurements.items():
        merged = merge_measurements(list(rounds.values()))
        known[key] = {**known[key], "measurement": merged,
                      "status": "above_31" if merged["all_samples_above_threshold"] else "at_or_below_31"}
    return known


def scaled_parameters(group, original, grow):
    """Double/halve a workload dimension while preserving its broadcast pattern."""
    p = copy.deepcopy(original)
    factor = 2 if grow else 0.5
    changed = []

    def value(v):
        return max(1, int(v * factor))

    axes = group["grow"]
    if axes not in (["tensor"], ["elements"]):
        for key in axes:
            old = p[key]; p[key] = value(old)
            if p[key] != old:
                changed.append(key)
        if not changed and not grow:
            # The scanned dimension is already one; explicitly label a change to a formerly fixed dimension.
            for key in ("N", "M", "B", "K", "H", "W", "Cin", "Cout"):
                if key in p and key not in axes and p[key] > 1:
                    p[key] = value(p[key]); changed.append("fixed:" + key); break
        if group["op"] in {"Convolution", "ConvTranspose"}:
            kernel = p.get("kernel", [3, 3])
            for i, key in enumerate(("H", "W")):
                if group["op"] == "Convolution":
                    p[key] = max(kernel[i], p[key])
            for key in ("Cin", "Cout"):
                div = p.get("G", 1)
                p[key] = max(div, (p[key] // div) * div)
    else:
        shape = p["a"]
        op, identifier = group["op"], group["id"]
        if op in {"ReduceSum", "LayerNorm", "Softmax"}:
            axis = p["axis"] % len(shape)
            if identifier.endswith("fixed_elements"):
                dim = 0
            elif identifier.endswith("rank"):
                dim = max(range(max(1, axis)), key=lambda i: shape[i])
            else:
                dim = axis
        elif identifier.endswith("channel"):
            dim = 1
        elif identifier.endswith("alignment"):
            dim = len(shape) - 1
        else:
            dim = max(range(len(shape)), key=lambda i: shape[i])
        if not grow and op in {"ReduceSum", "LayerNorm", "Softmax"} and dim == p["axis"] % len(shape) and shape[dim] <= 2:
            candidates = [i for i, v in enumerate(shape) if i != dim and v > 1]
            if candidates:
                dim = max(candidates, key=lambda i: shape[i])
                changed.append("fixed-dimension")
        if not grow and shape[dim] == 1:
            candidates = [i for i, v in enumerate(shape) if v > 1]
            if candidates:
                dim = max(candidates, key=lambda i: shape[i])
                changed.append("fixed-dimension")
        old = shape[dim]; shape[dim] = value(old)
        if old != shape[dim]:
            changed.append("a." + str(dim))
            if "b" in p:
                # Right-aligned ONNX broadcasting: singleton axes stay singleton.
                bdim = dim - (len(shape) - len(p["b"]))
                if 0 <= bdim < len(p["b"]) and p["b"][bdim] == old and old != 1:
                    p["b"][bdim] = shape[dim]
    return p, changed


def memory_bytes(case):
    items = case["inputs"]
    inputs = sum(math.prod(t["shape"]) * g.np_dtype(g.input_dtype(case, t)).itemsize for t in items)
    shape = list(items[0]["shape"])
    op, attrs = case["op"], case.get("attributes", {})
    if op in {"MatMul", "GemmD", "BatchMatMul"}:
        shape[-1] = items[1]["shape"][-1]
    elif op in {"Add", "Mul", "PRelu"}:
        shape = g.np.broadcast_shapes(shape, items[1]["shape"])
    elif op == "ReduceSum":
        for axis in attrs["axes"]:
            shape[axis] = 1
    elif op in {"Convolution", "ConvTranspose"}:
        weight = items[1]["shape"]
        shape[1] = weight[0] if op == "Convolution" else weight[1] * attrs["group"]
        for i in range(2):
            k, stride = weight[2+i], attrs["strides"][i]
            shape[2+i] = (shape[2+i] - k) // stride + 1 if op == "Convolution" else (shape[2+i] - 1) * stride + k
    return inputs + math.prod(shape) * g.np_dtype(g.input_dtype(case, items[0])).itemsize


def proposals(groups, known, max_bytes=256 * 1024 * 1024):
    pending, decisions = {}, []

    def search(group, member, grow):
        original = member["parameters"]
        p = copy.deepcopy(original)
        changes = []
        seen = {g.digest(p)}
        fallback_dim = None
        fallback_from = None
        for step in range(1, 25):
            if fallback_dim is None:
                p, changed = scaled_parameters(group, p, grow)
            else:
                old = p["a"][fallback_dim]
                p["a"][fallback_dim] = max(1, int(old * (2 if grow else 0.5)))
                changed = ["fixed:a." + str(fallback_dim)] if p["a"][fallback_dim] != old else []
            changes += changed
            if not changed or g.digest(p) in seen:
                return dict(status="minimum_dimension" if not grow else "cannot_scale", changed=changes)
            seen.add(g.digest(p))
            try:
                case = build_case(group["op"], p)
            except ValueError as exc:
                return dict(status="invalid_extension", error=str(exc), parameters=p)
            key = g.digest(g.model_config(case))[:12]
            if memory_bytes(case) > max_bytes:
                return dict(status="memory_limit", parameters=p, bytes=memory_bytes(case))
            result = known.get(key)
            wanted = "above_31" if grow else "at_or_below_31"
            info = dict(hash=key, parameters=p, config=g.operator_config(case), changed=changes, steps=step)
            if fallback_from:
                info["fallback_from_failed_hash"] = fallback_from
            if result is None:
                pending[key] = g.operator_config(case)
                return dict(status="pending", **info)
            if result["status"] == wanted:
                return dict(status="replacement_above_31" if grow else "smaller_control", **info)
            if result["status"] in {"compile_failed", "profile_failed"}:
                if fallback_dim is None and group["op"] in {"ReduceSum", "LayerNorm", "Softmax"}:
                    axis = original["axis"] % len(original["a"])
                    independent = list(range(axis)) if group["op"] == "LayerNorm" else [i for i in range(len(original["a"])) if i != axis]
                    if not grow:
                        independent = [i for i in independent if original["a"][i] > 1]
                    if independent:
                        # Keep a supported reduction length and vary independent rows instead.
                        fallback_dim = max(independent, key=lambda i: original["a"][i])
                        fallback_from = key
                        p = copy.deepcopy(original)
                        changes = []
                        continue
                return dict(status="extension_unmeasured", error=result.get("error"), **info)
        return dict(status="search_limit", parameters=p)

    for group in groups:
        statuses = [known.get(c["hash"]) for c in group["cases"]]
        if any(s is None for s in statuses):
            continue
        for member, status in zip(group["cases"], statuses):
            if status["status"] == "at_or_below_31":
                decisions.append(dict(group=group["id"], original_hash=member["hash"], direction="larger",
                                      result=search(group, member, True)))
            elif status["status"] in {"compile_failed", "profile_failed"}:
                p = member["parameters"]
                if group["op"] == "ReduceSum" and p["a"][p["axis"]] == 1:
                    decisions.append(dict(group=group["id"], original_hash=member["hash"], direction="identity_to_reduction",
                                          result=search(group, member, True)))
                else:
                    decisions.append(dict(group=group["id"], original_hash=member["hash"], direction="unresolved",
                                          result={"status": "unmeasured", "error": status.get("error")}))
        if all(s["status"] == "above_31" for s in statuses):
            # Keep the original eight/twelve cases; add controls below the smallest workload.
            member = min(group["cases"], key=lambda c: memory_bytes(build_case(group["op"], c["parameters"])))
            decisions.append(dict(group=group["id"], original_hash=member["hash"], direction="smaller",
                                  result=search(group, member, False)))
    return pending, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "results_sweep_adapt")
    args = parser.parse_args()
    known = load_known(args.runs)
    pending, decisions = proposals(plan(), known)
    args.out.mkdir(parents=True, exist_ok=True)
    g.save_json(args.out / "next.json", list(pending.values()))
    g.save_json(args.out / "decisions.json", decisions)
    print(json.dumps({"known": len(known), "pending": len(pending), "decisions": len(decisions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
