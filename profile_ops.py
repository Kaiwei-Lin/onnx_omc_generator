#!/usr/bin/env python3
"""Profile compiled generator outputs via HDC; count only the selected OMC kernel."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess

import numpy as np
import onnxruntime as ort

from generate_ops import save_json, sha256

CORE_TYPES = {
    "MatMul": {"MatMulV2", "MatMul"}, "GemmD": {"GemmD", "GemmV3"},
    "BatchMatMul": {"BatchMatMulV2", "BatchMatMul"}, "Convolution": {"Convolution"},
    "ConvTranspose": {"ConvTranspose"}, "Add": {"Eltwise", "Add"},
    "Mul": {"Eltwise", "Mul"}, "Activation": {"Activation"}, "PRelu": {"PRelu"},
    "ReduceSum": {"ReduceSum"}, "LayerNorm": {"LayerNorm"}, "Softmax": {"Softmax"},
    "CastT": {"CastT"},
}
DTYPES = {"DT_FLOAT16": np.dtype("float16"), "DT_FLOAT": np.dtype("float32")}
TOOL_TYPES = {"DT_FLOAT16": "FP16", "DT_FLOAT": "FP32"}


def physical_shape(tensor):
    shape = tensor["shape"]
    layout = tensor["layout"]
    if layout == "NC1HWC0":
        if len(shape) != 4 or tensor["dtype"] not in DTYPES:
            raise ValueError("Only floating point rank-4 NC1HWC0 interfaces are supported")
        n, c, h, w = shape
        return n, (c + 15) // 16, h, w, 16
    if layout not in {"NCHW", "ND"}:
        raise ValueError("Unverified interface layout: " + layout)
    return tuple(shape)


def pack(array, tensor):
    physical_shape(tensor)
    if list(array.shape) != tensor["shape"] or array.dtype != DTYPES[tensor["dtype"]]:
        raise ValueError("Input shape/dtype differs from OMC interface")
    if tensor["layout"] == "NC1HWC0":
        n, c, h, w = array.shape
        padded = np.zeros((n, (c + 15) // 16 * 16, h, w), dtype=array.dtype)
        padded[:, :c] = array
        array = padded.reshape(n, -1, 16, h, w).transpose(0, 1, 3, 4, 2)
    payload = array.tobytes()
    # ROMC standard interfaces allocate logical bytes; blocked interfaces retain descriptor padding.
    size = tensor["buffer_bytes"] if tensor["layout"] == "NC1HWC0" else len(payload)
    if tensor["buffer_bytes"] < len(payload):
        raise ValueError("OMC descriptor is smaller than the input")
    if size < len(payload):
        raise ValueError("OMC buffer is smaller than packed input")
    return payload + bytes(size - len(payload))


def unpack(payload, tensor):
    shape = physical_shape(tensor)
    dtype = DTYPES[tensor["dtype"]]
    count = math.prod(shape)
    if len(payload) < count * dtype.itemsize:
        raise ValueError("Output file is smaller than its physical tensor")
    value = np.frombuffer(payload, dtype=dtype, count=count).reshape(shape)
    if tensor["layout"] == "NC1HWC0":
        n, c, h, w = tensor["shape"]
        value = value.transpose(0, 1, 4, 2, 3).reshape(n, -1, h, w)[:, :c]
    return value


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{str(k).strip().lower(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(stream, skipinitialspace=True)]


def durations(rows, field, value, times):
    found = [r for r in rows if r.get(field) == value]
    values = [float(r["total time(us)"]) for r in found]
    if len(values) != times or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f"Expected {times} positive samples of {value}, got {values}")
    return found, values


def stats(values):
    return dict(samples=len(values), mean_us=statistics.fmean(values), median_us=statistics.median(values),
                min_us=min(values), max_us=max(values), stddev_us=statistics.pstdev(values))


class Device:
    def __init__(self, hdc, target, tools):
        self.hdc, self.target, self.tools = str(hdc), target, tools

    def call(self, *args, timeout=180):
        p = subprocess.run([self.hdc, "-t", self.target, *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        return p.stdout + p.stderr

    def shell(self, command, timeout=180):
        marker = "__BENCH_EXIT__="
        result = self.call("shell", command + "; code=$?; echo " + marker + "$code", timeout=timeout)
        matches = re.findall(re.escape(marker) + r"(\d+)", result)
        if not matches or matches[-1] != "0":
            raise RuntimeError(result)
        return result.rsplit(marker, 1)[0].strip()

    def send(self, local, remote):
        self.call("file", "send", str(local), remote)
        if self.shell("sha256sum " + shlex.quote(remote)).split()[0] != sha256(local):
            raise RuntimeError("Push hash mismatch: " + str(local))

    def receive(self, remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        self.call("file", "recv", remote, str(local))
        if not local.is_file() or sha256(local) != self.shell("sha256sum " + shlex.quote(remote)).split()[0]:
            raise RuntimeError("Pull hash mismatch: " + str(local))


def profile(device, record_path, out, remote_root, times, warmup, perf, threshold, allow_transdata=False):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    compiled = record["compile"]
    if compiled.get("status") != "COMPILED":
        raise ValueError("Model compilation did not pass: " + str(record_path))
    directory = record_path.parents[2]
    omc, onnx = directory / compiled["omc"], directory / record["onnx"]
    if sha256(omc) != compiled["omc_sha256"] or sha256(onnx) != record["onnx_sha256"]:
        raise ValueError("Unverified or changed model: " + str(record_path))
    audit = compiled["actual_omc"]
    if audit["types"].get("TransData", 0) and not allow_transdata:
        raise ValueError("TransData is present")
    core = [n for n in audit["nodes"] if n["type"] in CORE_TYPES[record["op"]]]
    if len(core) != 1:
        raise ValueError("Expected exactly one target compute node")
    identifier = record["op"] + "_" + record["config_sha256"][:12]
    dest = out / identifier
    if dest.exists():
        raise FileExistsError("Use a fresh --out for each run: " + str(dest))
    dest.mkdir(parents=True)
    remote = remote_root + "/" + identifier
    device.shell("mkdir -p " + shlex.quote(remote + "/run") + " " + shlex.quote(remote + "/csv"))
    remote_model = remote + "/" + identifier + ".omc"
    device.send(omc, remote_model)
    inputs = {x["name"]: x["tensors"][0] for x in audit["interfaces"] if x["kind"] == "Data"}
    outputs = [t for x in audit["interfaces"] if x["kind"] == "NetOutput" for t in x["tensors"]]
    ordered_inputs = [inputs[x["name"]] for x in record["onnx_info"]["inputs"]]
    if len(outputs) != 1:
        raise ValueError("Expected one model output")
    rng = np.random.default_rng(20261111)
    feed, remote_inputs = {}, []
    for index, info in enumerate(record["onnx_info"]["inputs"]):
        tensor = inputs[info["name"]]
        value = rng.normal(0, 0.1, info["shape"]).astype(DTYPES[tensor["dtype"]])
        feed[info["name"]] = value
        local = dest / f"input_{index}.bin"
        local.write_bytes(pack(value, tensor))
        remote_input = remote + "/" + local.name
        device.send(local, remote_input)
        remote_inputs.append(remote_input)
    command = ["timeout", "150", "env", "LD_LIBRARY_PATH=" + device.tools,
               device.tools + "/model_run_tool_internal", "--model=" + remote_model,
               "--input=" + ",".join(remote_inputs), "--output_dir=" + remote + "/run",
               "--modelmanager_itf=ROMC", "--enable_item=1", "--open_pmu=1", f"--times={times}",
               f"--dev_perf={perf}", "--input_type=" + ",".join(TOOL_TYPES[t["dtype"]] for t in ordered_inputs),
               "--output_type=" + ",".join(TOOL_TYPES[t["dtype"]] for t in outputs)]
    if any(t["layout"] == "NC1HWC0" for t in [*ordered_inputs, *outputs]):
        command += ["--input_format=" + ",".join(t["layout"] for t in ordered_inputs),
                    "--output_format=" + ",".join(t["layout"] for t in outputs)]
    save_json(dest / "invocation.json", dict(command=command, record=str(record_path), omc_sha256=sha256(omc),
              onnx_sha256=sha256(onnx), times=times, warmup=warmup, perf=perf, core=core[0],
              interfaces=audit["interfaces"], compiler_platform=compiled["compiler"]["config"].get("platform", "kirin9030"),
              input="seeded finite normal(0,0.1); NC1HWC0 packed with descriptor padding; NCHW/ND logical bytes"))
    thermal = "date; for z in /sys/class/thermal/thermal_zone*; do printf '%s ' \"$z\"; cat \"$z/type\" \"$z/temp\" 2>/dev/null; done; true"
    (dest / "thermal_before.txt").write_text(device.shell(thermal), encoding="utf-8")
    try:
        log = device.shell(shlex.join(command))
    except RuntimeError as error:
        (dest / "run.log").write_text(str(error), encoding="utf-8")
        raise
    (dest / "run.log").write_text(log, encoding="utf-8")
    if f"Load model:{identifier} success" not in log or "Execute result :Success" not in log:
        raise RuntimeError("No confirmed model load/inference success; see " + str(dest / "run.log"))
    # The exporter can consume the shared spool. Preserve it before exporting, never delete it.
    device.shell("tar -czf " + shlex.quote(remote + "/profiling_raw.tar.gz") + " -C /data/local/tmp profiling")
    device.receive(remote + "/profiling_raw.tar.gz", dest / "profiling_raw.tar.gz")
    export = device.shell(shlex.join(["env", "LD_LIBRARY_PATH=" + device.tools,
                                     device.tools + "/data_proc_tool_internal", "--result_path=" + remote + "/csv"]))
    (dest / "export.log").write_text(export, encoding="utf-8")
    for name in device.shell("ls " + shlex.quote(remote + "/csv")).splitlines():
        if name.startswith(identifier + "_") and name.endswith(".csv") and "/" not in name:
            device.receive(remote + "/csv/" + name, dest / "csv" / name)
    op_files = list((dest / "csv").glob("*_op.csv"))
    model_files = list((dest / "csv").glob("*_model.csv"))
    run_files = device.shell("ls -l " + shlex.quote(remote + "/run"))
    (dest / "output_files.txt").write_text(run_files, encoding="utf-8")
    binaries = [n for n in device.shell("ls " + shlex.quote(remote + "/run")).splitlines()
                if re.fullmatch(r"output_\d+(?:\.bin)?", n)]
    if len(binaries) != 1:
        raise ValueError("Expected exactly one output file; inspect output_files.txt")
    output_file = dest / "output.bin"
    device.receive(remote + "/run/" + binaries[0], output_file)
    actual = unpack(output_file.read_bytes(), outputs[0]).astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
    ref_types = {"FLOAT": np.float32, "FLOAT16": np.float16}
    reference_feed = {t["name"]: feed[t["name"]].astype(ref_types[t["dtype"]]) for t in record["onnx_info"]["inputs"]}
    expected = session.run(None, reference_feed)[0].astype(np.float32)
    numeric = dict(passed=bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=0.03, atol=0.003)),
                   atol=0.003, rtol=0.03, max_abs_error=float(np.max(np.abs(actual - expected))),
                   rmse=float(np.sqrt(np.mean((actual - expected) ** 2))))
    save_json(dest / "numerical_check.json", numeric)
    if not numeric["passed"]:
        raise ValueError("Device output differs from ONNX; inspect numerical_check.json")
    if len(op_files) != 1 or len(model_files) != 1 or not list((dest / "csv").glob("*_pmu.csv")):
        raise ValueError("No unambiguous NPU op/PMU CSV; model latency is not a target-kernel measurement")
    op_rows = csv_rows(op_files[0])
    name = core[0]["name"] + "(" + core[0]["type"] + ")"
    target_rows, values = durations(op_rows, "op name", name, times)
    if any(r["device type"] != "npu_aicore" for r in target_rows):
        raise ValueError("Target is not executing on npu_aicore")
    _, model_values = durations([r for r in csv_rows(model_files[0]) if r.get("item event") == "inference end"],
                                "model name", identifier, times)
    result = dict(op=record["op"], id=identifier, config=record["case"], target=name,
                  target_device_types=sorted({r["device type"] for r in target_rows}),
                  **stats(values[warmup:]), threshold_us=threshold,
                  all_samples_above_threshold=min(values[warmup:]) > threshold,
                  model=stats(model_values[warmup:]), values_us=values, warmup=warmup,
                  all_runtime_ops=sorted({r["op name"] for r in op_rows}),
                  transdata_count=audit["types"].get("TransData", 0), numerical_check=numeric,
                  record=str(record_path), onnx=str(onnx), omc=str(omc),
                  op_csv=str(op_files[0]), profile=str(dest))
    save_json(dest / "statistics.json", result)
    print(json.dumps({k: result[k] for k in ("op", "id", "target", "mean_us", "min_us", "max_us", "all_samples_above_threshold")}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hdc", default=shutil.which("hdc"))
    p.add_argument("--target", required=True)
    p.add_argument("--tools", default="/data/local/tmp/tools_sysdbg")
    p.add_argument("--only", nargs="+")
    p.add_argument("--times", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--perf", type=int, choices=(1, 2, 3, 4, 103), default=3)
    p.add_argument("--threshold", type=float, default=31)
    p.add_argument("--allow-transdata", action="store_true", help="Diagnostic comparison only; target timing still excludes TransData")
    a = p.parse_args()
    if not a.hdc or not 0 <= a.warmup < a.times or not math.isfinite(a.threshold) or a.threshold < 0:
        p.error("Need HDC, 0 <= warmup < times, and finite nonnegative threshold")
    records = sorted(a.models.resolve().glob("*/logs/*/record.json"))
    records = [r for r in records if not a.only or r.parents[2].name in a.only or r.parent.name in a.only]
    if not records:
        p.error("No matching records")
    device = Device(a.hdc, a.target, a.tools)
    active = device.shell("ps -A | grep -E 'model_run_tool_internal|data_proc_tool_internal' || true")
    if active:
        raise RuntimeError("Another profiling task is active: " + active)
    a.out = a.out.resolve()
    if (a.out / "device.json").exists():
        raise FileExistsError("Use a fresh output directory for each run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote = "/data/local/tmp/onnx_31us_" + timestamp
    save_json(a.out / "device.json", dict(target=a.target, chip=device.shell("param get ohos.boot.chiptype"),
              product=device.shell("param get const.product.model"),
              software=device.shell("param get const.product.software.version"),
              tool_version=device.shell(shlex.join(["env", "LD_LIBRARY_PATH=" + device.tools,
                                                    device.tools + "/model_run_tool_internal", "--version"])),
              host_time=timestamp, remote=remote, times=a.times, warmup=a.warmup, dev_perf=a.perf, open_pmu=1))
    results, failures = [], []
    for record in records:
        try:
            results.append(profile(device, record, a.out, remote, a.times, a.warmup, a.perf, a.threshold, a.allow_transdata))
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            failures.append(dict(record=str(record), error=str(error)))
            print("PROFILE FAILED " + str(record) + " " + str(error), flush=True)
        save_json(a.out / "summary.json", dict(results=results, failures=failures))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
