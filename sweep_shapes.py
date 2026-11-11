"""Compile a shape matrix with two workers; profile the USB device serially."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import generate_ops as g
from profile_ops import Device, profile

HERE = Path(__file__).resolve().parent


def linux_path(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + "/" + p.as_posix()[3:]


def compile_case(case, out, args):
    key = g.digest(g.model_config(case))[:12]
    cfg = out / "jobs" / (key + ".json")
    g.save_json(cfg, g.operator_config(case))
    record = out / "models" / case["op"] / "logs" / key / "record.json"
    if args.profile_only:
        if not record.is_file():
            return key, record, "No existing model record for --profile-only"
        compiled = json.loads(record.read_text(encoding="utf-8")).get("compile", {})
        return key, record, None if compiled.get("status") == "COMPILED" else compiled.get("error", "Model is not compiled")
    command = ["wsl.exe", "-d", args.distro, "--cd", linux_path(HERE), args.python,
               "generate_ops.py", "--config", linux_path(cfg), "--compiler-config", linux_path(out / "compiler.json"),
               "--ddk-root", linux_path(args.ddk_root), "--out", linux_path(out / "models")]
    log = out / "compile" / (key + ".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n"); stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=360)
    if result.returncode:
        detail = json.loads(record.read_text()).get("compile", {}).get("error") if record.exists() else None
        return key, record, detail or log.read_text(encoding="utf-8", errors="replace")[-2000:]
    return key, record, None


def summary(out, cases):
    rows = []
    for case in cases:
        key = g.digest(g.model_config(case))[:12]
        p = out / "status" / (key + ".json")
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    g.save_json(out / "summary.json", {"total": len(cases), "finished": len(rows), "counts": counts, "cases": rows})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/shape_cases.json")
    parser.add_argument("--out", type=Path, default=HERE / "results_sweep")
    parser.add_argument("--ddk-root", type=Path, required=True)
    parser.add_argument("--hdc", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--python", default="/home/lkw/miniconda3/envs/cannkit_torch_alg/bin/python")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--times", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tools", default="/data/local/tmp/tools_sysdbg")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--repeat-measured", action="store_true", help="Repeat selected successful cases and retain every earlier round")
    parser.add_argument("--profile-only", action="store_true", help="Measure existing verified models without recompiling or changing generator provenance")
    parser.add_argument("--allow-multiple-core", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.times:
        parser.error("Need 0 <= warmup < times")
    cases = g.load_config(args.config)[0]
    cases = list({g.digest(g.model_config(c))[:12]: c for c in cases}.values())
    if args.only:
        cases = [c for c in cases if c["op"] in args.only or g.digest(g.model_config(c))[:12] in args.only]
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "status", "profiles"):
        (out / name).mkdir(exist_ok=True)
    fingerprint = {name: g.sha256(HERE / name) for name in ("generate_ops.py", "ge_nd.py", "compiler.json")}
    signature = out / "generator.json"
    if args.profile_only and not signature.exists():
        raise ValueError("--profile-only requires an existing generator.json")
    if signature.exists() and json.loads(signature.read_text()) != fingerprint and not args.profile_only:
        raise ValueError("Generator changed; use a fresh --out to keep compiled variants separate")
    if not args.profile_only:
        g.save_json(signature, fingerprint)
        shutil.copy2(HERE / "compiler.json", out / "compiler.json")
    pending = []
    for c in cases:
        p = out / "status" / (g.digest(g.model_config(c))[:12] + ".json")
        old_status = json.loads(p.read_text())["status"] if p.exists() else None
        if not p.exists() or (args.retry_failed and old_status in {"compile_failed", "profile_failed"}) or (
                args.repeat_measured and old_status in {"above_31", "at_or_below_31"}):
            pending.append(c)
    device = Device(args.hdc, args.target, args.tools)
    if device.shell("ps -A | grep -E 'model_run_tool_internal|data_proc_tool_internal' || true"):
        raise RuntimeError("Another device profiler is active")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote = "/data/local/tmp/shape_sweep_" + stamp
    metadata = dict(target=args.target, chip=device.shell("param get ohos.boot.chiptype"),
                    product=device.shell("param get const.product.model"),
                    software=device.shell("param get const.product.software.version"),
                    host_time=stamp, times=args.times, warmup=args.warmup, dev_perf=3, open_pmu=1,
                    tool_version=device.shell(shlex.join(["env", "LD_LIBRARY_PATH=" + args.tools,
                                                         args.tools + "/model_run_tool_internal", "--version"])))
    g.save_json(out / ("device_" + stamp + ".json"), metadata)
    print(f"START total={len(cases)} pending={len(pending)} device={metadata['chip']}", flush=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(compile_case, c, out, args): c for c in pending}
    processed = 0
    try:
        for future in as_completed(futures):
            case = futures[future]; key = g.digest(g.model_config(case))[:12]
            row = dict(hash=key, op=case["op"], config=g.operator_config(case))
            previous = out / "status" / (key + ".json")
            if previous.exists():
                old = json.loads(previous.read_text(encoding="utf-8"))
                row["previous_measurements"] = old.get("previous_measurements", [])
                if "measurement" in old:
                    row["previous_measurements"].append(old["measurement"])
            try:
                _, record, error = future.result()
            except (OSError, subprocess.TimeoutExpired) as exc:
                record, error = None, str(exc)
            if error:
                row.update(status="compile_failed", error=error)
            else:
                row["record"] = str(record)
                run = out / "profiles" / key / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                g.save_json(run / "device.json", metadata)
                try:
                    measured = profile(device, record, run, remote, args.times, args.warmup, 3, 31,
                                       allow_multiple_core=args.allow_multiple_core)
                    all_passed = measured["all_samples_above_threshold"] and all(
                        m["all_samples_above_threshold"] for m in row.get("previous_measurements", []))
                    row.update(status="above_31" if all_passed else "at_or_below_31",
                               measurement=measured)
                    # Remove only this task's archived model/input/output copies, never the global profiling spool.
                    identifier = case["op"] + "_" + key
                    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
                        raise ValueError("Unsafe generated remote identifier")
                    try:
                        device.shell("rm -rf -- " + shlex.quote(remote + "/" + identifier))
                    except RuntimeError as exc:
                        row["cleanup_warning"] = str(exc)
                except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    # A lost USB connection stops the sweep instead of misclassifying every remaining model.
                    device.shell("echo device_ready")
                    row.update(status="profile_failed", error=str(exc), attempt=str(run))
            g.save_json(out / "status" / (key + ".json"), row)
            processed += 1
            counts = summary(out, cases)
            print(json.dumps(dict(done=sum(counts.values()), total=len(cases), op=case["op"], hash=key,
                                  status=row["status"], counts=counts,
                                  processed_this_run=processed, pending_this_run=len(pending))), flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        summary(out, cases)


if __name__ == "__main__":
    main()
