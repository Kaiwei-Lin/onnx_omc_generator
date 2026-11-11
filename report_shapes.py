"""Export shareable shape-sweep summaries, excluding raw profiles, logs and device identifiers."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import generate_ops as g
from adapt_shapes import load_known, proposals
from profile_ops import CORE_TYPES
from shape_plan import plan

HERE = Path(__file__).resolve().parent
TIMING = ("samples", "rounds", "min_us", "median_us", "mean_us", "max_us", "stddev_us", "all_samples_above_threshold")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: compact(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def public_error(error):
    # Share the cause without workstation paths or device command output.
    if not error:
        return None
    if "loading model" in error and "failed" in error:
        return "Device rejected the compiled model during loading"
    if "__BENCH_EXIT__" in error:
        return "Device execution failed; diagnostic log retained locally"
    if "Out of range float values" in error:
        return "Numerical validation encountered a nonfinite output/error"
    return re.sub(r"(?:[A-Z]:[\\/]|/mnt/[a-z]/)[^\s;]+", "<local-path>", str(error))[:1500]


def case_summary(key, row, roots, groups):
    candidates = [Path(row["measurement"]["record"])] if "measurement" in row else []
    candidates += [r / "models" / row["op"] / "logs" / key / "record.json" for r in reversed(roots)]
    record = next((json.loads(p.read_text(encoding="utf-8")) for p in candidates if p.exists()), {})
    compiled = record.get("compile", {})
    audit = compiled.get("actual_omc", {})
    cores = [n for n in audit.get("nodes", []) if n["type"] in CORE_TYPES[row["op"]]]
    m = row.get("measurement", {})
    numeric = m.get("numerical_check", {})
    info = record.get("onnx_info", {})
    inputs = row["config"]["params"]["inputs"]
    if m:
        assert m["numerical_check"]["passed"] and m["transdata_count"] == 0
        assert m["samples"] == len(m["values_us"][m["warmup"]:])
        assert (row["status"] == "above_31") == (m["min_us"] > m["threshold_us"])
    result = dict(hash=key, op=row["op"], groups=groups, origin="original" if groups else "added",
                  status=row["status"], config=row["config"],
                  inputs=[dict(name=t["name"], shape=t["shape"], constant="init" in t,
                               dtype=t.get("dtype", row["config"]["params"].get("dtype", "float16"))) for t in inputs],
                  outputs=info.get("outputs", []), actual_types=audit.get("types", {}),
                  cores=[{k: n[k] for k in ("name", "type", "inputs", "outputs")} for n in cores],
                  interfaces=audit.get("interfaces", []), transdata_count=audit.get("types", {}).get("TransData"),
                  core_count=len(cores), **{k: m[k] for k in TIMING if k in m},
                  per_core=[{k: c[k] for k in ("target", *TIMING) if k in c} for c in m.get("per_core", [])],
                  numerical_check=numeric, onnx_sha256=record.get("onnx_sha256"),
                  omc_sha256=compiled.get("omc_sha256"), compiler_fingerprint=compiled.get("fingerprint"),
                  compiler_platform=compiled.get("compiler", {}).get("config", {}).get("platform"),
                  ddk=Path(compiled.get("compiler", {}).get("ddk_root", "unknown")).name,
                  route=compiled.get("route", "ONNX -> OMG"))
    if not m:
        result["error"] = public_error(row.get("error"))
    return result


def report(roots, out):
    groups = plan()
    known = load_known(roots)
    if not known:
        raise ValueError("No status records in the requested --runs directories")
    pending, decisions = proposals(groups, known)
    membership = defaultdict(list)
    for group in groups:
        for c in group["cases"]:
            membership[c["hash"]].append(group["id"])
    rows = [case_summary(k, row, roots, membership.get(k, [])) for k, row in sorted(known.items())]
    by_key = {r["hash"]: r for r in rows}
    group_rows = []
    for group in groups:
        members = [by_key.get(c["hash"], {}) for c in group["cases"]]
        counts = Counter(m.get("status", "pending") for m in members)
        measured = [m for m in members if "min_us" in m]
        group_rows.append(dict(group=group["id"], op=group["op"], title=group["title"], total=len(members),
                               above_31=counts["above_31"], at_or_below_31=counts["at_or_below_31"],
                               unmeasured=sum(v for k, v in counts.items() if k not in {"above_31", "at_or_below_31"}),
                               min_us=min((m["min_us"] for m in measured), default=None),
                               max_us=max((m["max_us"] for m in measured), default=None)))
    adjustments, larger, smaller = [], {}, {}
    for d in decisions:
        original = by_key[d["original_hash"]]
        result = d["result"]; replacement = by_key.get(result.get("hash"), {})
        entry = dict(group=d["group"], direction=d["direction"], original_hash=d["original_hash"],
                     original_config=original["config"], original_status=original["status"],
                     original_min_us=original.get("min_us"), original_max_us=original.get("max_us"),
                     result={k: v for k, v in result.items() if k != "error"},
                     adjusted_min_us=replacement.get("min_us"), adjusted_max_us=replacement.get("max_us"),
                     adjusted_samples=replacement.get("samples"), adjusted_rounds=replacement.get("rounds"))
        if result.get("error"):
            entry["result"]["error"] = public_error(result["error"])
        adjustments.append(entry)
        if result["status"] == "replacement_above_31":
            larger[result["hash"]] = result["config"]
        if result["status"] == "smaller_control":
            smaller[result["hash"]] = result["config"]
    total_original = len(membership)
    original_rows = [r for r in rows if r["origin"] == "original"]
    operator_rows = []
    for op in dict.fromkeys(group["op"] for group in groups):
        subset = [r for r in original_rows if r["op"] == op]
        c = Counter(r["status"] for r in subset)
        operator_rows.append(dict(op=op, total=len(subset), above_31=c["above_31"],
                                  at_or_below_31=c["at_or_below_31"],
                                  unmeasured=len(subset)-c["above_31"]-c["at_or_below_31"],
                                  measured_types=sorted({n["type"] for r in subset if "min_us" in r for n in r["cores"]})))
    counts = dict(Counter(r["status"] for r in original_rows))
    added = dict(Counter(r["status"] for r in rows if r["origin"] == "added"))
    devices, settings = {}, {}
    for root in roots:
        for path in root.glob("device_*.json"):
            d = json.loads(path.read_text(encoding="utf-8"))
            public = {k: d.get(k) for k in ("chip", "product", "software")}
            devices[compact(public)] = public
            setting = {k: d.get(k) for k in ("times", "warmup", "dev_perf", "open_pmu")}
            settings[compact(setting)] = setting
    chip = ", ".join(sorted({d["chip"] for d in devices.values() if d["chip"]})) or "未记录"
    platform = ", ".join(sorted({r["compiler_platform"] for r in rows if r["compiler_platform"]}))
    ddks = ", ".join(sorted({r["ddk"] for r in rows}))
    confirmed = {r["hash"] for r in rows if r["per_core"] and min(c["samples"] for c in r["per_core"]) >= 60}
    recommended = set(larger) | set(smaller)
    metadata = dict(threshold_us=31, decision="every kept individual NPU core sample > 31; no kernel-time sums",
                    compiler_platform=platform, device_chip=chip, ddk=ddks,
                    devices=list(devices.values()), measurement_settings=list(settings.values()),
                    original_total=total_original, original_completed=len(original_rows), original_counts=counts,
                    added_counts=added, pending_extensions=len(pending),
                    confirmed_cases=len(confirmed), recommended_cases=len(recommended),
                    recommended_confirmed=len(recommended & confirmed),
                    recommended_needing_confirmation=sorted(recommended - confirmed),
                    generator_signatures=[dict(run=r.name, hashes=json.loads((r / "generator.json").read_text()))
                                          for r in roots if (r / "generator.json").exists()])
    out.mkdir(parents=True, exist_ok=True)
    g.save_json(out / "shape_sweep.json", dict(metadata=metadata, operators=operator_rows, groups=group_rows, cases=rows, adjustments=adjustments))
    write_csv(out / "shape_sweep_cases.csv", rows)
    write_csv(out / "shape_sweep_groups.csv", group_rows)
    write_csv(out / "shape_sweep_adjustments.csv", adjustments)
    g.save_json(HERE / "configs/shape_larger.json", list(larger.values()))
    g.save_json(HERE / "configs/shape_smaller_controls.json", list(smaller.values()))
    lines = ["# Shape 实测与 31 µs 调整", "",
             f"原表展开为 {len(groups)} 组、492 个组内条目，去重为 {total_original} 个配置（含两种 Add）。已完成 {len(original_rows)} 个："
             f"{counts.get('above_31', 0)} 个所有有效采样均 >31 µs，{counts.get('at_or_below_31', 0)} 个至少一次 ≤31 µs，"
             f"{len(original_rows)-counts.get('above_31', 0)-counts.get('at_or_below_31', 0)} 个未取得符合要求的计时。", "",
             f"新增并实测 {sum(added.values())} 个配置。当前映射给出 {len(larger)} 个去重后的放大配置、{len(smaller)} 个小 shape 对照；待测扩展 {len(pending)} 个。", "",
             f"{len(confirmed)} 个配置的每个目标内核已累计至少 60 个有效样本；推荐的 {len(recommended)} 个去重配置中，{len(recommended & confirmed)} 个完成此复测。", "",
             "## 测量口径", "",
             f"编译平台 {platform}，DDK {ddks}；USB 设备实际上报 {chip}。HIGH（dev_perf=3），PMU=1。"
             "初筛运行 25 次、每个核心去掉前 5 次预热；推荐配置复测通常再运行 45 次、去掉前 5 次，累计 60 个有效样本。汇总保留所有有效轮次，包括复测前 ≤31 µs 的样本。"
             "每个配置的 samples/rounds 见 CSV；20 次是筛选，复测配置通常为 60 次。整数微秒计时有量化和运行波动，不能据此保证其他设备或 DDK 上的耗时。", "",
             "阈值只比较 op.csv 中目标 npu_aicore 内核。分组卷积若拆成多个 Convolution，每个内核分别判断，"
             "case 通过需所有内核的所有有效样本 >31；samples 为所有内核样本数，不能理解为重复轮数，也不把内核耗时相加。"
             "所有有效结果均通过零 TransData 检查及数值验证：优先与 ONNX Runtime 比较（atol=0.003、rtol=0.03）；"
             "FP16 ReduceSum 若超出该门限，仅在与逐步 FP16 累加参考逐位一致时保留计时，并标记 reference=fp16_sequential_exact。"
             "onnxruntime_passed=false 及原误差仍保留；这种结果表示累加精度差异，不表示与 FP32 累加等精度。", "",
             "## 参数与布局", "",
             "- MatMul 保留 A 常量、B 动态，ONNX 数学形状为 A[M,K]×B[K,N]。兼容改写执行 Bᵀ×Aᵀ，再转置输出，"
             "因此实际内核的 M/N 方向与逻辑配置不同；两个 Permute 的耗时不计入 MatMul。",
             "- GemmD 是生成器的家族名；本 DDK 实际内核类型见 actual_types/per_core，通常为 GemmV3。",
             "- 卷积默认 stride=1、padding=0；分组卷积默认 H=W=32。反卷积沿用 stride=[1,2]，明确扫描 stride 的组除外；上采样组未指定 Cout，沿用 Cout=1。",
             "- Add 两种形式分别保留；Mul 的广播及动态/常量角色逐项保留。Activation 为 Sigmoid。默认 FP16，常量为固定种子合成值。",
             "- Mul 两路核心、LayerNorm 核心、ReduceSum 核心输入要求 ND。ReduceSum 和多维部分广播 Add 使用 AIR 修正原编译路径的 TransData/加载问题，逻辑 shape 不变。AIR 的公开 Data/NetOutput 可以标记 NCHW；"
             "汇总分别列出 interfaces 和 cores，ND 结论以实际核心端口为准。",
             "- 非四维 Add/Sigmoid/PRelu/Softmax 使用 GE AIR 尝试保留用户 rank；失败不自动改写为四维，也不当作 ≤31 µs。", "",
             "## 调整规则", "",
             "保留原表全部配置。对存在 ≤31 µs 样本的配置，沿扫描维度逐次放大，复用已经测得的较大配置；"
             "对原始配置全部通过的组，从最小数据量案例逐次缩小到 ≤31 µs 的对照。若原扫描维度已为 1，"
             "映射以 fixed: 或 fixed-dimension 明确标记改变其他维度。放大/缩小后不再承诺原组的固定总元素数或固定 FLOPs。"
             "输入、权重和输出总量设置 256 MiB 上限；编译失败、被优化掉或布局不合要求的条目单独列出，不能用模型总时延替代。", "",
             "若原配置可正确执行，但放大或缩小归约/归一化长度后失效，搜索会保留原有效长度，改为调整独立行数；"
             "映射以 fixed:a.* 和 fallback_from_failed_hash 标记，不能再称为固定行数的长度扫描。", "",
             "## 各算子结果（原表去重）", "", "| 算子 | 配置数 | >31 | 存在 ≤31 | 未测成 | 实测核心类型 |", "|---|---:|---:|---:|---:|---|" ]
    for row in operator_rows:
        lines.append(f"| {row['op']} | {row['total']} | {row['above_31']} | {row['at_or_below_31']} | {row['unmeasured']} | {', '.join(row['measured_types']) or '—'} |")
    lines += ["", "当前 DDK 的限制：ConvTranspose 仍不满足一维 ND bias 要求；PRelu 的标量和非通道 slope 广播有编译失败；"
              "LayerNorm 部分长归一化和特殊 rank 存在数值或布局问题。失败配置的 shape 与原因逐项保留，"
              "原表没有 CastT，本轮未新增 CastT 的独立 NPU 计时。", "",
              "## 各组结果", "", "| 组 | >31 | 存在 ≤31 | 未测成 | 单内核 min–max（µs） |", "|---|---:|---:|---:|---|"]
    for row in group_rows:
        span = "—" if row["min_us"] is None else f"{row['min_us']:g}–{row['max_us']:g}"
        lines.append(f"| {row['group']} | {row['above_31']} | {row['at_or_below_31']} | {row['unmeasured']} | {span} |")
    unresolved = Counter((d["direction"], d["result"]["status"]) for d in decisions)
    lines += ["", "## 文件与复现", "",
              "- [逐配置汇总](verification/shape_sweep_cases.csv)：逻辑输入/权重/输出 shape、dtype、实际核心和接口布局、样本数、时间范围、错误。",
              "- [逐组汇总](verification/shape_sweep_groups.csv) 与 [完整 JSON 汇总](verification/shape_sweep.json)。",
              "- [AIR 一致性检查](verification/shape_air_compatibility.json)：最终生成器对已编译 AIR 案例的字节一致性核验；实际编译路径以各 case 的 route 字段为准。",
              "- [原配置→调整配置映射](verification/shape_sweep_adjustments.csv)：每项原参数、放大/缩小参数、实际计时和未解决原因。",
              "- 可直接运行的参数文件：[原表](configs/shape_cases.json)、[放大配置](configs/shape_larger.json)、[小 shape 对照](configs/shape_smaller_controls.json)。",
              "- 原始 profiling、op.csv、设备日志、ONNX/OMC 留在本机 results_sweep*，不提交本次原始结果包。", "",
              "```powershell", "python shape_plan.py",
              "python sweep_shapes.py --ddk-root <DDK目录> --hdc <hdc.exe> --target <设备ID> --allow-multiple-core",
              "python adapt_shapes.py --runs results_sweep results_sweep_ext",
              "python sweep_shapes.py --config results_sweep_adapt/next.json --out results_sweep_ext --ddk-root <DDK目录> --hdc <hdc.exe> --target <设备ID> --allow-multiple-core",
              "# 重复 adapt / sweep，直到 next.json 为空；设备 profiling 必须串行。",
              "python report_shapes.py --runs results_sweep results_sweep_ext", "```", "",
              "复测使用 --profile-only --repeat-measured --times 45 --warmup 5 --only <配置hash...>，并指定原模型所在 --out；"
              "旧轮次保存于 previous_measurements，并由 adapt/report 合并判断。"
              "更换生成器、DDK 或布局模式时使用新输出目录。", "",
              "调整状态（按组内映射计数，包含重复引用）：`" + compact({f"{a}/{b}": n for (a,b),n in unresolved.items()}) + "`。", ""]
    (HERE / "SHAPE_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")
    print(compact(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "verification")
    args = parser.parse_args()
    report(args.runs, args.out)
