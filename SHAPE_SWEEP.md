# Shape 实测与 31 µs 调整

原表展开为 59 组、492 个组内条目，去重为 462 个配置（含两种 Add）。已完成 462 个：249 个所有有效采样均 >31 µs，154 个至少一次 ≤31 µs，59 个未取得符合要求的计时。

新增并实测 139 个配置。当前映射给出 61 个去重后的放大配置、19 个小 shape 对照；待测扩展 0 个。

122 个配置的每个目标内核已累计至少 60 个有效样本；推荐的 80 个去重配置中，80 个完成此复测。

## 测量口径

编译平台 kirin9030，DDK DDK-tools-next-6.1.1.0；USB 设备实际上报 Kirin9020。HIGH（dev_perf=3），PMU=1。初筛运行 25 次、每个核心去掉前 5 次预热；推荐配置复测通常再运行 45 次、去掉前 5 次，累计 60 个有效样本。汇总保留所有有效轮次，包括复测前 ≤31 µs 的样本。每个配置的 samples/rounds 见 CSV；20 次是筛选，复测配置通常为 60 次。整数微秒计时有量化和运行波动，不能据此保证其他设备或 DDK 上的耗时。

阈值只比较 op.csv 中目标 npu_aicore 内核。分组卷积若拆成多个 Convolution，每个内核分别判断，case 通过需所有内核的所有有效样本 >31；samples 为所有内核样本数，不能理解为重复轮数，也不把内核耗时相加。所有有效结果均通过零 TransData 检查及数值验证：优先与 ONNX Runtime 比较（atol=0.003、rtol=0.03）；FP16 ReduceSum 若超出该门限，仅在与逐步 FP16 累加参考逐位一致时保留计时，并标记 reference=fp16_sequential_exact。onnxruntime_passed=false 及原误差仍保留；这种结果表示累加精度差异，不表示与 FP32 累加等精度。

## 参数与布局

- MatMul 保留 A 常量、B 动态，ONNX 数学形状为 A[M,K]×B[K,N]。兼容改写执行 Bᵀ×Aᵀ，再转置输出，因此实际内核的 M/N 方向与逻辑配置不同；两个 Permute 的耗时不计入 MatMul。
- GemmD 是生成器的家族名；本 DDK 实际内核类型见 actual_types/per_core，通常为 GemmV3。
- 卷积默认 stride=1、padding=0；分组卷积默认 H=W=32。反卷积沿用 stride=[1,2]，明确扫描 stride 的组除外；上采样组未指定 Cout，沿用 Cout=1。
- Add 两种形式分别保留；Mul 的广播及动态/常量角色逐项保留。Activation 为 Sigmoid。默认 FP16，常量为固定种子合成值。
- Mul 两路核心、LayerNorm 核心、ReduceSum 核心输入要求 ND。ReduceSum 和多维部分广播 Add 使用 AIR 修正原编译路径的 TransData/加载问题，逻辑 shape 不变。AIR 的公开 Data/NetOutput 可以标记 NCHW；汇总分别列出 interfaces 和 cores，ND 结论以实际核心端口为准。
- 非四维 Add/Sigmoid/PRelu/Softmax 使用 GE AIR 尝试保留用户 rank；失败不自动改写为四维，也不当作 ≤31 µs。

## 调整规则

保留原表全部配置。对存在 ≤31 µs 样本的配置，沿扫描维度逐次放大，复用已经测得的较大配置；对原始配置全部通过的组，从最小数据量案例逐次缩小到 ≤31 µs 的对照。若原扫描维度已为 1，映射以 fixed: 或 fixed-dimension 明确标记改变其他维度。放大/缩小后不再承诺原组的固定总元素数或固定 FLOPs。输入、权重和输出总量设置 256 MiB 上限；编译失败、被优化掉或布局不合要求的条目单独列出，不能用模型总时延替代。

若原配置可正确执行，但放大或缩小归约/归一化长度后失效，搜索会保留原有效长度，改为调整独立行数；映射以 fixed:a.* 和 fallback_from_failed_hash 标记，不能再称为固定行数的长度扫描。

## 各算子结果（原表去重）

| 算子 | 配置数 | >31 | 存在 ≤31 | 未测成 | 实测核心类型 |
|---|---:|---:|---:|---:|---|
| MatMul | 64 | 50 | 14 | 0 | MatMulV2 |
| GemmD | 30 | 11 | 19 | 0 | GemmV3 |
| BatchMatMul | 35 | 28 | 7 | 0 | BatchMatMulV2 |
| Convolution | 45 | 27 | 18 | 0 | Convolution |
| ConvTranspose | 38 | 0 | 0 | 38 | — |
| Add | 48 | 24 | 24 | 0 | Add, Eltwise |
| Mul | 24 | 12 | 12 | 0 | Mul |
| Activation | 35 | 13 | 22 | 0 | Activation |
| PRelu | 23 | 4 | 4 | 15 | PRelu |
| ReduceSum | 38 | 14 | 24 | 0 | ReduceSum |
| LayerNorm | 40 | 28 | 6 | 6 | LayerNorm |
| Softmax | 42 | 38 | 4 | 0 | Softmax |

当前 DDK 的限制：ConvTranspose 仍不满足一维 ND bias 要求；PRelu 的标量和非通道 slope 广播有编译失败；LayerNorm 部分长归一化和特殊 rank 存在数值或布局问题。失败配置的 shape 与原因逐项保留，原表没有 CastT，本轮未新增 CastT 的独立 NPU 计时。

## 各组结果

| 组 | >31 | 存在 ≤31 | 未测成 | 单内核 min–max（µs） |
|---|---:|---:|---:|---|
| MatMul_small_m | 8 | 0 | 0 | 258–704 |
| MatMul_small_n | 8 | 0 | 0 | 216–394 |
| MatMul_k | 6 | 2 | 0 | 23–341 |
| MatMul_square | 4 | 4 | 0 | 18–170 |
| MatMul_small_k | 8 | 0 | 0 | 61–172 |
| MatMul_long_reduce | 0 | 8 | 0 | 16–39 |
| MatMul_same_flops | 8 | 0 | 0 | 54–12821 |
| MatMul_alignment | 12 | 0 | 0 | 37–115 |
| GemmD_small_m | 8 | 0 | 0 | 33–88 |
| GemmD_n | 1 | 7 | 0 | 20–59 |
| GemmD_k | 4 | 4 | 0 | 24–81 |
| GemmD_alignment | 0 | 12 | 0 | 19–59 |
| BatchMatMul_batch | 6 | 2 | 0 | 23–705 |
| BatchMatMul_m | 8 | 0 | 0 | 45–77 |
| BatchMatMul_n | 6 | 2 | 0 | 27–173 |
| BatchMatMul_k | 8 | 0 | 0 | 34–166 |
| BatchMatMul_same_flops | 5 | 3 | 0 | 25–580 |
| Convolution_width | 3 | 5 | 0 | 12–81 |
| Convolution_cin | 5 | 3 | 0 | 19–207 |
| Convolution_cout | 5 | 3 | 0 | 26–125 |
| Convolution_kernel | 5 | 3 | 0 | 20–195 |
| Convolution_group | 4 | 4 | 0 | 11–90 |
| Convolution_aspect | 8 | 0 | 0 | 270–600 |
| ConvTranspose_spatial | 0 | 0 | 8 | — |
| ConvTranspose_cout | 0 | 0 | 8 | — |
| ConvTranspose_stride | 0 | 0 | 8 | — |
| ConvTranspose_kernel | 0 | 0 | 8 | — |
| ConvTranspose_group | 0 | 0 | 8 | — |
| Add_small | 0 | 8 | 0 | 4–6 |
| Add_small_constant | 0 | 8 | 0 | 4–6 |
| Add_large | 4 | 4 | 0 | 5–952 |
| Add_large_constant | 4 | 4 | 0 | 5–937 |
| Mul_small | 0 | 8 | 0 | 4–6 |
| Mul_large | 4 | 4 | 0 | 5–954 |
| Activation_small | 0 | 8 | 0 | 4–7 |
| Activation_large | 3 | 5 | 0 | 5–733 |
| Add_broadcast | 8 | 0 | 0 | 44–142 |
| Add_broadcast_constant | 8 | 0 | 0 | 44–171 |
| Mul_broadcast | 8 | 0 | 0 | 44–4001 |
| Activation_shape | 8 | 0 | 0 | 49–132 |
| Activation_alignment | 3 | 9 | 0 | 10–132 |
| PRelu_scalar | 0 | 0 | 8 | — |
| PRelu_channel | 4 | 4 | 0 | 14–148 |
| PRelu_broadcast | 1 | 0 | 7 | 40–43 |
| ReduceSum_short | 1 | 7 | 0 | 29–232 |
| ReduceSum_long | 1 | 7 | 0 | 6–77 |
| ReduceSum_fixed_elements | 6 | 2 | 0 | 25–201 |
| ReduceSum_axis0 | 6 | 2 | 0 | 21–1302 |
| ReduceSum_direction | 0 | 8 | 0 | 13–41 |
| LayerNorm_short | 8 | 0 | 0 | 207–282 |
| LayerNorm_long | 0 | 5 | 3 | 7–12 |
| LayerNorm_fixed_elements | 6 | 1 | 1 | 21–875 |
| LayerNorm_rank | 5 | 0 | 3 | 56–99 |
| LayerNorm_alignment | 12 | 0 | 0 | 55–85 |
| Softmax_short | 8 | 0 | 0 | 61–329 |
| Softmax_long | 4 | 4 | 0 | 5–225 |
| Softmax_fixed_elements | 8 | 0 | 0 | 82–449 |
| Softmax_four_d | 8 | 0 | 0 | 44–233 |
| Softmax_alignment | 12 | 0 | 0 | 36–184 |

## 文件与复现

- [逐配置汇总](verification/shape_sweep_cases.csv)：逻辑输入/权重/输出 shape、dtype、实际核心和接口布局、样本数、时间范围、错误。
- [逐组汇总](verification/shape_sweep_groups.csv) 与 [完整 JSON 汇总](verification/shape_sweep.json)。
- [AIR 一致性检查](verification/shape_air_compatibility.json)：最终生成器对已编译 AIR 案例的字节一致性核验；实际编译路径以各 case 的 route 字段为准。
- [原配置→调整配置映射](verification/shape_sweep_adjustments.csv)：每项原参数、放大/缩小参数、实际计时和未解决原因。
- 可直接运行的参数文件：[原表](configs/shape_cases.json)、[放大配置](configs/shape_larger.json)、[小 shape 对照](configs/shape_smaller_controls.json)。
- 原始 profiling、op.csv、设备日志、ONNX/OMC 留在本机 results_sweep*，不提交本次原始结果包。

```powershell
python shape_plan.py
python sweep_shapes.py --ddk-root <DDK目录> --hdc <hdc.exe> --target <设备ID> --allow-multiple-core
python adapt_shapes.py --runs results_sweep results_sweep_ext
python sweep_shapes.py --config results_sweep_adapt/next.json --out results_sweep_ext --ddk-root <DDK目录> --hdc <hdc.exe> --target <设备ID> --allow-multiple-core
# 重复 adapt / sweep，直到 next.json 为空；设备 profiling 必须串行。
python report_shapes.py --runs results_sweep results_sweep_ext
```

复测使用 --profile-only --repeat-measured --times 45 --warmup 5 --only <配置hash...>，并指定原模型所在 --out；旧轮次保存于 previous_measurements，并由 adapt/report 合并判断。更换生成器、DDK 或布局模式时使用新输出目录。

调整状态（按组内映射计数，包含重复引用）：`{"smaller/smaller_control":19,"larger/replacement_above_31":160,"unresolved/unmeasured":62}`。
