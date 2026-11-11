# 实机测试：目标算子超过 31 µs（2026-11-11）

**历史对照：本文先于 ND 布局修正。** Mul、LayerNorm、ConvTranspose 的旧结果不满足后续提出的原节点布局要求；最新结果见 [BOUNDARY_31US.md](BOUNDARY_31US.md)。重现本文的普通配置时需显式添加 `--io-layout native`，不能作为 ND 结果使用。

当时 12 类算子找到达标配置；CastT 尚未取得独立 NPU 节点耗时。每个配置独立运行 3 组，每组 25 次，去掉前 5 次预热后保留 20 次：以下每项的 60 个目标节点样本全部 >31 µs，均来自 `op.csv` 的 `npu_aicore` 记录，OMC 的 TransData 数量均为 0。

连接设备实际上报 **Kirin9020 / MOR**，系统 `MOR-W52 6.1.0.125(SP10ENTC302E101R1P1log)`。编译沿用用户指定的 **Kirin9030 OMG**：`DDK-tools-next-6.1.1.0`，`--platform kirin9030`。这些模型已在该设备上实际运行，但下面的时间**不是 Kirin9030 真机实测值**。

## 推荐参数与实测结果

shape 使用 ONNX 逻辑维度。除 LayerNorm 特别说明外，推荐 ONNX/OMC 计算数据类型均为 FP16。未列出的参数保留 handfree 预设；Softmax 沿用通用预设。

| 家族 → 实际 OMC 类型 | 输入、权重与关键参数 | 输出 shape | 中位数 µs | 最小值 µs |
|---|---|---|---:|---:|
| MatMul → MatMulV2 | 常量 A=`[514,6144]`，运行时 B=`[6144,2]`；启用常量左乘改写 | `[514,2]` | 260 | 259 |
| GemmD → **GemmV3** | A=`[128,288]`，W=`[288,768]`，bias=`[768]`；transA=0，transB=0，alpha=beta=1 | `[128,768]` | 38 | 37 |
| BatchMatMul → BatchMatMulV2 | A=`[8,2,256]`，常量 B=`[8,256,256]` | `[8,2,256]` | 61 | 60 |
| Convolution | X=`[1,96,5,512]`，W=`[96,96,5,1]`，bias=`[96]`；kernel=`[5,1]`，stride=`[1,1]`，pads=0，group=1 | `[1,96,1,512]` | 47 | 46 |
| ConvTranspose | X=`[1,16,128,128]`，W=`[16,1,1,3]`，bias=`[1]`；kernel=`[1,3]`，stride=`[1,2]`，pads/output_padding=0 | `[1,1,128,257]` | 67 | 65 |
| Add → Eltwise(sum) | 两个运行时输入均为 `[1,16,256,128]` | `[1,16,256,128]` | 60 | 58 |
| Mul → Eltwise(product) | 两个运行时输入均为 `[1,256,2048,1]` | `[1,256,2048,1]` | 60 | 58 |
| Activation | Sigmoid，X=`[1,16,256,128]` | `[1,16,256,128]` | 51 | 49 |
| PRelu | X=`[1,64,64,128]`，slope=`[64,1,1]`，斜率=0.25 | `[1,64,64,128]` | 42 | 41 |
| ReduceSum | X=`[4,65536,1]`，axes=`[0]`，keepdims=1 | `[1,65536,1]` | 53 | 52 |
| LayerNorm | X=`[256,64,1,1]`，scale/bias=`[1,64,1,1]`；axis=1，epsilon≈1e-5；专用编译配置见下 | `[256,64,1,1]` | 56 | 56 |
| Softmax | X=`[1,4096,64,1]`，axis=2 | `[1,4096,64,1]` | 55 | 55 |

以上是已验证的可用配置，没有穷举所有参数，也不代表达到 31 µs 的最小 shape。硬件调度随 shape 改变，耗时并不必然单调：例如 BatchMatMul 的 M 从 2 增至 64 后，本次实测反而更快。

部分原始样本有毫秒级长耗时，例如 8197 µs。这里展示中位数和最小值；**没有删除这些长耗时样本**。均值、最大值、标准差、全部原始样本保存在汇总和各组 `statistics.json` 中，不能把表中中位数当成所有样本的均值。

MatMul 的常量左乘通过 `(B.T @ A.T).T` 实现，ONNX 实际保存的转置权重为 `[6144,514]`。表中耗时只取 `__g_product_T_MatMul(MatMulV2)`，不包含前后两个 Permute。GemmD 是家族名称，本 DDK 产物确实为 GemmV3，没有将其改名冒充 GemmD。

## LayerNorm：保留原规模并真正运行在 NPU

原预设的 opset 17 原生 `LayerNormalization` 虽然没有 TransData，但解析发现其 `GraphOp.cl_name=CPUCL`，实机也没有导出该节点的 NPU 耗时。仅看节点名称或 TransData 数量不足以确认 NPU 执行。

推荐配置改用 **FP32 ONNX 分解表达**（ReduceMean/Sub/Pow/Add/Sqrt/Div/Mul/Add），编译时显式设置 **FP16 输入输出和 NC1HWC0 接口**。当前 DDK 将其融合成单个 `Y_Add(LayerNorm)`，没有 CastT / TransData；实机 `op.csv` 确认设备类型为 `npu_aicore`。

`[256,64,1,1]` 和沿 axis=1 的归一化对应原 `[256,64]` 的逻辑运算，新增的是两个单例维度。ONNX 参考计算为 FP32，OMC 输入输出/核心为 FP16，这个精度选择需要明确保留。所有推荐配置均用固定种子的有限随机输入与 ORT 对照，按 `atol=0.003, rtol=0.03` 逐元素检查，通过才计入结果。这是数值冒烟检查，并非原训练权重的完整精度验证。

## CastT：目前未确认 NPU 阈值

本次已实际测试：

- 普通布局、FP32→FP16，原 shape `[2,2,2,257]`：模型执行成功，只有 `model.csv`，没有目标 `op.csv` / PMU CSV。
- 放大到 `[2,2,512,257]`：模型执行成功，输出数值校验通过，仍没有 NPU CastT 节点耗时。
- 强制 NC1HWC0，shape `[1,16,256,128]`：OMG 编译成功且无 TransData，但实机推理失败。
- FP32 Sigmoid 上下文自动插入两个 CastT：OMC 中有 CastT，但 NPU `op.csv` 只含 Activation 和 TransData，仍不能测出 CastT 本身。

原 handfree 的 CastT 使用 `generated_lib`，本机 DDK 对照产物为 `elementary_lib`；这是可核对的实现差异，尚不足以断言唯一根因。当前结果不支持“只增大 shape 就能让独立 NPU CastT 超过 31 µs”。需要可提供该 CastT NPU 实现的 DDK/运行时或不同的算子构建路径继续验证。没有用 CPU 耗时、整模型时间或其他节点的时间替代它。

## 复现

11 类普通推荐配置与 LayerNorm 分开编译，以保持 `op/params` 配置简单，且避免全局修改其他算子的 I/O 类型：

```bash
python generate_ops.py --config configs/over31us.json --all \
  --ddk-root /path/to/DDK-tools-next-6.1.1.0 --out results_rebuilt
python generate_ops.py --config configs/layernorm_31us.json \
  --compiler-config configs/layernorm_npu_compiler.json \
  --ddk-root /path/to/DDK-tools-next-6.1.1.0 --out results_rebuilt
```

Windows 主机可用 HDC 读取 Linux/WSL 生成的模型。设备需已有与当前固件兼容的 `model_run_tool_internal`、`data_proc_tool_internal` 及依赖库，默认放在 `/data/local/tmp/tools_sysdbg`：

```powershell
python profile_ops.py --models results_rebuilt --out measured_run1 `
  --target 0123456789ABCDEF --hdc "D:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
```

使用新的 `--out` 重复三组。默认 `--times 25 --warmup 5 --perf 3 --threshold 31`，启用 PMU。`dev_perf=3` 是工具请求的 HIGH 档位，不等于实测固定时钟。每组保存设备信息、温度读数、命令、模型哈希、原始 profiling 压缩包、CSV、输出和数值检查。不要并发运行多个设备 profiling 进程，它们共享设备的 profiling 目录。

脚本按 OMC 描述打包 NC1HWC0 输入（C0=16）并保留其缓冲区尾部 padding；普通 NCHW/ND 接口按照本机 ROMC 工具的逻辑字节数供数。此规则经过本次设备验证，其他 SDK/运行时需重新核对。脚本保存并导出 profiling 数据，不删除设备的全局 profiling 目录。

## 结果位置

- `configs/over31us.json`：11 类推荐参数，只含 `op/params`。
- `configs/layernorm_31us.json` 和 `configs/layernorm_npu_compiler.json`：纯 NPU LayerNorm 的模型及编译参数。
- `verification/benchmark_31us.json`：12 类推荐配置、输入/输出/权重 shape、I/O 布局、哈希与统计值。
- `results_31us/recommended/<算子>/`：本机已验证的 ONNX、OMC、参数及编译记录。
- `results_31us/summary.json`、`summary.csv`：本机汇总；各实验子目录保留成功、失败和未达阈值的证据。
- `results_31us_verified.zip`：可携带的模型及三组原始测量证据，解压后的模型结构为 `results/<算子>/<算子>_<参数>.onnx/.omc`。

本次没有测量功耗，也没有在 Kirin9030 真机或公司 DDK 上复测。阈值结论限定于本文记录的设备、编译器、接口和性能档位。
