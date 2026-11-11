# 验证记录：简化配置与零 TransData（2026-11-11）

本文是旧版结构对照。最新的 ND Mul/LayerNorm、两种 Add、严格 bias 校验和实机临界点见 [BOUNDARY_31US.md](BOUNDARY_31US.md)；核心端口的原模型/当前模型对照见 `verification/layout_comparison.json`。最新默认模式已不使用下表的 Mul/LayerNorm 编译路径。

后续实机测试见 [BENCHMARK_31US.md](BENCHMARK_31US.md)：12 类推荐配置已通过 NPU 逐节点计时、输出数值检查和三组复测。下表保留此前的默认产物结构记录；其中原生 LayerNorm 已确认进入 CPUCL，CastT 默认产物没有导出 NPU 节点耗时，不能将“编译成功/零 TransData”当作 NPU 执行验证。

13 类默认样例均通过 ONNX 检查、CPU 试运行、Kirin9030 OMG 编译及 OMC 结构解析。TransData 总数从 **18 减少到 0**。Windows 和 WSL 的 **26 项测试分别全部通过**，包括二维/四维配置的数值对应关系、配置回读、NPY 路径、布尔编译参数和残留 TransData 拒绝测试。

| 算子家族 | 原 TransData | 当前 TransData | 当前 OMC 计算节点 |
|---|---:|---:|---|
| MatMul | 0 | 0 | MatMulV2 × 1, Permute × 2 |
| GemmD | 0 | 0 | GemmV3 × 1 |
| BatchMatMul | 0 | 0 | BatchMatMulV2 × 1 |
| Convolution | 2 | 0 | Convolution × 1 |
| ConvTranspose | 2 | 0 | ConvTranspose × 1 |
| Add | 3 | 0 | Eltwise × 1 |
| Mul | 3 | 0 | Eltwise × 1 |
| Activation | 2 | 0 | Activation × 1 |
| PRelu | 2 | 0 | PRelu × 1 |
| ReduceSum | 0 | 0 | ReduceSum × 1 |
| LayerNorm | 2 | 0 | CastT × 2, LayerNorm × 1 |
| Softmax | 2 | 0 | Softmax × 1 |
| CastT | 0 | 0 | CastT × 1 |

本机环境：DDK-tools-next-6.1.1.0 / kirin9030，WSL Ubuntu-22.04，Python 3.10.21，ONNX 1.14.1，ORT 1.15.1，NumPy 1.26.4。Windows 自检：Python 3.10.20，ONNX/ORT 1.23.2，NumPy 2.2.6。

## 配置和接口变化

- 默认配置及每个样例的同名 JSON 只有 `op` 和 `params`。编译信息在 `logs/<摘要>/record.json`，编译选项在独立的 `compiler.json`。
- 卷积、Add/Mul、Activation、PRelu、Softmax 的 OMC 接口使用 NC1HWC0，调用者必须按 SDK tensor 描述打包数据并分配实际 buffer。ONNX shape 是逻辑 shape，不能将普通 NCHW 数据直接当作 NC1HWC0 内存使用。
- Mul 的 `[1,256]` 补为 `[1,256,1,1]`，实际类型变为 Eltwise（乘法语义）；Softmax 的 `[256,64]` 补为 `[1,256,64,1]`，axis 从 -1 改为 2。CPU 对照测试验证元素计算对应关系。
- LayerNorm 保留 `[256,64]` 和 axis=1，使用 opset 17 原生 LayerNormalization，FP16 默认产物仍有两个 CastT；没有把它报告成纯单节点模型。
- 可选 `configs/layernorm_fp32.json` 也已经真实编译，结果只有 **LayerNorm × 1**，无 CastT / TransData。FP32 输入输出与默认 FP16 的数据类型不同。
- 原二维/分解图参数保留在 `configs/logical_2d.json`；使用 `--io-layout standard` 可生成普通布局接口，允许 TransData。

所有实际 shape、dtype、layout、buffer_bytes、文件哈希见 `verification/results.json` 和各样例日志。`native` 模式会在发布 OMC 前检查 TransData 数量；不是通过删改已编译二进制或过滤统计结果实现。

这批模型用于算子测试，权重为合成值，Softmax 为通用样例，GemmD 家族在本 DDK 下仍为 GemmV3。布局、形状映射、实现和 dtype 的变化可能影响功耗及执行时间；已通过的 CPU 检查不能证明与原模型的 NPU 浮点舍入或性能一致。本节原记录仅覆盖编译和 CPU 检查；后续板端数值、时间测试以链接报告为准，尚未测量功耗。
