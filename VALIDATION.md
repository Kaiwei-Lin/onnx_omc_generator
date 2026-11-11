# 本机验证记录（2026-11-11）

13 类默认配置全部通过 ONNX checker、shape 推导、ONNX Runtime CPU 试运行，以及 Kirin9030 OMG 编译。每个生成的 OMC 都由 IMOD/GE 解析器重新读取，核对真实类型及输入输出描述。22 项单元/数值测试在 Windows 和 WSL 环境中分别全部通过。

编译环境为 WSL Ubuntu-22.04、Python 3.10.21、ONNX 1.14.1、NumPy 1.26.4、ONNX Runtime 1.15.1、protobuf 3.20.3、DDK-tools-next-6.1.1.0，平台 kirin9030。Windows 测试环境为 Python 3.10.20、ONNX 1.23.2、NumPy 2.2.6、ONNX Runtime 1.23.2。

| 请求家族 / 目录 | OMC 中实际计算节点（包含布局转换） |
|---|---|
| MatMul | MatMulV2 × 1, Permute × 2 |
| GemmD | GemmV3 × 1 |
| BatchMatMul | BatchMatMulV2 × 1 |
| Convolution | Convolution × 1, TransData × 2 |
| ConvTranspose | ConvTranspose × 1, TransData × 2 |
| Add | Eltwise × 1, TransData × 3 |
| Mul | Mul × 1, TransData × 3 |
| Activation | Activation × 1, TransData × 2 |
| PRelu | PRelu × 1, TransData × 2 |
| ReduceSum | ReduceSum × 1 |
| LayerNorm | LayerNorm × 1, TransData × 2 |
| Softmax | Softmax × 1, TransData × 2 |
| CastT | CastT × 1 |

完整 shape、权重 shape、dtype、文件哈希汇总见 `verification/results.json`；详细节点属性和编译命令见每对 ONNX/OMC 同名 JSON 及其日志。测试输出见 `verification/test_windows.txt` 和 `verification/test_linux.txt`。压缩包仅含最终通过的默认样例，失败探索记录保留在原项目输出目录。

默认 LayerNorm 的逻辑 shape / axis / epsilon 来自 handfree；为满足当前 DDK 融合要求，统计量与 affine 部分在 ONNX 中按 FP32 计算，再转换为 FP16 输出。OMG 最终融合出一个 LayerNorm 和两个 TransData。原 OMC 的内部舍入与该 ONNX 未进行数值对比。

GemmD 目录内的默认 OMC 实际为 GemmV3。切换到公司 DDK 后必须重新解析，不能把目录名当成编译类型保证。Softmax 是明确标注的通用样例。带权重算子使用合成测试值。

验证范围不包括手机/NPU 上的执行、数值误差、时间或功耗，也不包括所有可能参数组合。上述版本和默认参数的通过结果不能保证其他 DDK 或任意配置都可编译。
