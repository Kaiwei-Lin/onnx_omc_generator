# ONNX / OMC 单算子生成器

生成 13 类算子的 ONNX 和 Kirin OMC。可编辑 JSON 只保留 `op`（算子类型）和 `params`（参数）。默认开启去 TransData 的编译模式；每个 OMC 生成后都会重新解析，发现残留 TransData 就报错，不会把它当成成功结果。

默认参数以 handfree 模型为依据，权重是可重复的合成测试值。Softmax 没有找到可独立恢复的原节点，使用通用样例。实际 OMC 类型由 DDK 决定，例如本机 GemmD 家族实际编译为 GemmV3。

## 快速使用

在 Linux / WSL 中，使用已经能够运行 OMG 的 Python 环境：

```bash
git clone git@github.com:Kaiwei-Lin/onnx_omc_generator.git
cd onnx_omc_generator
python -m pip install -r requirements.txt
python generate_ops.py --all --ddk-root /path/to/DDK-tools --out results_company
```

Python 3.10 或更新版本。DDK 需要自行安装，默认平台为 `kirin9030`。Windows 可用 `--onnx-only` 生成 ONNX，Linux ELF 版 OMG 需要在 Linux / WSL 中运行。

```bash
python generate_ops.py --list
python generate_ops.py --op Convolution --onnx-only --check-runtime
python -m zipfile -e results.zip .
```

`results.zip` 是本机已验证的 13 对样例，解压得到 `results/`。重新编译使用新输出目录，避免覆盖不同 DDK 的结果。建议使用较短路径，OMG 有路径长度限制。

## 配置只包含类型和参数

`defaults_handfree.json` 是 13 项配置的列表。单个配置文件也可以直接传给 `--config`，例如：

```json
{
  "op": "Add",
  "params": {
    "dtype": "float16",
    "inputs": [
      {"name": "X", "shape": [1, 16, 2, 128]},
      {"name": "W", "shape": [1, 16, 2, 128]}
    ]
  }
}
```

```bash
python generate_ops.py --config add.json --ddk-root /path/to/DDK-tools --out results_add
```

同名输出 JSON 也采用这个格式，可以直接修改后再次作为 `--config` 使用。来源、编译命令、节点描述、哈希和验证结果不再混在可编辑参数中，详细诊断统一放在 `logs/` 中。

| 参数 | 含义 |
|---|---|
| `dtype` | 默认 float16；也支持 float32 等 ONNX 类型，具体取决于算子和 DDK |
| `inputs` | 按输入端口顺序填写 name、shape，可单独指定 dtype |
| `inputs[].init` | 存在时是常量，不存在时是运行时输入 |
| `attributes` | ONNX 属性，例如 transB、strides、pads、axis、epsilon |
| `seed` | 合成常量的随机种子，可省略，缺省为 0 |
| `opset` | 默认 13；原生 LayerNormalization 需要 >=17 |

shape 必须是静态正整数，标量写 `[]`。输出 shape 自动推导。同一 ONNX 算子的操作数需满足它的 dtype 和广播约束。常量初始化支持：

```json
{"kind": "normal", "mean": 0.0, "std": 0.02}
{"kind": "uniform", "low": -0.1, "high": 0.1}
{"kind": "fill", "value": 0.25}
{"kind": "values", "values": [1, 2, 3, 4]}
{"kind": "npy", "path": "weights/B.npy"}
```

NPY 路径相对于配置文件目录，shape 必须一致，禁止 pickle。至少保留一个运行时输入，避免整图常量折叠。

也可以快速覆盖单个选中的配置：

```bash
python generate_ops.py --op GemmD --set attributes.transB=1 --set 'inputs.1.shape=[768,288]' --onnx-only --out results_transB1
python generate_ops.py --op Activation --set 'activation="LeakyRelu"' --set attributes.alpha=0.1 --onnx-only --out results_leaky
```

`--set` 的右侧必须是 JSON 值，字符串保留双引号；也接受 `params.attributes.transB=1`。可在列表中复制同一算子以批量生成多个配置。旧版 `{ "compiler": ..., "cases": [...] }` 格式仍可读取。

## 各类算子的参数

| 家族 → ONNX | 主要参数 |
|---|---|
| MatMul / BatchMatMul → MatMul | A/B shape、常量状态；`transpose_a` / `transpose_b` 对最后两维转置；支持 batch 广播 |
| GemmD → Gemm | transA、transB、alpha、beta；二维 A/B，偏置可广播 |
| Convolution → Conv | kernel_shape、strides、pads、dilations、group、auto_pad；W=`[Cout,Cin/group,k...]` |
| ConvTranspose → ConvTranspose | 上述属性及 output_padding、output_shape；W=`[Cin,Cout/group,k...]` |
| Add / Mul | 两个输入的 shape、dtype、是否常量，支持广播 |
| Activation → 指定激活 | `activation` 选择 Sigmoid、Relu、Tanh、LeakyRelu、Elu、Selu、HardSigmoid、Softplus、Softsign、ThresholdedRelu、HardSwish、Gelu、Clip；相应属性放 attributes |
| PRelu → PRelu | X 和 slope 的 shape / 数值；NCHW 按通道 slope 常用 `[C,1,1]` |
| ReduceSum → ReduceSum | axes、keepdims、noop_with_empty_axes；opset>=13 时自动生成 axes 输入 |
| LayerNorm → LayerNormalization / 分解图 | axis、epsilon；`implementation` 为 native 或 decomposed；`accumulation_dtype` 控制内部统计精度 |
| Softmax → Softmax | axis；建议 opset>=13 |
| CastT → Cast | to 可用 dtype 名称或 ONNX 类型枚举，不能直接使用 OMC dtype 编码 |

卷积使用 channel-first 逻辑布局，pads 顺序为 `[begin_H,begin_W,end_H,end_W]`。使用 auto_pad 时删除 pads。Clip 的 min/max 在 opset>=11 中是标量输入；Gelu 原生算子需要较新 opset/DDK。去掉第三个输入可生成无偏置 Gemm、Conv、ConvTranspose 或 LayerNorm。

LayerNorm 的 decomposed 模式使用 ReduceMean/Sub/Pow/Add/Sqrt/Div/Mul 等标准 ONNX 节点；可用 `affine_dtype` 控制 scale/bias 运算精度。native 模式使用标准 LayerNormalization，affine 精度由输入类型决定。不同内部精度可能带来舍入差异。

## 如何去除 TransData

完整模型的相邻算子可以共享内部布局。独立 ONNX 暴露普通布局接口时，OMG 会在边界加入布局转换。脚本通过编译接口选项去除这些转换，不修改编译后的 OMC 二进制。

`compiler.json` 的 `io_layout` 默认为 `native`：

- 卷积、Add/Mul、Activation、PRelu、Softmax 使用 NC1HWC0 接口。脚本传入 `--use_origin_format=false`、按输入名指定 `--input_format`，并按输出**生产节点名**指定 `--output_format`。
- 原生 LayerNormalization 使用本 DDK 的 ND 实现，不强制套 NC1HWC0；MatMul/Gemm/ReduceSum/Cast 等也保留各自的兼容接口。
- 所有默认产物必须经解析确认 `TransData=0` 才会发布到结果目录。

这会改变部分 OMC 的外部数据布局。**NC1HWC0 模型不能直接输入普通 NCHW 连续内存。** 板端需根据 SDK 的 tensor 描述准备、打包和解包数据，并按实际 buffer size 分配；不能只用逻辑 shape 的元素数估算缓冲区。接口的 layout、shape、dtype、buffer_bytes 记录在 `logs/<摘要>/record.json` 的 `compile.actual_omc.interfaces` 中。

本机 DDK 的 NC1HWC0 外部接口要求四维。默认值为此做了以下明确调整，数学运算的元素对应关系保持一致：

| 算子 | 原配置 | 当前默认配置 |
|---|---|---|
| Mul | 两个 `[1,256]` 输入 | 两个 `[1,256,1,1]` 输入 |
| Softmax | `[256,64]`，axis=-1 | `[1,256,64,1]`，axis=2 |
| LayerNorm | `[256,64]`，axis=1，分解图 | shape/axis 不变，改为 opset 17 原生 LayerNormalization |

保留原二维/分解图配置的文件是 `configs/logical_2d.json`。若要普通外部布局或恢复这些表达：

```bash
python generate_ops.py --config configs/logical_2d.json --io-layout standard --ddk-root /path/to/DDK-tools --out results_logical
```

standard 模式允许 TransData，保留普通 ONNX 接口。新参数若不能满足 native 模式的约束，会报错并保留诊断文件；不会自动更改用户的 shape 或掩盖残留节点。

**没有 TransData 不代表只有一个计算节点。** MatMul 的常量左乘兼容改写仍有两个 Permute；本机默认 FP16 LayerNorm 仍有两个 CastT。要试验 FP32 LayerNorm，使用 `configs/layernorm_fp32.json`。实际节点见 `VALIDATION.md` 和结果记录。尺寸映射、实现版本及 dtype 都会影响功耗，不能把新模型当成原节点的硬件执行完全复现。

## 输出与编译选项

```text
results/<算子>/<算子>_<shape和参数>_<摘要>.onnx
results/<算子>/<算子>_<shape和参数>_<摘要>.omc
results/<算子>/<算子>_<shape和参数>_<摘要>.json   # 只有 op / params
results/<算子>/logs/<摘要>/record.json             # 自动诊断信息
results/<算子>/logs/<摘要>/attempt_*/convert.log
```

文件名中的摘要覆盖完整参数，避免截短和浮点数显示导致重名。源 ONNX、已有 OMC 或参数文件被修改后，不会静默覆盖。相同参数、相同编译器指纹可复用已验证产物。每次命令还会保存 `run_*.json` 清单；任一任务失败时整体退出码为 1。

`compiler.json` 独立存放平台、io_layout、weight_data_type、io_types、timeout 等编译选项。用 `--compiler-config` 选择公司配置，用 `--ddk-root` / `DDK_PATH` 指定 DDK，用 `--platform` 覆盖平台。`io_types=model` 保留各输入输出的独立 dtype，特别是 CastT 的 FP32→FP16。FP32 ONNX 若需要 FP32 编译权重，也要设置 `weight_data_type=FP32`。

高级选项仍支持 `omg`、`loader`、`env`、`extra_args`、`ge_proto`、`version_label`，以及完整 argv 数组 `command`。自定义 command 支持 `{model}`、`{output}`、`{platform}`、`{ddk_root}`、`{input_format}`、`{output_format}` 占位符；需由公司 wrapper 提供相应编译选项。native 模式最终仍核验 TransData 数量。切换/替换 DDK 或 wrapper 时使用新输出目录。

## 验证与边界

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python generate_ops.py --all --check-runtime --ddk-root /path/to/DDK-tools --out results_check
```

`--check-runtime` 是关闭 ORT 图优化的 CPU 数值/接口试运行，避免旧 ORT 的 FP16 融合问题。实际 OMG 产物和布局核验见 `VALIDATION.md`。OMC 解析支持已验证的未加密 IMOD/GE 格式；未包含 DDK、原训练权重或板端功耗测试。所有样例都应在目标设备上验证数值和数据布局后再做功耗测量。
