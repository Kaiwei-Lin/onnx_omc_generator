# ONNX / OMC 单算子模型生成器

这套工具生成 MatMul、GemmD、BatchMatMul、Convolution、ConvTranspose、Add、Mul、Activation、PRelu、ReduceSum、LayerNorm、Softmax、CastT 的参数化测试模型。运行 `generate_ops.py` 即可先生成标准 ONNX，再调用你指定版本的 Kirin OMG，解析 OMC 并记录实际算子类型、输入输出 shape、权重 shape、dtype 和编译参数。

默认参数在 `defaults_handfree.json`。除 Softmax 外，shape 和主要属性取自此前解析的 `handfree_ainr_fuse.omc`。这是算子结构的测试复现，常量使用可重复的合成数据；不包含原模型训练权重。

## 从 GitHub 获取

```bash
git clone git@github.com:Kaiwei-Lin/onnx_omc_generator.git
cd onnx_omc_generator
```

仓库中的 `results.zip` 保存本机已验证的 13 对 ONNX/OMC 样例、参数记录和编译日志。需要查看现成结果时，在仓库根目录解压：

```bash
python -m zipfile -e results.zip .
```

解压后得到下文所述的 `results/` 目录。生成的 `results/`、`results_*/` 和 Python 缓存已加入 Git 忽略规则。DDK 需要在使用者的机器上自行配置；更换 DDK 后请使用新的输出目录重新编译。

## 1. 快速使用

Python 3.10 或更新版本。优先使用公司已经能成功运行 OMG 的 Python 环境，缺少依赖时再安装：

```bash
python -m pip install -r requirements.txt
python generate_ops.py --list
```

在 **Linux 或 WSL** 中，指定实际 DDK 根目录，一次生成全部 13 类：

```bash
python generate_ops.py --all --ddk-root /path/to/DDK-tools --out results
```

加上 `--check-runtime`，会用 ONNX Runtime CPU 做输入输出 shape、dtype、有限值试运行。OMG 和板端运行仍使用各自的环境。

CPU 自检关闭 ORT 图优化，以验证导出的原始图，避免旧 ORT 将 FP16 分解图融合为自己不支持的 LayerNorm 内核。这不影响 OMG 的优化和融合。

```bash
python generate_ops.py --all --ddk-root /path/to/DDK-tools --out results --check-runtime
```

在 **Windows** 中可以先只生成 ONNX：

```powershell
python generate_ops.py --all --onnx-only --check-runtime --out results_onnx
```

Windows 不能直接运行 Linux ELF 版 OMG。在 WSL 终端中进入工具包目录，再运行 Linux 命令；DDK 和文件路径须使用 `/mnt/d/...` 等 Linux 路径。脚本不依赖本项目的绝对路径。

本机验证命令（PowerShell，在 `D:\WHU\huawei_power_opt` 下运行）：

```powershell
wsl.exe -d Ubuntu-22.04 --cd /mnt/d/WHU/huawei_power_opt /home/lkw/miniconda3/envs/cannkit_torch_alg/bin/python tools/onnx_omc_generator/generate_ops.py --all --ddk-root spatialnet_field_kit/microbenchmark/.local/DDK-tools-next-6.1.1.0 --out tools/onnx_omc_generator/results --check-runtime
```

工具包附带的 `results/` 是本机 DDK 的结果。到公司首次重新编译时，使用新目录，例如 `--out results_company`；若希望目录名正好为 `results`，将源码文件放入另一个空目录后执行默认命令。

建议解压到较短目录（例如 `D:\omc_ops` 或 `~/omc_ops`）。OMG 自身也有路径长度限制，避免在很深的聊天附件临时目录中直接编译。

## 2. 输出结构及命名

```text
results/
  GemmD/
    GemmD_f16_in0-128x288_w1-288x768_w2-768_out-128x768_..._<配置摘要>.onnx
    GemmD_f16_in0-128x288_w1-288x768_w2-768_out-128x768_..._<配置摘要>.omc
    GemmD_f16_in0-128x288_w1-288x768_w2-768_out-128x768_..._<配置摘要>.json
    logs/<配置摘要>/attempt_<编号>/command.json
    logs/<配置摘要>/attempt_<编号>/convert.log
  Convolution/
    ...
  run_<编号>.json
```

ONNX 和 OMC 使用同一文件名主体。`in0` 是运行时输入端口 0，`w1` 是常量端口 1，`out` 是推导出的输出 shape。文件名包含可读参数和 12 位配置哈希；过长的可读部分会截短，完整参数始终在同名 JSON 的 `case` 中。哈希覆盖形状、属性、dtype、随机种子、初始化配置和外部 NPY 内容摘要，避免截短或浮点数显示精度导致重名。

同名 JSON 中主要字段：

| 字段 | 含义 |
|---|---|
| `case` | 本次完整配置及 handfree 来源 |
| `matches_handfree_preset` | 是否仍是发布的默认预设，含已披露的编译适配 |
| `onnx_info.inputs / outputs` | ONNX 外部接口 shape / dtype |
| `onnx_info.initializers` | 实际保存的常量名称、shape、dtype，包括转置或提升精度后的权重 |
| `compile.actual_omc.types` | 解析编译产物得到的真实算子名称和数量 |
| `compile.actual_omc.nodes` | 每个计算节点的类型、shape、dtype、layout、属性；常量输入附原 Const 描述 |
| `compile.compiler` | DDK 路径、OMG / GE schema 哈希、平台、选项和 Python 环境标识 |
| `runtime_check` | 可选 CPU 试运行结果，不代表 NPU 数值或性能验证 |

`run_<编号>.json` 只记录该次命令选择的任务；不会覆盖之前的运行清单。批量任务遇到单例失败会继续，最终退出码为 1，并保留日志。返回码为 0、OMC 结构解析成功、源 ONNX 未变化后，才会将 OMC 发布到上述成对文件路径；失败的临时文件留在对应日志目录。

重复运行可复用哈希匹配的产物。不同 ONNX 内容、不同编译器配置或手动修改过的 OMC 不会被静默覆盖；请更换 `--out`。不要并发向同一输出目录写入同一配置。

## 3. handfree 默认值

原 OMC SHA-256：`414032aa61ba24f8d068e1eef431a50882036ad9033ecb16690b309b26180cc7`。

下表描述逻辑 shape，不是 `FRACTAL_NZ / FRACTAL_Z / NC1HWC0` 物理存储尺寸。除 CastT 输入外，默认外部数据为 FP16；`C` / `B` 为偏置时也计入常量输入。

| 目录 | handfree 节点 ID | 输入 / 常量 shape | 输出 shape | 主要参数 |
|---|---:|---|---|---|
| MatMul | 3739 | 常量 A `[514,6144]`，运行时 B `[6144,2]` | `[514,2]` | 常量左乘等价改写 |
| GemmD | 3019 | A `[128,288]`，W `[288,768]`，C `[768]` | `[128,768]` | transA=0，transB=0，alpha=beta=1 |
| BatchMatMul | 2917 | A `[8,2,256]`，W `[8,256,256]` | `[8,2,256]` | 无转置 |
| Convolution | 3062 | X `[1,96,5,128]`，W `[96,96,5,1]`，B `[96]` | `[1,96,1,128]` | kernel `[5,1]`，stride `[1,1]`，group=1，pads=0 |
| ConvTranspose | 2174 | X `[1,16,2,128]`，W `[16,1,1,3]`，B `[1]` | `[1,1,2,257]` | kernel `[1,3]`，stride `[1,2]`，pads=0，output_padding=0 |
| Add | 1416 | 两个运行时输入 `[1,16,2,128]` | `[1,16,2,128]` | 逐元素加 |
| Mul | 413 | 两个运行时输入 `[1,256]` | `[1,256]` | 逐元素乘 |
| Activation | 520 | X `[1,16,2,128]` | `[1,16,2,128]` | Sigmoid |
| PRelu | 2392 | X `[1,64,2,128]`，slope `[64,1,1]` | `[1,64,2,128]` | 合成 slope=0.25 |
| ReduceSum | 3514 | X `[4,256,1]` | `[1,256,1]` | axes `[0]`，keepdims=1，noop_with_empty_axes=1 |
| LayerNorm | 2257 | X `[256,64]`，scale/bias `[1,64]` | `[256,64]` | axis=1，epsilon≈1e-5；内部计算精度见配置 |
| Softmax | 通用样例 | X `[256,64]` | `[256,64]` | axis=-1 |
| CastT | 3 | X `[2,2,2,257]`，FP32 | `[2,2,2,257]`，FP16 | to=float16 |

完整节点名称、区域、原始描述和每项适配见 `case.source`。Softmax 在已解析 handfree OMC 中没有找到能独立对应的 Softmax/SoftmaxV2 节点；这里借用 LayerNorm 输入尺寸制作通用样例，axis=-1 是测试选择。

## 4. 修改任意合法参数

最稳妥的方法是复制 `defaults_handfree.json` 为 `my_cases.json`，修改 `cases` 中需要的项，执行：

```bash
python generate_ops.py --config my_cases.json --all --ddk-root /path/to/DDK-tools --out results_custom
```

可以删除不需要的项，也可以复制同一算子多个配置并给不同 `id`。`--op` 接受大小写不敏感的算子名或精确 case ID；优先按 ID 匹配，再按算子家族匹配。不指定 `--op` 时生成所有 case。`Convalution`、`Conv` 都作为 `Convolution` 别名接受。

### 4.1 快速覆盖单个配置

```bash
# Gemm: [128,288] @ [768,288].T -> [128,768]
python generate_ops.py --op GemmD --set attributes.transB=1 --set 'inputs.1.shape=[768,288]' --onnx-only --out results_transB1

# 所有操作数改为 FP32，初始偏置也同步变为 FP32
python generate_ops.py --op GemmD --set 'dtype="float32"' --set attributes.transB=1 --set 'inputs.1.shape=[768,288]' --ddk-root /path/to/DDK-tools --out results_gemm_fp32

# 改为 LeakyRelu
python generate_ops.py --op Activation --set 'activation="LeakyRelu"' --set attributes.alpha=0.1 --onnx-only --out results_leaky

# 改 Softmax 输入和归一化轴
python generate_ops.py --op Softmax --set 'inputs.0.shape=[2,8,128,128]' --set attributes.axis=-1 --onnx-only --out results_softmax
```

`--set` 的等号右侧必须是 JSON 值，字符串需要 JSON 双引号；只允许对单个选中的 case 使用，避免误改整批。上例适合 Bash 和 PowerShell 7；旧版 PowerShell 的原生命令引号规则不同，建议直接编辑 JSON。使用 FP32 ONNX 时，默认编译选项仍是 `weight_data_type=FP16`；如果希望编译权重保留 FP32，同时修改 `compiler.weight_data_type`，最终以解析出的 OMC dtype 为准。

### 4.2 case 公共字段

| 字段 | 用法 |
|---|---|
| `op` | 13 个算子家族之一 |
| `id` | 可选，批量参数实验的唯一标签 |
| `dtype` | 默认 float16；支持 float16/32/64、int8/16/32/64、uint8/16/32/64、bool；具体算子合法类型由 ONNX schema 和 DDK 限制 |
| `opset` | 默认 13，脚本接受 >=11；本机默认验证为 13。Native LayerNorm 至少 17，原生 Gelu 至少 20 |
| `seed` | 非负整数，控制合成权重 |
| `inputs` | 按 ONNX 输入顺序填写，包含 name、shape、可选 dtype 和 init |
| `attributes` | ONNX 算子属性；拼写错误或该 opset 不支持的属性会报错 |
| `expect_omc_type` | 可选，例如 `"GemmD"`；实际 OMC 未包含该类型时任务失败并保留产物供查看 |
| `source` | 追溯记录，不影响计算；自定义配置可以删除 |

shape 使用静态正整数列表；标量使用 `[]`。不支持动态维度。修改输入后输出 shape 自动推导。输入名仅用字母开头的字母、数字、下划线，`Y` 保留为输出名。不要将 `source.outputs` 当成可编辑的目标输出，它只是原始来源。

某个输入存在 `init` 时生成 initializer，否则成为运行时输入。去掉偏置输入即可测试无偏置 Gemm/Conv/ConvTranspose/LayerNorm。至少保留一个运行时输入，避免整个模型被常量折叠。

初始化方式：

```json
{"kind": "normal", "mean": 0.0, "std": 0.02}
{"kind": "uniform", "low": -0.1, "high": 0.1}
{"kind": "fill", "value": 0.25}
{"kind": "values", "values": [1, 2, 3, 4]}
{"kind": "npy", "path": "weights/B.npy"}
```

`values` 元素数须等于 shape 元素数。`npy` 路径相对于配置文件目录，shape 须完全一致，数据转换为所选 dtype；不加载 pickle，并将文件内容哈希计入配置身份。不同机器使用同一 seed 时仍可能受到 NumPy 版本影响；需要逐字节一致时携带 `.npy` 或生成好的 ONNX。

### 4.3 各算子参数

| 家族 → ONNX | 可改参数及约束 |
|---|---|
| MatMul / BatchMatMul → MatMul | A/B shape、dtype、是否常量；支持 batch 广播；case 顶层 `transpose_a` / `transpose_b` 对最后两维转置。BatchMatMul 至少一侧 rank>=3 |
| GemmD → Gemm | `transA`、`transB`、`alpha`、`beta`；A/B 必须二维，偏置允许单向广播。不要仅修改 transB 而忘记调整 B shape |
| Convolution → Conv | `kernel_shape`、`strides`、`pads`、`dilations`、`group`、`auto_pad`；NCHW / NCDHW 等 channel-first 逻辑布局；W=`[Cout,Cin/group,k...]` |
| ConvTranspose → ConvTranspose | 上述参数及 `output_padding`、`output_shape`；W=`[Cin,Cout/group,k...]`；output_shape 仅填空间维 |
| Add / Mul | 输入 shape/dtype/常量状态，支持 ONNX 多向广播 |
| Activation → 对应 ONNX 激活 | 顶层 `activation` 支持 Relu、Sigmoid、Tanh、LeakyRelu、Elu、Selu、HardSigmoid、Softplus、Softsign、ThresholdedRelu、HardSwish、Gelu、Clip；alpha/beta/gamma 等放 attributes，按该 op 的 schema 设置 |
| PRelu → PRelu | X/slope shape 和值；NCHW 按通道 slope 常用 `[C,1,1]`，不是 `[C]` |
| ReduceSum → ReduceSum | `axes`、`keepdims`、`noop_with_empty_axes`；opset>=13 的 axes 自动变为 INT64 initializer。缺省 axes 或空 axes 的含义遵循对应 opset |
| LayerNorm → 分解图 / LayerNormalization | `axis`、`epsilon`；归一化 axis 到末尾。顶层 `implementation` 为 decomposed 或 native；见下文 |
| Softmax → Softmax | `axis`；建议 opset>=13，以使用沿指定轴归一化的语义 |
| CastT → Cast | `to` 支持 dtype 名称或 ONNX dtype 枚举整数；不要将 OMC 的 src_dtype/dst_dtype 编码直接当作 ONNX 编码 |

卷积 padding 顺序为 `[begin_H,begin_W,end_H,end_W]`，高维按同样规则。`auto_pad` 使用 SAME_UPPER/SAME_LOWER/VALID 时删除 `pads`。stride/dilation/group 和相应输入/权重必须相容。ONNX 支持不表示该 DDK 支持所有 rank、group、dtype 或激活类型；不支持时日志给出真实失败信息。

Clip 在 opset>=11 的 min/max 是可选标量输入，放到 `inputs` 的第 2、3 项，通常用 `init: {"kind":"fill", "value":...}`；不要将 min/max 写进 attributes。

LayerNorm 默认使用标准 ONNX 分解图，主要过程为 ReduceMean→Sub→Pow→ReduceMean→Add(epsilon)→Sqrt→Div→Mul(scale)→Add(bias)。`accumulation_dtype` 控制统计量和归一化计算精度，`affine_dtype` 控制最后 scale/bias 运算精度，最终输出转换回 X 的 dtype。`affine_dtype` 缺省为 X 的 dtype。不同内部精度会改变舍入，不声称与原 OMC 逐位一致。

要测试原生 `LayerNormalization`，设置 `implementation="native"`、`opset>=17`、`accumulation_dtype="float32"`，删除 `affine_dtype` 或将其改为输入 dtype。`attributes.stash_type` 可指定 native 的内部精度；该选项受 ONNX schema/runtime/DDK 支持范围限制。只生成主输出 Y，不将原 OMC 的辅助输出强行映射为 ONNX Mean/InvStdDev。

## 5. DDK 与 GemmD / GemmV3

`GemmD` 是请求的算子家族和输出目录名，ONNX 节点是标准 `Gemm`。到底生成 GemmD、GemmV3 或其他实现，由 DDK、平台、shape、dtype、转置、常量和编译选项共同决定。

本机 `DDK-tools-next-6.1.1.0 + kirin9030` 将 handfree 默认 Gemm 配置编译为 **GemmV3**。脚本不会把 GemmV3 文件改名成“已验证 GemmD”来掩盖这个事实。到公司用对应 DDK 编译后，读取同名 JSON：

```text
compile.actual_omc.types
compile.actual_omc.nodes[*].type
compile.actual_omc.nodes[*].inputs
compile.actual_omc.nodes[*].attributes
```

想严格要求产物包含 GemmD，可在该 case 增加 `"expect_omc_type": "GemmD"`。这只用于核验，不会强制编译器选择实现。

编译配置示例：

```json
{
  "compiler": {
    "platform": "kirin9030",
    "weight_data_type": "FP16",
    "use_origin_format": true,
    "io_types": "model",
    "timeout": 300,
    "extra_args": [],
    "env": {},
    "version_label": "company-DDK-version"
  },
  "cases": []
}
```

将 `cases` 替换为实际配置后运行。DDK 可由 `--ddk-root`、`compiler.ddk_root`、`DDK_PATH` 或 `DDK_INSTALL_PATH` 指定。`--platform` 可覆盖平台，`--ge-proto` 可指定该版本 DDK 的 `ge_ir_pb2.py`。默认 schema 路径为 `tools/tools_ascendc/package/python/opc_tool/ge_ir_pb2.py`。

`io_types="model"` 根据每个 ONNX 输入/输出分别生成 `--input_type` / `--output_type`，保留 CastT 的 FP32→FP16 接口。`io_types="auto"` 不传这两个参数，交由编译器决定；这样可能引入接口 Cast 或消除原 Cast。`weight_data_type=null` 可省略权重类型选项。

脚本按本机已验证方式配置 DDK_INSTALL_PATH、DDK_PATH、TOOLCHAIN_HOME、CCE_LIB_DIR、PATH、LD_LIBRARY_PATH、PYTHONPATH、SOC_VERSION。`compiler.env` 在这些设置之后覆盖。默认检测 `/lib64/ld-linux-x86-64.so.2` 来启动 OMG；其他环境可设置 `compiler.loader=null` 直接执行，或指定其他 loader。`compiler.omg` 可指定不同相对位置的 OMG 可执行文件。

公司已有完整封装命令时，可以通过 `compiler.command` 提供**完整 argv 数组**，例如：

```json
{
  "command": [
    "/path/to/company_omg_wrapper",
    "--model", "{model}",
    "--output", "{output}",
    "--platform", "{platform}"
  ],
  "extra_args": [],
  "ge_proto": "/path/to/matching/ge_ir_pb2.py"
}
```

占位符支持 `{model}`、`{output}`、`{platform}`、`{ddk_root}`。自定义 command 时不自动追加默认 OMG 选项，由 wrapper 或数组提供；`extra_args` 仍追加。不要使用 shell 管道/重定向字符串。若 wrapper 内含文字花括号，用 `{{` / `}}` 转义。

编译器指纹记录可执行文件/schema/config/主要环境标识，不能证明整个 DDK 安装树未变化。在原目录替换 DDK 的库、算子实现或公司 wrapper 后，设置新的 `version_label` 并使用新的输出目录，保证重新编译。

## 6. 已知适配与验证边界

1. 本机 OMG 不支持常量左乘 `W @ X`。默认 MatMul 使用 `(X.T @ W.T).T`，W 在导出时预转置为 initializer，保留原外部 shape；会多出两个 Transpose/Permute。`constant_left_rewrite=false` 可关闭，是否可编译由 DDK 决定。
2. 原 OMC Gemm/Conv 部分 bias 描述为 FP32，但标准 ONNX 对这些操作数要求同类型，默认测试 bias 使用 FP16。实际编译 bias dtype 另见 OMC 审计。
3. 本机 LayerNorm 融合要求 epsilon 为 FP32，默认 `accumulation_dtype=float32`、`affine_dtype=float32`；外部接口仍用 handfree 的 FP16 和原 shape。这样 scale/bias 可一起融合为一个 LayerNorm，另有布局转换节点。
4. Add 可能编译为 Eltwise，MatMul/BatchMatMul 可能为 V2 版本；布局转换节点也可能出现在 OMC 中。不能假定一个 ONNX 节点只对应一个设备计算节点。
5. 没有从 OMC 的 blocked 权重中恢复原始训练数值，也没有声称恢复了原始完整 ONNX。`source` 记录保留恢复依据和适配。
6. OMC 解析器支持经验证的未加密 IMOD/GE 容器。遇到新格式或不同 schema 会失败并留日志；不会用字符串扫描猜测成功。
7. 本工具不测量设备执行时间和功耗。CPU 试运行、OMG 编译和 OMC 结构审计不能代替板端数值/性能验证。测试算子任意新参数时，应核对 actual_omc.nodes 后再做功耗测量。

## 7. 自检

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

自检覆盖 13 类默认值、Gemm 转置和缩放、矩阵广播、常量左乘改写、卷积/反卷积、归约、LayerNorm 数值、dtype、文件命名、修改保护和编译失败隔离。实际 OMG 编译结果另见工具包的 `VALIDATION.md`。
