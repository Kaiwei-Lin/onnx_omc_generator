"""The user's planned shape matrix, with explicit constants and fixed attributes."""
import copy
from pathlib import Path

from generate_ops import load_config, model_config, digest, operator_config, save_json, validate_case

HERE = Path(__file__).resolve().parent
BASE = {c["op"]: c for c in load_config(HERE / "defaults_handfree.json")[0]}


def build_case(op, p):
    c = copy.deepcopy(BASE[op])
    x = c["inputs"]
    if op in {"MatMul", "GemmD", "BatchMatMul"}:
        m, n, k = (p[v] for v in ("M", "N", "K"))
        batch = [p["B"]] if op == "BatchMatMul" else []
        x[0]["shape"], x[1]["shape"] = batch + [m, k], batch + [k, n]
        if op == "GemmD":
            x[2]["shape"] = [n]
    elif op in {"Convolution", "ConvTranspose"}:
        ci, co, h, w = (p[v] for v in ("Cin", "Cout", "H", "W"))
        group = p.get("G", 1)
        kernel = p.get("kernel", [3, 3])
        x[0]["shape"] = [1, ci, h, w]
        x[1]["shape"] = ([co, ci // group] if op == "Convolution" else [ci, co // group]) + kernel
        x[2]["shape"] = [co]
        c["attributes"].update(group=group, kernel_shape=kernel,
                               strides=p.get("stride", [1, 1] if op == "Convolution" else [1, 2]),
                               pads=[0, 0, 0, 0], dilations=[1, 1])
    else:
        x[0]["shape"] = list(p["a"])
        if op in {"Add", "Mul", "PRelu"}:
            x[1]["shape"] = list(p.get("b", p["a"]))
            if op != "PRelu":
                x[1].pop("init", None)
                if p.get("constant", False):
                    x[1]["init"] = {"kind": "normal", "std": 0.1}
        if op in {"Softmax", "LayerNorm"}:
            axis = p["axis"] % len(p["a"])
            c["attributes"]["axis"] = axis
            if op == "LayerNorm":
                for t in x[1:]:
                    t["shape"] = [1] * axis + p["a"][axis:]
        if op == "ReduceSum":
            c["attributes"]["axes"] = [p["axis"]]
    validate_case(c)
    return c


def plan():
    groups = []

    def add(op, slug, title, parameters, grow):
        members = []
        for p in parameters:
            c = build_case(op, p)
            members.append({"parameters": p, "config": operator_config(c), "hash": digest(model_config(c))[:12]})
        groups.append({"id": op + "_" + slug, "op": op, "title": title, "grow": grow, "cases": members})

    def mm(op, slug, title, values, fixed, axes):
        pars = [{**fixed, **(dict(zip(axes, v)) if isinstance(v, (list, tuple)) else {axes[0]: v})} for v in values]
        add(op, slug, title, pars, axes)

    mm("MatMul", "small_m", "小 M、大 N、大 K", [1,2,4,8,16,32,64,128], dict(N=512,K=4096), ["M"])
    mm("MatMul", "small_n", "小 N、大 M、大 K", [1,2,4,8,16,32,64,128], dict(M=514,K=6144), ["N"])
    mm("MatMul", "k", "固定 M/N 扫描 K", [128,256,512,1024,2048,4096,6144,8192], dict(M=514,N=2), ["K"])
    mm("MatMul", "square", "规则大矩阵", [(v,v) for v in [32,64,96,128,192,256,384,512]], dict(K=1024), ["M","N"])
    mm("MatMul", "small_k", "大 M/N、小 K", [8,16,32,64,96,128,192,256], dict(M=1024,N=1024), ["K"])
    mm("MatMul", "long_reduce", "小 M/N、长归约", [256,512,1024,2048,3072,4096,6144,8192], dict(M=8,N=8), ["K"])
    mm("MatMul", "same_flops", "相同 FLOPs、不同矩阵比例", [(8,1024,8192),(1024,8,8192),(16,1024,4096),(1024,16,4096),(64,1024,1024),(256,256,1024),(512,512,256),(1024,1024,64)], {}, ["M","N","K"])
    align = [31,32,33,63,64,65,127,128,129,255,256,257]
    mm("MatMul", "alignment", "Tile 对齐边界", align, dict(N=256,K=1024), ["M"])
    mm("GemmD", "small_m", "小 M、宽输出", [1,4,8,16,32,64,128,256], dict(N=768,K=288), ["M"])
    mm("GemmD", "n", "输出 N 扩展", [16,32,64,128,256,384,512,768], dict(M=128,K=288), ["N"])
    mm("GemmD", "k", "归约 K 扩展", [16,32,64,128,256,288,512,1024], dict(M=128,N=768), ["K"])
    mm("GemmD", "alignment", "非对齐 N", align, dict(M=128,K=288), ["N"])
    mm("BatchMatMul", "batch", "Batch 数量", [1,2,4,8,16,32,64,128], dict(M=2,N=256,K=256), ["B"])
    mm("BatchMatMul", "m", "单批 M", [1,2,4,8,16,32,64,128], dict(B=8,N=256,K=256), ["M"])
    mm("BatchMatMul", "n", "输出 N", [16,32,64,128,256,384,512,1024], dict(B=8,M=2,K=256), ["N"])
    mm("BatchMatMul", "k", "归约 K", [32,64,128,192,256,384,512,1024], dict(B=8,M=2,N=256), ["K"])
    mm("BatchMatMul", "same_flops", "固定 B×M=128", [(1,128),(2,64),(4,32),(8,16),(16,8),(32,4),(64,2),(128,1)], dict(N=256,K=256), ["B","M"])

    channels = [8,16,32,48,64,96,128,256]
    groups_g = [1,2,3,4,6,8,16,96]
    kernels = [[1,1],[1,3],[3,1],[3,3],[5,1],[5,3],[3,5],[5,5]]
    add("Convolution", "width", "原始长宽比", [dict(Cin=96,Cout=96,H=5,W=v,kernel=[5,1]) for v in [16,32,64,128,256,512,768,1024]], ["W"])
    add("Convolution", "cin", "输入通道", [dict(Cin=v,Cout=96,H=32,W=32) for v in channels], ["Cin"])
    add("Convolution", "cout", "输出通道", [dict(Cin=96,Cout=v,H=32,W=32) for v in channels], ["Cout"])
    add("Convolution", "kernel", "Kernel 几何", [dict(Cin=96,Cout=96,H=32,W=32,kernel=v) for v in kernels], ["H","W"])
    add("Convolution", "group", "Group / Depthwise", [dict(Cin=96,Cout=96,H=32,W=32,G=v) for v in groups_g], ["H","W"])
    add("Convolution", "aspect", "H×W=8192", [dict(Cin=96,Cout=96,H=h,W=w) for h,w in [(8,1024),(16,512),(32,256),(64,128),(128,64),(256,32),(512,16),(1024,8)]], ["H","W"])
    add("ConvTranspose", "spatial", "原始空间规模", [dict(Cin=16,Cout=1,H=v,W=v,kernel=[1,3]) for v in [8,16,32,64,96,128,192,256]], ["H","W"])
    add("ConvTranspose", "cout", "输出通道", [dict(Cin=16,Cout=v,H=64,W=64) for v in [1,2,4,8,16,32,64,128]], ["Cout"])
    add("ConvTranspose", "stride", "上采样倍率", [dict(Cin=16,Cout=1,H=64,W=64,stride=list(v)) for v in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,3),(4,4)]], ["H","W"])
    add("ConvTranspose", "kernel", "Kernel 几何", [dict(Cin=16,Cout=16,H=64,W=64,kernel=v) for v in [[1,1],[1,3],[3,1],[3,3],[3,5],[5,3],[5,5],[7,7]]], ["H","W"])
    add("ConvTranspose", "group", "分组反卷积", [dict(Cin=96,Cout=96,H=32,W=32,G=v) for v in groups_g], ["H","W"])

    small = [32,64,128,256,512,1024,2048,4096]
    large = [8192,32768,65536,131072,262144,524288,2097152,8388608]
    for op in ("Add", "Mul", "Activation"):
        for label, sizes in (("small",small),("large",large)):
            for const in ((False,True) if op=="Add" else (False,)):
                add(op,label+("_constant" if const else ""),label+("：一方常量" if const else ""),
                    [dict(a=[e],constant=const) for e in sizes], ["elements"])
    add_b = [[1],[1,1,1,1],[1,16,1,1],[1,1,256,1],[1,1,1,128],[1,16,256,1],[1,16,1,128],[1,16,256,128]]
    for const in (False,True):
        add("Add","broadcast"+("_constant" if const else ""),"广播"+("：一方常量" if const else ""),
            [dict(a=[1,16,256,128],b=b,constant=const) for b in add_b],["tensor"])
    mul_b = [([1],False),([1,1,1,1],False),([1,256,1,1],False),([1,1,2048,1],False),([],True),([1,256,2048,1],False),([1,256,1,1],True),([1,1,2048,1],True)]
    add("Mul","broadcast","广播与可复用常量",[dict(a=[1,256,2048,1],b=b,constant=const) for b,const in mul_b],["tensor"])
    add("Activation","shape","同元素数不同 rank",[dict(a=a) for a in [[524288],[512,1024],[1024,512],[1,16,256,128],[1,256,2048,1],[1,32,128,128],[1,64,128,64],[1,128,64,64]]],["tensor"])
    add("Activation","alignment","L 对齐边界",[dict(a=[2048,l]) for l in align],["tensor"])
    add("PRelu","scalar","标量 slope",[dict(a=[e],b=[1]) for e in [128,512,2048,8192,32768,131072,524288,2097152]],["elements"])
    add("PRelu","channel","通道 slope",[dict(a=[1,c,64,128],b=[1,c,1,1]) for c in channels],["tensor"])
    add("PRelu","broadcast","slope 广播粒度",[dict(a=[1,64,64,128],b=b) for b in [[1],[1,1,1,1],[1,64,1,1],[1,1,64,1],[1,1,1,128],[1,64,64,1],[1,64,1,128],[1,64,64,128]]],["tensor"])
    add("ReduceSum","short","短归约",[dict(a=[4096,l],axis=1) for l in [1,2,4,8,16,32,64,128]],["tensor"])
    add("ReduceSum","long","长归约",[dict(a=[16,l],axis=1) for l in [128,256,512,1024,2048,4096,8192,16384]],["tensor"])
    add("ReduceSum","fixed_elements","固定 262144 元素",[dict(a=[262144//l,l],axis=1) for l in [8,16,32,64,128,256,512,1024]],["tensor"])
    add("ReduceSum","axis0","原始 axis=0",[dict(a=[r,65536,1],axis=0) for r in [1,2,4,8,16,32,64,128]],["tensor"])
    add("ReduceSum","direction","连续/跨步归约",[dict(a=[r,l] if axis==1 else [l,r],axis=axis) for r,l in [(64,1024),(256,256),(1024,64),(4096,16)] for axis in (1,0)],["tensor"])
    add("LayerNorm","short","短归一化",[dict(a=[1024,l],axis=1) for l in [8,16,32,64,128,192,256,384]],["tensor"])
    add("LayerNorm","long","长归一化",[dict(a=[16,l],axis=1) for l in [128,256,512,1024,2048,4096,8192,16384]],["tensor"])
    add("LayerNorm","fixed_elements","固定 65536 元素",[dict(a=[65536//l,l],axis=1) for l in [16,32,64,128,256,512,1024,4096]],["tensor"])
    add("LayerNorm","rank","原始及等价布局",[dict(a=a,axis=axis) for a,axis in [([256,64,1,1],1),([256,64,1],1),([256,64],1),([1,256,64],2),([1,256,64,1],2),([1,1,256,64],3),([16,16,64],2),([4,64,64],2)]],["tensor"])
    add("LayerNorm","alignment","归一化长度对齐",[dict(a=[256,l],axis=1) for l in align],["tensor"])
    add("Softmax","short","短行",[dict(a=[4096,l],axis=1) for l in [8,16,32,64,96,128,192,256]],["tensor"])
    add("Softmax","long","长行",[dict(a=[16,l],axis=1) for l in [128,256,512,1024,2048,4096,8192,16384]],["tensor"])
    add("Softmax","fixed_elements","固定 262144 元素",[dict(a=[262144//l,l],axis=1) for l in [8,16,32,64,128,256,512,1024]],["tensor"])
    add("Softmax","four_d","原始四维",[dict(a=[1,4096,l,1],axis=2) for l in [16,32,48,64,96,128,256,512]],["tensor"])
    add("Softmax","alignment","归约边界",[dict(a=[2048,l],axis=1) for l in align],["tensor"])
    return groups


if __name__ == "__main__":
    groups = plan()
    cases = {c["hash"]: c["config"] for group in groups for c in group["cases"]}
    save_json(HERE / "verification/shape_plan.json", groups)
    save_json(HERE / "configs/shape_cases.json", list(cases.values()))
    print(f"{len(groups)} groups, {sum(len(g['cases']) for g in groups)} memberships, {len(cases)} unique configurations")
