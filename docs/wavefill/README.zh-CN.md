# Wavefill 中文概要

Wavefill 是 [NInfer](https://github.com/Neroued/ninfer) 的一个分支，专门加速 RTX 5090 上长 prompt 的 prefill。NInfer 是一个从零写的 C++/CUDA 推理引擎，引擎本身的设计、功能和绝大部分代码都出自上游作者；Wavefill 的改动是上游 `9e163eee` 之上的三个提交。

## 效果（Qwen3.8-27B NVFP4，BF16 KV，MTP3，单卡 RTX 5090）

| | NInfer | Wavefill | 提速 |
|---|---:|---:|---:|
| 32k prompt prefill | 4.25 s | 4.01 s | **1.06×** |
| 64k prompt prefill | 11.29 s | 10.11 s | **1.12×** |
| 128k prompt prefill | 33.8 s | 28.6 s | **1.18×** |
| prompt 注意力内核（64k 上下文，BF16 / INT8 / FP8 KV） | | | **1.40× / 1.29× / 1.35×** |
| 无推测 decode | 76.9 tok/s | 77.9 tok/s | **+1.25%** |

端到端提速比内核提速低，原因是长 prefill 时整卡顶在 600 W 功耗上限：新内核下 SM 频率中位数从 2745 MHz 降到 2595 MHz。这一点已经用 nsys 和逐 100 ms 的功耗采样确认。

## 做了什么

1. **cluster split-KV。** 原来一个 1024 token 的分块是 384 个各占满一个 SM 的 CTA，在 170 个 SM 上要跑 3 波，最后一波只有 26% 满。现在同一块的 key 范围由一个线程块簇里的 S 个 CTA 分担，最后通过分布式共享内存合并，不需要额外的全局显存。S 由主机端根据这块卡实测的簇容量和上下文长度来选。
2. **不会溢出的 FP16 P·V 累加。** FP16 累加的吞吐是 FP32 累加的 2 倍。为了不溢出，P 按每个 64-key 块自己的最大值计算再乘 2⁻⁶，这样每块的部分和不超过 max|V|；折回 FP32 累加器时再乘 exp(m块 − m全局)。所以在有注意力 sink 的长上下文里也不会丢精度。
3. **NVFP4 下投影 GEMV。** 原来全展开的内核代码有 102 KiB，decode 时每次都要冷取指；把循环改成按 4 展开，输出逐位不变。

## 怎么验证的

- NInfer 自带的 FP64 oracle 测试全部通过，并新增了会触发 split 的用例；开发期间还强制 S=2/3/5/8 各跑了一遍全套。
- 专门写了数值压力测试：64k 上下文下的注意力 sink、|V| 最大到 3×10⁴ 等情况，新旧内核误差相当。
- compute-sanitizer 四个工具共 48 次运行，0 错误。
- 对整个引擎做 SASS 对比，确认只有目标内核发生了变化。

详细设计、数据和被否决的方案见 [long-context-attention.md](long-context-attention.md) 和 [nvfp4-down-gemv.md](nvfp4-down-gemv.md)。开发过程借助了 AI 编程助手（Claude Code）。许可证沿用 NInfer 的 Apache 2.0。
