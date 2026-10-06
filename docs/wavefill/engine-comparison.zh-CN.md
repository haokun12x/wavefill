# Wavefill 与 vLLM、SGLang、NInfer 的同条件对比实验报告

> 硬件：单卡 GeForce RTX 5090（32 GB，600 W 功耗墙）
> 模型：Qwen3.8-27B NVFP4（unsloth checkpoint；NInfer/Wavefill 使用同一份权重的 `.ninfer` 转换）
> 被测引擎：Wavefill（本仓库 `main`）、NInfer `9e163eee`（上游）、vLLM 0.27.1、SGLang 0.5.17
> 日期：2026-10-03 至 2026-10-04。原始数据在 [`results/engine-compare/`](results/engine-compare/)，脚本在 [`tools/wavefill/engine-compare/`](../../tools/wavefill/engine-compare/)。
> 开发过程借助了 AI 编程助手（Claude Code）。

## 摘要

[技术报告](technical-report.zh-CN.md)只回答了 Wavefill 相对上游 NInfer 快多少。这份报告回答另一个问题：**在同一块卡、同一份权重、同样的请求下，它和主流服务框架相比如何？** 四个引擎都通过 HTTP Chat Completions 接收完全相同的消息，计时在客户端，包含模板渲染、分词和 HTTP 开销，也就是 agent 客户端实际等待的时间。

| 测试（中位数） | **Wavefill** | NInfer `9e163eee` | vLLM 0.27.1 | SGLang 0.5.17 |
|---|---:|---:|---:|---:|
| 冷启动到首 token，32k prompt | **3.97 s** | 4.38 s | 4.29 s | 4.16 s |
| 冷启动到首 token，64k prompt | **10.00 s** | 11.47 s | 11.01 s | 10.56 s |
| 冷启动到首 token，128k prompt | **28.5 s** | 34.3 s | 31.7 s | 不支持¹ |
| agent 单轮，64k 历史 + 128 token | **162 ms** | 192 ms | 221 ms | 338 ms |
| agent 单轮，64k 历史 + 512 token | **230 ms** | 272 ms | 418 ms | 453 ms |
| agent 单轮，64k 历史 + 4096 token | **1071 ms** | 1271 ms | 1401 ms | 1338 ms |
| decode，短 prompt，512 token | **77.6 tok/s** | 76.9 | 56.5 | 41.3 |

¹ SGLang 在这块卡上能分出的 BF16 KV 池只有 85,007 token，128k 的请求被拒绝（HTTP 400）。

**Wavefill 在所有测试里都最快。** 相对每一项里第二快的引擎：冷启动快 1.05–1.11×，agent 单轮快 1.18–1.19×（这几项第二快的都是上游 NInfer），decode 比 vLLM 快 1.37×。

用两个简单模型拆解这些数字，可以得到三个结论：

1. **冷启动的差距几乎全部来自注意力。** 用 $t(N) = aN + bN^2$ 拟合每个引擎的三个点，误差都在 1% 以内。四个引擎的线性系数 $a$（GEMM、GDN 等）只差 7%；二次项反推出的注意力有效吞吐却从 142 到 199 TFLOP/s 不等，Wavefill 最高（§4.1）。
2. **agent 单轮的差距一半在注意力，一半在“复用粒度”。** NInfer/Wavefill 在每轮结束处保存精确的检查点，每轮只重算新增的 d + 15 个 token。vLLM 对混合架构模型只能按 784 token 的整块复用，每轮平均多算约 390 个 token。一个“按整块复用”的模型解释了 vLLM 9 个单轮延迟 99.6% 的方差（r = 0.998）（§4.2）。
3. **decode 的差距在带宽利用率。** 四个引擎每个 token 读的权重字节数相同，带宽下界都是 11.4 ms。Wavefill/NInfer 用到 88%，vLLM 64%，SGLang 47%（§4.3）。

这份对比只覆盖**单请求**场景。多请求并发吞吐是 vLLM/SGLang 的设计重点，这里没有测（§5）。

---

## 1. 实验目的与问题

agent 的工作负载有三个特点（详见技术报告 §4.1）：

- 上下文单调增长到 64k–128k；
- 步与步严格串行，每一步的到首 token 时间直接累加进任务时长；
- 冷启动（整段长 prompt）和热增量（在长历史上追加 d 个 token）两种 prefill 并存。

所以本实验测三类指标：

| 指标 | 对应 agent 的哪个环节 | 主要受什么限制 |
|---|---|---|
| 冷启动到首 token（32k/64k/128k） | 首轮、缓存未命中、上下文压缩之后 | GEMM（线性项）+ 注意力（二次项） |
| agent 单轮到首 token（64k 历史 + d） | 每一步工具调用之后 | 前缀复用粒度 + 短查询 × 长 key 的注意力 + 服务路径开销 |
| decode 速度 | 生成工具调用和回复 | DRAM 带宽 |

## 2. 实验设置

### 2.1 公平性控制

| 项目 | 设置 | 说明 |
|---|---|---|
| 权重 | 同一个 unsloth NVFP4 checkpoint | FP8（逐通道）+ NVFP4（16 元素块缩放）混合，每 token 要读的权重字节数相同 |
| KV 缓存 | 四个引擎都是 BF16 | vLLM 的 `--kv-cache-dtype auto` 会按 checkpoint 的 `kv_cache_scheme` 选 FP8，所以显式设为 `bfloat16`；FP8 的结果单列在 §3.3 |
| 并发 | 1 个请求 | `--max-concurrency 1` / `--max-num-seqs 1` / `--max-running-requests 1` |
| 投机解码 | 全部关闭 | |
| 前缀缓存 | 全部开启 | vLLM `--enable-prefix-caching`（混合模型自动进入 mamba `align` 模式），SGLang radix cache，NInfer 端点检查点 |
| 最大上下文 | 142,336 token | |
| 请求 | 完全相同的 messages；`chat_template_kwargs.enable_thinking=false`，温度 0 | 文本由模型自己的分词器把 token 序列解码而来，重新分词后长度一致（每个引擎报告的 `prompt_tokens` 相同） |
| 计时 | 客户端墙钟，`max_tokens=1` | 等于到首 token 时间，包含模板、分词、HTTP |
| 测试顺序 | 每个引擎单独启动、测完、关闭，GPU 上没有其他负载 | 每 0.5 s 采一次 SM 频率和功耗 |

### 2.2 各引擎的配置

| 引擎 | 关键配置 | 注意力内核 |
|---|---|---|
| Wavefill / NInfer | `ninfer-serve --kv-dtype bf16 --kv-capacity auto --max-concurrency 1 --prefill-chunk 1024` | NInfer 自己的 FA2 式内核；Wavefill 加了 cluster split-KV 和 FP16 P·V |
| vLLM | `--kv-cache-dtype bfloat16 --kv-cache-memory-bytes 9GiB --max-num-batched-tokens 2048 --no-enable-flashinfer-autotune`，图像/视频输入限制为 0 | FlashAttention 2（vLLM 在 BF16 KV 时自动选择） |
| SGLang | `--kv-cache-dtype bf16 --mem-fraction-static 0.94 --attention-backend flashinfer --fp4-gemm-backend flashinfer_cudnn`，chunked prefill 2048，decode 用完整 CUDA Graph | FlashInfer |

两处调整需要说明：

- **vLLM 关闭了 FlashInfer FP4 自动调优。** 在 32 GB 的卡上，调优在 2048 batched token 时显存不够；即使把 KV 缓存压到 1 GiB 也一样。关闭后，vLLM 的 FP4 GEMM 用默认 tactic，可能比调优后的略慢。§4.1 里 vLLM 的线性系数比其他引擎高 5–7%，上限也就在这个量级。注意力内核不受影响。
- **SGLang 的 `mem-fraction-static` 是 0.94。** 0.97 在预热时显存不够，0.94 能分出 85,007 token 的 BF16 KV 池，所以 128k 的测试无法进行。

### 2.3 测试流程（[`bench_engines.py`](../../tools/wavefill/engine-compare/bench_engines.py)）

- **冷启动**：先用 2k 和 8k 的请求做全局预热（JIT、Graph 捕获）。然后每个长度测 2 次，每个请求开头加一行唯一标记，防止前缀缓存命中。
- **agent 单轮**：先发一个 64k 的请求作为历史（不计时），之后每轮把上一轮的回复和 d 个新 token 追加进对话。先预热 1 轮，再计时 3 轮。每轮的 `prompt_tokens` 和 `cached_tokens` 都记录下来。
- **decode**：一个 38 token 的短 prompt，流式输出 512 个 token（vLLM/SGLang 设 `ignore_eos`），用首尾 token 的时间差算 tok/s。测 3 次，丢掉第 1 次。

## 3. 结果

### 3.1 主表

见摘要。重复测量之间的差异：

- 冷启动：同一引擎两次之间不超过 1%；
- decode：两次之间不超过 0.1%；
- agent 单轮：Wavefill/NInfer 3 轮之间差 1–8%；vLLM 同一格内最慢和最快相差到 1.86×，原因见 §4.2。

### 3.2 相对倍数

| 测试 | Wavefill 相对 NInfer | 相对 vLLM | 相对 SGLang |
|---|---:|---:|---:|
| 冷启动 32k / 64k / 128k | 1.10 / 1.15 / 1.20× | 1.08 / 1.10 / 1.11× | 1.05 / 1.06× / — |
| 单轮 +128 / +512 / +4096 | 1.18 / 1.18 / 1.19× | 1.36 / 1.82 / 1.31× | 2.08 / 1.97 / 1.25× |
| decode | 1.01× | 1.37× | 1.88× |
| decode 测试的首 token（38 token prompt） | 34 ms 对 34 ms | 34 对 89 ms | 34 对 94 ms |

这里 Wavefill 相对 NInfer 的冷启动倍数（1.10–1.20×），与技术报告 §8.3 用 `ninfer_bench`（BF16 KV + MTP3）测的 1.06–1.18× 是同一量级。两者的服务路径和投机设置不同，所以数字不完全一样。

### 3.3 vLLM 的默认配置：FP8 KV

不改 `--kv-cache-dtype`，vLLM 会按 checkpoint 选 FP8 KV。这时注意力内核换成 FlashInfer，块大小变成 1568 token：

| 测试 | vLLM，BF16 KV（主表） | vLLM，FP8 KV（默认） |
|---|---:|---:|
| 冷启动 32k / 64k / 128k | 4.29 / 11.01 / 31.7 s | 4.55 / 12.01 / 36.0 s |
| 单轮 +128 / +512 / +4096 | 221 / 418 / 1401 ms | 206 / 453 / 1530 ms |
| decode | 56.5 tok/s | 57.0 tok/s |

FP8 KV 没有让 prefill 变快，反而更慢：注意力有效吞吐从 175 降到 142 TFLOP/s（§4.1）。decode 也没有变快，因为 decode 的字节数由权重主导，KV 只占很小一部分。

### 3.4 GPU 状态

| 引擎 | SM 频率中位数 | 功耗中位数 |
|---|---:|---:|
| Wavefill | **2640 MHz** | 600 W |
| NInfer | 2745 MHz | 577 W |
| vLLM（BF16 KV） | 2752 MHz | 599 W |
| vLLM（FP8 KV） | 2767 MHz | 543 W |
| SGLang | 2677 MHz | 599 W |

只统计功耗超过 400 W 的采样，也就是负载期间的采样。Wavefill 的频率最低，但在所有测试里最快。它的优势来自每个时钟周期做了更多有效工作（空转的 SM 更少，P·V 走双倍吞吐的累加路径），而不是更高的频率。这与技术报告 §8.5 的功耗墙分析一致：长 prefill 期间卡持续顶在 600 W，张量核心越忙，频率越低。

## 4. 分析

### 4.1 冷启动：线性部分几乎一样，差距在注意力

技术报告 §2.4 推导过，N token 的 prompt 分块 prefill 的时间是

$$t(N) = aN + bN^2,\qquad b = \frac{196{,}608}{R_\text{attn}}$$

- $a$ 是每 token 的线性开销（GEMM、48 层 GDN、归一化等）；
- 二次项来自 16 层因果注意力：每对 (query, key) 在 16 层里共 393,216 FLOP，因果对数约 $N^2/2$；
- $R_\text{attn}$ 是注意力在整段 prefill 上的平均有效吞吐。

对每个引擎，用 32k、64k、128k 三个点做两参数最小二乘拟合（[`analyze.py`](../../tools/wavefill/engine-compare/analyze.py)）：

| 引擎 | 注意力内核 | $a$（µs/token） | $R_\text{attn}$（TFLOP/s） | 拟合最大偏差 | 交叉点 $N^* = aR/196608$ |
|---|---|---:|---:|---:|---:|
| **Wavefill** | NInfer + split-KV + FP16 PV | 87.8 | **199** | 0.5% | 89k |
| NInfer | NInfer FA2 式 | 89.4 | 150 | 0.9% | 68k |
| vLLM（BF16 KV） | FlashAttention 2 | 94.2 | 175 | 0.2% | 84k |
| vLLM（FP8 KV） | FlashInfer | 92.6 | 142 | 0.4% | 67k |
| SGLang | FlashInfer | 92.8 | 189 | 只有 2 个点，精确解 | 89k |

几点观察：

1. **两参数模型对每个引擎都成立**，三个点的偏差都在 1% 以内。这说明“线性部分 + 二次注意力”这个分解对四个完全不同的实现都适用。
2. **线性部分几乎一样**：$a$ 只在 87.8–94.2 µs/token 之间。线性部分的主体是 FP8/NVFP4 GEMM。prefill 时它们都是计算受限的大矩阵乘，各家实现都能接近张量核心峰值，结果殊途同归。
3. **差距几乎全部来自注意力。** $R_\text{attn}$ 从 142 到 199，相差 1.4×。
4. **这个拟合可以和内核测量互相印证。**
   - NInfer 拟合出的 150 TFLOP/s，与技术报告 §4.4 在 1024 token 分块、64k 上下文下单独测的 151 TFLOP/s 一致。
   - Wavefill 与 NInfer 的比值 199/150 = 1.33，与技术报告 §8.5 引擎内实测的注意力提速 1.32×（受功耗墙限制）一致。
5. **SGLang 和 vLLM 的注意力为什么在 NInfer 和 Wavefill 之间？** 它们的 prefill 分块是 2048 token，不是 1024。按技术报告 §4.4 的波次模型，2048 token 的分块有 768 个 tile，波次效率是 90.4%，而 1024 token 只有 75.3%。也就是说，**更大的分块本身就缓解了波次量化**，这正是 Wavefill 用 split-KV 在 1024 分块上解决的问题。Wavefill 在更小的分块、更低的频率下，有效吞吐仍然最高。
6. **交叉点 $N^*$** 是注意力开始占 prefill 一半以上时间的长度。Wavefill 把它从 68k 推迟到 89k。

### 4.2 agent 单轮：复用粒度和注意力各占一半

每一轮要重新 prefill 的 token 数等于 `prompt_tokens − cached_tokens`。四个引擎在这一点上的设计差别很大：

| 引擎 | 前缀复用的粒度 | 每轮实际重算（d = 128 时） |
|---|---|---:|
| Wavefill / NInfer | 每轮结束处保存精确的端点检查点（KV + GDN 状态） | 143 = d + 15（15 是对话模板的 token） |
| SGLang | 实测命中长度全是 64 的整数倍 | 154–203 |
| vLLM（BF16 KV） | 784 token 的整块 | 154–795（模型推算） |
| vLLM（FP8 KV） | 1568 token 的整块 | 154–1579（模型推算） |

**vLLM 为什么是 784 token 的整块？** Qwen3.8 是混合架构，48 层 GDN 的递推状态不能截断，只能在保存过的边界上恢复。vLLM 的做法是让注意力 KV 页和 mamba 状态页共用同一个分配器，于是把注意力块放大到与状态页一样大（日志：“Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size”），并且只在块边界上保存状态（`align` 模式）。所以前缀只能按整块命中：

$$\text{重算} = P - \left\lfloor \frac{P_\text{prev}}{784} \right\rfloor \cdot 784$$

其中 $P$ 是本轮的 prompt 长度，$P_\text{prev}$ 是上一轮的 prompt 长度。vLLM 不报告 `cached_tokens`，所以这里用这个公式推算每轮的重算量，再对延迟做线性拟合：

| 引擎 | 拟合：延迟 = c + k · 重算 token 数 | 相关系数 r | 最大残差 |
|---|---|---:|---:|
| **Wavefill** | c = 118 ms，**k = 231 µs/token** | 0.999 | 28 ms |
| NInfer | c = 138 ms，k = 277 µs/token | 0.999 | 53 ms |
| vLLM（BF16 KV，整块模型） | c = 147 ms，k = 270 µs/token | **0.998** | 63 ms |
| vLLM（FP8 KV，整块模型，块 = 1568） | c = 93 ms，k = 316 µs/token | 0.996 | 94 ms |
| SGLang | c = 288 ms，k = 258 µs/token | 0.995 | 81 ms |

整块模型对 vLLM 的 9 个点给出 r = 0.998。看具体数据就很清楚：d = 512 的三轮推算重算 780 / 1306 / 1048 个 token，实测 316 / 497 / 418 ms，顺序完全一致。所以 vLLM 单轮延迟的大幅波动不是测量噪声，而是复用粒度造成的。按 k = 270 µs/token 算，**平均多算的约 392 个 token 每轮要多付约 106 ms**。

这张表还能拆出另外两件事：

- **k 是在 66k 历史上每多 prefill 一个 token 的边际代价**，主要由“短查询 × 长 key”的注意力决定（技术报告 §2.5）。Wavefill 的 231 µs/token 最低，比 NInfer 低 17%。这就是 split-KV 在短增量上把波次填满的效果。
- **c 是每轮的固定开销**，包含模板渲染、分词、调度，以及小增量时 GPU 利用率低的部分。SGLang 的 c = 288 ms 明显高于其他引擎，这是它在 +128 时最慢的原因。我们没有拆解这部分开销。

**Wavefill 自己还剩下的空间。** 技术报告 §8.4 直接调用 Engine API，64k 历史 + 128 token 的单轮是 65 ms。走 HTTP 服务路径是 162 ms，服务端日志里的 TTFT 是 159–160 ms（其中 queue 16–17 ms）。所以有约 95 ms 花在服务路径上，而不是注意力内核上：例如每轮对 66k token 的完整对话重新做模板渲染和分词、前缀匹配、恢复端点状态。这部分还没有拆解，是下一步在 agent 场景下最值得做的优化。

### 4.3 decode：同样的字节，不同的带宽利用率

四个引擎每生成一个 token 都要读 19.05 GB 权重和约 0.30 GB 的 GDN 状态，按实测 1.70 TB/s 的读带宽，下界是 11.4 ms（技术报告 §2.2）：

| 引擎 | tok/s | ms/token | 达到下界的比例 | 超出下界 |
|---|---:|---:|---:|---:|
| **Wavefill** | 77.6 | 12.9 | **88.3%** | 1.5 ms |
| NInfer | 76.9 | 13.0 | 87.5% | 1.6 ms |
| vLLM | 56.5 | 17.7 | 64.3% | 6.3 ms |
| SGLang | 41.3 | 24.2 | 47.0% | 12.8 ms |

字节数相同，所以差距只能来自带宽利用率和内核之间的空隙。Wavefill 比 NInfer 快的 0.9% 来自 NVFP4 下投影 GEMV 的修改（技术报告 §3.2）。我们**没有**对 vLLM 和 SGLang 做剖析，以下只是根据 roofline 做的定位，不是测量结论：

- vLLM 的 FP8 层走 W8A8（`CutlassFP8ScaledMMLinearKernel`，每层要先动态量化激活），NVFP4 层走 FlashInfer CUTLASS W4A4 GEMM（同样要先量化激活）。这些是为 M ≥ 16 设计的分块 GEMM。M = 1 时，每读一遍权重的工作量不变，但在途读请求是否足够喂饱 DRAM（技术报告 §1.2 的 Little 定律：全卡约 1 MiB），取决于 tactic 选择。而 tactic 自动调优在这次测试里是关掉的。
- NInfer 在 M = 1 时用专门的 GEMV，在途字节、展开深度和取指都针对带宽受限场景调过（技术报告 §3）。

## 5. 局限与有效性威胁

1. **只测了单请求。** vLLM 和 SGLang 的核心优势是连续批处理下的多请求吞吐，这次完全没有涉及。NInfer 的设计上限是 8 个并发请求。所以结论只适用于“单用户、长上下文 agent”的延迟。
2. **样本量小。** 冷启动每点 2 次，单轮每格 3 轮，decode 2 次。好在同一引擎重复测量之间的差异（冷启动 ≤ 1%，decode < 0.1%）远小于引擎之间的差距。
3. **vLLM 关闭了 FP4 自动调优**（32 GB 显存不够）。这可能让它的 GEMM 和 decode 偏慢；对冷启动的影响上限，大约就是它比其他引擎高出的 5–7% 线性系数。
4. **分块大小不同**：NInfer/Wavefill 1024，vLLM/SGLang 2048。更大的分块对 FA2 式内核的波次效率更有利（§4.1），所以这个差异并没有让对手吃亏。
5. **SGLang 没有 128k 的数据**：显存限制，KV 池只有 85k token。
6. **只比较了速度，没有比较输出质量。** Wavefill 的数值正确性见技术报告 §9（FP64 oracle、数值压力测试）。四个引擎的权重格式相同，但各自的内核实现和累加顺序不同，输出不会逐位一致。
7. **各引擎依次测试**，没有交替。每次运行都采样了频率和功耗（§3.4），没有看到异常的降频。

## 6. 结论

1. **在单请求的长上下文场景里，Wavefill 在全部 7 项测试中都最快**：冷启动比第二名快 5–11%，agent 单轮快 18–19%，decode 比 vLLM 快 37%。
2. **冷启动的差距可以精确归因到注意力。** 两参数模型对四个引擎都拟合到 1% 以内：线性部分各家几乎一样，注意力有效吞吐是 142–199 TFLOP/s，Wavefill 最高，而且它的频率最低。
3. **agent 单轮的差距有两个来源。**
   - 前缀复用的粒度：精确的端点检查点对比 784/1568 token 的整块，整块模型以 r = 0.998 解释了 vLLM 的延迟波动；
   - 短增量的注意力效率：Wavefill 每 token 边际代价最低，231 µs。
   对混合架构模型来说，“在哪里保存 GDN 状态”和注意力内核同样重要。
4. **decode 由带宽决定。** 同样的字节，Wavefill/NInfer 用到下界的 88%，vLLM 64%，SGLang 47%。
5. **下一步**：Wavefill 在 HTTP 路径上每轮约有 95 ms 不在 GPU 计算上，这是 agent 场景下剩下的最大一块。

## 附录：复现

```bash
# 需要：本仓库构建出的 ninfer-serve、上游 9e163eee 构建出的 ninfer-serve、vLLM、SGLang 环境
export HF_MODEL_DIR=<unsloth/Qwen3.8-27B-NVFP4 的本地快照>
export NINFER_ARTIFACT=<qwen3_8_27b_nvfp4.ninfer>
export CORPUS_IDS=<空白分隔的 token id，至少 262144 个>
export CLIENT_PY=<装有 tokenizers 的 python>
export WAVEFILL_SERVE=build/apps/ninfer-serve NINFER_SERVE=<upstream>/apps/ninfer-serve
export VLLM=<vllm 可执行文件> SGLANG_PY=<SGLang 环境的 python>
OUT=results tools/wavefill/engine-compare/run_all.sh wavefill ninfer vllm sglang
# 分析（vLLM 的块大小见其日志 "Setting attention block size to ..."）
python3 tools/wavefill/engine-compare/analyze.py --block vllm=784 --block vllm-fp8kv=1568 \
    docs/wavefill/results/engine-compare wavefill ninfer vllm vllm-fp8kv sglang
```

FlashInfer 的 JIT 需要 `nvcc`（vLLM）和 `ninja`（SGLang），两者都要在 `PATH` 里。

**数据**（[`results/engine-compare/`](results/engine-compare/)）：

| 文件 | 内容 |
|---|---|
| `<engine>.json`、`<engine>.client.log` | 每个请求的耗时、`prompt_tokens`、`cached_tokens`、decode tok/s |
| `<engine>.clock.csv` | 每 0.5 s 的 SM 频率、功耗、显存 |
| `server_settings.txt` | 各引擎日志里与配置有关的行（KV 容量、注意力后端、块大小、SGLang 的参数） |

`vllm.*` 是 BF16 KV 的主结果，`vllm-fp8kv.*` 是 vLLM 默认的 FP8 KV。
