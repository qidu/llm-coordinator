# TRINITY: An Evolved LLM Coordinator — 核心机制笔记 (verified)

> 来源：ICLR 2026, Sakana AI  (arXiv:2512.04695v3, 27 Apr 2026)
> 关键作者：Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, Yujin Tang
> 本笔记基于对原文 HTML 的逐节校验，重点是 Section 3 (机制) + Appendix A.4 (head 架构) + Section 4.7/4.8 (separable 性质 + sep-CMA-ES vs 其他)。

## 1. 设计目标

把"高难度复杂任务的多步求解 + 验证"这两件事从**中心协调器**中卸载出去，扔给外部的 LLM 池。
协调器本身保持**极致轻量**（~0.6B SLM + ≤20K 训练参数，**其中 head ≤10K**），但通过**sep-CMA-ES** 进化训练出非平凡的路由策略。

## 2. 状态机 (Section 3.2)

每一轮 k，系统维护完整对话历史轨迹 $\mathcal{C}_{k-1}=(Q, O_1, \ldots, O_{k-1})$。
协调器产生两个决策：
- $A_k$：从模型池里选哪个 LLM
- $R_k \in \{T, W, V\}$：赋予什么角色

三种角色：

| 角色 | 输入 | 输出 | 终止? |
|---|---|---|---|
| **Thinker (T)** | $\mathcal{C}_{k-1}$ | 高层策略、子任务拆解、对当前解法的批判 | 否 |
| **Worker (W)** | $\mathcal{C}_{k-1}$ | 具体执行（推导、写代码、计算） | 否 |
| **Verifier (V)** | $\mathcal{C}_{k-1}$ | $u_k \in \{$ACCEPT, REVISE$\}$ + 诊断 $\delta_k$ | 是 iff $u_k$=ACCEPT |

**Termination rule** (论文 Eq):
$$\tau = \min\{k \leq K : R_k = V \text{ and } u_k = \texttt{ACCEPT}\}$$
若没有 ACCEPT，则 $\tau = K$。最终答案返回 $O_\tau$。

论文实验默认 **K = 5** 轮 (Section 4.1: "the maximum number of coordination turns to five")。

## 3. 协调器架构 (Section 3.1 + Appendix A.4)

**两层结构**：

1. **SLM backbone** (Qwen3-0.6B)
   - 取**倒数第二层** (second-to-last layer) 的 hidden state
   - 在该层上做 **Singular Value Fine-tuning (SVF)**：对选定的权重矩阵做 SVD，只学习奇异值的 scale，正交矩阵冻结
   - SVF 引入 **9,216 个可训练参数**
   - **重要**：coordinator 不需要生成完整文本，只取 hidden state，所以可以做"早 token"决策

2. **Lightweight head** (Appendix A.4 详细比较了 4 种)
   - 输入：hidden state $h \in \mathbb{R}^{d_h}$ ($d_h$=1024)
   - 输出：logits $z \in \mathbb{R}^{n_a}$ ($n_a = L + 3 = 7+3 = 10$)
   - 论文实验了 4 种架构 + argmax/softmax：

| 架构 | 参数数 ($d_h$=1024, $n_a$=10) | 备注 |
|---|---|---|
| **Linear** | 10,240 | $\mathbf{z} = \mathbf{W} \mathbf{h}$, 无 bias |
| **Low-rank** (r=14) | 14,592 + 140 = 14,732 | $\mathbf{u}=\text{ELU}(\mathbf{U}\mathbf{h})$, $\mathbf{z}=\mathbf{V}\mathbf{u}\cdot\sigma$ |
| **Sparse** | 10,240 + 1024 + 2 = 11,266 | $\mathbf{z} = \mathbf{W}(\mathbf{h} \odot \alpha)$, learnable $\alpha$, top-k mask |
| **Block-diagonal-2** | 5,120 | 把 hidden 维度分成 2 块 |
| **Block-diagonal-10** | **1,024** | 每个 logit 一块，$z_j = w_j^\top h_j$ |

**关键观察 (Section 4.7)**：
- Linear 最稳定，但参数最多
- Block-diagonal-10 + argmax 在保持竞争力（mid-tier 表现）的同时，把 head 参数从 10,240 砍到 1,024（10× 更少）
- 配合 **argmax** 输出，决策只依赖最大 logit，扰动非 max block 不会影响 fitness attribution → 大幅减少 cross-block interference
- 这支持论文的核心论点："the coordination objective exhibits strong block-$\varepsilon$ separability"

**Table 6 (in paper)** — 全部训练参数分布：
```
                 SVF  linear  low-rank  sparse  block-diag-2  block-diag-10
Parameter Size  9216  10240   ~14700    11266    5120          1024
```

**Total 训练参数 = SVF + head ≈ 10K-20K**，远低于典型 fine-tuning。

## 4. 训练：sep-CMA-ES (Section 3.3 + Appendix A.1)

**为什么不是 RL** (论文 Section 3.3)：
- block-$\varepsilon$ separable 几何 → 弱 inter-block 信号
- 终端 binary reward 让 policy gradient 方差爆炸
- "noisy global returns swamp weak inter-block signals, yielding ill-conditioned gradients, poor credit assignment, and unstable learning"

**sep-CMA-ES** (Hansen 2016) 关键点：
- 维护 $\theta$（head 全部权重），加 per-dimension step size $\sigma$（**对角协方差**）
- 论文实验：$\lambda = \lceil 4 + 3 \ln n \rceil$，$n \approx 10K \Rightarrow \lambda \approx 32$
- 复制次数：$m_{\text{CMA}}=16$（每个候选评估 16 次取平均），$m_{\text{RS}}=32$
- 预算对齐：每个 gen 跑 $16T$ 个 RS 候选

**为什么 sep-CMA-ES 优于 RS**（论文 Proposition 1+2）：
- 小 T 区域：sep-CMA-ES 提升线性增长，RS 仅 $\ln$ 增长
- 稳态区域：每步剩余误差衰减 $\sim 1/n$，常数 $\sim \Theta(1)$ (CMA recombination efficiency)
- RS 即使多 round 也只对数增长

**Empirical comparison (Table 4, Section 4.8)**:
| Method | LiveCodeBench | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| REINFORCE | 0.253 | 0.459 | 0.500 | 0.266 |
| RS | 0.374 | 0.794 | 0.897 | 0.345 |
| SFT | 0.592 | 0.786 | 0.906 | 0.360 |
| **sep-CMA-ES** | **0.615** | **0.880** | **0.916** | **0.401** |

sep-CMA-ES 在所有 4 个 benchmark 上都胜出。

**Figure 6 (in paper)**: sep-CMA-ES 学会偏向高性能 LLM 的非均匀分布；REINFORCE 保持接近均匀；RS 经常塌缩到 unipolar（只选一个 agent/role），多样性丧失。

## 5. 与原型的差异 (诚实清单)

| 维度 | 论文 | 我的原型 | 备注 |
|---|---|---|---|
| SLM backbone | Qwen3-0.6B + SVF | **无** (16-dim 手写特征替代) | 真正的"再实现"需要 backbone；`QwenCoordinator` 接口已留好 |
| Head 架构 | Linear / Low-rank / Sparse / **Block-diagonal-10** | **MLP (16→32→32)** | 我用的是 2 层 MLP，没复现 4 种 head |
| Output conversion | Softmax / **Argmax** | Softmax + temperature | Argmax 在论文中是 block-diag-10 的搭档 |
| 模型池 | 7 个真实 LLM (GPT-5, Gemini-2.5-pro, Claude-4, Gemma-3-27B, DeepSeek-R1-Distill-32B, Qwen3-32B direct/reasoning) | 2 mock (small/strong) | 玩具化以便训练 |
| $n_a$ | 10 (7 模型 + 3 角色) | 5 (2 模型 + 3 角色) | |
| Max turns K | 5 | 6 | |
| Population size $\lambda$ | $\lceil 4+3\ln n \rceil$ (n≈10K → 32) | 12-16 (用户指定) | 论文公式未自动化 |
| sep-CMA-ES | ✅ (理论 + 实证) | ✅ (实现) | |
| RS baseline | ✅ (Table 4) | ❌ | **下一步加** |
| 终止规则 | τ = min{k≤K: R_k=V and u_k=ACCEPT} | ✅ | |
| 训练任务 | MATH500, MMLU, RLPR, LiveCodeBench | 玩具算术/逻辑/字符串 | |

## 6. 复现路径

- **MVP (今天)**: 启发式 + MLP 模拟 + sep-CMA-ES 训练 + 玩具算术任务 ✅
- **下一步 (本 PR)**: 加 Block-diagonal head + Argmax + RS baseline
- **下一阶段**: 换 Qwen2-0.5B 取 hidden state（替代不可达的 Qwen3-0.6B）
- **进阶**: SVF on Qwen backbone + 接入真实 LLM API + 对比启发式 vs 训练后路由的 pass@1
