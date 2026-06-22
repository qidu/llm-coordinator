# TRINITY: An Evolved LLM Coordinator — 核心机制笔记

> 来源：ICLR 2026, Sakana AI
> 本笔记聚焦 Section 4 (S4) 的状态机/路由逻辑及可复现的关键点。

## 1. 设计目标

把"高难度复杂任务的多步求解 + 验证"这两件事从**中心协调器**中卸载出去，扔给外部的 LLM 池。
协调器本身保持**极致轻量**（0.6B 小模型 + ~10K 参数的线性 Head），但通过**进化算法**（sep-CMA-ES）训练出非平凡的路由策略。

## 2. 状态机

每一轮 k，系统维护完整对话历史轨迹 $\mathcal{C}_{k-1}$。
协调器产生两个决策：

- $A_k$：从模型池里选哪个 LLM
- $R_k$：赋予什么角色

三种角色：

| 角色 | 输入 | 输出 | 终止? |
|---|---|---|---|
| **Thinker** | 当前上下文 + 问题 | 高层策略、子任务拆解、对当前解法的批判 | 否 |
| **Worker** | Thinker 的策略 + 上下文 | 具体执行（推导、写代码、计算） | 否 |
| **Verifier** | 累积中间步骤 + 初始问题 | `ACCEPT` (终止, 输出最终答案) 或 `REVISE` (带诊断 $\delta_k$ 进入下一轮) | 是/否 |

## 3. 协调器 (Coordinator) 实现

论文里协调器**不是 if-else**，而是一个训练出来的分类器：

1. 把 transcript $\mathcal{C}_{k-1}$ 喂给小模型（如 Qwen-0.6B）
2. 取**最后一个 token** 的 hidden state $h \in \mathbb{R}^d$
3. 过一个 ~10K 参数的线性层：$\text{logits} = W h + b$，$W \in \mathbb{R}^{C \times d}$
4. argmax 选 $(A_k, R_k)$

可选的简化：用一个固定 tokenizer 把 transcript 编码成向量 (不依赖外部 LLM) 再过同样的 Head。

## 4. 训练：sep-CMA-ES

放弃 RL 的原因：
- 奖励稀疏（只有 ACCEPT/REVISE）
- LLM 输出随机性强，policy gradient 方差爆炸
- 路由决策是离散的，难以反向传播

sep-CMA-ES 流程：
- 维护参数 $\theta$（线性 Head 的所有权重）
- 每个维度独立维护一个 step size $\sigma_i$
- 每一代采样 $\lambda$ 个候选，按 fitness（= 多任务 rollout 的平均正确率）排序
- 更新 $\mu$ 和 $\sigma_i$（基于排名权重，类似 CMA-ES 经典公式）

奖励：binary reward $= \mathbb{1}[\text{final answer correct}]$。

## 5. 与原型代码的差异

你给的 OpenAI 跑版用 if-else 做路由；本项目把它换成：
1. **特征提取器**：从 transcript 提取低维特征向量（turn count、last role、context length、是否被 reject 过等）
2. **MLP 路由器**：2 层感知机，~5K-10K 参数
3. **进化训练器**：自实现的 sep-CMA-ES（不依赖 pycma，纯净 numpy）

## 6. 任务设计

为了让 binary reward 有梯度，训练任务需要**答案可验证**：
- 算术题（带明确数值答案）
- 简单逻辑题（True/False）
- 字符串操作（反转、计数）

每轮 rollout 让 Trinity 系统解题，最终答案与 ground truth 比对给出 0/1 奖励。

## 7. 复现路径

- **MVP (今天)**: 启发式 + MLP 模拟 + sep-CMA-ES 训练 + 玩具算术任务
- **下一阶段**: 换 Qwen2-0.5B（替代不可达的 Qwen-0.6B）取 hidden state
- **进阶**: 接入真实 LLM API，对比启发式 vs 训练后路由的 pass@1

