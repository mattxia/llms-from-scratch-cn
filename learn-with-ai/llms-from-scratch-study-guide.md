# LLMs From Scratch 项目学习指南

> 本指南基于 Datawhale 的「动手学 LLM」项目，帮助你系统掌握大语言模型的核心技术和架构实现。

---

## 📖 项目概览

这是一个**从 0 构建大语言模型**的完整教程项目，包含两大部分：

### 1. 基础知识部分
基于《Build a Large Language Model From Scratch》的中文教程，涵盖：
- 文本数据处理
- 注意力机制实现
- GPT 模型从零实现
- 预训练与微调

### 2. 模型架构部分
10+ 种主流 LLM 架构实现：
| 模型 | Notebook | 特点 |
|------|----------|------|
| Llama3 | `llama3-from-scratch.ipynb` | 经典 Transformer 架构 |
| ChatGLM4 | `chatglm4-guide.ipynb` | 国产优秀中文模型 |
| RWKV V2-V6 | `rwkv-v*-guide.ipynb` | RNN 架构系列 |
| Mamba | `demo.ipynb` | SSM 新架构 |
| MiniCPM | `MiniCPM.ipynb` | 轻量级模型 |
| Phi-3 | `phi-3.ipynb` | 微软小模型 |

---

## 🎯 分阶段学习路径

### 阶段 1：基础准备（1-2 周）

**学习目录**：
```
Codes/appendix-A/  → PyTorch 基础
Codes/ch02/        → 文本数据处理
```

**核心知识点**：
- ✅ PyTorch 张量操作
- ✅ 文本分词（Tokenization）
- ✅ BytePair Encoding (BPE)
- ✅ 词嵌入（Embedding）
- ✅ 数据加载器（DataLoader）

**必做练习**：
```python
# Codes/ch02/01_main-chapter-code/ch02.ipynb
# 1. 实现 BPE 分词器
# 2. 创建自定义 Dataset 和 DataLoader
# 3. 理解滑动窗口采样
```

**检查清单**：
- [ ] 能解释什么是 Tokenization
- [ ] 能手写 BPE 算法核心逻辑
- [ ] 能使用 tiktoken 库进行分词
- [ ] 理解 `GPTDatasetV1` 的工作原理

---

### 阶段 2：注意力机制（2-3 周）

**学习目录**：
```
Codes/ch03/  → 编写注意力机制
```

**核心知识点**：
- ✅ 自注意力（Self-Attention）
- ✅ 缩放点积注意力
- ✅ 多头注意力（Multi-Head Attention）
- ✅ 因果掩码（Causal Mask）
- ✅ 位置编码基础

**核心代码结构**：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        # W_query, W_key, W_value 权重矩阵
        # 多头分割逻辑
        # 因果掩码注册
        
    def forward(self, x):
        # 1. 计算 Q, K, V
        # 2. 多头分割与转置
        # 3. 缩放点积注意力
        # 4. 合并多头输出
```

**检查清单**：
- [ ] 能推导注意力公式：Attention(Q,K,V) = softmax(QK^T/√d)V
- [ ] 理解为什么需要多头注意力
- [ ] 能解释因果掩码的作用
- [ ] 能手写 `MultiHeadAttention` 类

---

### 阶段 3：GPT 模型实现（3-4 周）⭐

**学习目录**：
```
Codes/ch04/  → 从零开始实现 GPT 模型
```

**核心文件**：`gpt.py`（约 200 行核心代码）

**模型组件详解**：

| 组件 | 代码行数 | 作用 |
|------|----------|------|
| `LayerNorm` | 15 行 | 层归一化，稳定训练 |
| `GELU` | 10 行 | 激活函数，引入非线性 |
| `FeedForward` | 15 行 | 前馈神经网络 |
| `TransformerBlock` | 25 行 | 完整 Transformer 块 |
| `GPTModel` | 20 行 | 完整 GPT 模型 |
| `generate_text_simple` | 20 行 | 文本生成函数 |

**架构流程图**：
```
输入 Token → Token Embedding + Position Embedding
                ↓
        [TransformerBlock] × n_layers
                ↓
           LayerNorm
                ↓
           Output Head → Logits
```

**关键配置**（124M 模型）：
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # 词表大小
    "ctx_len": 1024,       # 上下文长度
    "emb_dim": 768,        # 嵌入维度
    "n_heads": 12,         # 注意力头数
    "n_layers": 12,        # Transformer 层数
    "drop_rate": 0.1,      # Dropout 率
    "qkv_bias": False      # QKV 偏置
}
```

**检查清单**：
- [ ] 能独立写出 `TransformerBlock`（含残差连接）
- [ ] 理解 LayerNorm 与 BatchNorm 的区别
- [ ] 能解释 GELU 激活函数的优势
- [ ] 能手写文本生成循环

---

### 阶段 4：预训练与微调（4-6 周）

**学习目录**：
```
Codes/ch05/  → 预训练（未标记数据）
Codes/ch06/  → 微调（文本分类）
Codes/ch07/  → 人类反馈微调（DPO）
```

#### 5.1 预训练核心

**训练循环关键代码**：
```python
for epoch in range(num_epochs):
    for input_batch, target_batch in dataloader:
        optimizer.zero_grad()
        logits = model(input_batch)
        loss = cross_entropy(logits, target_batch)
        loss.backward()
        optimizer.step()
```

**关键概念**：
- 下一词预测任务（Next Token Prediction）
- 交叉熵损失计算
- 学习率调度
- 模型检查点保存

#### 5.2 微调技术

**文本分类微调**：
- 添加分类头（Classification Head）
- 冻结部分层 vs 全量微调
- 处理不平衡数据

**指令微调**（Chapter 7）：
- 构建指令数据集
- 使用 Ollama 生成数据
- DPO（Direct Preference Optimization）

**检查清单**：
- [ ] 能实现完整的训练循环
- [ ] 理解预训练与微调的区别
- [ ] 能加载 HuggingFace 权重
- [ ] 了解 DPO 基本原理

---

### 阶段 5：主流模型架构研究（选学）

**推荐学习顺序**：

#### 5.1 Llama3 架构 ⭐⭐⭐
```
Model_Architecture_Discussions/llama3/
```

**核心特性**：
- RoPE 位置编码（旋转位置编码）
- SwiGLU 激活函数
- Grouped Query Attention (GQA)
- RMSNorm 归一化

**关键公式**：
```python
# RoPE 旋转编码
def rope_embedding(x, freqs):
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2)
```

#### 5.2 ChatGLM4 架构 ⭐⭐⭐
```
Model_Architecture_Discussions/ChatGLM4/
```

**核心特性**：
- 混合注意力机制
- 多查询注意力（MQA）
- 中文优化词表

#### 5.3 RWKV V6 架构 ⭐⭐
```
Model_Architecture_Discussions/rwkv-v6/
```

**核心特性**：
- 线性注意力（Linear Attention）
- Token Shift 机制
- RNN 式推理（O(1) 推理）

**改进点**（V6 vs V5）：
```python
# 数据依赖线性插值（ddlerp）
def ddlerp(a, b, x):
    lora = lambda x + tanh(x @ A) @ B
    return a + (b - a) * lora(a + (b - a) * mu_x)
```

#### 5.4 Mamba 架构 ⭐⭐
```
Model_Architecture_Discussions/mamba/
```

**核心特性**：
- 状态空间模型（SSM）
- 选择性状态机制
- 线性复杂度序列建模

---

## 💡 实践建议

### 环境配置
```bash
# 基础环境
Python 3.9+
PyTorch 2.1+
tiktoken 0.6+

# 安装依赖
pip install torch tiktoken matplotlib numpy pandas
```

### 代码实践技巧

| 技巧 | 说明 |
|------|------|
| **逐行调试** | 用 debugger 逐行跟踪注意力计算过程 |
| **可视化** | 绘制注意力热力图、损失曲线 |
| **小模型实验** | 先用 emb_dim=128 等小配置快速验证 |
| **对比实验** | 修改超参数观察效果变化 |
| **代码复现** | 盖住源码，尝试独立实现 |

### 调试技巧
```python
# 1. 打印形状，追踪张量变化
print(f"Q shape: {queries.shape}")  # (B, H, T, D)

# 2. 检查梯度
for name, param in model.named_parameters():
    print(f"{name}: grad={param.grad is not None}")

# 3. 可视化注意力
plt.imshow(attn_weights[0, 0].detach().cpu())
```

---

## 📚 资源对照表

| 学习目标 | 推荐资源 | 预计时间 |
|----------|----------|----------|
| 快速入门 | `Codes/ch02-ch04` 主代码 | 2 周 |
| 深入理解 | `Translated_Book/` 详细教程 | 4 周 |
| 架构研究 | `Model_Architecture_Discussions/` | 4-8 周 |
| 完整掌握 | 全部内容 + 独立复现 | 3-6 个月 |

---

## 🎓 学习检查清单

### 基础阶段
- [ ] 理解 Tokenization 原理
- [ ] 能实现 BPE 算法
- [ ] 掌握 PyTorch Dataset/DataLoader

### 进阶阶段
- [ ] 理解自注意力机制
- [ ] 能手写 Multi-Head Attention
- [ ] 理解 Transformer 架构

### 高级阶段
- [ ] 独立实现 GPT 模型
- [ ] 实现预训练循环
- [ ] 掌握微调技术
- [ ] 理解至少 2 种模型架构

---

## 🔗 下一步行动

1. **今天**：配置环境，运行 `ch02.ipynb`
2. **本周**：完成第 2 章所有练习
3. **本月**：完成 ch02-ch04，能独立生成文本
4. **本季**：完成 ch05-ch06，理解预训练流程

---

## 📞 学习支持

- **项目仓库**：https://github.com/datawhalechina/llms-from-scratch-cn
- **Issue 反馈**：遇到问题在 GitHub 提 Issue
- **Discussion 讨论**：参与社区讨论

---

> 💡 **关键建议**：不要只看不练！每个 notebook 都要亲手运行，每段核心代码都要尝试独立复现。

**祝学习顺利！** 🚀
