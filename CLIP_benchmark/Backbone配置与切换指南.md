# 🏗️ CLIP Backbone 配置与切换指南

## 📊 当前配置

你目前使用的是：**ViT-L/14** (Vision Transformer Large, patch size 14)

这是论文中**主要使用且性能最好**的 Backbone！

---

## 🎯 论文中评估的所有 Backbone

### 主要 Backbone (ViT-L/14) - 论文重点

| 模型 | Backbone | 参数量 | Clean Acc | Adv ε=2/255 | Adv ε=4/255 |
|------|----------|--------|-----------|-------------|-------------|
| **OpenAI CLIP** | **ViT-L/14** | 428M | **75.5%** | **0%** | **0%** |
| **FARE² (ε=2)** | **ViT-L/14** | 428M | **73.8%** | **56.8%** | **20.5%** |
| **FARE² (ε=4)** | **ViT-L/14** | 428M | **71.2%** | **59.4%** | **32.4%** |
| **TeCoA (ε=2)** | **ViT-L/14** | 428M | **71.3%** | **54.4%** | **27.0%** |
| **TeCoA (ε=4)** | **ViT-L/14** | 428M | **68.5%** | **57.0%** | **31.9%** |

**结论**：
- ✅ **ViT-L/14 是最好的选择**（最高准确率）
- ✅ **你当前使用的就是这个！**
- ✅ **FARE² (ε=2)** 在干净样本和对抗鲁棒性之间达到最佳平衡

---

### 其他 Backbone (基础尺寸) - 补充实验

论文还提供了**较小的 Backbone**，用于资源受限场景：

#### ViT-B/32 (较小，速度快)

| 模型 | Backbone | 参数量 | Clean | Adv ε=2/255 | Adv ε=4/255 |
|------|----------|--------|-------|-------------|-------------|
| OpenAI CLIP | ViT-B/32 | 151M | 63.2% | 0% | 0% |
| FARE⁴ (OpenAI) | ViT-B/32 | 151M | 48.6% | 33.7% | 21.9% |
| FARE⁴ (LAION 2B) | ViT-B/32 | 151M | 53.8% | 35.5% | 21.2% |

**特点**：
- ⚡ **速度最快**（patch size 32 = 更少的 tokens）
- 💾 **内存最小**（151M 参数）
- ⚠️ **准确率较低**（比 ViT-L/14 低约 10-20%）

#### ViT-B/16 (中等)

| 模型 | Backbone | 参数量 | Clean | Adv ε=2/255 | Adv ε=4/255 |
|------|----------|--------|-------|-------------|-------------|
| FARE⁴ (LAION 2B) | ViT-B/16 | 149M | 56.6% | 39.2% | 23.5% |
| TeCoA⁴ (LAION 2B) | ViT-B/16 | 149M | 51.5% | 38.4% | 26.4% |

**特点**：
- ⚖️ **平衡选择**（速度 vs 准确率）
- 📈 **比 B/32 好**，但仍低于 L/14

#### ConvNeXt-B (卷积架构)

| 模型 | Backbone | 参数量 | Clean | Adv ε=2/255 | Adv ε=4/255 |
|------|----------|--------|-------|-------------|-------------|
| FARE⁴ (LAION 2B) | ConvNeXt-B | 198M | 60.2% | 44.1% | 28.4% |
| TeCoA⁴ (LAION 2B) | ConvNeXt-B | 198M | 56.2% | 44.1% | 31.8% |

**特点**：
- 🔄 **卷积架构**（非 Transformer）
- 🎯 **对抗鲁棒性好**（特别是 ε=4/255）
- ⚠️ **需要不同的代码支持**

---

## 🔧 如何切换 Backbone

### 方法 1: 使用现有的 ViT-L/14 模型（推荐）

你已经下载了 **ViT-L/14** 的权重，直接使用即可：

```bash
# 编辑模型配置
nano ~/data/KeyToken/CLIP_benchmark/benchmark/models_local.txt
```

**当前配置**：
```
# ViT-L/14 架构（最好的选择）
ViT-L-14,openai                                    # OpenAI 原始模型
ViT-L-14,~/data/KeyToken/models/fare_eps_2.pt     # FARE² (ε=2)
ViT-L-14,~/data/KeyToken/models/fare_eps_4.pt     # FARE² (ε=4)
```

### 方法 2: 下载其他 Backbone 的权重

如果你想测试**较小的 Backbone**：

#### ViT-B/32 (OpenAI 预训练)

```bash
cd ~/data/KeyToken/models

# FARE² (ε=1)
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/cCgQAS8QW9arj9d/download/vitb32_fare_eps_1.pt

# FARE² (ε=4)
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/3nMxBKEwbWnDymT/download/vitb32_fare_eps_4.pt

# TeCoA (ε=4)
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/RiWGQzBrqYNCaDk/download/vitb32_tecoa_eps_4.pt
```

然后修改 `models_local.txt`：
```
ViT-B-32,openai
ViT-B-32,~/data/KeyToken/models/vitb32_fare_eps_1.pt
ViT-B-32,~/data/KeyToken/models/vitb32_fare_eps_4.pt
ViT-B-32,~/data/KeyToken/models/vitb32_tecoa_eps_4.pt
```

#### ViT-B/16 (LAION 2B 预训练)

```bash
cd ~/data/KeyToken/models

# 使用 huggingface-cli 下载
huggingface-cli download chs20/FARE4-ViT-B-16-laion2B-s34B-b88K --local-dir vitb16_fare_eps_4

huggingface-cli download chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K --local-dir vitb16_tecoa_eps_4
```

然后修改 `models_local.txt`：
```
ViT-B-16,laion2b_s34b_b88k
ViT-B-16,~/data/KeyToken/models/vitb16_fare_eps_4
ViT-B-16,~/data/KeyToken/models/vitb16_tecoa_eps_4
```

---

## 📝 完整的 Backbone 对比

### 性能对比（Zero-Shot 分类，13 个数据集平均）

| Backbone | 参数量 | 推理速度 | Clean Acc | Adv ε=2 | Adv ε=4 | 推荐场景 |
|----------|--------|----------|-----------|---------|---------|----------|
| **ViT-L/14** | **428M** | **基准** | **✅ 最高** | **✅ 最高** | **✅ 最高** | **📊 论文复现（强烈推荐）** |
| ViT-B/32 | 151M | ⚡ 2-3x 快 | ⚠️ 中等 | ⚠️ 中等 | ⚠️ 较低 | 🚀 快速原型/资源受限 |
| ViT-B/16 | 149M | ⚡ 1.5x 快 | ⚠️ 中上 | ⚠️ 中上 | ⚠️ 中等 | ⚖️ 平衡选择 |
| ConvNeXt-B | 198M | 🐢 稍慢 | ✅ 较高 | ✅ 较高 | ✅ 高 | 🔬 对抗鲁棒性研究 |

### GPU 内存需求（评估时）

| Backbone | Batch Size=128 | Batch Size=256 | Batch Size=512 |
|----------|----------------|----------------|----------------|
| ViT-L/14 | ~8 GB | ~14 GB | ~24 GB |
| ViT-B/32 | ~4 GB | ~7 GB | ~12 GB |
| ViT-B/16 | ~5 GB | ~9 GB | ~16 GB |
| ConvNeXt-B | ~6 GB | ~10 GB | ~18 GB |

**你的硬件**：2x RTX 3090 (24GB each) ✅ 足够运行任何 Backbone！

---

## 💡 推荐配置

### 场景 1: 论文完整复现（推荐）⭐⭐⭐

**使用当前配置**：
```
ViT-L-14,openai
ViT-L-14,~/data/KeyToken/models/fare_eps_2.pt
ViT-L-14,~/data/KeyToken/models/fare_eps_4.pt
```

**原因**：
- ✅ 论文主要使用 ViT-L/14
- ✅ 性能最好
- ✅ 你的 GPU 完全支持

### 场景 2: 快速测试/消融实验

添加 ViT-B/32 进行对比：
```
# 主力模型 (ViT-L/14)
ViT-L-14,openai
ViT-L-14,~/data/KeyToken/models/fare_eps_2.pt

# 快速测试 (ViT-B/32)
ViT-B-32,openai
ViT-B-32,~/data/KeyToken/models/vitb32_fare_eps_4.pt
```

### 场景 3: 完整 Backbone 对比研究

```
# ViT-L/14 (主力)
ViT-L-14,openai
ViT-L-14,~/data/KeyToken/models/fare_eps_2.pt
ViT-L-14,~/data/KeyToken/models/fare_eps_4.pt

# ViT-B/32 (快速)
ViT-B-32,openai
ViT-B-32,~/data/KeyToken/models/vitb32_fare_eps_4.pt

# ViT-B/16 (平衡)
ViT-B-16,laion2b_s34b_b88k
ViT-B-16,~/data/KeyToken/models/vitb16_fare_eps_4
```

---

## 🎯 总结与建议

### 关键要点

1. ✅ **你当前使用 ViT-L/14 是正确的选择**
2. ✅ **这是论文中性能最好的 Backbone**
3. ✅ **无需更换，除非有特殊需求**

### 何时考虑切换 Backbone？

| 场景 | 推荐 Backbone | 原因 |
|------|---------------|------|
| **论文复现（默认）** | **ViT-L/14** | **主要模型，性能最优** |
| GPU 内存不足 | ViT-B/32 | 内存需求最低 |
| 快速迭代测试 | ViT-B/32 | 推理速度快 2-3 倍 |
| 对抗鲁棒性研究 | ConvNeXt-B | ε=4/255 鲁棒性最好 |
| 平衡性能和速度 | ViT-B/16 | 中间选择 |

### 我的建议

**保持当前的 ViT-L/14 配置**：
```bash
# 无需修改，当前配置已经是最优的！
cat ~/data/KeyToken/CLIP_benchmark/benchmark/models_local.txt
```

如果需要对比实验，可以添加 ViT-B/32 作为补充。

---

## 📌 快速参考

### 当前你的模型文件

```bash
ls -lh ~/data/KeyToken/models/*.pt
```

输出：
```
fare_eps_2.pt  (1.2GB)  # ViT-L/14, ε=2 ✅ 最佳平衡
fare_eps_4.pt  (1.2GB)  # ViT-L/14, ε=4 ✅ 最强鲁棒性
```

### 运行评估

```bash
cd ~/data/KeyToken/CLIP_benchmark

# 使用当前 ViT-L/14 模型评估
./bash/eval_local_clean.sh
./bash/eval_local_adv.sh
```

---

**结论**: 你当前的 ViT-L/14 Backbone 配置是最优选择，无需更改！🎉
