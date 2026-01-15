# 🎯 CLIP Backbone 性能快速对比

## 当前配置：ViT-L/14 ✅

**这是论文中性能最好的 Backbone！无需更改。**

---

## 📊 性能对比表

### ViT-L/14 模型（当前使用）⭐ 推荐

| 模型 | Clean Acc | Adv ε=1/255 | Adv ε=2/255 | Adv ε=4/255 | 推荐场景 |
|------|-----------|-------------|-------------|-------------|----------|
| **OpenAI CLIP** | **75.5%** | 0% | 0% | 0% | 🔬 基线对比 |
| **FARE² (ε=2)** | **73.8%** | 46.3% | **56.8%** | 20.5% | ⭐ **日常评估（推荐）** |
| **FARE² (ε=4)** | **71.2%** | 50.9% | 59.4% | **32.4%** | 💪 强对抗攻击 |
| TeCoA (ε=2) | 71.3% | 42.6% | 54.4% | 27.0% | 🔬 方法对比 |
| TeCoA (ε=4) | 68.5% | 45.6% | 57.0% | 31.9% | 🔬 方法对比 |

**你已下载的模型** ✅：
- `fare_eps_2.pt` - 平衡型，推荐日常使用
- `fare_eps_4.pt` - 强鲁棒型，用于强攻击评估

---

### 其他 Backbone（参考）

#### ViT-B/32（较小，速度快）

| 模型 | Clean | Adv ε=2 | Adv ε=4 | 参数量 | 速度 |
|------|-------|---------|---------|--------|------|
| OpenAI CLIP | 63.2% | 0% | 0% | 151M | ⚡ 2-3x |
| FARE⁴ (OpenAI) | 48.6% | 33.7% | 21.9% | 151M | ⚡ 2-3x |
| FARE⁴ (LAION 2B) | 53.8% | 35.5% | 21.2% | 151M | ⚡ 2-3x |

**特点**：速度快但准确率低 10-20%

#### ViT-B/16（中等）

| 模型 | Clean | Adv ε=2 | Adv ε=4 | 参数量 | 速度 |
|------|-------|---------|---------|--------|------|
| FARE⁴ (LAION 2B) | 56.6% | 39.2% | 23.5% | 149M | ⚡ 1.5x |
| TeCoA⁴ (LAION 2B) | 51.5% | 38.4% | 26.4% | 149M | ⚡ 1.5x |

**特点**：平衡选择，但仍低于 ViT-L/14

#### ConvNeXt-B（卷积架构）

| 模型 | Clean | Adv ε=2 | Adv ε=4 | 参数量 | 特点 |
|------|-------|---------|---------|--------|------|
| FARE⁴ (LAION 2B) | 60.2% | 44.1% | 28.4% | 198M | 🎯 高鲁棒性 |
| TeCoA⁴ (LAION 2B) | 56.2% | 44.1% | 31.8% | 198M | 🎯 高鲁棒性 |

**特点**：对抗鲁棒性强，但需要不同代码

---

## 🔑 关键结论

### 1. 你当前使用的是最好的 Backbone ✅

**ViT-L/14 的优势**：
- ✅ **准确率最高**（比 ViT-B/32 高约 15-20%）
- ✅ **鲁棒性最强**（比小模型高 15-25%）
- ✅ **论文主要使用**（所有主表格数据都是 ViT-L/14）
- ✅ **你的硬件完全支持**（2x RTX 3090, 24GB each）

### 2. 模型选择建议

| 如果你想... | 选择 | 配置 |
|-------------|------|------|
| **复现论文结果** | ✅ **保持当前配置** | `ViT-L-14,openai` + `fare_eps_2.pt` |
| 日常评估 | FARE² (ε=2) | `~/data/KeyToken/models/fare_eps_2.pt` |
| 强对抗攻击评估 | FARE² (ε=4) | `~/data/KeyToken/models/fare_eps_4.pt` |
| 快速原型测试 | ViT-B/32 | 需要另外下载 |
| 节省 GPU 内存 | ViT-B/32 | 需要另外下载 |

### 3. GPU 内存需求（Batch Size = 128）

| Backbone | 单卡内存 | 你的硬件 (2x 24GB) |
|----------|----------|--------------------|
| **ViT-L/14** | **~8 GB** | ✅ **完全够用** |
| ViT-B/32 | ~4 GB | ✅ 大材小用 |
| ViT-B/16 | ~5 GB | ✅ 大材小用 |

---

## 🚀 如何切换 Backbone

### 保持 ViT-L/14（推荐）

```bash
# 查看当前配置
cat ~/data/KeyToken/CLIP_benchmark/benchmark/models_local.txt
```

**已配置**：
```
ViT-L-14,openai                                    # 基线
ViT-L-14,~/data/KeyToken/models/fare_eps_2.pt     # 平衡型 ⭐
ViT-L-14,~/data/KeyToken/models/fare_eps_4.pt     # 强鲁棒型
```

### 添加 ViT-B/32（可选）

如果想对比实验：

```bash
cd ~/data/KeyToken/models

# 下载 ViT-B/32 权重
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/3nMxBKEwbWnDymT/download/vitb32_fare_eps_4.pt
```

然后编辑 `models_local.txt`：
```bash
nano ~/data/KeyToken/CLIP_benchmark/benchmark/models_local.txt
```

添加：
```
# ViT-B/32 (对比实验)
ViT-B-32,openai
ViT-B-32,~/data/KeyToken/models/vitb32_fare_eps_4.pt
```

---

## 📈 预期性能（13 个数据集平均）

### 干净样本

```
ViT-L/14 + OpenAI:      75.5%  ⭐ 最高
ViT-L/14 + FARE² (ε=2): 73.8%  ⭐ 次高（仅降低 1.7%）
ViT-B/32 + OpenAI:      63.2%  ⚠️  低 12%
```

### 对抗样本 (ε=2/255)

```
ViT-L/14 + FARE² (ε=2): 56.8%  ⭐ 最高
ViT-L/14 + OpenAI:       0.0%  ❌ 完全失败
ViT-B/32 + FARE⁴:       33.7%  ⚠️  低 23%
```

---

## 💡 最终建议

### 推荐配置（当前）⭐⭐⭐

```bash
# 无需修改！当前配置已经是最优的
cd ~/data/KeyToken/CLIP_benchmark

# 直接运行评估
./bash/eval_local_clean.sh    # 干净样本评估
./bash/eval_local_adv.sh      # 对抗样本评估
```

**理由**：
1. ✅ ViT-L/14 是论文主要使用的 Backbone
2. ✅ 性能最好（准确率 + 鲁棒性）
3. ✅ 你的硬件完全支持
4. ✅ 已经下载好所有必要的权重

---

## 📌 快速查询

### 我的 Backbone 是什么？

```bash
# 查看配置
head -1 ~/data/KeyToken/CLIP_benchmark/benchmark/models_local.txt | grep "ViT"
```

**答案**：`ViT-L-14` (Vision Transformer Large, patch size 14)

### 这是最好的吗？

**是的！** ViT-L/14 是：
- ✅ 论文中性能最好的架构
- ✅ 参数量最大（428M）
- ✅ 准确率最高
- ✅ 对抗鲁棒性最强

### 需要换吗？

**不需要！** 除非：
- ❌ GPU 内存不足（你有 2x 24GB，完全够）
- ❌ 需要极快推理速度（评估不需要）
- ❌ 研究不同架构的影响（科研需求）

---

**总结**：你的 ViT-L/14 配置是最优选择，直接开始评估即可！🎉
