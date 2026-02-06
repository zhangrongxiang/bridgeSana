# Bridge Training for Sana Model

基于 ViBT (Vision Bridge Transformer) 算法的 Sana 模型 Bridge 训练脚本。

## 📋 目录

- [概述](#概述)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [推理](#推理)
- [核心算法](#核心算法)

---

## 🎯 概述

本训练脚本实现了 **Data-to-Data Bridge** 训练范式，与传统的 Noise-to-Data 扩散模型不同：

| 特性 | 传统扩散模型 | Bridge 模型 |
|------|-------------|------------|
| 起点 | 随机噪声 | 源图像 latents |
| 终点 | 目标图像 | 目标图像 |
| 适用场景 | 文生图 | 图像翻译/风格化 |
| 训练目标 | 噪声预测 | 速度场预测 |

### 核心创新

1. **Brownian Bridge 框架**：从源数据到目标数据的随机桥接
2. **稳定化速度匹配**：解决 t→1 时的数值不稳定问题
3. **LoRA 微调**：高效训练，仅需 ~20K 步

---

## 📁 数据集准备

### 数据集结构

你的数据集应该按照以下结构组织：

```
/cache/omnic/3D_Chibi/
├── src/              # 源图像
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── tar/              # 目标图像（风格化后）
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── caption/          # 文本描述（可选）
│   ├── 001.txt
│   ├── 002.txt
│   └── ...
└── train.jsonl       # 元数据文件
```

### train.jsonl 格式

每行一个 JSON 对象：

```json
{"src": "3D_Chibi/src/001.png", "tar": "3D_Chibi/tar/001.png", "prompt": "3D Chibi Style, A cute character..."}
{"src": "3D_Chibi/src/002.png", "tar": "3D_Chibi/tar/002.png", "prompt": "3D Chibi Style, Another character..."}
```

**字段说明：**
- `src`: 源图像相对路径
- `tar`: 目标图像相对路径
- `prompt`: 文本描述（描述目标风格）

---

## 🚀 训练

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install diffusers transformers accelerate peft torch torchvision
pip install wandb tensorboard  # 可选，用于日志记录
```

### 2. 配置训练参数

编辑 `launch_bridge_training.sh` 中的参数：

```bash
# 模型和数据路径
export MODEL_PATH="/cache/SANA1.5_4.8B_1024px_diffusers"
export DATA_DIR="/cache/omnic/3D_Chibi"
export OUTPUT_DIR="./output/bridge_3d_chibi"

# 训练配置
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4  # 有效 batch size = 1 * 4 = 4
export MAX_TRAIN_STEPS=20000
export LEARNING_RATE=1e-4

# LoRA 配置
export LORA_RANK=128
export LORA_ALPHA=128

# Bridge 特定参数
export NOISE_SCALE=1.0  # 推荐值：0.5-2.0
export USE_STABILIZED_VELOCITY="--use_stabilized_velocity"  # 强烈推荐开启
```

### 3. 启动训练

```bash
cd /home/ma-user/workspace/rongxiang/bridgeSana/train_scripts
bash launch_bridge_training.sh
```

### 4. 监控训练

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir=./output/bridge_3d_chibi/logs
```

---

## 🎨 推理

训练完成后，使用推理脚本进行图像翻译：

```bash
python inference_bridge.py \
  --model_path="/cache/SANA1.5_4.8B_1024px_diffusers" \
  --lora_path="./output/bridge_3d_chibi/final_checkpoint/pytorch_lora_weights.bin" \
  --source_image="/path/to/source.png" \
  --prompt="3D Chibi Style, A cute character with big eyes" \
  --output_path="output.png" \
  --num_inference_steps=28 \
  --guidance_scale=4.5 \
  --noise_scale=1.0 \
  --seed=42
```

### 参数说明

- `--source_image`: 输入的源图像
- `--prompt`: 目标风格的文本描述
- `--num_inference_steps`: 推理步数（推荐 20-50）
- `--guidance_scale`: 引导强度（推荐 3.0-6.0）
- `--noise_scale`: 噪声尺度（推荐 0.5-2.0）

---

## 🔬 核心算法

### Brownian Bridge 训练公式

#### 1. 中间状态构造

给定源 latent $x_0$ 和目标 latent $x_1$，在时间 $t \in [0,1]$ 构造中间状态：

$$x_t = (1-t) \cdot x_0 + t \cdot x_1 + \sqrt{t(1-t)} \cdot \epsilon$$

其中 $\epsilon \sim \mathcal{N}(0, I)$

#### 2. 速度目标

$$u_t = \frac{x_1 - x_t}{1 - t}$$

#### 3. 稳定化归一化因子

$$\alpha^2 = 1 + \frac{t \cdot D}{(1-t) \cdot \|x_1 - x_0\|^2}$$

其中 $D$ 是 latent 维度数。

#### 4. 训练损失

$$\mathcal{L} = \mathbb{E}_{t,\epsilon,x_0,x_1}\left[\left\|\frac{v_\theta(x_t, t)}{\alpha} - \frac{u_t}{\alpha}\right\|^2\right]$$

### 代码实现位置

- **数据集加载**: `BridgeDataset` 类（第 219-268 行）
- **损失计算**: `compute_bridge_loss` 函数（第 283-323 行）
- **训练循环**: `main` 函数中的训练循环（第 420-500 行）

---

## 📊 训练建议

### 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-4 | 使用 AdamW 优化器 |
| `lora_rank` | 128 | 平衡性能和质量 |
| `noise_scale` | 0.5-2.0 | 图像编辑用 0.5，视频用 2.0 |
| `max_train_steps` | 20000 | 根据数据集大小调整 |
| `gradient_accumulation_steps` | 4 | 有效 batch size = 4 |

### 常见问题

**Q: 训练损失不下降？**
- 检查 `use_stabilized_velocity` 是否开启
- 尝试降低学习率到 5e-5
- 检查数据集质量（源图和目标图是否对齐）

**Q: 生成结果不稳定？**
- 调整 `noise_scale`（降低到 0.5）
- 增加推理步数到 50
- 检查 prompt 是否准确描述目标风格

**Q: 显存不足？**
- 减小 `train_batch_size` 到 1
- 减小 `lora_rank` 到 64
- 使用梯度检查点（已默认开启）

---

## 📚 参考文献

- **ViBT 论文**: Vision Bridge Transformer
- **Sana 模型**: https://huggingface.co/Efficient-Large-Model/Sana
- **Diffusers 文档**: https://huggingface.co/docs/diffusers

---

## 🙏 致谢

本训练脚本基于：
- ViBT (Vision Bridge Transformer) 算法
- Diffusers 库的 Sana 实现
- LoRA (Low-Rank Adaptation) 技术
