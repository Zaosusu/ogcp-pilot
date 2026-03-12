# OGCP 机器学习 — 和弦识别

**任务**: 14 类吉他和弦分类  
**模型**: Mel 频谱图 + CNN  
**框架**: PyTorch + torchaudio + soundfile

---

## 环境安装

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy soundfile matplotlib
```

> RTX 5070 Ti 用 CUDA 12.8，上面命令已匹配。  
> 使用 soundfile 读取音频，无需安装 FFmpeg。

---

## 文件说明

| 文件 | 说明 |
|:---|:---|
| `dataset.py` | 数据加载 & Mel 频谱提取，含麦克风模拟增强（`_simulate_mic`，50% 概率触发） |
| `dataset1.py` | 同上，纯净版，无模拟增强，适合对照实验 |
| `model.py` | CNN 模型定义 |
| `train.py` | 训练脚本（含自动生成训练曲线） |
| `predict.py` | 推理脚本 |

---

## 快速开始

所有命令从**项目根目录**运行。

### 1. 训练

```bash
# 默认参数（小模型，50 epoch）
python ml/train.py --data_dir dataset/raw

# 自定义参数
python ml/train.py \
    --data_dir   dataset/raw \
    --epochs     80 \
    --batch_size 32 \
    --lr         3e-4 \
    --model      small
```

训练结束后 `models/` 目录下自动生成：

- `best_model.pth` — 验证集最优模型权重
- `train_log.csv` — 每 epoch 的 loss / acc 记录
- `training_curves.png` — Loss & Accuracy 训练曲线图

### 2. 推理

```bash
# 单文件
python ml/predict.py --wav dataset/raw/C/open-down-enya-direct-001.wav

# 批量测试某个和弦文件夹
python ml/predict.py --wav_dir dataset/raw/Am
```

---

## 模型说明

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| `small` | ~620K | 660 样本首选，不易过拟合 |
| `large` | ~2M | 数据增强后或样本扩充后使用 |

### 输入处理

- 采样率：44100 Hz（匹配恩雅 NEXG2 直进 Cubase 录音）
- Mel bins：128
- 时间帧：128 帧（约 1.49 秒，不足补零，超出截断）
- 归一化：Z-score per sample

### 训练策略

- Label Smoothing 0.1 — 防止过拟合
- AdamW + Cosine LR Decay
- 数据增强：随机音量 ±3dB + 随机时移
- Gradient Clipping (max_norm=1.0)

---

## 下一步

1. **消融实验**：MFCC vs Mel vs CQT 特征对比
2. **难度递增**：随机划分 → 把位划分 → 演奏法划分
3. **Baseline**：SVM + MFCC 作为传统方法对比
4. **可视化**：t-SNE 嵌入空间 + Grad-CAM 热图
5. **数据扩充**：补录更多和弦类型、木吉他麦克风采样、电吉他采样
