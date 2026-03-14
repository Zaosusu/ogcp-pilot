# OGCP-Pilot: A Physics-Aware Guitar Chord Dataset for Robust Recognition

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Zaosusu/ogcp-pilot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18979053.svg)](https://doi.org/10.5281/zenodo.18979053)

> **English** | [中文介绍](#中文介绍)

**OGCP (Open Guitar Chord Project) Pilot Study v1.0**

660 high-fidelity guitar chord samples with physical annotations for domain-robust chord recognition.

## Installation

```bash
pip install ogcp
```

## Quick Start

```python
from ogcp import OGCPDataset, plot_fretboard
import matplotlib.pyplot as plt

# 1. Load dataset (auto-downloads from Hugging Face on first run)
dataset = OGCPDataset(root_dir='dataset/raw')
print(len(dataset))  # 660

# 2. Dataset statistics
print(dataset.statistics())

# 3. Filter by chord type
dm_dataset = OGCPDataset(root_dir='dataset/raw', chord_filter=['D:min'])
print(len(dm_dataset))  # Dm samples only

# 4. Get all samples for a specific chord
samples = dataset.get_by_chord('C:maj')
sample = samples[0]
print(sample)  # ChordSample(C:maj, open/down, fretboard=[x, 3, 2, 0, 1, 0])

# 5. Plot fingerboard diagram
fig = plot_fretboard(
    sample.fretboard,
    chord_name=sample.chord_name,
    position=sample.position
)
plt.show()

# 6. Chord distribution
print(dataset.get_chord_distribution())
```

> For training and inference, see [`ml/README.md`](ml/README.md).

## Data Distribution

| Component | Location |
|:---|:---|
| **Code & SDK** | This GitHub repo |
| **Audio + Annotations** | [Hugging Face](https://huggingface.co/datasets/Zaosusu/ogcp-pilot) (auto-downloaded to `dataset/raw/` on first use) |

## Dataset Statistics

| Attribute | Value |
|:---|:---|
| Total Samples | 660 |
| Chord Classes | 14 |
| Recording Device | Enya NEXG2xCCS |

---

## 中文介绍

**OGCP (Open Guitar Chord Project) Pilot Study v1.0**

660 个高保真吉他和弦样本，包含物理标注，用于鲁棒性和弦识别研究。

### 安装

```bash
pip install ogcp
```

### 快速开始

```python
from ogcp import OGCPDataset, plot_fretboard
import matplotlib.pyplot as plt

# 1. 加载数据集（首次运行自动从 Hugging Face 下载）
dataset = OGCPDataset(root_dir='dataset/raw')
print(len(dataset))  # 660

# 2. 查看数据集统计
print(dataset.statistics())

# 3. 按和弦类型过滤
dm_dataset = OGCPDataset(root_dir='dataset/raw', chord_filter=['D:min'])
print(len(dm_dataset))  # 只含 Dm 样本

# 4. 获取某个和弦的所有样本
samples = dataset.get_by_chord('C:maj')
sample = samples[0]
print(sample)  # ChordSample(C:maj, open/down, fretboard=[x, 3, 2, 0, 1, 0])

# 5. 生成把位图
fig = plot_fretboard(
    sample.fretboard,
    chord_name=sample.chord_name,
    position=sample.position
)
plt.show()

# 6. 查看和弦分布
print(dataset.get_chord_distribution())
```

> 训练与推理说明请见 [`ml/README.md`](ml/README.md)。

### 数据分布

| 组件 | 位置 |
|:---|:---|
| **代码与 SDK** | 本 GitHub 仓库 |
| **音频 + 标注文件** | [Hugging Face](https://huggingface.co/datasets/Zaosusu/ogcp-pilot)（首次使用时自动下载到 `dataset/raw/`）|

### 数据集统计

| 属性 | 数值 |
|:---|:---|
| 总样本数 | 660 |
| 和弦类别 | 14 |
| 录音设备 | Enya NEXG2xCCS |

---

## Citation / 引用

```bibtex
@dataset{qiao2026ogcp,
  author = {Qin Qiao},
  title = {OGCP Pilot: A Physics-Aware Guitar Chord Dataset},
  year = {2026},
  version = {1.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18979053},
  url = {https://doi.org/10.5281/zenodo.18979053}
}
```
